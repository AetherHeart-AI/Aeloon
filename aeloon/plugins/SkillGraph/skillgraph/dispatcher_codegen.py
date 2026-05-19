"""Dispatcher artifact generator for toolkit-style skills."""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .models import IOField, RuntimeDependency, RuntimeManifest
from .package import AssetType, SkillPackage

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_FENCE_RE = re.compile(r"```(?P<lang>[^\n`]*)\n(?P<body>.*?)```", re.DOTALL)
_ANGLE_RE = re.compile(r"<([a-zA-Z0-9_-]+)>")
_SQUARE_RE = re.compile(r"\[([a-zA-Z0-9_-]+)\]")
_SHELL_PREFIXES = (
    "bash ",
    "sh ",
    "python ",
    "python3 ",
    "npx ",
    "ts-node ",
    "pip ",
    "pip3 ",
    "grep ",
    "find ",
    "sed ",
    "awk ",
    "ls ",
    "mv ",
    "cp ",
    "mkdir ",
    "rm ",
    "du ",
    "sort ",
    "uniq ",
    "curl ",
    "wget ",
    "git ",
    "pytest ",
    "mvn ",
    "gradle ",
    "npm ",
    "pnpm ",
    "yarn ",
    "node ",
    "go ",
    "cargo ",
    "make ",
    "ffmpeg ",
    "ffprobe ",
    "trivy ",
    "browser ",
    "openclaw ",
    "md5 ",
)
_BUILTINS = {
    "echo",
    "cat",
    "test",
    "true",
    "false",
    "cd",
    "export",
    "source",
    ".",
    "set",
    "unset",
    "read",
    "printf",
    "eval",
    "exec",
    "exit",
    "return",
    "shift",
    "trap",
    "wait",
    "jobs",
    "fg",
    "bg",
    "kill",
    "umask",
    "alias",
    "unalias",
    "type",
    "builtin",
    "command",
    "local",
    "declare",
    "typeset",
    "readonly",
}
_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "into",
    "this",
    "that",
    "step",
    "file",
    "files",
    "skill",
    "using",
    "into",
    "your",
}


@dataclass
class DispatcherCapability:
    name: str
    title: str
    kind: str
    summary: str
    inputs: list[IOField]
    commands: list[str]
    code_example: str
    source_refs: list[dict[str, Any]]
    keywords: list[str]
    script_path: str = ""
    input_bindings: dict[str, str] | None = None
    can_produce_final_answer: bool = False
    confidence: float = 1.0
    validation_status: str = "static"
    provenance: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "kind": self.kind,
            "summary": self.summary,
            "inputs": [field.model_dump() for field in self.inputs],
            "commands": list(self.commands),
            "code_example": self.code_example,
            "source_refs": list(self.source_refs),
            "keywords": list(self.keywords),
            "script_path": self.script_path,
            "input_bindings": dict(self.input_bindings or {}),
            "can_produce_final_answer": self.can_produce_final_answer,
            "confidence": self.confidence,
            "validation_status": self.validation_status,
            "provenance": dict(self.provenance or {}),
        }


def generate_dispatcher(
    *,
    package: SkillPackage,
    entry_skill: Path,
    output_path: str | Path,
    base_url: str,
    llm_model: str,
) -> tuple[Path, RuntimeManifest, list[DispatcherCapability]]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    capabilities = extract_dispatcher_capabilities(package, entry_skill)
    manifest = build_dispatcher_manifest(package, capabilities)
    module_name = output_path.stem.removesuffix("_workflow")
    description = _extract_description(entry_skill) or f"Dispatcher artifact for {package.slug}"
    code = _build_dispatcher_code(
        workflow_name=module_name,
        description=description,
        capabilities=capabilities,
        base_url=base_url,
        llm_model=llm_model,
    )
    output_path.write_text(code, encoding="utf-8")
    manifest.save(output_path.with_suffix(".manifest.json"))

    config_path = output_path.parent / "skill_config.json"
    if not config_path.exists():
        config_data = {
            "runtime": {
                "api_key": "",
                "base_url": base_url,
                "model": llm_model,
            }
        }
        config_path.write_text(json.dumps(config_data, indent=2), encoding="utf-8")

    return output_path, manifest, capabilities


def extract_dispatcher_capabilities(
    package: SkillPackage, entry_skill: Path
) -> list[DispatcherCapability]:
    content = entry_skill.read_text(encoding="utf-8") if entry_skill.exists() else ""
    generic_capabilities = _generic_office_capabilities(package, entry_skill)
    blocks = list(_iter_blocks(content))
    local_modules = {
        Path(asset.path).stem
        for asset in package.assets
        if asset.asset_type == AssetType.script and asset.path.endswith(".py")
    }
    capabilities: list[DispatcherCapability] = []
    seen_names: dict[str, int] = {}

    for block in blocks:
        name = _capability_name(
            block["heading"] or block["heading_path"][-1] or f"capability_{len(capabilities) + 1}"
        )
        seen_names.setdefault(name, 0)
        seen_names[name] += 1
        if seen_names[name] > 1:
            name = f"{name}_{seen_names[name]}"

        summary = block["summary"] or block["heading"] or "Capability extracted from SKILL.md"
        source_ref = {
            "path": entry_skill.name,
            "line": block["line"],
            "snippet": (block["summary"] or block["heading"] or block["body"]).strip()[:220],
        }

        commands = block["commands"]
        code_example = block["body"].strip()
        inputs = _infer_inputs_from_block(block["body"], commands)
        keywords = _build_keywords(
            " ".join(
                [
                    *block["heading_path"],
                    summary,
                    code_example,
                    " ".join(commands),
                    " ".join(sorted(local_modules)),
                ]
            )
        )

        capabilities.append(
            DispatcherCapability(
                name=name,
                title=block["heading"] or name.replace("_", " ").title(),
                kind=block["kind"],
                summary=summary,
                inputs=inputs,
                commands=commands,
                code_example=code_example,
                source_refs=[source_ref],
                keywords=keywords,
            )
        )

    for asset in package.assets:
        if asset.asset_type != AssetType.script:
            continue
        if not asset.path.endswith((".py", ".sh", ".ts", ".js", ".mjs", ".cjs")):
            continue
        if not any(
            Path(asset.path).name in cap.code_example or Path(asset.path).stem in cap.keywords
            for cap in capabilities
        ):
            script_name = _capability_name(Path(asset.path).stem)
            seen_names.setdefault(script_name, 0)
            seen_names[script_name] += 1
            if seen_names[script_name] > 1:
                script_name = f"{script_name}_{seen_names[script_name]}"
            capabilities.append(
                DispatcherCapability(
                    name=script_name,
                    title=f"Use {Path(asset.path).name}",
                    kind="script_asset",
                    summary=f"Bundled script available at {asset.path}",
                    inputs=[],
                    commands=[],
                    code_example=asset.path,
                    source_refs=[{"path": asset.path, "line": None, "snippet": asset.path}],
                    keywords=_build_keywords(asset.path),
                    script_path=asset.path,
                )
            )

    if generic_capabilities and package.slug in {"docx", "pptx"}:
        capabilities = [cap for cap in capabilities if cap.kind != "shell_command"]
    return _dedupe_capabilities(generic_capabilities + capabilities)


def _apply_refined_dispatcher_payload(
    capabilities: list[DispatcherCapability],
    data: dict[str, Any],
) -> tuple[list[DispatcherCapability], list[str], bool]:
    if not isinstance(data, dict):
        return capabilities, ["refinement_payload_not_object"], False
    items = data.get("operators")
    if not isinstance(items, list):
        return capabilities, ["refinement_missing_operators"], False

    by_source = {
        str(item.get("source_name") or ""): item for item in items if isinstance(item, dict)
    }
    issues: list[str] = []
    refined: list[DispatcherCapability] = []
    changed = False
    for cap in capabilities:
        item = by_source.get(cap.name)
        if item is None:
            refined.append(cap)
            continue
        if item.get("include") is False:
            issues.append(f"dropped_operator:{cap.name}")
            changed = True
            continue
        new_inputs = _refined_inputs(cap, item.get("inputs"), issues)
        keywords = _list_strings(item.get("keywords"), limit=120) or list(cap.keywords)
        refined_cap = replace(
            cap,
            name=_capability_name(str(item.get("name") or cap.name)),
            title=_bounded_text(item.get("title"), cap.title, limit=120),
            summary=_bounded_text(item.get("summary"), cap.summary, limit=700),
            inputs=new_inputs,
            keywords=keywords,
            confidence=_confidence(item.get("confidence"), default=cap.confidence),
            validation_status="llm_refined_validated",
            provenance={"refinement": "llm_refined", "source_name": cap.name},
        )
        changed = changed or refined_cap.to_dict() != cap.to_dict()
        refined.append(refined_cap)

    known = {cap.name for cap in capabilities}
    for source_name in sorted(set(by_source) - known):
        issues.append(f"ignored_unknown_operator:{source_name}")
    if not refined:
        return capabilities, issues + ["refinement_dropped_all_operators"], False
    return _dedupe_capabilities(refined), issues, changed


def _refined_inputs(
    cap: DispatcherCapability,
    raw_inputs: Any,
    issues: list[str],
) -> list[IOField]:
    if not isinstance(raw_inputs, list):
        return [IOField(**field.model_dump()) for field in cap.inputs]
    existing_by_name = {field.name: field for field in cap.inputs}
    existing_by_normalized = {_capability_name(field.name): field for field in cap.inputs}
    if not existing_by_name:
        if raw_inputs:
            issues.append(f"ignored_new_inputs_for_schema_free_operator:{cap.name}")
        return []
    refined: list[IOField] = []
    seen: set[str] = set()
    for item in raw_inputs:
        if isinstance(item, str):
            raw_name = item
            description = ""
        elif isinstance(item, dict):
            raw_name = str(item.get("name") or "")
            description = str(item.get("description") or "")
        else:
            continue
        source = existing_by_name.get(raw_name) or existing_by_normalized.get(
            _capability_name(raw_name)
        )
        if source is None:
            issues.append(f"ignored_unknown_input:{cap.name}.{raw_name}")
            continue
        if source.name in seen:
            continue
        field_data = source.model_dump()
        if description:
            field_data["description"] = description[:240].rstrip()
        refined.append(IOField(**field_data))
        seen.add(source.name)
    return refined or [IOField(**field.model_dump()) for field in cap.inputs]


def _bounded_text(value: Any, fallback: str, *, limit: int) -> str:
    text = str(value or "").strip()
    return text[:limit].rstrip() if text else fallback


def _list_strings(value: Any, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text and text not in out:
            out.append(text[:80].rstrip())
        if len(out) >= limit:
            break
    return out


def _confidence(value: Any, *, default: float) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return default


def _field(
    name: str,
    *,
    field_type: str = "string",
    required: bool = True,
    description: str = "",
) -> IOField:
    return IOField(name=name, type=field_type, required=required, description=description)


def _generic_office_capabilities(
    package: SkillPackage,
    entry_skill: Path,
) -> list[DispatcherCapability]:
    if package.slug == "docx":
        return _generic_docx_capabilities(entry_skill)
    if package.slug == "pptx":
        return _generic_pptx_capabilities(entry_skill)
    return []


def _generic_source_ref(entry_skill: Path, snippet: str) -> list[dict[str, Any]]:
    return [{"path": entry_skill.name or "SKILL.md", "line": None, "snippet": snippet}]


def _generic_docx_capabilities(entry_skill: Path) -> list[DispatcherCapability]:
    source_refs = _generic_source_ref(
        entry_skill,
        "Generic DOCX executable operators derived only from the docx skill package.",
    )
    return [
        DispatcherCapability(
            name="inspect_docx",
            title="Inspect DOCX structure",
            kind="docx_inspect",
            summary=(
                "Generic executable operator that summarizes DOCX paragraphs, tables, "
                "headers, footers, and placeholder-like text."
            ),
            inputs=[_field("path", description="Input DOCX workspace path.")],
            commands=[],
            code_example="inspect_docx(path)",
            source_refs=source_refs,
            keywords=_build_keywords("docx inspect structure paragraphs tables headers footers placeholders"),
        ),
        DispatcherCapability(
            name="replace_docx_text",
            title="Replace DOCX text",
            kind="docx_replace_text",
            summary=(
                "Generic executable operator for split-run-safe text replacement across "
                "DOCX body, tables, headers, and footers."
            ),
            inputs=[
                _field("input_path", description="Input DOCX workspace path."),
                _field("output_path", description="Output DOCX workspace path."),
                _field(
                    "replacements",
                    field_type="object",
                    description="Mapping of exact text/placeholders to replacement text.",
                ),
                _field(
                    "include_headers_footers",
                    field_type="boolean",
                    required=False,
                    description="Whether to process headers and footers. Defaults to true.",
                ),
                _field(
                    "include_tables",
                    field_type="boolean",
                    required=False,
                    description="Whether to process table cells. Defaults to true.",
                ),
            ],
            commands=[],
            code_example="replace_docx_text(input_path, output_path, replacements)",
            source_refs=source_refs,
            keywords=_build_keywords("docx replace placeholders split runs headers footers tables template"),
        ),
        DispatcherCapability(
            name="remove_or_keep_marked_blocks",
            title="Remove or keep marked DOCX blocks",
            kind="docx_marked_blocks",
            summary=(
                "Generic executable operator that removes or keeps text blocks delimited "
                "by caller-provided markers."
            ),
            inputs=[
                _field("input_path", description="Input DOCX workspace path."),
                _field("output_path", description="Output DOCX workspace path."),
                _field(
                    "markers",
                    field_type="array",
                    description="Marker specs with name/start/end and optional keep flag.",
                ),
                _field(
                    "conditions",
                    field_type="object",
                    required=False,
                    description="Optional mapping from marker name to keep/remove boolean.",
                ),
            ],
            commands=[],
            code_example="remove_or_keep_marked_blocks(input_path, output_path, markers)",
            source_refs=source_refs,
            keywords=_build_keywords("docx conditional section marker remove keep block template"),
        ),
        DispatcherCapability(
            name="validate_docx_text",
            title="Validate DOCX text",
            kind="docx_validate_text",
            summary="Generic executable operator that checks required and forbidden text in a DOCX.",
            inputs=[
                _field("path", description="Input DOCX workspace path."),
                _field(
                    "must_contain",
                    field_type="array",
                    required=False,
                    description="Strings that must appear in the DOCX text.",
                ),
                _field(
                    "must_not_contain",
                    field_type="array",
                    required=False,
                    description="Strings that must not appear in the DOCX text.",
                ),
            ],
            commands=[],
            code_example="validate_docx_text(path, must_contain, must_not_contain)",
            source_refs=source_refs,
            keywords=_build_keywords("docx validate text placeholders markers required forbidden"),
        ),
    ]


def _generic_pptx_capabilities(entry_skill: Path) -> list[DispatcherCapability]:
    source_refs = _generic_source_ref(
        entry_skill,
        "Generic PPTX executable operators derived only from the pptx skill package.",
    )
    return [
        DispatcherCapability(
            name="inventory_pptx",
            title="Inventory PPTX",
            kind="pptx_inventory",
            summary=(
                "Generic executable operator that inventories slides, text boxes, shape ids, "
                "text, and positions from a PPTX."
            ),
            inputs=[
                _field("input_pptx", description="Input PPTX workspace path."),
                _field("output_json", required=False, description="Optional output JSON report path."),
                _field(
                    "include_positions",
                    field_type="boolean",
                    required=False,
                    description="Whether to include shape positions. Defaults to true.",
                ),
            ],
            commands=[],
            code_example="inventory_pptx(input_pptx, output_json)",
            source_refs=source_refs,
            keywords=_build_keywords("pptx inventory slides shapes text boxes positions ids"),
        ),
        DispatcherCapability(
            name="update_pptx_text_boxes",
            title="Update PPTX text boxes",
            kind="pptx_update_text_boxes",
            summary=(
                "Generic executable operator that updates PPTX text boxes selected by "
                "slide index, shape id, shape name, exact text, or substring."
            ),
            inputs=[
                _field(
                    "input_pptx",
                    description="Input PPTX workspace path. Aliases: input_path, pptx_path, path.",
                ),
                _field("output_pptx", description="Output PPTX workspace path. Alias: output_path."),
                _field(
                    "edits",
                    field_type="array",
                    description=(
                        "Text-box edit specs with selectors and replacement/style fields. "
                        "Aliases: updates, text_boxes, or targets plus shared style fields."
                    ),
                ),
            ],
            commands=[],
            code_example="update_pptx_text_boxes(input_pptx, output_pptx, edits=[...])",
            source_refs=source_refs,
            keywords=_build_keywords("pptx update text boxes shapes font position edit"),
        ),
        DispatcherCapability(
            name="add_pptx_reference_slide",
            title="Add PPTX reference slide",
            kind="pptx_add_reference_slide",
            summary=(
                "Generic executable operator that appends a slide containing a title and "
                "numbered or bulleted items supplied at runtime."
            ),
            inputs=[
                _field(
                    "input_pptx",
                    description="Input PPTX workspace path. Aliases: input_path, pptx_path, path.",
                ),
                _field("output_pptx", description="Output PPTX workspace path. Alias: output_path."),
                _field("title", required=False, description="Reference slide title."),
                _field("items", field_type="array", description="Slide list items. Aliases: references, titles."),
                _field(
                    "numbered",
                    field_type="boolean",
                    required=False,
                    description="Whether to use real auto-numbering bullets. Defaults to true.",
                ),
            ],
            commands=[],
            code_example="add_pptx_reference_slide(input_pptx, output_pptx, title='Reference', items=titles)",
            source_refs=source_refs,
            keywords=_build_keywords("pptx add reference slide numbered list items"),
        ),
        DispatcherCapability(
            name="validate_pptx_file",
            title="Validate PPTX file",
            kind="pptx_validate_file",
            summary="Generic executable operator that checks PPTX zip and python-pptx loadability.",
            inputs=[_field("path", description="Input PPTX workspace path.")],
            commands=[],
            code_example="validate_pptx_file(path)",
            source_refs=source_refs,
            keywords=_build_keywords("pptx validate zip integrity presentation load"),
        ),
    ]


def _dedupe_capabilities(capabilities: list[DispatcherCapability]) -> list[DispatcherCapability]:
    counts: dict[str, int] = {}
    deduped: list[DispatcherCapability] = []
    for cap in capabilities:
        counts[cap.name] = counts.get(cap.name, 0) + 1
        if counts[cap.name] > 1:
            cap.name = f"{cap.name}_{counts[cap.name]}"
        deduped.append(cap)
    return deduped


def build_dispatcher_manifest(
    package: SkillPackage, capabilities: list[DispatcherCapability]
) -> RuntimeManifest:
    deps: dict[tuple[str, str], RuntimeDependency] = {}
    local_modules = {
        Path(asset.path).stem: asset.path
        for asset in package.assets
        if asset.asset_type == AssetType.script and asset.path.endswith(".py")
    }

    def add(kind: str, name: str, step_id: str, *, required: bool = True) -> None:
        key = (kind, name)
        if key in deps:
            if step_id not in deps[key].source_step_ids:
                deps[key].source_step_ids.append(step_id)
            if required:
                deps[key].required = True
            return
        deps[key] = RuntimeDependency(
            kind=kind,
            name=name,
            required=required,
            source_step_ids=[step_id],
            check_command=_make_check_command(kind, name),
        )

    for capability in capabilities:
        if capability.kind == "shell_command":
            for command in capability.commands:
                tokens = _safe_split(command)
                if not tokens:
                    continue
                binary = tokens[0]
                if binary not in _BUILTINS and not binary.startswith("builtin:"):
                    if binary.endswith(".py") or binary.startswith(("./", "scripts/")):
                        add("script_file", binary, capability.name)
                    else:
                        add("cli_binary", binary, capability.name)
                for token in tokens[1:]:
                    if token.endswith(".py") or token.startswith(("./", "scripts/")):
                        add("script_file", token, capability.name)
        if capability.kind in {"python_api", "script_asset"}:
            for module_name in _extract_imports(capability.code_example):
                if module_name in local_modules:
                    add("script_file", local_modules[module_name], capability.name)
                else:
                    add("python_package", module_name, capability.name, required=False)

    return RuntimeManifest(dependencies=sorted(deps.values(), key=lambda dep: (dep.kind, dep.name)))


def _build_dispatcher_code(
    *,
    workflow_name: str,
    description: str,
    capabilities: list[DispatcherCapability],
    base_url: str,
    llm_model: str,
) -> str:
    capabilities_payload = [cap.to_dict() for cap in capabilities]
    meta = {
        "name": workflow_name,
        "description": description,
        "global_inputs": [
            {
                "name": "task",
                "type": "string",
                "required": False,
                "description": "Natural language description of what you want the skill to help with.",
            },
            {
                "name": "operation",
                "type": "string",
                "required": False,
                "description": "Exact capability name to run or inspect.",
            },
            {
                "name": "arguments",
                "type": "dict",
                "required": False,
                "description": "Structured arguments for the selected capability.",
            },
            {
                "name": "execute",
                "type": "bool",
                "required": False,
                "description": "If true, execute shell capabilities after resolving placeholders.",
            },
        ],
    }
    template = _strip_template_margin(
        f'''\
        #!/usr/bin/env python3
        """
        Auto-generated dispatcher artifact: {workflow_name}
        """

        import argparse
        import json
        import logging
        import os
        import re
        import subprocess
        from typing import Any

        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
        logger = logging.getLogger(__name__)

        DEFAULT_BASE_URL = {base_url!r}
        DEFAULT_LLM_MODEL = {llm_model!r}
        DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skill_config.json")
        MANIFEST_FILENAME = {f"{workflow_name}_workflow.manifest.json"!r}
        SANDBOX_DIRNAME = {f"{workflow_name}_workflow.sandbox"!r}
        SKILL_META = __SKILL_META__
        CAPABILITIES = __CAPABILITIES__

        class _DispatcherGraph:
            def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
                return run_until_blocked(state)


        def build_graph() -> _DispatcherGraph:
            return _DispatcherGraph()


        def _load_runtime_config(config_path: str | None) -> dict[str, Any]:
            path = config_path or os.getenv("SKILLGRAPH_CONFIG_PATH", DEFAULT_CONFIG_PATH)
            path = os.path.abspath(os.path.expanduser(path))
            file_cfg = {{}}
            if os.path.exists(path):
                try:
                    file_cfg = json.loads(open(path, "r", encoding="utf-8").read() or "{{}}")
                except Exception as exc:
                    logger.warning("[config] failed to parse %s: %s", path, exc)
            runtime = file_cfg.get("runtime", file_cfg) if isinstance(file_cfg, dict) else {{}}
            if not isinstance(runtime, dict):
                runtime = {{}}
            return {{
                "api_key": runtime.get("api_key") or os.getenv("SKILLGRAPH_API_KEY") or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or "",
                "base_url": runtime.get("base_url") or os.getenv("SKILLGRAPH_BASE_URL") or DEFAULT_BASE_URL,
                "model": runtime.get("model") or os.getenv("SKILLGRAPH_RUNTIME_MODEL") or DEFAULT_LLM_MODEL,
                "config_path": path,
            }}


        def _sandbox_dir() -> str:
            return os.path.join(os.path.dirname(os.path.abspath(__file__)), SANDBOX_DIRNAME)


        def _sandbox_skill_dir() -> str:
            return os.path.join(_sandbox_dir(), "skill")


        def _sandbox_env() -> dict[str, Any]:
            path = os.path.join(_sandbox_dir(), "env.json")
            if not os.path.exists(path):
                return {{}}
            try:
                return json.loads(open(path, "r", encoding="utf-8").read() or "{{}}")
            except Exception:
                return {{}}


        def _which_with_sandbox(name: str, sandbox_bin: str, sandbox_skill: str) -> str:
            import shutil
            node_bin = os.path.join(sandbox_skill, "node_modules", ".bin")
            extra = sandbox_bin + os.pathsep + node_bin
            old_path = os.environ.get("PATH", "")
            try:
                os.environ["PATH"] = extra + os.pathsep + old_path
                return shutil.which(name) or ""
            finally:
                os.environ["PATH"] = old_path


        def preflight_check(project_dir: str) -> list[str]:
            sandbox_dir = _sandbox_dir()
            if not os.path.isdir(sandbox_dir):
                return ["Skill sandbox not found: " + sandbox_dir]
            bootstrap_path = os.path.join(sandbox_dir, "bootstrap.json")
            if not os.path.exists(bootstrap_path):
                return ["Skill sandbox bootstrap metadata not found: " + bootstrap_path]
            try:
                bootstrap = json.loads(open(bootstrap_path, "r", encoding="utf-8").read() or "{{}}")
            except Exception as exc:
                return [f"Failed to read sandbox bootstrap metadata: {{exc}}"]
            if bootstrap.get("status") != "ready":
                failures = []
                for check in bootstrap.get("checks", []):
                    if not check.get("ok", False):
                        failures.append(f"Sandbox bootstrap failed: {{check.get('name')}} — {{check.get('detail', '')}}")
                return failures or ["Skill sandbox bootstrap is not ready"]
            manifest_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), MANIFEST_FILENAME)
            if not os.path.exists(manifest_path):
                return ["Runtime manifest not found: " + manifest_path]
            try:
                manifest = json.loads(open(manifest_path, "r", encoding="utf-8").read())
            except Exception as exc:
                return [f"Failed to read runtime manifest: {{exc}}"]
            sandbox_env = _sandbox_env()
            sandbox_bin = sandbox_env.get("SANDBOX_BIN_DIR", os.path.join(sandbox_dir, "bin"))
            sandbox_skill = _sandbox_skill_dir()
            failures = []
            for dep in manifest.get("dependencies", []):
                if not dep.get("required", True):
                    continue
                kind = dep.get("kind", "")
                name = dep.get("name", "")
                if kind == "cli_binary":
                    if not _which_with_sandbox(name, sandbox_bin, sandbox_skill):
                        failures.append(f"CLI tool not found: {{name}}")
                elif kind == "script_file":
                    if not os.path.isfile(os.path.join(sandbox_skill, name)):
                        failures.append(f"Required script not found in sandbox: {{name}}")
                elif kind == "env_var":
                    if not os.getenv(name):
                        failures.append(f"Required environment variable not set: {{name}}")
            return failures


        def _canonical_name(value: str) -> str:
            text = (value or "").strip().lower()
            text = re.sub(r"[^a-z0-9_\\-]+", "_", text)
            return re.sub(r"_+", "_", text).strip("_-")


        def _keywords(text: str) -> set[str]:
            tokens = set(re.findall(r"[a-z0-9_]+", (text or "").lower()))
            return {{token for token in tokens if len(token) >= 3 and token not in {sorted(_STOPWORDS)!r}}}


        def _select_capability(operation: str, task: str) -> dict[str, Any] | None:
            if operation:
                want = _canonical_name(operation)
                for capability in CAPABILITIES:
                    if _canonical_name(capability.get("name", "")) == want:
                        return capability
                    if _canonical_name(capability.get("title", "")) == want:
                        return capability
            if not task:
                return None
            wanted = _keywords(task)
            best = None
            best_score = 0
            for capability in CAPABILITIES:
                haystack = wanted.intersection(set(capability.get("keywords", [])))
                score = len(haystack)
                if score > best_score:
                    best = capability
                    best_score = score
            return best if best_score > 0 else None


        def _resolve_input(name: str, global_inputs: dict[str, Any], arguments: dict[str, Any]) -> Any:
            if name in arguments:
                return arguments[name]
            return global_inputs.get(name)


        def _render_template(text: str, global_inputs: dict[str, Any], arguments: dict[str, Any]) -> tuple[str, list[str]]:
            missing = []

            def repl(match):
                raw = match.group(1)
                key = raw.strip().lower().replace("-", "_")
                value = _resolve_input(key, global_inputs, arguments)
                if value is None:
                    missing.append(key)
                    return match.group(0)
                return str(value)

            rendered = re.sub(r"<([a-zA-Z0-9_-]+)>", repl, text)
            rendered = re.sub(r"\\[([a-zA-Z0-9_-]+)\\]", repl, rendered)
            return rendered, missing


        def _pending(message: str, *, current_step: str | None, state: dict[str, Any], details: dict[str, Any] | None = None, suggested_actions: list[str] | None = None) -> dict[str, Any]:
            return {{
                "status": "blocked",
                "current_step": current_step,
                "final_output": None,
                "step_results": state.get("step_results", {{}}),
                "block": {{
                    "message": message,
                    "details": details or {{}},
                    "suggested_actions": suggested_actions or [],
                }},
                "graph_state": state,
            }}


        def _completed(state: dict[str, Any], capability: dict[str, Any], final_output: str, step_result: dict[str, Any]) -> dict[str, Any]:
            step_results = dict(state.get("step_results", {{}}))
            step_results[capability["name"]] = step_result
            graph_state = dict(state)
            graph_state["step_results"] = step_results
            graph_state["final_output"] = final_output
            return {{
                "status": "completed",
                "current_step": capability["name"],
                "final_output": final_output,
                "step_results": step_results,
                "block": None,
                "graph_state": graph_state,
            }}


        def _build_guidance(capability: dict[str, Any], global_inputs: dict[str, Any], arguments: dict[str, Any]) -> str:
            lines = [f"Capability: {{capability['title']}}", capability.get("summary", "")]
            if capability.get("commands"):
                lines.append("Commands:")
                for command in capability["commands"]:
                    rendered, _missing = _render_template(command, global_inputs, arguments)
                    lines.append(f"- {{rendered}}")
            if capability.get("code_example"):
                lines.append("Code example:")
                lines.append(capability["code_example"])
            if capability.get("source_refs"):
                refs = ", ".join(ref.get("path", "") for ref in capability["source_refs"] if ref.get("path"))
                if refs:
                    lines.append(f"Sources: {{refs}}")
            return "\\n".join(line for line in lines if line)


        def _shell_env() -> dict[str, str]:
            env = os.environ.copy()
            sandbox_env = _sandbox_env()
            prepend = sandbox_env.get("PATH_PREPEND", "")
            if prepend:
                env["PATH"] = prepend + os.pathsep + env.get("PATH", "")
            return env


        def _run_shell(commands: list[str], cwd: str) -> tuple[bool, list[dict[str, Any]]]:
            outputs = []
            env = _shell_env()
            for command in commands:
                proc = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=cwd, env=env)
                outputs.append({{
                    "command": command,
                    "exit_code": proc.returncode,
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                }})
                if proc.returncode != 0:
                    return False, outputs
            return True, outputs


        def run_until_blocked(state: dict[str, Any]) -> dict[str, Any]:
            global_inputs = dict(state.get("global_inputs", {{}}))
            arguments = global_inputs.get("arguments") if isinstance(global_inputs.get("arguments"), dict) else {{}}
            operation = str(global_inputs.get("operation") or "")
            task = str(global_inputs.get("task") or global_inputs.get("user_request") or "")
            capability = _select_capability(operation, task)
            if capability is None:
                available = [cap.get("name", "") for cap in CAPABILITIES]
                return _pending(
                    "No dispatcher capability matched the request.",
                    current_step=None,
                    state=state,
                    details={{"available_operations": available}},
                    suggested_actions=[
                        "Call the workflow again with `operation` set to one of the available capability names.",
                        "Or provide a more specific `task` describing the desired operation.",
                    ],
                )

            missing = []
            for field in capability.get("inputs", []):
                if not field.get("required", True):
                    continue
                if _resolve_input(field.get("name", ""), global_inputs, arguments) is None:
                    missing.append(field.get("name", ""))
            if missing:
                return _pending(
                    f"Capability '{{capability['name']}}' is missing required inputs: {{', '.join(missing)}}",
                    current_step=capability["name"],
                    state=state,
                    details={{"missing_inputs": missing, "capability": capability["name"]}},
                    suggested_actions=[
                        "Provide the missing inputs under the `arguments` object and call resume_workflow.",
                    ],
                )

            execute = bool(global_inputs.get("execute", False))
            if execute and capability.get("kind") == "shell_command" and capability.get("commands"):
                rendered = []
                unresolved = []
                for command in capability["commands"]:
                    cmd, missing_inputs = _render_template(command, global_inputs, arguments)
                    rendered.append(cmd)
                    unresolved.extend(missing_inputs)
                if unresolved:
                    return _pending(
                        f"Capability '{{capability['name']}}' still has unresolved placeholders: {{', '.join(sorted(set(unresolved)))}}",
                        current_step=capability["name"],
                        state=state,
                        details={{"missing_inputs": sorted(set(unresolved)), "commands": rendered}},
                        suggested_actions=["Provide the missing inputs and call resume_workflow."],
                    )
                ok, outputs = _run_shell(rendered, str(global_inputs.get("project_dir") or "."))
                if not ok:
                    return _pending(
                        f"Capability '{{capability['name']}}' failed while executing shell commands.",
                        current_step=capability["name"],
                        state=state,
                        details={{"outputs": outputs}},
                        suggested_actions=["Inspect the command output, repair the environment if needed, then call resume_workflow."],
                    )
                final_output = json.dumps({{"capability": capability["name"], "outputs": outputs}}, ensure_ascii=False, indent=2)
                return _completed(state, capability, final_output, {{"mode": "executed", "outputs": outputs}})

            final_output = _build_guidance(capability, global_inputs, arguments)
            return _completed(state, capability, final_output, {{"mode": "guided", "capability": capability["name"]}})


        def resume_from_state(state: dict[str, Any]) -> dict[str, Any]:
            return run_until_blocked(state)


        def main() -> int:
            parser = argparse.ArgumentParser(description={description!r})
            parser.add_argument("--project", default=".")
            parser.add_argument("--config", default=None)
            parser.add_argument("--task", default="")
            parser.add_argument("--operation", default="")
            parser.add_argument("--arguments", default="{{}}")
            parser.add_argument("--execute", action="store_true")
            args = parser.parse_args()

            runtime_config = _load_runtime_config(args.config)
            failures = preflight_check(args.project)
            if failures:
                print(json.dumps({{"status": "blocked", "preflight": failures}}, ensure_ascii=False, indent=2))
                return 2
            try:
                arguments = json.loads(args.arguments or "{{}}")
            except Exception:
                arguments = {{}}
            state = {{
                "global_inputs": {{
                    "project_dir": args.project,
                    "task": args.task,
                    "operation": args.operation,
                    "arguments": arguments,
                    "execute": args.execute,
                    "_runtime_config": runtime_config,
                }},
                "step_results": {{}},
                "error": None,
                "final_output": None,
            }}
            result = run_until_blocked(state)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result.get("status") == "completed" else 1


        if __name__ == "__main__":
            raise SystemExit(main())
        '''
    )
    return template.replace("__SKILL_META__", repr(meta)).replace(
        "__CAPABILITIES__", repr(capabilities_payload)
    )


def _extract_description(entry_skill: Path) -> str:
    try:
        text = entry_skill.read_text(encoding="utf-8")
    except Exception:
        return ""
    match = re.search(r"^description:\s*(.+)$", text, re.MULTILINE)
    return match.group(1).strip().strip('"') if match else ""


def _iter_blocks(content: str) -> list[dict[str, Any]]:
    lines = content.splitlines()
    blocks: list[dict[str, Any]] = []
    heading_stack: list[tuple[int, str]] = []
    recent_text: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        heading_match = _HEADING_RE.match(line)
        if heading_match:
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            recent_text = []
            index += 1
            continue

        stripped = line.strip()
        if stripped.startswith("```"):
            language = stripped[3:].strip().lower()
            start_line = index + 1
            body_lines: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                body_lines.append(lines[index])
                index += 1
            body = "\n".join(body_lines).strip()
            heading_path = [title for _level, title in heading_stack]
            heading = heading_path[-1] if heading_path else "Capability"
            summary = " ".join(recent_text[-3:]).strip()
            commands = _extract_shell_commands(body, language)
            kind = (
                "shell_command"
                if commands
                else "python_api"
                if language in {"python", "py"}
                else "reference"
            )
            blocks.append(
                {
                    "line": start_line,
                    "heading": heading,
                    "heading_path": heading_path,
                    "language": language,
                    "body": body,
                    "summary": summary,
                    "commands": commands,
                    "kind": kind,
                }
            )
            recent_text = []
            if index < len(lines) and lines[index].strip().startswith("```"):
                index += 1
            continue

        if stripped and not stripped.startswith("#"):
            recent_text.append(stripped)
            recent_text = recent_text[-4:]
        elif not stripped:
            recent_text = recent_text[-2:]
        index += 1
    return [block for block in blocks if block["kind"] != "reference"]


def _extract_shell_commands(body: str, language: str) -> list[str]:
    if language and language not in {"bash", "sh", "shell", "zsh", "console", ""}:
        return []
    commands: list[str] = []
    for raw in body.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(_SHELL_PREFIXES) or re.match(
            r"^(?:\./|[A-Za-z0-9_.-]+\s+-[A-Za-z])", line
        ):
            commands.append(line)
    return commands


def _infer_inputs_from_block(body: str, commands: list[str]) -> list[IOField]:
    fields: dict[str, IOField] = {}
    for command in commands or [body]:
        for raw in _ANGLE_RE.findall(command):
            name = raw.strip().lower().replace("-", "_")
            fields.setdefault(
                name, IOField(name=name, description=f"Value for <{raw}>", required=True)
            )
        for raw in _SQUARE_RE.findall(command):
            name = raw.strip().lower().replace("-", "_")
            fields.setdefault(
                name, IOField(name=name, description=f"Optional value for [{raw}]", required=False)
            )
    return list(fields.values())


def _build_keywords(text: str) -> list[str]:
    tokens = set(re.findall(r"[a-z0-9_]+", text.lower()))
    return sorted(token for token in tokens if len(token) >= 3 and token not in _STOPWORDS)


def _capability_name(title: str) -> str:
    text = re.sub(r"^\d+\.?\s*", "", (title or "capability").strip().lower())
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "capability"


def _safe_split(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except Exception:
        return command.split()


def _extract_imports(code: str) -> list[str]:
    imports: list[str] = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("import "):
            parts = stripped.removeprefix("import ").split(",")
            for part in parts:
                module = part.strip().split()[0].split(".")[0]
                if module:
                    imports.append(module)
        elif stripped.startswith("from "):
            module = stripped.removeprefix("from ").split()[0].split(".")[0]
            if module:
                imports.append(module)
    return sorted(set(imports))


def _make_check_command(kind: str, name: str) -> str:
    if kind == "cli_binary":
        return f"which {name}"
    if kind == "script_file":
        return f"test -f {name}"
    if kind == "env_var":
        return f'test -n "${name}"'
    if kind == "python_package":
        return f'python3 -c "import {name}"'
    return ""


def _strip_template_margin(template: str, margin: str = "        ") -> str:
    lines = template.splitlines()
    stripped = [line[len(margin) :] if line.startswith(margin) else line for line in lines]
    return "\n".join(stripped).lstrip() + "\n"
