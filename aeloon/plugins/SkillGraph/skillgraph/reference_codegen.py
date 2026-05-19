"""Reference adapter generator for knowledge- and guide-style skills."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import RuntimeManifest
from .package import SkillPackage

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "into",
    "this",
    "that",
    "your",
    "skill",
    "guide",
    "notes",
    "section",
    "using",
}


@dataclass
class ReferenceSection:
    title: str
    heading_path: list[str]
    summary: str
    body: str
    line: int
    formulas: list[str]
    keywords: list[str]
    query_aliases: list[str] = field(default_factory=list)
    use_cases: list[str] = field(default_factory=list)
    confidence: float = 1.0
    validation_status: str = "static"

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "heading_path": list(self.heading_path),
            "summary": self.summary,
            "body": self.body,
            "line": self.line,
            "formulas": list(self.formulas),
            "keywords": list(self.keywords),
            "query_aliases": list(self.query_aliases),
            "use_cases": list(self.use_cases),
            "confidence": self.confidence,
            "validation_status": self.validation_status,
        }


def _apply_refined_reference_payload(
    sections: list[ReferenceSection],
    data: dict[str, Any],
) -> tuple[list[ReferenceSection], list[str], bool]:
    if not isinstance(data, dict):
        return sections, ["refinement_payload_not_object"], False
    items = data.get("sections")
    if not isinstance(items, list):
        return sections, ["refinement_missing_sections"], False

    refined: list[ReferenceSection] = []
    issues: list[str] = []
    changed = False
    for item in items:
        if not isinstance(item, dict):
            continue
        indexes = item.get("source_indexes")
        if not isinstance(indexes, list):
            indexes = [item.get("source_index")]
        source_sections: list[ReferenceSection] = []
        for raw_index in indexes:
            try:
                source_index = int(raw_index)
            except (TypeError, ValueError):
                issues.append(f"ignored_invalid_source_index:{raw_index}")
                continue
            if source_index < 1 or source_index > len(sections):
                issues.append(f"ignored_invalid_source_index:{source_index}")
                continue
            source_sections.append(sections[source_index - 1])
        if not source_sections:
            continue

        first = source_sections[0]
        formulas: list[str] = []
        for section in source_sections:
            for formula in section.formulas:
                if formula not in formulas:
                    formulas.append(formula)

        refined.append(
            ReferenceSection(
                title=_bounded_text(item.get("title"), first.title, limit=160),
                heading_path=list(first.heading_path),
                summary=_bounded_text(item.get("summary"), first.summary, limit=700),
                body="\n\n".join(section.body for section in source_sections),
                line=first.line,
                formulas=formulas,
                keywords=_list_strings(item.get("keywords"), limit=120) or list(first.keywords),
                query_aliases=_list_strings(item.get("query_aliases"), limit=80),
                use_cases=_list_strings(item.get("use_cases"), limit=80),
                confidence=_confidence(item.get("confidence"), default=first.confidence),
                validation_status="llm_refined_validated",
            )
        )
        changed = True

    if not refined:
        return sections, issues + ["refinement_produced_no_sections"], False
    return refined, issues, changed


def _bounded_text(value: Any, fallback: str, *, limit: int) -> str:
    text = str(value or "").strip()
    return text[:limit].rstrip() if text else fallback


def _list_strings(value: Any, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text and text not in result:
            result.append(text[:120].rstrip())
        if len(result) >= limit:
            break
    return result


def _confidence(value: Any, *, default: float) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return default


def generate_reference_adapter(
    *,
    package: SkillPackage,
    entry_skill: Path,
    output_path: str | Path,
    base_url: str,
    llm_model: str,
) -> tuple[Path, RuntimeManifest, list[ReferenceSection]]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sections = extract_reference_sections(entry_skill)
    manifest = RuntimeManifest(dependencies=[])
    module_name = output_path.stem.removesuffix("_workflow")
    description = _extract_description(entry_skill) or f"Reference adapter for {package.slug}"
    code = _build_reference_code(
        workflow_name=module_name,
        description=description,
        sections=sections,
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

    return output_path, manifest, sections


def extract_reference_sections(entry_skill: Path) -> list[ReferenceSection]:
    content = entry_skill.read_text(encoding="utf-8") if entry_skill.exists() else ""
    lines = content.splitlines()
    sections: list[ReferenceSection] = []
    heading_stack: list[tuple[int, str]] = []
    current_heading = "Overview"
    current_line = 1
    buffer: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        body = "\n".join(buffer).strip()
        if not body:
            return
        heading_path = [title for _level, title in heading_stack] or [current_heading]
        title = heading_path[-1]
        summary = _first_meaningful_line(body)
        formulas = _extract_formula_lines(body)
        keywords = _build_keywords(" ".join([*heading_path, summary, body]))
        sections.append(
            ReferenceSection(
                title=title,
                heading_path=heading_path,
                summary=summary,
                body=body,
                line=current_line,
                formulas=formulas,
                keywords=keywords,
            )
        )

    for index, line in enumerate(lines, start=1):
        match = _HEADING_RE.match(line)
        if match:
            flush()
            buffer = []
            level = len(match.group(1))
            title = match.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            current_heading = title
            current_line = index
            continue
        buffer.append(line)
    flush()

    if not sections:
        body = content.strip()
        sections.append(
            ReferenceSection(
                title="Overview",
                heading_path=["Overview"],
                summary=_first_meaningful_line(body),
                body=body,
                line=1,
                formulas=_extract_formula_lines(body),
                keywords=_build_keywords(body),
            )
        )
    return sections


def _extract_description(entry_skill: Path) -> str:
    try:
        text = entry_skill.read_text(encoding="utf-8")
    except Exception:
        return ""
    match = re.search(r"^description:\s*(.+)$", text, re.MULTILINE)
    return match.group(1).strip().strip('"') if match else ""


def _first_meaningful_line(body: str) -> str:
    for raw in body.splitlines():
        stripped = raw.strip()
        if stripped and not stripped.startswith(("```", "|", "#")):
            return stripped[:220]
    return body.strip().splitlines()[0][:220] if body.strip() else ""


def _extract_formula_lines(body: str) -> list[str]:
    formulas: list[str] = []
    in_fence = False
    for raw in body.splitlines():
        stripped = raw.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if not stripped:
            continue
        if in_fence or ("=" in stripped and len(stripped) <= 120):
            formulas.append(stripped)
    return formulas[:10]


def _build_keywords(text: str) -> list[str]:
    tokens = set(re.findall(r"[a-z0-9_]+", text.lower()))
    return sorted(token for token in tokens if len(token) >= 3 and token not in _STOPWORDS)


def _build_reference_code(
    *,
    workflow_name: str,
    description: str,
    sections: list[ReferenceSection],
    base_url: str,
    llm_model: str,
) -> str:
    sections_payload = [section.to_dict() for section in sections]
    meta = {
        "name": workflow_name,
        "description": description,
        "global_inputs": [
            {
                "name": "task",
                "type": "string",
                "required": False,
                "description": "Task or question that should be answered using the reference skill.",
            },
            {
                "name": "topic",
                "type": "string",
                "required": False,
                "description": "Optional exact topic or heading to look up.",
            },
            {
                "name": "operation",
                "type": "string",
                "required": False,
                "description": "Use `list_topics` to enumerate headings or `lookup` to retrieve matching guidance.",
            },
            {
                "name": "max_results",
                "type": "integer",
                "required": False,
                "description": "Maximum number of matching sections to return.",
            },
        ],
    }
    template = _strip_template_margin(
        f'''\
        #!/usr/bin/env python3
        """
        Auto-generated reference adapter: {workflow_name}
        """

        import argparse
        import json
        import os
        import re
        from typing import Any

        DEFAULT_BASE_URL = {base_url!r}
        DEFAULT_LLM_MODEL = {llm_model!r}
        DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skill_config.json")
        MANIFEST_FILENAME = {f"{workflow_name}_workflow.manifest.json"!r}
        SANDBOX_DIRNAME = {f"{workflow_name}_workflow.sandbox"!r}
        SKILL_META = __SKILL_META__
        REFERENCE_SECTIONS = __SECTIONS__

        class _ReferenceGraph:
            def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
                return run_until_blocked(state)


        def build_graph() -> _ReferenceGraph:
            return _ReferenceGraph()


        def _load_runtime_config(config_path: str | None) -> dict[str, Any]:
            path = config_path or os.getenv("SKILLGRAPH_CONFIG_PATH", DEFAULT_CONFIG_PATH)
            path = os.path.abspath(os.path.expanduser(path))
            file_cfg = {{}}
            if os.path.exists(path):
                try:
                    file_cfg = json.loads(open(path, "r", encoding="utf-8").read() or "{{}}")
                except Exception:
                    file_cfg = {{}}
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
            return []


        def _canonical(value: str) -> str:
            text = (value or "").strip().lower()
            text = re.sub(r"[^a-z0-9_\\-]+", "_", text)
            return re.sub(r"_+", "_", text).strip("_-")


        def _keywords(text: str) -> set[str]:
            tokens = set(re.findall(r"[a-z0-9_]+", (text or "").lower()))
            return {{token for token in tokens if len(token) >= 3 and token not in {sorted(_STOPWORDS)!r}}}


        def _section_score(section: dict[str, Any], query: str, topic: str) -> int:
            score = 0
            if topic:
                want = _canonical(topic)
                if _canonical(section.get("title", "")) == want:
                    score += 10
                if any(_canonical(item) == want for item in section.get("heading_path", [])):
                    score += 8
            query_words = _keywords(query)
            if query_words:
                score += len(query_words.intersection(set(section.get("keywords", []))))
            return score


        def _top_sections(query: str, topic: str, max_results: int) -> list[dict[str, Any]]:
            ranked = []
            for section in REFERENCE_SECTIONS:
                score = _section_score(section, query, topic)
                if score > 0:
                    ranked.append((score, section))
            ranked.sort(key=lambda item: item[0], reverse=True)
            return [section for _score, section in ranked[:max_results]]


        def _pending(message: str, *, state: dict[str, Any], details: dict[str, Any] | None = None) -> dict[str, Any]:
            return {{
                "status": "blocked",
                "current_step": None,
                "final_output": None,
                "step_results": state.get("step_results", {{}}),
                "block": {{
                    "message": message,
                    "details": details or {{}},
                    "suggested_actions": ["Provide a more specific `task` or `topic` and call resume_workflow."],
                }},
                "graph_state": state,
            }}


        def _completed(state: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
            graph_state = dict(state)
            graph_state["final_output"] = payload
            return {{
                "status": "completed",
                "current_step": "reference_lookup",
                "final_output": json.dumps(payload, ensure_ascii=False, indent=2),
                "step_results": {{"reference_lookup": payload}},
                "block": None,
                "graph_state": graph_state,
            }}


        def run_until_blocked(state: dict[str, Any]) -> dict[str, Any]:
            global_inputs = dict(state.get("global_inputs", {{}}))
            operation = str(global_inputs.get("operation") or "lookup").strip().lower()
            task = str(global_inputs.get("task") or "")
            topic = str(global_inputs.get("topic") or "")
            max_results = int(global_inputs.get("max_results") or 3)
            max_results = max(1, min(max_results, 10))

            if operation == "list_topics":
                payload = {{
                    "topics": [section.get("title", "") for section in REFERENCE_SECTIONS],
                    "count": len(REFERENCE_SECTIONS),
                }}
                return _completed(state, payload)

            query = topic or task
            if not query:
                return _pending(
                    "Reference adapter needs a `task` or `topic` to decide what guidance to return.",
                    state=state,
                    details={{"available_topics": [section.get("title", "") for section in REFERENCE_SECTIONS[:20]]}},
                )

            matches = _top_sections(query, topic, max_results)
            if not matches:
                return _pending(
                    "No matching reference sections found for the provided task/topic.",
                    state=state,
                    details={{"query": query, "available_topics": [section.get("title", "") for section in REFERENCE_SECTIONS[:20]]}},
                )

            payload = {{
                "query": query,
                "matches": [
                    {{
                        "title": section.get("title", ""),
                        "heading_path": section.get("heading_path", []),
                        "summary": section.get("summary", ""),
                        "body": section.get("body", ""),
                        "formulas": section.get("formulas", []),
                        "line": section.get("line"),
                    }}
                    for section in matches
                ],
            }}
            return _completed(state, payload)


        def resume_from_state(state: dict[str, Any]) -> dict[str, Any]:
            return run_until_blocked(state)


        def main() -> int:
            parser = argparse.ArgumentParser(description={description!r})
            parser.add_argument("--project", default=".")
            parser.add_argument("--config", default=None)
            parser.add_argument("--task", default="")
            parser.add_argument("--topic", default="")
            parser.add_argument("--operation", default="lookup")
            parser.add_argument("--max-results", type=int, default=3)
            args = parser.parse_args()
            _ = _load_runtime_config(args.config)
            failures = preflight_check(args.project)
            if failures:
                print(json.dumps({{"status": "blocked", "preflight": failures}}, ensure_ascii=False, indent=2))
                return 2
            state = {{
                "global_inputs": {{
                    "project_dir": args.project,
                    "task": args.task,
                    "topic": args.topic,
                    "operation": args.operation,
                    "max_results": args.max_results,
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
        "__SECTIONS__", repr(sections_payload)
    )


def _strip_template_margin(template: str, margin: str = "        ") -> str:
    lines = template.splitlines()
    stripped = [line[len(margin) :] if line.startswith(margin) else line for line in lines]
    return "\n".join(stripped).lstrip() + "\n"
