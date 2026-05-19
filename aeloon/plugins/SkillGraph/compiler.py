"""Compile workspace skills into compiled workflow runtimes."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aeloon.core.config.paths import get_storage_gateway
from aeloon.providers.base import LLMProvider


@dataclass(frozen=True)
class SkillCompilerRequest:
    skill_path: str
    model: str | None = None
    runtime_model: str | None = None
    strict_validate: bool = False


@dataclass(frozen=True)
class SkillCompilerResult:
    skill_path: Path
    package_slug: str
    workflow_name: str
    output_path: Path
    manifest_path: Path
    sandbox_path: Path
    report_path: Path
    config_path: Path
    model: str
    runtime_model: str
    base_url: str


def build_skill_compiler_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="/skill_compiler", add_help=False)
    parser.add_argument("skill_path")
    parser.add_argument("--model", default="")
    parser.add_argument("--runtime-model", default="")
    parser.add_argument("--strict-validate", action="store_true")
    return parser


def compile_skill_to_workspace(
    *,
    workspace: Path,
    provider: LLMProvider,
    default_model: str,
    request: SkillCompilerRequest,
) -> SkillCompilerResult:
    api = _load_skillgraph_api()
    resolved_skill_path = _resolve_skill_path(workspace, request.skill_path)
    package = api.build_skill_package(resolved_skill_path)
    storage = get_storage_gateway(workspace)
    compiled_dir = storage.project_compiled_skills_root()
    cache_dir = storage.project_cache_root("skillgraph")

    output_path = compiled_dir / f"{package.slug}_workflow.py"
    report_path = cache_dir / f"{package.slug}.report.json"
    model = request.model or default_model
    runtime_model = request.runtime_model or default_model
    api_key = provider.api_key or ""
    base_url = provider.api_base or "https://openrouter.ai/api/v1"

    api.compile_skill(
        skill_path=resolved_skill_path,
        output_path=output_path,
        api_key=api_key,
        base_url=base_url,
        model=model,
        runtime_model=runtime_model,
        cache_dir=cache_dir,
        strict_validate=request.strict_validate,
        report_path=report_path,
    )

    workflow_name = _discover_workflow_name(output_path)
    return SkillCompilerResult(
        skill_path=resolved_skill_path,
        package_slug=package.slug,
        workflow_name=workflow_name,
        output_path=output_path,
        manifest_path=output_path.with_suffix(".manifest.json"),
        sandbox_path=output_path.with_suffix(".sandbox"),
        report_path=report_path,
        config_path=compiled_dir / "skill_config.json",
        model=model,
        runtime_model=runtime_model,
        base_url=base_url,
    )


def format_skill_compiler_success(result: SkillCompilerResult, refreshed: bool) -> str:
    lines = [
        "Skill compiled successfully.",
        f"- source: {result.skill_path}",
        f"- package_slug: {result.package_slug}",
        f"- workflow_name: {result.workflow_name}",
        f"- tool: run_{result.workflow_name}",
        f"- output: {result.output_path}",
        f"- manifest: {result.manifest_path}",
        f"- sandbox: {result.sandbox_path}",
        f"- report: {result.report_path}",
        f"- runtime config: {result.config_path}",
        f"- analyzer model: {result.model}",
        f"- runtime model: {result.runtime_model}",
        f"- workflow tools refreshed: {'yes' if refreshed else 'no'}",
        f"You can use it next with `run_{result.workflow_name}`.",
    ]
    return "\n".join(lines)


@dataclass(frozen=True)
class _SkillgraphApi:
    build_skill_package: Any
    compile_skill: Any


def _resolve_skill_path(workspace: Path, raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    workspace_candidate = workspace / candidate
    if workspace_candidate.exists():
        return workspace_candidate.resolve()

    project_skills_root = get_storage_gateway(workspace).project_skills_root(create=False)
    parts = candidate.parts
    if parts and parts[0] == "skills":
        project_skill_candidate = project_skills_root.joinpath(*parts[1:])
        if project_skill_candidate.exists():
            return project_skill_candidate.resolve()

    direct_project_skill_candidate = project_skills_root / candidate
    if direct_project_skill_candidate.exists():
        return direct_project_skill_candidate.resolve()

    return workspace_candidate.resolve()


def _discover_workflow_name(output_path: Path) -> str:
    module_name = f"aeloon_skill_compiler_{output_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, output_path)
    if spec is None or spec.loader is None:
        return output_path.stem.removesuffix("_workflow")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    meta = getattr(module, "SKILL_META", None) or {}
    return str(meta.get("name") or output_path.stem.removesuffix("_workflow"))


def _load_skillgraph_api() -> _SkillgraphApi:
    try:
        embedded_skillgraph = importlib.import_module("aeloon.plugins.SkillGraph.skillgraph")
    except ImportError as exc:  # pragma: no cover - exercised via tests with monkeypatch
        raise RuntimeError(
            "embedded SkillGraph compiler is not available. Ensure optional compiler dependencies are installed."
        ) from exc
    return _SkillgraphApi(
        build_skill_package=getattr(embedded_skillgraph, "build_skill_package"),
        compile_skill=getattr(embedded_skillgraph, "compile"),
    )
