"""Discovery and loading of compiled workflows in the workspace."""

from __future__ import annotations

import importlib.util
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from loguru import logger

from aeloon.core.config.paths import get_storage_gateway


@dataclass
class WorkflowMetadata:
    name: str
    description: str
    path: Path
    global_inputs: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class LoadedWorkflow:
    metadata: WorkflowMetadata
    graph: Any
    module: ModuleType
    config_path: Path | None = None
    manifest_path: Path | None = None
    sandbox_path: Path | None = None


class WorkflowLoader:
    """Discover compiled workflows under the project compiled-skill cache."""

    def __init__(self, workspace: Path, directory_name: str = "compiled_skills") -> None:
        self.workspace = workspace
        if directory_name == "compiled_skills":
            self.directory = get_storage_gateway(workspace).project_compiled_skills_root(
                create=False
            )
            legacy_directory = workspace / "compiled_skills"
            self._legacy_directories = (
                [legacy_directory] if legacy_directory != self.directory else []
            )
        else:
            self.directory = workspace / directory_name
            self._legacy_directories = []
        self._loaded: dict[str, LoadedWorkflow] = {}
        self.refresh()

    def refresh(self) -> None:
        loaded: dict[str, LoadedWorkflow] = {}
        for directory in self._workflow_directories():
            for path in sorted(directory.glob("*_workflow.py")):
                workflow = self._load_one(path)
                if workflow is not None and workflow.metadata.name not in loaded:
                    loaded[workflow.metadata.name] = workflow
        self._loaded = loaded

    def list_workflows(self) -> list[WorkflowMetadata]:
        self.refresh()
        return [workflow.metadata for workflow in self._loaded.values()]

    def get_workflow(self, name: str) -> LoadedWorkflow | None:
        self.refresh()
        direct = self._loaded.get(name)
        if direct is not None:
            return direct
        want = self._canonical_name(name)
        for workflow_name, workflow in self._loaded.items():
            if self._canonical_name(workflow_name) == want:
                return workflow
        return None

    def build_summary(self) -> str:
        self.refresh()
        if not self._loaded:
            return ""
        lines = ["<compiled-workflows>"]
        for workflow in self._loaded.values():
            tool_name = f"run_{workflow.metadata.name}"
            lines.append(
                f'<workflow name="{workflow.metadata.name}" tool="{tool_name}" path="{workflow.metadata.path}">{workflow.metadata.description or "Compiled workflow"}'
            )
            for input_field in workflow.metadata.global_inputs:
                required = "true" if input_field.get("required", True) else "false"
                lines.append(
                    f'  <input name="{input_field.get("name", "")}" type="{input_field.get("type", "string")}" required="{required}">{input_field.get("description", "")}</input>'
                )
            lines.append("</workflow>")
        lines.append("</compiled-workflows>")
        return "\n".join(lines)

    def _workflow_directories(self) -> list[Path]:
        return [
            directory
            for directory in [self.directory, *self._legacy_directories]
            if directory.exists()
        ]

    def load_runtime_config(self, workflow: LoadedWorkflow) -> dict[str, Any]:
        if workflow.config_path is None or not workflow.config_path.exists():
            return {}
        try:
            raw = json.loads(workflow.config_path.read_text(encoding="utf-8") or "{}")
        except Exception:
            return {}
        if isinstance(raw, dict):
            runtime = raw.get("runtime", raw)
            return runtime if isinstance(runtime, dict) else {}
        return {}

    def run_preflight(self, workflow: LoadedWorkflow, project_dir: str) -> list[str]:
        preflight_fn = getattr(workflow.module, "preflight_check", None)
        if callable(preflight_fn):
            try:
                return list(preflight_fn(project_dir) or [])
            except Exception as exc:
                return [f"preflight_check raised: {exc}"]
        return []

    def execute(
        self,
        workflow: LoadedWorkflow,
        state: dict[str, Any],
        *,
        resume: bool = False,
    ) -> dict[str, Any]:
        if resume:
            resume_fn = getattr(workflow.module, "resume_from_state", None)
            if callable(resume_fn):
                return resume_fn(state)
        else:
            run_fn = getattr(workflow.module, "run_until_blocked", None)
            if callable(run_fn):
                return run_fn(state)
        return workflow.graph.invoke(state)

    def _load_one(self, path: Path) -> LoadedWorkflow | None:
        module_name = f"aeloon_compiled_{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            logger.warning("Failed to create spec for workflow {}", path)
            return None
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            logger.warning("Failed to import workflow {}: {}", path, exc)
            return None

        build_graph = getattr(module, "build_graph", None)
        if not callable(build_graph):
            logger.warning("Workflow {} missing build_graph()", path)
            return None

        meta = getattr(module, "SKILL_META", None) or {}
        name = str(meta.get("name") or path.stem.removesuffix("_workflow"))
        description = str(meta.get("description") or "")
        global_inputs = list(meta.get("global_inputs") or [])
        config_path = path.parent / "skill_config.json"
        manifest_path = path.with_suffix(".manifest.json")
        sandbox_path = path.with_suffix(".sandbox")

        return LoadedWorkflow(
            metadata=WorkflowMetadata(
                name=name,
                description=description,
                path=path,
                global_inputs=global_inputs,
            ),
            graph=build_graph(),
            module=module,
            config_path=config_path if config_path.exists() else None,
            manifest_path=manifest_path if manifest_path.exists() else None,
            sandbox_path=sandbox_path if sandbox_path.exists() else None,
        )

    @staticmethod
    def _canonical_name(name: str) -> str:
        text = (name or "").strip().lower()
        text = re.sub(r"[^a-z0-9_\-]+", "", text)
        return text.strip("_-")
