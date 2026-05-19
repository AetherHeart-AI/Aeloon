"""Tools for running and resuming compiled workflows inside the agent loop."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from aeloon.core.agent.tools.base import Tool
from aeloon.core.agent.turn import TurnContext
from aeloon.plugins.SkillGraph.workflow_bridge import make_llm_callable
from aeloon.plugins.SkillGraph.workflow_loader import WorkflowLoader
from aeloon.plugins.SkillGraph.workflow_state import WorkflowExecutionState, WorkflowStateStore
from aeloon.providers.base import LLMProvider


class _BaseWorkflowTool(Tool):
    def __init__(self, *, loader: WorkflowLoader, state_store: WorkflowStateStore) -> None:
        self._loader = loader
        self._state_store = state_store
        self._turn_ctx: TurnContext | None = None

    def on_turn_start(self, ctx: TurnContext) -> None:
        self._turn_ctx = ctx
        super().on_turn_start(ctx)

    def _session_key(self) -> str:
        if self._turn_ctx and self._turn_ctx.session_key:
            return self._turn_ctx.session_key
        if self._turn_ctx:
            return f"{self._turn_ctx.channel}:{self._turn_ctx.chat_id}"
        return "default"


class WorkflowTool(_BaseWorkflowTool):
    def __init__(
        self,
        *,
        loader: WorkflowLoader,
        workflow_name: str,
        provider: LLMProvider,
        model: str,
        workspace: str,
        state_store: WorkflowStateStore,
    ) -> None:
        super().__init__(loader=loader, state_store=state_store)
        self._workflow_name = workflow_name
        self._provider = provider
        self._model = model
        self._workspace = workspace

    @property
    def name(self) -> str:
        return f"run_{self._workflow_name}"

    @property
    def description(self) -> str:
        workflow = self._loader.get_workflow(self._workflow_name)
        if workflow is None:
            return f"Run the compiled workflow '{self._workflow_name}'."
        required_inputs = [
            field.get("name", "")
            for field in workflow.metadata.global_inputs
            if field.get("name") not in {"project_dir"} and field.get("required", True)
        ]
        required_text = (
            f" Required inputs: {', '.join(required_inputs)}." if required_inputs else ""
        )
        boundary = _workflow_boundary(workflow)
        strategy_hint = _workflow_strategy_hint(boundary)
        planlet_hint = _workflow_planlet_hint(boundary)
        return (
            f"Run the compiled workflow '{self._workflow_name}' when it exactly matches the task."
            f" {workflow.metadata.description or ''}{required_text} "
            "Pass boundary fields such as `planlet`, `operation`, `arguments`, and `execute` directly, "
            "or put them under `inputs`. "
            f"{strategy_hint} {planlet_hint} "
            "If the workflow becomes blocked, repair the issue with normal tools and then call `resume_workflow`."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        workflow = self._loader.get_workflow(self._workflow_name)
        if workflow is not None:
            boundary = _workflow_boundary(workflow)
            tool_schema = boundary.get("tool_schema") if isinstance(boundary, dict) else None
            if isinstance(tool_schema, dict) and tool_schema.get("type") == "object":
                return tool_schema
        return {
            "type": "object",
            "properties": {
                "inputs": {"type": "object"},
                "request": {
                    "type": "string",
                    "description": "Natural-language request for operator selection.",
                },
                "task": {
                    "type": "string",
                    "description": "Optional task summary for operator selection.",
                },
                "planlet": {
                    "type": "string",
                    "description": "Optional reusable skill-level planlet to execute when its bindings are available.",
                },
                "operation": {
                    "type": "string",
                    "description": "Exact compiled boundary operation to run.",
                },
                "arguments": {
                    "type": "object",
                    "description": "Structured arguments for the selected operation.",
                },
                "execute": {
                    "type": "boolean",
                    "description": "Set true to execute adapter or operator workflows when inputs are available.",
                },
                "allow_network": {
                    "type": "boolean",
                    "description": "Explicitly allow network-risk operations.",
                },
                "files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional workspace files relevant to the operation.",
                },
            },
            "required": [],
        }

    async def execute(self, inputs: dict[str, Any] | None = None, **kwargs: Any) -> str:
        workflow_name = self._workflow_name
        workflow = self._loader.get_workflow(workflow_name)
        if workflow is None:
            available = (
                ", ".join(sorted(item.name for item in self._loader.list_workflows())) or "none"
            )
            return json.dumps(
                {
                    "status": "failed",
                    "workflow_name": workflow_name,
                    "error": f"workflow_not_found: {workflow_name}",
                    "available": available,
                },
                ensure_ascii=False,
            )

        runtime_inputs = dict(inputs) if isinstance(inputs, dict) else {}
        runtime_inputs.update({key: value for key, value in kwargs.items() if value is not None})

        runtime_cfg = self._loader.load_runtime_config(workflow)
        sandbox_dir = str(workflow.sandbox_path) if workflow.sandbox_path else ""
        state = {
            "global_inputs": {
                "project_dir": self._workspace,
                "sandbox_dir": sandbox_dir,
                "_runtime_config": runtime_cfg,
                "_llm_callable": make_llm_callable(self._provider, self._model),
                **runtime_inputs,
            },
            "step_results": {},
            "error": None,
            "final_output": None,
        }
        return await self._run_and_wrap(workflow_name=workflow_name, workflow=workflow, state=state)

    async def _run_and_wrap(
        self,
        *,
        workflow_name: str,
        workflow: Any,
        state: dict[str, Any],
        workflow_run_id: str | None = None,
    ) -> str:
        failures = self._loader.run_preflight(workflow, self._workspace)
        if failures:
            block = {
                "message": "Workflow preflight failed because required runtime dependencies are missing.",
                "details": {"preflight": failures},
                "suggested_actions": [
                    "Use normal tools to install or configure the missing dependency.",
                    "Then call resume_workflow with the workflow_run_id.",
                ],
            }
            workflow_state = self._save_state(
                workflow_name=workflow_name,
                status="blocked",
                graph_state=state,
                block=block,
                workflow_run_id=workflow_run_id,
            )
            return json.dumps(
                {
                    "status": "blocked",
                    "workflow_name": workflow_name,
                    "workflow_run_id": workflow_state.workflow_run_id,
                    "current_step": None,
                    "final_output": None,
                    "step_results": state.get("step_results", {}),
                    "block": block,
                },
                ensure_ascii=False,
            )

        try:
            result = await asyncio.to_thread(
                self._loader.execute,
                workflow,
                state,
                resume=bool(workflow_run_id),
            )
        except Exception as exc:
            block = {
                "message": f"Workflow raised an exception: {exc}",
                "details": {"exception": str(exc)},
                "suggested_actions": [
                    "Inspect the error, use normal tools to repair the environment, then call resume_workflow.",
                ],
            }
            workflow_state = self._save_state(
                workflow_name=workflow_name,
                status="blocked",
                graph_state=state,
                block=block,
                workflow_run_id=workflow_run_id,
            )
            return json.dumps(
                {
                    "status": "blocked",
                    "workflow_name": workflow_name,
                    "workflow_run_id": workflow_state.workflow_run_id,
                    "current_step": None,
                    "final_output": None,
                    "step_results": state.get("step_results", {}),
                    "block": block,
                },
                ensure_ascii=False,
            )

        if isinstance(result, dict) and result.get("status") in {"blocked", "completed", "failed"}:
            return self._wrap_envelope_result(
                workflow_name=workflow_name,
                result=result,
                workflow_run_id=workflow_run_id,
            )

        if result.get("error"):
            block = {
                "message": result.get("error"),
                "details": {"error": result.get("error")},
                "suggested_actions": [
                    "Inspect the failing step and repair the issue using normal tools, then call resume_workflow.",
                ],
            }
            workflow_state = self._save_state(
                workflow_name=workflow_name,
                status="blocked",
                graph_state=result,
                block=block,
                workflow_run_id=workflow_run_id,
            )
            return json.dumps(
                {
                    "status": "blocked",
                    "workflow_name": workflow_name,
                    "workflow_run_id": workflow_state.workflow_run_id,
                    "current_step": None,
                    "final_output": result.get("final_output"),
                    "step_results": result.get("step_results", {}),
                    "block": block,
                },
                ensure_ascii=False,
            )

        workflow_state = self._save_state(
            workflow_name=workflow_name,
            status="completed",
            graph_state=result,
            block=None,
            workflow_run_id=workflow_run_id,
        )
        return json.dumps(
            {
                "status": "completed",
                "workflow_name": workflow_name,
                "workflow_run_id": workflow_state.workflow_run_id,
                "current_step": None,
                "final_output": result.get("final_output"),
                "step_results": result.get("step_results", {}),
                "block": None,
            },
            ensure_ascii=False,
        )

    def _wrap_envelope_result(
        self,
        *,
        workflow_name: str,
        result: dict[str, Any],
        workflow_run_id: str | None,
    ) -> str:
        status = str(result.get("status") or "failed")
        graph_state = dict(result.get("graph_state") or {})
        graph_state.setdefault("global_inputs", result.get("global_inputs") or {})
        graph_state.setdefault("step_results", result.get("step_results") or {})
        graph_state.setdefault("error", result.get("error"))
        graph_state.setdefault("final_output", result.get("final_output"))
        current_step = result.get("current_step")
        block = result.get("block") if isinstance(result.get("block"), dict) else None
        workflow_state = self._save_state(
            workflow_name=workflow_name,
            status=status,
            graph_state=graph_state,
            block=block,
            workflow_run_id=workflow_run_id,
            current_step=current_step,
        )
        payload = {
            "status": status,
            "workflow_name": workflow_name,
            "workflow_run_id": workflow_state.workflow_run_id,
            "current_step": current_step,
            "final_output": result.get("final_output"),
            "step_results": result.get("step_results", {}),
            "block": block,
        }
        if result.get("error") and not payload["block"]:
            payload["block"] = {
                "message": str(result.get("error")),
                "details": {"error": result.get("error")},
                "suggested_actions": [
                    "Inspect the error, repair it with normal tools, then call resume_workflow.",
                ],
            }
        return json.dumps(payload, ensure_ascii=False)

    def _save_state(
        self,
        *,
        workflow_name: str,
        status: str,
        graph_state: dict[str, Any],
        block: dict[str, Any] | None,
        workflow_run_id: str | None,
        current_step: str | None = None,
    ) -> WorkflowExecutionState:
        if workflow_run_id:
            state = WorkflowExecutionState(
                workflow_run_id=workflow_run_id,
                workflow_name=workflow_name,
                session_key=self._session_key(),
                status=status,
                graph_state=graph_state,
                current_step=current_step,
                block=block,
            )
            self._state_store.save(state)
            return state
        return self._state_store.create(
            workflow_name=workflow_name,
            session_key=self._session_key(),
            graph_state=graph_state,
            status=status,
            current_step=current_step,
            block=block,
        )


def _workflow_boundary(workflow: Any) -> dict[str, Any]:
    manifest_path = getattr(workflow, "manifest_path", None)
    if manifest_path is not None and manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8") or "{}")
        except Exception:
            manifest = {}
        metadata = manifest.get("metadata") if isinstance(manifest, dict) else {}
        boundary = metadata.get("artifact_boundary") if isinstance(metadata, dict) else {}
        if isinstance(boundary, dict) and boundary:
            return boundary
    boundary = getattr(getattr(workflow, "module", None), "ARTIFACT_BOUNDARY", None)
    return boundary if isinstance(boundary, dict) else {}


def _workflow_strategy_hint(boundary: dict[str, Any]) -> str:
    if not boundary:
        return ""
    boundary_kind = str(boundary.get("boundary") or "adapter")
    selection = boundary.get("selection_policy") if isinstance(boundary, dict) else {}
    direct_level = str((selection or {}).get("direct_call_level") or "")
    doc_policy = str((selection or {}).get("doc_read_policy") or "")
    ops = _boundary_operation_names(boundary)
    op_text = f" Available operations: {', '.join(ops[:8])}." if ops else ""
    if boundary_kind == "adapter" and direct_level == "inspect_first":
        return (
            "Boundary policy: adapter/inspect_first. Prefer `search_skill_docs` or "
            "`get_skill_asset` with `arguments`; do not set `execute=true` for workspace "
            "files unless you intentionally selected an executable operation with all "
            f"required arguments.{op_text}"
        )
    if direct_level in {"operator_first", "solver_first"} or boundary_kind in {
        "typed_operator",
        "workflow_solver",
        "jit_solver",
    }:
        return (
            f"Boundary policy: {boundary_kind}/{direct_level or 'operator_first'}. "
            "Prefer an exact `operation` plus structured `arguments`; pass workspace "
            "files through `arguments` and/or `files` instead of relying only on a "
            f"natural-language request.{op_text}"
        )
    if doc_policy == "required":
        return (
            f"Boundary policy: {boundary_kind}; read targeted preserved guidance first "
            f"with `search_skill_docs` or `get_skill_asset` before manual edits.{op_text}"
        )
    return f"Boundary policy: {boundary_kind}.{op_text}"


def _workflow_planlet_hint(boundary: dict[str, Any]) -> str:
    planlets = [item for item in boundary.get("planlets") or [] if isinstance(item, dict)]
    if not planlets:
        return ""
    compact = []
    for planlet in planlets[:4]:
        bindings = planlet.get("bindings") if isinstance(planlet.get("bindings"), dict) else {}
        required = ", ".join(str(item) for item in bindings.get("required") or [])
        required_text = f" requires: {required}" if required else ""
        compact.append(f"{planlet.get('id')} ({planlet.get('intent')}{required_text})")
    return (
        "Reusable planlets available: "
        + "; ".join(str(item) for item in compact if item)
        + ". Prefer a matching `planlet` over manually chaining operations when bindings are known."
    )


def _boundary_operation_names(boundary: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for op in boundary.get("operators") or []:
        if not isinstance(op, dict):
            continue
        name = str(op.get("name") or op.get("id") or "")
        if name and name not in names:
            names.append(name)
    return names


class ResumeWorkflowTool(_BaseWorkflowTool):
    def __init__(
        self,
        *,
        loader: WorkflowLoader,
        provider: LLMProvider,
        model: str,
        workspace: str,
        state_store: WorkflowStateStore,
    ) -> None:
        super().__init__(loader=loader, state_store=state_store)
        self._provider = provider
        self._model = model
        self._workspace = workspace

    @property
    def name(self) -> str:
        return "resume_workflow"

    @property
    def description(self) -> str:
        return (
            "Resume a previously blocked compiled workflow after fixing configuration, dependencies, or runtime issues. "
            "Use the workflow_run_id returned by a prior `run_<workflow_name>` call."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "workflow_run_id": {"type": "string"},
                "inputs": {"type": "object"},
            },
            "required": [],
        }

    async def execute(
        self,
        workflow_run_id: str | None = None,
        inputs: dict[str, Any] | None = None,
        **_: Any,
    ) -> str:
        workflow_run_id = workflow_run_id or ""
        state = (
            self._state_store.load(self._session_key(), workflow_run_id)
            if workflow_run_id
            else None
        )
        if state is None:
            latest = self._state_store.latest_blocked(self._session_key())
            if latest is not None and not workflow_run_id:
                state = latest
                workflow_run_id = latest.workflow_run_id
            else:
                latest_id = latest.workflow_run_id if latest else None
                return json.dumps(
                    {
                        "status": "failed",
                        "error": f"workflow_state_not_found: {workflow_run_id}",
                        "latest_blocked": latest_id,
                    },
                    ensure_ascii=False,
                )
        workflow = self._loader.get_workflow(state.workflow_name)
        if workflow is None:
            return json.dumps(
                {
                    "status": "failed",
                    "workflow_name": state.workflow_name,
                    "workflow_run_id": workflow_run_id,
                    "error": f"workflow_not_found: {state.workflow_name}",
                },
                ensure_ascii=False,
            )

        graph_state = dict(state.graph_state or {})
        globals_in = dict(graph_state.get("global_inputs") or {})
        runtime_cfg = self._loader.load_runtime_config(workflow)
        globals_in.update(
            {
                "project_dir": self._workspace,
                "sandbox_dir": str(workflow.sandbox_path) if workflow.sandbox_path else "",
                "_runtime_config": runtime_cfg,
                "_llm_callable": make_llm_callable(self._provider, self._model),
            }
        )
        if inputs:
            globals_in.update(inputs)
        graph_state["global_inputs"] = globals_in
        graph_state["error"] = None

        runner = WorkflowTool(
            loader=self._loader,
            workflow_name=state.workflow_name,
            provider=self._provider,
            model=self._model,
            workspace=self._workspace,
            state_store=self._state_store,
        )
        if self._turn_ctx:
            runner.on_turn_start(self._turn_ctx)
        return await runner._run_and_wrap(
            workflow_name=state.workflow_name,
            workflow=workflow,
            state=graph_state,
            workflow_run_id=workflow_run_id,
        )
