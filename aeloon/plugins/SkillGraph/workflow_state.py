"""Persistent state store for resumable compiled workflow executions."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from aeloon.core.config.paths import get_storage_gateway


@dataclass
class WorkflowExecutionState:
    workflow_run_id: str
    workflow_name: str
    session_key: str
    status: str
    graph_state: dict[str, Any] = field(default_factory=dict)
    current_step: str | None = None
    block: dict[str, Any] | None = None
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_run_id": self.workflow_run_id,
            "workflow_name": self.workflow_name,
            "session_key": self.session_key,
            "status": self.status,
            "graph_state": _sanitize_json(self.graph_state),
            "current_step": self.current_step,
            "block": _sanitize_json(self.block),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowExecutionState":
        return cls(
            workflow_run_id=str(data.get("workflow_run_id", "")),
            workflow_name=str(data.get("workflow_name", "")),
            session_key=str(data.get("session_key", "")),
            status=str(data.get("status", "")),
            graph_state=dict(data.get("graph_state") or {}),
            current_step=data.get("current_step"),
            block=data.get("block"),
            updated_at=str(data.get("updated_at") or datetime.now().isoformat()),
        )


class WorkflowStateStore:
    """File-backed state store for blocked/running workflow executions."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.root = get_storage_gateway(workspace).project_workflows_root()
        self.root.mkdir(parents=True, exist_ok=True)

    def create(
        self,
        *,
        workflow_name: str,
        session_key: str,
        graph_state: dict[str, Any],
        status: str,
        current_step: str | None = None,
        block: dict[str, Any] | None = None,
    ) -> WorkflowExecutionState:
        run_id = f"wf_{uuid.uuid4().hex[:12]}"
        state = WorkflowExecutionState(
            workflow_run_id=run_id,
            workflow_name=workflow_name,
            session_key=session_key,
            status=status,
            graph_state=graph_state,
            current_step=current_step,
            block=block,
        )
        self.save(state)
        return state

    def save(self, state: WorkflowExecutionState) -> None:
        path = self._path(state.session_key, state.workflow_run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        state.updated_at = datetime.now().isoformat()
        path.write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, session_key: str, workflow_run_id: str) -> WorkflowExecutionState | None:
        path = self._path(session_key, workflow_run_id)
        if not path.exists():
            return None
        try:
            return WorkflowExecutionState.from_dict(
                json.loads(path.read_text(encoding="utf-8") or "{}")
            )
        except Exception:
            return None

    def latest_blocked(self, session_key: str) -> WorkflowExecutionState | None:
        session_dir = self.root / self._safe_session(session_key)
        if not session_dir.exists():
            return None
        newest: tuple[float, WorkflowExecutionState] | None = None
        for path in session_dir.glob("*.json"):
            state = self.load(session_key, path.stem)
            if state is None or state.status != "blocked":
                continue
            mtime = path.stat().st_mtime
            if newest is None or mtime > newest[0]:
                newest = (mtime, state)
        return newest[1] if newest else None

    def _path(self, session_key: str, workflow_run_id: str) -> Path:
        return self.root / self._safe_session(session_key) / f"{workflow_run_id}.json"

    @staticmethod
    def _safe_session(session_key: str) -> str:
        return session_key.replace(":", "__") or "default"


def _sanitize_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if key == "_llm_callable":
                continue
            cleaned[str(key)] = _sanitize_json(item)
        return cleaned
    if isinstance(value, list):
        return [_sanitize_json(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_json(item) for item in value]
    return repr(value)
