"""LLM refinement metadata helpers for SkillGraph compilation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMRefinementResult:
    mode: str
    status: str
    data: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, int] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""


def record_refinement_metadata(
    metadata: dict[str, Any] | None,
    *,
    kind: str,
    result: LLMRefinementResult,
    model: str,
    applied: bool,
    issues: list[str],
) -> None:
    """Attach refinement status and token usage to the compiler metadata."""

    if metadata is None:
        return
    compiler = metadata.setdefault("compiler", {})
    refinement = compiler.setdefault("refinement", {})
    refinement[kind] = {
        "mode": result.mode,
        "status": result.status,
        "model": model,
        "applied": applied,
        "issues": list(issues),
    }
    if result.error:
        refinement[kind]["error"] = result.error

    token_usage = compiler.setdefault("token_usage", {})
    aggregate = token_usage.setdefault(
        "refinement",
        {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "llm_calls": 0,
        },
    )
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "llm_calls"):
        aggregate[key] = int(aggregate.get(key) or 0) + int(result.usage.get(key) or 0)
    token_usage.setdefault("events", []).extend(result.events)


def call_json_refiner(**_: Any) -> LLMRefinementResult:
    """Placeholder refiner used when the full Pro LLM refinement loop is unavailable."""

    return LLMRefinementResult(mode="disabled", status="skipped")
