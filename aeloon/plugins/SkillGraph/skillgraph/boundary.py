"""Boundary metadata builders for SkillGraph artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PlanletSpec:
    id: str
    intent: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    bindings: dict[str, Any] = field(default_factory=dict)
    keywords: list[str] = field(default_factory=list)
    applicability: dict[str, Any] = field(default_factory=dict)
    stop_policy: dict[str, Any] = field(default_factory=dict)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ArtifactBoundarySpec:
    skill_name: str
    description: str
    boundary: str
    operators: list[dict[str, Any]] = field(default_factory=list)
    planlets: list[PlanletSpec] = field(default_factory=list)
    scope: dict[str, Any] = field(default_factory=dict)
    selection_policy: dict[str, Any] = field(default_factory=dict)
    execution_policy: dict[str, Any] = field(default_factory=dict)
    risk_flags: list[str] = field(default_factory=list)
    tool_schema: dict[str, Any] = field(default_factory=dict)

    def model_dump(self) -> dict[str, Any]:
        data = asdict(self)
        data["planlets"] = [
            planlet.model_dump() if hasattr(planlet, "model_dump") else dict(planlet)
            for planlet in self.planlets
        ]
        return data


def build_boundary_spec(
    *,
    strategy: str,
    skill_name: str,
    description: str,
    runtime_manifest: Any,
    package: Any,
    metadata: dict[str, Any] | None = None,
    dispatcher_capabilities: list[Any] | None = None,
    reference_sections: list[Any] | None = None,
) -> ArtifactBoundarySpec:
    del runtime_manifest, reference_sections
    metadata = metadata or {}
    operators = [
        cap.to_dict() if hasattr(cap, "to_dict") else dict(cap)
        for cap in dispatcher_capabilities or []
    ]
    package_slug = str(getattr(package, "slug", skill_name) or skill_name)
    generic_office = package_slug in {"docx", "pptx"} and strategy == "dispatcher"
    boundary = "typed_operator" if generic_office else "adapter"
    planlets = _office_planlets(package_slug) if generic_office else []
    selection_policy = {
        "direct_call_level": "operator_first" if generic_office else "inspect_first",
        "doc_read_policy": "optional" if generic_office else "required",
    }
    scope = {
        "kind": "skill_package_generic" if generic_office else "skill_package",
        "dynamic_sources_used_for_codegen": False,
    }
    classification = metadata.get("compilability") if isinstance(metadata, dict) else {}
    risk_flags: list[str] = []
    if isinstance(classification, dict) and classification.get("risk_flags"):
        risk_flags = [str(flag) for flag in classification.get("risk_flags") or []]

    return ArtifactBoundarySpec(
        skill_name=skill_name,
        description=description,
        boundary=boundary,
        operators=operators,
        planlets=planlets,
        scope=scope,
        selection_policy=selection_policy,
        execution_policy={"timeout_sec": 20, "allow_network_by_default": False},
        risk_flags=risk_flags,
        tool_schema=_tool_schema(operators, planlets),
    )


def normalize_boundary_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, ArtifactBoundarySpec):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    return {}


def summarize_boundary(value: Any) -> dict[str, Any]:
    data = normalize_boundary_metadata(value)
    return {
        "boundary": data.get("boundary", ""),
        "operator_count": len(data.get("operators") or []),
        "planlet_count": len(data.get("planlets") or []),
        "skill_name": data.get("skill_name", ""),
    }


def runtime_adapter_available(value: Any) -> bool:
    data = normalize_boundary_metadata(value)
    return bool(data.get("operators") or data.get("planlets"))


def build_boundary_cache_fingerprint(value: Any) -> str:
    payload = json.dumps(normalize_boundary_metadata(value), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _office_planlets(package_slug: str) -> list[PlanletSpec]:
    if package_slug == "docx":
        return [
            PlanletSpec(
                id="docx_template_fill_and_validate",
                intent="Fill and validate a DOCX template.",
                steps=[
                    {
                        "id": "inspect_template",
                        "operation": "inspect_docx",
                        "optional": True,
                        "arguments": {"path": "${input_path}"},
                    },
                    {
                        "id": "fill_template",
                        "operation": "replace_docx_text",
                        "arguments": {
                            "input_path": "${input_path}",
                            "output_path": "${output_path}",
                            "replacements": "${replacements}",
                        },
                    },
                    {
                        "id": "validate_text",
                        "operation": "validate_docx_text",
                        "arguments": {
                            "path": "${output_path}",
                            "must_contain": "${must_contain:[]}",
                            "must_not_contain": "${must_not_contain:[]}",
                        },
                    },
                ],
                bindings={
                    "required": ["input_path", "output_path", "replacements"],
                    "optional": ["must_contain", "must_not_contain"],
                    "autobind": {"input_path": ["files[0]"]},
                },
                keywords=["docx", "template", "fill"],
                applicability={"file_extensions": [".docx"]},
            )
        ]
    if package_slug == "pptx":
        return [
            PlanletSpec(
                id="pptx_update_and_validate",
                intent="Update a PPTX and validate the resulting deck.",
                steps=[
                    {
                        "id": "update_text",
                        "operation": "update_pptx_text_boxes",
                        "arguments": {
                            "input_pptx": "${input_pptx}",
                            "output_pptx": "${output_pptx}",
                            "edits": "${edits}",
                        },
                    },
                    {
                        "id": "validate_deck",
                        "operation": "validate_pptx_file",
                        "arguments": {"path": "${output_pptx}"},
                    },
                ],
                bindings={"required": ["input_pptx", "output_pptx", "edits"]},
                keywords=["pptx", "deck", "update"],
                applicability={"file_extensions": [".pptx"]},
            )
        ]
    return []


def _tool_schema(operators: list[dict[str, Any]], planlets: list[PlanletSpec]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": [
                    str(op.get("name") or op.get("id"))
                    for op in operators
                    if op.get("name") or op.get("id")
                ],
            },
            "planlet": {
                "type": "string",
                "enum": [planlet.id for planlet in planlets],
            },
            "arguments": {"type": "object"},
            "files": {"type": "array", "items": {"type": "string"}},
            "execute": {"type": "boolean"},
        },
        "required": [],
        "x-operator-schemas": [
            {
                "operation": str(op.get("name") or op.get("id")),
                "inputs": list(op.get("inputs") or []),
            }
            for op in operators
        ],
    }
