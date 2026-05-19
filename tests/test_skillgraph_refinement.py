from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from aeloon.plugins.SkillGraph.skillgraph.dispatcher_codegen import (
    DispatcherCapability,
    _apply_refined_dispatcher_payload,
)
from aeloon.plugins.SkillGraph.skillgraph.llm_refinement import (
    LLMRefinementResult,
    record_refinement_metadata,
)
from aeloon.plugins.SkillGraph.skillgraph.models import IOField
from aeloon.plugins.SkillGraph.skillgraph.reference_codegen import (
    ReferenceSection,
    _apply_refined_reference_payload,
)
from aeloon.plugins.SkillGraph.tools import WorkflowTool


def test_record_refinement_metadata_aggregates_token_usage() -> None:
    metadata = {
        "compiler": {
            "token_usage": {
                "analyzer": {
                    "prompt_tokens": 2,
                    "completion_tokens": 3,
                    "total_tokens": 5,
                    "llm_calls": 1,
                },
                "events": [],
            }
        }
    }
    result = LLMRefinementResult(
        mode="llm_refined",
        status="ok",
        usage={
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
            "llm_calls": 1,
        },
        events=[
            {
                "label": "dispatcher_llm_refine",
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
                "llm_calls": 1,
            }
        ],
    )

    record_refinement_metadata(
        metadata,
        kind="dispatcher",
        result=result,
        model="model",
        applied=True,
        issues=[],
    )

    compiler = metadata["compiler"]
    assert compiler["refinement"]["dispatcher"]["applied"] is True
    assert compiler["token_usage"]["refinement"]["total_tokens"] == 18
    assert compiler["token_usage"]["events"][0]["label"] == "dispatcher_llm_refine"


def test_dispatcher_refinement_rejects_unknown_operator_and_input() -> None:
    cap = DispatcherCapability(
        name="mesh_tool",
        title="Mesh Tool",
        kind="script_asset",
        summary="Static summary",
        inputs=[IOField(name="path", required=True)],
        commands=[],
        code_example="scripts/mesh.py",
        source_refs=[{"path": "scripts/mesh.py", "line": None, "snippet": "mesh"}],
        keywords=["mesh"],
        script_path="scripts/mesh.py",
    )

    refined, issues, applied = _apply_refined_dispatcher_payload(
        [cap],
        {
            "operators": [
                {
                    "source_name": "mesh_tool",
                    "name": "mesh_volume",
                    "summary": "Compute mesh volume from an existing bundled script.",
                    "inputs": [
                        {"name": "path", "description": "Input mesh path."},
                        {"name": "invented", "description": "Must be ignored."},
                    ],
                    "keywords": ["mesh", "volume"],
                    "confidence": 0.77,
                },
                {"source_name": "missing_tool", "name": "bad"},
            ]
        },
    )

    assert applied is True
    assert refined[0].name == "mesh_volume"
    assert refined[0].script_path == "scripts/mesh.py"
    assert [field.name for field in refined[0].inputs] == ["path"]
    assert refined[0].inputs[0].description == "Input mesh path."
    assert refined[0].validation_status == "llm_refined_validated"
    assert "ignored_unknown_operator:missing_tool" in issues
    assert "ignored_unknown_input:mesh_tool.invented" in issues


def test_reference_refinement_preserves_source_body_and_invalid_indexes() -> None:
    sections = [
        ReferenceSection(
            title="Alpha",
            heading_path=["Alpha"],
            summary="Alpha summary",
            body="Alpha body",
            line=1,
            formulas=["a+b"],
            keywords=["alpha"],
        ),
        ReferenceSection(
            title="Beta",
            heading_path=["Beta"],
            summary="Beta summary",
            body="Beta body",
            line=8,
            formulas=["c+d"],
            keywords=["beta"],
        ),
    ]

    refined, issues, applied = _apply_refined_reference_payload(
        sections,
        {
            "sections": [
                {
                    "source_indexes": [1, 2, 99],
                    "title": "Combined",
                    "summary": "Combined guidance",
                    "keywords": ["combined"],
                    "query_aliases": ["how to combine"],
                    "use_cases": ["combined lookup"],
                    "confidence": 0.9,
                }
            ]
        },
    )

    assert applied is True
    assert refined[0].title == "Combined"
    assert refined[0].body == "Alpha body\n\nBeta body"
    assert refined[0].formulas == ["a+b", "c+d"]
    assert refined[0].query_aliases == ["how to combine"]
    assert refined[0].validation_status == "llm_refined_validated"
    assert "ignored_invalid_source_index:99" in issues


@dataclass
class _Workflow:
    manifest_path: Path


class _Loader:
    def __init__(self, workflow: _Workflow) -> None:
        self.workflow = workflow

    def get_workflow(self, name: str) -> _Workflow:
        return self.workflow


def test_workflow_tool_parameters_uses_boundary_tool_schema(tmp_path: Path) -> None:
    schema = {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["inspect_docx"]},
            "arguments": {"type": "object"},
        },
        "x-operator-schemas": [{"operation": "inspect_docx"}],
    }
    manifest_path = tmp_path / "demo.manifest.json"
    manifest_path.write_text(
        json.dumps({"metadata": {"artifact_boundary": {"tool_schema": schema}}}),
        encoding="utf-8",
    )
    tool = WorkflowTool(
        loader=_Loader(_Workflow(manifest_path=manifest_path)),
        workflow_name="demo",
        provider=object(),
        model="model",
        workspace=str(tmp_path),
        state_store=object(),
    )

    assert tool.parameters == schema
