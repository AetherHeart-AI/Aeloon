"""Validation for grounded SkillGraph IR."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .manifest import extract_manifest
from .models import RuntimeManifest, SkillGraph, Step, StepType
from .normalize import command_has_template

SHELL_PREFIXES = (
    "openclaw ",
    "browser ",
    "bash ",
    "sh ",
    "cat ",
    "ls ",
    "echo ",
    "which ",
    "python ",
    "python3 ",
    "node ",
    "npm ",
    "pnpm ",
    "yarn ",
    "git ",
    "pip ",
    "pip3 ",
    "curl ",
    "wget ",
    "docker ",
    "docker-compose ",
    "go ",
    "govulncheck ",
    "cargo ",
    "make ",
    "./",
)


class ValidationIssue(BaseModel):
    level: str
    code: str
    message: str
    step_id: str = ""


class ValidationResult(BaseModel):
    errors: list[ValidationIssue] = Field(default_factory=list)
    warnings: list[ValidationIssue] = Field(default_factory=list)
    argv_backed_tool_steps: int = 0
    binding_backed_tool_steps: int = 0
    shell_string_only_steps: int = 0
    unresolved_tool_steps: int = 0

    def has_errors(self) -> bool:
        return bool(self.errors)

    def add_error(self, code: str, message: str, step_id: str = "") -> None:
        self.errors.append(
            ValidationIssue(level="error", code=code, message=message, step_id=step_id)
        )

    def add_warning(self, code: str, message: str, step_id: str = "") -> None:
        self.warnings.append(
            ValidationIssue(level="warning", code=code, message=message, step_id=step_id)
        )


def validate_graph(graph: SkillGraph) -> ValidationResult:
    result = ValidationResult()

    for err in graph.validate():
        result.add_error("graph.invalid", err)

    global_input_names = {i.name for i in graph.global_inputs}
    step_map = {s.id: s for s in graph.steps}

    for step in graph.steps:
        _validate_step_execution(step, result)
        _validate_required_inputs(step, graph, step_map, global_input_names, result)

    _compute_execution_metrics(graph, result)

    manifest = extract_manifest(graph)
    _validate_manifest(manifest, result)

    return result


def workflow_blocking_issues(result: ValidationResult) -> list[ValidationIssue]:
    """Return validation issues that should block executable workflow builds."""
    return list(result.errors)


def _validate_step_execution(step: Step, result: ValidationResult) -> None:
    if step.step_type == StepType.TOOL_CALL:
        if not step.execution_spec or not step.execution_spec.command:
            result.add_warning(
                "step.tool.ungrounded",
                "tool_call step has no grounded execution command",
                step.id,
            )
        elif step.execution_spec.kind.value == "shell":
            cmd = step.execution_spec.command.strip().lower()
            if not cmd.startswith(SHELL_PREFIXES):
                result.add_warning(
                    "step.tool.suspicious_command",
                    f"shell command does not look executable: {step.execution_spec.command[:120]}",
                    step.id,
                )
            if command_has_template(step.execution_spec.command):
                result.add_error(
                    "step.tool.unresolved_template",
                    "tool_call step contains unresolved command template markers",
                    step.id,
                )

    if step.step_type == StepType.LLM_GENERATE:
        if step.execution_spec and step.execution_spec.kind.value != "llm":
            result.add_warning(
                "step.llm.kind_mismatch",
                "llm_generate step execution kind is not llm",
                step.id,
            )

    if (
        step.step_type == StepType.TOOL_CALL
        and step.execution_spec
        and step.execution_spec.kind.value == "noop"
    ):
        result.add_warning(
            "step.tool.noop",
            "tool_call step normalized to noop because no deterministic command survived normalization",
            step.id,
        )

    if step.step_type in {StepType.CONDITION, StepType.DATA_TRANSFORM}:
        if not step.execution_spec:
            result.add_warning(
                "step.det.missing_exec_spec",
                "deterministic step has no execution_spec",
                step.id,
            )

    if step.risk_level == "high" and not step.guards:
        result.add_warning(
            "step.risk.unguarded",
            "high-risk step has no guard policy",
            step.id,
        )


def _validate_required_inputs(
    step: Step,
    graph: SkillGraph,
    step_map: dict[str, Step],
    global_input_names: set[str],
    result: ValidationResult,
) -> None:
    upstream_ids = graph.upstream_of(step.id)
    upstream_outputs: set[str] = set()
    for sid in upstream_ids:
        upstream = step_map.get(sid)
        if not upstream:
            continue
        upstream_outputs.update(o.name for o in upstream.outputs)

    for inp in step.inputs:
        if not inp.required:
            continue
        if inp.name in global_input_names:
            continue
        if inp.name in upstream_outputs:
            continue
        result.add_warning(
            "step.input.unresolved",
            f"required input '{inp.name}' not provided by globals or upstream outputs",
            step.id,
        )


def _compute_execution_metrics(graph: SkillGraph, result: ValidationResult) -> None:
    for step in graph.steps:
        if step.step_type != StepType.TOOL_CALL:
            continue
        spec = step.execution_spec
        if not spec or (not spec.command and not spec.argv):
            result.unresolved_tool_steps += 1
            continue
        if spec.argv:
            result.argv_backed_tool_steps += 1
            if spec.arg_bindings:
                result.binding_backed_tool_steps += 1
        elif spec.command:
            result.shell_string_only_steps += 1


def _validate_manifest(manifest: RuntimeManifest, result: ValidationResult) -> None:
    """Warn about potentially problematic runtime dependencies."""
    for dep in manifest.dependencies:
        if dep.kind == "script_file" and dep.required:
            result.add_warning(
                "manifest.script_file",
                f"workflow requires script file '{dep.name}' at runtime "
                f"(used by: {', '.join(dep.source_step_ids)})",
            )
        if dep.kind == "env_var" and dep.required:
            result.add_warning(
                "manifest.env_var",
                f"workflow requires environment variable '{dep.name}' "
                f"(used by: {', '.join(dep.source_step_ids)})",
            )
