"""
SkillGraph: Compile AI agent skills (SKILL.md) into executable LangGraph workflows.

Usage:
    from aeloon.plugins.SkillGraph.skillgraph import compile

    # One-step: SKILL.md (or skill directory) → LangGraph .py file
    output = compile(
        skill_path="path/to/skill_or_SKILL_md",
        output_path="compiled/my_workflow.py",
        api_key="sk-...",
        base_url="https://openrouter.ai/api/v1",
        model="openai/gpt-5.4",
        runtime_model="openai/gpt-5.4",
        strict_validate=False,
    )

    # Two-step: analyze separately, then generate
    from aeloon.plugins.SkillGraph.skillgraph.analyzer import Analyzer
    from aeloon.plugins.SkillGraph.skillgraph.codegen import generate

    analyzer = Analyzer(model="openai/gpt-5.4", api_key="sk-...")
    graph = analyzer.analyze("path/to/SKILL.md", cache_path="cache/my_skill.json")
    generate(graph, "compiled/my_workflow.py", api_key="sk-...")
"""

from __future__ import annotations

import logging
from pathlib import Path

from .analyzer import Analyzer
from .codegen import generate
from .compilability import assess_compilability
from .dispatcher_codegen import generate_dispatcher
from .models import SkillGraph
from .normalize import normalize_graph
from .package import SkillPackage, build_skill_package
from .reference_codegen import generate_reference_adapter
from .report import CompileReport, build_report
from .sandbox import bootstrap_sandbox
from .validator import ValidationResult, validate_graph

logger = logging.getLogger(__name__)

__all__ = [
    "compile",
    "Analyzer",
    "generate",
    "SkillGraph",
    "normalize_graph",
    "SkillPackage",
    "CompileReport",
    "build_skill_package",
    "build_report",
    "ValidationResult",
    "validate_graph",
]


def compile(
    skill_path: str | Path,
    output_path: str | Path,
    api_key: str = "",
    base_url: str = "https://openrouter.ai/api/v1",
    model: str = "openai/gpt-5.4",
    runtime_model: str | None = None,
    cache_dir: str | Path | None = None,
    strict_validate: bool = False,
    report_path: str | Path | None = None,
    task_context_path: str | Path | None = None,
    verifier_command: str | None = None,
    trace_path: str | Path | None = None,
    target_outputs: list[str] | None = None,
    compile_goal: str | None = None,
    artifact_policy: str | None = None,
) -> Path:
    """
    Compile a SKILL.md (or skill directory) into a standalone LangGraph Python file.

    This is the main API — one call does everything:
    1. Parse SKILL.md
    2. Analyze with LLM → SkillGraph JSON (cached if available)
    3. Generate LangGraph Python code

    Args:
        skill_path: Path to SKILL.md or skill directory
        output_path: Where to write the generated .py file
        api_key: API key for LLM (analysis + runtime LLM nodes)
        base_url: OpenAI-compatible API base URL
        model: LLM model for analysis
        runtime_model: Optional model for generated runtime LLM nodes.
            Defaults to the analysis model.
        cache_dir: Directory to cache SkillGraph JSON (skips re-analysis)
        strict_validate: If true, fail compile on SkillGraph validation errors.
            If false (default), only warn and continue code generation.
        report_path: Optional explicit path for compile report JSON.
        task_context_path: Optional task instructions/tests path for task-aware compilation.
        verifier_command: Optional command that verifies generated outputs.
        trace_path: Optional successful trace to solidify into an adapter.
        target_outputs: Optional output files the compiled artifact should produce.
        compile_goal: Optional goal hint: guidance, adapter, workflow, or solver.
        artifact_policy: Optional evidence policy: generic, trace, or family.

    Returns:
        Path to the generated .py file
    """
    skill_path = Path(skill_path)
    output_path = Path(output_path)

    package = build_skill_package(skill_path)
    skill_root = Path(package.skill_root)
    entry_skill = skill_root / package.entry_skill

    logger.info(
        "Package: %s | entry=%s | assets=%d",
        package.slug,
        package.entry_skill,
        len(package.assets),
    )

    # Determine cache path
    cache_path = None
    use_cache = True
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{package.slug}.json"
        manifest_path = cache_dir / f"{package.slug}.manifest.json"

        if manifest_path.exists():
            try:
                previous = SkillPackage.load(manifest_path)
                if previous.package_hash != package.package_hash:
                    use_cache = False
                    logger.info("Package changed; bypassing cached graph: %s", cache_path)
            except Exception as e:
                use_cache = False
                logger.warning("Failed to load previous manifest (%s): %s", manifest_path, e)

        package.save(manifest_path)

    # Step 0: Compilability gate
    compilability = assess_compilability(package, entry_skill)
    if not compilability["compilable"]:
        raise ValueError(
            f"Skill '{package.slug}' is not suitable for graph compilation:\n"
            f"  Reason: {compilability['reason']}\n"
            f"  Suggestion: {compilability['suggestion']}"
        )
    logger.info(
        "Compilability: %s (kind=%s, strategy=%s, confidence=%s)",
        "OK",
        compilability.get("kind", "unknown"),
        compilability.get("strategy", "workflow"),
        compilability.get("confidence", "unknown"),
    )

    runtime_llm = runtime_model or model

    strategy = compilability.get("strategy")
    if strategy in {"dispatcher", "reference"}:
        if strategy == "dispatcher":
            result, runtime_manifest, _capabilities = generate_dispatcher(
                package=package,
                entry_skill=entry_skill,
                output_path=output_path,
                base_url=base_url,
                llm_model=runtime_llm,
            )
        else:
            result, runtime_manifest, _sections = generate_reference_adapter(
                package=package,
                entry_skill=entry_skill,
                output_path=output_path,
                base_url=base_url,
                llm_model=runtime_llm,
            )
        sandbox_result = bootstrap_sandbox(
            package,
            None,
            output_path,
            runtime_manifest=runtime_manifest,
        )
        if sandbox_result.status != "ready":
            failed = [check for check in sandbox_result.checks if not check.ok]
            if failed:
                details = "\n".join(f"- {check.name}: {check.detail}" for check in failed)
                logger.warning(
                    "Skill sandbox bootstrap incomplete for %s:\n%s", package.slug, details
                )
            else:
                logger.info("Skill sandbox bootstrap complete (no checks failed).")

        report_target = None
        if report_path:
            report_target = Path(report_path)
        elif cache_dir:
            report_target = Path(cache_dir) / f"{package.slug}.report.json"

        if report_target:
            report = build_report(
                SkillGraph(skill_name=package.slug),
                package,
                result,
                ValidationResult(),
                compilability=compilability,
                runtime_manifest=runtime_manifest,
            )
            report.save(report_target)
            logger.info("Report: %s", report_target)

        logger.info("Compiled %s artifact: %s", strategy, result)
        return result

    # Step 1+2: Analyze (with cache)
    analyzer = Analyzer(model=model, api_key=api_key, base_url=base_url)
    graph = analyzer.analyze(entry_skill, cache_path=cache_path, use_cache=use_cache)
    graph = normalize_graph(graph)

    logger.info(f"Graph: {len(graph.steps)} steps, {len(graph.edges)} edges")

    validation = validate_graph(graph)
    if validation.errors:
        details = "\n".join(f"- [{i.code}] {i.message}" for i in validation.errors)
        if strict_validate:
            raise ValueError(f"SkillGraph validation failed:\n{details}")
        logger.warning("SkillGraph validation warnings (continuing in loose mode):\n%s", details)

    if validation.warnings:
        warning_details = "\n".join(
            f"- [{i.code}] {i.message} ({i.step_id or 'graph'})" for i in validation.warnings
        )
        logger.warning("SkillGraph warnings:\n%s", warning_details)

    # Step 3: Prepare sandbox
    sandbox_result = bootstrap_sandbox(package, graph, output_path)
    if sandbox_result.status != "ready":
        failed = [c for c in sandbox_result.checks if not c.ok]
        if failed:
            details = "\n".join(f"- {c.name}: {c.detail}" for c in failed)
            logger.warning("Skill sandbox bootstrap incomplete for %s:\n%s", package.slug, details)
        else:
            logger.info("Skill sandbox bootstrap complete (no checks failed).")

    # Step 4: Generate code
    result = generate(graph, output_path, api_key=api_key, base_url=base_url, llm_model=runtime_llm)

    # Optional compile report
    report_target = None
    if report_path:
        report_target = Path(report_path)
    elif cache_dir:
        report_target = Path(cache_dir) / f"{package.slug}.report.json"

    if report_target:
        report = build_report(
            graph,
            package,
            result,
            validation,
            compilability=compilability,
        )
        report.save(report_target)
        logger.info("Report: %s", report_target)

    logger.info(f"Compiled: {result}")
    return result
