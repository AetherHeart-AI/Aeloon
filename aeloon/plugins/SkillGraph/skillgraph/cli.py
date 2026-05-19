from __future__ import annotations

import argparse
import os
from pathlib import Path

from aeloon.core.config.paths import get_storage_gateway

from . import (
    Analyzer,
    SkillPackage,
    build_report,
    build_skill_package,
    normalize_graph,
    validate_graph,
)
from . import (
    compile as compile_skill,
)
from .validator import workflow_blocking_issues


def _prepare_cache(package, cache_dir: Path) -> tuple[Path, bool, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{package.slug}.json"
    manifest_path = cache_dir / f"{package.slug}.manifest.json"
    use_cache = True

    if manifest_path.exists():
        try:
            previous = SkillPackage.load(manifest_path)
            if previous.package_hash != package.package_hash:
                use_cache = False
        except Exception:
            use_cache = False

    package.save(manifest_path)
    return cache_path, use_cache, manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile SKILL.md (or a skill directory) into a standalone LangGraph workflow."
    )
    parser.add_argument("skill", help="Path to SKILL.md or a skill directory")
    parser.add_argument(
        "--workspace",
        default="",
        help="Aeloon workspace used for default project cache exits (default: current directory)",
    )
    parser.add_argument(
        "-o", "--output", default="", help="Output Python file path (required for full compile)"
    )
    parser.add_argument("--model", default="openai/gpt-5.4", help="Analyzer model")
    parser.add_argument(
        "--runtime-model", default="", help="Optional runtime model for generated LLM nodes"
    )
    parser.add_argument(
        "--base-url", default="https://openrouter.ai/api/v1", help="OpenAI-compatible base URL"
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENROUTER_API_KEY", ""),
        help="API key (default: OPENROUTER_API_KEY)",
    )
    parser.add_argument(
        "--cache-dir",
        default="",
        help="Graph cache directory (default: project cache/skillgraph under --workspace)",
    )
    parser.add_argument(
        "--report-path", default="", help="Optional explicit path for compile report JSON"
    )
    parser.add_argument("--task-context-path", default="", help="Optional task context path")
    parser.add_argument("--verifier-command", default="", help="Optional verifier command")
    parser.add_argument("--trace-path", default="", help="Optional successful trace path")
    parser.add_argument("--target-output", action="append", default=[], help="Expected output path")
    parser.add_argument("--compile-goal", default="", help="guidance, adapter, workflow, or solver")
    parser.add_argument(
        "--artifact-policy",
        default="",
        help="Evidence policy for artifact promotion: generic, trace, or family",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only run analysis+grounding and cache the graph",
    )
    parser.add_argument(
        "--validate-only", action="store_true", help="Only run analysis+validation and emit report"
    )
    parser.add_argument(
        "--strict-validate",
        action="store_true",
        help=(
            "Compatibility flag. Workflow compilation always fails on blocking "
            "validation issues; validate-only exits nonzero when blockers exist."
        ),
    )
    return parser


def _workspace_root(raw_workspace: str) -> Path:
    if raw_workspace:
        return Path(raw_workspace).expanduser()
    return Path.cwd()


def _cache_dir(raw_cache_dir: str, workspace: Path) -> Path:
    if raw_cache_dir:
        return Path(raw_cache_dir).expanduser()
    return get_storage_gateway(workspace).project_cache_root("skillgraph")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.analyze_only and args.validate_only:
        parser.error("--analyze-only and --validate-only cannot be used together")

    if not args.output and not args.analyze_only and not args.validate_only:
        parser.error("-o/--output is required unless using --analyze-only or --validate-only")

    skill_path = Path(args.skill)
    workspace = _workspace_root(args.workspace)
    cache_dir = _cache_dir(args.cache_dir, workspace)

    if args.analyze_only or args.validate_only:
        package = build_skill_package(skill_path)
        cache_path, use_cache, _ = _prepare_cache(package, cache_dir)
        entry_skill = Path(package.skill_root) / package.entry_skill

        analyzer = Analyzer(model=args.model, api_key=args.api_key, base_url=args.base_url)
        graph = analyzer.analyze(entry_skill, cache_path=cache_path, use_cache=use_cache)
        graph = normalize_graph(graph)

        if args.analyze_only:
            print(f"analyzed: {package.slug} | steps={len(graph.steps)} edges={len(graph.edges)}")
            print(cache_path)
            return

        validation = validate_graph(graph)
        if validation.errors:
            for issue in validation.errors:
                print(f"ERROR [{issue.code}] {issue.message} ({issue.step_id or 'graph'})")
        if validation.warnings:
            for issue in validation.warnings:
                print(f"WARN  [{issue.code}] {issue.message} ({issue.step_id or 'graph'})")
        blocking = workflow_blocking_issues(validation)

        report_target = (
            Path(args.report_path)
            if args.report_path
            else cache_dir / f"{package.slug}.report.json"
        )
        report = build_report(graph, package, args.output or "(validate-only)", validation)
        report.save(report_target)
        print(report_target)

        if args.strict_validate and blocking:
            raise SystemExit(1)
        return

    output = compile_skill(
        skill_path=skill_path,
        output_path=Path(args.output),
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        runtime_model=args.runtime_model or None,
        cache_dir=cache_dir,
        strict_validate=args.strict_validate,
        report_path=Path(args.report_path) if args.report_path else None,
        task_context_path=Path(args.task_context_path) if args.task_context_path else None,
        verifier_command=args.verifier_command or None,
        trace_path=Path(args.trace_path) if args.trace_path else None,
        target_outputs=list(args.target_output or []),
        compile_goal=args.compile_goal or None,
        artifact_policy=args.artifact_policy or None,
    )
    print(output)


if __name__ == "__main__":
    main()
