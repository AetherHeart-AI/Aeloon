from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aeloon.core.config.paths import get_storage_gateway
from aeloon.plugins.SkillGraph.skillgraph import cli


@dataclass(frozen=True)
class _FakePackage:
    slug: str
    skill_root: str
    entry_skill: str = "SKILL.md"
    package_hash: str = "hash"

    def save(self, path: Path) -> None:
        path.write_text("{}", encoding="utf-8")


@dataclass(frozen=True)
class _FakeGraph:
    steps: list[str]
    edges: list[str]


def test_analyze_only_defaults_cache_to_gateway_project_cache(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    skill_dir = tmp_path / "demo"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("# Demo", encoding="utf-8")
    package = _FakePackage(slug="demo", skill_root=str(skill_dir))
    captured: dict[str, Any] = {}

    class _FakeAnalyzer:
        def __init__(self, **kwargs: Any) -> None:
            captured["analyzer_kwargs"] = kwargs

        def analyze(self, entry_skill: Path, *, cache_path: Path, use_cache: bool) -> _FakeGraph:
            captured["entry_skill"] = entry_skill
            captured["cache_path"] = cache_path
            captured["use_cache"] = use_cache
            cache_path.write_text("{}", encoding="utf-8")
            return _FakeGraph(steps=["step"], edges=[])

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "build_skill_package", lambda path: package)
    monkeypatch.setattr(cli, "Analyzer", _FakeAnalyzer)
    monkeypatch.setattr(cli, "normalize_graph", lambda graph: graph)
    monkeypatch.setattr(
        sys,
        "argv",
        ["skillgraph-compile", str(skill_dir), "--analyze-only", "--api-key", "key"],
    )

    cli.main()

    expected_cache_dir = get_storage_gateway(tmp_path).project_cache_root(
        "skillgraph",
        create=False,
    )
    expected_cache_path = expected_cache_dir / "demo.json"
    assert captured["entry_skill"] == skill_dir / "SKILL.md"
    assert captured["cache_path"] == expected_cache_path
    assert captured["use_cache"] is True
    assert expected_cache_path.exists()
    assert not (tmp_path / "output" / "graphs").exists()
    assert str(expected_cache_path) in capsys.readouterr().out


def test_cache_dir_override_remains_explicit_developer_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    explicit_cache = tmp_path / "custom-cache"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "skillgraph-compile",
            "demo",
            "--analyze-only",
            "--cache-dir",
            str(explicit_cache),
        ],
    )

    args = cli.build_parser().parse_args()

    assert cli._cache_dir(args.cache_dir, tmp_path) == explicit_cache
