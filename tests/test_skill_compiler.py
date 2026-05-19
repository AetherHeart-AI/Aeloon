from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest

from aeloon.core.config.paths import get_storage_gateway
from aeloon.plugins.SkillGraph.compiler import SkillCompilerRequest, compile_skill_to_workspace


@dataclass(frozen=True)
class _FakePackage:
    slug: str


@dataclass(frozen=True)
class _FakeApi:
    build_skill_package: object
    compile_skill: object


def _install_fake_skillgraph(monkeypatch: pytest.MonkeyPatch, expected_skill_path: Path) -> None:
    def _fake_build_skill_package(skill_path):
        assert skill_path == expected_skill_path.resolve()
        return _FakePackage(slug="demo")

    def _fake_compile_skill(**kwargs):
        output_path = kwargs["output_path"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            'SKILL_META = {"name": "demo", "description": "Demo workflow", "global_inputs": []}\n'
            "class _Graph:\n"
            "    def invoke(self, state):\n"
            "        return state\n"
            "def build_graph():\n"
            "    return _Graph()\n",
            encoding="utf-8",
        )
        output_path.with_suffix(".manifest.json").write_text(
            '{"dependencies": []}', encoding="utf-8"
        )
        sandbox = output_path.with_suffix(".sandbox")
        (sandbox / "skill").mkdir(parents=True, exist_ok=True)
        (sandbox / "bootstrap.json").write_text(
            '{"status": "ready", "checks": [], "env": {}}', encoding="utf-8"
        )
        (sandbox / "env.json").write_text("{}", encoding="utf-8")
        config_path = output_path.parent / "skill_config.json"
        config_path.write_text(
            '{"runtime": {"api_key": "", "base_url": "https://example.com", "model": "runtime-model"}}',
            encoding="utf-8",
        )
        kwargs["report_path"].write_text('{"ok": true}', encoding="utf-8")
        return output_path

    monkeypatch.setattr(
        "aeloon.plugins.SkillGraph.compiler._load_skillgraph_api",
        lambda: _FakeApi(_fake_build_skill_package, _fake_compile_skill),
    )


def test_compile_skill_to_workspace_writes_into_compiled_skills(tmp_path, monkeypatch) -> None:
    skills_dir = tmp_path / "skills" / "demo"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text("# Demo", encoding="utf-8")
    _install_fake_skillgraph(monkeypatch, skills_dir)

    provider = type("Provider", (), {"api_key": "key", "api_base": "https://example.com"})()
    result = compile_skill_to_workspace(
        workspace=tmp_path,
        provider=provider,
        default_model="default-model",
        request=SkillCompilerRequest(skill_path="skills/demo", runtime_model="runtime-model"),
    )

    assert result.workflow_name == "demo"
    assert result.output_path == (
        get_storage_gateway(tmp_path).project_compiled_skills_root(create=False)
        / "demo_workflow.py"
    )
    assert result.manifest_path.exists()
    assert result.sandbox_path.exists()
    assert result.report_path.exists()
    assert result.config_path.exists()


def test_compile_skill_to_workspace_resolves_gateway_skill_root(
    tmp_path,
    monkeypatch,
) -> None:
    storage = get_storage_gateway(tmp_path)
    skill_dir = storage.project_skills_root() / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# Demo", encoding="utf-8")
    _install_fake_skillgraph(monkeypatch, skill_dir)

    provider = type("Provider", (), {"api_key": "key", "api_base": "https://example.com"})()
    result = compile_skill_to_workspace(
        workspace=tmp_path,
        provider=provider,
        default_model="default-model",
        request=SkillCompilerRequest(skill_path="skills/demo", runtime_model="runtime-model"),
    )

    assert result.skill_path == skill_dir.resolve()
    assert result.output_path.parent == storage.project_compiled_skills_root(create=False)
    assert not (tmp_path / "skills").exists()


def test_compiled_skill_cache_can_be_deleted_and_rebuilt_without_source_loss(
    tmp_path,
    monkeypatch,
) -> None:
    storage = get_storage_gateway(tmp_path)
    skill_dir = storage.project_skills_root() / "demo"
    skill_file = skill_dir / "SKILL.md"
    skill_dir.mkdir(parents=True)
    skill_file.write_text("# Demo", encoding="utf-8")
    _install_fake_skillgraph(monkeypatch, skill_dir)
    provider = type("Provider", (), {"api_key": "key", "api_base": "https://example.com"})()
    request = SkillCompilerRequest(skill_path="skills/demo", runtime_model="runtime-model")

    first = compile_skill_to_workspace(
        workspace=tmp_path,
        provider=provider,
        default_model="default-model",
        request=request,
    )
    shutil.rmtree(storage.project_compiled_skills_root(create=False))
    shutil.rmtree(storage.project_cache_root("skillgraph", create=False))

    assert first.output_path.exists() is False
    assert skill_file.exists()

    second = compile_skill_to_workspace(
        workspace=tmp_path,
        provider=provider,
        default_model="default-model",
        request=request,
    )

    assert second.skill_path == skill_dir.resolve()
    assert second.output_path.exists()
    assert second.report_path.exists()
    assert skill_file.exists()


def test_compile_skill_to_workspace_raises_when_skillgraph_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "aeloon.plugins.SkillGraph.compiler._load_skillgraph_api",
        lambda: (_ for _ in ()).throw(RuntimeError("skillgraph is not available")),
    )

    provider = type("Provider", (), {"api_key": "key", "api_base": "https://example.com"})()
    with pytest.raises(RuntimeError, match="skillgraph is not available"):
        compile_skill_to_workspace(
            workspace=tmp_path,
            provider=provider,
            default_model="default-model",
            request=SkillCompilerRequest(skill_path="skills/demo"),
        )
