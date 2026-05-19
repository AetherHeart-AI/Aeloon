from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aeloon.core.agent.tools.registry import ToolRegistry
from aeloon.core.config.paths import get_storage_gateway
from aeloon.plugins._sdk.api import PluginAPI
from aeloon.plugins._sdk.registry import PluginRegistry
from aeloon.plugins._sdk.runtime import PluginRuntime
from aeloon.plugins._sdk.types import CommandContext
from aeloon.plugins.SkillGraph.compiler import SkillCompilerResult
from aeloon.plugins.SkillGraph.plugin import SkillGraphPlugin


class _FakeAPI:
    def __init__(self, workspace: Path) -> None:
        provider = type(
            "Provider",
            (),
            {"api_key": "test-key", "api_base": "https://example.com"},
        )()
        agent_loop = type(
            "AgentLoop",
            (),
            {
                "workspace": workspace,
                "provider": provider,
                "model": "test-model",
                "tools": ToolRegistry(),
            },
        )()
        self.runtime = type("Runtime", (), {"agent_loop": agent_loop})()
        self.commands: dict[str, object] = {}
        self.cli: dict[str, tuple[object, tuple[object, ...]]] = {}

    def register_command(self, name: str, handler, *, description: str = "") -> None:
        self.commands[name] = (handler, description)

    def register_cli(
        self,
        name: str,
        builder=None,
        *,
        commands=(),
        handler=None,
        description: str = "",
    ) -> None:
        self.cli[name] = (builder, commands)
        if handler is not None:
            self.register_command(name, handler, description=description)


def _make_context() -> CommandContext:
    async def _reply(_text: str) -> None:
        return None

    async def _progress(*_args, **_kwargs) -> None:
        return None

    return CommandContext(
        session_key="cli:chat",
        channel="cli",
        reply=_reply,
        send_progress=_progress,
        plugin_config={},
    )


def test_skill_compiler_register_creates_pending_cli(tmp_path: Path) -> None:
    agent_loop = MagicMock()
    agent_loop.workspace = tmp_path
    agent_loop.provider = MagicMock(api_key="test-key", api_base="https://example.com")
    agent_loop.model = "test-model"
    agent_loop.tools = ToolRegistry()

    registry = PluginRegistry()
    runtime = PluginRuntime(
        agent_loop=agent_loop,
        plugin_id="aeloon.skillgraph",
        config={},
        storage_base=tmp_path,
    )
    api = PluginAPI(
        plugin_id="aeloon.skillgraph",
        version="0.1.0",
        config={},
        runtime=runtime,
        registry=registry,
    )

    plugin = SkillGraphPlugin()
    plugin.register(api)

    assert any(record.name == "skill_compiler" for record in api._pending_commands)
    assert any(record.name == "skill_compiler" for record in api._pending_cli)


@pytest.mark.asyncio
async def test_skill_compiler_command_requires_path(tmp_path, monkeypatch) -> None:
    plugin = SkillGraphPlugin()
    api = _FakeAPI(tmp_path)
    plugin.register(api)
    await plugin.activate(api)

    response = await plugin._handle_command(_make_context(), "")

    assert response is not None
    assert "usage: /skill_compiler" in response.lower()


@pytest.mark.asyncio
async def test_skill_compiler_command_compiles_and_refreshes(tmp_path, monkeypatch) -> None:
    plugin = SkillGraphPlugin()
    api = _FakeAPI(tmp_path)
    plugin.register(api)
    await plugin.activate(api)

    storage = get_storage_gateway(tmp_path)
    compiled_dir = storage.project_compiled_skills_root(create=False)
    skillgraph_cache = storage.project_cache_root("skillgraph", create=False)
    fake_result = SkillCompilerResult(
        skill_path=storage.project_skills_root(create=False) / "demo",
        package_slug="demo",
        workflow_name="demo",
        output_path=compiled_dir / "demo_workflow.py",
        manifest_path=compiled_dir / "demo_workflow.manifest.json",
        sandbox_path=compiled_dir / "demo_workflow.sandbox",
        report_path=skillgraph_cache / "demo.report.json",
        config_path=compiled_dir / "skill_config.json",
        model="test-model",
        runtime_model="override-model",
        base_url="https://example.com",
    )
    refreshed: list[bool] = []

    monkeypatch.setattr(
        "aeloon.plugins.SkillGraph.plugin.compile_skill_to_workspace",
        lambda **_: fake_result,
    )
    monkeypatch.setattr(plugin, "_refresh_tools", lambda: refreshed.append(True) or True)

    response = await plugin._handle_command(
        _make_context(),
        "skills/demo --runtime-model override-model",
    )

    assert response is not None
    assert "skill compiled successfully" in response.lower()
    assert "run_demo" in response
    assert refreshed == [True]
