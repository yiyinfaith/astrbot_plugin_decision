"""Exercise the real plugin lifecycle/API with a minimal AstrBot host double."""

from __future__ import annotations

import copy
import importlib.util
import json
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def host(monkeypatch, tmp_path):
    def module(name, **attrs):
        value = ModuleType(name)
        value.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, value)
        return value

    def decorator(*args, **kwargs):
        return lambda target: target

    class Star:
        def __init__(self, context):
            self.context = context

    class Config(dict):
        config_path = str(tmp_path / "data" / "config" / "astrbot_plugin_decision_config.json")

    module("astrbot", logger=logging.getLogger("test.astrbot"))
    module("astrbot.api")
    module(
        "astrbot.api.event",
        AstrMessageEvent=object,
        filter=SimpleNamespace(
            on_llm_request=decorator,
            event_message_type=decorator,
            on_llm_response=decorator,
            command=decorator,
        ),
    )
    module("astrbot.api.provider", ProviderRequest=object)
    module("astrbot.api.star", Context=object, Star=Star, register=decorator)
    module("astrbot.core", AstrBotConfig=Config)
    module("astrbot.core.agent")
    module("astrbot.core.agent.handoff", HandoffTool=type("HandoffTool", (), {}))
    module("astrbot.core.agent.tool", FunctionTool=SimpleNamespace, ToolSet=SimpleNamespace)
    module("astrbot.core.platform", MessageType=SimpleNamespace(GROUP_MESSAGE=1, FRIEND_MESSAGE=2))
    module("astrbot.core.star")
    module("astrbot.core.star.filter")
    module(
        "astrbot.core.star.filter.event_message_type",
        EventMessageType=SimpleNamespace(GROUP_MESSAGE=1, PRIVATE_MESSAGE=2),
    )
    module("astrbot.core.utils")
    module(
        "astrbot.core.utils.astrbot_path",
        get_astrbot_config_path=lambda: str(tmp_path / "data" / "config"),
    )
    body = {}

    async def request_json(**kwargs):
        return copy.deepcopy(body)

    module(
        "astrbot.api.web",
        json_response=lambda data: data,
        error_response=lambda message, status_code: {"error": message, "status_code": status_code},
        request=SimpleNamespace(json=request_json),
    )
    package = module("_decision_config_test")
    package.__path__ = [str(ROOT)]
    spec = importlib.util.spec_from_file_location("_decision_config_test.main", ROOT / "main.py")
    integration = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, integration)
    spec.loader.exec_module(integration)
    schema = json.loads((ROOT / "_conf_schema.json").read_text(encoding="utf-8"))
    config = Config({key: copy.deepcopy(value["default"]) for key, value in schema.items()})
    manager = SimpleNamespace(func_list=[], iter_builtin_tools=lambda: [])
    context = SimpleNamespace(
        add_llm_tools=lambda tool: manager.func_list.append(tool),
        register_web_api=Mock(),
        get_llm_tool_manager=lambda: manager,
    )
    return SimpleNamespace(
        module=integration,
        config=config,
        schema=schema,
        context=context,
        body=body,
        path=integration.DecisionPlugin._get_settings_path(config),
    )


def test_native_schema_contains_only_globals_and_module_switches(host):
    assert set(host.schema) == {
        "provider",
        "base_url",
        "systemone_path",
        "api_key",
        "model",
        "timeout_sec",
        "retries",
        "retry_backoff_seconds",
        "model_context_tokens",
        "history_max_messages",
        "history_max_chars",
        "tools_subagents_decision_enabled",
        "proactive_reply_enabled",
    }
    assert not (host.schema.keys() & host.module.DETAIL_DEFAULTS.keys())


@pytest.mark.asyncio
async def test_save_reload_keeps_native_config_clean_and_webui_settings_intact(host):
    original = copy.deepcopy(host.config)
    plugin = host.module.DecisionPlugin(host.context, host.config)
    host.body.update(
        always_keep_tools=["jev_decide"],
        always_keep_recommend_tools=[],
        jev_pre_prompt="custom policy",
        main_llm_post_prompt="",
        tool_filter_enabled=False,
        proactive={"proactive_whitelist": ["12345"], "cooldown_seconds": 25.0},
    )
    result = await plugin.page_save_settings()
    assert result["saved"] is True
    assert host.config == original
    assert set(host.config) == set(host.schema)
    assert plugin._bool("tool_filter_enabled", True) is False
    assert plugin._main_llm_post_prompt() == ""
    assert host.path.parent == Path(host.config.config_path).parent
    assert not host.path.is_relative_to(ROOT)
    persisted = json.loads(host.path.read_text(encoding="utf-8"))
    assert not (persisted.keys() & host.schema.keys())
    reloaded = host.module.DecisionPlugin(host.context, host.config)
    settings = await reloaded.page_settings()
    assert settings["jev_pre_prompt"] == "custom policy"
    assert settings["main_llm_post_prompt"] == ""
    assert settings["proactive"]["proactive_whitelist"] == ["12345"]
    assert settings["proactive"]["cooldown_seconds"] == 25.0
    assert settings["always_keep_tools"] == ["jev_decide"]
    assert host.config == original


@pytest.mark.asyncio
async def test_failed_write_does_not_change_live_policy_history_or_previous_file(host, monkeypatch):
    plugin = host.module.DecisionPlugin(host.context, host.config)
    host.body.update(always_keep_tools=["jev_decide"], tool_filter_enabled=False)
    assert (await plugin.page_save_settings())["saved"] is True
    before_file = host.path.read_bytes()
    before_settings = copy.deepcopy(plugin._detail_settings)
    before_config = copy.deepcopy(host.config)
    reconfigure = Mock()
    monkeypatch.setattr(plugin.proactive, "reconfigure", reconfigure)

    def deny_replace(*args):
        raise PermissionError("simulated disk failure")

    monkeypatch.setattr(host.module.os, "replace", deny_replace)
    host.body.update(tool_filter_enabled=True, proactive={"proactive_history_max_messages": 1})
    result = await plugin.page_save_settings()
    assert result["status_code"] == 500
    assert plugin._detail_settings == before_settings
    assert host.config == before_config
    assert host.path.read_bytes() == before_file
    reconfigure.assert_not_called()
    assert list(host.path.parent.glob("*.tmp")) == []


@pytest.mark.asyncio
async def test_detail_file_cannot_override_globals_or_module_switches(host):
    host.path.parent.mkdir(parents=True)
    host.path.write_text(
        json.dumps(
            {"model": "wrong-model", "proactive_reply_enabled": True, "jev_pre_prompt": "my policy"}
        ),
        encoding="utf-8",
    )
    plugin = host.module.DecisionPlugin(host.context, host.config)
    assert plugin._setting("model") == "jev-latest"
    settings = await plugin.page_settings()
    assert settings["jev_pre_prompt"] == "my policy"
    assert settings["proactive_reply_enabled"] is False
    host.config["proactive_reply_enabled"] = True
    assert (await plugin.page_settings())["proactive_reply_enabled"] is True
    assert set(host.config) == set(host.schema)


def test_legacy_live_detail_keys_are_preserved_then_removed_from_native_config(host):
    host.config.update(jev_pre_prompt="legacy policy", proactive_whitelist=["group-1"])
    plugin = host.module.DecisionPlugin(host.context, host.config)
    assert set(host.config) == set(host.schema)
    assert plugin._setting("jev_pre_prompt") == "legacy policy"
    assert json.loads(host.path.read_text(encoding="utf-8"))["proactive_whitelist"] == ["group-1"]


def test_path_fallback_uses_astrbot_data_directory(host, monkeypatch, tmp_path):
    plugin_directory = tmp_path / "plugin_installation"
    plugin_directory.mkdir()
    monkeypatch.chdir(plugin_directory)
    assert host.module.DecisionPlugin._get_settings_path({}) == host.path
