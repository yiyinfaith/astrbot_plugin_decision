"""Exercise the real plugin lifecycle/API with a minimal AstrBot host double."""

from __future__ import annotations

import copy
import importlib.util
import json
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

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
    handoff = sys.modules["astrbot.core.agent.handoff"].HandoffTool()
    handoff.name = "agent_alpha"
    handoff.description = "alpha"
    manager = SimpleNamespace(func_list=[handoff], iter_builtin_tools=lambda: [])
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
        always_keep_subagents=["agent_alpha"],
        always_keep_recommend_subagents=[],
        jev_pre_prompt="custom policy",
        main_llm_post_prompt="",
        tool_decision_enabled=False,
        subagent_decision_enabled=True,
        tool_filter_enabled=False,
        proactive={"proactive_whitelist": ["12345"], "cooldown_seconds": 25.0},
    )
    result = await plugin.page_save_settings()
    assert result["saved"] is True
    assert host.config == original
    assert set(host.config) == set(host.schema)
    assert plugin._bool("tool_filter_enabled", True) is False
    assert plugin._bool("tool_decision_enabled", True) is False
    assert plugin._bool("subagent_decision_enabled", False) is True
    assert plugin._main_llm_post_prompt() == ""
    assert host.path.parent == Path(host.config.config_path).parent
    assert not host.path.is_relative_to(ROOT)
    persisted = json.loads(host.path.read_text(encoding="utf-8"))
    assert not (persisted.keys() & host.schema.keys())
    reloaded = host.module.DecisionPlugin(host.context, host.config)
    settings = await reloaded.page_settings()
    assert settings["jev_pre_prompt"] == "custom policy"
    assert settings["main_llm_post_prompt"] == ""
    assert settings["tool_decision_enabled"] is False
    assert settings["subagent_decision_enabled"] is True
    assert settings["proactive"]["proactive_whitelist"] == ["12345"]
    assert settings["proactive"]["cooldown_seconds"] == 25.0
    assert settings["always_keep_tools"] == ["jev_decide"]
    assert settings["always_keep_subagents"] == ["agent_alpha"]
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


def test_legacy_migration_keeps_native_values_when_disk_write_fails(host, monkeypatch):
    host.config.update(jev_pre_prompt="legacy policy", proactive_whitelist=["group-1"])

    def deny_replace(*args):
        raise PermissionError("simulated migration failure")

    monkeypatch.setattr(host.module.os, "replace", deny_replace)
    plugin = host.module.DecisionPlugin(host.context, host.config)

    assert host.config["jev_pre_prompt"] == "legacy policy"
    assert host.config["proactive_whitelist"] == ["group-1"]
    assert plugin._setting("jev_pre_prompt") == "legacy policy"


@pytest.mark.asyncio
async def test_mapping_tool_manager_is_supported_by_page_and_save(host):
    manager = host.context.get_llm_tool_manager()
    agent = manager.func_list[0]
    plugin = host.module.DecisionPlugin(host.context, host.config)
    manager.func_list = {agent.name: agent, "jev_decide": plugin._decision_tool}

    listed = (await plugin.page_tools())["tools"]
    assert [tool["name"] for tool in listed if tool["handoff"]] == [agent.name]
    decision = next(tool for tool in listed if tool["name"] == "jev_decide")
    assert decision["builtin"] is False
    assert decision["origin_display"] == "AstrBot 智能决策引擎"

    host.body.update(
        always_keep_tools=["jev_decide"],
        always_keep_subagents=[agent.name],
        always_keep_recommend_subagents=[agent.name],
    )
    result = await plugin.page_save_settings()
    assert result["saved"] is True
    assert result["always_keep_subagents"] == [agent.name]


def test_path_fallback_uses_astrbot_data_directory(host, monkeypatch, tmp_path):
    plugin_directory = tmp_path / "plugin_installation"
    plugin_directory.mkdir()
    monkeypatch.chdir(plugin_directory)
    assert host.module.DecisionPlugin._get_settings_path({}) == host.path


@pytest.mark.asyncio
async def test_dynamic_subagent_is_listed_saved_and_restored_without_global_registration(host):
    manager = host.context.get_llm_tool_manager()
    agent = manager.func_list.pop()
    host.context.subagent_orchestrator = SimpleNamespace(handoffs=[agent])
    plugin = host.module.DecisionPlugin(host.context, host.config)
    assert agent not in manager.func_list
    listed = (await plugin.page_tools())["tools"]
    agents = [tool for tool in listed if tool["handoff"]]
    assert [tool["name"] for tool in agents] == [agent.name]
    assert agents[0]["origin_display"] == "AstrBot SubAgent"
    host.body.update(
        always_keep_tools=["jev_decide", agent.name],
        always_keep_subagents=[agent.name, "jev_decide"],
        always_keep_recommend_subagents=[agent.name, "jev_decide"],
    )
    result = await plugin.page_save_settings()
    assert result["saved"] is True
    assert result["always_keep_tools"] == ["jev_decide"]
    assert result["always_keep_subagents"] == [agent.name]
    assert result["always_keep_recommend_subagents"] == [agent.name]
    reloaded = host.module.DecisionPlugin(host.context, host.config)
    settings = await reloaded.page_settings()
    assert settings["always_keep_subagents"] == [agent.name]
    assert settings["always_keep_recommend_subagents"] == [agent.name]
    assert not set(host.module.DETAIL_DEFAULTS).intersection(host.config)


@pytest.mark.asyncio
async def test_subagent_catalog_deduplicates_and_follows_orchestrator_reloads(host):
    agent = host.context.get_llm_tool_manager().func_list[0]
    orchestrator = SimpleNamespace(handoffs=[agent])
    host.context.subagent_orchestrator = orchestrator
    plugin = host.module.DecisionPlugin(host.context, host.config)
    assert len([t for t in (await plugin.page_tools())["tools"] if t["handoff"]]) == 1
    host.context.get_llm_tool_manager().func_list.remove(agent)
    orchestrator.handoffs = []
    assert not [t for t in (await plugin.page_tools())["tools"] if t["handoff"]]
    replacement = host.module.HandoffTool()
    replacement.name = "transfer_to_new_agent"
    replacement.description = "new"
    orchestrator.handoffs = [replacement]
    assert [t["name"] for t in (await plugin.page_tools())["tools"] if t["handoff"]] == [
        replacement.name
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("recommend", [False, True])
async def test_saved_dynamic_subagent_reaches_request_routing(host, monkeypatch, recommend):
    agent = host.context.get_llm_tool_manager().func_list.pop()
    host.context.subagent_orchestrator = SimpleNamespace(handoffs=[agent])
    plugin = host.module.DecisionPlugin(host.context, host.config)
    host.body.update(
        always_keep_tools=[],
        always_keep_subagents=[agent.name],
        always_keep_recommend_subagents=[agent.name] if recommend else [],
        subagent_filter_enabled=True,
        main_llm_post_prompt="Tools: {tools}; SubAgents: {subagents}",
    )
    assert (await plugin.page_save_settings())["saved"] is True
    toolset = SimpleNamespace(tools=[agent, plugin._decision_tool])
    toolset.get_tool = lambda name: next((t for t in toolset.tools if t.name == name), None)
    req = SimpleNamespace(func_tool=toolset, system_prompt="Existing persona")
    monkeypatch.setattr(plugin, "_decision_batches", Mock(return_value=[("test state", {})]))
    monkeypatch.setattr(plugin, "_evaluate", AsyncMock(return_value=SimpleNamespace(answers={})))
    await plugin.filter_tools_before_llm(None, req)
    assert req.func_tool.tools == [agent]
    assert req.system_prompt.startswith("Existing persona")
    assert (agent.name in req.system_prompt) is recommend


@pytest.mark.asyncio
async def test_both_filters_off_skip_jev_keep_all_and_use_manual_recommendations(host, monkeypatch):
    manager = host.context.get_llm_tool_manager()
    agent = manager.func_list[0]
    ordinary = SimpleNamespace(name="ordinary_tool", description="ordinary", active=True)
    manager.func_list.append(ordinary)
    host.context.subagent_orchestrator = SimpleNamespace(handoffs=[agent])
    plugin = host.module.DecisionPlugin(host.context, host.config)
    plugin._detail_settings.update(
        {
            "always_keep_tools": [ordinary.name],
            "always_keep_recommend_tools": [ordinary.name],
            "always_keep_subagents": [agent.name],
            "always_keep_recommend_subagents": [agent.name],
            "always_keep_tools_customized": True,
            "tool_filter_enabled": False,
            "subagent_filter_enabled": False,
            "tool_decision_enabled": False,
            "subagent_decision_enabled": False,
            "main_llm_post_prompt": "Tools={tools}; SubAgents={subagents}",
        }
    )
    evaluate = AsyncMock(side_effect=AssertionError("Jev must not be called"))
    monkeypatch.setattr(plugin, "_evaluate", evaluate)
    tools = [ordinary, agent, plugin._decision_tool]
    toolset = SimpleNamespace(tools=tools)
    toolset.get_tool = lambda name: next(
        (tool for tool in toolset.tools if tool.name == name), None
    )
    req = SimpleNamespace(func_tool=toolset, system_prompt="persona", prompt="", contexts=[])

    await plugin.filter_tools_before_llm(None, req)

    assert req.func_tool.tools == tools
    assert "Tools=ordinary_tool" in req.system_prompt
    assert f"SubAgents={agent.name}" in req.system_prompt
    evaluate.assert_not_awaited()


@pytest.mark.asyncio
async def test_only_enabled_category_is_sent_to_jev(host, monkeypatch):
    manager = host.context.get_llm_tool_manager()
    agent = manager.func_list[0]
    ordinary = SimpleNamespace(name="ordinary_tool", description="ordinary", active=True)
    manager.func_list.append(ordinary)
    host.context.subagent_orchestrator = SimpleNamespace(handoffs=[agent])
    plugin = host.module.DecisionPlugin(host.context, host.config)
    plugin._detail_settings.update(
        {
            "always_keep_tools": [ordinary.name],
            "always_keep_recommend_tools": [],
            "always_keep_subagents": [agent.name],
            "always_keep_recommend_subagents": [agent.name],
            "always_keep_tools_customized": True,
            "tool_filter_enabled": True,
            "subagent_filter_enabled": False,
            "tool_decision_enabled": True,
            "subagent_decision_enabled": False,
            "main_llm_post_prompt": "Tools={tools}; SubAgents={subagents}",
        }
    )
    seen_questions = []

    async def evaluate(*, state, questions):
        seen_questions.append(questions)
        return SimpleNamespace(
            answers={qid: {"noul": 1.0} for qid in questions},
        )

    monkeypatch.setattr(plugin, "_evaluate", evaluate)
    tools = [ordinary, agent, plugin._decision_tool]
    toolset = SimpleNamespace(tools=tools)
    toolset.get_tool = lambda name: next(
        (tool for tool in toolset.tools if tool.name == name), None
    )
    req = SimpleNamespace(func_tool=toolset, system_prompt="persona", prompt="", contexts=[])

    await plugin.filter_tools_before_llm(None, req)

    assert seen_questions
    assert all(qid.startswith("tool_") for batch in seen_questions for qid in batch)
    assert req.func_tool.tools == [ordinary, agent]
    assert "Tools=ordinary_tool" in req.system_prompt
    assert f"SubAgents={agent.name}" in req.system_prompt


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("category", "filter_enabled", "decision_enabled"),
    [
        ("tool", True, True),
        ("tool", False, True),
        ("tool", True, False),
        ("tool", False, False),
        ("subagent", True, True),
        ("subagent", False, True),
        ("subagent", True, False),
        ("subagent", False, False),
    ],
)
async def test_category_decision_and_filter_switch_matrix(
    host, monkeypatch, category, filter_enabled, decision_enabled
):
    """Filtering and judging are independent for each candidate category."""

    manager = host.context.get_llm_tool_manager()
    agent = manager.func_list[0]
    ordinary = SimpleNamespace(name="ordinary_tool", description="ordinary", active=True)
    manager.func_list.append(ordinary)
    host.context.subagent_orchestrator = SimpleNamespace(handoffs=[agent])
    plugin = host.module.DecisionPlugin(host.context, host.config)
    plugin._detail_settings.update(
        {
            "always_keep_tools": [],
            "always_keep_recommend_tools": [],
            "always_keep_subagents": [],
            "always_keep_recommend_subagents": [],
            "always_keep_tools_customized": True,
            "tool_filter_enabled": filter_enabled if category == "tool" else False,
            "subagent_filter_enabled": filter_enabled if category == "subagent" else False,
            "tool_decision_enabled": decision_enabled if category == "tool" else False,
            "subagent_decision_enabled": decision_enabled if category == "subagent" else False,
            "main_llm_post_prompt": "Tools={tools}; SubAgents={subagents}",
        }
    )

    async def evaluate(*, state, questions):
        return SimpleNamespace(answers={qid: {"noul": 1.0} for qid in questions})

    evaluate = AsyncMock(side_effect=evaluate)
    monkeypatch.setattr(plugin, "_evaluate", evaluate)
    tools = [ordinary, agent, plugin._decision_tool]
    toolset = SimpleNamespace(tools=tools)
    toolset.get_tool = lambda name: next(
        (tool for tool in toolset.tools if tool.name == name), None
    )
    req = SimpleNamespace(func_tool=toolset, system_prompt="persona", prompt="", contexts=[])

    await plugin.filter_tools_before_llm(None, req)

    candidate = ordinary if category == "tool" else agent
    candidate_name = candidate.name
    # A filtered category keeps only manual/Jev selections.  An unfiltered
    # category keeps every active candidate regardless of its decision switch.
    assert (candidate in req.func_tool.tools) is (not filter_enabled or decision_enabled)
    assert (candidate_name in req.system_prompt) is decision_enabled
    if decision_enabled:
        evaluate.assert_awaited_once()
    else:
        evaluate.assert_not_awaited()


@pytest.mark.asyncio
async def test_enabled_tools_and_subagents_share_one_jev_request(host, monkeypatch):
    """Both candidate sets are sent together; batching is only for context overflow."""

    manager = host.context.get_llm_tool_manager()
    agent = manager.func_list[0]
    ordinary = SimpleNamespace(name="ordinary_tool", description="ordinary", active=True)
    manager.func_list.append(ordinary)
    host.context.subagent_orchestrator = SimpleNamespace(handoffs=[agent])
    plugin = host.module.DecisionPlugin(host.context, host.config)
    plugin._detail_settings.update(
        {
            "always_keep_tools": [],
            "always_keep_recommend_tools": [],
            "always_keep_subagents": [],
            "always_keep_recommend_subagents": [],
            "always_keep_tools_customized": True,
            "tool_filter_enabled": False,
            "subagent_filter_enabled": False,
            "tool_decision_enabled": True,
            "subagent_decision_enabled": True,
            "main_llm_post_prompt": "Tools={tools}; SubAgents={subagents}",
        }
    )
    seen_questions = []

    async def evaluate(*, state, questions):
        seen_questions.append(questions)
        return SimpleNamespace(answers={qid: {"noul": 1.0} for qid in questions})

    monkeypatch.setattr(plugin, "_evaluate", evaluate)
    tools = [ordinary, agent, plugin._decision_tool]
    toolset = SimpleNamespace(tools=tools)
    toolset.get_tool = lambda name: next(
        (tool for tool in toolset.tools if tool.name == name), None
    )
    req = SimpleNamespace(func_tool=toolset, system_prompt="persona", prompt="", contexts=[])

    await plugin.filter_tools_before_llm(None, req)

    assert len(seen_questions) == 1
    assert any(qid.startswith("tool_") for qid in seen_questions[0])
    assert any(qid.startswith("subagent_") for qid in seen_questions[0])
    assert ordinary.name in req.system_prompt
    assert agent.name in req.system_prompt
