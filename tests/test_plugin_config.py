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
        "conversation_flow_enabled",
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
        proactive={
            "proactive_whitelist": ["12345"],
            "cooldown_seconds": 25.0,
            "conversation_flow_window": 35,
        },
        dialogue_enhancement={"conversation_flow_analysis_enabled": False},
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
    assert settings["dialogue_enhancement"]["conversation_flow_analysis_enabled"] is False
    assert settings["proactive"]["conversation_flow_window"] == 30
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
    assert decision["origin_display"] == "jev决策综合插件"

    host.body.update(
        always_keep_tools=["jev_decide"],
        always_keep_subagents=[agent.name],
        always_keep_recommend_subagents=[agent.name],
    )
    result = await plugin.page_save_settings()
    assert result["saved"] is True
    assert result["always_keep_subagents"] == [agent.name]


@pytest.mark.asyncio
async def test_dialogue_enhancement_scopes_and_output_pipeline_are_page_settings(host):
    plugin = host.module.DecisionPlugin(host.context, host.config)
    host.body.update(
        always_keep_tools=[],
        always_keep_recommend_tools=[],
        always_keep_subagents=[],
        always_keep_recommend_subagents=[],
        dialogue_enhancement={
            "conversation_flow_analysis_enabled": True,
            "conversation_flow_window": 10,
            "conversation_flow_scope": ["group-1"],
            "conversation_flow_scope_blacklist": False,
            "output_enhancement_enabled": True,
            "output_scope": ["group-2"],
            "output_scope_blacklist": True,
            "output_pipeline": {
                "pipeline": {"steps": ["split(分段回复)"]},
                "split": {"max_length": 240},
            },
        }
    )
    result = await plugin.page_save_settings()
    assert result["saved"] is True
    settings = await plugin.page_settings()
    dialogue = settings["dialogue_enhancement"]
    assert dialogue["conversation_flow_scope"] == ["group-1"]
    assert dialogue["conversation_flow_scope_blacklist"] is False
    assert dialogue["output_scope"] == ["group-2"]
    assert dialogue["output_scope_blacklist"] is True
    assert dialogue["output_pipeline"]["split"]["max_length"] == 240
    # The optional runtime is fail-open in this host double and must not make
    # a valid WebUI save fail.
    assert plugin._bool("output_enhancement_enabled", False) is True


@pytest.mark.asyncio
async def test_tools_scope_is_page_only_and_persists_with_empty_list_semantics(host):
    plugin = host.module.DecisionPlugin(host.context, host.config)
    host.body.update(
        always_keep_tools=[],
        always_keep_recommend_tools=[],
        always_keep_subagents=[],
        always_keep_recommend_subagents=[],
        tools_scope=["group-1", "group-1", ""],
        tools_scope_blacklist=False,
    )
    result = await plugin.page_save_settings()
    assert result["saved"] is True
    assert plugin._detail_settings["tools_scope"] == ["group-1"]
    assert plugin._detail_settings["tools_scope_blacklist"] is False
    assert (await plugin.page_settings())["tools_scope"] == ["group-1"]
    assert (await plugin.page_settings())["tools_scope_blacklist"] is False
    # A whitelist only applies to an explicitly matching group.
    matching = SimpleNamespace(
        unified_msg_origin="qq:GroupMessage:group-1",
        get_message_type=lambda: 1,
        get_group_id=lambda: "group-1",
    )
    other = SimpleNamespace(
        unified_msg_origin="qq:GroupMessage:group-2",
        get_message_type=lambda: 1,
        get_group_id=lambda: "group-2",
    )
    assert plugin._tools_scope_allows(matching)
    assert not plugin._tools_scope_allows(other)

    # Empty whitelist disables the module everywhere; empty blacklist enables
    # it everywhere, matching the WebUI copy.
    plugin._detail_settings.update({"tools_scope": [], "tools_scope_blacklist": False})
    assert not plugin._tools_scope_allows(matching)
    plugin._detail_settings["tools_scope_blacklist"] = True
    assert plugin._tools_scope_allows(other)


@pytest.mark.asyncio
async def test_tools_scope_bypass_keeps_request_untouched_and_skips_jev(host, monkeypatch):
    manager = host.context.get_llm_tool_manager()
    ordinary = SimpleNamespace(name="ordinary_tool", description="ordinary", active=True)
    manager.func_list.append(ordinary)
    plugin = host.module.DecisionPlugin(host.context, host.config)
    plugin._detail_settings.update(
        {
            "tools_scope": ["blocked-group"],
            "tools_scope_blacklist": True,
            "always_keep_tools": [],
            "always_keep_tools_customized": True,
        }
    )
    evaluate = AsyncMock(side_effect=AssertionError("out-of-scope must not call Jev"))
    monkeypatch.setattr(plugin, "_evaluate", evaluate)
    toolset = SimpleNamespace(tools=[ordinary], get_tool=lambda name: ordinary if name == ordinary.name else None)
    req = SimpleNamespace(func_tool=toolset, system_prompt="persona")
    event = SimpleNamespace(
        unified_msg_origin="qq:GroupMessage:blocked-group",
        get_message_type=lambda: 1,
        get_group_id=lambda: "blocked-group",
        get_extra=lambda key, default=None: default,
    )

    await plugin.filter_tools_before_llm(event, req)

    assert req.func_tool.tools == [ordinary]
    assert req.system_prompt == "persona"
    evaluate.assert_not_awaited()


def test_output_pipeline_always_reports_and_normalizes_builtin_order(host):
    plugin = host.module.DecisionPlugin(host.context, host.config)
    plugin._detail_settings["output_pipeline"] = {"pipeline": {"lock_order": False}}
    assert plugin._output_page_settings()["output_pipeline"]["pipeline"]["lock_order"] is True


@pytest.mark.asyncio
async def test_builtin_tools_fall_back_to_reserved_module_prefix(host):
    builtin = SimpleNamespace(
        name="builtin_search",
        description="search",
        active=True,
        handler_module_path="astrbot.builtin_stars.web_searcher.main",
    )
    host.context.get_llm_tool_manager().func_list.append(builtin)
    plugin = host.module.DecisionPlugin(host.context, host.config)

    assert plugin._builtin_tool_names() == {builtin.name}
    listed = await plugin.page_tools()
    item = next(item for item in listed["tools"] if item["name"] == builtin.name)
    assert item["builtin"] is True
    assert item["origin_display"] == "Astrbot内置工具"


@pytest.mark.asyncio
async def test_conversation_flow_global_switch_disables_existing_output_runtime(host):
    plugin = host.module.DecisionPlugin(host.context, host.config)
    runtime = SimpleNamespace(prepare_message=AsyncMock(), run=AsyncMock())
    plugin._output_runtime = runtime
    host.config["conversation_flow_enabled"] = False
    event = SimpleNamespace(
        unified_msg_origin="qq:FriendMessage:user-1",
        get_group_id=lambda: "",
        get_sender_id=lambda: "user-1",
    )

    await plugin.output_enhancement_message(event)
    await plugin.output_enhancement(event)

    runtime.prepare_message.assert_not_awaited()
    runtime.run.assert_not_awaited()


@pytest.mark.asyncio
async def test_proactive_provider_request_marks_event_as_consumed(host, monkeypatch):
    plugin = host.module.DecisionPlugin(host.context, host.config)
    host.config["proactive_reply_enabled"] = True
    plugin._detail_settings["proactive_scope_blacklist"] = True
    plugin.provider = object()
    plugin._ready = True
    host.context.conversation_manager = SimpleNamespace(
        get_curr_conversation_id=AsyncMock(return_value="conversation-1"),
        get_conversation=AsyncMock(return_value=SimpleNamespace()),
    )
    monkeypatch.setattr(
        plugin,
        "_evaluate",
        AsyncMock(
            return_value=SimpleNamespace(
                answers={
                    "is_addressing_bot": {"noul": 1.0},
                    "should_interject": {"noul": 1.0},
                    "conversation_relevance": {"noul": 1.0},
                    "timing": {"noul": 1.0},
                    "continuity": {"noul": 1.0},
                }
            )
        ),
    )
    event = SimpleNamespace(
        unified_msg_origin="qq:GroupMessage:group-1",
        is_at_or_wake_command=False,
        call_llm=False,
        get_message_type=lambda: 1,
        get_message_str=lambda: "普通消息",
        get_sender_id=lambda: "user-1",
        get_sender_name=lambda: "用户",
        get_self_id=lambda: "bot-1",
        get_messages=lambda: [],
        get_extra=lambda key, default=None: default,
        get_message_outline=lambda: "普通消息",
        request_llm=lambda **kwargs: ("provider-request", kwargs),
        should_call_llm=lambda value: setattr(event, "call_llm", value),
    )

    yielded = [item async for item in plugin.proactive_reply(event)]

    assert len(yielded) == 1
    assert yielded[0][0] == "provider-request"
    assert event.is_at_or_wake_command is True
    assert event.call_llm is True


def test_path_fallback_uses_astrbot_data_directory(host, monkeypatch, tmp_path):
    plugin_directory = tmp_path / "plugin_installation"
    plugin_directory.mkdir()
    monkeypatch.chdir(plugin_directory)
    assert host.module.DecisionPlugin._get_settings_path({}) == host.path


def test_scope_modes_have_explicit_empty_list_semantics_and_namespaces(host):
    plugin = host.module.DecisionPlugin(host.context, host.config)

    group = SimpleNamespace(
        unified_msg_origin="qq:GroupMessage:group-1",
        get_message_type=lambda: 1,
        get_group_id=lambda: "group-1",
        get_sender_id=lambda: "user-1",
    )
    private = SimpleNamespace(
        unified_msg_origin="qq:FriendMessage:user-1",
        get_message_type=lambda: 2,
        get_group_id=lambda: "",
        get_sender_id=lambda: "user-1",
    )

    # Blacklist + empty means every conversation applies.
    plugin._detail_settings.update(
        {
            "proactive_whitelist": [],
            "proactive_scope_blacklist": True,
            "conversation_flow_scope": [],
            "conversation_flow_scope_blacklist": True,
        }
    )
    assert plugin._proactive_whitelist_allows(group)
    assert plugin._conversation_flow_enabled_for(private)

    # The native module switch gates both the input analysis and output
    # runtime, independently of the WebUI child switch.
    host.config["conversation_flow_enabled"] = False
    assert not plugin._conversation_flow_enabled_for(private)
    host.config["conversation_flow_enabled"] = True

    # Whitelist + empty means no conversation applies.
    plugin._detail_settings.update(
        {
            "proactive_scope_blacklist": False,
            "conversation_flow_scope_blacklist": False,
        }
    )
    assert not plugin._proactive_whitelist_allows(group)
    assert not plugin._conversation_flow_enabled_for(private)

    # A raw private user id must not match a group with the same sender id.
    plugin._detail_settings.update(
        {
            "proactive_whitelist": ["user-1"],
            "proactive_scope_blacklist": False,
            "conversation_flow_scope": ["user-1"],
            "conversation_flow_scope_blacklist": False,
        }
    )
    assert not plugin._proactive_whitelist_allows(group)
    assert not plugin._conversation_flow_enabled_for(group)
    assert plugin._proactive_whitelist_allows(private)
    assert plugin._conversation_flow_enabled_for(private)


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


@pytest.mark.asyncio
async def test_mcp_is_listed_separately_and_joins_the_single_jev_request(host, monkeypatch):
    manager = host.context.get_llm_tool_manager()
    mcp_tool = SimpleNamespace(
        name="filesystem_read",
        description="read files",
        mcp_server_name="filesystem",
        active=True,
    )
    ordinary = SimpleNamespace(name="ordinary_tool", description="ordinary", active=True)
    manager.func_list.extend([mcp_tool, ordinary])
    plugin = host.module.DecisionPlugin(host.context, host.config)
    host.body.update(
        always_keep_tools=[],
        always_keep_recommend_tools=[],
        always_keep_mcp=[],
        always_keep_recommend_mcp=[],
        always_keep_subagents=[],
        always_keep_recommend_subagents=[],
        always_keep_tools_customized=True,
        tool_filter_enabled=True,
        mcp_filter_enabled=True,
        subagent_filter_enabled=False,
        tool_decision_enabled=True,
        mcp_decision_enabled=True,
        subagent_decision_enabled=False,
        main_llm_post_prompt="Tools={tools}; MCP={mcps}; SubAgents={subagents}",
    )
    assert (await plugin.page_save_settings())["saved"] is True
    listed = (await plugin.page_tools())["tools"]
    mcp_rows = [item for item in listed if item["name"] == mcp_tool.name]
    assert len(mcp_rows) == 1
    assert mcp_rows[0]["mcp"] is True
    assert mcp_rows[0]["handoff"] is False

    seen_questions = []

    async def evaluate(*, state, questions):
        seen_questions.append(questions)
        return SimpleNamespace(
            answers={qid: {"noul": 1.0} for qid in questions},
        )

    monkeypatch.setattr(plugin, "_evaluate", evaluate)
    toolset = SimpleNamespace(tools=[ordinary, mcp_tool, plugin._decision_tool])
    toolset.get_tool = lambda name: next(
        (tool for tool in toolset.tools if tool.name == name), None
    )
    req = SimpleNamespace(
        func_tool=toolset, system_prompt="persona", prompt="read a file", contexts=[]
    )
    await plugin.filter_tools_before_llm(None, req)
    assert len(seen_questions) == 1
    assert any(qid.startswith("tool_") for qid in seen_questions[0])
    assert any(qid.startswith("mcp_") for qid in seen_questions[0])
    assert mcp_tool in req.func_tool.tools
    assert "MCP=filesystem_read" in req.system_prompt


@pytest.mark.asyncio
async def test_mcp_manual_keep_and_recommend_are_persisted_separately(host):
    mcp_tool = SimpleNamespace(
        name="filesystem_read",
        description="read files",
        mcp_server_name="filesystem",
        active=True,
    )
    host.context.get_llm_tool_manager().func_list.append(mcp_tool)
    plugin = host.module.DecisionPlugin(host.context, host.config)
    host.body.update(
        always_keep_tools=[],
        always_keep_recommend_tools=[],
        always_keep_mcp=[mcp_tool.name],
        always_keep_recommend_mcp=[mcp_tool.name],
        always_keep_subagents=[],
        always_keep_recommend_subagents=[],
    )
    result = await plugin.page_save_settings()
    assert result["always_keep_tools"] == []
    assert result["always_keep_mcp"] == [mcp_tool.name]
    assert result["always_keep_recommend_mcp"] == [mcp_tool.name]
    settings = await plugin.page_settings()
    assert settings["always_keep_mcp"] == [mcp_tool.name]
    assert settings["always_keep_recommend_mcp"] == [mcp_tool.name]


@pytest.mark.asyncio
async def test_jev_failure_keeps_manual_recommendations(host, monkeypatch):
    """A fail-open Jev request must not suppress manual recommendations."""

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
            "tool_filter_enabled": True,
            "subagent_filter_enabled": True,
            "tool_decision_enabled": True,
            "subagent_decision_enabled": True,
            "main_llm_post_prompt": "Tools={tools}; SubAgents={subagents}",
        }
    )

    async def fail(*, state, questions):
        raise RuntimeError("simulated Jev failure")

    monkeypatch.setattr(plugin, "_evaluate", fail)
    tools = [ordinary, agent, plugin._decision_tool]
    toolset = SimpleNamespace(tools=tools)
    toolset.get_tool = lambda name: next(
        (tool for tool in toolset.tools if tool.name == name), None
    )
    req = SimpleNamespace(func_tool=toolset, system_prompt="persona", prompt="", contexts=[])

    await plugin.filter_tools_before_llm(None, req)

    assert ordinary in req.func_tool.tools
    assert agent in req.func_tool.tools
    assert "Tools=ordinary_tool" in req.system_prompt
    assert f"SubAgents={agent.name}" in req.system_prompt


@pytest.mark.asyncio
async def test_jev_failure_respects_unfiltered_tools_category(host, monkeypatch):
    ordinary = SimpleNamespace(name="ordinary_tool", description="ordinary", active=True)
    manager = host.context.get_llm_tool_manager()
    manager.func_list.append(ordinary)
    plugin = host.module.DecisionPlugin(host.context, host.config)
    plugin._detail_settings.update(
        {
            "always_keep_tools": [],
            "always_keep_tools_customized": True,
            "tool_filter_enabled": False,
            "tool_decision_enabled": True,
            "subagent_decision_enabled": False,
        }
    )
    monkeypatch.setattr(plugin, "_evaluate", AsyncMock(side_effect=RuntimeError("jev down")))
    tools = [ordinary, plugin._decision_tool]
    toolset = SimpleNamespace(tools=tools)
    toolset.get_tool = lambda name: next(
        (tool for tool in toolset.tools if tool.name == name), None
    )
    req = SimpleNamespace(func_tool=toolset, system_prompt="persona", prompt="", contexts=[])

    await plugin.filter_tools_before_llm(None, req)

    assert req.func_tool.tools == tools


@pytest.mark.asyncio
async def test_matched_command_bypasses_jev_routing(host, monkeypatch):
    """A native or plugin command is handled directly by AstrBot's pipeline."""

    manager = host.context.get_llm_tool_manager()
    ordinary = SimpleNamespace(name="ordinary_tool", description="ordinary", active=True)
    manager.func_list.append(ordinary)
    plugin = host.module.DecisionPlugin(host.context, host.config)
    evaluate = AsyncMock(side_effect=AssertionError("matched commands must not call Jev"))
    monkeypatch.setattr(plugin, "_evaluate", evaluate)
    tools = [ordinary, plugin._decision_tool]
    toolset = SimpleNamespace(tools=tools)
    req = SimpleNamespace(func_tool=toolset, system_prompt="persona", prompt="/xxx", contexts=[])
    event = SimpleNamespace(
        get_extra=lambda key, default=None: (
            {"command_handler": {"argument": "value"}}
            if key == "handlers_parsed_params"
            else default
        )
    )

    await plugin.filter_tools_before_llm(event, req)

    assert req.func_tool.tools == tools
    assert req.system_prompt == "persona"
    evaluate.assert_not_awaited()
