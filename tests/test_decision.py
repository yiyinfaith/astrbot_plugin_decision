from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from decision.context import (
    build_decision_state,
    context_lines,
    is_mcp_tool,
    is_skill_tool,
    tool_summary,
)
from decision.context import question_id as make_question_id
from decision.proactive import (
    DialogueInference,
    ProactiveRecord,
    ProactiveState,
    ProactiveStatus,
    aggregate_scores,
    infer_dialogue_target,
    normalize_prefixes,
    parse_noul_scores,
    text_matches_prefix,
)
from decision.routing import (
    add_routing_hint,
    append_system_prompt,
    choose_tools,
    decision_tool_parameters,
    recommendations_from_noul,
    render_routing_prompt,
)
from decision.skills import filter_skills_prompt
from decision.tokens import estimate_request_tokens, estimate_tokens, fit_request_state
from tests.dashboard_source import dashboard_source


def test_routing_always_keep_handoff_and_decision_tool():
    ordinary = SimpleNamespace(name="ordinary", description="ordinary", active=True)
    keep = SimpleNamespace(name="keep", description="keep", active=True)
    handoff = type("HandoffTool", (), {"name": "transfer_to_search", "description": "search"})()
    decision = SimpleNamespace(name="decision_judge", description="decision")
    original = [ordinary, keep, handoff]
    outcome = choose_tools(
        original,
        {"q": {"noul": 0.1}},
        {"q": "ordinary"},
        threshold=0.2,
        always_keep={"keep", "decision_judge"},
        decision_tool=decision,
        always_keep_recommend={"keep"},
    )
    assert [tool.name for tool in outcome.selected] == [
        "keep",
        "transfer_to_search",
        "decision_judge",
    ]
    assert [tool.name for tool in outcome.handoffs] == ["transfer_to_search"]
    assert outcome.recommended_tools == ["keep"]


def test_routing_manual_subagent_keep_and_recommend_are_independent():
    agent_keep = type("HandoffTool", (), {"name": "agent_keep", "description": "keep"})()
    agent_recommend = type(
        "HandoffTool", (), {"name": "agent_recommend", "description": "recommend"}
    )()
    agent_decision = type("HandoffTool", (), {"name": "agent_decision", "description": "decision"})()
    answers = {
        "keep_q": {"noul": 0.1},
        "recommend_q": {"noul": 0.1},
        "decision_q": {"noul": 0.9},
    }
    outcome = choose_tools(
        [agent_keep, agent_recommend, agent_decision],
        answers,
        {},
        threshold=0.65,
        always_keep=set(),
        decision_tool=None,
        always_keep_handoffs={"agent_keep", "agent_recommend"},
        always_keep_handoffs_recommend={"agent_recommend"},
        question_to_handoff={
            "keep_q": "agent_keep",
            "recommend_q": "agent_recommend",
            "decision_q": "agent_decision",
        },
        filter_handoffs=True,
    )
    assert [tool.name for tool in outcome.selected] == [
        "agent_keep",
        "agent_recommend",
        "agent_decision",
    ]
    assert outcome.recommended_subagents == ["agent_recommend", "agent_decision"]


def test_routing_never_readds_inactive_always_kept_tool():
    inactive = SimpleNamespace(name="inactive", description="offline", active=False)
    outcome = choose_tools(
        [inactive],
        {},
        {},
        threshold=0.2,
        always_keep={"inactive"},
        decision_tool=None,
        filter_ordinary=False,
    )
    assert outcome.selected == []


def test_each_noul_can_recommend_zero_one_or_many_subagents():
    answers = {
        "a": {"noul": 0.1},
        "b": {"noul": 0.8},
        "c": {"noul": 0.9},
    }
    mapping = {"a": "agent_a", "b": "agent_b", "c": "agent_c"}
    assert recommendations_from_noul(answers, mapping, threshold=0.2) == ["agent_b", "agent_c"]
    assert (
        recommendations_from_noul({key: {"noul": 0.1} for key in mapping}, mapping, threshold=0.2)
        == []
    )


def test_filter_switches_keep_all_or_filter_each_candidate_independently():
    ordinary = SimpleNamespace(name="ordinary", description="ordinary", active=True)
    dropped = SimpleNamespace(name="dropped", description="dropped", active=True)
    handoff_a = type("HandoffTool", (), {"name": "agent_a", "description": "a"})()
    handoff_b = type("HandoffTool", (), {"name": "agent_b", "description": "b"})()
    decision = SimpleNamespace(name="decision_judge", description="decision")
    answers = {
        "ordinary_q": {"noul": 0.8},
        "dropped_q": {"noul": 0.1},
        "agent_a_q": {"noul": 0.8},
        "agent_b_q": {"noul": 0.1},
    }
    ordinary_questions = {"ordinary_q": "ordinary", "dropped_q": "dropped"}
    handoff_questions = {"agent_a_q": "agent_a", "agent_b_q": "agent_b"}

    keep_all = choose_tools(
        [ordinary, dropped, handoff_a, handoff_b],
        answers,
        ordinary_questions,
        threshold=0.2,
        always_keep={"decision_judge"},
        decision_tool=decision,
        question_to_handoff=handoff_questions,
        filter_ordinary=False,
        filter_handoffs=False,
    )
    assert [tool.name for tool in keep_all.selected] == [
        "ordinary",
        "dropped",
        "agent_a",
        "agent_b",
        "decision_judge",
    ]
    assert keep_all.recommended_tools == ["ordinary"]

    filter_tools_only = choose_tools(
        [ordinary, dropped, handoff_a, handoff_b],
        answers,
        ordinary_questions,
        threshold=0.2,
        always_keep={"decision_judge"},
        decision_tool=decision,
        question_to_handoff=handoff_questions,
        filter_ordinary=True,
        filter_handoffs=False,
    )
    assert [tool.name for tool in filter_tools_only.selected] == [
        "ordinary",
        "agent_a",
        "agent_b",
        "decision_judge",
    ]
    assert filter_tools_only.recommended_tools == ["ordinary"]

    filter_both = choose_tools(
        [ordinary, dropped, handoff_a, handoff_b],
        answers,
        ordinary_questions,
        threshold=0.2,
        always_keep={"decision_judge"},
        decision_tool=decision,
        question_to_handoff=handoff_questions,
        filter_ordinary=True,
        filter_handoffs=True,
    )
    assert [tool.name for tool in filter_both.selected] == [
        "ordinary",
        "agent_a",
        "decision_judge",
    ]
    assert filter_both.recommended_tools == ["ordinary"]


def test_decision_selection_recommends_kept_tool_but_manual_only_keep_stays_silent():
    keep = SimpleNamespace(name="keep", description="keep", active=True)
    decision = SimpleNamespace(name="decision_judge", description="decision")
    outcome = choose_tools(
        [keep],
        {"keep_q": {"noul": 0.9}},
        {"keep_q": "keep"},
        threshold=0.2,
        always_keep={"keep", "decision_judge"},
        decision_tool=decision,
        always_keep_recommend=set(),
    )
    assert [tool.name for tool in outcome.selected] == ["keep", "decision_judge"]
    assert outcome.recommended_tools == ["keep"]
    manual_only = choose_tools(
        [keep],
        {},
        {},
        threshold=0.2,
        always_keep={"keep", "decision_judge"},
        decision_tool=decision,
        always_keep_recommend=set(),
    )
    assert manual_only.recommended_tools == []

    opt_out = choose_tools(
        [keep],
        {},
        {},
        threshold=0.2,
        always_keep=set(),
        decision_tool=decision,
        filter_ordinary=False,
    )
    assert [tool.name for tool in opt_out.selected] == ["keep"]


def test_builtin_decision_tool_can_be_manually_recommended_from_page_settings():
    decision = SimpleNamespace(name="decision_judge", description="decision")
    outcome = choose_tools(
        [],
        {},
        {},
        threshold=0.2,
        always_keep={"decision_judge"},
        decision_tool=decision,
        always_keep_recommend={"decision_judge"},
    )
    assert [tool.name for tool in outcome.selected] == ["decision_judge"]
    assert outcome.recommended_tools == ["decision_judge"]

    existing = choose_tools(
        [decision],
        {},
        {},
        threshold=0.2,
        always_keep={"decision_judge"},
        decision_tool=decision,
        always_keep_recommend={"decision_judge"},
    )
    assert existing.recommended_tools == ["decision_judge"]


def test_recommendation_is_appended_to_system_prompt_and_preserves_other_plugins():
    req = SimpleNamespace(
        system_prompt="AstrBot persona\n\nOther plugin prompt", extra_user_content_parts=[]
    )
    add_routing_hint(
        req,
        recommended_tools=["search", "calendar"],
        recommended_subagents=["transfer_to_search", "transfer_to_code"],
    )
    hint = req.system_prompt
    assert "当前场景需要根据用户请求选择合适的能力" in hint
    assert "search, calendar" in hint
    assert "transfer_to_search, transfer_to_code" in hint
    assert "AstrBot persona" in hint
    assert "Other plugin prompt" in hint
    assert req.extra_user_content_parts == []


def test_custom_routing_template_replaces_only_supported_placeholders():
    rendered = render_routing_prompt(
        'tools={tools}; agents={subagents}; json={"keep": true}',
        recommended_tools=["search"],
        recommended_subagents=[],
    )
    assert rendered == 'tools=search; agents=(无); json={"keep": true}'


def test_routing_template_replaces_mcp_placeholder():
    rendered = render_routing_prompt(
        "tools={tools}; mcp={mcps}; agents={subagents}",
        recommended_tools=[],
        recommended_mcp=["filesystem_read"],
        recommended_subagents=[],
    )
    assert rendered == "tools=(无); mcp=filesystem_read; agents=(无)"


def test_mcp_tool_detection_uses_server_identity():
    mcp_tool = SimpleNamespace(name="filesystem_read", mcp_server_name="filesystem", active=True)
    ordinary = SimpleNamespace(name="filesystem_read", active=True)
    assert is_mcp_tool(mcp_tool)
    assert not is_mcp_tool(ordinary)


def test_mcp_filter_keep_and_recommend_are_independent():
    selected = SimpleNamespace(
        name="filesystem_read", description="read files", mcp_server_name="filesystem", active=True
    )
    dropped = SimpleNamespace(
        name="filesystem_write",
        description="write files",
        mcp_server_name="filesystem",
        active=True,
    )
    outcome = choose_tools(
        [selected, dropped],
        {"selected_q": {"noul": 0.9}, "dropped_q": {"noul": 0.1}},
        {},
        threshold=0.65,
        always_keep=set(),
        decision_tool=None,
        always_keep_mcp={"filesystem_write"},
        always_keep_mcp_recommend={"filesystem_write"},
        question_to_mcp={"selected_q": "filesystem_read", "dropped_q": "filesystem_write"},
        filter_mcp=True,
    )
    assert [tool.name for tool in outcome.selected] == ["filesystem_read", "filesystem_write"]
    assert outcome.recommended_mcp == ["filesystem_read", "filesystem_write"]


def test_mcp_filter_off_keeps_unselected_and_still_recommends_only_selected():
    mcp_tool = SimpleNamespace(
        name="filesystem_read", description="read files", mcp_server_name="filesystem", active=True
    )
    outcome = choose_tools(
        [mcp_tool],
        {"mcp_q": {"noul": 0.1}},
        {},
        threshold=0.65,
        always_keep=set(),
        decision_tool=None,
        question_to_mcp={"mcp_q": "filesystem_read"},
        filter_mcp=False,
    )
    assert [tool.name for tool in outcome.selected] == ["filesystem_read"]
    assert outcome.recommended_mcp == []


def test_skill_routing_supports_filter_and_manual_keep_recommend_independently():
    selected = SimpleNamespace(name="selected_skill", description="selected", active=True)
    manual = SimpleNamespace(name="manual_skill", description="manual", active=True)
    dropped = SimpleNamespace(name="dropped_skill", description="dropped", active=True)
    outcome = choose_tools(
        [],
        {
            "selected_q": {"noul": 0.9},
            "dropped_q": {"noul": 0.1},
        },
        {},
        threshold=0.65,
        always_keep=set(),
        decision_tool=None,
        skills=[selected, manual, dropped],
        always_keep_skills={"manual_skill"},
        always_keep_skills_recommend={"manual_skill"},
        question_to_skill={"selected_q": "selected_skill", "dropped_q": "dropped_skill"},
        filter_skills=True,
    )
    assert [item.name for item in outcome.selected_skills] == ["selected_skill", "manual_skill"]
    assert outcome.recommended_skills == ["selected_skill", "manual_skill"]


def test_skill_filter_off_keeps_all_but_manual_recommendations_need_the_checkbox():
    skills = [
        SimpleNamespace(name="a", description="a", active=True),
        SimpleNamespace(name="b", description="b", active=True),
    ]
    outcome = choose_tools(
        [],
        {"a_q": {"noul": 0.9}, "b_q": {"noul": 0.1}},
        {},
        threshold=0.65,
        always_keep=set(),
        decision_tool=None,
        skills=skills,
        always_keep_skills={"a"},
        always_keep_skills_recommend=set(),
        question_to_skill={"a_q": "a", "b_q": "b"},
        filter_skills=False,
    )
    assert [item.name for item in outcome.selected_skills] == ["a", "b"]
    assert outcome.recommended_skills == ["a"]


def test_skill_detection_is_narrow_and_skill_prompt_filter_preserves_other_sections():
    class SkillInfo:
        name = "x"
        skill_name = "x"
        source_type = "local_only"
        path = "/skills/x/SKILL.md"
        active = True

    skill = SkillInfo()
    # A real SkillInfo-like class is accepted; a plain function tool is not.
    assert is_skill_tool(skill)
    assert not is_skill_tool(SimpleNamespace(name="tool", description="tool"))
    prompt = (
        "persona\n## Skills\n\n### Available skills\n\n"
        "- **keep**: keep it\n  File: `/skills/keep/SKILL.md`\n"
        "- **drop**: drop it\n  File: `/skills/drop/SKILL.md`\n"
        "\n### Skill rules\n\n1. rule\n## Other\nrest"
    )
    filtered = filter_skills_prompt(prompt, ["keep"])
    assert "**keep**" in filtered
    assert "**drop**" not in filtered
    assert "## Other\nrest" in filtered


def test_append_system_prompt_is_idempotent_for_this_plugin_section():
    req = SimpleNamespace(system_prompt="persona")
    append_system_prompt(req, "recommendation")
    first = req.system_prompt
    append_system_prompt(req, "recommendation")
    assert req.system_prompt == first


def test_state_excludes_system_prompt_and_tool_schema():
    state = build_decision_state(
        policy="short policy",
        current_prompt="find cats",
        contexts=[{"role": "user", "content": "older"}],
        tools=[{"name": "search", "description": "search the web"}],
        subagents=[],
    )
    assert "short policy" in state
    assert "find cats" in state
    assert "search the web" in state
    assert "system_prompt" not in state
    assert "parameters" not in state


def test_decision_state_has_a_separate_skill_section():
    state = build_decision_state(
        policy="policy",
        current_prompt="make a spreadsheet",
        contexts=[],
        skills=[{"name": "spreadsheet_skill", "description": "spreadsheet help"}],
    )
    assert "[Available Skills]" in state
    assert "spreadsheet_skill: spreadsheet help" in state


def test_public_schema_contains_only_global_settings_and_master_switches():
    schema = json.loads(
        (Path(__file__).resolve().parents[1] / "_conf_schema.json").read_text(encoding="utf-8")
    )
    assert set(schema) == {
        "decision_provider_id",
        "fallback_decision_provider_ids",
        "retries",
        "retry_backoff_seconds",
        "model_context_tokens",
        "history_max_messages",
        "history_max_chars",
        "tools_subagents_decision_enabled",
        "proactive_reply_enabled",
        "conversation_flow_enabled",
    }
    assert all("[" not in item.get("description", "") for item in schema.values())
    assert schema["tools_subagents_decision_enabled"]["default"] is True
    assert (
        schema["tools_subagents_decision_enabled"]["description"] == "Tool/MCP/Skill/SubAgent 决策"
    )
    assert "不影响主动对话" in schema["tools_subagents_decision_enabled"]["hint"]
    assert schema["proactive_reply_enabled"]["default"] is False
    assert "详细配置在插件 WebUI" in schema["proactive_reply_enabled"]["hint"]
    assert schema["retry_backoff_seconds"]["default"] == 0.0
    assert "默认 1" in schema["retries"]["hint"]
    assert schema["model_context_tokens"]["default"] == 32000
    assert schema["decision_provider_id"]["options"][0] == ""
    assert "请求超时" not in str(schema)
    detailed_fields = {
        "decision_pre_prompt",
        "main_llm_post_prompt",
        "tool_filter_enabled",
        "mcp_filter_enabled",
        "subagent_filter_enabled",
        "tool_noul_threshold",
        "always_keep_tools",
        "always_keep_recommend_tools",
        "always_keep_mcp",
        "always_keep_recommend_mcp",
        "always_keep_subagents",
        "always_keep_recommend_subagents",
        "proactive_whitelist",
        "direct_reply_prefixes",
        "force_reply_when_summoned",
        "proactive_score_threshold",
        "cooldown_seconds",
    }
    assert not detailed_fields.intersection(schema)


def test_builtin_decision_tool_is_default_keep_only_in_settings_page():
    page = dashboard_source()
    assert "<title>判断决策综合插件设置</title>" in page
    assert "<h1>判断决策综合插件设置</h1>" in page
    assert "始终推荐给主 LLM" in page
    assert "tool.builtin" in page
    assert "builtin" in page
    assert "#ba96ff" in page
    assert "#6ce0bf" in page
    assert "origin_display" in page
    assert "Vue.createApp" not in page
    assert "createApp" in page
    # The optional glass library is bundled and loaded on demand, never from a CDN.
    assert "./vendor/liquid-glass.js" in page
    assert "glass.goose" not in page and "unpkg.com" not in page
    assert "setTabs" in page
    assert "background: linear-gradient" in page
    assert "只负责结构化判断的 Decision Model" in page
    assert "当前场景需要根据用户请求选择合适的能力" in page
    assert "payloadOf" in page
    assert "waitForBridge" in page
    assert "fallbackBridge" not in page
    assert "JSON.parse(JSON.stringify(body))" in page
    assert "TOOL_THRESHOLD_DEFAULT = .65" in page
    assert "数值越高越严格" in page
    assert "always_keep_subagents" in page
    assert "always_keep_recommend_subagents" in page
    assert "tool_decision_enabled" in page
    assert "subagent_decision_enabled" in page
    assert "tools_scope_blacklist" in page
    assert "tools_scope" in page
    assert "锁定内置顺序" not in page
    assert "始终按内置顺序执行" in page
    assert "routing-rules-table" in page
    assert "四类能力的决策判断会合并为一次请求" in page
    assert "mcp_decision_enabled" in page
    assert "mcp_filter_enabled" in page
    assert "always_keep_mcp" in page
    assert "always_keep_skills" in page
    assert "always_keep_recommend_skills" in page
    assert "skill_decision_enabled" in page
    assert "skill_filter_enabled" in page
    assert "Tool/MCP/Skill/SubAgent" in page
    assert "<h2>对话流分析</h2>" in page
    assert "conversation_flow_analysis_enabled" in page
    assert "conversation_flow_window" in page
    for field in (
        "outputList('summary','quotes_files')",
        "outputConfig.error.custom_msg",
        "outputConfig.block.block_reread",
        "outputConfig.at.at_str",
        "outputConfig.clean.punctuation",
        "outputList('clean','lead')",
        "outputList('replace','words')",
        "outputConfig.typo.tone_error_rate",
        "outputConfig.tts.character_id",
        "outputConfig.t2i.pillowmd_style_dir",
        "outputConfig.reply.threshold",
        "outputConfig.forward.threshold",
        "outputList('recall','keywords')",
        "outputConfig.split.delay_scope_str",
        "outputList('split','tail_punc')",
    ):
        assert field in page


def test_proactive_details_are_page_only_and_excluded_from_public_schema():
    schema = json.loads(
        (Path(__file__).resolve().parents[1] / "_conf_schema.json").read_text(encoding="utf-8")
    )
    page = dashboard_source()
    for field in (
        "proactive_whitelist",
        "direct_reply_prefixes",
        "force_reply_when_summoned",
        "conversation_flow_analysis_enabled",
        "conversation_flow_window",
        "proactive_score_threshold",
        "observation_timeout_seconds",
    ):
        assert field in page
        assert field not in schema
    descriptions = " ".join(item.get("description", "") for item in schema.values())
    for excluded in ("工具调用提示", "安抚机制", "AI人格设定", "接管astrbot原生上下文"):
        assert excluded not in descriptions


def test_context_limits_history_and_truncates_tool_results():
    lines = context_lines(
        [
            {"role": "user", "content": "old"},
            {"role": "tool", "content": "x" * 1000},
            {"role": "user", "content": "current"},
        ],
        max_messages=3,
        max_chars=700,
        current_prompt="current",
    )
    assert not any(line.endswith("current") for line in lines)
    assert len(next(line for line in lines if line.startswith("tool:"))) <= 506


def test_token_budget_helpers_keep_requests_strictly_under_limit():
    questions = {"q": {"type": "noul", "instructions": "判断"}}
    state = "旧消息\n" * 200 + "当前消息"
    assert estimate_tokens("中文") >= 2
    fitted = fit_request_state(state, questions, 64)
    assert estimate_request_tokens(fitted, questions) < 64
    assert fitted.endswith("当前消息")


def test_tool_summary_does_not_apply_a_per_tool_description_cap():
    tool = SimpleNamespace(name="large", description="x" * 1000)
    assert len(tool_summary(tool)["description"]) == 1000


def test_question_id_is_stable_and_collision_resistant():
    assert make_question_id("tool", "web-search", 0) == make_question_id("tool", "web-search", 0)
    assert make_question_id("tool", "web-search", 0) != make_question_id("tool", "web search", 0)


def test_decision_tool_schema_has_valid_array_items():
    schema = decision_tool_parameters()
    assert schema["properties"]["choice_options"]["items"]["type"] == "object"
    assert schema["properties"]["score_levels"]["items"] == {"type": "string"}


def test_choose_tools_fail_open_shape_preserves_handoffs_and_decision_tool():
    ordinary = SimpleNamespace(name="ordinary", description="ordinary", active=True)
    handoff = type("HandoffTool", (), {"name": "transfer_to_search", "description": "search"})()
    decision = SimpleNamespace(name="decision_judge", description="decision")
    outcome = choose_tools(
        [ordinary, handoff],
        {},
        {"ordinary": "missing-answer"},
        threshold=0.2,
        always_keep={"decision_judge"},
        decision_tool=decision,
    )
    assert [tool.name for tool in outcome.selected] == ["transfer_to_search", "decision_judge"]


def test_proactive_history_records_bot_and_user_roles():
    state = ProactiveState(max_messages=3)
    state.add("g", ProactiveRecord("Alice", "1", "hello"))
    state.add("g", ProactiveRecord("bot", "0", "reply", True))
    assert state.history_lines("g", 200) == ["[Alice]: hello", "[bot]: reply"]


def test_dialogue_flow_prefers_explicit_bot_and_other_targets():
    bot = ProactiveRecord("Alice", "u1", "@bot hi", at_targets=(("bot-id", "Bot"),), timestamp=100)
    result = infer_dialogue_target(bot, [], bot_id="bot-id", now=100)
    assert result == DialogueInference("bot", "你", 1.0, "explicit_at_bot")

    other = ProactiveRecord("Alice", "u1", "@Bob hi", at_targets=(("u2", "Bob"),), timestamp=100)
    result = infer_dialogue_target(other, [], bot_id="bot-id", now=100)
    assert result.target_id == "u2"
    assert result.target_name == "Bob"
    assert result.reason == "explicit_at_other"

    # AstrBot's AtAll is represented as a target named ``all`` by some
    # adapters; it addresses the group and must not become a user target.
    group_mention = ProactiveRecord(
        "Alice", "u1", "@all hi", at_targets=(("all", "全体成员"),), timestamp=100
    )
    result = infer_dialogue_target(group_mention, [], bot_id="bot-id", now=100)
    assert result == DialogueInference(reason="default_group")


def test_dialogue_flow_uses_reply_bot_aba_and_conservative_group_fallback():
    reply = ProactiveRecord("Alice", "u1", "thanks", reply_to_id="bot-id", timestamp=100)
    assert infer_dialogue_target(reply, [], bot_id="bot-id", now=100).target_id == "bot"

    recent_bot = ProactiveRecord(
        "bot", "bot-id", "answer", True, timestamp=90, talking_to="u1", talking_to_name="Alice"
    )
    short_ack = ProactiveRecord("Alice", "u1", "好的", timestamp=100)
    assert (
        infer_dialogue_target(short_ack, [recent_bot], bot_id="bot-id", now=100).reason
        == "bot_recently_replied"
    )

    previous = ProactiveRecord(
        "Bob", "u2", "问 Alice", timestamp=90, talking_to="u1", talking_to_name="Alice"
    )
    follow = ProactiveRecord("Alice", "u1", "我来了", timestamp=100)
    result = infer_dialogue_target(follow, [previous], bot_id="bot-id", now=100)
    assert result.target_id == "u2"
    assert result.reason == "aba_pattern"

    unrelated = ProactiveRecord("Alice", "u1", "随便说说", timestamp=100)
    result = infer_dialogue_target(
        unrelated,
        [ProactiveRecord("Bob", "u2", "群里说话", timestamp=70)],
        bot_id="bot-id",
        now=100,
    )
    assert result.target_id == "group"
    assert result.reason == "default_group"


def test_proactive_history_can_render_identity_aware_dialogue_flow():
    state = ProactiveState(max_messages=4)
    state.add(
        "g",
        ProactiveRecord("Alice", "u1", "hello", talking_to="u2", talking_to_name="Bob"),
    )
    assert state.history_lines("g", 200, include_dialogue=True) == [
        "[Alice (u1) → Bob (u2)]: hello"
    ]


def test_proactive_reconfigure_applies_live_history_and_session_limits():
    state = ProactiveState(max_messages=5, max_sessions=3)
    for index in range(5):
        state.add("g", ProactiveRecord("user", str(index), str(index)))
    state.reconfigure(max_messages=2, max_sessions=1)
    assert state.history_lines("g", 200) == ["[user]: 3", "[user]: 4"]
    state.add("other", ProactiveRecord("user", "x", "x"))
    state.add("third", ProactiveRecord("user", "y", "y"))
    assert len(state.sessions) <= 1


def test_proactive_cooldown_and_window():
    state = ProactiveState()
    assert state.allow_reply("g", cooldown_seconds=60, window_seconds=600, max_replies=2)
    assert not state.allow_reply("g", cooldown_seconds=60, window_seconds=600, max_replies=2)
    assert not state.allow_reply("disabled", cooldown_seconds=0, window_seconds=600, max_replies=0)


def test_proactive_cancel_reply_releases_unqueued_slot():
    state = ProactiveState()
    assert state.allow_reply("g", cooldown_seconds=60, window_seconds=600, max_replies=1)
    state.cancel_reply("g")
    assert not state.has_pending_reply("g")
    assert state.allow_reply("g", cooldown_seconds=60, window_seconds=600, max_replies=1)


def test_proactive_reply_in_flight_blocks_a_second_reservation():
    """The proactive request runs outside the session lock, so a concurrent
    message must not reserve a second slot while the first reply is pending."""

    state = ProactiveState()
    assert state.allow_reply("g", cooldown_seconds=0, window_seconds=600, max_replies=5)
    assert not state.allow_reply("g", cooldown_seconds=0, window_seconds=600, max_replies=5)
    first = state.pending_reply_token("g")
    state.mark_reply_finished("g")
    assert state.pending_reply_token("g") is None
    assert state.allow_reply("g", cooldown_seconds=0, window_seconds=600, max_replies=5)
    assert state.pending_reply_token("g") != first


def test_hung_proactive_request_stops_blocking_after_the_timeout():
    from decision import proactive as proactive_module

    state = ProactiveState()
    assert state.allow_reply("g", cooldown_seconds=0, window_seconds=600, max_replies=5)
    state.sessions["g"].pending_since -= proactive_module.PENDING_REPLY_TIMEOUT_SECONDS + 1
    assert not state.has_pending_reply("g")
    assert state.allow_reply("g", cooldown_seconds=0, window_seconds=600, max_replies=5)


def test_reply_tokens_stay_unique_when_a_session_is_recreated():
    state = ProactiveState()
    assert state.allow_reply("g", cooldown_seconds=0, window_seconds=600, max_replies=5)
    old = state.pending_reply_token("g")
    state.sessions.pop("g")
    assert state.allow_reply("g", cooldown_seconds=0, window_seconds=600, max_replies=5)
    assert state.pending_reply_token("g") != old


def test_cancel_reply_with_a_stale_token_keeps_the_newer_reservation():
    state = ProactiveState()
    assert state.allow_reply("g", cooldown_seconds=0, window_seconds=600, max_replies=5)
    old = state.pending_reply_token("g")
    state.mark_reply_finished("g")
    assert state.allow_reply("g", cooldown_seconds=0, window_seconds=600, max_replies=5)
    newer = state.pending_reply_token("g")
    state.cancel_reply("g", old)
    assert state.pending_reply_token("g") == newer


def test_proactive_analysis_cooldown_does_not_drop_new_history():
    state = ProactiveState(max_messages=4)
    state.add("g", ProactiveRecord("Alice", "1", "first"))
    state.mark_analysis("g", success=True, no_reply_cooldown=60)
    assert not state.can_analyze("g")
    state.add("g", ProactiveRecord("Bob", "2", "during cooldown"))
    assert state.history_lines("g", 200)[-1] == "[Bob]: during cooldown"


def test_direct_prefix_matching_reserves_at_for_real_message_components():
    assert normalize_prefixes(["/", "@", "", "/"]) == ["/", "@"]
    assert text_matches_prefix("/hello", ["/", "@"])
    assert not text_matches_prefix("hello @bot", ["/", "@"])
    assert not text_matches_prefix("@bot hello", ["@"])


def test_proactive_scores_are_strict_and_weighted():
    scores = parse_noul_scores(
        {
            "good": {"noul": 0.8},
            "bad": {"noul": "0.9"},
            "clamped": {"noul": 2},
            "bool": {"noul": True},
        },
        ["good", "bad", "clamped", "bool"],
    )
    assert scores == {"good": 0.8, "clamped": 1.0}
    assert aggregate_scores(scores, {"good": 3, "clamped": 1}) == pytest.approx(0.85)


def test_proactive_state_tracks_echo_status_and_per_session_locks():
    state = ProactiveState(max_messages=5)
    state.add("g", ProactiveRecord("Alice", "1", "same"))
    state.add("g", ProactiveRecord("Bob", "2", "same"))
    state.add("g", ProactiveRecord("Carol", "3", "same"))
    status = state.observe_message(
        "g",
        text="same",
        sender_id="4",
        echo_threshold=3,
        echo_window=60,
        dense_threshold=99,
    )
    assert status == ProactiveStatus.GETTING_FAMILIAR
    assert state.lock_for("g") is state.lock_for("g")
    assert state.lock_for("g") is not state.lock_for("other")


def test_proactive_state_session_limit_does_not_grow_without_bound():
    state = ProactiveState(max_sessions=2)
    state.add("one", ProactiveRecord("a", "1", "x"))
    state.add("two", ProactiveRecord("b", "2", "x"))
    state.add("three", ProactiveRecord("c", "3", "x"))
    assert len(state.sessions) <= 2


def test_normal_llm_response_does_not_create_proactive_state():
    state = ProactiveState()
    assert not state.has_pending_reply("ordinary")
    assert "ordinary" not in state.sessions
