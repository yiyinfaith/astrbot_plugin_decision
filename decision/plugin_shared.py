"""Module-level names shared by main.py and the DecisionPlugin mixins.

main.py re-exports the public constants so existing imports keep working.
"""

from __future__ import annotations

import asyncio
import copy
import json
import math
import os
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star
from astrbot.core import AstrBotConfig
from astrbot.core.agent.handoff import HandoffTool
from astrbot.core.agent.tool import FunctionTool, ToolSet
from astrbot.core.platform import MessageType
from astrbot.core.star.filter.event_message_type import EventMessageType
from astrbot.core.utils.astrbot_path import get_astrbot_config_path

from . import usage_meter
from .context import (
    build_decision_state,
    is_handoff_tool,
    is_mcp_tool,
    question_id,
    short_description,
    tool_is_active,
    tool_summary,
)
from .life_config import life_schema, normalize_life, validate_umo
from .models import DecisionProviderError
from .proactive import (
    DEFAULT_REPLY_STARTERS,
    GROUP_TARGET_ID,
    GROUP_TARGET_NAME,
    DialogueInference,
    ProactiveRecord,
    ProactiveState,
    ProactiveStatus,
    aggregate_scores,
    bounded_float,
    infer_dialogue_target,
    normalize_prefixes,
    parse_noul_scores,
    text_matches_prefix,
)
from .providers.astrbot import (
    AstrBotDecisionProvider,
    decision_provider_catalog,
)
from .routing import (
    DECISION_TOOL_NAME,
    DEFAULT_MAIN_LLM_POST_PROMPT,
    add_routing_hint,
    choose_tools,
    compact_answer,
    decision_tool_parameters,
)
from .skills import filter_skills_prompt
from .tokens import estimate_request_tokens, fit_request_state

PLUGIN_NAME = "astrbot_plugin_decision"
PLUGIN_DISPLAY_NAME = "jev决策综合插件"
_ON_DECORATING_RESULT = getattr(filter, "on_decorating_result", None)
DEFAULT_TOOL_NOUL_THRESHOLD = 0.65
# Event extra marking the request issued by the proactive (message interjection) path.
PROACTIVE_REPLY_EXTRA = "decision_proactive_reply"
# Life actions that mutate the plan, after which the WebUI wants a fresh read.
INSPECT_AFTER_ACTIONS = frozenset(
    {
        "generate",
        "rewrite_schedule",
        "rewrite_outfit",
        "generate_timeline",
        "regenerate_timeline",
        "import_timeline",
        "approve_timeline",
        "reject_timeline",
        "add_requirement",
    }
)
PROACTIVE_PAGE_DEFAULTS: dict[str, Any] = {
    "proactive_whitelist": [],
    "proactive_scope_blacklist": False,
    "direct_reply_prefixes": ["/", "@"],
    "force_reply_when_summoned": True,
    "proactive_alias": "AI|助手",
    "proactive_score_threshold": 0.68,
    "addressing_threshold": 0.7,
    "interject_threshold": 0.85,
    "cooldown_seconds": 60.0,
    "reply_window_seconds": 600.0,
    "max_replies_per_window": 2,
    "proactive_history_max_messages": 12,
    "proactive_max_sessions": 500,
    "no_reply_cooldown_seconds": 3.0,
    "observation_timeout_seconds": 600.0,
    "echo_detection_threshold": 3,
    "echo_detection_window_seconds": 30.0,
    "dense_conversation_threshold": 30,
    "dense_conversation_window_seconds": 600.0,
    "min_participant_count": 2,
    # WebUI child switch for the input dialogue-flow analysis.  The native
    # schema owns the module-level ``conversation_flow_enabled`` switch.
    "conversation_flow_analysis_enabled": True,
    "conversation_flow_window": 8,
    # The input/output conversation-flow enhancement has its own scope.  A
    # blacklist with no entries intentionally means every conversation.
    "conversation_flow_scope": [],
    "conversation_flow_scope_blacklist": True,
    # Off by default: pure media messages keep the old text-only behaviour
    # unless the operator explicitly opts in.
    "non_text_proactive_enabled": False,
}
OUTPUT_PAGE_DEFAULTS: dict[str, Any] = {
    "output_enhancement_enabled": False,
    "output_scope": [],
    "output_scope_blacklist": True,
    "output_pipeline": {},
}
DEFAULT_POLICY = (
    "你是一个只负责结构化判断的 Decision Model。\n"
    "请严格根据当前场景、用户请求和候选能力逐项判断。\n"
    "不要生成最终回答，不要执行工具，不要发明候选名称。"
)
# The JEV Provider (decision_provider_id) is deliberately not a detail key: it
# lives only in AstrBot's native plugin config.  The dashboard edits it there
# with a dropdown populated from AstrBot's live Decision Providers.
NATIVE_PROVIDER_KEY = "decision_provider_id"
# Errors _save_detail_settings can raise: disk errors, plus multimodal
# normalisation and JSON serialisation of a bad value.
SETTINGS_WRITE_ERRORS = (OSError, TypeError, ValueError)
DETAIL_DEFAULTS: dict[str, Any] = {
    "message_interjection_enabled": True,
    # Multi-modal understanding lives under 对话流增强 → 输入增强. It is a
    # first-class detail section so load, save and reload all see it.
    "multimodal": {},
    "life": {},
    "always_keep_tools": [],
    "always_keep_recommend_tools": [],
    "always_keep_subagents": [],
    "always_keep_recommend_subagents": [],
    "always_keep_mcp": [],
    "always_keep_recommend_mcp": [],
    "always_keep_skills": [],
    "always_keep_recommend_skills": [],
    "always_keep_tools_customized": False,
    "tool_filter_enabled": True,
    "subagent_filter_enabled": False,
    "mcp_filter_enabled": True,
    "skill_filter_enabled": True,
    # Per-category Jev decision switches.  Filtering and judging are
    # deliberately independent: a category can be judged only to produce
    # recommendations while remaining fully visible to the main LLM, or can
    # be filtered from the manually kept set without making a Jev request.
    "tool_decision_enabled": True,
    "subagent_decision_enabled": True,
    "mcp_decision_enabled": True,
    "skill_decision_enabled": True,
    # One shared scope gates the complete Tool/MCP/Skill/SubAgent decision module.
    # Blacklist mode with no entries preserves the historical default: apply
    # the module to every conversation.  Switching to whitelist mode with an
    # empty list intentionally disables it everywhere.
    "tools_scope": [],
    "tools_scope_blacklist": True,
    "tool_noul_threshold": DEFAULT_TOOL_NOUL_THRESHOLD,
    "jev_pre_prompt": DEFAULT_POLICY,
    "main_llm_post_prompt": DEFAULT_MAIN_LLM_POST_PROMPT,
    # Display preference for the plugin view: the optional liquid-glass theme.
    "ui_liquid_glass": False,
    **OUTPUT_PAGE_DEFAULTS,
    **PROACTIVE_PAGE_DEFAULTS,
}


def _safe_error(error: BaseException) -> str:
    text = str(error).replace("\n", " ").strip()
    return text[:240] or error.__class__.__name__
