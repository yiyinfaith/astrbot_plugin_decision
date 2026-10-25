from __future__ import annotations

import asyncio
import json
import math
import os
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from astrbot import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, register
from astrbot.core import AstrBotConfig
from astrbot.core.agent.handoff import HandoffTool
from astrbot.core.agent.tool import FunctionTool, ToolSet
from astrbot.core.platform import MessageType
from astrbot.core.star.filter.event_message_type import EventMessageType
from astrbot.core.utils.astrbot_path import get_astrbot_config_path

from .decision.context import (
    build_decision_state,
    is_handoff_tool,
    is_mcp_tool,
    question_id,
    short_description,
    tool_is_active,
    tool_summary,
)
from .decision.life_config import life_schema, normalize_life, validate_umo
from .decision.models import DecisionProviderError
from .decision.proactive import (
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
from .decision.providers.systemone import SystemOneProvider
from .decision.routing import (
    DECISION_TOOL_NAME,
    DEFAULT_MAIN_LLM_POST_PROMPT,
    add_routing_hint,
    choose_tools,
    compact_answer,
    decision_tool_parameters,
)
from .decision.skills import filter_skills_prompt
from .decision.tokens import estimate_request_tokens, fit_request_state

PLUGIN_NAME = "astrbot_plugin_decision"
PLUGIN_DISPLAY_NAME = "jev决策综合插件"
_ON_DECORATING_RESULT = getattr(filter, "on_decorating_result", None)
DEFAULT_TOOL_NOUL_THRESHOLD = 0.65
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
    **OUTPUT_PAGE_DEFAULTS,
    **PROACTIVE_PAGE_DEFAULTS,
}


class DecisionFunctionTool(FunctionTool):
    """Keep framework ownership, disable and reload tracking on this plugin."""


@register(
    PLUGIN_NAME,
    "yiyinfaith",
    "jev决策综合插件：用 Jev 判断 Tool、MCP、Skill、SubAgent 和主动对话，并保留 AstrBot 主 LLM 的最终控制权。",
    "0.4.1",
)
class DecisionPlugin(Star):
    """AstrBot integration layer for a provider-neutral decision engine."""

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.context = context
        self.config = config
        self._settings_path = self._get_settings_path(config)
        self._detail_settings = self._load_detail_settings()
        # AstrBot also renders live config keys. Never merge WebUI settings
        # into this framework-owned object, even only in memory.
        legacy = {key: config[key] for key in DETAIL_DEFAULTS if key in config}
        if legacy:
            try:
                self._save_detail_settings({**legacy, **self._detail_settings})
            except OSError as exc:
                # Do not remove framework-owned values until the durable
                # migration succeeds.  Keeping them in memory preserves the
                # user's policy for this run and allows a later retry.
                self._detail_settings = {**legacy, **self._detail_settings}
                logger.error(
                    "Decision Engine could not migrate legacy settings; native values were kept: %s",
                    _safe_error(exc),
                )
            else:
                for key in legacy:
                    config.pop(key, None)
        self.provider: SystemOneProvider | None = None
        self._ready = False
        self._active_reply_warning_emitted = False
        self._output_runtime: Any | None = None
        self._life_runtime: Any | None = None
        self._multimodal: Any | None = None
        self._multimodal_tools: list[Any] = []
        self._life_media: Any | None = None
        self._personal_network: Any | None = None
        self._life_lock = asyncio.Lock()
        self._life_asset_lock = asyncio.Lock()
        self._life_jobs: dict[str, dict[str, Any]] = {}
        # Track the background WebUI jobs so a hot reload can cancel them instead
        # of leaving orphaned tasks mutating a dead runtime.
        self._life_tasks: set[asyncio.Task] = set()
        self._life_job_limit = 20
        self.proactive = ProactiveState(
            self._int("proactive_history_max_messages", 12),
            self._int("proactive_max_sessions", 500),
        )
        self._last_call_latency_ms: float | None = None
        self._call_count = 0
        self._failure_count = 0

        self._decision_tool = DecisionFunctionTool(
            name=DECISION_TOOL_NAME,
            description=(
                "Use the configured Decision Engine for one fast structured noul, choice, "
                "or score judgment. It currently uses Jev/SystemOne and never executes a tool."
            ),
            parameters=decision_tool_parameters(),
            handler=self._run_decision_tool,
        )
        # Context.add_llm_tools is the current AstrBot API. This tool is also
        # explicitly re-added to a request after filtering so it stays available.
        self.context.add_llm_tools(self._decision_tool)
        self._life_tools = []
        for name, description, handler, properties, required in (
            (
                "take_photo",
                "生成并发送一张照片：可以拍自己、拍别人、拍周围的景物或物品，"
                "也可以生成手机截图、聊天截图等屏幕内容。通过 aigen 使用当前 AstrBot 人格的"
                "外貌参考图与今日穿搭；不是真实摄像头。",
                self._take_photo,
                {
                    "scene": {
                        "type": "string",
                        "description": "画面内容、动作、构图与氛围要求，可留空使用当前活动。",
                    },
                    "subject": {
                        "type": "string",
                        "description": (
                            "拍摄对象：self=自己，other=别人，scene=景色/环境，"
                            "object=物品，screenshot=手机截图或录屏画面。"
                        ),
                        "enum": ["self", "other", "scene", "object", "screenshot"],
                    },
                    "count": {
                        "type": "integer",
                        "description": "生成张数；必须在管理员配置的数量区间内，越界会被自动收敛。",
                    },
                },
                [],
            ),
            (
                "take_video",
                "生成并发送一段短视频：可以拍自己、拍别人、拍场景，也可以生成手机录屏。"
                "通过 aigen 保持外貌与今日穿搭；不访问真实摄像头。",
                self._take_video,
                {
                    "scene": {"type": "string", "description": "视频内容、动作和镜头要求，可留空。"},
                    "subject": {
                        "type": "string",
                        "description": (
                            "拍摄对象：self=自己，other=别人，scene=景色/环境，"
                            "object=物品，screenshot=手机录屏画面。"
                        ),
                        "enum": ["self", "other", "scene", "object", "screenshot"],
                    },
                    "count": {
                        "type": "integer",
                        "description": "生成段数；必须在管理员配置的数量区间内，越界会被自动收敛。",
                    },
                },
                [],
            ),
            (
                "send_voice",
                "生成并发送一段音频，用于偶尔开口说话、朗读或唱歌；也可以生成环境音等描述性音频。"
                "通过 aigen 的音频接口实现，可指定音色。",
                self._send_voice,
                {
                    "text": {
                        "type": "string",
                        "description": "要说出、朗读或唱出的内容；提示词模式下则是音频的描述。",
                    },
                    "style": {
                        "type": "string",
                        "description": "语气、情绪与表达风格，例如温柔、兴奋、轻声哼唱，可留空。",
                    },
                    "count": {
                        "type": "integer",
                        "description": "生成段数；必须在管理员配置的数量区间内，越界会被自动收敛。",
                    },
                },
                ["text"],
            ),
            (
                "get_virtual_daily_schedule",
                "查询当前 AstrBot 人格的今日日程、穿搭和当前活动。",
                self._get_virtual_daily_schedule,
                {},
                [],
            ),
            (
                "get_long_term_timeline",
                "查询当前 AstrBot 人格已批准的长期生活阶段。",
                self._get_long_term_timeline,
                {},
                [],
            ),
            (
                "add_schedule_requirement",
                "仅当当前管理员明确要求时，为未来日期添加一次性日程要求。",
                self._add_schedule_requirement,
                {
                    "date_range": {
                        "type": "string",
                        "description": "日期或区间，例如 2026-12-01..2026-12-03",
                    },
                    "requirement": {"type": "string", "description": "管理员要求，最多500字"},
                },
                ["date_range", "requirement"],
            ),
        ):
            parameters = {"type": "object", "properties": properties}
            if required:
                parameters["required"] = required
            tool = DecisionFunctionTool(
                name=name, description=description, parameters=parameters, handler=handler
            )
            self._life_tools.append(tool)
            self.context.add_llm_tools(tool)
        from .decision.life_control_tools import control_tools

        for name, description, handler, properties, required in control_tools(self):
            tool = DecisionFunctionTool(name=name, description=description,
                parameters={"type": "object", "properties": properties, "required": required},
                handler=handler)
            self._life_tools.append(tool)
            self.context.add_llm_tools(tool)
        # The merged Personal Network tools keep their upstream names so stored
        # prompts and habits keep working. They are registered unconditionally
        # and removed per request when the feature or a sub-switch is off.
        for tool in (
            DecisionFunctionTool(
                name="update_personal_network",
                description=(
                    "更新当前 AstrBot 人格的人际网络：新增或修正人物、人物之间的关系，"
                    "以及已经发生的共同经历。只写入对话中明确、持久的人物与关系事实。"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "characters": {
                            "type": "array",
                            "description": "要新增或更新的人物列表，最多 20 个。",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "ref": {"type": "string", "description": "本次调用内的临时引用名"},
                                    "id": {"type": "string", "description": "已存在人物的 UUID"},
                                    "name": {"type": "string", "description": "人物姓名"},
                                    "aliases": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "别名或昵称",
                                    },
                                    "bio": {"type": "string", "description": "人物简介"},
                                    "personality": {"type": "string", "description": "性格"},
                                    "preferences": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "偏好",
                                    },
                                    "facts": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "确定的事实",
                                    },
                                    "current_sender": {
                                        "type": "boolean",
                                        "description": "该人物是否为当前消息发送者",
                                    },
                                },
                                "required": ["name"],
                            },
                        },
                        "relationships": {
                            "type": "array",
                            "description": "要新增或更新的人物关系列表，最多 30 条。",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "source": {"type": "string", "description": "主体，可用 persona 表示当前人格"},
                                    "target": {"type": "string", "description": "关系目标"},
                                    "type": {"type": "string", "description": "关系类型"},
                                    "strength": {"type": "integer", "description": "亲密度 0-100"},
                                    "status": {
                                        "type": "string",
                                        "enum": ["active", "ended", "uncertain"],
                                    },
                                    "description": {"type": "string", "description": "关系说明"},
                                },
                                "required": ["source", "target", "type"],
                            },
                        },
                        "interactions": {
                            "type": "array",
                            "description": "要记录的共同经历，最多 30 条。",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "participants": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "参与者，至少两位",
                                    },
                                    "type": {"type": "string", "description": "经历类型"},
                                    "summary": {"type": "string", "description": "经历摘要"},
                                    "occurred_at": {"type": "string", "description": "ISO 8601 时间"},
                                    "importance": {"type": "integer", "description": "重要度 0-100"},
                                    "emotional_tone": {"type": "string", "description": "情绪基调"},
                                },
                                "required": ["participants", "type", "summary"],
                            },
                        },
                    },
                },
                handler=self.network_update_tool,
            ),
            DecisionFunctionTool(
                name="query_personal_network",
                description="查询当前 AstrBot 人格的人际网络：人物、关系与共同经历。",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "要查询的人物姓名、别名或关键词，留空返回总览。",
                        }
                    },
                },
                handler=self.network_query_tool,
            ),
        ):
            self._life_tools.append(tool)
            self.context.add_llm_tools(tool)
        for endpoint, handler, methods in (
            ("life", self.page_life, ["GET"]),
            ("life/action", self.page_life_action, ["POST"]),
            ("life/control", self.page_life_control, ["POST"]),
            ("life/appearance", self.page_life_appearance, ["POST"]),
            ("life/personas", self.page_life_personas, ["GET"]),
            ("life/sessions", self.page_life_sessions, ["GET"]),
            ("life/billing", self.page_life_billing, ["GET"]),
        ):
            self.context.register_web_api(
                f"/{PLUGIN_NAME}/{endpoint}", handler, methods, "Virtual Life WebUI"
            )

        self.context.register_web_api(
            f"/{PLUGIN_NAME}/multimodal/models",
            self.page_multimodal_models,
            ["GET"],
            "Multi-modal model catalog",
        )

        self.context.register_web_api(
            f"/{PLUGIN_NAME}/tools",
            self.page_tools,
            ["GET"],
            "List tools available to the Decision Engine settings page",
        )
        self.context.register_web_api(
            f"/{PLUGIN_NAME}/settings",
            self.page_settings,
            ["GET"],
            "Read Decision Engine page settings",
        )
        self.context.register_web_api(
            f"/{PLUGIN_NAME}/settings/save",
            self.page_save_settings,
            ["POST"],
            "Save Decision Engine page settings",
        )
        for route, handler, methods in (
            ("network/personas", self.network_personas, ["GET"]),
            ("network/network", self.network_network, ["GET"]),
            ("network/network/enabled", self.network_set_enabled, ["POST"]),
            ("network/character/save", self.network_save_character, ["POST"]),
            ("network/character/delete", self.network_delete_character, ["POST"]),
            ("network/character/merge", self.network_merge_characters, ["POST"]),
            ("network/relationship/save", self.network_save_relationship, ["POST"]),
            ("network/relationship/delete", self.network_delete_relationship, ["POST"]),
            ("network/life-event/save", self.network_save_life_event, ["POST"]),
            ("network/life-event/delete", self.network_delete_life_event, ["POST"]),
            ("network/generation/generate", self.network_generate, ["POST"]),
            ("network/generation/status", self.network_generation_status, ["POST"]),
            ("network/generation/validate", self.network_generation_validate, ["POST"]),
            ("network/generation/apply", self.network_generation_apply, ["POST"]),
            ("network/avatar/<character_id>", self.network_avatar, ["GET", "POST"]),
            ("network/export", self.network_export, ["GET"]),
            ("network/import/<persona_key>/preview", self.network_import_preview, ["POST"]),
            ("network/import/apply", self.network_import_apply, ["POST"]),
        ):
            self.context.register_web_api(
                f"/{PLUGIN_NAME}/{route}", handler, methods, f"Personal network {route}"
            )

    @staticmethod
    def _get_settings_path(config: AstrBotConfig) -> Path:
        config_path = getattr(config, "config_path", "")
        if config_path:
            return Path(str(config_path)).resolve().with_name(f"{PLUGIN_NAME}_settings.json")
        return Path(get_astrbot_config_path()) / f"{PLUGIN_NAME}_settings.json"

    def _load_detail_settings(self) -> dict[str, Any]:
        try:
            if not self._settings_path.is_file():
                return {}
            with self._settings_path.open(encoding="utf-8-sig") as file:
                value = json.load(file)
            payload = (
                {key: value[key] for key in DETAIL_DEFAULTS if key in value}
                if isinstance(value, Mapping)
                else {}
            )
            if "multimodal" in payload:
                from .decision.multimodal import normalize_settings

                payload["multimodal"] = normalize_settings(payload["multimodal"])
            return payload
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            logger.warning(
                "Decision Engine could not read WebUI settings; defaults will be used: %s",
                _safe_error(exc),
            )
            return {}

    def _save_detail_settings(self, values: Mapping[str, Any]) -> None:
        payload = {key: values[key] for key in DETAIL_DEFAULTS if key in values}
        if "multimodal" in payload:
            from .decision.multimodal import normalize_settings

            payload["multimodal"] = normalize_settings(payload["multimodal"])
        self._settings_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            dir=str(self._settings_path.parent),
            prefix=f".{self._settings_path.name}.",
            suffix=".tmp",
        )
        committed = False
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False, indent=2)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, self._settings_path)
            committed = True
        finally:
            if not committed:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
        self._detail_settings = payload

    def _setting(self, key: str, default: Any = None) -> Any:
        if key in DETAIL_DEFAULTS:
            return self._detail_settings.get(key, default)
        return self.config.get(key, default)

    async def initialize(self) -> None:
        provider_name = str(self._setting("provider", "systemone_jev"))
        if provider_name != "systemone_jev":
            logger.warning(
                "Decision Engine provider %s is unavailable; plugin stays fail-open.", provider_name
            )
            await self._configure_output_runtime()
            return
        self.provider = SystemOneProvider(
            base_url=str(self._setting("base_url", "")),
            path=str(self._setting("systemone_path", "/v1/systemone")),
            api_key=str(self._setting("api_key", "")),
            model=str(self._setting("model", "jev-latest")),
            timeout_sec=self._float("timeout_sec", 5.0),
            retries=self._int("retries", 1),
            retry_backoff_sec=self._float("retry_backoff_seconds", 0.0),
            retry_logger=self._log_provider_retry,
        )
        # Create the reusable session during plugin initialization. A missing key
        # is reported only as a configuration warning and does not break AstrBot.
        try:
            await self.provider.start()
        except Exception:
            logger.warning(
                "Decision Engine could not initialize its HTTP client; calls will fail open."
            )
        self._ready = True
        await self._configure_output_runtime()
        # Multi-modal understanding is independent of 主动对话: its tools and
        # auto-understanding have their own switches.
        try:
            await self._ensure_multimodal()
        except Exception as exc:
            logger.warning("Decision Engine 多模态初始化失败：%s", _safe_error(exc))
        await self._configure_life_runtime()
        if not self.provider.api_key:
            logger.warning(
                "Decision Engine API key is empty; tool filtering and proactive decisions are disabled."
            )

    async def _configure_output_runtime(self) -> None:
        """Enable the vendored output pipeline only when explicitly selected."""

        enabled = self._bool("conversation_flow_enabled", True) and self._bool(
            "output_enhancement_enabled", False
        )
        if not enabled:
            if self._output_runtime is not None:
                await self._output_runtime.close()
                self._output_runtime = None
            return
        if self._output_runtime is None:
            from .decision.outputpro_runtime import OutputPipelineRuntime

            self._output_runtime = OutputPipelineRuntime(
                self.context,
                self._settings_path.parent / PLUGIN_NAME,
            )
        config = self._setting("output_pipeline", {})
        if not isinstance(config, Mapping):
            config = {}
        await self._output_runtime.configure(dict(config))

    def _log_provider_retry(
        self, retry_number: int, max_retries: int, reason: str, backoff_seconds: float
    ) -> None:
        logger.warning(
            "Decision Engine Jev 请求失败，将进行第 %d/%d 次重试；间隔 %.2f 秒：%s",
            retry_number,
            max_retries,
            backoff_seconds,
            _safe_error(RuntimeError(reason)),
        )

    async def terminate(self) -> None:
        # Cancel WebUI jobs first: they hold the runtime and would otherwise keep
        # mutating a runtime that is about to be torn down.
        pending = [task for task in self._life_tasks if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._life_tasks.clear()
        self._life_jobs.clear()
        self._life_media = None
        if self._multimodal is not None:
            try:
                await self._multimodal.terminate()
            except Exception as exc:
                logger.warning(
                    "Decision Engine 多模态关闭失败：%s", _safe_error(exc)
                )
            self._multimodal = None
        if self._life_runtime is not None:
            try:
                await self._life_runtime.terminate()
            except Exception as exc:
                logger.warning("Decision Engine 虚拟生活关闭失败：%s", _safe_error(exc))
            self._life_runtime = None
        if self._personal_network is not None:
            try:
                await self._personal_network.terminate()
            except Exception as exc:
                logger.warning(
                    "Decision Engine 人际网络关闭失败：%s", _safe_error(exc)
                )
            self._personal_network = None
        if self.provider is not None:
            try:
                await self.provider.close()
            except Exception as exc:
                logger.warning("Decision Engine Provider 关闭失败：%s", _safe_error(exc))
        self.provider = None
        self._ready = False
        if self._output_runtime is not None:
            try:
                await self._output_runtime.close()
            except Exception as exc:
                logger.warning("Decision Engine 输出增强关闭失败：%s", _safe_error(exc))
            self._output_runtime = None

    def _int(self, key: str, default: int) -> int:
        try:
            return int(self._setting(key, default))
        except (TypeError, ValueError):
            return default

    def _float(self, key: str, default: float) -> float:
        try:
            return float(self._setting(key, default))
        except (TypeError, ValueError):
            return default

    def _bool(self, key: str, default: bool = False) -> bool:
        value = self._setting(key, default)
        return value if isinstance(value, bool) else bool(value)

    def _policy(self) -> str:
        return str(self._setting("jev_pre_prompt", "")).strip() or DEFAULT_POLICY

    def _main_llm_post_prompt(self) -> str:
        configured = self._setting("main_llm_post_prompt")
        if configured is None:
            return DEFAULT_MAIN_LLM_POST_PROMPT
        # An explicitly empty value is a supported way to disable this
        # optional advisory section without disabling Tool filtering.
        return str(configured)

    def _output_page_settings(self) -> dict[str, Any]:
        """Return a complete, editable OutputPro-compatible configuration."""

        from .decision.outputpro_runtime import _deep_merge, output_config_defaults

        configured = self._setting("output_pipeline", {})
        merged = _deep_merge(
            output_config_defaults(),
            configured if isinstance(configured, Mapping) else {},
        )
        pipeline = merged.get("pipeline")
        if not isinstance(pipeline, dict):
            pipeline = {}
            merged["pipeline"] = pipeline
        pipeline["lock_order"] = True
        return {
            "conversation_flow_analysis_enabled": self._bool(
                "conversation_flow_analysis_enabled", True
            ),
            "conversation_flow_window": self._int("conversation_flow_window", 8),
            "conversation_flow_scope": [
                str(item)
                for item in (self._setting("conversation_flow_scope", []) or [])
                if str(item).strip()
            ],
            "conversation_flow_scope_blacklist": self._bool(
                "conversation_flow_scope_blacklist", True
            ),
            "output_enhancement_enabled": self._bool("output_enhancement_enabled", False),
            "output_scope": [
                str(item) for item in (self._setting("output_scope", []) or []) if str(item).strip()
            ],
            "output_scope_blacklist": self._bool("output_scope_blacklist", True),
            "output_pipeline": merged,
            "multimodal": self._multimodal_page_settings(),
        }

    def _multimodal_page_settings(self) -> dict[str, Any]:
        """Editable multi-modal section plus the provider dropdown."""

        from .decision.multimodal import page_settings
        from .decision.multimodal.adapters import provider_catalog
        from .decision.multimodal.config import MultimodalSettings

        settings = MultimodalSettings(self._multimodal_settings())
        payload = page_settings(settings)
        payload["models"] = provider_catalog(self.context)
        return payload

    def _tool_list(self, req: ProviderRequest) -> list[Any]:
        return list(getattr(getattr(req, "func_tool", None), "tools", None) or [])

    def _ensure_request_toolset(self, req: ProviderRequest) -> ToolSet:
        """Keep the provider request usable even when AstrBot supplied no ToolSet.

        The Decision Engine tool is registered globally through ``Context``. A
        request can still arrive with ``func_tool=None`` when the active
        persona has no ordinary tools, so add a local ToolSet rather than
        silently making ``jev_decide`` unavailable.
        """

        tool_set = getattr(req, "func_tool", None)
        if tool_set is None:
            tool_set = ToolSet(tools=[])
            req.func_tool = tool_set
        if tool_set.get_tool(DECISION_TOOL_NAME) is None:
            tool_set.add_tool(self._decision_tool)
        return tool_set

    @staticmethod
    def _command_handler_matched(event: AstrMessageEvent | None) -> bool:
        """Return whether AstrBot already matched a registered command.

        AstrBot records parsed command handlers during its waking-check stage.
        Such events must stay on the command pipeline: running Jev routing or
        proactive analysis first could delay, alter, or duplicate a native or
        plugin command. An empty mapping means that no command matched.
        """

        if event is None:
            return False
        try:
            matched = event.get_extra("handlers_parsed_params", {})
        except (AttributeError, TypeError, ValueError):
            return False
        return isinstance(matched, Mapping) and bool(matched)

    def _builtin_tool_objects(self) -> list[Any]:
        """Return AstrBot's native tools without making them request-scoped."""

        manager = self.context.get_llm_tool_manager()
        iterator = getattr(manager, "iter_builtin_tools", None)
        if callable(iterator):
            try:
                values = iterator()
                enumerated = list(values.values()) if isinstance(values, Mapping) else list(values)
                if enumerated:
                    return enumerated
            except (KeyError, RuntimeError, TypeError, ValueError):
                logger.debug("Decision Engine could not enumerate AstrBot builtin tools.")

        # AstrBot 4.28.x keeps builtin and plugin tools in the same
        # ``func_list`` and identifies their owner through
        # ``handler_module_path``.  There is no public iterator in that
        # version, so use the stable reserved module prefix as a fallback.
        return [tool for tool in self._manager_tool_objects() if self._is_builtin_tool(tool)]

    @staticmethod
    def _is_builtin_tool(tool: Any) -> bool:
        """Return whether a tool belongs to AstrBot's reserved builtin stars."""

        module_path = str(getattr(tool, "handler_module_path", "") or "").strip()
        return module_path == "astrbot.builtin_stars" or module_path.startswith(
            "astrbot.builtin_stars."
        )

    def _manager_tool_objects(self) -> list[Any]:
        """Normalize AstrBot tool-manager collections across supported versions."""

        try:
            manager = self.context.get_llm_tool_manager()
            values = getattr(manager, "func_list", []) or []
            return list(values.values()) if isinstance(values, Mapping) else list(values)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            logger.debug("Decision Engine could not enumerate registered tools.")
            return []

    def _builtin_tool_names(self) -> set[str]:
        return {
            str(getattr(tool, "name", "")).strip()
            for tool in self._builtin_tool_objects()
            if str(getattr(tool, "name", "")).strip()
        }

    def _subagent_tool_objects(self) -> list[Any]:
        """Return dynamic SubAgent handoffs managed outside ``func_list``.

        AstrBot's dynamic ``SubAgentOrchestrator`` keeps these handoffs in its
        own ``handoffs`` collection and injects them into each request later;
        they are intentionally not registered in the global function-tool
        manager.  The WebUI and save validation must read that collection too.
        """

        candidates: list[Any] = []
        candidates.extend(self._manager_tool_objects())
        orchestrator = getattr(self.context, "subagent_orchestrator", None)
        handoffs = getattr(orchestrator, "handoffs", []) or []
        candidates.extend(handoffs.values() if isinstance(handoffs, Mapping) else handoffs)
        result: list[Any] = []
        seen: set[str] = set()
        for tool in candidates:
            if not is_handoff_tool(tool):
                continue
            name = str(getattr(tool, "name", "")).strip()
            if name and name not in seen:
                seen.add(name)
                result.append(tool)
        return result

    def _skill_objects(self) -> list[Any]:
        """Return AstrBot's installed SkillInfo entries when available.

        Skills are managed independently from the function-tool registry.  The
        optional import keeps the plugin compatible with AstrBot versions that
        predate the SkillManager; in that case the category simply stays empty.
        """

        try:
            from astrbot.core.skills.skill_manager import SkillManager

            context_config = self.context.get_config()
            if not isinstance(context_config, Mapping):
                context_config = {}
            runtime = str(context_config.get("computer_use_runtime", "local") or "local")
            values = SkillManager().list_skills(active_only=False, runtime=runtime)
            return list(values or [])
        except (
            ImportError,
            AttributeError,
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            logger.debug("Decision Engine could not enumerate AstrBot Skills.")
            return []

    def _tool_origin(self, tool: Any, *, builtin: bool = False) -> str:
        """Resolve the human-readable owner shown in the settings page."""

        if getattr(tool, "name", None) in {
            DECISION_TOOL_NAME,
            "take_photo",
            "take_video",
            "send_voice",
            "get_virtual_daily_schedule",
            "get_long_term_timeline",
            "add_schedule_requirement",
            "adjust_contact_frequency",
            "future_contact",
            "save_life_thought",
            "update_personal_network",
            "query_personal_network",
        }:
            return PLUGIN_DISPLAY_NAME
        if builtin or self._is_builtin_tool(tool):
            return "Astrbot内置工具"
        module_path = str(getattr(tool, "handler_module_path", "") or "").strip()
        if module_path:
            getter = getattr(self.context, "get_all_stars", None)
            if callable(getter):
                try:
                    stars = getter()
                except (AttributeError, KeyError, TypeError, ValueError):
                    stars = []
                for star in stars or []:
                    star_module = str(getattr(star, "module_path", "") or "")
                    if module_path == star_module or module_path.startswith(f"{star_module}."):
                        return str(
                            getattr(star, "display_name", None)
                            or getattr(star, "name", None)
                            or module_path
                        )
            if is_mcp_tool(tool):
                mcp_server = str(getattr(tool, "mcp_server_name", "") or "").strip()
                return f"MCP · {mcp_server}" if mcp_server else "MCP"
            return module_path
        mcp_server = str(getattr(tool, "mcp_server_name", "") or "").strip()
        if mcp_server:
            # A plugin-provided MCP tool may expose no handler module. Keep its
            # server identity as a standalone MCP source rather than guessing a
            # plugin owner.
            return f"MCP · {mcp_server}"
        if is_mcp_tool(tool):
            return "MCP"
        if is_handoff_tool(tool):
            return "AstrBot SubAgent"
        return "未知来源"

    def _skill_origin(self, skill: Any) -> str:
        """Resolve a Skill's plugin owner, or keep standalone Skills separate."""

        plugin_name = str(getattr(skill, "plugin_name", "") or "").strip()
        if not plugin_name:
            return "Skill"
        getter = getattr(self.context, "get_all_stars", None)
        if callable(getter):
            try:
                stars = getter()
            except (AttributeError, KeyError, TypeError, ValueError):
                stars = []
            for star in stars or []:
                root_name = str(
                    getattr(star, "root_dir_name", "") or getattr(star, "module_path", "") or ""
                ).strip()
                if root_name == plugin_name or str(getattr(star, "name", "")) == plugin_name:
                    return str(
                        getattr(star, "display_name", None)
                        or getattr(star, "name", None)
                        or plugin_name
                    )
        return plugin_name

    def _state_for_request(
        self,
        req: ProviderRequest,
        tools: list[Any],
        mcp_tools: list[Any],
        skills: list[Any],
        handoffs: list[Any],
    ) -> str:
        return build_decision_state(
            policy=self._policy(),
            current_prompt=str(req.prompt or ""),
            contexts=req.contexts,
            tools=[tool_summary(tool) for tool in tools],
            mcp_tools=[tool_summary(tool) for tool in mcp_tools],
            skills=[tool_summary(skill) for skill in skills],
            subagents=[tool_summary(tool) for tool in handoffs],
            history_max_messages=self._int("history_max_messages", 8),
            history_max_chars=self._int("history_max_chars", 6000),
        )

    def _context_budget(self) -> int:
        return max(1, self._int("model_context_tokens", 32000))

    def _fit_active_state(
        self,
        *,
        history: list[str],
        sender_name: str,
        sender_id: str,
        text: str,
        status: ProactiveStatus,
        summoned: bool,
        dialogue: DialogueInference,
        dialogue_enabled: bool,
        questions: Mapping[str, Mapping[str, Any]],
    ) -> str:
        """Trim the oldest proactive history until the full Jev request fits."""

        budget = self._context_budget()

        def make_state(lines: list[str]) -> str:
            if dialogue_enabled:
                current_message = (
                    f"[Current Message]\n{sender_name} ({sender_id}) → "
                    f"{dialogue.target_name} ({dialogue.target_id}): {text}\n"
                    f"[Dialogue Flow Analysis]\n"
                    f"Current addressee: {dialogue.target_name} ({dialogue.target_id})\n"
                    f"Inference: {dialogue.reason}; confidence={dialogue.confidence:.2f}\n"
                    "Treat sender IDs as stable identity anchors; nicknames may be duplicated or changed.\n"
                    "When the addressee is uncertain, assume the message is for the group rather than the Bot.\n"
                )
            else:
                current_message = f"[Current Message]\n{sender_name} ({sender_id}): {text}\n"
            return (
                f"[Decision Policy]\n{self._policy()}\n\n"
                "[Conversation]\n"
                f"{chr(10).join(lines) or '(none)'}\n\n"
                f"{current_message}"
                f"[Interaction State]\n{status.value}\n"
                f"Directly summoned: {str(summoned).lower()}"
            )

        while history:
            state = make_state(history)
            if estimate_request_tokens(state, questions) < budget:
                return state
            history.pop(0)

        state = make_state(history)
        if estimate_request_tokens(state, questions) < budget:
            return state
        # An unusually large current message or custom policy can exceed the
        # budget even without history. Preserve the tail, which contains the
        # current message and interaction state, as a final safety fallback.
        return fit_request_state(state, questions, budget)

    def _decision_batches(
        self,
        req: ProviderRequest,
        ordinary: list[Any],
        mcp_tools: list[Any],
        skills: list[Any],
        handoffs: list[Any],
        questions: Mapping[str, Mapping[str, Any]],
        question_to_tool: Mapping[str, str],
        question_to_mcp: Mapping[str, str],
        question_to_skill: Mapping[str, str],
        question_to_handoff: Mapping[str, str],
    ) -> list[tuple[str, dict[str, dict[str, Any]]]]:
        """Split candidate judgments only when the complete Jev request is oversized."""

        full_state = self._state_for_request(req, ordinary, mcp_tools, skills, handoffs)
        budget = self._context_budget()
        total_tokens = estimate_request_tokens(full_state, questions)
        if total_tokens < budget:
            return [(full_state, dict(questions))]

        entries: list[tuple[str, Any, str]] = []
        for tool in ordinary:
            name = str(getattr(tool, "name", ""))
            qid = next((key for key, value in question_to_tool.items() if value == name), "")
            if qid:
                entries.append(("tool", tool, qid))
        for tool in handoffs:
            name = str(getattr(tool, "name", ""))
            qid = next((key for key, value in question_to_handoff.items() if value == name), "")
            if qid:
                entries.append(("handoff", tool, qid))
        for tool in mcp_tools:
            name = str(getattr(tool, "name", ""))
            qid = next((key for key, value in question_to_mcp.items() if value == name), "")
            if qid:
                entries.append(("mcp", tool, qid))
        for skill in skills:
            name = str(getattr(skill, "name", ""))
            qid = next((key for key, value in question_to_skill.items() if value == name), "")
            if qid:
                entries.append(("skill", skill, qid))
        if not entries:
            return [(full_state, dict(questions))]

        batch_count = min(len(entries), max(2, math.ceil(total_tokens / budget)))
        buckets: list[list[tuple[str, Any, str]]] = [[] for _ in range(batch_count)]
        loads = [0] * batch_count
        weighted_entries = sorted(
            entries,
            key=lambda entry: estimate_request_tokens(
                f"{getattr(entry[1], 'name', '')}\n{getattr(entry[1], 'description', '')}",
                {entry[2]: questions[entry[2]]},
            ),
            reverse=True,
        )
        for entry in weighted_entries:
            weight = estimate_request_tokens(
                f"{getattr(entry[1], 'name', '')}\n{getattr(entry[1], 'description', '')}",
                {entry[2]: questions[entry[2]]},
            )
            target = min(range(batch_count), key=loads.__getitem__)
            buckets[target].append(entry)
            loads[target] += weight

        def render(bucket: list[tuple[str, Any, str]]) -> tuple[str, dict[str, dict[str, Any]]]:
            bucket_tools = [tool for kind, tool, _ in bucket if kind == "tool"]
            bucket_mcp = [tool for kind, tool, _ in bucket if kind == "mcp"]
            bucket_skills = [tool for kind, tool, _ in bucket if kind == "skill"]
            bucket_handoffs = [tool for kind, tool, _ in bucket if kind == "handoff"]
            bucket_questions = {qid: dict(questions[qid]) for _, _, qid in bucket}
            return (
                self._state_for_request(
                    req, bucket_tools, bucket_mcp, bucket_skills, bucket_handoffs
                ),
                bucket_questions,
            )

        rendered = [render(bucket) for bucket in buckets if bucket]
        # The initial count follows ceil(total/budget). Fixed prompt overhead is
        # duplicated in each request, so add batches if a particular bucket is
        # still too large. This keeps every request strictly below the limit.
        while True:
            oversized = next(
                (
                    index
                    for index, (state, batch_questions) in enumerate(rendered)
                    if estimate_request_tokens(state, batch_questions) >= budget
                ),
                None,
            )
            if oversized is None or len(rendered) >= len(entries):
                break
            bucket = buckets[oversized]
            if len(bucket) <= 1:
                break
            midpoint = max(1, len(bucket) // 2)
            buckets[oversized : oversized + 1] = [bucket[:midpoint], bucket[midpoint:]]
            rendered = [render(item) for item in buckets if item]
        return [
            (fit_request_state(state, batch_questions, budget), batch_questions)
            for state, batch_questions in rendered
        ]

    async def _evaluate(
        self,
        *,
        state: str,
        questions: Mapping[str, Mapping[str, Any]],
    ):
        if not self.provider or not self._ready:
            raise DecisionProviderError("Decision Engine is not initialized")
        self._call_count += 1
        started = time.perf_counter()
        try:
            result = await self.provider.evaluate(
                state=state,
                questions=questions,
                model=str(self._setting("model", "jev-latest")),
            )
            self._last_call_latency_ms = (time.perf_counter() - started) * 1000
            return result
        except Exception:
            self._failure_count += 1
            self._last_call_latency_ms = (time.perf_counter() - started) * 1000
            raise

    def _log_routing_outcome(self, outcome: Any) -> None:
        selected_tools = [
            str(getattr(tool, "name", ""))
            for tool in outcome.selected
            if getattr(tool, "name", None) and not is_handoff_tool(tool)
        ]
        selected_subagents = [
            str(getattr(tool, "name", ""))
            for tool in outcome.selected
            if getattr(tool, "name", None) and is_handoff_tool(tool)
        ]
        selected_mcp = [
            str(getattr(tool, "name", ""))
            for tool in outcome.selected
            if getattr(tool, "name", None) and is_mcp_tool(tool)
        ]
        # AstrBot's default console handler can hide INFO records.  Keep the
        # two normal decision summaries at WARNING so operators see them in
        # the console without enabling a separate debug switch.
        logger.warning(
            "Decision Engine 共筛选出如下工具：Tool=%s；MCP=%s；Skill=%s；SubAgent=%s；推荐 Tool=%s；推荐 MCP=%s；推荐 Skill=%s；推荐 SubAgent=%s",
            ", ".join(selected_tools) or "无",
            ", ".join(selected_mcp) or "无",
            ", ".join(
                str(getattr(skill, "name", ""))
                for skill in getattr(outcome, "selected_skills", [])
                if getattr(skill, "name", None)
            )
            or "无",
            ", ".join(selected_subagents) or "无",
            ", ".join(outcome.recommended_tools) or "无",
            ", ".join(getattr(outcome, "recommended_mcp", [])) or "无",
            ", ".join(getattr(outcome, "recommended_skills", [])) or "无",
            ", ".join(getattr(outcome, "recommended_subagents", [])) or "无",
        )

    @staticmethod
    def _apply_skill_visibility(req: ProviderRequest, selected_skills: list[Any]) -> None:
        """Filter only AstrBot's generated Skill inventory in the system prompt."""

        req.system_prompt = filter_skills_prompt(
            str(getattr(req, "system_prompt", "") or ""),
            [str(getattr(skill, "name", "")) for skill in selected_skills],
        )

    @filter.on_llm_request(priority=1000)
    async def filter_tools_before_llm(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
    ) -> None:
        """Run one fan-out decision request before AstrBot's main LLM call."""

        # A registered AstrBot/plugin command has already been parsed by the
        # waking-check pipeline and should execute directly without entering
        # this LLM routing hook. The command handler owns the event.
        if self._command_handler_matched(event):
            return
        if not self._bool("tools_subagents_decision_enabled", True):
            return
        if not self._tools_scope_allows(event):
            return
        tool_set = self._ensure_request_toolset(req)
        tool_filter_enabled = self._bool("tool_filter_enabled", True)
        mcp_filter_enabled = self._bool("mcp_filter_enabled", True)
        skill_filter_enabled = self._bool("skill_filter_enabled", True)
        subagent_filter_enabled = self._bool("subagent_filter_enabled", False)
        tool_decision_enabled = self._bool("tool_decision_enabled", True)
        mcp_decision_enabled = self._bool("mcp_decision_enabled", True)
        skill_decision_enabled = self._bool("skill_decision_enabled", True)
        subagent_decision_enabled = self._bool("subagent_decision_enabled", True)
        original = list(tool_set.tools)
        handoffs = [tool for tool in original if is_handoff_tool(tool) and tool_is_active(tool)]
        mcp_tools = [tool for tool in original if is_mcp_tool(tool) and tool_is_active(tool)]
        skills = [skill for skill in self._skill_objects() if tool_is_active(skill)]
        if not original and not skills:
            return
        ordinary = [
            tool
            for tool in original
            if not is_handoff_tool(tool)
            and not is_mcp_tool(tool)
            and getattr(tool, "name", None) != DECISION_TOOL_NAME
            and tool_is_active(tool)
        ]
        always_keep = {
            str(name).strip()
            for name in (self._setting("always_keep_tools", []) or [])
            if str(name).strip()
        }
        # A stochastic tool decision cannot remove the persistence mechanism
        # required to honour this turn's explicit time/preference request.
        if event is not None and event.get_extra("decision_temporal_guard", False):
            always_keep.add("future_contact")
        if event is not None and event.get_extra("decision_contact_feedback", ""):
            always_keep.add("adjust_contact_frequency")
        if not self._bool("always_keep_tools_customized", False):
            always_keep.update(self._builtin_tool_names())
            # jev_decide is registered by this plugin rather than AstrBot's
            # builtin manager, but it is still a builtin Decision Engine tool
            # and must start in the WebUI's Always Keep selection.
            always_keep.add(DECISION_TOOL_NAME)
        always_keep_recommend = {
            str(name).strip()
            for name in (self._setting("always_keep_recommend_tools", []) or [])
            if str(name).strip()
        } & always_keep
        always_keep_subagents = {
            str(name).strip()
            for name in (self._setting("always_keep_subagents", []) or [])
            if str(name).strip()
        }
        always_keep_recommend_subagents = {
            str(name).strip()
            for name in (self._setting("always_keep_recommend_subagents", []) or [])
            if str(name).strip()
        } & always_keep_subagents
        always_keep_mcp = {
            str(name).strip()
            for name in (self._setting("always_keep_mcp", []) or [])
            if str(name).strip()
        }
        always_keep_skills = {
            str(name).strip()
            for name in (self._setting("always_keep_skills", []) or [])
            if str(name).strip()
        }
        always_keep_recommend_skills = {
            str(name).strip()
            for name in (self._setting("always_keep_recommend_skills", []) or [])
            if str(name).strip()
        } & always_keep_skills
        # Before MCP had its own WebUI list, MCP tools were accepted by the
        # ordinary Tool list.  Keep those selections effective until the user
        # next saves the new three-section settings page.
        configured_tool_keeps = {
            str(name).strip()
            for name in (self._setting("always_keep_tools", []) or [])
            if str(name).strip()
        }
        always_keep_mcp.update(
            name
            for name in configured_tool_keeps
            if any(str(getattr(tool, "name", "")) == name for tool in mcp_tools)
        )
        always_keep_recommend_mcp = {
            str(name).strip()
            for name in (self._setting("always_keep_recommend_mcp", []) or [])
            if str(name).strip()
        } & always_keep_mcp
        always_keep_recommend_mcp.update(
            name
            for name in (self._setting("always_keep_recommend_tools", []) or [])
            if str(name).strip() in always_keep_mcp
            and any(str(getattr(tool, "name", "")) == str(name).strip() for tool in mcp_tools)
        )

        unknown_always_keep = always_keep - {
            str(getattr(tool, "name", "")) for tool in original if getattr(tool, "name", None)
        }
        if unknown_always_keep:
            logger.debug(
                "Decision Engine ignored %d Always Keep names not present in this request.",
                len(unknown_always_keep),
            )
        # Only the categories whose decision switches are enabled need Jev
        # questions.  Filtering is intentionally independent: a judged
        # category with filtering disabled remains fully visible while its
        # Jev-selected candidates can still be recommended.
        decision_ordinary = ordinary if tool_decision_enabled else []
        decision_mcp = mcp_tools if mcp_decision_enabled else []
        decision_skills = skills if skill_decision_enabled else []
        decision_handoffs = handoffs if subagent_decision_enabled else []
        question_to_tool: dict[str, str] = {}
        question_to_mcp: dict[str, str] = {}
        question_to_skill: dict[str, str] = {}
        questions: dict[str, dict[str, Any]] = {}
        for index, tool in enumerate(decision_ordinary):
            name = str(getattr(tool, "name", ""))
            qid = question_id("tool", name, index)
            question_to_tool[qid] = name
            questions[qid] = {
                "type": "noul",
                "instructions": (
                    "Should this tool be available to and recommended to the main LLM "
                    "for the current request? Return a high probability only when it "
                    "could materially help; zero recommendations are allowed."
                ),
            }

        for index, tool in enumerate(decision_mcp):
            name = str(getattr(tool, "name", ""))
            if not name:
                continue
            qid = question_id("mcp", name, index)
            question_to_mcp[qid] = name
            questions[qid] = {
                "type": "noul",
                "instructions": (
                    "Should this MCP tool be available to and recommended to the main LLM "
                    "for the current request? Return a high probability only when it could "
                    "materially help; zero recommendations are allowed."
                ),
            }

        for index, skill in enumerate(decision_skills):
            name = str(getattr(skill, "name", ""))
            if not name:
                continue
            qid = question_id("skill", name, index)
            question_to_skill[qid] = name
            questions[qid] = {
                "type": "noul",
                "instructions": (
                    "Should this Skill be available to and recommended to the main LLM "
                    "for the current request? Return a high probability only when it "
                    "could materially help; zero recommendations are allowed."
                ),
            }

        subagent_question_to_name: dict[str, str] = {}
        for index, tool in enumerate(decision_handoffs):
            name = str(getattr(tool, "name", ""))
            if not name:
                continue
            qid = question_id("subagent", name, index)
            subagent_question_to_name[qid] = name
            questions[qid] = {
                "type": "noul",
                "instructions": (
                    "Should the main LLM consider delegating this request to this "
                    "SubAgent? Return a high probability only when delegation could "
                    "materially help; zero or multiple recommendations are allowed."
                ),
            }

        decision_tool = self._decision_tool
        if not questions:
            # Even when there are no ordinary tools, the handoffs and the
            # explicitly kept Decision Engine tool are routed consistently.
            outcome = choose_tools(
                original,
                {},
                {},
                threshold=self._float("tool_noul_threshold", DEFAULT_TOOL_NOUL_THRESHOLD),
                always_keep=always_keep,
                decision_tool=decision_tool,
                always_keep_recommend=always_keep_recommend,
                always_keep_handoffs=always_keep_subagents,
                always_keep_handoffs_recommend=always_keep_recommend_subagents,
                question_to_handoff=subagent_question_to_name,
                filter_ordinary=tool_filter_enabled,
                always_keep_mcp=always_keep_mcp,
                always_keep_mcp_recommend=always_keep_recommend_mcp,
                question_to_mcp=question_to_mcp,
                filter_mcp=mcp_filter_enabled,
                skills=skills,
                always_keep_skills=always_keep_skills,
                always_keep_skills_recommend=always_keep_recommend_skills,
                question_to_skill=question_to_skill,
                filter_skills=skill_filter_enabled,
                filter_handoffs=subagent_filter_enabled,
            )
            add_routing_hint(
                req,
                template=self._main_llm_post_prompt(),
                recommended_tools=outcome.recommended_tools,
                recommended_mcp=outcome.recommended_mcp,
                recommended_skills=outcome.recommended_skills,
                recommended_subagents=outcome.recommended_subagents,
            )
            if skill_filter_enabled:
                self._apply_skill_visibility(req, outcome.selected_skills)
            req.func_tool.tools = outcome.selected
            self._log_routing_outcome(outcome)
            return

        try:
            batches = self._decision_batches(
                req,
                decision_ordinary,
                decision_mcp,
                decision_skills,
                decision_handoffs,
                questions,
                question_to_tool,
                question_to_mcp,
                question_to_skill,
                subagent_question_to_name,
            )
            tasks = [
                asyncio.create_task(self._evaluate(state=state, questions=batch_questions))
                for state, batch_questions in batches
            ]
            try:
                results = await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
        except Exception as exc:
            # Tool filtering is deliberately fail-open: leave the original
            # request untouched. Respect an explicit decision-tool opt-out.
            logger.warning("Decision Engine tool filter unavailable: %s", _safe_error(exc))
            # When ordinary Tool filtering is disabled, the whole ordinary
            # category is pass-through even if Jev itself is unavailable.
            # Keep ``jev_decide`` in that case just as the normal routing path
            # does; remove it only when filtering is actually enabled and the
            # user did not select Always Keep.
            if tool_filter_enabled and DECISION_TOOL_NAME not in always_keep:
                req.func_tool.tools = [
                    tool
                    for tool in tool_set.tools
                    if getattr(tool, "name", None) != DECISION_TOOL_NAME
                ]
            # Manual recommendations do not depend on Jev's availability.
            # Preserve them in the main LLM system prompt even when the
            # decision request failed and routing fell back to fail-open.
            manual_recommended_tools = [
                str(getattr(tool, "name", ""))
                for tool in original
                if getattr(tool, "name", None)
                and not is_handoff_tool(tool)
                and str(getattr(tool, "name", "")) in always_keep_recommend
            ]
            manual_recommended_subagents = [
                str(getattr(tool, "name", ""))
                for tool in original
                if getattr(tool, "name", None)
                and is_handoff_tool(tool)
                and str(getattr(tool, "name", "")) in always_keep_recommend_subagents
            ]
            manual_recommended_mcp = [
                str(getattr(tool, "name", ""))
                for tool in original
                if getattr(tool, "name", None)
                and is_mcp_tool(tool)
                and str(getattr(tool, "name", "")) in always_keep_recommend_mcp
            ]
            manual_recommended_skills = [
                str(getattr(skill, "name", ""))
                for skill in skills
                if getattr(skill, "name", None)
                and str(getattr(skill, "name", "")) in always_keep_recommend_skills
            ]
            add_routing_hint(
                req,
                template=self._main_llm_post_prompt(),
                recommended_tools=manual_recommended_tools,
                recommended_mcp=manual_recommended_mcp,
                recommended_skills=manual_recommended_skills,
                recommended_subagents=manual_recommended_subagents,
            )
            return

        noul_answers = {
            qid: answer
            for result in results
            for qid, answer in result.answers.items()
            if qid in question_to_tool
            or qid in question_to_mcp
            or qid in question_to_skill
            or qid in subagent_question_to_name
        }
        outcome = choose_tools(
            original,
            noul_answers,
            question_to_tool,
            threshold=min(
                1.0,
                max(0.0, self._float("tool_noul_threshold", DEFAULT_TOOL_NOUL_THRESHOLD)),
            ),
            always_keep=always_keep,
            decision_tool=decision_tool,
            always_keep_recommend=always_keep_recommend,
            always_keep_handoffs=always_keep_subagents,
            always_keep_handoffs_recommend=always_keep_recommend_subagents,
            question_to_handoff=subagent_question_to_name,
            filter_ordinary=tool_filter_enabled,
            always_keep_mcp=always_keep_mcp,
            always_keep_mcp_recommend=always_keep_recommend_mcp,
            question_to_mcp=question_to_mcp,
            filter_mcp=mcp_filter_enabled,
            skills=skills,
            always_keep_skills=always_keep_skills,
            always_keep_skills_recommend=always_keep_recommend_skills,
            question_to_skill=question_to_skill,
            filter_skills=skill_filter_enabled,
            filter_handoffs=subagent_filter_enabled,
        )
        recommended_subagents = outcome.recommended_subagents
        recommended_tools = outcome.recommended_tools
        recommended_mcp = outcome.recommended_mcp
        add_routing_hint(
            req,
            template=self._main_llm_post_prompt(),
            recommended_tools=recommended_tools,
            recommended_mcp=recommended_mcp,
            recommended_skills=outcome.recommended_skills,
            recommended_subagents=recommended_subagents,
        )
        if skill_filter_enabled:
            self._apply_skill_visibility(req, outcome.selected_skills)
        req.func_tool.tools = outcome.selected
        outcome.recommended_subagents = recommended_subagents
        self._log_routing_outcome(outcome)

    async def _run_decision_tool(
        self,
        event: AstrMessageEvent,
        state: str,
        decision_type: str,
        instructions: str,
        choice_options: list[dict[str, str]] | None = None,
        score_levels: list[str] | None = None,
    ) -> str:
        """Handler exposed to the main LLM as ``jev_decide``."""

        if decision_type not in {"noul", "choice", "score"}:
            return "Decision Engine error: decision_type must be noul, choice, or score."
        question: dict[str, Any] = {"type": decision_type, "instructions": instructions}
        if decision_type == "choice":
            options = choice_options or []
            criteria = {
                str(item.get("key", "")).strip(): str(item.get("description", "")).strip()
                for item in options
                if isinstance(item, Mapping)
                and str(item.get("key", "")).strip()
                and str(item.get("description", "")).strip()
            }
            if len(criteria) < 2:
                return (
                    "Decision Engine error: choice_options must contain at least two keyed options."
                )
            question["criteria"] = criteria
        elif decision_type == "score":
            levels = [str(level).strip() for level in (score_levels or []) if str(level).strip()]
            if len(levels) < 2:
                return "Decision Engine error: score_levels must contain at least two levels."
            question["criteria"] = levels
        try:
            result = await self._evaluate(state=str(state), questions={"decision": question})
        except Exception as exc:
            return f"Decision Engine unavailable: {_safe_error(exc)}"
        return compact_answer(result.answers["decision"])

    def _builtin_active_reply_enabled(self, event: AstrMessageEvent) -> bool:
        try:
            cfg = self.context.get_config(umo=event.unified_msg_origin)
            return bool(
                cfg.get("provider_ltm_settings", {}).get("active_reply", {}).get("enable", False)
            )
        except Exception:
            return False

    def _message_address_flags(self, event: AstrMessageEvent) -> tuple[bool, bool, bool, bool]:
        """Return (at-self, reply-to-self, at-all, parse-failed) safely."""

        at_self = reply_self = at_all = False
        try:
            from astrbot.api.message_components import At, AtAll, Reply

            self_id = str(event.get_self_id() or "")
            for component in event.get_messages():
                if isinstance(component, AtAll):
                    at_all = True
                elif isinstance(component, At):
                    target = getattr(component, "qq", None) or getattr(component, "target", None)
                    if str(target or "") == self_id:
                        at_self = True
                elif isinstance(component, Reply):
                    sender_id = str(
                        getattr(component, "sender_id", None)
                        or getattr(component, "sender", None)
                        or ""
                    )
                    if sender_id and sender_id == self_id:
                        reply_self = True
        except (AttributeError, ImportError, KeyError, TypeError, ValueError):
            return at_self, reply_self, at_all, True
        return at_self, reply_self, at_all, False

    def _dialogue_message_metadata(
        self, event: AstrMessageEvent
    ) -> tuple[tuple[tuple[str, str], ...], str]:
        """Extract only stable At/Reply identity metadata for flow analysis."""

        targets: list[tuple[str, str]] = []
        reply_to_id = ""
        try:
            from astrbot.api.message_components import At, AtAll, Reply

            for component in event.get_messages():
                # ``AtAll`` subclasses ``At`` in AstrBot.  It addresses the
                # whole group, so it must never become a concrete dialogue
                # target such as the literal ``all`` user ID.
                if isinstance(component, AtAll):
                    continue
                if isinstance(component, At):
                    target_id = str(
                        getattr(component, "qq", None) or getattr(component, "target", None) or ""
                    ).strip()
                    if target_id.lower() == "all":
                        continue
                    if target_id:
                        target_name = str(getattr(component, "name", None) or target_id).strip()
                        targets.append((target_id, target_name or target_id))
                elif isinstance(component, Reply):
                    reply_to_id = str(
                        getattr(component, "sender_id", None)
                        or getattr(component, "sender", None)
                        or ""
                    ).strip()
        except (AttributeError, ImportError, KeyError, TypeError, ValueError):
            return (), reply_to_id
        # Preserve order while avoiding duplicate At components from adapters.
        unique: list[tuple[str, str]] = []
        seen: set[str] = set()
        for target_id, target_name in targets:
            if target_id in seen:
                continue
            seen.add(target_id)
            unique.append((target_id, target_name))
        return tuple(unique), reply_to_id

    def _infer_proactive_dialogue(
        self,
        event: AstrMessageEvent,
        *,
        session: str,
        sender: str,
        sender_id: str,
        text: str,
        at_targets: tuple[tuple[str, str], ...] | None = None,
        reply_to_id: str | None = None,
    ) -> DialogueInference:
        """Infer the current addressee before the record enters session history."""

        if not self._conversation_flow_enabled_for(event):
            return DialogueInference()
        if at_targets is None or reply_to_id is None:
            at_targets, reply_to_id = self._dialogue_message_metadata(event)
        current = ProactiveRecord(
            sender=sender,
            sender_id=sender_id,
            text=text,
            reply_to_id=reply_to_id,
            at_targets=at_targets,
        )
        return infer_dialogue_target(
            current,
            self.proactive.history_records(session)[
                -max(2, min(30, self._int("conversation_flow_window", 8))) :
            ],
            bot_id=str(event.get_self_id() or ""),
            reply_starters=DEFAULT_REPLY_STARTERS,
        )

    def _record_proactive_input(
        self,
        event: AstrMessageEvent,
        *,
        session: str,
        sender: str,
        sender_id: str,
        text: str,
    ) -> tuple[list[str], DialogueInference]:
        """Record one non-command message and return its Jev-facing snapshot."""

        flow_enabled = self._conversation_flow_enabled_for(event)
        flow_window = max(2, min(30, self._int("conversation_flow_window", 8)))
        history = self.proactive.history_lines(
            session,
            self._int("history_max_chars", 6000),
            include_dialogue=flow_enabled,
            max_messages=(flow_window if flow_enabled else None),
        )
        at_targets, reply_to_id = self._dialogue_message_metadata(event)
        dialogue = self._infer_proactive_dialogue(
            event,
            session=session,
            sender=sender,
            sender_id=sender_id,
            text=text,
            at_targets=at_targets,
            reply_to_id=reply_to_id,
        )
        self.proactive.add(
            session,
            ProactiveRecord(
                sender,
                sender_id,
                text,
                reply_to_id=reply_to_id,
                at_targets=at_targets,
                talking_to=dialogue.target_id,
                talking_to_name=dialogue.target_name,
            ),
        )
        return history, dialogue

    def _direct_reply_requested(self, event: AstrMessageEvent) -> bool:
        """Implement AngelHeart's direct prefix feature without text @ matching."""

        prefixes = normalize_prefixes(self._setting("direct_reply_prefixes", ["/", "@"]))
        at_self, _, at_all, parse_failed = self._message_address_flags(event)
        if not parse_failed and not at_all and "@" in prefixes and at_self:
            return True
        try:
            outline = event.get_message_outline()
        except (AttributeError, TypeError):
            outline = event.get_message_str()
        return text_matches_prefix(str(outline or ""), prefixes)

    def _proactive_summoned(self, event: AstrMessageEvent) -> bool:
        at_self, reply_self, at_all, parse_failed = self._message_address_flags(event)
        if not parse_failed and not at_all and (at_self or reply_self):
            return True
        try:
            text = str(event.get_message_outline() or event.get_message_str() or "")
        except (AttributeError, TypeError):
            text = str(event.get_message_str() or "")
        aliases = [
            item.strip()
            for item in str(self._setting("proactive_alias", "AI|助手") or "").split("|")
            if item.strip()
        ]
        return bool(aliases and any(alias in text for alias in aliases))

    def _event_scope_candidates(self, event: AstrMessageEvent | None) -> set[str]:
        """Collect stable IDs accepted by both scope editors."""

        candidates: set[str] = set()
        try:
            origin = str(event.unified_msg_origin).strip()
            if origin:
                candidates.add(origin)
        except (AttributeError, TypeError):
            pass

        # Keep group and private identifiers in their own namespaces.  A
        # private-chat user ID in the allowlist must not accidentally enable
        # proactive replies for that same user's messages in every group.
        try:
            message_type = event.get_message_type()
        except (AttributeError, TypeError, ValueError):
            message_type = None
        if message_type == MessageType.GROUP_MESSAGE:
            method_names = ("get_group_id",)
        elif message_type == MessageType.FRIEND_MESSAGE:
            method_names = ("get_sender_id",)
        else:
            method_names = ()
        for method_name in method_names:
            method = getattr(event, method_name, None)
            if not callable(method):
                continue
            try:
                value = method()
            except (AttributeError, TypeError, ValueError):
                continue
            if value is not None and str(value).strip():
                candidates.add(str(value).strip())
        return candidates

    def _scope_allows(
        self,
        event: AstrMessageEvent | None,
        values_key: str,
        blacklist_key: str,
        *,
        default_blacklist: bool,
    ) -> bool:
        """Apply the UI's empty-list semantics for a black/white scope.

        Blacklist mode + empty list means all conversations apply.  Whitelist
        mode + empty list means no conversation applies.  This is deliberately
        shared by active replies and both dialogue-flow enhancements so the
        two editors cannot drift apart.
        """

        configured = self._setting(values_key, [])
        values = (
            {str(item).strip() for item in configured if str(item).strip()}
            if isinstance(configured, (list, tuple, set))
            else set()
        )
        matched = bool(self._event_scope_candidates(event) & values)
        blacklist = self._bool(blacklist_key, default_blacklist)
        return (not matched) if blacklist else matched

    def _proactive_whitelist_allows(self, event: AstrMessageEvent) -> bool:
        """Apply the independent active-conversation black/white scope."""

        return self._scope_allows(
            event,
            "proactive_whitelist",
            "proactive_scope_blacklist",
            default_blacklist=False,
        )

    def _tools_scope_allows(self, event: AstrMessageEvent | None) -> bool:
        """Apply the shared Tool/MCP/Skill/SubAgent decision black/white list."""

        return self._scope_allows(
            event,
            "tools_scope",
            "tools_scope_blacklist",
            default_blacklist=True,
        )

    def _configured_scope_values(self, key: str) -> list[str]:
        """Return a normalized scope list for WebUI responses."""

        configured = self._setting(key, [])
        if not isinstance(configured, (list, tuple, set)):
            return []
        return [str(item).strip() for item in configured if str(item).strip()]

    def _conversation_flow_enabled_for(self, event: AstrMessageEvent) -> bool:
        """Return whether input/output dialogue enhancement applies here."""

        return (
            self._bool("conversation_flow_enabled", True)
            and self._bool("conversation_flow_analysis_enabled", True)
            and self._scope_allows(
                event,
                "conversation_flow_scope",
                "conversation_flow_scope_blacklist",
                default_blacklist=True,
            )
        )

    def _should_skip_proactive(self, event: AstrMessageEvent) -> bool:
        message_type = event.get_message_type()
        if message_type not in {MessageType.GROUP_MESSAGE, MessageType.FRIEND_MESSAGE}:
            return True
        if self._command_handler_matched(event):
            return True
        sender_id = str(event.get_sender_id() or "")
        self_id = str(event.get_self_id() or "")
        if sender_id and self_id and sender_id == self_id:
            return True
        # AstrBot has already decided this is a wake/command event; allow its
        # native pipeline to handle it and do not issue a second Jev request.
        if bool(getattr(event, "is_at_or_wake_command", False)):
            return True
        if message_type == MessageType.GROUP_MESSAGE:
            _, _, at_all, _ = self._message_address_flags(event)
            return at_all
        return False

    @filter.event_message_type(
        EventMessageType.GROUP_MESSAGE | EventMessageType.PRIVATE_MESSAGE,
        priority=1000,
    )
    async def proactive_reply(self, event: AstrMessageEvent):
        """Use Jev to decide whether AstrBot's normal Agent should interject."""

        if not self._bool("message_interjection_enabled", True):
            return
        if not self._bool("proactive_reply_enabled", False):
            return
        if not self._proactive_whitelist_allows(event):
            return
        if self._should_skip_proactive(event):
            return

        # A direct prefix is always handled by AstrBot's native Agent path;
        # native active-reply settings must not prevent this explicit wake-up.
        if self._direct_reply_requested(event):
            session = str(event.unified_msg_origin)
            text = str(event.get_message_str() or "").strip()
            if text:
                sender_id = str(event.get_sender_id() or "")
                sender_name = str(event.get_sender_name() or sender_id or "user")
                async with self.proactive.lock_for(session):
                    self._record_proactive_input(
                        event,
                        session=session,
                        sender=sender_name,
                        sender_id=sender_id,
                        text=text,
                    )
            event.is_at_or_wake_command = True
            self.proactive.set_status(session, ProactiveStatus.SUMMONED)
            logger.warning("Decision Engine 因为命中直接回复前缀或 @ 机器人，所以主动对话。")
            return

        if not self.provider or not self._ready:
            return

        if self._builtin_active_reply_enabled(event):
            if not self._active_reply_warning_emitted:
                logger.warning(
                    "Decision Engine proactive reply is disabled for this event because AstrBot active_reply is enabled."
                )
                self._active_reply_warning_emitted = True
            return

        session = str(event.unified_msg_origin)
        text = str(event.get_message_str() or "").strip()
        media_refs = await self._media_refs_for(event)
        if not text:
            # Pure media messages only become candidates when the operator turns
            # this on; understanding itself is controlled separately.
            if not self._bool("non_text_proactive_enabled", False) or not media_refs:
                return
            text = await self._media_placeholder(media_refs, session, event)
        elif media_refs and self._bool("non_text_proactive_enabled", False):
            attachment_context = await self._media_placeholder(media_refs, session, event)
            text = f"{text}\n{attachment_context}"
        sender_id = str(event.get_sender_id() or "")
        sender_name = str(event.get_sender_name() or sender_id or "user")
        at_self, reply_self, at_all, parse_failed = self._message_address_flags(event)
        explicit_summon = not parse_failed and not at_all and (at_self or reply_self)
        summoned = explicit_summon or self._proactive_summoned(event)
        async with self.proactive.lock_for(session):
            history, dialogue = self._record_proactive_input(
                event,
                session=session,
                sender=sender_name,
                sender_id=sender_id,
                text=text,
            )
            flow_enabled = self._conversation_flow_enabled_for(event)
            # Keep every allowlisted message in the bounded session history,
            # while the analysis cooldown suppresses only the Jev request.
            if not self.proactive.can_analyze(session):
                return
            status = self.proactive.observe_message(
                session,
                text=text,
                sender_id=sender_id,
                summoned=summoned,
                echo_threshold=self._int("echo_detection_threshold", 3),
                echo_window=self._float("echo_detection_window_seconds", 30.0),
                dense_threshold=self._int("dense_conversation_threshold", 30),
                dense_window=self._float("dense_conversation_window_seconds", 600.0),
                min_participants=self._int("min_participant_count", 2),
                observation_timeout=self._float("observation_timeout_seconds", 600.0),
            )
            if explicit_summon and self._bool("force_reply_when_summoned", True):
                should_reply = True
                scores: dict[str, float] = {}
                aggregate = 1.0
                logger.warning(
                    "Decision Engine 因为明确 @ 或回复机器人且启用强制回复，所以主动对话。"
                )
            else:
                questions = {
                    "is_addressing_bot": {
                        "type": "noul",
                        "instructions": "Is the current message meaningfully addressing the bot?",
                    },
                    "should_interject": {
                        "type": "noul",
                        "instructions": "Would a brief, helpful bot reply now be natural rather than disruptive?",
                    },
                    "conversation_relevance": {
                        "type": "noul",
                        "instructions": "Would the bot add relevant value to this conversation right now?",
                    },
                    "timing": {
                        "type": "noul",
                        "instructions": "Is this a good moment for one concise bot reply?",
                    },
                    "continuity": {
                        "type": "noul",
                        "instructions": "Would replying continue the current conversation naturally?",
                    },
                }
                state = self._fit_active_state(
                    history=history,
                    sender_name=sender_name,
                    sender_id=sender_id,
                    text=text,
                    status=status,
                    summoned=summoned,
                    dialogue=dialogue,
                    dialogue_enabled=flow_enabled,
                    questions=questions,
                )
                try:
                    result = await self._evaluate(state=state, questions=questions)
                except Exception as exc:
                    self.proactive.mark_analysis(
                        session,
                        success=False,
                        # Retry backoff controls the gap between transport
                        # attempts inside one Jev request.  Once all attempts
                        # fail, use the proactive analysis interval instead
                        # of zero so an API outage cannot trigger one new
                        # request for every incoming message.
                        no_reply_cooldown=self._float("no_reply_cooldown_seconds", 3.0),
                    )
                    logger.warning(
                        "Decision Engine proactive check closed after failure: %s", _safe_error(exc)
                    )
                    return
                names = tuple(questions)
                scores = parse_noul_scores(result.answers, names)
                weights = {
                    "is_addressing_bot": 1.4,
                    "should_interject": 1.4,
                    "conversation_relevance": 1.2,
                    "timing": 0.8,
                    "continuity": 0.8,
                }
                aggregate = aggregate_scores(scores, weights)
                addressing = scores.get("is_addressing_bot", 0.0)
                interject = scores.get("should_interject", 0.0)
                should_reply = bool(
                    len(scores) == len(names)
                    and (
                        aggregate
                        >= bounded_float(self._setting("proactive_score_threshold", 0.68), 0.68)
                        or (
                            addressing
                            >= bounded_float(self._setting("addressing_threshold", 0.7), 0.7)
                            and interject
                            >= bounded_float(self._setting("interject_threshold", 0.85), 0.85)
                        )
                    )
                )
                logger.warning(
                    "Decision Engine 因为 Jev 综合分 %.3f（指向 %.3f，介入 %.3f），所以%s。",
                    aggregate,
                    addressing,
                    interject,
                    "主动对话" if should_reply else "不主动对话",
                )
            self.proactive.mark_analysis(
                session,
                success=True,
                no_reply_cooldown=self._float("no_reply_cooldown_seconds", 3.0),
            )
            if not should_reply:
                self.proactive.set_status(session, ProactiveStatus.OBSERVATION)
                return
            if not self.proactive.allow_reply(
                session,
                cooldown_seconds=self._float("cooldown_seconds", 60),
                window_seconds=self._float("reply_window_seconds", 600),
                max_replies=self._int("max_replies_per_window", 2),
            ):
                logger.warning("Decision Engine 因为主动回复冷却或窗口次数限制，所以不主动对话。")
                return

            try:
                cid = await self.context.conversation_manager.get_curr_conversation_id(session)
                conversation = await self.context.conversation_manager.get_conversation(
                    session, cid
                )
                if not conversation:
                    self.proactive.cancel_reply(session)
                    return
                event.is_at_or_wake_command = True
                # This request is already being sent through the handler's
                # ProviderRequest path.  Mark it as consumed so AstrBot's
                # ProcessStage does not issue a second native LLM request
                # merely because the event is also marked as awakened.
                event.should_call_llm(True)
                yield event.request_llm(
                    prompt=text,
                    # Continue AstrBot's native conversation so persona and
                    # other plugin system prompts remain intact.
                    session_id=cid,
                    conversation=conversation,
                )
            except Exception as exc:
                self.proactive.cancel_reply(session)
                logger.debug(
                    "Decision Engine proactive request was not queued: %s", _safe_error(exc)
                )

    @filter.event_message_type(
        EventMessageType.GROUP_MESSAGE | EventMessageType.PRIVATE_MESSAGE,
        priority=1000,
    )
    async def output_enhancement_message(self, event: AstrMessageEvent) -> None:
        """Collect the small amount of input state required by OutputPro."""

        runtime = self._output_runtime
        if (
            runtime is None
            or not self._bool("conversation_flow_enabled", True)
            or not self._scope_allows(
                event,
                "output_scope",
                "output_scope_blacklist",
                default_blacklist=True,
            )
        ):
            return
        await runtime.prepare_message(event)

    @(
        _ON_DECORATING_RESULT(priority=15)
        if callable(_ON_DECORATING_RESULT)
        else (lambda target: target)
    )
    async def output_enhancement(self, event: AstrMessageEvent) -> None:
        """Run the optional OutputPro-compatible last-mile pipeline.

        The scope is intentionally independent from active conversation.  A
        user can therefore enable output formatting for a set of groups while
        keeping Jev proactive decisions disabled there.
        """

        runtime = self._output_runtime
        if (
            runtime is None
            or not self._bool("conversation_flow_enabled", True)
            or not self._scope_allows(
                event,
                "output_scope",
                "output_scope_blacklist",
                default_blacklist=True,
            )
        ):
            return
        await runtime.run(event)

    @filter.on_llm_response()
    async def remember_bot_response(self, event: AstrMessageEvent, response: Any) -> None:
        if event.get_extra("decision_life_proactive", False):
            return
        if not self._bool("proactive_reply_enabled", False):
            return
        if event.get_message_type() not in {
            MessageType.GROUP_MESSAGE,
            MessageType.FRIEND_MESSAGE,
        }:
            return
        if not self._proactive_whitelist_allows(event):
            return
        text = str(getattr(response, "completion_text", "") or "").strip()
        session = str(event.unified_msg_origin)
        # The response hook also sees ordinary user-addressed Agent replies;
        # only a slot reserved by our proactive path may update its state.
        if self.proactive.has_pending_reply(session):
            self.proactive.mark_reply_success(session)
        if text:
            replied_to_id = str(event.get_sender_id() or "")
            replied_to_name = str(event.get_sender_name() or replied_to_id or "用户")
            self.proactive.add(
                session,
                ProactiveRecord(
                    "bot",
                    str(event.get_self_id() or "bot"),
                    text,
                    True,
                    talking_to=replied_to_id or GROUP_TARGET_ID,
                    talking_to_name=replied_to_name or GROUP_TARGET_NAME,
                ),
            )

    def _is_admin(self, event: AstrMessageEvent) -> bool:
        try:
            admins = self.context.get_config().get("admins_id", [])
            return str(event.get_sender_id()) in {str(item) for item in admins}
        except Exception:
            return False

    @filter.command("decision")
    async def decision_command(self, event: AstrMessageEvent):
        """管理员诊断：/decision status、/decision test、/decision tools。"""

        if not self._is_admin(event):
            event.stop_event()
            return
        text = str(event.get_message_str() or "").strip().lstrip("/")
        parts = text.split(maxsplit=1)
        action = parts[1].strip().lower() if len(parts) > 1 else "status"
        if action == "status":
            tools = self._manager_tool_objects()
            handoffs = sum(isinstance(item, HandoffTool) for item in tools)
            output = (
                f"Tool/MCP/Skill/SubAgent decision: {'enabled' if self._bool('tools_subagents_decision_enabled', True) else 'disabled'}\n"
                f"proactive_reply={self._bool('proactive_reply_enabled', False)}\n"
                f"provider={self._setting('provider', 'systemone_jev')}\n"
                f"endpoint={self._setting('base_url', '')}{self._setting('systemone_path', '/v1/systemone')}\n"
                f"model={self._setting('model', 'jev-latest')}\n"
                f"tool_filter={self._bool('tool_filter_enabled', True)} mcp_filter={self._bool('mcp_filter_enabled', True)} skill_filter={self._bool('skill_filter_enabled', True)} subagent_filter={self._bool('subagent_filter_enabled', False)} "
                f"tool_decision={self._bool('tool_decision_enabled', True)} mcp_decision={self._bool('mcp_decision_enabled', True)} skill_decision={self._bool('skill_decision_enabled', True)} subagent_decision={self._bool('subagent_decision_enabled', True)} "
                f"threshold={self._float('tool_noul_threshold', DEFAULT_TOOL_NOUL_THRESHOLD):.3f}\n"
                f"registered_tools={len(tools)} handoffs={handoffs} always_keep={len(self._setting('always_keep_tools', []) or [])}\n"
                f"calls={self._call_count} failures={self._failure_count} latency_ms={self._last_call_latency_ms or 0:.1f}"
            )
            yield event.plain_result(output)
        elif action == "test":
            try:
                result = await self._evaluate(
                    state="The service is available.",
                    questions={
                        "service": {
                            "type": "noul",
                            "instructions": "Is the service available?",
                        }
                    },
                )
                yield event.plain_result(
                    f"SystemOne OK: {compact_answer(result.answers['service'])}"
                )
            except Exception as exc:
                yield event.plain_result(f"SystemOne failed: {_safe_error(exc)}")
        elif action == "tools":
            tools = self._manager_tool_objects()
            names = [str(getattr(item, "name", "")) for item in tools]
            shown = names[:80]
            suffix = f"\n... 共 {len(names)} 个" if len(names) > len(shown) else ""
            yield event.plain_result("\n".join(shown) + suffix)
        else:
            yield event.plain_result("用法：/decision status | /decision test | /decision tools")
        event.stop_event()

    async def page_tools(self):
        from astrbot.api.web import json_response

        tools = []
        seen_names: set[str] = set()
        builtin_names = self._builtin_tool_names()
        allowed_mm = self._mm_allowed_tool_names(self._multimodal_settings())
        for tool in [*self._manager_tool_objects(), *self._subagent_tool_objects()]:
            name = str(getattr(tool, "name", ""))
            if not name or name in seen_names:
                continue
            if name.startswith("understand_") and name not in allowed_mm:
                continue
            seen_names.add(name)
            tools.append(
                {
                    "name": name,
                    "description": short_description(
                        getattr(tool, "description", ""),
                    ),
                    "handoff": is_handoff_tool(tool),
                    "mcp": is_mcp_tool(tool),
                    "skill": False,
                    "active": tool_is_active(tool),
                    "builtin": name in builtin_names,
                    "origin_display": self._tool_origin(tool, builtin=name in builtin_names),
                }
            )
        for tool in self._builtin_tool_objects():
            name = str(getattr(tool, "name", ""))
            if not name or name in seen_names:
                continue
            seen_names.add(name)
            tools.append(
                {
                    "name": name,
                    "description": short_description(
                        getattr(tool, "description", "") or "AstrBot 内置工具",
                    ),
                    "handoff": False,
                    "mcp": False,
                    "skill": False,
                    "active": tool_is_active(tool),
                    "builtin": True,
                    "origin_display": "Astrbot内置工具",
                }
            )
        if DECISION_TOOL_NAME not in seen_names:
            tools.insert(
                0,
                {
                    "name": DECISION_TOOL_NAME,
                    "description": "调用 Jev/SystemOne 对当前场景执行一次结构化判断；不会执行其他工具。",
                    "handoff": False,
                    "mcp": False,
                    "skill": False,
                    "active": True,
                    "builtin": False,
                    "origin_display": PLUGIN_DISPLAY_NAME,
                },
            )
        for skill in self._skill_objects():
            name = str(getattr(skill, "name", ""))
            if not name:
                continue
            tools.append(
                {
                    "name": name,
                    "description": short_description(getattr(skill, "description", "")),
                    "handoff": False,
                    "mcp": False,
                    "skill": True,
                    "active": tool_is_active(skill),
                    "builtin": False,
                    "origin_display": self._skill_origin(skill),
                }
            )
        tools.sort(
            key=lambda item: (
                bool(item.get("builtin")),
                str(item.get("origin_display", "")).casefold(),
                str(item.get("name", "")).casefold(),
            )
        )
        return json_response({"tools": tools})

    def _multimodal_settings(self) -> dict[str, Any]:
        from .decision.multimodal import normalize_settings

        return normalize_settings(dict(self._setting("multimodal", {}) or {}))

    def _get_multimodal(self):
        """Create the multi-modal runtime on first use."""

        if self._multimodal is not None:
            return self._multimodal
        from .decision.multimodal.runtime import MultimodalRuntime

        base = (
            Path(self._settings_path).parent
            if getattr(self, "_settings_path", None)
            else Path.cwd()
        )
        runtime = MultimodalRuntime(
            self, data_dir=base, settings=self._multimodal_settings()
        )
        self._multimodal = runtime
        return runtime

    async def _ensure_multimodal(self):
        runtime = self._get_multimodal()
        await runtime.initialize()
        self._sync_multimodal_tools(runtime)
        await self._apply_multimodal_tool_state(self._multimodal_settings())
        return runtime

    def _sync_multimodal_tools(self, runtime) -> None:
        """(Re)register the understanding tools according to the current switches."""

        from .decision.multimodal.tools import build_tools

        wanted = set(runtime.settings.tool_kinds)
        existing = {tool.name: tool for tool in self._multimodal_tools}
        for name, tool in list(existing.items()):
            if name not in {
                f"understand_{kind}" for kind in wanted
            }:
                self._life_tools.remove(tool) if tool in self._life_tools else None
                self._multimodal_tools.remove(tool)
        for tool in build_tools(runtime, tuple(wanted)):
            if any(item.name == tool.name for item in self._multimodal_tools):
                continue
            self._multimodal_tools.append(tool)
            self._life_tools.append(tool)
            self.context.add_llm_tools(tool)

    @staticmethod
    def _media_kind_summary(refs: Any) -> str:
        labels = {"image": "图片", "video": "视频", "audio": "音频", "file": "文件"}
        counts: dict[str, int] = {}
        for ref in refs:
            counts[ref.kind] = counts.get(ref.kind, 0) + 1
        return "、".join(f"{labels.get(kind, kind)}×{count}" for kind, count in counts.items())

    async def _media_placeholder(self, refs: Any, umo: str, event=None) -> str:
        """Give Jev attachment semantics when this turn's auto task is ready."""

        summary = self._media_kind_summary(refs)
        cached: list[str] = []
        runtime = self._multimodal
        if runtime is not None:
            if event is not None:
                items = await runtime.collect_for_event(
                    event,
                    list(refs[:4]),
                    wait=float(runtime.settings.int("mm_auto_wait_seconds")),
                )
                cached.extend(item.summary for item in items if getattr(item, "ok", False))
            else:
                for ref in refs[:4]:
                    item = await runtime.cache.get_for_media(umo, ref.content_key)
                    if item is not None and getattr(item, "ok", False):
                        cached.append(item.summary)
            if not cached:
                for item in await runtime.cache.recent_understandings(umo, 2):
                    cached.append(item.summary)
        detail = ""
        if cached:
            detail = "\n已有理解结果：" + "\n".join(cached)[:800]
        return f"[附件资料] 对方发送了{summary}。{detail}"

    async def _media_refs_for(self, event: AstrMessageEvent) -> list[Any]:
        """Media on this message, independent of the understanding engine.

        The 非文字消息 switch must work even when multi-modal understanding is
        disabled, so detection never depends on decision_mm_refs being produced.
        """

        refs = event.get_extra("decision_mm_refs")
        if refs:
            return list(refs)
        from .decision.multimodal.media import extract_media

        try:
            return await extract_media(event)
        except Exception as exc:
            logger.debug("Decision Engine 媒体探测失败：%s", _safe_error(exc))
            return []

    async def _get_life_runtime(self):
        async with self._life_lock:
            if self._life_runtime is None:
                from .decision.life_runtime import LifeRuntime

                settings = normalize_life(dict(self._setting("life", {}) or {}))
                runtime = LifeRuntime(self, settings)
                try:
                    await runtime.initialize()
                except BaseException:
                    await runtime.terminate()
                    raise
                self._life_runtime = runtime
            return self._life_runtime

    async def _configure_life_runtime(self):
        if self._life_runtime:
            async with self._life_lock:
                await self._life_runtime.terminate()
                self._life_runtime = None
        # 主动发起 is enabled by its own per-kind switches (私聊主动 / 群聊主动),
        # so there is no separate master switch to consult here.
        await self._get_life_runtime()

    def _get_life_media(self):
        if self._life_media is None:
            from .decision.life_media import LifeMedia

            self._life_media = LifeMedia(
                self,
                self._settings_path.parent.parent / "plugin_data" / PLUGIN_NAME / "virtual_life",
            )
        return self._life_media

    def _get_personal_network(self):
        """Return the merged Personal Network service (created lazily)."""

        if self._personal_network is None:
            from .decision.personal_network import PersonalNetworkService

            self._personal_network = PersonalNetworkService(
                self,
                self._settings_path.parent.parent
                / "plugin_data"
                / PLUGIN_NAME
                / "personal_network",
            )
        return self._personal_network

    async def _take_photo(
        self, event: AstrMessageEvent, scene: str = "", subject: str = "self", count: int = 1
    ) -> str:
        return await self._get_life_media().capture(
            event, kind="photo", scene=scene, subject=subject, count=count
        )

    async def _take_video(
        self, event: AstrMessageEvent, scene: str = "", subject: str = "self", count: int = 1
    ) -> str:
        return await self._get_life_media().capture(
            event, kind="video", scene=scene, subject=subject, count=count
        )

    async def _send_voice(
        self, event: AstrMessageEvent, text: str, style: str = "", count: int = 1
    ) -> str:
        return await self._get_life_media().capture(
            event, kind="audio", scene=style, audio_text=text, count=count
        )

    async def _get_virtual_daily_schedule(self, event: AstrMessageEvent) -> str:
        runtime = await self._get_life_runtime()
        return await runtime.get_virtual_daily_schedule(event)

    async def _get_long_term_timeline(self, event: AstrMessageEvent) -> str:
        runtime = await self._get_life_runtime()
        return await runtime.get_long_term_timeline(event)

    async def _add_schedule_requirement(
        self, event: AstrMessageEvent, date_range: str, requirement: str
    ) -> str:
        if getattr(event, "role", "member") != "admin":
            return "只有管理员可以添加日程要求。"
        runtime = await self._get_life_runtime()
        return await runtime._add_schedule_requirement(event, date_range, requirement)

    async def _adjust_contact_frequency(self, event, action="status", interval_minutes=0, pause_until=""):
        from .decision.life_temporal import parse_time

        runtime = await self._get_life_runtime()
        persona = await runtime.personas.resolve(str(event.unified_msg_origin))
        try:
            until = parse_time(pause_until, runtime.timezone).timestamp() if pause_until else 0
            return runtime.contacts.preference(event, persona.id, action, interval_minutes, until)
        except (ValueError, TypeError) as exc:
            return "未修改联系偏好：" + str(exc)

    async def _future_contact(self, event, action="list", task_id="", run_at="", delay_seconds=0,
                              note="", origin="user", kind="reminder", notify=True):
        runtime = await self._get_life_runtime()
        try:
            return await runtime.temporal.manage(event, action=action, task_id=task_id, run_at=run_at,
                delay_seconds=delay_seconds, note=note, origin=origin, kind=kind, notify=notify)
        except (ValueError, TypeError) as exc:
            return "未来任务未修改：" + str(exc)

    async def _save_life_thought(self, event, text):
        from .decision.virtual_life.core.utils import timeline_item_at

        runtime = await self._get_life_runtime()
        persona = await runtime.personas.resolve(str(event.unified_msg_origin))
        plan = runtime.visible_plan(runtime.storage.get_plan(runtime._now().date().isoformat(), persona.id))
        item = timeline_item_at(plan, runtime._now()) if plan else None
        if not item:
            return "没有可验证的当前日程活动，未保存想法。"
        try:
            ident = await runtime.ideas.save_thought(persona.id, text,
                {"activity": item.activity, "location": item.location, "at": runtime._now().isoformat()}, scope=str(event.unified_msg_origin))
            return "想法已暂存，是否分享由限频与偏好决定。记录 " + ident
        except ValueError as exc:
            return str(exc)

    @filter.on_llm_request(priority=1300)
    async def life_control_tool_state(self, event, req):
        runtime = self._life_runtime
        tools = getattr(req, "func_tool", None)
        if runtime is None or tools is None:
            return
        settings = runtime.config
        hide = set()
        for name, group, flag in (("adjust_contact_frequency", "adaptive_settings", "preference_tool_enable"),
            ("future_contact", "temporal_settings", "task_tool_enable"),
            ("save_life_thought", "thought_settings", "share_tool_enable")):
            if not settings[group]["enable"] or not settings[group][flag]:
                hide.add(name)
        if settings["temporal_settings"]["enable"]:
            hide.add("future_task")
        if event.get_extra("decision_life_proactive", False):
            # Only Decision wrappers can enforce the one-kind/count budget.
            hide.update({"generate_image", "generate_video", "generate_audio"})
            kind = event.get_extra("decision_life_media_kind", "")
            for name, media_kind in (("take_photo", "photo"), ("take_video", "video"), ("send_voice", "audio")):
                if kind != media_kind:
                    hide.add(name)
        if event.get_extra("decision_temporal_guard", False):
            hide.add("send_message_to_user")
        tools.tools = [tool for tool in tools.tools if getattr(tool, "name", "") not in hide]

    @filter.event_message_type(
        EventMessageType.GROUP_MESSAGE | EventMessageType.PRIVATE_MESSAGE, priority=1100
    )
    async def life_observe_message(self, event: AstrMessageEvent):
        if event.get_extra("decision_life_proactive", False):
            return
        if self._life_runtime:
            if self._command_handler_matched(event):
                self._life_runtime._mark_conversation_activity(str(event.unified_msg_origin))
                return
            await self._life_runtime.temporal.prepare_event(event)
            if self._life_runtime.config["continuity_settings"]["enable"] or self._life_runtime.config["temporal_settings"]["enable"]:
                persona = await self._life_runtime.personas.resolve(str(event.unified_msg_origin))
                self._life_runtime.delivery.wrap_event(event, persona.id)
            await self._life_runtime.record_incoming(event)

    @filter.event_message_type(
        EventMessageType.GROUP_MESSAGE | EventMessageType.PRIVATE_MESSAGE, priority=1001
    )
    async def multimodal_observe_message(self, event: AstrMessageEvent):
        """Index media and start background understanding for any message.

        This deliberately runs for messages that are not addressed to the bot: the
        listener only marks ``is_wake`` for plugin handlers and never sets
        ``is_at_or_wake_command``, so no reply is triggered here.
        """

        if event.get_extra("decision_life_proactive", False):
            return
        settings = self._multimodal_settings()
        if not settings.get("mm_enabled"):
            return
        try:
            runtime = self._get_multimodal()
            allowed = self._mm_scope_allows(event, settings)
            refs = await runtime.observe(event, auto_allowed=allowed)
            if refs:
                event.set_extra("decision_mm_refs", refs)
        except Exception as exc:
            logger.debug(
                "Decision Engine 多模态观察跳过：%s", _safe_error(exc)
            )

    @staticmethod
    def _mm_allowed_tool_names(settings: dict[str, Any]) -> set[str]:
        from .decision.multimodal.config import MEDIA_KINDS

        if not settings.get("mm_enabled") or not settings.get("mm_tools_enabled"):
            return set()
        return {
            f"understand_{kind}"
            for kind in MEDIA_KINDS
            if settings.get(f"mm_tool_{kind}")
        }

    async def _apply_multimodal_tool_state(self, settings: dict[str, Any]) -> None:
        """Activate or deactivate the understanding tools inside AstrBot itself.

        ``Context.activate_llm_tool`` / ``deactivate_llm_tool`` flip the global
        active flag used by JEV routing, the tool list and the actual call path, so
        a disabled tool is really unavailable instead of just hidden.
        """

        from .decision.multimodal.config import MEDIA_KINDS

        allowed = self._mm_allowed_tool_names(settings)
        for kind in MEDIA_KINDS:
            name = f"understand_{kind}"
            try:
                if name in allowed:
                    activate = getattr(self.context, "activate_llm_tool_async", None)
                    if callable(activate):
                        await activate(name)
                else:
                    deactivate = getattr(self.context, "deactivate_llm_tool_async", None)
                    if callable(deactivate):
                        await deactivate(name)
            except Exception as exc:  # pragma: no cover - host API differences
                logger.debug(
                    "Decision Engine 切换多模态工具 %s 失败：%s", name, _safe_error(exc)
                )

    def _mm_scope_allows(self, event: AstrMessageEvent, settings: dict[str, Any]) -> bool:
        """Apply the multi-modal auto-understanding black/white list."""

        configured = settings.get("mm_auto_scope") or []
        values = {str(item).strip() for item in configured if str(item).strip()}
        matched = bool(self._event_scope_candidates(event) & values)
        if settings.get("mm_auto_scope_blacklist"):
            return not matched
        return matched

    @filter.on_llm_request(priority=900)
    async def multimodal_prepare_request(
        self, event: AstrMessageEvent, req: ProviderRequest
    ):
        """Make GIFs provider-safe and inject the media digest."""

        settings = self._multimodal_settings()
        if not settings.get("mm_enabled"):
            return
        runtime = self._get_multimodal()
        try:
            await runtime.sanitize_request_images(req)
        except Exception as exc:
            logger.warning("Decision Engine 动图兼容处理失败：%s", _safe_error(exc))
        refs = event.get_extra("decision_mm_refs") or []
        try:
            # Always call in: a later text-only question must still recall the
            # media understood earlier in this same conversation.
            await runtime.inject(event, req, refs)
        except Exception as exc:
            logger.warning("Decision Engine 注入多模态结果失败：%s", _safe_error(exc))

    @filter.on_llm_request(priority=1100)
    async def life_inject_state(self, event: AstrMessageEvent, req: ProviderRequest):
        if self._life_runtime and not self._command_handler_matched(event):
            if (
                self._life_runtime.enabled
                and not event.get_extra("decision_life_proactive", False)
                and not event.get_extra("decision_life_counted", False)
            ):
                await self._life_runtime.mark_agent_active(event, None)
                event.set_extra("decision_life_counted", True)
            await self._life_runtime.inject_cached_state(event, req)

    @getattr(filter, "on_waiting_llm_request", lambda **kw: lambda fn: fn)(priority=1100)
    async def life_waiting_request(self, event: AstrMessageEvent):
        if (
            not self._life_runtime
            or not self._life_runtime.enabled
            or self._command_handler_matched(event)
        ):
            return
        if not self._life_runtime.config["schedule_settings"]["daily_schedule_generation_enable"]:
            return
        # Explicit summons keep the immediate native reply contract.
        if event.get_extra("decision_life_proactive", False) or self._direct_reply_requested(event):
            return
        at_bot, reply_bot, _, _ = self._message_address_flags(event)
        if at_bot or reply_bot:
            return
        await self._life_runtime.queue_reply_delay(event)

    @getattr(filter, "on_agent_begin", lambda **kw: lambda fn: fn)()
    async def life_agent_begin(self, event, run_context):
        if (
            self._life_runtime
            and not event.get_extra("decision_life_proactive", False)
            and not event.get_extra("decision_life_counted", False)
        ):
            await self._life_runtime.mark_agent_active(event, run_context)
            event.set_extra("decision_life_counted", True)

    @getattr(filter, "on_agent_done", lambda **kw: lambda fn: fn)()
    async def life_agent_done(self, event, run_context, response):
        if self._life_runtime and event.get_extra("decision_life_counted", False):
            event.set_extra("decision_life_counted", False)
            await self._life_runtime.defer_agent_release(event, run_context, response)

    @getattr(filter, "after_message_sent", lambda **kw: lambda fn: fn)()
    async def life_after_send(self, event):
        if self._life_runtime:
            if self._life_runtime.config["adaptive_settings"]["enable"] and getattr(event, "__is_llm_reply", False):
                persona = await self._life_runtime.personas.resolve(str(event.unified_msg_origin))
                self._life_runtime.contacts.normal_reply(event, persona.id, self._life_runtime._now())
        if self._life_runtime and not event.get_extra("decision_life_proactive", False):
            await self._life_runtime.mark_active_conversation(event)

    @filter.on_llm_response()
    async def life_track_response(self, event, response):
        if self._life_runtime:
            await self._life_runtime.temporal.guard_response(event, response)
            await self._life_runtime.track_delayed_llm_response(event, response)
            if not hasattr(filter, "on_agent_done"):
                await self.life_agent_done(event, None, response)

    @getattr(filter, "on_astrbot_loaded", lambda **kw: lambda fn: fn)()
    async def network_loaded(self) -> None:
        if self._network_enabled():
            await self._network().on_astrbot_loaded()

    def _network_enabled(self) -> bool:
        settings = (self._setting("life", {}) or {}).get("personal_network_integration", {})
        return bool(isinstance(settings, Mapping) and settings.get("enable", False))

    def _network_settings(self) -> Mapping:
        settings = (self._setting("life", {}) or {}).get("personal_network_integration", {})
        return settings if isinstance(settings, Mapping) else {}

    @filter.on_llm_request(priority=1200)
    async def network_inject_context(self, event: AstrMessageEvent, req: ProviderRequest):
        """Gate the network tools and inject relationship context.

        This runs *before* the Tool decision hook (priority 1000) so a disabled
        network tool is removed from the candidate set instead of being judged
        and possibly re-added. When enabled, the tools stay in the request and
        are judged like any other plugin tool.
        """

        settings = self._network_settings()
        enabled = bool(settings.get("enable", False)) and not self._command_handler_matched(event)
        self._hide_network_tools(
            req,
            update=not (enabled and settings.get("enable_llm_update_tool", True)),
            query=not (enabled and settings.get("enable_llm_query_tool", True)),
        )
        if enabled:
            await self._network().inject_context(event, req)

    @staticmethod
    def _hide_network_tools(req: ProviderRequest, *, update: bool, query: bool) -> None:
        tool_set = getattr(req, "func_tool", None)
        if tool_set is None or not callable(getattr(tool_set, "remove_tool", None)):
            return
        if update:
            tool_set.remove_tool("update_personal_network")
        if query:
            tool_set.remove_tool("query_personal_network")

    async def network_update_tool(
        self,
        event: AstrMessageEvent,
        characters: list[dict[str, Any]] | None = None,
        relationships: list[dict[str, Any]] | None = None,
        interactions: list[dict[str, Any]] | None = None,
    ) -> str:
        return await self._network().update_tool(
            event, characters=characters, relationships=relationships, interactions=interactions
        )

    async def network_query_tool(self, event: AstrMessageEvent, query: str = "") -> str:
        return await self._network().query_tool(event, query)

    @filter.command("关系查询", alias={"查询关系", "人际关系"})
    async def network_query_command(self, event: AstrMessageEvent, query: str = ""):
        if not self._network_enabled():
            yield event.plain_result("人际网络已关闭，请在插件 WebUI 的主动发起页面开启。")
        else:
            yield event.plain_result(await self._network().query_command_text(event, query))
        event.stop_event()

    @filter.command("生活")
    async def life_command(self, event: AstrMessageEvent, action: str = "状态"):
        if action not in {"状态", "日程", "穿搭", "立即"}:
            yield event.plain_result(
                "用法：/生活 状态｜日程｜穿搭｜立即（管理员）；其他设置请打开插件 WebUI。"
            )
            return
        if action == "立即" and getattr(event, "role", "member") != "admin":
            yield event.plain_result("只有管理员可以请求立即发起。")
            return
        runtime = await self._get_life_runtime()
        if action == "状态":
            yield event.plain_result(
                json.dumps(runtime.status(str(event.unified_msg_origin)), ensure_ascii=False)
            )
        elif action == "日程":
            async for result in runtime.show_schedule(event):
                yield result
        elif action == "穿搭":
            async for result in runtime.show_outfit(event):
                yield result
        else:
            persona, plan = await runtime._ensure_plan_for_umo(event.unified_msg_origin)
            sent, reason = await runtime._attempt_unsolicited(
                event.unified_msg_origin,
                persona,
                plan,
                "管理员请求此刻自然地发起一次聊天",
                "manual",
            )
            yield event.plain_result("已发起。" if sent else f"本次未发起：{reason}")
        event.stop_event()

    def _reserve_life_job(self, ident: str, action: str, umo: str) -> bool:
        """Register a new WebUI job, evicting finished ones first.

        Returns False when every retained job is still running, so the caller can
        answer 429 without leaving a half-registered entry behind.
        """

        while len(self._life_jobs) >= self._life_job_limit:
            removable = next(
                (key for key, value in self._life_jobs.items() if value["state"] != "running"),
                None,
            )
            if removable is None:
                return False
            self._life_jobs.pop(removable, None)
        self._life_jobs[ident] = {"state": "running", "action": action, "umo": umo}
        return True

    def _life_tasks_payload(self) -> dict[str, dict[str, Any]]:
        """Serialise recent life jobs without the embedded base64 images.

        ``/life`` is polled every two seconds while a job runs. Returning the raw
        results meant re-sending every past screenshot on each poll (tens of MB
        once a few render jobs had accumulated), so images are reported as a
        count and only text parts are kept.
        """

        payload: dict[str, dict[str, Any]] = {}
        for ident, job in self._life_jobs.items():
            entry = {key: value for key, value in job.items() if key != "result"}
            result = job.get("result")
            if isinstance(result, list):
                texts = [
                    part for part in result
                    if isinstance(part, Mapping) and part.get("type") == "text"
                ]
                images = sum(
                    1 for part in result
                    if isinstance(part, Mapping) and part.get("type") == "image"
                )
                entry["result"] = texts
                if images:
                    entry["image_count"] = images
            elif result is not None:
                entry["result"] = result
            payload[ident] = entry
        return payload

    async def page_multimodal_models(self):
        """List the providers the multi-modal understanding can use."""

        from astrbot.api.web import json_response

        from .decision.multimodal.adapters import provider_catalog

        return json_response({"models": provider_catalog(self.context)})

    async def page_life(self):
        from astrbot.api.web import json_response

        providers = []
        getter = getattr(self.context, "get_all_providers", None)
        if callable(getter):
            for provider in getter():
                try:
                    meta = provider.meta()
                    providers.append({"id": meta.id, "name": meta.id})
                except (AttributeError, RuntimeError, TypeError):
                    continue
        return json_response(
            {
                "providers": providers,
                "schema": life_schema(),
                "settings": normalize_life(dict(self._setting("life", {}) or {})),
                "status": self._life_runtime.status()
                if self._life_runtime
                else {"enabled": False, "jobs": [], "audit": [], "sessions": {}},
                "tasks": self._life_tasks_payload(),
                "aigen_available": self._get_life_media().aigen() is not None,
            }
        )

    async def page_life_personas(self):
        """List AstrBot personas for the appearance selector."""

        from astrbot.api.web import json_response

        personas = await self._persona_options()
        return json_response({"personas": personas})

    async def _persona_options(self) -> list[dict[str, str]]:
        getter = getattr(self.context.persona_manager, "get_all_personas", None)
        rows: list[dict[str, str]] = []
        if callable(getter):
            try:
                for persona in await getter():
                    persona_id = str(getattr(persona, "persona_id", "") or "").strip()
                    if persona_id:
                        rows.append({"id": persona_id, "name": persona_id})
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                logger.debug("Decision Engine could not list personas: %s", _safe_error(exc))
        if not any(row["id"] == "default" for row in rows):
            rows.append({"id": "default", "name": "default"})
        rows.sort(key=lambda row: row["id"].casefold())
        return rows

    async def page_life_sessions(self):
        """List AstrBot conversations so the WebUI can offer a searchable picker."""

        from astrbot.api.web import json_response, request

        query = str((request.query or {}).get("search", "") or "").strip().casefold()
        manager = getattr(self.context, "conversation_manager", None)
        getter = getattr(manager, "get_conversations", None)
        conversations = []
        if callable(getter):
            try:
                conversations = await getter()
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                logger.debug("Decision Engine could not list conversations: %s", _safe_error(exc))
        persona_names = {row["id"]: row["name"] for row in await self._persona_options()}
        sessions: list[dict[str, str]] = []
        seen: set[str] = set()
        for conversation in conversations or []:
            umo = str(getattr(conversation, "user_id", "") or "").strip()
            if not umo or umo in seen:
                continue
            seen.add(umo)
            persona_id = str(getattr(conversation, "persona_id", "") or "").strip()
            title = str(getattr(conversation, "title", "") or "").strip()
            platform, _, remainder = umo.partition(":")
            message_type, _, _ = remainder.partition(":")
            row = {
                "umo": umo,
                "title": title,
                "platform": platform,
                "message_type": message_type,
                "persona_id": persona_id,
                "persona_name": persona_names.get(persona_id, persona_id),
            }
            if query and query not in " ".join(row.values()).casefold():
                continue
            sessions.append(row)
            if len(sessions) >= 2000:
                break
        # AstrBot's "自定义规则" can exist before a conversation row is
        # created.  Include concrete routing entries as selectable sessions so
        # the appearance page can resolve their persona immediately.
        try:
            acm = getattr(self.context, "astrbot_config_mgr", None)
            router = getattr(acm, "ucr", None)
            routes = getattr(router, "umop_to_conf_id", {}) or {}
            conf_meta = getattr(acm, "abconf_data", {}) or {}
            for routed_umo, conf_id in routes.items():
                routed_umo = str(routed_umo).strip()
                if not routed_umo or "*" in routed_umo or routed_umo in seen:
                    continue
                parts = routed_umo.split(":", 2)
                if len(parts) != 3 or parts[1] not in {"FriendMessage", "GroupMessage"}:
                    continue
                seen.add(routed_umo)
                meta = conf_meta.get(str(conf_id), {}) if isinstance(conf_meta, dict) else {}
                title = str(meta.get("name", "") or "自定义规则")
                row = {
                    "umo": routed_umo,
                    "title": title,
                    "platform": parts[0],
                    "message_type": parts[1],
                    "persona_id": "",
                    "persona_name": "",
                }
                if not query or query in " ".join(row.values()).casefold():
                    sessions.append(row)
        except (AttributeError, TypeError, ValueError):
            pass
        sessions.sort(key=lambda row: (row["message_type"], row["umo"]))
        return json_response({"sessions": sessions})

    def _network(self):
        return self._get_personal_network()

    async def network_personas(self):
        return await self._network().api_personas()

    async def network_network(self):
        return await self._network().api_network()

    async def network_set_enabled(self):
        return await self._network().api_set_enabled()

    async def network_save_character(self):
        return await self._network().api_save_character()

    async def network_delete_character(self):
        return await self._network().api_delete_character()

    async def network_merge_characters(self):
        return await self._network().api_merge_characters()

    async def network_save_relationship(self):
        return await self._network().api_save_relationship()

    async def network_delete_relationship(self):
        return await self._network().api_delete_relationship()

    async def network_save_life_event(self):
        return await self._network().api_save_life_event()

    async def network_delete_life_event(self):
        return await self._network().api_delete_life_event()

    async def network_generate(self):
        return await self._network().api_generate_network()

    async def network_generation_status(self):
        return await self._network().api_generation_status()

    async def network_generation_validate(self):
        return await self._network().api_validate_generation()

    async def network_generation_apply(self):
        return await self._network().api_apply_generation()

    async def network_avatar(self, character_id: str):
        return await self._network().api_avatar(character_id)

    async def network_export(self):
        return await self._network().api_export()

    async def network_import_preview(self, persona_key: str):
        return await self._network().api_import_preview(persona_key)

    async def network_import_apply(self):
        return await self._network().api_import_apply()

    async def page_life_billing(self):
        """List the platforms and aigen billing accounts for the pickers.

        Platforms come from AstrBot's own configuration; accounts come from
        aigen's billing ledger, which is the authoritative list of identities
        that can actually be charged. Both reads are best-effort so a missing
        aigen install never breaks the settings page.
        """

        from astrbot.api.web import json_response

        platforms: list[dict[str, str]] = []
        try:
            configured = self.context.get_config().get("platform", []) or []
        except (AttributeError, TypeError, ValueError):
            configured = []
        seen: set[str] = set()
        for row in configured:
            if not isinstance(row, Mapping):
                continue
            # aigen keys the account by the adapter *type*, not the instance id.
            name = str(row.get("type") or row.get("id") or "").strip()
            if name and name not in seen:
                seen.add(name)
                platforms.append({"id": name, "name": name})
        platforms.sort(key=lambda item: item["id"].casefold())

        accounts = await self._aigen_billing_accounts()
        # Keep the runtime fallback in sync with what the picker shows, so an
        # unset account can never disagree with the visible list.
        try:
            self._get_life_media().refresh_billing_accounts(accounts)
        except Exception as exc:
            logger.debug("Decision Engine 同步计费账户失败：%s", _safe_error(exc))
        return json_response({"platforms": platforms, "accounts": accounts})

    async def _aigen_billing_accounts(self) -> list[dict[str, Any]]:
        """Read aigen's balance rows without importing or mutating its store."""

        getter = getattr(self.context, "get_all_stars", None)
        if not callable(getter):
            return []
        try:
            stars = list(getter())
        except (AttributeError, TypeError, ValueError):
            return []
        directory = None
        for star in stars:
            if getattr(star, "name", "") != "astrbot_plugin_aigen":
                continue
            root = str(getattr(star, "root_dir_name", "") or "")
            module = getattr(star, "module_path", "") or ""
            if root or module:
                directory = root or Path(module).parent.name
            break
        if not directory:
            return []
        # <data>/plugin_data/<plugin>/billing.sqlite3
        db_path = (
            Path(get_astrbot_config_path()).parent
            / "plugin_data"
            / directory
            / "billing.sqlite3"
        )
        if not db_path.is_file():
            return []

        def read() -> list[dict[str, Any]]:
            import sqlite3

            connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
            try:
                rows = connection.execute(
                    "SELECT owner, amount FROM balances ORDER BY amount DESC"
                ).fetchall()
            finally:
                connection.close()
            result = []
            for owner, amount in rows:
                try:
                    platform, user_id = json.loads(owner)
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
                platform, user_id = str(platform), str(user_id)
                if not platform or not user_id:
                    continue
                result.append(
                    {
                        "platform": platform,
                        "user_id": user_id,
                        "amount": int(amount or 0),
                        "label": f"{platform} · {user_id}",
                    }
                )
            return result

        try:
            return await asyncio.to_thread(read)
        except Exception as exc:
            logger.debug("Decision Engine 读取 aigen 记账账户失败：%s", _safe_error(exc))
            return []

    async def page_life_appearance(self):
        from astrbot.api.web import error_response, json_response, request

        payload = await request.json(default={})
        try:
            if not isinstance(payload, Mapping):
                raise ValueError("请求必须是对象")
            runtime = await self._get_life_runtime()
            persona_id = str(payload.get("persona_id", "") or "").strip()
            if not persona_id:
                # Legacy callers selected a conversation; resolve its persona.
                umo = validate_umo(str(payload.get("umo", "")))
                from astrbot.core.platform.message_session import MessageSession

                MessageSession.from_str(umo)
                persona_id = (await runtime.personas.resolve(umo)).id
            else:
                persona_id = (await runtime.personas.resolve_id(persona_id)).id
            store = self._get_life_media().appearance
            async with self._life_asset_lock:
                if payload.get("action") == "upload":
                    await store.upload(
                        persona_id, payload.get("name", "reference"), payload.get("data_url", "")
                    )
                elif payload.get("action") == "set_description":
                    await store.set_description(persona_id, str(payload.get("description", "")))
                elif payload.get("action") == "delete":
                    await store.remove(persona_id, str(payload.get("id", "")))
                elif payload.get("action", "list") != "list":
                    raise ValueError("未知外貌操作")
                return json_response(
                    {
                        "persona_id": persona_id,
                        "images": await store.list(persona_id, previews=True),
                        "description": await store.get_description(persona_id)
                        or str(
                            ((self._setting("life", {}) or {}).get("appearance_settings", {}) or {}).get(
                                "description", ""
                            )
                        ),
                    }
                )
        except (ValueError, OSError, TypeError) as exc:
            return error_response(_safe_error(exc), status_code=400)

    async def page_life_control(self):
        from astrbot.api.web import error_response, json_response, request

        from .decision.life_actions import perform_control_action

        payload = await request.json(default={})
        if not isinstance(payload, Mapping):
            return error_response("请求必须是对象", status_code=400)
        if not isinstance(payload.get("action"), str) or payload["action"] not in {"create_task", "edit_task", "cancel_task", "resolve_topic", "dismiss_idea"}:
            return error_response("未知会话控制操作", status_code=400)
        try:
            umo = validate_umo(str(payload.get("umo", "")))
            runtime = await self._get_life_runtime()
            return json_response(await perform_control_action(runtime, umo, payload))
        except (ValueError, TypeError) as exc:
            return error_response(_safe_error(exc), status_code=400)

    async def page_life_action(self):
        from astrbot.api.web import error_response, json_response, request

        payload = await request.json(default={})
        if not isinstance(payload, Mapping):
            return error_response("请求必须是对象", status_code=400)
        allowed = {
            "inspect",
            "generate",
            "rewrite_schedule",
            "rewrite_outfit",
            "add_requirement",
            "generate_timeline",
            "regenerate_timeline",
            "import_timeline",
            "approve_timeline",
            "reject_timeline",
            "settle_delay",
            "initiate",
            "render_schedule",
            "render_outfit",
            "render_timeline",
            "render_stage",
        }
        action, umo = str(payload.get("action", "")), str(payload.get("umo", ""))
        if action not in allowed:
            return error_response("未知生活操作", status_code=400)
        try:
            from astrbot.core.platform.message_session import MessageSession

            validate_umo(umo)
            MessageSession.from_str(umo)
            runtime = await self._get_life_runtime()
        except (ValueError, RuntimeError) as exc:
            return error_response(_safe_error(exc), status_code=400)
        import uuid

        ident = uuid.uuid4().hex
        if not self._reserve_life_job(ident, action, umo):
            return error_response("已有太多后台操作，请稍后再试", status_code=429)

        async def run():
            try:
                from .decision.life_actions import perform_life_action

                # The inspect that follows a mutation must run under the same
                # lock, otherwise a concurrent action can write a plan between
                # the mutation and its read-back and the UI shows stale state.
                async with runtime.action_lock:
                    value = await perform_life_action(
                        runtime,
                        umo,
                        action,
                        str(payload.get("text", ""))[:50000],
                        str(payload.get("date_range", "")),
                    )
                    inspection = None
                    if action in INSPECT_AFTER_ACTIONS:
                        inspection = await perform_life_action(runtime, umo, "inspect")
                job = self._life_jobs.get(ident)
                if job is not None:
                    job.update(state="done", result=value)
                    if inspection is not None:
                        job["inspection"] = inspection
            except asyncio.CancelledError:
                job = self._life_jobs.get(ident)
                if job is not None:
                    job.update(state="cancelled")
                raise
            except Exception as exc:
                job = self._life_jobs.get(ident)
                if job is not None:
                    job.update(state="error", error=_safe_error(exc))

        task = runtime.spawn(run())
        if isinstance(task, asyncio.Task):
            self._life_tasks.add(task)
            task.add_done_callback(self._life_tasks.discard)
        return json_response({"task_id": ident})

    async def page_settings(self):
        from astrbot.api.web import json_response

        always_keep = [str(item) for item in (self._setting("always_keep_tools", []) or [])]
        if not self._bool("always_keep_tools_customized", False):
            always_keep.extend(sorted(self._builtin_tool_names()))
            always_keep.append(DECISION_TOOL_NAME)
        always_keep = list(dict.fromkeys(always_keep))
        always_keep_recommend = [
            str(item)
            for item in (self._setting("always_keep_recommend_tools", []) or [])
            if str(item) in always_keep
        ]
        always_keep_subagents = [
            str(item) for item in (self._setting("always_keep_subagents", []) or [])
        ]
        always_keep_recommend_subagents = [
            str(item)
            for item in (self._setting("always_keep_recommend_subagents", []) or [])
            if str(item) in always_keep_subagents
        ]
        always_keep_mcp = [str(item) for item in (self._setting("always_keep_mcp", []) or [])]
        always_keep_recommend_mcp = [
            str(item)
            for item in (self._setting("always_keep_recommend_mcp", []) or [])
            if str(item) in always_keep_mcp
        ]
        always_keep_skills = [str(item) for item in (self._setting("always_keep_skills", []) or [])]
        always_keep_recommend_skills = [
            str(item)
            for item in (self._setting("always_keep_recommend_skills", []) or [])
            if str(item) in always_keep_skills
        ]
        proactive = {}
        for key, default in PROACTIVE_PAGE_DEFAULTS.items():
            value = self._setting(key, default)
            if isinstance(default, list):
                value = [str(item) for item in value] if isinstance(value, (list, tuple)) else []
            elif isinstance(default, bool):
                value = self._bool(key, default)
            elif isinstance(default, int):
                value = self._int(key, default)
            elif isinstance(default, float):
                value = self._float(key, default)
            else:
                value = str(value if value is not None else default)
            proactive[key] = value
        return json_response(
            {
                "always_keep_tools": always_keep,
                "message_interjection_enabled": self._bool("message_interjection_enabled", True),
                "life": normalize_life(dict(self._setting("life", {}) or {})),
                "always_keep_recommend_tools": always_keep_recommend,
                "always_keep_subagents": always_keep_subagents,
                "always_keep_recommend_subagents": always_keep_recommend_subagents,
                "always_keep_mcp": always_keep_mcp,
                "always_keep_recommend_mcp": always_keep_recommend_mcp,
                "always_keep_skills": always_keep_skills,
                "always_keep_recommend_skills": always_keep_recommend_skills,
                "jev_pre_prompt": self._policy(),
                "main_llm_post_prompt": self._main_llm_post_prompt(),
                "tool_filter_enabled": self._bool("tool_filter_enabled", True),
                "mcp_filter_enabled": self._bool("mcp_filter_enabled", True),
                "skill_filter_enabled": self._bool("skill_filter_enabled", True),
                "subagent_filter_enabled": self._bool("subagent_filter_enabled", False),
                "tool_decision_enabled": self._bool("tool_decision_enabled", True),
                "mcp_decision_enabled": self._bool("mcp_decision_enabled", True),
                "skill_decision_enabled": self._bool("skill_decision_enabled", True),
                "subagent_decision_enabled": self._bool("subagent_decision_enabled", True),
                "tools_scope": self._configured_scope_values("tools_scope"),
                "tools_scope_blacklist": self._bool("tools_scope_blacklist", True),
                "tool_noul_threshold": self._float(
                    "tool_noul_threshold", DEFAULT_TOOL_NOUL_THRESHOLD
                ),
                "tools_subagents_decision_enabled": self._bool(
                    "tools_subagents_decision_enabled", True
                ),
                "proactive_reply_enabled": self._bool("proactive_reply_enabled", False),
                "conversation_flow_enabled": self._bool("conversation_flow_enabled", True),
                "proactive": proactive,
                "dialogue_enhancement": self._output_page_settings(),
            }
        )

    async def page_save_settings(self):
        from astrbot.api.web import error_response, json_response, request

        payload = await request.json(default={})
        if not isinstance(payload, Mapping):
            return error_response("request body must be an object", status_code=400)
        life_values = None
        if "life" in payload:
            try:
                life_values = normalize_life(payload["life"])
            except (ValueError, TypeError) as exc:
                return error_response(str(exc), status_code=400)
        interjection = payload.get("message_interjection_enabled")
        if interjection is not None and not isinstance(interjection, bool):
            return error_response("消息接话开关必须是布尔值", status_code=400)
        values = payload.get("always_keep_tools")
        recommendation_values = payload.get("always_keep_recommend_tools", [])
        mcp_values = payload.get("always_keep_mcp", [])
        mcp_recommendation_values = payload.get("always_keep_recommend_mcp", [])
        skill_values = payload.get("always_keep_skills", [])
        skill_recommendation_values = payload.get("always_keep_recommend_skills", [])
        subagent_values = payload.get("always_keep_subagents", [])
        subagent_recommendation_values = payload.get("always_keep_recommend_subagents", [])
        if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
            return error_response("always_keep_tools must be a string list", status_code=400)
        if not isinstance(recommendation_values, list) or any(
            not isinstance(item, str) for item in recommendation_values
        ):
            return error_response(
                "always_keep_recommend_tools must be a string list", status_code=400
            )
        if not isinstance(mcp_values, list) or any(
            not isinstance(item, str) for item in mcp_values
        ):
            return error_response("always_keep_mcp must be a string list", status_code=400)
        if not isinstance(mcp_recommendation_values, list) or any(
            not isinstance(item, str) for item in mcp_recommendation_values
        ):
            return error_response(
                "always_keep_recommend_mcp must be a string list", status_code=400
            )
        if not isinstance(skill_values, list) or any(
            not isinstance(item, str) for item in skill_values
        ):
            return error_response("always_keep_skills must be a string list", status_code=400)
        if not isinstance(skill_recommendation_values, list) or any(
            not isinstance(item, str) for item in skill_recommendation_values
        ):
            return error_response(
                "always_keep_recommend_skills must be a string list", status_code=400
            )
        if not isinstance(subagent_values, list) or any(
            not isinstance(item, str) for item in subagent_values
        ):
            return error_response("always_keep_subagents must be a string list", status_code=400)
        if not isinstance(subagent_recommendation_values, list) or any(
            not isinstance(item, str) for item in subagent_recommendation_values
        ):
            return error_response(
                "always_keep_recommend_subagents must be a string list", status_code=400
            )
        jev_pre_prompt = payload.get("jev_pre_prompt")
        main_llm_post_prompt = payload.get("main_llm_post_prompt")
        tool_filter_enabled = payload.get("tool_filter_enabled")
        mcp_filter_enabled = payload.get("mcp_filter_enabled")
        skill_filter_enabled = payload.get("skill_filter_enabled")
        subagent_filter_enabled = payload.get("subagent_filter_enabled")
        tool_decision_enabled = payload.get("tool_decision_enabled")
        mcp_decision_enabled = payload.get("mcp_decision_enabled")
        skill_decision_enabled = payload.get("skill_decision_enabled")
        subagent_decision_enabled = payload.get("subagent_decision_enabled")
        tools_scope = payload.get("tools_scope")
        tools_scope_blacklist = payload.get("tools_scope_blacklist")
        tool_noul_threshold = payload.get("tool_noul_threshold")
        if jev_pre_prompt is not None and not isinstance(jev_pre_prompt, str):
            return error_response("jev_pre_prompt must be a string", status_code=400)
        if main_llm_post_prompt is not None and not isinstance(main_llm_post_prompt, str):
            return error_response("main_llm_post_prompt must be a string", status_code=400)
        if tool_filter_enabled is not None and not isinstance(tool_filter_enabled, bool):
            return error_response("tool_filter_enabled must be a boolean", status_code=400)
        if mcp_filter_enabled is not None and not isinstance(mcp_filter_enabled, bool):
            return error_response("mcp_filter_enabled must be a boolean", status_code=400)
        if skill_filter_enabled is not None and not isinstance(skill_filter_enabled, bool):
            return error_response("skill_filter_enabled must be a boolean", status_code=400)
        if subagent_filter_enabled is not None and not isinstance(subagent_filter_enabled, bool):
            return error_response("subagent_filter_enabled must be a boolean", status_code=400)
        if tool_decision_enabled is not None and not isinstance(tool_decision_enabled, bool):
            return error_response("tool_decision_enabled must be a boolean", status_code=400)
        if mcp_decision_enabled is not None and not isinstance(mcp_decision_enabled, bool):
            return error_response("mcp_decision_enabled must be a boolean", status_code=400)
        if skill_decision_enabled is not None and not isinstance(skill_decision_enabled, bool):
            return error_response("skill_decision_enabled must be a boolean", status_code=400)
        if subagent_decision_enabled is not None and not isinstance(
            subagent_decision_enabled, bool
        ):
            return error_response("subagent_decision_enabled must be a boolean", status_code=400)
        if tools_scope is not None and (
            not isinstance(tools_scope, list)
            or any(not isinstance(item, str) for item in tools_scope)
        ):
            return error_response("tools_scope must be a string list", status_code=400)
        if tools_scope_blacklist is not None and not isinstance(tools_scope_blacklist, bool):
            return error_response("tools_scope_blacklist must be a boolean", status_code=400)
        if tool_noul_threshold is not None and (
            isinstance(tool_noul_threshold, bool)
            or not isinstance(tool_noul_threshold, (int, float))
            or not math.isfinite(float(tool_noul_threshold))
        ):
            return error_response("tool_noul_threshold must be a number", status_code=400)
        proactive_payload = payload.get("proactive", {})
        if not isinstance(proactive_payload, Mapping):
            return error_response("proactive must be an object", status_code=400)
        dialogue_payload = payload.get("dialogue_enhancement", {})
        if not isinstance(dialogue_payload, Mapping):
            return error_response("dialogue_enhancement must be an object", status_code=400)
        proactive_payload = dict(proactive_payload)
        for key in (
            "conversation_flow_analysis_enabled",
            "conversation_flow_window",
            "conversation_flow_scope",
            "conversation_flow_scope_blacklist",
        ):
            if key in dialogue_payload:
                proactive_payload[key] = dialogue_payload[key]
        pending_multimodal: dict[str, Any] | None = None
        multimodal_payload = dialogue_payload.get("multimodal")
        if multimodal_payload is not None:
            if not isinstance(multimodal_payload, Mapping):
                return error_response("multimodal must be an object", status_code=400)
            from .decision.multimodal import validate_page_payload

            cleaned_mm, mm_error = validate_page_payload(multimodal_payload)
            if mm_error:
                return error_response(mm_error, status_code=400)
            if cleaned_mm:
                # Staged; the whole config is written once validation succeeds.
                pending_multimodal = {**self._multimodal_settings(), **cleaned_mm}
        output_enabled = dialogue_payload.get("output_enhancement_enabled")
        output_scope = dialogue_payload.get("output_scope")
        output_scope_blacklist = dialogue_payload.get("output_scope_blacklist")
        output_pipeline = dialogue_payload.get("output_pipeline")
        if output_enabled is not None and not isinstance(output_enabled, bool):
            return error_response("output_enhancement_enabled must be a boolean", status_code=400)
        if output_scope is not None and (
            not isinstance(output_scope, list)
            or any(not isinstance(item, str) for item in output_scope)
        ):
            return error_response("output_scope must be a string list", status_code=400)
        if output_scope_blacklist is not None and not isinstance(output_scope_blacklist, bool):
            return error_response("output_scope_blacklist must be a boolean", status_code=400)
        if output_pipeline is not None and not isinstance(output_pipeline, Mapping):
            return error_response("output_pipeline must be an object", status_code=400)
        if output_pipeline is not None:
            try:
                if len(json.dumps(output_pipeline, ensure_ascii=False)) > 300_000:
                    return error_response("output_pipeline is too large", status_code=400)
            except (TypeError, ValueError):
                return error_response("output_pipeline must be JSON serializable", status_code=400)
        cleaned_proactive: dict[str, Any] = {}
        for key, default in PROACTIVE_PAGE_DEFAULTS.items():
            if key not in proactive_payload:
                continue
            value = proactive_payload[key]
            if isinstance(default, list):
                if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
                    return error_response(f"{key} must be a string list", status_code=400)
                cleaned_proactive[key] = list(
                    dict.fromkeys(item.strip() for item in value if item.strip())
                )[:500]
            elif isinstance(default, bool):
                if not isinstance(value, bool):
                    return error_response(f"{key} must be a boolean", status_code=400)
                cleaned_proactive[key] = value
            elif isinstance(default, int):
                if isinstance(value, bool) or not isinstance(value, int):
                    return error_response(f"{key} must be an integer", status_code=400)
                if key == "conversation_flow_window":
                    value = min(30, max(2, value))
                cleaned_proactive[key] = value
            elif isinstance(default, float):
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    return error_response(f"{key} must be a number", status_code=400)
                cleaned_proactive[key] = float(value)
            else:
                if not isinstance(value, str):
                    return error_response(f"{key} must be a string", status_code=400)
                cleaned_proactive[key] = value[:1000]
        available = {
            str(getattr(tool, "name", ""))
            for tool in self._manager_tool_objects()
            if getattr(tool, "name", None) and not is_handoff_tool(tool) and not is_mcp_tool(tool)
        }
        available.add(DECISION_TOOL_NAME)
        available.update(self._builtin_tool_names())
        cleaned = list(
            dict.fromkeys(
                item.strip() for item in values if item.strip() and item.strip() in available
            )
        )[:500]
        cleaned_recommend = list(
            dict.fromkeys(
                item.strip()
                for item in recommendation_values
                if item.strip() in cleaned and item.strip() in available
            )
        )[:500]
        available_mcp = {
            str(getattr(tool, "name", ""))
            for tool in self._manager_tool_objects()
            if getattr(tool, "name", None) and is_mcp_tool(tool)
        }
        cleaned_mcp = list(
            dict.fromkeys(
                item.strip()
                for item in mcp_values
                if item.strip() and item.strip() in available_mcp
            )
        )[:500]
        cleaned_recommend_mcp = list(
            dict.fromkeys(
                item.strip() for item in mcp_recommendation_values if item.strip() in cleaned_mcp
            )
        )[:500]
        available_skills = {
            str(getattr(skill, "name", ""))
            for skill in self._skill_objects()
            if getattr(skill, "name", None)
        }
        cleaned_skills = list(
            dict.fromkeys(
                item.strip()
                for item in skill_values
                if item.strip() and item.strip() in available_skills
            )
        )[:500]
        cleaned_recommend_skills = list(
            dict.fromkeys(
                item.strip()
                for item in skill_recommendation_values
                if item.strip() in cleaned_skills
            )
        )[:500]
        available_subagents = {
            str(getattr(tool, "name", "")) for tool in self._subagent_tool_objects()
        }
        cleaned_subagents = list(
            dict.fromkeys(
                item.strip()
                for item in subagent_values
                if item.strip() and item.strip() in available_subagents
            )
        )[:500]
        cleaned_recommend_subagents = list(
            dict.fromkeys(
                item.strip()
                for item in subagent_recommendation_values
                if item.strip() in cleaned_subagents
            )
        )[:500]
        detail_values = {
            key: self._setting(key, default) for key, default in DETAIL_DEFAULTS.items()
        }
        if life_values is not None:
            detail_values["life"] = life_values
        if interjection is not None:
            detail_values["message_interjection_enabled"] = interjection
        detail_values.update(
            always_keep_tools=cleaned,
            always_keep_recommend_tools=cleaned_recommend,
            always_keep_mcp=cleaned_mcp,
            always_keep_recommend_mcp=cleaned_recommend_mcp,
            always_keep_skills=cleaned_skills,
            always_keep_recommend_skills=cleaned_recommend_skills,
            always_keep_subagents=cleaned_subagents,
            always_keep_recommend_subagents=cleaned_recommend_subagents,
            always_keep_tools_customized=True,
        )
        if tool_filter_enabled is not None:
            detail_values["tool_filter_enabled"] = tool_filter_enabled
        if mcp_filter_enabled is not None:
            detail_values["mcp_filter_enabled"] = mcp_filter_enabled
        if skill_filter_enabled is not None:
            detail_values["skill_filter_enabled"] = skill_filter_enabled
        if subagent_filter_enabled is not None:
            detail_values["subagent_filter_enabled"] = subagent_filter_enabled
        if tool_decision_enabled is not None:
            detail_values["tool_decision_enabled"] = tool_decision_enabled
        if mcp_decision_enabled is not None:
            detail_values["mcp_decision_enabled"] = mcp_decision_enabled
        if skill_decision_enabled is not None:
            detail_values["skill_decision_enabled"] = skill_decision_enabled
        if subagent_decision_enabled is not None:
            detail_values["subagent_decision_enabled"] = subagent_decision_enabled
        if tools_scope is not None:
            detail_values["tools_scope"] = list(
                dict.fromkeys(item.strip() for item in tools_scope if item.strip())
            )[:500]
        if tools_scope_blacklist is not None:
            detail_values["tools_scope_blacklist"] = tools_scope_blacklist
        if jev_pre_prompt is not None:
            detail_values["jev_pre_prompt"] = jev_pre_prompt[:50000]
        if main_llm_post_prompt is not None:
            detail_values["main_llm_post_prompt"] = main_llm_post_prompt[:50000]
        if tool_noul_threshold is not None:
            detail_values["tool_noul_threshold"] = min(1.0, max(0.0, float(tool_noul_threshold)))
        detail_values.update(cleaned_proactive)
        if output_enabled is not None:
            detail_values["output_enhancement_enabled"] = output_enabled
        if output_scope is not None:
            detail_values["output_scope"] = list(
                dict.fromkeys(item.strip() for item in output_scope if item.strip())
            )[:500]
        if output_scope_blacklist is not None:
            detail_values["output_scope_blacklist"] = output_scope_blacklist
        from .decision.outputpro_runtime import _deep_merge, output_config_defaults

        if output_pipeline is not None:
            detail_values["output_pipeline"] = _deep_merge(
                output_config_defaults(), dict(output_pipeline)
            )
        # The output pipeline has one built-in execution order.  Keep the
        # legacy field for schema compatibility, but never persist a
        # user-editable false value, including when the current save payload
        # does not include output settings.
        configured_output = detail_values.get("output_pipeline")
        normalized_output = _deep_merge(
            output_config_defaults(),
            dict(configured_output) if isinstance(configured_output, Mapping) else {},
        )
        output_pipeline_node = normalized_output.get("pipeline")
        if not isinstance(output_pipeline_node, dict):
            output_pipeline_node = {}
            normalized_output["pipeline"] = output_pipeline_node
        output_pipeline_node["lock_order"] = True
        detail_values["output_pipeline"] = normalized_output
        if pending_multimodal is not None:
            detail_values["multimodal"] = pending_multimodal
        # Commit first: a failed disk write must not change the running policy
        # or prune proactive history while reporting a failed save to the UI.
        try:
            self._save_detail_settings(detail_values)
        except OSError as exc:
            logger.error("Decision Engine WebUI settings save failed: %s", _safe_error(exc))
            return error_response("配置写入失败，原设置保持不变", status_code=500)
        self.proactive.reconfigure(
            self._int("proactive_history_max_messages", 12),
            self._int("proactive_max_sessions", 500),
        )
        if pending_multimodal is not None and self._multimodal is not None:
            self._multimodal.refresh(self._multimodal_settings())
            self._sync_multimodal_tools(self._multimodal)
            try:
                await self._apply_multimodal_tool_state(self._multimodal_settings())
            except Exception as exc:  # pragma: no cover - host API differences
                logger.warning(
                    "Decision Engine 多模态工具状态刷新失败：%s", _safe_error(exc)
                )
        await self._configure_output_runtime()
        if life_values is not None:
            try:
                await self._configure_life_runtime()
            except Exception as exc:
                logger.error("Decision Engine 生活设置已保存，后台启动失败：%s", _safe_error(exc))
                return json_response({"saved": True, "runtime_warning": _safe_error(exc)})
        return json_response(
            {
                "saved": True,
                "always_keep_tools": cleaned,
                "always_keep_recommend_tools": cleaned_recommend,
                "always_keep_mcp": cleaned_mcp,
                "always_keep_recommend_mcp": cleaned_recommend_mcp,
                "always_keep_skills": cleaned_skills,
                "always_keep_recommend_skills": cleaned_recommend_skills,
                "always_keep_subagents": cleaned_subagents,
                "always_keep_recommend_subagents": cleaned_recommend_subagents,
                "tool_filter_enabled": self._bool("tool_filter_enabled", True),
                "mcp_filter_enabled": self._bool("mcp_filter_enabled", True),
                "skill_filter_enabled": self._bool("skill_filter_enabled", True),
                "subagent_filter_enabled": self._bool("subagent_filter_enabled", False),
                "tool_decision_enabled": self._bool("tool_decision_enabled", True),
                "mcp_decision_enabled": self._bool("mcp_decision_enabled", True),
                "skill_decision_enabled": self._bool("skill_decision_enabled", True),
                "subagent_decision_enabled": self._bool("subagent_decision_enabled", True),
                "tools_scope": self._configured_scope_values("tools_scope"),
                "tools_scope_blacklist": self._bool("tools_scope_blacklist", True),
                "tool_noul_threshold": self._float(
                    "tool_noul_threshold", DEFAULT_TOOL_NOUL_THRESHOLD
                ),
                "proactive": {
                    key: self._setting(key, default)
                    for key, default in PROACTIVE_PAGE_DEFAULTS.items()
                },
                "dialogue_enhancement": self._output_page_settings(),
            }
        )


def _safe_error(error: BaseException) -> str:
    text = str(error).replace("\n", " ").strip()
    return text[:240] or error.__class__.__name__
