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
from .decision.tokens import estimate_request_tokens, fit_request_state

PLUGIN_NAME = "astrbot_plugin_decision"
PLUGIN_DISPLAY_NAME = "jev决策综合插件"
_ON_DECORATING_RESULT = getattr(filter, "on_decorating_result", None)
DEFAULT_TOOL_NOUL_THRESHOLD = 0.65
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
    "always_keep_tools": [],
    "always_keep_recommend_tools": [],
    "always_keep_subagents": [],
    "always_keep_recommend_subagents": [],
    "always_keep_mcp": [],
    "always_keep_recommend_mcp": [],
    "always_keep_tools_customized": False,
    "tool_filter_enabled": True,
    "subagent_filter_enabled": False,
    "mcp_filter_enabled": True,
    # Per-category Jev decision switches.  Filtering and judging are
    # deliberately independent: a category can be judged only to produce
    # recommendations while remaining fully visible to the main LLM, or can
    # be filtered from the manually kept set without making a Jev request.
    "tool_decision_enabled": True,
    "subagent_decision_enabled": True,
    "mcp_decision_enabled": True,
    # One shared scope gates the complete Tools/MCP/SubAgent decision module.
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


@register(
    PLUGIN_NAME,
    "yiyinfaith",
    "jev决策综合插件：用 Jev 判断 Tools、MCP、SubAgents 和主动对话，并保留 AstrBot 主 LLM 的最终控制权。",
    "0.1.0",
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
        self.proactive = ProactiveState(
            self._int("proactive_history_max_messages", 12),
            self._int("proactive_max_sessions", 500),
        )
        self._last_call_latency_ms: float | None = None
        self._call_count = 0
        self._failure_count = 0

        self._decision_tool = FunctionTool(
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
            return (
                {key: value[key] for key in DETAIL_DEFAULTS if key in value}
                if isinstance(value, Mapping)
                else {}
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            logger.warning(
                "Decision Engine could not read WebUI settings; defaults will be used: %s",
                _safe_error(exc),
            )
            return {}

    def _save_detail_settings(self, values: Mapping[str, Any]) -> None:
        payload = {key: values[key] for key in DETAIL_DEFAULTS if key in values}
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
        if not self.provider.api_key:
            logger.warning(
                "Decision Engine API key is empty; tool filtering and proactive decisions are disabled."
            )

    async def _configure_output_runtime(self) -> None:
        """Enable the vendored output pipeline only when explicitly selected."""

        enabled = (
            self._bool("conversation_flow_enabled", True)
            and self._bool("output_enhancement_enabled", False)
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
        if self.provider is not None:
            await self.provider.close()
        self.provider = None
        self._ready = False
        if self._output_runtime is not None:
            await self._output_runtime.close()
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
                str(item)
                for item in (self._setting("output_scope", []) or [])
                if str(item).strip()
            ],
            "output_scope_blacklist": self._bool("output_scope_blacklist", True),
            "output_pipeline": merged,
        }

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

    def _tool_origin(self, tool: Any, *, builtin: bool = False) -> str:
        """Resolve the human-readable owner shown in the settings page."""

        if getattr(tool, "name", None) == DECISION_TOOL_NAME:
            return PLUGIN_DISPLAY_NAME
        if builtin or self._is_builtin_tool(tool):
            return "Astrbot内置工具"
        mcp_server = str(getattr(tool, "mcp_server_name", "") or "").strip()
        if mcp_server:
            return f"MCP · {mcp_server}"
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
            return module_path
        if is_handoff_tool(tool):
            return "AstrBot SubAgent"
        return "未知来源"

    def _state_for_request(
        self,
        req: ProviderRequest,
        tools: list[Any],
        mcp_tools: list[Any],
        handoffs: list[Any],
    ) -> str:
        return build_decision_state(
            policy=self._policy(),
            current_prompt=str(req.prompt or ""),
            contexts=req.contexts,
            tools=[tool_summary(tool) for tool in tools],
            mcp_tools=[tool_summary(tool) for tool in mcp_tools],
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
        handoffs: list[Any],
        questions: Mapping[str, Mapping[str, Any]],
        question_to_tool: Mapping[str, str],
        question_to_mcp: Mapping[str, str],
        question_to_handoff: Mapping[str, str],
    ) -> list[tuple[str, dict[str, dict[str, Any]]]]:
        """Split candidate judgments only when the complete Jev request is oversized."""

        full_state = self._state_for_request(req, ordinary, mcp_tools, handoffs)
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
            bucket_handoffs = [tool for kind, tool, _ in bucket if kind == "handoff"]
            bucket_questions = {qid: dict(questions[qid]) for _, _, qid in bucket}
            return (
                self._state_for_request(req, bucket_tools, bucket_mcp, bucket_handoffs),
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
            "Decision Engine 共筛选出如下工具：Tools=%s；MCP=%s；SubAgents=%s；推荐 Tools=%s；推荐 MCP=%s；推荐 SubAgents=%s",
            ", ".join(selected_tools) or "无",
            ", ".join(selected_mcp) or "无",
            ", ".join(selected_subagents) or "无",
            ", ".join(outcome.recommended_tools) or "无",
            ", ".join(getattr(outcome, "recommended_mcp", [])) or "无",
            ", ".join(getattr(outcome, "recommended_subagents", [])) or "无",
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
        subagent_filter_enabled = self._bool("subagent_filter_enabled", False)
        tool_decision_enabled = self._bool("tool_decision_enabled", True)
        mcp_decision_enabled = self._bool("mcp_decision_enabled", True)
        subagent_decision_enabled = self._bool("subagent_decision_enabled", True)
        original = list(tool_set.tools)
        if not original:
            return
        handoffs = [
            tool for tool in original if is_handoff_tool(tool) and tool_is_active(tool)
        ]
        mcp_tools = [
            tool
            for tool in original
            if is_mcp_tool(tool) and tool_is_active(tool)
        ]
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
        # Before MCP had its own WebUI list, MCP tools were accepted by the
        # ordinary Tools list.  Keep those selections effective until the user
        # next saves the new three-section settings page.
        configured_tool_keeps = {
            str(name).strip()
            for name in (self._setting("always_keep_tools", []) or [])
            if str(name).strip()
        }
        always_keep_mcp.update(
            name for name in configured_tool_keeps if any(
                str(getattr(tool, "name", "")) == name for tool in mcp_tools
            )
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
        decision_handoffs = handoffs if subagent_decision_enabled else []
        question_to_tool: dict[str, str] = {}
        question_to_mcp: dict[str, str] = {}
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
                filter_handoffs=subagent_filter_enabled,
            )
            add_routing_hint(
                req,
                template=self._main_llm_post_prompt(),
                recommended_tools=outcome.recommended_tools,
                recommended_mcp=outcome.recommended_mcp,
                recommended_subagents=outcome.recommended_subagents,
            )
            req.func_tool.tools = outcome.selected
            self._log_routing_outcome(outcome)
            return

        try:
            batches = self._decision_batches(
                req,
                decision_ordinary,
                decision_mcp,
                decision_handoffs,
                questions,
                question_to_tool,
                question_to_mcp,
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
            add_routing_hint(
                req,
                template=self._main_llm_post_prompt(),
                recommended_tools=manual_recommended_tools,
                recommended_mcp=manual_recommended_mcp,
                recommended_subagents=manual_recommended_subagents,
            )
            return

        noul_answers = {
            qid: answer
            for result in results
            for qid, answer in result.answers.items()
            if qid in question_to_tool or qid in question_to_mcp or qid in subagent_question_to_name
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
            recommended_subagents=recommended_subagents,
        )
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
                        getattr(component, "qq", None)
                        or getattr(component, "target", None)
                        or ""
                    ).strip()
                    if target_id.lower() == "all":
                        continue
                    if target_id:
                        target_name = str(
                            getattr(component, "name", None) or target_id
                        ).strip()
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
        values = {
            str(item).strip()
            for item in configured
            if str(item).strip()
        } if isinstance(configured, (list, tuple, set)) else set()
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
        """Apply the shared Tools/MCP/SubAgent decision black/white list."""

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
        if not text:
            return
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
                        no_reply_cooldown=self._float(
                            "no_reply_cooldown_seconds", 3.0
                        ),
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
                f"Tools/MCP/SubAgent decision: {'enabled' if self._bool('tools_subagents_decision_enabled', True) else 'disabled'}\n"
                f"proactive_reply={self._bool('proactive_reply_enabled', False)}\n"
                f"provider={self._setting('provider', 'systemone_jev')}\n"
                f"endpoint={self._setting('base_url', '')}{self._setting('systemone_path', '/v1/systemone')}\n"
                f"model={self._setting('model', 'jev-latest')}\n"
                f"tool_filter={self._bool('tool_filter_enabled', True)} mcp_filter={self._bool('mcp_filter_enabled', True)} subagent_filter={self._bool('subagent_filter_enabled', False)} "
                f"tool_decision={self._bool('tool_decision_enabled', True)} mcp_decision={self._bool('mcp_decision_enabled', True)} subagent_decision={self._bool('subagent_decision_enabled', True)} "
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
        for tool in [*self._manager_tool_objects(), *self._subagent_tool_objects()]:
            name = str(getattr(tool, "name", ""))
            if not name or name in seen_names:
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
                    "active": True,
                    "builtin": False,
                    "origin_display": PLUGIN_DISPLAY_NAME,
                },
            )
        tools.sort(
            key=lambda item: (
                bool(item.get("builtin")),
                str(item.get("origin_display", "")).casefold(),
                str(item.get("name", "")).casefold(),
            )
        )
        return json_response({"tools": tools})

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
        always_keep_mcp = [
            str(item) for item in (self._setting("always_keep_mcp", []) or [])
        ]
        always_keep_recommend_mcp = [
            str(item)
            for item in (self._setting("always_keep_recommend_mcp", []) or [])
            if str(item) in always_keep_mcp
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
                "always_keep_recommend_tools": always_keep_recommend,
                "always_keep_subagents": always_keep_subagents,
                "always_keep_recommend_subagents": always_keep_recommend_subagents,
                "always_keep_mcp": always_keep_mcp,
                "always_keep_recommend_mcp": always_keep_recommend_mcp,
                "jev_pre_prompt": self._policy(),
                "main_llm_post_prompt": self._main_llm_post_prompt(),
                "tool_filter_enabled": self._bool("tool_filter_enabled", True),
                "mcp_filter_enabled": self._bool("mcp_filter_enabled", True),
                "subagent_filter_enabled": self._bool("subagent_filter_enabled", False),
                "tool_decision_enabled": self._bool("tool_decision_enabled", True),
                "mcp_decision_enabled": self._bool("mcp_decision_enabled", True),
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
        values = payload.get("always_keep_tools")
        recommendation_values = payload.get("always_keep_recommend_tools", [])
        mcp_values = payload.get("always_keep_mcp", [])
        mcp_recommendation_values = payload.get("always_keep_recommend_mcp", [])
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
        subagent_filter_enabled = payload.get("subagent_filter_enabled")
        tool_decision_enabled = payload.get("tool_decision_enabled")
        mcp_decision_enabled = payload.get("mcp_decision_enabled")
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
        if subagent_filter_enabled is not None and not isinstance(subagent_filter_enabled, bool):
            return error_response("subagent_filter_enabled must be a boolean", status_code=400)
        if tool_decision_enabled is not None and not isinstance(tool_decision_enabled, bool):
            return error_response("tool_decision_enabled must be a boolean", status_code=400)
        if mcp_decision_enabled is not None and not isinstance(mcp_decision_enabled, bool):
            return error_response("mcp_decision_enabled must be a boolean", status_code=400)
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
            if getattr(tool, "name", None)
            and not is_handoff_tool(tool)
            and not is_mcp_tool(tool)
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
                item.strip()
                for item in mcp_recommendation_values
                if item.strip() in cleaned_mcp
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
        detail_values.update(
            always_keep_tools=cleaned,
            always_keep_recommend_tools=cleaned_recommend,
            always_keep_mcp=cleaned_mcp,
            always_keep_recommend_mcp=cleaned_recommend_mcp,
            always_keep_subagents=cleaned_subagents,
            always_keep_recommend_subagents=cleaned_recommend_subagents,
            always_keep_tools_customized=True,
        )
        if tool_filter_enabled is not None:
            detail_values["tool_filter_enabled"] = tool_filter_enabled
        if mcp_filter_enabled is not None:
            detail_values["mcp_filter_enabled"] = mcp_filter_enabled
        if subagent_filter_enabled is not None:
            detail_values["subagent_filter_enabled"] = subagent_filter_enabled
        if tool_decision_enabled is not None:
            detail_values["tool_decision_enabled"] = tool_decision_enabled
        if mcp_decision_enabled is not None:
            detail_values["mcp_decision_enabled"] = mcp_decision_enabled
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
        await self._configure_output_runtime()
        return json_response(
            {
                "saved": True,
                "always_keep_tools": cleaned,
                "always_keep_recommend_tools": cleaned_recommend,
                "always_keep_mcp": cleaned_mcp,
                "always_keep_recommend_mcp": cleaned_recommend_mcp,
                "always_keep_subagents": cleaned_subagents,
                "always_keep_recommend_subagents": cleaned_recommend_subagents,
                "tool_filter_enabled": self._bool("tool_filter_enabled", True),
                "mcp_filter_enabled": self._bool("mcp_filter_enabled", True),
                "subagent_filter_enabled": self._bool("subagent_filter_enabled", False),
                "tool_decision_enabled": self._bool("tool_decision_enabled", True),
                "mcp_decision_enabled": self._bool("mcp_decision_enabled", True),
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
