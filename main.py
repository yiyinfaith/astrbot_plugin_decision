from __future__ import annotations

import time
from collections.abc import Mapping
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

from .decision.context import (
    build_decision_state,
    is_handoff_tool,
    question_id,
    short_description,
    tool_is_active,
    tool_summary,
)
from .decision.models import DecisionProviderError
from .decision.proactive import ProactiveRecord, ProactiveState
from .decision.providers.systemone import SystemOneProvider
from .decision.routing import (
    DECISION_TOOL_NAME,
    DEFAULT_MAIN_LLM_POST_PROMPT,
    add_routing_hint,
    choose_tools,
    compact_answer,
    decision_tool_parameters,
    recommendations_from_noul,
)

PLUGIN_NAME = "astrbot_plugin_decision"
DEFAULT_POLICY = (
    "你是一个只负责结构化判断的 Decision Model。\n"
    "请严格根据当前场景、用户请求和候选能力逐项判断。\n"
    "不要生成最终回答，不要执行工具，不要发明候选名称。"
)


@register(
    PLUGIN_NAME,
    "yiyinfaith",
    "AstrBot 智能决策引擎：Tools 逐个过滤与推荐、SubAgent 逐个推荐、主动回复判断和结构化决策。",
    "0.1.0",
)
class DecisionPlugin(Star):
    """AstrBot integration layer for a provider-neutral decision engine."""

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.context = context
        self.config = config
        self.provider: SystemOneProvider | None = None
        self._ready = False
        self._active_reply_warning_emitted = False
        self.proactive = ProactiveState(self._int("proactive_history_max_messages", 12))
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

    async def initialize(self) -> None:
        provider_name = str(self.config.get("provider", "systemone_jev"))
        if provider_name != "systemone_jev":
            logger.warning(
                "Decision Engine provider %s is unavailable; plugin stays fail-open.", provider_name
            )
            return
        self.provider = SystemOneProvider(
            base_url=str(self.config.get("base_url", "")),
            path=str(self.config.get("systemone_path", "/v1/systemone")),
            api_key=str(self.config.get("api_key", "")),
            model=str(self.config.get("model", "jev-latest")),
            timeout_sec=self._float("timeout_sec", 10.0),
            retries=self._int("retries", 1),
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
        if not self.provider.api_key:
            logger.warning(
                "Decision Engine API key is empty; tool filtering and proactive decisions are disabled."
            )

    async def terminate(self) -> None:
        if self.provider is not None:
            await self.provider.close()
        self.provider = None
        self._ready = False

    def _int(self, key: str, default: int) -> int:
        try:
            return int(self.config.get(key, default))
        except (TypeError, ValueError):
            return default

    def _float(self, key: str, default: float) -> float:
        try:
            return float(self.config.get(key, default))
        except (TypeError, ValueError):
            return default

    def _bool(self, key: str, default: bool = False) -> bool:
        value = self.config.get(key, default)
        return value if isinstance(value, bool) else bool(value)

    def _policy(self) -> str:
        return str(self.config.get("jev_pre_prompt", "")).strip() or DEFAULT_POLICY

    def _main_llm_post_prompt(self) -> str:
        configured = self.config.get("main_llm_post_prompt")
        if configured is None:
            return DEFAULT_MAIN_LLM_POST_PROMPT
        # An explicitly empty value is a supported way to disable this
        # optional advisory section without disabling Tool filtering.
        return str(configured)

    def _tool_list(self, req: ProviderRequest) -> list[Any]:
        return list(getattr(getattr(req, "func_tool", None), "tools", None) or [])

    def _ensure_request_toolset(self, req: ProviderRequest) -> ToolSet:
        """Keep the provider request usable even when AstrBot supplied no ToolSet.

        The Decision Engine tool is registered globally through ``Context``. A
        request can still arrive with ``func_tool=None`` when the active
        persona has no ordinary tools, so add a local ToolSet rather than
        silently making ``decision_evaluate`` unavailable.
        """

        tool_set = getattr(req, "func_tool", None)
        if tool_set is None:
            tool_set = ToolSet(tools=[])
            req.func_tool = tool_set
        if tool_set.get_tool(DECISION_TOOL_NAME) is None:
            tool_set.add_tool(self._decision_tool)
        return tool_set

    def _state_for_request(
        self,
        req: ProviderRequest,
        tools: list[Any],
        handoffs: list[Any],
    ) -> str:
        max_desc = self._int("tool_description_max_chars", 240)
        return build_decision_state(
            policy=self._policy(),
            current_prompt=str(req.prompt or ""),
            contexts=req.contexts,
            tools=[tool_summary(tool, max_desc) for tool in tools],
            subagents=[tool_summary(tool, max_desc) for tool in handoffs],
            history_max_messages=self._int("history_max_messages", 8),
            history_max_chars=self._int("history_max_chars", 6000),
        )

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
                model=str(self.config.get("model", "jev-latest")),
            )
            self._last_call_latency_ms = (time.perf_counter() - started) * 1000
            return result
        except Exception:
            self._failure_count += 1
            self._last_call_latency_ms = (time.perf_counter() - started) * 1000
            raise

    @filter.on_llm_request(priority=1000)
    async def filter_tools_before_llm(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
    ) -> None:
        """Run one fan-out decision request before AstrBot's main LLM call."""

        if not self._bool("enable", True):
            return
        tool_set = self._ensure_request_toolset(req)
        subagent_recommendation_enabled = self._bool("subagent_recommendation_enabled", True)
        original = list(tool_set.tools)
        if not original:
            return
        handoffs = [tool for tool in original if is_handoff_tool(tool)]
        ordinary = [
            tool
            for tool in original
            if not is_handoff_tool(tool)
            and getattr(tool, "name", None) != DECISION_TOOL_NAME
            and tool_is_active(tool)
        ]
        always_keep = {
            str(name).strip()
            for name in (self.config.get("always_keep_tools", []) or [])
            if str(name).strip()
        }
        always_keep_recommend = {
            str(name).strip()
            for name in (self.config.get("always_keep_recommend_tools", []) or [])
            if str(name).strip()
        } & always_keep
        unknown_always_keep = always_keep - {
            str(getattr(tool, "name", "")) for tool in original if getattr(tool, "name", None)
        }
        if unknown_always_keep:
            logger.debug(
                "Decision Engine ignored %d Always Keep names not present in this request.",
                len(unknown_always_keep),
            )
        question_to_tool: dict[str, str] = {}
        questions: dict[str, dict[str, Any]] = {}
        for index, tool in enumerate(ordinary):
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

        subagent_question_to_name: dict[str, str] = {}
        if subagent_recommendation_enabled:
            for index, tool in enumerate(handoffs):
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
            # explicit Decision Engine tool must survive unchanged.
            outcome = choose_tools(
                original,
                {},
                {},
                threshold=self._float("tool_noul_threshold", 0.2),
                always_keep=always_keep,
                decision_tool=decision_tool,
                always_keep_recommend=always_keep_recommend,
            )
            req.func_tool.tools = outcome.selected
            return

        state = self._state_for_request(req, ordinary, handoffs)
        try:
            result = await self._evaluate(state=state, questions=questions)
        except Exception as exc:
            # Tool filtering is deliberately fail-open: leave the original
            # request untouched, apart from making the built-in decision tool
            # available when AstrBot supplied a tool set.
            logger.warning("Decision Engine tool filter unavailable: %s", _safe_error(exc))
            if tool_set.get_tool(DECISION_TOOL_NAME) is None:
                tool_set.add_tool(decision_tool)
            return

        noul_answers = {
            qid: answer
            for qid, answer in result.answers.items()
            if qid in question_to_tool or qid in subagent_question_to_name
        }
        outcome = choose_tools(
            original,
            noul_answers,
            question_to_tool,
            threshold=min(1.0, max(0.0, self._float("tool_noul_threshold", 0.2))),
            always_keep=always_keep,
            decision_tool=decision_tool,
            always_keep_recommend=always_keep_recommend,
        )
        threshold = min(1.0, max(0.0, self._float("tool_noul_threshold", 0.2)))
        recommended_subagents = recommendations_from_noul(
            noul_answers,
            subagent_question_to_name,
            threshold=threshold,
        )
        recommended_tools = outcome.recommended_tools
        add_routing_hint(
            req,
            template=self._main_llm_post_prompt(),
            recommended_tools=recommended_tools,
            recommended_subagents=recommended_subagents,
        )
        req.func_tool.tools = outcome.selected
        if self._bool("debug_log", False):
            selected_names = [str(getattr(tool, "name", "")) for tool in outcome.selected]
            logger.debug(
                "Decision routing: input_tools=%d selected=%d always_keep=%d always_keep_recommend=%d subagents=%d recommended_tools=%s recommended_subagents=%s",
                len(original),
                len(selected_names),
                len(always_keep),
                len(always_keep_recommend),
                len(handoffs),
                ",".join(recommended_tools) or "none",
                ",".join(recommended_subagents) or "none",
            )

    async def _run_decision_tool(
        self,
        event: AstrMessageEvent,
        state: str,
        decision_type: str,
        instructions: str,
        choice_options: list[dict[str, str]] | None = None,
        score_levels: list[str] | None = None,
    ) -> str:
        """Handler exposed to the main LLM as ``decision_evaluate``."""

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

    def _should_skip_proactive(self, event: AstrMessageEvent) -> bool:
        if event.get_message_type() != MessageType.GROUP_MESSAGE:
            return True
        if bool(getattr(event, "is_at_or_wake_command", False)):
            return True
        if event.get_extra("handlers_parsed_params", {}):
            return True
        try:
            from astrbot.api.message_components import Reply

            self_id = str(event.get_self_id() or "")
            for component in event.get_messages():
                if isinstance(component, Reply):
                    sender_id = str(
                        getattr(component, "sender_id", None)
                        or getattr(component, "sender", None)
                        or ""
                    )
                    if sender_id and sender_id == self_id:
                        return True
        except Exception:
            pass
        return False

    @filter.event_message_type(EventMessageType.GROUP_MESSAGE, priority=1000)
    async def proactive_reply(self, event: AstrMessageEvent):
        """Optionally wake AstrBot's normal Agent loop for ambient group messages."""

        if not self._bool("enable", True) or not self._bool("proactive_reply_enabled", False):
            return
        if not self.provider or not self._ready or self._should_skip_proactive(event):
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
        history = self.proactive.history_lines(session, self._int("history_max_chars", 6000))
        self.proactive.add(session, ProactiveRecord(sender_name, sender_id, text))
        state = (
            f"[Decision Policy]\n{self._policy()}\n\n"
            "[Group Conversation]\n"
            f"{chr(10).join(history) or '(none)'}\n\n"
            f"[Current Message]\n{sender_name} ({sender_id}): {text}\n"
            "The current message was not directly addressed to the bot."
        )
        questions = {
            "is_addressing_bot": {
                "type": "noul",
                "instructions": "Is the current message meaningfully addressing the bot?",
            },
            "should_interject": {
                "type": "noul",
                "instructions": "Would a brief, helpful bot reply now be natural rather than disruptive?",
            },
        }
        try:
            result = await self._evaluate(state=state, questions=questions)
        except Exception as exc:
            logger.debug(
                "Decision Engine proactive check closed after failure: %s", _safe_error(exc)
            )
            return
        addressing = float(result.answers["is_addressing_bot"].get("noul", 0.0))
        interject = float(result.answers["should_interject"].get("noul", 0.0))
        if addressing < self._float("addressing_threshold", 0.7) and interject < self._float(
            "interject_threshold", 0.85
        ):
            return
        if not self.proactive.allow_reply(
            session,
            cooldown_seconds=self._float("cooldown_seconds", 60),
            window_seconds=self._float("reply_window_seconds", 600),
            max_replies=self._int("max_replies_per_window", 2),
        ):
            return

        try:
            cid = await self.context.conversation_manager.get_curr_conversation_id(session)
            conversation = await self.context.conversation_manager.get_conversation(session, cid)
            if not conversation:
                return
            yield event.request_llm(
                prompt=text,
                # AstrBot's native active-reply flow passes the current
                # conversation id as session_id so the request continues the
                # existing Agent conversation instead of starting a new one.
                session_id=cid,
                conversation=conversation,
            )
        except Exception as exc:
            logger.debug("Decision Engine proactive request was not queued: %s", _safe_error(exc))

    @filter.on_llm_response()
    async def remember_bot_response(self, event: AstrMessageEvent, response: Any) -> None:
        if not self._bool("proactive_reply_enabled", False):
            return
        if event.get_message_type() != MessageType.GROUP_MESSAGE:
            return
        text = str(getattr(response, "completion_text", "") or "").strip()
        if text:
            self.proactive.add(
                str(event.unified_msg_origin),
                ProactiveRecord("bot", str(event.get_self_id() or "bot"), text, True),
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
            tools = self.context.get_llm_tool_manager().func_list
            handoffs = sum(isinstance(item, HandoffTool) for item in tools)
            output = (
                f"Decision Engine: {'enabled' if self._bool('enable', True) else 'disabled'}\n"
                f"provider={self.config.get('provider', 'systemone_jev')}\n"
                f"endpoint={self.config.get('base_url', '')}{self.config.get('systemone_path', '/v1/systemone')}\n"
                f"model={self.config.get('model', 'jev-latest')}\n"
                f"tool_filter=always_on threshold={self._float('tool_noul_threshold', 0.2):.3f}\n"
                f"registered_tools={len(tools)} handoffs={handoffs} always_keep={len(self.config.get('always_keep_tools', []) or [])}\n"
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
            tools = self.context.get_llm_tool_manager().func_list
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
        for tool in self.context.get_llm_tool_manager().func_list:
            if getattr(tool, "name", None) == DECISION_TOOL_NAME:
                continue
            tools.append(
                {
                    "name": str(getattr(tool, "name", "")),
                    "description": short_description(
                        getattr(tool, "description", ""),
                        self._int("tool_description_max_chars", 240),
                    ),
                    "handoff": is_handoff_tool(tool),
                    "active": tool_is_active(tool),
                }
            )
        return json_response({"tools": tools})

    async def page_settings(self):
        from astrbot.api.web import json_response

        always_keep = [str(item) for item in (self.config.get("always_keep_tools", []) or [])]
        always_keep_recommend = [
            str(item)
            for item in (self.config.get("always_keep_recommend_tools", []) or [])
            if str(item) in always_keep
        ]
        return json_response(
            {
                "always_keep_tools": always_keep,
                "always_keep_recommend_tools": always_keep_recommend,
                "jev_pre_prompt": self._policy(),
                "main_llm_post_prompt": self._main_llm_post_prompt(),
                "description_max_chars": self._int("tool_description_max_chars", 240),
            }
        )

    async def page_save_settings(self):
        from astrbot.api.web import error_response, json_response, request

        payload = await request.json(default={})
        if not isinstance(payload, Mapping):
            return error_response("request body must be an object", status_code=400)
        values = payload.get("always_keep_tools")
        recommendation_values = payload.get("always_keep_recommend_tools", [])
        if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
            return error_response("always_keep_tools must be a string list", status_code=400)
        if not isinstance(recommendation_values, list) or any(
            not isinstance(item, str) for item in recommendation_values
        ):
            return error_response(
                "always_keep_recommend_tools must be a string list", status_code=400
            )
        jev_pre_prompt = payload.get("jev_pre_prompt")
        main_llm_post_prompt = payload.get("main_llm_post_prompt")
        if jev_pre_prompt is not None and not isinstance(jev_pre_prompt, str):
            return error_response("jev_pre_prompt must be a string", status_code=400)
        if main_llm_post_prompt is not None and not isinstance(main_llm_post_prompt, str):
            return error_response("main_llm_post_prompt must be a string", status_code=400)
        available = {
            str(getattr(tool, "name", ""))
            for tool in self.context.get_llm_tool_manager().func_list
            if getattr(tool, "name", None) and getattr(tool, "name", None) != DECISION_TOOL_NAME
        }
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
        self.config["always_keep_tools"] = cleaned
        self.config["always_keep_recommend_tools"] = cleaned_recommend
        if jev_pre_prompt is not None:
            self.config["jev_pre_prompt"] = jev_pre_prompt[:50000]
        if main_llm_post_prompt is not None:
            self.config["main_llm_post_prompt"] = main_llm_post_prompt[:50000]
        save = getattr(self.config, "save_config", None)
        if callable(save):
            save()
        return json_response(
            {
                "saved": True,
                "always_keep_tools": cleaned,
                "always_keep_recommend_tools": cleaned_recommend,
                "jev_pre_prompt": self._policy(),
                "main_llm_post_prompt": self._main_llm_post_prompt(),
            }
        )


def _safe_error(error: BaseException) -> str:
    text = str(error).replace("\n", " ").strip()
    return text[:240] or error.__class__.__name__
