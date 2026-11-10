"""Capability routing, address detection and conversation scope for DecisionPlugin."""

# ruff: noqa: F403, F405
from __future__ import annotations

from .input_enhancement.group_wake_suppression import WAKE_SUPPRESSED_EXTRA
from .plugin_shared import *
from .plugin_shared import _ON_DECORATING_RESULT, _safe_error  # noqa: F401


class RoutingMixin:
    """Tool/MCP/Skill/SubAgent inventory, Jev batches and message-interjection helpers."""

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
        feature: str = "能力决策",
    ):
        if not self.provider or not self._ready:
            raise DecisionProviderError("Decision Engine is not initialized")
        self._call_count += 1
        started = time.perf_counter()
        prompt = str(state) + json.dumps(questions, ensure_ascii=False, default=str)
        try:
            result = await self.provider.evaluate(
                state=state,
                questions=questions,
            )
            self._last_call_latency_ms = (time.perf_counter() - started) * 1000
            usage_meter.record(
                "Jev·" + feature,
                prompt=prompt,
                output=json.dumps(getattr(result, "answers", {}), ensure_ascii=False, default=str),
            )
            return result
        except Exception:
            self._failure_count += 1
            self._last_call_latency_ms = (time.perf_counter() - started) * 1000
            usage_meter.record("Jev·" + feature, prompt=prompt, output="", ok=False)
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
        """Return (at-self, reply-to-self, at-all, parse-failed) safely.

        Several hooks ask for the same message; parse it once per event.
        """

        cached = getattr(event, "_decision_address_flags", None)
        if cached is not None:
            return cached
        flags = self._parse_address_flags(event)
        try:
            event._decision_address_flags = flags
        except (AttributeError, TypeError):
            pass
        return flags

    def _parse_address_flags(self, event: AstrMessageEvent) -> tuple[bool, bool, bool, bool]:
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
        if (
            not parse_failed
            and not at_all
            and "@" in prefixes
            and at_self
            and not self._wake_suppressed(event)
        ):
            return True
        try:
            outline = event.get_message_outline()
        except (AttributeError, TypeError):
            outline = event.get_message_str()
        return text_matches_prefix(str(outline or ""), prefixes)

    @staticmethod
    def _wake_suppressed(event: AstrMessageEvent) -> bool:
        """This group has turned off the wake-up this @ or reply would cause."""

        return bool(event.get_extra(WAKE_SUPPRESSED_EXTRA, False))

    def _proactive_summoned(self, event: AstrMessageEvent) -> bool:
        at_self, reply_self, at_all, parse_failed = self._message_address_flags(event)
        if (
            not parse_failed
            and not at_all
            and (at_self or reply_self)
            and not self._wake_suppressed(event)
        ):
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

    def _is_admin(self, event: AstrMessageEvent) -> bool:
        try:
            admins = self.context.get_config().get("admins_id", [])
            return str(event.get_sender_id()) in {str(item) for item in admins}
        except Exception:
            return False

    async def _filter_tools_before_llm_impl(
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

    async def _proactive_reply_impl(self, event: AstrMessageEvent):
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
        # Media only matters when the operator lets non-text messages into the
        # decision; understanding itself is controlled separately.
        non_text = self._bool("non_text_proactive_enabled", False)
        media_refs = await self._media_refs_for(event) if non_text else []
        if not text:
            if not media_refs:
                return
            text = await self._media_placeholder(media_refs, session, event)
        elif media_refs:
            attachment_context = await self._media_placeholder(media_refs, session, event)
            text = f"{text}\n{attachment_context}"
        sender_id = str(event.get_sender_id() or "")
        sender_name = str(event.get_sender_name() or sender_id or "user")
        at_self, reply_self, at_all, parse_failed = self._message_address_flags(event)
        explicit_summon = (
            not parse_failed
            and not at_all
            and (at_self or reply_self)
            and not self._wake_suppressed(event)
        )
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
            forced_summon = explicit_summon and self._bool("force_reply_when_summoned", True)
            if not self.proactive.can_analyze(session) and not forced_summon:
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
            if forced_summon:
                # Skip the Jev judgement and let AstrBot's native Agent answer,
                # like a direct prefix: an explicit summon is neither limited
                # by nor counted against the proactive cooldown and window.
                event.is_at_or_wake_command = True
                self.proactive.set_status(session, ProactiveStatus.SUMMONED)
                logger.warning(
                    "Decision Engine 因为明确 @ 或回复机器人且启用强制回复，所以交给原生 Agent 回复。"
                )
                return
            # A proactive reply is already in flight: allow_reply would reject
            # any result, so do not spend a Jev request on this message.
            if self.proactive.has_pending_reply(session):
                return
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
                result = await self._evaluate(state=state, questions=questions, feature="消息接话")
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
                        addressing >= bounded_float(self._setting("addressing_threshold", 0.7), 0.7)
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
            reply_token = self.proactive.allow_reply(
                session,
                cooldown_seconds=self._float("cooldown_seconds", 60),
                window_seconds=self._float("reply_window_seconds", 600),
                max_replies=self._int("max_replies_per_window", 2),
            )
            if reply_token is None:
                logger.warning("Decision Engine 因为主动回复冷却或窗口次数限制，所以不主动对话。")
                return

            try:
                cid = await self.context.conversation_manager.get_curr_conversation_id(session)
                conversation = await self.context.conversation_manager.get_conversation(
                    session, cid
                )
            except Exception as exc:
                self.proactive.cancel_reply(session, reply_token)
                logger.debug(
                    "Decision Engine proactive request was not queued: %s", _safe_error(exc)
                )
                return
            if not conversation:
                self.proactive.cancel_reply(session, reply_token)
                return
        # Leave the per-session lock first: AstrBot runs the yielded request
        # (the whole agent loop) before resuming this handler, and other
        # messages in the session, including an explicit @bot, must not wait
        # for that reply.  The reply slot reserved above keeps a second
        # proactive reply out while this one is in flight.
        queued = False
        try:
            event.is_at_or_wake_command = True
            set_extra = getattr(event, "set_extra", None)
            if callable(set_extra):
                # Lets remember_bot_response settle only this reservation.
                set_extra(PROACTIVE_REPLY_EXTRA, reply_token)
            # This request is already being sent through the handler's
            # ProviderRequest path.  Mark it as consumed so AstrBot's
            # ProcessStage does not issue a second native LLM request
            # merely because the event is also marked as awakened.
            event.should_call_llm(True)
            request = event.request_llm(
                prompt=text,
                # Continue AstrBot's native conversation so persona and
                # other plugin system prompts remain intact.
                session_id=cid,
                conversation=conversation,
            )
            queued = True
            yield request
        except Exception as exc:
            if not queued:
                self.proactive.cancel_reply(session, reply_token)
            logger.debug("Decision Engine proactive request failed: %s", _safe_error(exc))
        finally:
            # A delivered reply already settled this reservation in
            # remember_bot_response.  Otherwise (provider error, a hook stopped
            # by another plugin, a closed generator) release it here.  It stays
            # counted: the reply may still have reached the chat, and counting
            # errs towards interjecting less.  The token check leaves a newer
            # reservation untouched.
            if queued:
                self.proactive.mark_reply_finished(session, reply_token)
