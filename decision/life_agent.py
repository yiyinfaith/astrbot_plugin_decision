"""Dispatch unsolicited messages through AstrBot's native Agent and send stages."""

from __future__ import annotations

import asyncio
import copy
import inspect
from dataclasses import fields


async def dispatch_life_agent(
    runtime, umo: str, prompt: str, generation: int, persona_id: str, intent: str
) -> bool:
    from astrbot import logger
    from astrbot.api.message_components import Plain
    from astrbot.core.astr_main_agent import MainAgentBuildConfig, build_main_agent
    from astrbot.core.cron.events import CronMessageEvent
    from astrbot.core.message.message_event_result import ResultContentType
    from astrbot.core.pipeline.context import PipelineContext, call_event_hook
    from astrbot.core.pipeline.respond.stage import RespondStage
    from astrbot.core.pipeline.result_decorate.stage import ResultDecorateStage
    from astrbot.core.platform.message_session import MessageSession
    from astrbot.core.star.star_handler import EventType
    from astrbot.core.utils.session_lock import session_lock_manager

    ctx = runtime.context
    session = MessageSession.from_str(umo)
    event = CronMessageEvent(
        context=ctx,
        session=session,
        message=prompt,
        message_type=session.message_type,
        extras={"decision_life_proactive": True, "decision_life_generation": generation},
    )
    event.role = "member"  # A scheduled task never inherits administrator rights.
    platform = ctx.get_platform_inst(session.platform_id)
    if platform is None:
        raise RuntimeError("目标平台未连接")
    event.platform_meta = platform.meta()
    cfg = ctx.get_config(umo=umo)
    plugins = cfg.get("plugin_set", ["*"])
    event.plugins_name = None if plugins == ["*"] else plugins
    if plugins != ["*"] and "astrbot_plugin_decision" not in plugins:
        return False
    settings = cfg.get("provider_settings", {})
    manager = getattr(ctx, "_star_manager", None)
    if manager is None:
        raise RuntimeError("当前 AstrBot 不提供原生发送管道")
    pipe = PipelineContext(cfg, manager, event.get_platform_id())
    values = {f.name: settings[f.name] for f in fields(MainAgentBuildConfig) if f.name in settings}
    values.update(
        streaming_response=False, provider_settings=settings, timezone=cfg.get("timezone")
    )
    config = MainAgentBuildConfig(**values)
    selected = runtime.config["schedule_settings"].get("proactive_llm_provider", "")
    provider = ctx.get_provider_by_id(selected) if selected else None
    if selected and provider is None:
        raise RuntimeError("配置的主动消息模型不存在")

    sent = 0
    runner = None
    result = None
    initial_user = None
    split = False
    send_failure = None
    delivering_final = False
    final_delivery = False
    original_send = ctx.send_message
    delivery_context = copy.copy(ctx)

    async def guarded_send(target, chain):
        nonlocal sent, send_failure, final_delivery
        if str(target) != umo:
            raise RuntimeError("主动发起只能发送到本次已订阅的目标会话")
        if not await runtime.validate_commit(umo, generation, persona_id):
            raise RuntimeError("主动候选已过期或会话已更新")
        try:
            if not await original_send(session, chain):
                raise RuntimeError("平台未确认主动消息发送成功")
        except Exception as exc:
            send_failure = exc
            raise
        if not sent:
            runtime.confirm_delivery(umo)
        sent += 1
        final_delivery = final_delivery or delivering_final
        event._has_send_oper = True
        return True

    # A per-run Context copy also covers AstrBot's send_message_to_user tool,
    # which sends through Context directly rather than through event.send.
    # The real shared Context and other plugins are never patched.
    delivery_context.send_message = guarded_send
    event.context_obj = delivery_context

    # CronMessageEvent.send discards Context.send_message's bool. Track it
    # directly so a disconnected platform can never consume a daily quota.
    async def send(chain):
        if not await runtime.validate_commit(umo, generation, persona_id):
            raise RuntimeError("主动候选已过期或会话已更新")
        segments = [chain]
        if split and len(chain.chain) == 1 and isinstance(chain.chain[0], Plain):
            from astrbot.api.event import MessageChain

            segments = [
                MessageChain().message(text)
                for text in runtime.message_segmenter.split(
                    "".join(c.text for c in chain.chain)
                ).segments
            ]
        for index, part in enumerate(segments):
            if not await runtime.validate_commit(umo, generation, persona_id):
                raise RuntimeError("分段发送期间会话已更新")
            await guarded_send(session, part)
            if index + 1 < len(segments):
                await asyncio.sleep(
                    runtime.message_segmenter.interval_for("".join(c.text for c in part.chain))
                )

    event.send = send
    async with session_lock_manager.acquire_lock(umo):
        if not runtime.can_commit(umo, generation):
            return False
        result = await build_main_agent(
            event=event,
            plugin_context=delivery_context,
            config=config,
            provider=provider,
            apply_reset=False,
        )
        if not result:
            return False
        pending = result.reset_coro
        try:
            if await call_event_hook(event, EventType.OnLLMRequestEvent, result.provider_request):
                return False
            if pending:
                await pending
                pending = None
            runner = result.agent_runner
            initial_user = next(
                (
                    m
                    for m in reversed(runner.run_context.messages)
                    if getattr(m, "role", "") == "user"
                ),
                None,
            )
            async for _ in runner.step_until_done(int(settings.get("max_agent_step", 30))):
                if not runtime.can_commit(umo, generation):
                    return sent > 0
            response = runner.get_final_llm_resp()
            # MAIN_AGENT_HOOKS already dispatches OnLLMResponseEvent. Calling
            # it again would apply other plugins' state changes twice.
            if response and getattr(response, "role", "assistant") == "assistant":
                text = str(response.completion_text or "").strip()
                if text and runtime.can_commit(umo, generation):
                    delivering_final = True
                    result_chain = event.plain_result(text)
                    result_chain.set_result_content_type(ResultContentType.LLM_RESULT)
                    event.set_result(result_chain)
                    setattr(event, "__is_llm_reply", True)
                    for cls in (ResultDecorateStage, RespondStage):
                        stage = cls()
                        await stage.initialize(pipe)
                        if cls is RespondStage:
                            # Let native/output enhancement splitting take
                            # precedence. Never split the same reply twice.
                            native_split = (
                                cfg.get("platform_settings", {})
                                .get("segmented_reply", {})
                                .get("enable", False)
                            )
                            split = not native_split and not event.get_extra(
                                "decision_output_handled", False
                            )
                        step = stage.process(event)
                        if inspect.isasyncgen(step):
                            async for _ in step:
                                pass
                        else:
                            await step
                        if event.is_stopped():
                            break
            if not sent and send_failure:
                raise send_failure
            return sent > 0
        except Exception as exc:
            if not sent:
                raise
            logger.warning("Decision Engine 主动消息已部分发送，后续处理失败：%s", str(exc)[:200])
            return True
        finally:
            if sent and runner:
                # Persist the actual tool-loop messages, including photo/video
                # tool status. Do not invent a user message or discard tools.
                messages = runner.run_context.messages
                history = []
                skipped_initial_system = False
                for message in messages:
                    if getattr(message, "role", "") == "system" and not skipped_initial_system:
                        skipped_initial_system = True
                        continue
                    if getattr(message, "_no_save", False):
                        continue
                    if (
                        message is messages[-1]
                        and getattr(message, "role", "") == "assistant"
                        and not getattr(message, "tool_calls", None)
                        and not final_delivery
                    ):
                        # A tool may already have delivered media before a
                        # new human message invalidates the final text. Keep
                        # that tool history, never invent an unsent reply.
                        continue
                    if message is initial_user:
                        # Save an explicit Bot-origin note instead of the
                        # internal generation instructions as a human utterance.
                        note = runtime.config["delivery_settings"][
                            "proactive_history_note_template"
                        ].replace("{intent}", intent)
                        history.append(
                            {
                                "role": "user",
                                "content": "[机器人主动发起事件，并非用户发言]\n" + note,
                            }
                        )
                    else:
                        history.append(
                            message.model_dump() if hasattr(message, "model_dump") else message
                        )
                req = result.provider_request
                if req.conversation:
                    try:
                        await ctx.conversation_manager.update_conversation(
                            umo, req.conversation.cid, history=history
                        )
                    except Exception as exc:
                        logger.warning(
                            "Decision Engine 主动消息已发送，但历史保存失败：%s", str(exc)[:160]
                        )
                await runtime.storage.save_sessions()
            if pending and inspect.iscoroutine(pending):
                pending.close()
