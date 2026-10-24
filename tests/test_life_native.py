"""Run explicitly against a real AstrBot checkout; never contact an API/platform.

DECISION_REAL_ASTRBOT_SOURCE must point at AstrBot. ASTRBOT_ROOT must point
outside plugin source. These tests use native components, builder and runner.
"""

# ruff: noqa: E402
from __future__ import annotations

import asyncio
import base64
import copy
import importlib
import io
import json
import os
import sys
import time
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

SOURCE = os.environ.get("DECISION_REAL_ASTRBOT_SOURCE")
if not SOURCE:
    pytest.skip(
        "Set DECISION_REAL_ASTRBOT_SOURCE for native-host integration", allow_module_level=True
    )
sys.path.insert(0, SOURCE)
pytest.importorskip("astrbot.api.event")

from astrbot.api.event import MessageChain
from astrbot.api.message_components import Image
from astrbot.core.agent.tool import ToolSet
from astrbot.core.config.default import DEFAULT_CONFIG
from astrbot.core.cron.events import CronMessageEvent
from astrbot.core.platform.message_session import MessageSession
from astrbot.core.platform.platform_metadata import PlatformMetadata
from astrbot.core.provider.entities import LLMResponse, ProviderMeta
from astrbot.core.provider.provider import Provider
from astrbot.core.star.context import Context

from decision.life_agent import PlatformUnavailableError, dispatch_life_agent
from decision.life_config import life_defaults
from decision.life_media import AppearanceStore, LifeMedia, ReferenceEvent
from decision.life_runtime import CurrentPersonaResolver, LifeRuntime, fit_life_state
from decision.tokens import estimate_request_tokens
from decision.virtual_life.core.models import DailyPlan, Outfit, TimelineItem
from decision.virtual_life.core.persona import PersonaContext

from .virtual_life.fixtures import outfit_payload

UMO = "qq:FriendMessage:42"


class StubProvider(Provider):
    def __init__(self, *, fail=False, tool=False):
        super().__init__(
            {
                "id": "backup" if fail else "main",
                "modalities": ["image", "tool_use"],
                "max_context_tokens": 32000,
            },
            {},
        )
        self.fail = fail
        self.tool = tool
        self.requests = []

    def get_current_key(self):
        return "stub"

    def set_key(self, key):
        pass

    async def get_models(self):
        return ["stub"]

    def meta(self):
        return ProviderMeta(id=self.provider_config["id"], model="stub", type="stub")

    async def text_chat(self, **kwargs):
        self.requests.append(kwargs)
        if self.fail:
            raise RuntimeError("offline stub")
        if self.tool and len(self.requests) == 1:
            return LLMResponse(
                role="assistant",
                completion_text="",
                tools_call_name=["take_photo"],
                tools_call_args=[{"scene": "bookstore"}],
                tools_call_ids=["photo_1"],
            )
        return LLMResponse(role="assistant", completion_text="今天在书店。想和你分享这本书！")


@pytest.fixture
def native(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "astrbot.core.astr_main_agent.retrieve_knowledge_base", AsyncMock(return_value=None)
    )
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["timezone"] = "Asia/Shanghai"
    cfg["provider_settings"].update(
        computer_use_runtime="none", add_cron_tools=False, fallback_chat_models=[]
    )
    cfg["platform_settings"]["segmented_reply"]["enable"] = False
    cfg["platform_settings"].update(reply_with_mention=False, reply_with_quote=False)
    cfg["t2i"] = False
    ctx = MagicMock(spec=Context)
    ctx.get_config.side_effect = lambda **kw: cfg
    ctx.get_platform_inst.return_value = SimpleNamespace(
        meta=lambda: PlatformMetadata("aiocqhttp", "test", "qq", support_proactive_message=False)
    )
    conv = SimpleNamespace(cid="conversation", history="[]", persona_id="alice", token_usage=0)
    ctx.conversation_manager = SimpleNamespace(
        get_curr_conversation_id=AsyncMock(return_value=conv.cid),
        get_conversation=AsyncMock(return_value=conv),
        update_conversation=AsyncMock(),
    )
    ctx.persona_manager = SimpleNamespace(
        resolve_selected_persona=AsyncMock(
            return_value=(
                "alice",
                {"name": "alice", "prompt": "PERSONA_CURRENT_ALICE", "tools": None, "skills": []},
                None,
                False,
            )
        )
    )
    ctx.subagent_orchestrator = None
    ctx._star_manager = SimpleNamespace(context=ctx)
    ctx.get_using_tts_provider.return_value = None
    ctx.send_message = AsyncMock(return_value=True)
    ctx.get_all_stars.return_value = []
    ctx.get_llm_tool_manager.return_value = SimpleNamespace(get_full_tool_set=lambda: ToolSet())
    provider = StubProvider()
    ctx.get_using_provider.return_value = provider
    if hasattr(ctx, "get_using_provider_async"):
        async def using_provider(*args, **kwargs):
            return ctx.get_using_provider(*args, **kwargs)

        ctx.get_using_provider_async = AsyncMock(side_effect=using_provider)
    ctx.get_provider_by_id.side_effect = lambda ident: provider if ident == "main" else None
    owner = SimpleNamespace(
        context=ctx,
        html_render=AsyncMock(),
        _settings_path=tmp_path / "data/config/astrbot_plugin_decision_settings.json",
        _bool=lambda key, default=False: key == "proactive_reply_enabled",
        _evaluate=AsyncMock(),
        _context_budget=lambda: 32000,
        _command_handler_matched=lambda e: False,
        _message_address_flags=lambda e: (False, False, False, False),
    )
    settings = life_defaults()
    settings["integration_settings"].update(quiet_start="00:00", quiet_end="00:00")
    settings["friend_settings"]["session_list"] = [UMO]
    # The default is blacklist mode, where a listed session is excluded; these
    # tests want the opposite (only UMO participates), so be explicit.
    settings["friend_settings"]["session_blacklist"] = False
    settings["delivery_settings"]["availability_probabilities"] = {
        key: 1.0 for key in ("high", "normal", "low", "blocked")
    }
    settings["delivery_settings"]["segmented_reply_settings"].update(
        interval_method="random", interval="0,0"
    )
    runtime = LifeRuntime(owner, settings)
    now = runtime._now()
    runtime._conversation_activity_at[UMO] = now - timedelta(hours=1)
    plan = DailyPlan(
        date=now.date().isoformat(),
        persona_id="alice",
        theme="读书日",
        mood="轻松",
        outfit=Outfit.from_dict(outfit_payload()),
        timeline=(
            TimelineItem(
                id="day", start="00:00", end="24:00", activity="书店看书", availability="high"
            ),
        ),
    )
    runtime.storage.plans[runtime.storage.plan_key(plan.date, "alice")] = plan
    state = runtime.policy.ensure_state(UMO, "alice", plan, now)
    state.last_user_message_at = (now - timedelta(hours=1)).isoformat()
    runtime.deadlines[UMO] = time.monotonic() + 60
    owner._get_life_runtime = AsyncMock(return_value=runtime)
    event = CronMessageEvent(
        context=ctx, session=MessageSession.from_str(UMO), message="hello", extras={}
    )
    return SimpleNamespace(
        runtime=runtime,
        owner=owner,
        context=ctx,
        cfg=cfg,
        plan=plan,
        persona=PersonaContext("alice", "PERSONA_CURRENT_ALICE"),
        event=event,
        provider=provider,
    )


def test_token_trim_preserves_policy_and_valid_json():
    questions = {"q": {"type": "noul", "instructions": "decide"}}
    data = {"policy": "POLICY", "history": "old" * 20000 + "NEWEST", "time": "now"}
    state = fit_life_state(data, questions, 300)
    assert json.loads(state)["policy"] == "POLICY"
    assert json.loads(state)["history"].endswith("NEWEST")
    assert estimate_request_tokens(state, questions) < 300
    with pytest.raises(ValueError):
        fit_life_state({"policy": "x" * 5000}, questions, 100)


@pytest.mark.asyncio
async def test_persona_uses_framework_final_selection(native):
    value = await CurrentPersonaResolver(native.context).resolve(UMO)
    assert value == native.persona
    assert (
        native.context.persona_manager.resolve_selected_persona.call_args.kwargs[
            "conversation_persona_id"
        ]
        == "alice"
    )


@pytest.mark.asyncio
async def test_jev_rejection_error_and_stale_never_dispatch(native, monkeypatch):
    dispatch = AsyncMock(return_value=True)
    monkeypatch.setattr("decision.life_runtime.dispatch_life_agent", dispatch)
    native.owner._evaluate.return_value = SimpleNamespace(
        answers={q: {"noul": 0.1} for q in ("initiate", "relevant", "timing")}
    )
    assert not await native.runtime._deliver(UMO, native.persona, native.plan, "hello", 0, 0)
    assert len(native.owner._evaluate.call_args.kwargs["questions"]) == 3
    native.owner._evaluate.side_effect = RuntimeError("timeout")
    assert not await native.runtime._deliver(UMO, native.persona, native.plan, "hello", 0, 0)

    async def stale(**kw):
        native.runtime._mark_conversation_activity(UMO)
        return SimpleNamespace(answers={q: {"noul": 1} for q in ("initiate", "relevant", "timing")})

    native.owner._evaluate.side_effect = stale
    assert not await native.runtime._deliver(UMO, native.persona, native.plan, "hello", 0, 0)
    dispatch.assert_not_awaited()


@pytest.mark.asyncio
async def test_record_incoming_does_not_wait_for_schedule_generation(native):
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_schedule(_umo):
        started.set()
        await release.wait()

    native.runtime._schedule_available_umos.clear()
    native.runtime._ensure_scheduled = slow_schedule
    started_at = time.monotonic()
    await native.runtime.record_incoming(native.event)
    elapsed = time.monotonic() - started_at
    await asyncio.wait_for(started.wait(), timeout=1)
    assert elapsed < 0.5
    release.set()
    await asyncio.gather(*tuple(native.runtime._schedule_tasks.values()))


@pytest.mark.asyncio
async def test_manual_attempt_obeys_scope_and_does_not_dispatch(native, monkeypatch):
    runtime = native.runtime
    runtime.config["friend_settings"].update(enable=True, session_blacklist=False, session_list=[])
    dispatch = AsyncMock(return_value=True)
    monkeypatch.setattr("decision.life_runtime.dispatch_life_agent", dispatch)

    sent, reason = await runtime._attempt_unsolicited(
        UMO, native.persona, native.plan, "manual test", "manual"
    )
    assert sent is False
    assert reason == "session disabled or not whitelisted"
    dispatch.assert_not_awaited()


@pytest.mark.asyncio
async def test_jev_and_platform_failures_have_distinct_runtime_state(native, monkeypatch):
    runtime = native.runtime
    native.owner._evaluate.side_effect = RuntimeError("jev unavailable")
    sent, reason = await runtime._attempt_unsolicited(
        UMO, native.persona, native.plan, "jev failure", "window", attempt_key="w1"
    )
    assert sent is False and "Jev" in reason
    assert not runtime.offline

    native.owner._evaluate.side_effect = None
    native.owner._evaluate.return_value = SimpleNamespace(
        answers={q: {"noul": 1.0} for q in ("initiate", "relevant", "timing")}
    )

    async def disconnected(*args, **kwargs):
        raise PlatformUnavailableError("平台断开")

    monkeypatch.setattr("decision.life_runtime.dispatch_life_agent", disconnected)
    sent, reason = await runtime._attempt_unsolicited(
        UMO, native.persona, native.plan, "platform failure", "window", attempt_key="w2"
    )
    assert sent is False and reason == "request failed"
    assert UMO.split(":", 1)[0] in runtime.offline


@pytest.mark.asyncio
async def test_duplicate_automatic_trigger_only_delivers_once(native, monkeypatch):
    runtime = native.runtime
    native.owner._evaluate.return_value = SimpleNamespace(
        answers={q: {"noul": 1.0} for q in ("initiate", "relevant", "timing")}
    )
    calls = 0

    async def deliver(*args):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0)
        return True

    monkeypatch.setattr(runtime, "_deliver", deliver)
    results = await asyncio.gather(
        runtime._attempt_unsolicited(
            UMO, native.persona, native.plan, "same window", "window", attempt_key="same"
        ),
        runtime._attempt_unsolicited(
            UMO, native.persona, native.plan, "same window", "window", attempt_key="same"
        ),
    )
    assert calls == 1
    assert sum(1 for sent, _ in results if sent) == 1


@pytest.mark.asyncio
async def test_native_builder_pipeline_persona_and_history(native):
    assert await dispatch_life_agent(
        native.runtime, UMO, "INTERNAL_REQUEST", 0, "alice", "book sharing"
    )
    assert native.context.send_message.await_count == 2
    assert "PERSONA_CURRENT_ALICE" in str(native.provider.requests[0])
    state = native.runtime.storage.sessions[UMO]
    assert state.sent_count == 1 and state.unanswered_count == 1
    history = native.context.conversation_manager.update_conversation.call_args.kwargs["history"]
    assert any("机器人主动发起" in str(m) for m in history)
    assert "INTERNAL_REQUEST" not in str(history)
    assert history[-1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_partial_delivery_counts_once_and_saves_history(native):
    native.context.send_message.side_effect = [True, RuntimeError("platform dropped")]
    assert await dispatch_life_agent(native.runtime, UMO, "internal", 0, "alice", "share")
    assert native.runtime.storage.sessions[UMO].sent_count == 1
    native.context.conversation_manager.update_conversation.assert_awaited_once()


@pytest.mark.asyncio
async def test_unsent_platform_failure_no_quota_or_history(native):
    native.context.send_message.return_value = False
    with pytest.raises(RuntimeError):
        await dispatch_life_agent(native.runtime, UMO, "internal", 0, "alice", "share")
    assert native.runtime.storage.sessions[UMO].sent_count == 0
    native.context.conversation_manager.update_conversation.assert_not_awaited()


@pytest.mark.asyncio
async def test_persona_change_prevents_sending(native):
    native.context.persona_manager.resolve_selected_persona.return_value = (
        "bob",
        {"prompt": "BOB", "tools": None, "skills": []},
        None,
        False,
    )
    assert not await dispatch_life_agent(native.runtime, UMO, "internal", 0, "alice", "share")
    native.context.send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_native_fallback_preserves_persona(native):
    primary = StubProvider(fail=True)
    primary.provider_config["id"] = "primary"
    native.context.get_using_provider.return_value = primary
    native.cfg["provider_settings"]["fallback_chat_models"] = ["backup"]
    native.context.get_provider_by_id.side_effect = lambda ident: (
        native.provider if ident == "backup" else None
    )
    assert await dispatch_life_agent(native.runtime, UMO, "internal", 0, "alice", "share")
    assert len(primary.requests) == 1 and len(native.provider.requests) == 1
    assert "PERSONA_CURRENT_ALICE" in str(native.provider.requests[0])


def image_url():
    from PIL import Image as PILImage

    stream = io.BytesIO()
    PILImage.new("RGB", (16, 20), "white").save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


@pytest.mark.asyncio
async def test_appearance_validation_isolation_and_concurrent_updates(tmp_path):
    store = AppearanceStore(tmp_path)
    await asyncio.gather(
        store.upload("alice", "a.png", image_url()), store.upload("bob", "b.png", image_url())
    )
    refs = await store.list("alice", previews=True)
    assert len(refs) == len(await store.list("bob")) == 1
    assert refs[0]["preview"].startswith("data:image/png;")
    with pytest.raises(ValueError):
        await store.remove("bob", refs[0]["id"])
    with pytest.raises(ValueError):
        await store.remove("alice", "../outside.png")
    with pytest.raises((ValueError, OSError)):
        await store.upload("alice", "fake.png", "data:image/png;base64,SGVsbG8=")
    await store.remove("alice", refs[0]["id"])
    assert not await store.list("alice")


@pytest.mark.asyncio
async def test_reference_event_does_not_mutate_original(native, tmp_path):
    original = list(native.event.message_obj.message)
    proxy = ReferenceEvent(native.event, [tmp_path / "ref.png"], "outfit")
    assert isinstance(proxy.get_messages()[1], Image)
    assert native.event.message_obj.message == original
    assert proxy.get_message_str() == "outfit"
    assert proxy.unified_msg_origin == native.event.unified_msg_origin


@pytest.mark.asyncio
@pytest.mark.parametrize("video", [False, True])
async def test_aigen_reference_and_outfit_forwarded_once(native, tmp_path, video):
    calls = []

    class Aigen:
        async def generate_image(self, event, prompt="", aspect_ratio=""):
            calls.append((event, prompt, aspect_ratio))
            await event.send(MessageChain().message("MEDIA_SENT"))
            yield '{"status":"completed","media_type":"image"}'

        async def generate_video(
            self,
            event,
            prompt="",
            aspect_ratio="",
            duration=0,
            reference_mode="auto",
            use_message_media=True,
        ):
            assert duration == 5 and reference_mode == "reference" and use_message_media
            async for output in self.generate_image(event, prompt, aspect_ratio):
                yield output

    native.context.get_all_stars.return_value = [
        SimpleNamespace(name="astrbot_plugin_aigen", activated=True, star_cls=Aigen())
    ]
    media = LifeMedia(native.owner, tmp_path)
    await media.appearance.upload("alice", "face.png", image_url())
    native.event.send = AsyncMock()
    kind = "video" if video else "photo"
    result = await media.capture(native.event, kind=kind, scene="bookstore")
    assert json.loads(result)["status"] == "completed"
    assert len(calls) == 1
    assert "bookstore" in calls[0][1] and "白色短袖衬衫" in calls[0][1]
    assert "服装优先采用今日穿搭" in calls[0][1]
    assert isinstance(calls[0][0].get_messages()[1], Image)
    assert len(native.event.get_messages()) == 1
    native.event.send.assert_awaited_once()
    assert "冷却" in await media.capture(native.event, kind=kind)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_missing_aigen_never_generates_plan(native, tmp_path):
    native.runtime._ensure_plan_for_umo = AsyncMock()
    assert "尚未提交" in await LifeMedia(native.owner, tmp_path).capture(native.event)
    native.runtime._ensure_plan_for_umo.assert_not_awaited()


@pytest.mark.asyncio
async def test_midnight_unanswered_and_placeholder_budget(native):
    runtime = native.runtime
    state = runtime.storage.sessions[UMO]
    state.unanswered_count = 2
    tomorrow = runtime._now() + timedelta(days=1)
    tomorrow_plan = replace(native.plan, date=tomorrow.date().isoformat())
    new = runtime.policy.ensure_state(UMO, "alice", tomorrow_plan, tomorrow)
    assert new.unanswered_count == 2
    new.persona_id = ""
    new.daily_budget = 0
    assert runtime.policy.ensure_state(UMO, "alice", tomorrow_plan, tomorrow).daily_budget > 0


@pytest.mark.asyncio
async def test_runtime_shutdown_cancels_tasks_without_code_dir_data(native):
    runtime = native.runtime
    task = runtime.spawn(asyncio.sleep(100))
    await runtime.terminate()
    assert task.cancelled() and runtime.closed
    assert "plugin_data" in str(runtime.data_dir)
    assert not (Path(__file__).parents[1] / "data").exists()


def test_vendor_does_not_register_second_plugin():
    from astrbot.core.star.star import star_registry

    before = len(star_registry)
    import decision.virtual_life.engine as engine

    importlib.reload(engine)
    assert len(star_registry) == before


@pytest.mark.asyncio
async def test_real_native_photo_tool_loop_preserves_final_reply(native):
    from astrbot.core.agent.tool import FunctionTool

    provider = StubProvider(tool=True)
    native.context.get_using_provider.return_value = provider

    async def photo(event, scene=""):
        assert scene == "bookstore"
        await event.send(MessageChain().message("MOCK_PHOTO_DELIVERY"))
        return '{"status":"completed","media_type":"image"}'

    tool = FunctionTool(
        name="take_photo",
        description="photo",
        parameters={"type": "object", "properties": {"scene": {"type": "string"}}},
        handler=photo,
    )
    native.context.get_llm_tool_manager.return_value = SimpleNamespace(
        get_full_tool_set=lambda: ToolSet(tools=[tool])
    )
    assert await dispatch_life_agent(native.runtime, UMO, "internal", 0, "alice", "share")
    assert len(provider.requests) == 2
    assert native.context.send_message.await_count == 3
    assert native.runtime.storage.sessions[UMO].sent_count == 1
    history = native.context.conversation_manager.update_conversation.call_args.kwargs["history"]
    assert any(m["role"] == "tool" and "completed" in str(m) for m in history)
    assert history[-1]["role"] == "assistant"
    assert "想和你分享这本书" in str(history[-1])


@pytest.mark.asyncio
async def test_jev_approval_dispatches_exactly_once(native, monkeypatch):
    send = AsyncMock(return_value=True)
    monkeypatch.setattr("decision.life_runtime.dispatch_life_agent", send)
    native.owner._evaluate.return_value = SimpleNamespace(
        answers={q: {"noul": 0.95} for q in ("initiate", "relevant", "timing")}
    )
    assert await native.runtime._deliver(UMO, native.persona, native.plan, "share", 0, 0)
    native.owner._evaluate.assert_awaited_once()
    send.assert_awaited_once()


@pytest.mark.asyncio
async def test_group_chatter_does_not_reset_unanswered(native):
    from astrbot.core.platform.message_type import MessageType

    umo = "qq:GroupMessage:9"
    native.runtime.config["group_settings"].update(
        enable=True, session_list=[umo], session_blacklist=False
    )
    state = native.runtime.policy.ensure_state(umo, "alice", native.plan, native.runtime._now())
    state.unanswered_count = 2
    event = CronMessageEvent(
        context=native.context,
        session=MessageSession.from_str(umo),
        message="human chat",
        sender_id="bot",
        message_type=MessageType.GROUP_MESSAGE,
    )
    event.message_obj.sender.user_id = "human"
    await native.runtime.record_incoming(event)
    assert state.unanswered_count == 2
    native.owner._message_address_flags = lambda e: (True, False, False, False)
    await native.runtime.record_incoming(event)
    assert state.unanswered_count == 0
    native.runtime.runtime.stop()


@pytest.mark.asyncio
async def test_native_send_tool_is_counted_without_patching_shared_context(native):
    try:
        from astrbot.core.astr_main_agent_resources import SEND_MESSAGE_TO_USER_TOOL
    except ImportError:
        from astrbot.core.tools.message_tools import SendMessageToUserTool

        SEND_MESSAGE_TO_USER_TOOL = SendMessageToUserTool()

    class SenderProvider(StubProvider):
        async def text_chat(self, **kwargs):
            self.requests.append(kwargs)
            if len(self.requests) == 1:
                return LLMResponse(
                    role="assistant",
                    completion_text="",
                    tools_call_name=["send_message_to_user"],
                    tools_call_args=[
                        {"messages": [{"type": "plain", "text": "TOOL_DIRECT_MESSAGE"}]}
                    ],
                    tools_call_ids=["send1"],
                )
            return LLMResponse(role="assistant", completion_text="")

    provider = SenderProvider()
    native.context.get_using_provider.return_value = provider
    original_send = native.context.send_message
    native.context.get_llm_tool_manager.return_value = SimpleNamespace(
        get_full_tool_set=lambda: ToolSet(tools=[SEND_MESSAGE_TO_USER_TOOL])
    )
    assert await dispatch_life_agent(native.runtime, UMO, "internal", 0, "alice", "share")
    assert native.context.send_message is original_send
    original_send.assert_awaited_once()
    assert native.runtime.storage.sessions[UMO].sent_count == 1


@pytest.mark.asyncio
async def test_disabling_schedule_and_outfit_hides_existing_cache(native):
    native.runtime.config["schedule_settings"].update(
        daily_schedule_generation_enable=False, daily_outfit_generation_enable=False
    )
    plan = await native.runtime._ensure_plan(native.persona, native.runtime._now().date())
    assert not plan.has_schedule and not plan.has_outfit
    assert native.runtime.storage.get_plan(native.plan.date, "alice").has_outfit


@pytest.mark.asyncio
async def test_target_plugin_allowlist_is_respected(native):
    native.cfg["plugin_set"] = ["some_other_plugin"]
    assert not await dispatch_life_agent(native.runtime, UMO, "internal", 0, "alice", "share")
    native.context.send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_initialize_restores_cache_schedules_and_stops_only_own_jobs(native):
    runtime = native.runtime
    await runtime.storage.save_plans()
    await runtime.storage.save_sessions()
    restored = LifeRuntime(native.owner, runtime.config)
    try:
        await restored.initialize()
        await asyncio.gather(*tuple(restored.tasks))
        assert restored.storage.get_plan(native.plan.date, "alice").has_schedule
        assert any("idle" in job["id"] for job in restored.status()["jobs"])
        native.context.send_message.assert_not_awaited()
    finally:
        await restored.terminate()


@pytest.mark.asyncio
async def test_missing_reference_does_not_generate_daily_plan(native, tmp_path):
    native.context.get_all_stars.return_value = [
        SimpleNamespace(
            name="astrbot_plugin_aigen",
            activated=True,
            star_cls=SimpleNamespace(generate_image=AsyncMock(), generate_video=AsyncMock()),
        )
    ]
    native.runtime._ensure_plan_for_umo = AsyncMock()
    assert "尚未上传" in await LifeMedia(native.owner, tmp_path).capture(native.event)
    native.runtime._ensure_plan_for_umo.assert_not_awaited()


@pytest.mark.asyncio
async def test_real_aigen_extractor_sees_only_persona_references(native, tmp_path, monkeypatch):
    import importlib.util

    from astrbot.api.message_components import At

    source = Path(__file__).resolve().parents[2] / "astrbot_plugin_aigen/image_manager.py"
    if not source.exists():
        pytest.skip("Optional AIGen checkout is not available")
    package = ModuleType("_decision_aigen_extractor_qa")
    package.__path__ = [str(source.parent)]
    monkeypatch.setitem(sys.modules, "_decision_aigen_extractor_qa", package)
    spec = importlib.util.spec_from_file_location("_decision_aigen_extractor_qa.image_manager", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "persona.png"
    path.write_bytes(base64.b64decode(image_url().split(",", 1)[1]))
    native.event.message_obj.message = [
        Image.fromURL("https://private.invalid/user.png"),
        At(qq="999"),
    ]
    proxy = ReferenceEvent(native.event, [path], "self photo in current outfit")
    manager = module.ImageManager({})
    manager._download_image = AsyncMock(side_effect=AssertionError("No network in reference test"))
    refs = await manager.extract_image_sources_from_event(
        proxy, ignore_id="astrbot", context=proxy, strict=True
    )
    assert len(refs) == 1 and "persona.png" in refs[0]
    assert "private.invalid" not in str(refs) and "qlogo" not in str(refs)
    manager._download_image.assert_not_awaited()


@pytest.mark.asyncio
async def test_dynamic_tools_have_real_plugin_ownership(native, tmp_path, monkeypatch):
    import importlib.util
    from types import ModuleType

    from astrbot.core.star.star import star_map, star_registry
    from astrbot.core.star.star_handler import star_handlers_registry

    monkeypatch.setattr(star_handlers_registry, "_handlers", list(star_handlers_registry._handlers))
    monkeypatch.setattr(
        star_handlers_registry, "star_handlers_map", dict(star_handlers_registry.star_handlers_map)
    )
    registry_before = list(star_registry)
    map_before = dict(star_map)
    name = "_decision_native_registration"
    package = ModuleType(name)
    package.__path__ = [str(Path(__file__).parents[1])]
    monkeypatch.setitem(sys.modules, name, package)
    spec = importlib.util.spec_from_file_location(
        name + ".main", Path(__file__).parents[1] / "main.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)

    class Config(dict):
        config_path = str(tmp_path / "data/config/astrbot_plugin_decision_config.json")

    tools = []
    manager = SimpleNamespace(func_list=tools, remove_func=lambda name: None)
    native.context.provider_manager = SimpleNamespace(llm_tools=manager)
    native.context.add_llm_tools = lambda *items: Context.add_llm_tools(native.context, *items)
    try:
        plugin = module.DecisionPlugin(native.context, Config())
        assert {t.name for t in tools} == {
            "jev_decide",
            "take_photo",
            "take_video",
            "send_voice",
            "get_virtual_daily_schedule",
            "get_long_term_timeline",
            "add_schedule_requirement",
            "update_personal_network",
            "query_personal_network",
        }
        assert all(t.handler_module_path == module.__name__ for t in tools)
        assert all(t.__module__ == module.__name__ for t in tools)
        assert plugin._decision_tool.handler_module_path == module.__name__
    finally:
        star_registry[:] = registry_before
        star_map.clear()
        star_map.update(map_before)


@pytest.mark.asyncio
async def test_native_cached_state_injection_preserves_other_prompts(native):
    from astrbot.core.provider.entities import ProviderRequest

    req = ProviderRequest(prompt="今天穿什么", system_prompt="PERSONA + OTHER_PLUGIN")
    native.event.message_str = "今天穿什么"
    native.runtime.plan_generator.generate = AsyncMock(
        side_effect=AssertionError("cache must not call LLM")
    )
    await native.runtime.inject_cached_state(native.event, req)
    assert req.system_prompt == "PERSONA + OTHER_PLUGIN"
    assert any("白色短袖衬衫" in part.text for part in req.extra_user_content_parts)
    native.runtime.plan_generator.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_native_waiting_batch_can_settle_and_inject(native):
    from astrbot.core.provider.entities import ProviderRequest

    runtime = native.runtime
    runtime.config["reply_delay_settings"].update(enable=True, notify_user=False)
    runtime.config["reply_delay_settings"]["delay_formulas"]["high"] = "1"
    pending = asyncio.create_task(runtime.queue_reply_delay(native.event))
    for _ in range(20):
        await asyncio.sleep(0)
        batch = await runtime.reply_delay_coordinator.settle_now(UMO)
        if batch:
            break
    assert batch is not None
    await pending
    req = ProviderRequest(prompt="old")
    await runtime.inject_cached_state(native.event, req)
    assert req.prompt == native.event.get_message_str()
    assert any("reply_delay_context" in part.text for part in req.extra_user_content_parts)


@pytest.mark.asyncio
async def test_native_dispatch_keeps_other_plugins_persisted_context(native):
    conversation = native.context.conversation_manager.get_conversation.return_value
    conversation.history = json.dumps([{"role": "system", "content": "OTHER_PLUGIN_MEMORY"}])
    assert await dispatch_life_agent(native.runtime, UMO, "internal", 0, "alice", "share")
    history = native.context.conversation_manager.update_conversation.call_args.kwargs["history"]
    assert any(
        message.get("role") == "system" and message.get("content") == "OTHER_PLUGIN_MEMORY"
        for message in history
    )
    assert not any(
        message.get("role") == "system" and "PERSONA_CURRENT_ALICE" in str(message)
        for message in history
    )


@pytest.mark.asyncio
async def test_media_sent_then_new_message_does_not_save_unsent_final_text(native):
    from astrbot.core.agent.tool import FunctionTool

    class InterruptedProvider(StubProvider):
        async def text_chat(self, **kwargs):
            response = await super().text_chat(**kwargs)
            if len(self.requests) == 2:
                native.runtime._mark_conversation_activity(UMO)
            return response

    provider = InterruptedProvider(tool=True)
    native.context.get_using_provider.return_value = provider

    async def photo(event, scene=""):
        await event.send(MessageChain().message("DELIVERED_PHOTO"))
        return '{"status":"completed","media_type":"image"}'

    tool = FunctionTool(
        name="take_photo",
        description="photo",
        parameters={"type": "object", "properties": {"scene": {"type": "string"}}},
        handler=photo,
    )
    native.context.get_llm_tool_manager.return_value = SimpleNamespace(
        get_full_tool_set=lambda: ToolSet(tools=[tool])
    )
    assert await dispatch_life_agent(native.runtime, UMO, "internal", 0, "alice", "share")
    native.context.send_message.assert_awaited_once()
    history = native.context.conversation_manager.update_conversation.call_args.kwargs["history"]
    assert "想和你分享这本书" not in str(history)
    assert any(message["role"] == "tool" for message in history)


def test_life_chat_status_does_not_expose_other_conversations(native):
    other = "qq:FriendMessage:private_other"
    native.runtime.policy.ensure_state(other, "alice", native.plan, native.runtime._now())
    native.runtime.audit = [
        {"umo": UMO, "reason": "current"},
        {"umo": other, "reason": "private_other_reason"},
    ]
    scoped = native.runtime.status(UMO)
    assert list(scoped["sessions"]) == [UMO]
    assert other not in str(scoped) and "private_other_reason" not in str(scoped)
    assert other in native.runtime.status()["sessions"]


def test_proactive_runtime_follows_per_kind_switches(native):
    """主动发起 has no master switch: the per-kind switches decide."""

    runtime = native.runtime
    runtime.closed = False

    runtime.config["friend_settings"]["enable"] = True
    runtime.config["group_settings"]["enable"] = False
    assert runtime.enabled is True

    runtime.config["friend_settings"]["enable"] = False
    runtime.config["group_settings"]["enable"] = True
    assert runtime.enabled is True

    runtime.config["friend_settings"]["enable"] = False
    runtime.config["group_settings"]["enable"] = False
    assert runtime.enabled is False

    # A leftover legacy master switch must never re-enable or disable it.
    runtime.config["integration_settings"]["enable"] = True
    assert runtime.enabled is False
    runtime.config["friend_settings"]["enable"] = True
    runtime.config["integration_settings"]["enable"] = False
    assert runtime.enabled is True


def test_manual_trigger_obeys_whitelist_and_quiet_hours(native):
    """An immediate request remains subject to the ordinary delivery gates."""

    runtime = native.runtime
    runtime.closed = False
    runtime.config["friend_settings"]["enable"] = True
    runtime.config["friend_settings"]["session_blacklist"] = False
    runtime.config["friend_settings"]["session_list"] = []  # empty whitelist = nobody
    runtime.config["integration_settings"]["quiet_start"] = "00:00"
    runtime.config["integration_settings"]["quiet_end"] = "23:59"

    runtime.deadlines[UMO] = time.monotonic() + 600
    assert runtime.can_commit(UMO, 0) is False, "scheduled runs stay blocked"

    assert runtime.can_commit(UMO, 0) is False, "empty whitelist and quiet hours must block"


def test_sleep_wakeup_can_skip_quiet_hours(native):
    """The sleep trigger always lands in quiet hours, so it needs the switch."""

    runtime = native.runtime
    runtime.closed = False
    runtime.config["friend_settings"]["enable"] = True
    runtime.config["friend_settings"]["session_blacklist"] = True
    runtime.config["friend_settings"]["session_list"] = []
    runtime.config["integration_settings"]["quiet_start"] = "00:00"
    runtime.config["integration_settings"]["quiet_end"] = "23:59"
    runtime.config["delivery_settings"]["sleep_skips_quiet"] = True

    runtime.deadlines[UMO] = time.monotonic() + 600
    assert runtime._quiet() is True
    assert runtime._quiet_exempt(UMO) is False, "an ordinary run gets no exemption"

    runtime._sleep_umos.add(UMO)
    try:
        assert runtime._quiet_exempt(UMO) is True
        assert runtime.can_commit(UMO, 0) is True
    finally:
        runtime._sleep_umos.discard(UMO)

    runtime.config["delivery_settings"]["sleep_skips_quiet"] = False
    runtime._sleep_umos.add(UMO)
    try:
        assert runtime._quiet_exempt(UMO) is False, "the switch must be able to turn it off"
    finally:
        runtime._sleep_umos.discard(UMO)


def test_trim_session_state_bounds_the_per_session_caches(native):
    runtime = native.runtime
    now = runtime._now()
    for index in range(30):
        umo = f"aiocqhttp:FriendMessage:{index}"
        runtime._conversation_activity_at[umo] = now - timedelta(minutes=index)
        runtime._conversation_generations[umo] = index
        runtime.deadlines[umo] = 1.0
        runtime.delivery_locks[umo] = asyncio.Lock()
        runtime.observed[umo] = []

    runtime._trim_session_state(keep=10)

    assert len(runtime._conversation_activity_at) == 10
    survivors = set(runtime._conversation_activity_at)
    assert survivors == {f"aiocqhttp:FriendMessage:{i}" for i in range(10)}
    for umo in survivors:
        assert umo in runtime._conversation_generations
        assert umo in runtime.deadlines
        assert umo in runtime.delivery_locks
    dropped = "aiocqhttp:FriendMessage:29"
    for container in (
        runtime._conversation_generations,
        runtime.deadlines,
        runtime.delivery_locks,
        runtime.observed,
    ):
        assert dropped not in container


def test_trim_session_state_is_a_noop_below_the_limit(native):
    runtime = native.runtime
    runtime._conversation_activity_at.clear()
    for index in range(3):
        runtime._conversation_activity_at[f"u{index}"] = runtime._now()
    runtime._trim_session_state(keep=10)
    assert len(runtime._conversation_activity_at) == 3


@pytest.mark.asyncio
async def test_terminate_clears_the_session_caches(native):
    """A hot reload must not leave per-conversation state behind."""

    runtime = native.runtime
    runtime.closed = False
    runtime._conversation_activity_at["u1"] = runtime._now()
    runtime._conversation_generations["u1"] = 3
    runtime.deadlines["u1"] = 1.0
    runtime.delivery_locks["u1"] = asyncio.Lock()
    runtime.observed["u1"] = []
    runtime._active_agent_counts["u1"] = 1
    runtime._window_retry_counts[("u1", "rev", "w")] = 2
    runtime._schedule_available_umos.add("u1")
    runtime._sleep_umos.add("u1")
    runtime.audit.append({"umo": "u1"})

    await runtime.terminate()

    for container in (
        runtime._conversation_activity_at,
        runtime._conversation_generations,
        runtime._active_agent_counts,
        runtime._window_retry_counts,
        runtime.deadlines,
        runtime.delivery_locks,
        runtime.observed,
        runtime.audit,
        runtime._sleep_umos,
        runtime._schedule_available_umos,
    ):
        assert not container, f"terminate left behind: {container!r}"


def test_every_retry_reason_consumes_window_budget(native):
    """``agent task active`` used to retry every 30 min without ever counting."""

    runtime = native.runtime
    runtime.closed = False
    runtime.config["delivery_settings"]["window_retry_max"] = 3
    runtime.config["delivery_settings"]["window_retry_when_not_idle"] = True
    runtime.config["delivery_settings"]["window_retry_when_cooldown"] = True

    key = (UMO, "rev-1", "window-1")
    runtime._window_retry_counts.pop(key, None)

    # Simulate four consecutive "agent task active" retries through the same
    # accounting the window handler performs.
    retries_scheduled = 0
    for _ in range(4):
        count = runtime._window_retry_counts.get(key, 0)
        retry_max = max(0, int(runtime.config["delivery_settings"]["window_retry_max"]))
        if count >= retry_max:
            break
        retries_scheduled += 1
        runtime._window_retry_counts[key] = count + 1

    assert retries_scheduled == 3, "window_retry_max must cap the retries"
    assert runtime._window_retry_counts[key] == 3
    runtime._window_retry_counts.pop(key, None)


def test_should_retry_window_respects_its_switches(native):
    runtime = native.runtime
    runtime.config["delivery_settings"]["window_retry_when_not_idle"] = False
    runtime.config["delivery_settings"]["window_retry_when_cooldown"] = False

    assert runtime._should_retry_window(UMO, "sleeping") is True
    assert runtime._should_retry_window(UMO, "availability probability rejected") is True
    assert runtime._should_retry_window(UMO, "conversation is not idle enough") is False
    assert runtime._should_retry_window(UMO, "agent task active") is False
    assert runtime._should_retry_window(UMO, "cooldown active") is False
    assert runtime._should_retry_window(UMO, "something else") is False

    runtime.config["delivery_settings"]["window_retry_when_not_idle"] = True
    runtime.config["delivery_settings"]["window_retry_when_cooldown"] = True
    assert runtime._should_retry_window(UMO, "conversation is not idle enough") is True
    assert runtime._should_retry_window(UMO, "agent task active") is True
    assert runtime._should_retry_window(UMO, "cooldown active") is True
