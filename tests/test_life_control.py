from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest

from decision.life_config import life_defaults, normalize_life
from decision.life_contacts import ContactControl, group_authorized
from decision.life_ledger import LifeLedger
from decision.life_temporal import TemporalControl
from decision.virtual_life.core.proactive import ProactivePolicy

UMO = "qq:FriendMessage:42"
GROUP = "qq:GroupMessage:100"


class Event:
    def __init__(self, umo=UMO, *, sender="42", role="member", platform_role="member", text="六小时后提醒我看书", message_id="1"):
        self.unified_msg_origin, self.role, self.text, self.sender = umo, role, text, sender
        self.message_obj = SimpleNamespace(message_id=message_id, raw_message={"sender": {"role": platform_role}})
        self.extras = {}

    def get_sender_id(self):
        return self.sender

    def get_message_str(self):
        return self.text

    def get_extra(self, name, default=None):
        return self.extras.get(name, default)

    def set_extra(self, name, value):
        self.extras[name] = value


@pytest.fixture
def control(tmp_path):
    cfg = life_defaults()
    for key in ("adaptive_settings", "temporal_settings", "thought_settings"):
        cfg[key]["enable"] = True
    cfg["friend_settings"].update(enable=True, session_blacklist=True, session_list=[])
    now = [datetime(2026, 10, 25, 10, tzinfo=ZoneInfo("Asia/Shanghai"))]
    provider = SimpleNamespace(text_chat=AsyncMock(return_value=SimpleNamespace(completion_text="约好的读书时间到啦。")))
    host = SimpleNamespace(get_using_provider=lambda **kw: provider, cron_manager=SimpleNamespace(
        add_basic_job=AsyncMock(return_value=SimpleNamespace(job_id="native-job")), delete_job=AsyncMock()))
    runtime = SimpleNamespace(config=cfg, _now=lambda: now[0], timezone=now[0].tzinfo,
        personas=SimpleNamespace(resolve=AsyncMock(return_value=SimpleNamespace(id="alice", prompt="学生Alice，喜欢阅读"))),
        runtime=SimpleNamespace(add_date_job=lambda *args, **kwargs: None, remove=lambda *args: None),
        context=host, enabled=True, _quiet=lambda: False, delivery_locks={}, delivery_slots=asyncio.Semaphore(2), record_audit=lambda *a: None)
    runtime.policy = ProactivePolicy(cfg, SimpleNamespace(sessions={}), now[0].tzinfo)
    ledger = LifeLedger(tmp_path / "control.sqlite3")
    contacts = ContactControl(runtime, ledger)
    temporal = TemporalControl(runtime, ledger, contacts)
    runtime.ledger, runtime.contacts, runtime.temporal = ledger, contacts, temporal
    return SimpleNamespace(runtime=runtime, ledger=ledger, contacts=contacts, temporal=temporal, cfg=cfg, now=now, provider=provider)


def test_frequency_is_persistent_scoped_and_bounded(control):
    c = control
    initial = c.contacts.summary(UMO, "alice", c.now[0])["interval"]
    c.contacts.delivered(UMO, "alice", c.now[0])
    c.now[0] += timedelta(minutes=10)
    c.contacts.observe(Event(), "alice", c.now[0], targeted=True)
    assert c.contacts.summary(UMO, "alice", c.now[0])["interval"] < initial
    c.contacts.preference(Event(text="太吵了"), "alice", "less")
    quiet = c.contacts.summary(UMO, "alice", c.now[0])["interval"]
    clone = ContactControl(c.runtime, LifeLedger(c.ledger.path))
    assert clone.summary(UMO, "alice", c.now[0])["interval"] == quiet
    assert clone.summary(UMO, "bob", c.now[0])["interval"] == initial
    assert clone.summary("other:FriendMessage:42", "alice", c.now[0])["interval"] == initial
    for _ in range(30):
        c.now[0] += timedelta(days=3)
        c.contacts.delivered(UMO, "alice", c.now[0])
        c.now[0] += timedelta(hours=3)
        state = c.contacts.summary(UMO, "alice", c.now[0])
        assert 120 <= state["interval"] <= 2880
        old = state["interval"]
        assert c.contacts.summary(UMO, "alice", c.now[0])["interval"] == old  # once per missed reply
    c.now[0] += timedelta(days=3)
    assert c.contacts.gate(UMO, "alice", c.now[0])[0]
    c.contacts.preference(Event(), "alice", "pause")
    c.now[0] += timedelta(days=30)
    assert not c.contacts.gate(UMO, "alice", c.now[0])[0]
    c.contacts.preference(Event(), "alice", "resume")
    assert c.contacts.gate(UMO, "alice", c.now[0])[0]


@pytest.mark.parametrize("role,platform_role,listed,blacklist,expected", [
    ("admin", "member", False, False, True), ("member", "owner", False, False, True),
    ("member", "admin", False, False, True), ("member", "member", True, False, True),
    ("member", "member", False, False, False), ("member", "member", True, True, False),
    ("admin", "owner", True, True, False), ("member", "member", False, True, True),
])
def test_real_group_roles_and_qq_lists(control, role, platform_role, listed, blacklist, expected):
    settings = control.cfg["adaptive_settings"]
    settings.update(group_adjust_blacklist=blacklist, group_adjust_users=["42"] if listed else [])
    event = Event(GROUP, role=role, platform_role=platform_role)
    assert group_authorized(event, settings) is expected
    if not expected:
        with pytest.raises(ValueError, match="无权"):
            control.contacts.preference(event, "alice", "less")


def test_unrelated_group_messages_and_duplicate_hook_do_not_adjust(control):
    c = control
    event = Event(GROUP, text="太吵了", platform_role="owner")
    base = c.contacts.summary(GROUP, "alice", c.now[0])["interval"]
    c.contacts.observe(event, "alice", c.now[0], targeted=False)
    assert c.contacts.summary(GROUP, "alice", c.now[0])["interval"] == base
    event.message_obj.message_id = "2"
    c.contacts.observe(event, "alice", c.now[0], targeted=True)
    changed = c.contacts.summary(GROUP, "alice", c.now[0])["interval"]
    assert changed > base
    c.contacts.observe(event, "alice", c.now[0], targeted=True)
    c.contacts.preference(event, "alice", "less")
    assert c.contacts.summary(GROUP, "alice", c.now[0])["interval"] == changed


def test_active_normal_dialogue_can_increase_frequency_without_charging_quota(control):
    c = control
    before = c.contacts.summary(UMO, "alice", c.now[0])
    c.contacts.normal_reply(Event(), "alice", c.now[0])
    c.now[0] += timedelta(minutes=5)
    c.contacts.observe(Event(message_id="2"), "alice", c.now[0], targeted=True)
    after = c.contacts.summary(UMO, "alice", c.now[0])
    assert after["interval"] < before["interval"]
    assert after["today"] == {"ordinary": 0, "agreed": 0}


async def test_private_conversation_thought_never_leaks_to_another_session(control):
    c = control
    await c.temporal.save_thought("alice", "来自私聊的想法", {"activity": "阅读"}, scope=UMO)
    assert c.temporal.thought(UMO, "alice")
    assert c.temporal.thought("qq:FriendMessage:99", "alice") is None


async def test_time_guard_buffers_reply_before_host_streams_it(control):
    event = Event(text="你说六小时后做完，现在好了吗")
    await control.temporal.prepare_event(event)
    assert event.get_extra("enable_streaming") is False
    assert event.get_extra("decision_temporal_guard") is True


async def test_bot_cannot_claim_user_reservation_without_actual_user_request(control):
    event = Event(text="明天我有考试")
    with pytest.raises(ValueError, match="真实来信"):
        await control.temporal.manage(event, action="create", delay_seconds=3600, note="惊喜", origin="user")


async def test_failed_activity_generation_is_budgeted_and_not_retried_for_same_node(control):
    c = control
    from decision.virtual_life.core.models import DailyPlan, TimelineItem

    plan = DailyPlan(date=c.now[0].date().isoformat(), persona_id="alice", theme="阅读", mood="轻松",
        outfit=None, timeline=(TimelineItem(id="reading", start="00:00", end="24:00", activity="读书"),))
    persona = SimpleNamespace(id="alice", prompt="Alice爱读书")
    c.provider.text_chat.side_effect = TimeoutError()
    with pytest.raises(TimeoutError):
        await c.temporal.activity_thought(UMO, persona, plan)
    await c.temporal.activity_thought(UMO, persona, plan)
    assert c.provider.text_chat.await_count == 1


async def test_synthetic_proactive_event_cannot_force_preferences_or_use_user_quota(control):
    event = Event()
    event.set_extra("decision_life_proactive", True)
    with pytest.raises(ValueError, match="无权"):
        control.contacts.preference(event, "alice", "more")
    with pytest.raises(ValueError, match="冒充"):
        await control.temporal.manage(event, action="create", delay_seconds=3600, note="自发惊喜", origin="user")


async def test_bot_surprise_uses_decision_time_without_needing_to_reveal_it(control):
    c = control
    event = Event(text="最近挺忙的")
    result = json.loads(await c.temporal.manage(event, action="create", delay_seconds=172800, note="分享一个读书小惊喜", origin="bot"))
    task = c.ledger.get_task(result["task"]["id"])
    assert task["status"] == "pending"
    assert task["due"] == c.now[0].timestamp() + 172800
    with pytest.raises(ValueError, match="间隔"):
        await c.temporal.manage(event, action="create", delay_seconds=60, note="不允许绕过频率", origin="bot")


async def test_reserving_future_day_does_not_pretend_current_reply_window_expired(control):
    c = control
    c.contacts.delivered(UMO, "alice", c.now[0])
    before = c.contacts.summary(UMO, "alice", c.now[0])["interval"]
    await c.temporal.manage(Event(), action="create", delay_seconds=172800, note="未来提醒", origin="user")
    after = c.contacts.summary(UMO, "alice", c.now[0])
    assert after["interval"] == before and not after["negative_applied"]


async def create(c, **kwargs):
    result = await c.temporal.manage(Event(), action="create", delay_seconds=6 * 3600, note="提醒看书", **kwargs)
    return json.loads(result)["task"]["id"]


async def test_task_reservation_exact_time_duplicate_callback_and_native_scheduler(control, monkeypatch):
    c = control
    ident = await create(c)
    assert c.runtime.context.cron_manager.add_basic_job.await_count == 1
    send = AsyncMock()
    monkeypatch.setattr("decision.life_agent.send_prepared_contact", send)
    await c.temporal.prepare(ident, 1)
    assert c.provider.text_chat.await_count == 1
    c.now[0] += timedelta(hours=3)
    facts = json.loads(c.temporal.describe(UMO, "alice", c.now[0]))
    assert facts["tasks"][0]["remaining_seconds"] == 10800
    assert "尚未" in facts["tasks"][0]["progress"]
    await c.temporal.execute(ident, 1)
    assert send.await_count == 0
    c.now[0] += timedelta(hours=3)
    await asyncio.gather(c.temporal.execute(ident, 1), c.temporal.execute(ident, 1), c.temporal.execute(ident, 1))
    assert send.await_count == 1
    assert c.provider.text_chat.await_count == 1
    assert c.ledger.get_task(ident)["status"] == "delivered"
    summary = c.contacts.summary(UMO, "alice", c.now[0])
    assert summary["today"] == {"ordinary": 0, "agreed": 1}
    assert c.runtime.context.cron_manager.delete_job.await_count == 1


async def test_daily_reservation_cancel_and_scope(control):
    c = control
    ids = [await create(c) for _ in range(3)]
    with pytest.raises(ValueError, match="额度"):
        await create(c)
    with pytest.raises(ValueError, match="没有该任务"):
        await c.temporal.manage(Event("qq:FriendMessage:99"), action="cancel", task_id=ids[0])
    await c.temporal.manage(Event(), action="cancel", task_id=ids[0])
    assert await create(c)
    assert c.ledger.get_task(ids[0])["status"] == "cancelled"


async def test_bot_relative_time_starts_on_actual_delivery_and_no_budget_bypass(control):
    c = control
    event = Event()
    result = json.loads(await c.temporal.manage(event, action="create", delay_seconds=21600, note="读完小说", kind="virtual_finish", origin="bot", notify=False))
    ident = result["task"]["id"]
    assert c.ledger.get_task(ident)["status"] == "awaiting_delivery"
    c.now[0] += timedelta(minutes=5)
    c.provider.text_chat.return_value.completion_text = json.dumps({"commitments": [{"id": ident, "quote": "我六小时后读完小说"}]})
    await c.temporal.activate_from_event(event, "我六小时后读完小说")
    assert c.ledger.get_task(ident)["due"] == c.now[0].timestamp() + 21600
    c.now[0] += timedelta(hours=3)
    task = json.loads(c.temporal.describe(UMO, "alice", c.now[0]))["tasks"][0]
    assert task["remaining_seconds"] == 10800 and "尚未" in task["progress"]
    c.runtime.enabled = False
    assert "尚未" in c.temporal.describe(UMO, "alice", c.now[0])
    c.now[0] += timedelta(hours=3)
    await c.temporal.execute(ident, 1)
    assert c.ledger.get_task(ident)["status"] == "delivered"


async def test_persona_switch_suspend_recovery_and_unknown_send_no_retry(control, monkeypatch):
    c = control
    ident = await create(c)
    c.now[0] += timedelta(hours=6)
    c.runtime.personas.resolve.return_value.id = "bob"
    await c.temporal.execute(ident, 1)
    assert c.ledger.get_task(ident)["status"] == "suspended"
    c.runtime.personas.resolve.return_value.id = "alice"
    send = AsyncMock(side_effect=TimeoutError("platform acknowledgement lost"))
    monkeypatch.setattr("decision.life_agent.send_prepared_contact", send)
    with pytest.raises(TimeoutError):
        await c.temporal.execute(ident, 1)
    assert c.ledger.get_task(ident)["status"] == "unknown"
    await c.temporal.recover()
    assert send.await_count == 1


async def test_cancel_during_paid_preparation_discards_stale_result(control):
    c = control
    ident = await create(c)
    started, finish = asyncio.Event(), asyncio.Event()

    async def slow(**kw):
        started.set()
        await finish.wait()
        return SimpleNamespace(completion_text="旧提醒")

    c.provider.text_chat.side_effect = slow
    pending = asyncio.create_task(c.temporal.prepare(ident, 1))
    await started.wait()
    await c.temporal.manage(Event(), action="cancel", task_id=ident)
    finish.set()
    await pending
    assert c.ledger.get_task(ident)["status"] == "cancelled"
    assert not c.ledger.get_task(ident)["prepared"]


async def test_prepare_singleflight_does_not_wait_for_callers_long_lived_task(control):
    c = control
    ident = await create(c)
    started, finish, stay_alive = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def slow(**kw):
        started.set()
        await finish.wait()
        return SimpleNamespace(completion_text="准备好的提醒")

    async def worker():
        await c.temporal.prepare(ident, 1)
        await stay_alive.wait()

    c.provider.text_chat.side_effect = slow
    long_worker = asyncio.create_task(worker())
    await started.wait()
    second = asyncio.create_task(c.temporal.prepare(ident, 1))
    finish.set()
    await asyncio.wait_for(second, 1)
    assert c.provider.text_chat.await_count == 1
    assert not long_worker.done()
    stay_alive.set()
    await long_worker


async def test_guard_rewrites_false_completion_and_external_unconfirmed(control):
    c = control
    await create(c, kind="external_finish", notify=False)
    c.now[0] += timedelta(hours=3)
    c.provider.text_chat.return_value.completion_text = '{"conflict":true,"corrected":"还没好，约好六小时，目前只过了三小时。"}'
    response = SimpleNamespace(completion_text="已经完成了", tools_call_name=[])
    await c.temporal.guard_response(Event(), response)
    assert "还没好" in response.completion_text
    assert c.provider.text_chat.await_count == 1
    c.now[0] += timedelta(hours=3)
    facts = json.loads(c.temporal.describe(UMO, "alice", c.now[0]))
    assert "未经确认" in facts["tasks"][0]["progress"]
    c.provider.text_chat.side_effect = TimeoutError()
    response.completion_text = "完成啦"
    await c.temporal.guard_response(Event(), response)
    assert "未经确认" in response.completion_text


async def test_thought_ttl_scope_and_delivery_dedup(control):
    c = control
    ident = await c.temporal.save_thought("alice", "书的这个比喻很有趣", {"activity": "阅读", "at": c.now[0].isoformat()})
    assert c.temporal.thought(UMO, "alice")["id"] == ident
    assert c.temporal.thought(UMO, "bob") is None
    c.temporal.mark_shared(ident, UMO)
    assert c.temporal.thought(UMO, "alice") is None
    assert c.temporal.thought("qq:FriendMessage:99", "alice")["id"] == ident
    c.now[0] += timedelta(hours=7)
    assert c.temporal.thought("qq:FriendMessage:99", "alice") is None


@pytest.mark.parametrize("field,value", [("friend_min_minutes", 3000), ("positive_factor", 1), ("group_adjust_users", ["not-a-qq"]), ("negative_factor", float("nan"))])
def test_invalid_control_settings_rejected(field, value):
    settings = life_defaults()
    settings["adaptive_settings"][field] = value
    with pytest.raises(ValueError):
        normalize_life(settings)
