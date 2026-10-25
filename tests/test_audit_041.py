"""Regressions found by the 0.4.1 audit; assert real state and call counts."""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import timedelta

import pytest

from .test_life_continuity import continuity, source, topic_data  # noqa: F401
from .test_life_control import UMO, Event, control  # noqa: F401


@pytest.fixture
def audit(request):
    return request.getfixturevalue("continuity")


async def task(c, **extra):
    data = json.loads(await c.temporal.manage(Event(), action="create", delay_seconds=21600, note="外部处理", **extra))
    return data["task"]["id"]


async def test_persona_mismatch_never_spends_prepare_attempts(audit):
    c = audit
    ident = await task(c)
    c.runtime.personas.resolve.return_value.id = "bob"
    for _ in range(5):
        await c.temporal.prepare(ident, 1)
    assert c.provider.text_chat.await_count == 0
    assert c.ledger.get_task(ident)["attempts"] == 0
    c.runtime.personas.resolve.return_value.id = "alice"
    await c.temporal.prepare(ident, 1)
    assert c.provider.text_chat.await_count == 1
    assert c.ledger.get_task(ident)["status"] == "ready"


async def test_disabled_preparation_is_free_and_can_recover(audit):
    c = audit
    ident = await task(c)
    c.runtime.enabled = False
    await c.temporal.prepare(ident, 1)
    assert c.provider.text_chat.await_count == 0
    assert c.ledger.get_task(ident)["attempts"] == 0
    c.runtime.enabled = True
    await c.temporal.prepare(ident, 1)
    assert c.provider.text_chat.await_count == 1


async def test_cancel_while_waiting_for_model_slot_never_submits_request(audit):
    c = audit
    ident = await task(c)
    c.temporal.model_slots = asyncio.Semaphore(0)
    pending = asyncio.create_task(c.temporal.prepare(ident, 1))
    for _ in range(100):
        if c.ledger.get_task(ident)["status"] == "preparing":
            break
        await asyncio.sleep(.001)
    await c.temporal.manage(Event(), action="cancel", task_id=ident)
    c.temporal.model_slots.release()
    await asyncio.wait_for(pending, 1)
    assert c.provider.text_chat.await_count == 0
    assert c.ledger.get_task(ident)["status"] == "cancelled"
    assert c.ledger.get_task(ident)["attempts"] == 0


async def test_edit_clears_old_prepare_backoff_and_cached_text(audit):
    c = audit
    ident = await task(c)
    c.temporal.update(ident, 1, next_prepare_at=c.now[0].timestamp() + 86400, error="old failure", prepared="过时正文")
    await c.temporal.manage(Event(), action="edit", task_id=ident, delay_seconds=60, note="新提醒")
    await c.temporal.prepare(ident, 2)
    stored = c.ledger.get_task(ident)
    assert stored["next_prepare_at"] == 0 and stored["error"] == ""
    assert stored["status"] == "ready" and stored["prepared"] != "过时正文"
    assert c.provider.text_chat.await_count == 1


async def test_external_question_is_not_confirmation_and_proven_completion_invalidates_text(audit):
    c = audit
    ident = await task(c, kind="external_finish")
    await c.temporal.prepare(ident, 1)
    c.now[0] += timedelta(hours=6)
    c.provider.text_chat.return_value.completion_text = '{"confirmed":false,"quote":""}'
    with pytest.raises(ValueError, match="没有明确"):
        await c.temporal.manage(Event(text="外部处理做好了吗？"), action="complete", task_id=ident)
    assert not c.ledger.get_task(ident).get("external_confirmed")
    c.provider.text_chat.return_value.completion_text = '{"confirmed":true,"quote":"外部处理已经做好了"}'
    await c.temporal.manage(Event(text="外部处理已经做好了"), action="complete", task_id=ident)
    stored = c.ledger.get_task(ident)
    assert stored["external_confirmed"] is True
    assert stored["revision"] == 2 and stored["status"] == "pending" and stored["prepared"] == ""
    assert c.provider.text_chat.await_count == 3


def test_ambiguous_concatenations_cannot_overwrite_other_persona_topics(audit):
    c = audit
    a = source(c, "我明天有面试", sender="23")
    b = source(c, "我明天有面试", ident="s2", sender="3")
    c.runtime.conversation.apply(UMO, "alice1", [a], topic_data(c, a))
    c.runtime.conversation.apply(UMO, "alice12", [b], topic_data(c, b))
    first = c.runtime.conversation.topics(UMO, "alice1")
    second = c.runtime.conversation.topics(UMO, "alice12")
    assert len(first) == len(second) == 1 and first[0]["id"] != second[0]["id"]


def test_old_topic_ids_are_preserved_and_not_reopened(audit):
    c = audit
    src = source(c, "我明天有面试")
    data = topic_data(c, src)
    title = data["topics"][0]["title"]
    legacy = hashlib.sha256((UMO + "alice" + src["sender"] + title).encode()).hexdigest()
    with c.ledger.transaction() as db:
        db.execute("INSERT INTO topics VALUES (?,?,?,?,?,?,?,?)", (legacy, UMO, "alice", src["sender"], c.now[0].timestamp(), c.now[0].timestamp() + 86400, "asked", json.dumps({"title": title, "note": "已问过", "quote": src["text"]})))
    c.runtime.conversation.apply(UMO, "alice", [src], data)
    rows = c.runtime.conversation.topics(UMO, "alice")
    assert len(rows) == 1 and rows[0]["id"] == legacy and rows[0]["status"] == "asked"


@pytest.mark.parametrize("notify,delay", [("false", 1), (False, True), (True, "60")])
async def test_invalid_task_types_do_not_create_or_charge(audit, notify, delay):
    c = audit
    with pytest.raises(ValueError):
        await c.temporal.manage(Event(), action="create", delay_seconds=delay, note="不应创建", notify=notify)
    assert c.ledger.tasks() == [] and c.provider.text_chat.await_count == 0


async def test_commitment_waits_for_its_actual_text_and_uses_acknowledgement_time(audit):
    c = audit
    event = Event()
    ident = await task(c, origin="bot", kind="virtual_finish", notify=False)
    event.set_extra("decision_pending_commitments", [ident])
    c.provider.text_chat.return_value.completion_text = '{"commitments":[]}'
    await c.temporal.activate_from_event(event, "我先跟你说点别的。")
    assert c.ledger.get_task(ident)["status"] == "awaiting_delivery"
    c.now[0] += timedelta(hours=1)
    acknowledged_at = c.now[0]

    async def receipt(**kwargs):
        assert "我六小时后完成外部处理" in kwargs["prompt"]
        c.now[0] += timedelta(seconds=30)  # model latency cannot shift the promise
        return type("Response", (), {"completion_text": json.dumps({"commitments": [{"id": ident, "quote": "我六小时后完成外部处理"}]})})()

    c.provider.text_chat.side_effect = receipt
    await c.temporal.activate_from_event(event, "我六小时后完成外部处理")
    stored = c.ledger.get_task(ident)
    assert stored["status"] == "pending"
    assert stored["committed_at"] == acknowledged_at.timestamp()
    assert stored["due"] == acknowledged_at.timestamp() + 21600
    await c.temporal.activate_from_event(event, "其他内容")
    assert c.provider.text_chat.await_count == 2


async def test_failed_send_never_confirms_commitment_or_records_text(audit):
    from types import SimpleNamespace
    c = audit
    event = Event()
    ident = await task(c, origin="bot", kind="virtual_finish", notify=False)
    event.set_extra("decision_pending_commitments", [ident])

    async def failed(chain):
        event._has_send_oper = True
        raise RuntimeError("adapter failed after base hook")

    event.send = failed
    c.runtime.delivery.wrap_event(event, "alice")
    with pytest.raises(RuntimeError):
        await event.send(SimpleNamespace(chain=[SimpleNamespace(text="我六小时后完成外部处理")]))
    assert c.ledger.get_task(ident)["status"] == "awaiting_delivery"
    assert c.provider.text_chat.await_count == 0
    assert c.runtime.conversation.recent(UMO, "alice") == []


async def test_commitment_receipt_cannot_cross_scope_or_activate_cancelled_revision(audit):
    c = audit
    event = Event()
    ident = await task(c, origin="bot", kind="virtual_finish", notify=False)
    event.set_extra("decision_pending_commitments", [ident])
    c.runtime.personas.resolve.return_value.id = "bob"
    await c.temporal.activate_from_event(event, "我六小时后完成外部处理")
    assert c.provider.text_chat.await_count == 0
    c.runtime.personas.resolve.return_value.id = "alice"

    async def receipt(**kwargs):
        await c.temporal.manage(event, action="cancel", task_id=ident)
        return type("Response", (), {"completion_text": json.dumps({"commitments": [{"id": ident, "quote": "我六小时后完成外部处理"}]})})()

    c.provider.text_chat.side_effect = receipt
    await c.temporal.activate_from_event(event, "我六小时后完成外部处理")
    assert c.ledger.get_task(ident)["status"] == "cancelled"
    assert c.provider.text_chat.await_count == 1


@pytest.mark.parametrize("bad", [42, "invalid", {"unexpected": 1}, True])
def test_malformed_analysis_arrays_do_not_change_state(audit, bad):
    c = audit
    src = source(c)
    before = c.contacts.summary(UMO, "alice", c.now[0])
    c.runtime.conversation.apply(UMO, "alice", [src], {"signals": bad, "topics": bad})
    assert c.runtime.conversation.topics(UMO, "alice") == []
    assert c.contacts.summary(UMO, "alice", c.now[0]) == before


def test_malformed_quality_kind_does_not_crash_or_change_state(audit):
    c = audit
    src = source(c)
    c.runtime.conversation.apply(UMO, "alice", [src], {"signals": [{"source_id": src["id"], "quote": src["text"], "engagement": [], "confidence": .99}]})
    assert not c.contacts.summary(UMO, "alice", c.now[0]).get("engagement")


async def test_last_four_duplicate_baseline_survives_retention_without_time_limit(audit):
    c = audit
    for i in range(6):
        await c.runtime.delivery.record(UMO, "alice", f"old{i}", f"这是第{i}条足够长的真实主动发送消息，用于测试保留最近四条。")
        c.now[0] += timedelta(days=1)
    c.now[0] += timedelta(days=90)
    c.ledger.prune(c.now[0].timestamp())
    assert len(c.runtime.conversation.recent(UMO, "alice", 10)) == 4
    assert c.runtime.delivery.repeated(UMO, "alice", "这是第5条足够长的真实主动发送消息，用于测试保留最近四条。")
    assert not c.runtime.delivery.repeated(UMO, "bob", "这是第5条足够长的真实主动发送消息，用于测试保留最近四条。")


@pytest.mark.parametrize("invalid", [{}, {"conflict": "false"}, {"conflict": True, "corrected": {"text": "invalid"}}])
async def test_invalid_guard_schema_never_releases_unchecked_completion(audit, invalid):
    from types import SimpleNamespace
    c = audit
    await task(c, kind="external_finish")
    c.provider.text_chat.return_value.completion_text = json.dumps(invalid)
    response = SimpleNamespace(completion_text="外部处理已经全部完成！", tools_call_name=[])
    await c.temporal.guard_response(Event(), response)
    assert "已经全部完成" not in response.completion_text
    assert "尚未" in response.completion_text
    assert c.provider.text_chat.await_count == 1
