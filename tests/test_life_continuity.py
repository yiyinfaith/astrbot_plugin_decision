from __future__ import annotations

import asyncio
import json
from datetime import timedelta

import pytest

from decision.life_actions import perform_control_action
from decision.life_conversation import ConversationContinuity
from decision.life_delivery import LifeDelivery
from decision.life_ideas import LifeIdeas
from decision.life_state import PersonaLifeState
from decision.virtual_life.core.models import DailyPlan, TimelineItem

from .test_life_control import GROUP, UMO, Event, control  # noqa: F401


@pytest.fixture
async def continuity(request):
    c = request.getfixturevalue("control")
    c.runtime.closed = False
    c.runtime.contacts = c.contacts
    c.runtime.conversation = ConversationContinuity(c.runtime, c.ledger)
    c.runtime.ideas = LifeIdeas(c.runtime, c.ledger)
    c.runtime.delivery = LifeDelivery(c.runtime, c.ledger)
    c.runtime.life_state = PersonaLifeState(c.runtime, c.ledger)
    c.cfg["continuity_settings"]["analysis_interval_seconds"] = 0
    tasks = []

    def spawn(coro):
        task = asyncio.create_task(coro)
        tasks.append(task)
        return task

    c.runtime.spawn = spawn
    c.background = tasks
    yield c
    await c.runtime.conversation.close()
    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def source(c, text="我周五要去面试", *, ident="s1", sender="42"):
    return {"id": ident, "sender": sender, "at": c.now[0].isoformat(), "text": text, "can_adjust": False}


def topic_data(c, src, **extra):
    return {"signals": [], "topics": [{"source_id": src["id"], "quote": src["text"], "action": "upsert", "title": "面试结果", "note": "询问面试结果，不能假定成功", "follow_up_at": (c.now[0] + timedelta(hours=3)).isoformat(), **extra}]}


def test_neutral_closing_and_quality_are_not_message_count(continuity):
    c = continuity
    before = c.contacts.summary(UMO, "alice", c.now[0])["interval"]
    c.contacts.normal_reply(Event(), "alice", c.now[0])
    c.contacts.observe(Event(text="嗯嗯"), "alice", c.now[0], targeted=True, quality_pending=True)
    assert c.contacts.summary(UMO, "alice", c.now[0])["interval"] == before
    src = source(c, "嗯嗯")
    c.runtime.conversation.apply(UMO, "alice", [src], {"signals": [{"source_id": "s1", "quote": "嗯嗯", "engagement": "neutral", "confidence": .95}]})
    assert c.contacts.summary(UMO, "alice", c.now[0])["interval"] == before
    src = source(c, "这个话题太有意思了，你再给我讲讲", ident="s2")
    signal = {"source_id": "s2", "quote": src["text"], "engagement": "engaged", "confidence": .95}
    c.runtime.conversation.apply(UMO, "alice", [src], {"signals": [signal]})
    assert c.contacts.summary(UMO, "alice", c.now[0])["interval"] < before
    after = c.contacts.summary(UMO, "alice", c.now[0])["interval"]
    c.now[0] += timedelta(hours=7)
    c.runtime.conversation.apply(UMO, "alice", [src], {"signals": [signal]})
    assert c.contacts.summary(UMO, "alice", c.now[0])["interval"] == after


def test_temporary_busy_is_bounded_and_group_member_cannot_pause_the_group(continuity):
    c = continuity
    src = source(c, "我现在忙，晚点再聊")
    signal = {"source_id": "s1", "quote": src["text"], "engagement": "busy", "confidence": .99, "busy_minutes": 999999}
    c.runtime.conversation.apply(UMO, "alice", [src], {"signals": [signal]})
    profile = c.contacts.summary(UMO, "alice", c.now[0])
    assert profile["defer_until"] == c.now[0].timestamp() + 240 * 60
    assert not c.contacts.gate(UMO, "alice", c.now[0])[0]
    assert c.contacts.gate(UMO, "alice", c.now[0], agreed=True)[0]
    c.runtime.conversation.apply(GROUP, "alice", [src], {"signals": [signal]})
    group = c.contacts.summary(GROUP, "alice", c.now[0])
    assert not group.get("defer_until") and group["paused_until"] == 0
    assert group["member_busy"]["42"] > c.now[0].timestamp()


def test_topic_scope_resolution_expiry_and_member_ownership(continuity):
    c = continuity
    src = source(c)
    c.runtime.conversation.apply(UMO, "alice", [src], topic_data(c, src))
    topics = c.runtime.conversation.topics(UMO, "alice")
    assert len(topics) == 1 and topics[0]["quote"] == src["text"]
    assert c.runtime.conversation.topics(UMO, "bob") == []
    assert c.runtime.conversation.topics("qq:FriendMessage:99", "alice") == []
    assert c.runtime.conversation.due_topic(UMO, "alice") is None
    c.now[0] += timedelta(hours=3)
    assert c.runtime.conversation.due_topic(UMO, "alice")["id"] == topics[0]["id"]
    outcome = source(c, "面试已经通过了", ident="s2", sender="99")
    resolved = {"topics": [{"source_id": "s2", "quote": outcome["text"], "action": "resolve", "topic_id": topics[0]["id"]}]}
    c.runtime.conversation.apply(UMO, "alice", [outcome], resolved)
    assert c.runtime.conversation.topics(UMO, "alice")[0]["status"] == "open"
    outcome["sender"] = "42"
    c.runtime.conversation.apply(UMO, "alice", [outcome], resolved)
    assert c.runtime.conversation.topics(UMO, "alice")[0]["status"] == "resolved"
    assert c.runtime.conversation.due_topic(UMO, "alice") is None
    c.now[0] += timedelta(days=15)
    assert c.runtime.conversation.topics(UMO, "alice") == []


async def test_analysis_batches_dedup_sources_and_counts_real_provider_calls(continuity):
    c = continuity
    src = source(c)
    c.provider.text_chat.return_value.completion_text = json.dumps(topic_data(c, src))
    await asyncio.gather(c.runtime.conversation.analyze(UMO, "alice", [src]), c.runtime.conversation.analyze(UMO, "alice", [src]))
    assert c.provider.text_chat.await_count == 1
    assert len(c.runtime.conversation.topics(UMO, "alice")) == 1
    assert c.runtime.conversation.usage(UMO, "alice")["used"] == 1
    restored = ConversationContinuity(c.runtime, c.ledger)
    await restored.analyze(UMO, "alice", [src])
    assert c.provider.text_chat.await_count == 1
    c.cfg["continuity_settings"]["max_semantic_calls_day"] = 1
    await restored.analyze(UMO, "alice", [source(c, ident="s3")])
    assert c.provider.text_chat.await_count == 1


async def test_offered_idea_is_not_consumed_by_an_unrelated_actual_message(continuity):
    c = continuity
    ident = await c.runtime.ideas.save_thought("alice", "阅读像是用别人的眼睛重新观察世界", {"activity": "读书"})
    idea = c.runtime.ideas.thought(UMO, "alice")
    c.provider.text_chat.return_value.completion_text = '{"matches":[]}'
    await c.runtime.delivery.record(UMO, "alice", "out-1", "晚上好，今天过得怎么样？", ideas=[idea])
    await asyncio.gather(*c.background)
    assert c.runtime.ideas.thought(UMO, "alice")["id"] == ident
    assert c.provider.text_chat.await_count == 1
    await c.runtime.delivery.confirm("out-1")
    assert c.provider.text_chat.await_count == 1
    await c.runtime.delivery.record(UMO, "alice", "out-2", idea["text"], ideas=[idea])
    await asyncio.gather(*c.background)
    assert c.runtime.ideas.thought(UMO, "alice") is None
    assert c.provider.text_chat.await_count == 1  # exact actual content needs no model
    assert c.runtime.conversation.recent(UMO, "bob") == []


async def test_paraphrase_receipt_requires_actual_quote_and_valid_offered_id(continuity):
    c = continuity
    ident = await c.runtime.ideas.save_thought("alice", "阅读像是用别人的眼睛重新观察世界", {"activity": "读书"}, scope=UMO)
    idea = c.runtime.ideas.thought(UMO, "alice")
    text = "看书的时候，好像也借到了另一双看世界的眼睛。"
    c.provider.text_chat.return_value.completion_text = json.dumps({"matches": [{"kind": "idea", "id": ident, "quote": text}]})
    await c.runtime.delivery.record(UMO, "alice", "out-3", text, ideas=[idea])
    await asyncio.gather(*c.background)
    assert c.runtime.ideas.thought(UMO, "alice") is None
    c.provider.text_chat.return_value.completion_text = json.dumps({"matches": [{"kind": "idea", "id": ident, "quote": text}]})
    await c.runtime.delivery.record("qq:FriendMessage:99", "alice", "cross-scope", text, ideas=[idea])
    await asyncio.gather(*c.background)
    with c.ledger.transaction() as db:
        assert not db.execute("SELECT 1 FROM shares WHERE thought=? AND umo=?", (ident, "qq:FriendMessage:99")).fetchone()


async def test_topic_question_is_recorded_only_after_actual_sending(continuity):
    c = continuity
    src = source(c)
    c.runtime.conversation.apply(UMO, "alice", [src], topic_data(c, src))
    c.now[0] += timedelta(hours=4)
    topic = c.runtime.conversation.due_topic(UMO, "alice")
    text = "你之前提到的面试怎么样了，顺利吗？"
    c.provider.text_chat.return_value.completion_text = json.dumps({"matches": [{"kind": "topic", "id": topic["id"], "quote": text}]})
    assert c.runtime.conversation.due_topic(UMO, "alice")
    await c.runtime.delivery.record(UMO, "alice", "follow-up", text, topics=[topic])
    await asyncio.gather(*c.background)
    assert c.runtime.conversation.due_topic(UMO, "alice") is None
    assert c.runtime.conversation.topics(UMO, "alice")[0]["status"] == "asked"


def test_life_progress_survives_restart_and_is_not_external_completion(continuity):
    c = continuity
    plan = DailyPlan(date=c.now[0].date().isoformat(), persona_id="alice", theme="阅读", mood="轻松", outfit=None,
        timeline=(TimelineItem(id="sleep", start="00:00", end="10:00", activity="休息", state="sleep"),
                  TimelineItem(id="reading", start="10:00", end="16:00", activity="读小说", state="focus"),
                  TimelineItem(id="rest", start="16:00", end="24:00", activity="休息")))
    first = c.runtime.life_state.current("alice", plan, c.now[0])
    assert first["progress"] == 0 and first["phase"] == "刚进入活动"
    c.now[0] += timedelta(hours=3)
    middle = c.runtime.life_state.current("alice", plan, c.now[0])
    assert middle["progress"] == .5
    restored = PersonaLifeState(c.runtime, c.ledger).current("alice", plan, c.now[0])
    assert restored == middle
    c.now[0] += timedelta(hours=3)
    end = c.runtime.life_state.current("alice", plan, c.now[0])
    assert end["activity"] == "休息"
    assert end["affect"]["feeling"] == "逐渐放松" and "读小说" in end["affect"]["cause"]
    assert any("不是外部完成凭据" in e["fact"] for e in end["events"])
    c.now[0] += timedelta(minutes=31)
    restored_mood = c.runtime.life_state.current("alice", plan, c.now[0])
    assert restored_mood["affect"]["feeling"] == "轻松"


async def test_webui_controls_cannot_cancel_other_sessions_or_invent_user_outcomes(continuity):
    c = continuity
    created = await perform_control_action(c.runtime, UMO, {"action": "create_task", "note": "喝水", "run_at": (c.now[0] + timedelta(hours=1)).isoformat()})
    task_id = created["task"]["id"]
    with pytest.raises(ValueError, match="没有该任务"):
        await perform_control_action(c.runtime, "qq:FriendMessage:99", {"action": "cancel_task", "task_id": task_id})
    await perform_control_action(c.runtime, UMO, {"action": "cancel_task", "task_id": task_id})
    assert c.ledger.get_task(task_id)["status"] == "cancelled"
    src = source(c)
    c.runtime.conversation.apply(UMO, "alice", [src], topic_data(c, src))
    topic = c.runtime.conversation.topics(UMO, "alice")[0]
    await perform_control_action(c.runtime, UMO, {"action": "resolve_topic", "topic_id": topic["id"]})
    done = c.runtime.conversation.topics(UMO, "alice")[0]
    assert done["outcome_source"] == "authenticated_webui" and done["outcome"] == "管理员标记结束"


async def test_model_session_identity_includes_persona_and_full_umo(continuity):
    c = continuity
    await c.temporal.model(UMO, "隔离A", persona_id="alice")
    await c.temporal.model(UMO, "隔离B", persona_id="bob")
    await c.temporal.model("other:FriendMessage:42", "隔离C", persona_id="alice")
    identities = [call.kwargs["session_id"] for call in c.provider.text_chat.call_args_list]
    assert len(set(identities)) == 3


async def test_webui_edit_preserves_non_notifying_commitment(continuity):
    c = continuity
    task = json.loads(await c.temporal.manage(Event(), action="create", delay_seconds=21600, note="读书进度", kind="virtual_finish", notify=False))["task"]
    await perform_control_action(c.runtime, UMO, {"action": "edit_task", "task_id": task["id"], "note": "读书进度补充", "run_at": task["run_at"]})
    stored = c.ledger.get_task(task["id"])
    assert stored["notify"] is False and stored["kind"] == "virtual_finish"


def test_duplicate_text_suppression_is_scoped(continuity):
    c = continuity
    text = "突然想到一个有趣的事，阅读好像也是另一种旅行。"
    with c.ledger.transaction() as db:
        db.execute("INSERT INTO utterances VALUES (?,?,?,?,?)", ("old", UMO, "alice", c.now[0].timestamp(), json.dumps({"text": text})))
    assert c.runtime.delivery.repeated(UMO, "alice", text)
    assert not c.runtime.delivery.repeated(UMO, "bob", text)
    assert not c.runtime.delivery.repeated("qq:FriendMessage:99", "alice", text)


async def test_prompt_history_only_contains_actual_text_not_unshared_candidates(continuity):
    c = continuity
    ident = await c.runtime.ideas.save_thought("alice", "不应该被冒充为已经说过的想法", {"activity": "阅读"})
    idea = c.runtime.ideas.thought(UMO, "alice")
    c.provider.text_chat.return_value.completion_text = '{"matches":[]}'
    await c.runtime.delivery.record(UMO, "alice", "actual-only", "今天过得怎么样？", ideas=[idea])
    await asyncio.gather(*c.background)
    history = c.runtime.conversation.context(UMO, "alice")
    assert "今天过得怎么样" in str(history)
    assert ident not in str(history) and idea["text"] not in str(history)
