from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from decision.virtual_life.core.models import SessionState
from decision.virtual_life.core.storage import PluginStorage, session_key

from .test_life_control import GROUP, UMO, Event, control


@pytest.fixture
def human_control(tmp_path):
    # Reuse the established deterministic controller fixture without making
    # this regression module depend on pytest's per-module fixture discovery.
    return control.__wrapped__(tmp_path)


def test_session_state_isolated_by_umo_and_persona_and_legacy_migrates(tmp_path):
    path = Path(tmp_path) / "virtual-life"
    storage = PluginStorage(path)
    storage.sessions[UMO] = SessionState(
        date="2026-11-01",
        persona_id="alice",
        daily_budget=2,
        sent_count=1,
        unanswered_count=2,
    )
    storage.sessions[session_key(UMO, "bob")] = SessionState(
        date="2026-11-01",
        persona_id="bob",
        daily_budget=5,
        sent_count=0,
    )
    import asyncio

    asyncio.run(storage.save_sessions())

    restored = PluginStorage(path)
    asyncio.run(restored.load())
    assert restored.sessions.get_for(UMO, "alice").sent_count == 1
    assert restored.sessions.get_for(UMO, "alice").unanswered_count == 2
    assert restored.sessions.get_for(UMO, "bob").daily_budget == 5
    assert session_key(UMO, "alice") in restored.sessions
    assert UMO not in restored.sessions  # legacy bare keys are rewritten


def test_policy_does_not_reuse_daily_budget_after_persona_switch(tmp_path):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from decision.virtual_life.core.models import DailyPlan
    from decision.virtual_life.core.proactive import ProactivePolicy
    from tests.virtual_life.fixtures import outfit_payload

    storage = PluginStorage(Path(tmp_path) / "policy")
    config = {
        "friend_settings": {
            "enable": True,
            "session_blacklist": True,
            "session_list": [],
            "daily_budget_min": 2,
            "daily_budget_max": 2,
            "llm_bonus_max": 0,
            "daily_hard_max": 5,
            "cooldown_minutes": 0,
        },
        "group_settings": {
            "enable": True,
            "session_blacklist": True,
            "session_list": [],
            "daily_budget_min": 2,
            "daily_budget_max": 2,
            "llm_bonus_max": 0,
            "daily_hard_max": 5,
            "cooldown_minutes": 0,
        },
        "delivery_settings": {
            "max_unanswered": 3,
            "minimum_idle_for_window_minutes": 20,
            "availability_probabilities": {"blocked": 0, "low": 1, "normal": 1, "high": 1},
        },
    }
    plan = DailyPlan.from_dict(
        {
            "date": "2026-11-01",
            "persona_id": "alice",
            "theme": "日常",
            "mood": "平静",
            "outfit": outfit_payload(),
            "timeline": [
                {
                    "id": "all",
                    "start": "00:00",
                    "end": "24:00",
                    "activity": "阅读",
                    "state": "available",
                    "availability": "normal",
                }
            ],
            "proactive_windows": [],
            "budget_bonus": {},
        }
    )
    policy = ProactivePolicy(config, storage, ZoneInfo("Asia/Shanghai"))
    now = datetime(2026, 11, 1, 12, tzinfo=ZoneInfo("Asia/Shanghai"))
    alice = policy.ensure_state(UMO, "alice", plan, now)
    alice.sent_count = 2
    bob = policy.ensure_state(UMO, "bob", plan, now)
    assert bob is not alice and bob.sent_count == 0 and bob.unanswered_count == 0


def test_humanization_relationship_rhythm_and_intent_cooldown(human_control):
    c = human_control
    c.cfg["humanization_settings"].update(
        enable=True,
        relationship_memory_enable=True,
        content_diversity_enable=True,
        intent_cooldown_minutes=60,
    )
    now = c.now[0]
    c.contacts.record_intent(UMO, "alice", "根据联系偏好自然联系", "idle", now)
    allowed, reason = c.contacts.intent_allowed(
        UMO, "alice", "会话沉默了一段时间，希望自然地重新联系", "idle", now
    )
    assert not allowed and reason == "同类主动内容冷却中"
    c.now[0] += timedelta(hours=2)
    assert c.contacts.intent_allowed(
        UMO, "alice", "会话沉默了一段时间，希望自然地重新联系", "idle", c.now[0]
    )[0]

    c.contacts.delivered(UMO, "alice", c.now[0])
    c.now[0] += timedelta(minutes=5)
    c.contacts.observe(Event(text="我回来了"), "alice", c.now[0], targeted=True)
    summary = c.contacts.summary(UMO, "alice", c.now[0])
    assert summary["response_rhythm"]["bucket"] == "fast"
    assert summary["relationship"]["stage"] in {"初识", "熟悉", "稳定"}


def test_group_unrelated_feedback_does_not_change_relationship(human_control):
    c = human_control
    before = c.contacts.summary(GROUP, "alice", c.now[0])["relationship"]["score"]
    c.contacts.observe(
        Event(umo=GROUP, text="大家晚上好", sender="99"), "alice", c.now[0], targeted=False
    )
    after = c.contacts.summary(GROUP, "alice", c.now[0])["relationship"]["score"]
    assert after == before
