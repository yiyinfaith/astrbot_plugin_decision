"""0.4.11: lighter settings page, contact brief, day rhythm, weather bounds, wake control."""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from decision.input_enhancement.group_wake_suppression import (
    WAKE_SUPPRESSED_EXTRA,
    GroupWakeSuppressionModule,
)
from decision.input_enhancement.tool_history_context import (
    TOOL_HISTORY_PLACEHOLDER,
    ToolHistoryContextModule,
    sanitize_messages,
)
from decision.life_config import INTERNAL_SETTINGS, life_defaults, life_schema
from decision.life_contacts import ContactControl
from decision.life_ledger import LifeLedger

UMO = "qq:FriendMessage:42"


class _Event:
    def __init__(self, extras=None):
        self.extras = dict(extras or {})
        self.is_at_or_wake_command = True

    def is_stopped(self):
        return False

    def get_extra(self, key, default=None):
        return self.extras.get(key, default)

    def set_extra(self, key, value):
        self.extras[key] = value


def _contacts(tmp_path, **human):
    config = life_defaults()
    config["humanization_settings"].update(human)
    runtime = SimpleNamespace(
        config=config,
        _now=lambda: datetime(2026, 11, 3, 12, 0).astimezone(),
        policy=SimpleNamespace(settings_for=lambda umo: config["friend_settings"]),
    )
    return ContactControl(runtime, LifeLedger(tmp_path / "ledger.sqlite3"))


# ----- settings --------------------------------------------------------------


def test_internal_settings_exist_and_keep_working():
    schema = life_schema()
    for group, keys in INTERNAL_SETTINGS.items():
        for key in keys:
            assert schema[group]["items"][key]["internal"] is True
    assert schema["reply_delay_settings"]["items"]["notify_min_seconds"]["internal"] is True
    assert life_defaults()["weather_settings"]["cache_minutes"] == 60


# ----- what the models are told ---------------------------------------------


def test_brief_drops_bookkeeping_and_reports_elapsed_minutes(tmp_path):
    contacts = _contacts(tmp_path)
    now = datetime(2026, 11, 3, 12, 0).astimezone()
    summary = contacts.summary(UMO, "alice", now)
    summary.update(
        last_sent=(now - timedelta(minutes=90)).timestamp(),
        unanswered=2,
        paused_until=-1,
        defer_until=(now + timedelta(minutes=30)).timestamp(),
        member_busy={
            "a": (now + timedelta(minutes=10)).timestamp(),
            "b": (now - timedelta(minutes=10)).timestamp(),
        },
    )
    brief = contacts.brief(summary, now)
    assert brief["minutes_since_sent"] == 90
    assert brief["paused"] == "until_resumed"
    assert brief["busy_minutes"] == 30
    assert brief["member_busy_minutes"] == {"a": 10}
    assert brief["unanswered"] == 2
    assert brief["today"] == {"ordinary": 0, "agreed": 0}
    for field in (
        "relationship_updated_at",
        "recent_intents",
        "baseline",
        "umo",
        "persona",
        "paused_until",
        "defer_until",
        "member_busy",
    ):
        assert field not in brief
    assert "minutes_since_incoming" not in brief


def test_social_signals_follow_their_switches(tmp_path):
    summary = {"relationship": {"stage": "熟悉"}, "response_rhythm": {"bucket": "fast"}}
    assert _contacts(tmp_path).social(summary) == {
        "relationship": "熟悉",
        "response_rhythm": "fast",
    }
    assert _contacts(tmp_path, relationship_memory_enable=False).social(summary) == {
        "response_rhythm": "fast"
    }
    assert _contacts(tmp_path, enable=False).social(summary) == {}


def test_day_factor_is_stable_within_a_day_and_varies_across_days(tmp_path):
    contacts = _contacts(tmp_path)
    base = datetime(2026, 11, 3, 8, 0).astimezone()
    assert contacts._day_factor("alice", base) == contacts._day_factor(
        "alice", base + timedelta(hours=10)
    )
    factors = {contacts._day_factor("alice", base + timedelta(days=d)) for d in range(40)}
    assert len(factors) > 5
    assert all(0.75 <= factor <= 1.35 for factor in factors)
    assert _contacts(tmp_path, enable=False)._day_factor("alice", base) == 1.0


def test_day_rhythm_never_leaves_the_configured_interval_bounds(tmp_path):
    contacts = _contacts(tmp_path)
    low, high = contacts.bounds(UMO)
    base = datetime(2026, 11, 3, 8, 0).astimezone()
    days = [base + timedelta(days=d) for d in range(60)]
    at_low = {"interval": low, "baseline": low}
    at_high = {"interval": high, "baseline": high}
    assert all(contacts._day_interval(UMO, "alice", at_low, day) >= low for day in days)
    assert all(contacts._day_interval(UMO, "alice", at_high, day) <= high for day in days)


@pytest.mark.parametrize("feedback", ["more", "less", "set"])
def test_a_requested_frequency_is_left_exact(tmp_path, feedback):
    contacts = _contacts(tmp_path)
    data = {"interval": 360, "baseline": 360, "feedback": feedback}
    base = datetime(2026, 11, 3, 8, 0).astimezone()
    assert all(
        contacts._day_interval(UMO, "alice", data, base + timedelta(days=d)) == 360
        for d in range(30)
    )


def test_requested_frequency_survives_a_later_pause_and_resume(tmp_path):
    contacts = _contacts(tmp_path)
    now = datetime(2026, 11, 3, 8, 0).astimezone()
    data = {"interval": 360, "baseline": 360, "feedback": ""}
    contacts._feedback(data, UMO, "set", now, interval=360)
    contacts._feedback(data, UMO, "pause", now)
    contacts._feedback(data, UMO, "resume", now)
    assert data["feedback"] == "resume"
    assert contacts._day_interval(UMO, "alice", data, now + timedelta(days=1)) == 360


def test_brief_keeps_whose_signal_it_is_and_drops_stale_ones(tmp_path):
    contacts = _contacts(tmp_path)
    now = datetime(2026, 11, 3, 12, 0).astimezone()
    signal = {"kind": "busy", "quote": "在忙", "scope": "member", "sender": "x"}
    summary = {"engagement": {**signal, "at": (now - timedelta(minutes=5)).timestamp()}}
    assert contacts.brief(summary, now)["engagement"] == {
        "kind": "busy",
        "quote": "在忙",
        "scope": "member",
        "sender": "x",
        "minutes_ago": 5,
    }
    summary = {"engagement": {**signal, "at": (now - timedelta(days=2)).timestamp()}}
    assert "engagement" not in contacts.brief(summary, now)


def test_brief_reports_todays_effective_interval(tmp_path):
    contacts = _contacts(tmp_path)
    now = datetime(2026, 11, 3, 12, 0).astimezone()
    assert contacts.brief({"interval": 240, "today_interval": 300.4}, now)["interval"] == 300


def test_day_rhythm_cannot_undercut_the_baseline_floor(tmp_path):
    contacts = _contacts(tmp_path)
    data = {"interval": 320, "baseline": 400}
    base = datetime(2026, 11, 3, 8, 0).astimezone()
    assert all(
        contacts._day_interval(UMO, "alice", data, base + timedelta(days=d)) >= 320
        for d in range(60)
    )


# ----- group wake control ---------------------------------------------------


def test_suppression_marks_the_event_for_decisions_own_paths():
    module = GroupWakeSuppressionModule(logger=None)
    event = _Event({"handlers_parsed_params": {}, "activated_handlers": []})
    module.apply_suppression(event, {"remove_empty_mention": False})
    assert event.is_at_or_wake_command is False
    assert event.extras[WAKE_SUPPRESSED_EXTRA] is True


def test_suppression_leaves_a_recognised_command_alone():
    module = GroupWakeSuppressionModule(logger=None)
    handler = SimpleNamespace(handler_full_name="demo")
    event = _Event({"handlers_parsed_params": {"demo": {}}, "activated_handlers": [handler]})
    module.apply_suppression(event, {"remove_empty_mention": False})
    assert event.is_at_or_wake_command is True
    assert WAKE_SUPPRESSED_EXTRA not in event.extras


@pytest.mark.parametrize(
    ("blacklist", "groups", "expected"),
    [(True, [], True), (True, ["1"], False), (False, [], False), (False, ["1"], True)],
)
def test_wake_scope_follows_the_list_mode(blacklist, groups, expected):
    module = GroupWakeSuppressionModule(
        logger=None,
        disable_at_bot_wake=True,
        disable_at_bot_wake_scope_blacklist=blacklist,
        disable_at_bot_wake_group_ids=groups,
    )
    assert module._is_at_disabled_for_group("1") is expected


def test_direct_reply_and_summon_respect_a_suppressed_wake():
    from decision.plugin_routing import RoutingMixin

    class Host(RoutingMixin):
        def _setting(self, key, default=None):
            return default

        def _message_address_flags(self, event):
            return True, False, False, False

    event = _Event({WAKE_SUPPRESSED_EXTRA: True})
    event.get_message_outline = lambda: "[At:1] 你好"
    event.get_message_str = lambda: "你好"
    host = Host()
    assert host._direct_reply_requested(event) is False
    host._setting = lambda key, default=None: "" if key == "proactive_alias" else default
    assert host._proactive_summoned(event) is False
    del event.extras[WAKE_SUPPRESSED_EXTRA]
    assert host._direct_reply_requested(event) is True
    assert host._proactive_summoned(event) is True


# ----- tool history ---------------------------------------------------------


def test_tool_history_follows_the_session_scope():
    allowed = _Event()
    excluded = _Event()
    module = ToolHistoryContextModule(logger=None, applies=lambda event: event is allowed)
    assert module.applies_to_call(("stage", allowed), {}) is True
    assert module.applies_to_call(("stage", excluded), {}) is False
    assert module.applies_to_call((), {"event": excluded}) is False
    assert ToolHistoryContextModule(logger=None).applies_to_call(("stage", excluded), {}) is True


def test_tool_history_compresses_only_completed_rounds():
    pytest.importorskip("astrbot.core.agent.runners.tool_loop_agent_runner")
    messages = [
        {"role": "user", "content": "查一下"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "a"}]},
        {"role": "tool", "tool_call_id": "a", "content": "很长的结果"},
        {"role": "assistant", "content": "结论"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "b"}]},
        {"role": "tool", "tool_call_id": "b", "content": "进行中的结果"},
    ]
    sanitized, changed = sanitize_messages(messages)
    assert changed
    assert sanitized[2]["content"] == TOOL_HISTORY_PLACEHOLDER
    assert sanitized[5]["content"] == "进行中的结果"
    assert messages[2]["content"] == "很长的结果"
