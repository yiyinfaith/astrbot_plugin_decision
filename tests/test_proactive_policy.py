from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from decision.virtual_life.core.proactive import ProactivePolicy, session_kind

FRIEND = "aiocqhttp:FriendMessage:42"
GROUP = "aiocqhttp:GroupMessage:99"
OTHER_FRIEND = "aiocqhttp:FriendMessage:77"
OTHER_GROUP = "aiocqhttp:GroupMessage:88"
TZ = ZoneInfo("Asia/Shanghai")


def build(*, friend=None, group=None, sessions=None):
    config = {
        "friend_settings": {
            "enable": True,
            "session_blacklist": False,
            "session_list": [],
            "daily_budget_min": 0,
            "daily_budget_max": 0,
            "llm_bonus_max": 0,
            "daily_hard_max": 0,
            "cooldown_minutes": 0,
        },
        "group_settings": {
            "enable": True,
            "session_blacklist": False,
            "session_list": [],
            "daily_budget_min": 0,
            "daily_budget_max": 0,
            "llm_bonus_max": 0,
            "daily_hard_max": 0,
            "cooldown_minutes": 0,
        },
        "delivery_settings": {
            "max_unanswered": 3,
            "minimum_idle_for_window_minutes": 20,
            "availability_probabilities": {
                "blocked": 0.0,
                "low": 0.25,
                "normal": 0.7,
                "high": 1.0,
            },
        },
    }
    if friend is not None:
        config["friend_settings"].update(friend)
    if group is not None:
        config["group_settings"].update(group)
    storage = SimpleNamespace(sessions=dict(sessions or {}))
    return ProactivePolicy(config, storage, TZ)


def test_session_kind_is_case_insensitive():
    assert session_kind(GROUP) == "group"
    assert session_kind(FRIEND) == "friend"
    assert session_kind("aiocqhttp:groupmessage:1") == "group"


def test_whitelist_empty_list_disables_everything():
    policy = build()
    assert policy.is_enabled(FRIEND) is False
    assert policy.is_enabled(GROUP) is False
    assert policy.enabled_sessions() == []


def test_whitelist_membership():
    policy = build(friend={"session_list": [FRIEND]})
    assert policy.is_enabled(FRIEND) is True
    assert policy.is_enabled(OTHER_FRIEND) is False
    assert policy.is_enabled(GROUP) is False


def test_whitelist_does_not_mix_kinds():
    policy = build(friend={"session_list": [GROUP]}, group={"session_list": [FRIEND]})
    # A group UMO in the friend list must not enable it as a friend session.
    assert policy.is_enabled(GROUP) is False
    assert policy.is_enabled(FRIEND) is False


def test_enable_false_always_disables():
    policy = build(friend={"enable": False, "session_list": [FRIEND]})
    assert policy.is_enabled(FRIEND) is False
    assert policy.enabled_sessions() == []


def test_enabled_sessions_returns_only_matching_kind():
    policy = build(
        friend={"session_list": [FRIEND, OTHER_FRIEND]},
        group={"session_list": [GROUP]},
    )
    sessions = policy.enabled_sessions()
    assert FRIEND in sessions
    assert OTHER_FRIEND in sessions
    assert GROUP in sessions
    assert len(sessions) == 3


def test_blacklist_empty_list_allows_everything():
    policy = build(
        friend={"session_blacklist": True},
        group={"session_blacklist": True},
    )
    assert policy.is_enabled(FRIEND) is True
    assert policy.is_enabled(GROUP) is True
    assert policy.is_enabled(OTHER_GROUP) is True


def test_blacklist_membership():
    policy = build(friend={"session_blacklist": True, "session_list": [FRIEND]})
    assert policy.is_enabled(FRIEND) is False
    assert policy.is_enabled(OTHER_FRIEND) is True


def test_blacklist_enabled_sessions_includes_known_runtime_sessions():
    """Blacklist mode must fall back to known sessions, never return empty."""

    policy = build(
        friend={"session_blacklist": True},
        group={"session_blacklist": True},
        sessions={FRIEND: object(), GROUP: object()},
    )
    sessions = policy.enabled_sessions()
    assert FRIEND in sessions
    assert GROUP in sessions


def test_blacklist_enabled_sessions_excludes_listed_sessions():
    policy = build(
        friend={"session_blacklist": True, "session_list": [FRIEND]},
        group={"session_blacklist": True},
        sessions={FRIEND: object(), OTHER_FRIEND: object(), GROUP: object()},
    )
    sessions = policy.enabled_sessions()
    assert FRIEND not in sessions
    assert OTHER_FRIEND in sessions
    assert GROUP in sessions


def test_blacklist_does_not_leak_across_kinds():
    policy = build(
        friend={"session_blacklist": True},
        group={"enable": False, "session_blacklist": True},
        sessions={FRIEND: object(), GROUP: object()},
    )
    sessions = policy.enabled_sessions()
    assert FRIEND in sessions
    # group_settings.enable is False, so no group session may appear.
    assert GROUP not in sessions


def test_settings_for_picks_group_and_friend():
    policy = build()
    assert policy.settings_for(GROUP) is policy.config["group_settings"]
    assert policy.settings_for(FRIEND) is policy.config["friend_settings"]


def test_evaluate_rejects_disabled_session():
    policy = build()
    decision = policy.evaluate(
        umo=FRIEND,
        state=SimpleNamespace(unanswered_count=0, sent_count=0, last_proactive_at=""),
        current_item=None,
        now=datetime(2026, 7, 14, 12, 0, tzinfo=TZ),
        trigger="idle",
    )
    assert decision.allowed is False
    assert decision.reason == "session disabled or not whitelisted"


def test_evaluate_allows_blacklisted_mode_unlisted_session():
    policy = build(friend={"session_blacklist": True})
    decision = policy.evaluate(
        umo=FRIEND,
        state=SimpleNamespace(
            date="2026-07-14",
            unanswered_count=0,
            sent_count=0,
            daily_budget=1,
            last_proactive_at="",
            last_user_message_at="",
        ),
        current_item=None,
        now=datetime(2026, 7, 14, 12, 0, tzinfo=TZ),
        trigger="idle",
    )
    assert decision.allowed is True
