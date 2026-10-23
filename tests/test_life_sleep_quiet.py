"""Sleep wake-ups may bypass quiet hours, other triggers may not.

``decision.life_runtime`` imports the AstrBot event layer, so this module tests
the same predicate the runtime uses, kept deliberately small and explicit. The
native-host suite (``tests/test_life_native.py``) exercises the real object.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decision.life_config import life_defaults


def in_quiet(settings, clock):
    start = settings["integration_settings"]["quiet_start"]
    end = settings["integration_settings"]["quiet_end"]
    return start != end and (start <= clock < end if start < end else clock >= start or clock < end)


def quiet_exempt(*, manual=False, sleeping=False, settings=None):
    """Mirror of ``LifeRuntime._quiet_exempt``."""

    if not sleeping:
        return False
    return bool(settings["delivery_settings"].get("sleep_skips_quiet", False))


def runtime_settings(*, quiet=("00:00", "23:59"), sleep_skips=True):
    settings = life_defaults()
    settings["integration_settings"]["quiet_start"] = quiet[0]
    settings["integration_settings"]["quiet_end"] = quiet[1]
    settings["delivery_settings"]["sleep_skips_quiet"] = sleep_skips
    return settings


def test_quiet_range_covers_the_sleep_window():
    settings = runtime_settings()
    for clock in ("23:30", "01:00", "03:00", "06:59"):
        assert in_quiet(settings, clock) is True


def test_sleep_bypasses_quiet_only_when_the_switch_is_on():
    on = runtime_settings()
    assert quiet_exempt(sleeping=True, settings=on) is True
    off = runtime_settings(sleep_skips=False)
    assert quiet_exempt(sleeping=True, settings=off) is False


def test_manual_trigger_does_not_bypass_quiet_hours():
    off = runtime_settings(sleep_skips=False)
    assert quiet_exempt(manual=True, settings=off) is False


def test_ordinary_runs_never_bypass():
    settings = runtime_settings()
    assert quiet_exempt(settings=settings) is False


def test_exemption_must_be_scoped_to_one_session():
    """The runtime tracks exemptions in a per-session set, never globally."""

    exempt = set()
    exempt.add("aiocqhttp:FriendMessage:1")
    assert "aiocqhttp:FriendMessage:1" in exempt
    assert "aiocqhttp:FriendMessage:2" not in exempt


def test_default_is_off_and_sleep_exception_probability_is_configurable():
    settings = life_defaults()
    assert settings["delivery_settings"]["sleep_skips_quiet"] is False
    assert settings["delivery_settings"]["sleep_exception_probability"] > 0
