"""Cover the midnight splitter and the six rule refinements."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decision.virtual_life.core.models import DailyPlan, TimelineItem  # noqa: E402


def sleep_item(**over):
    base = {
        "id": "s1",
        "start": "23:00",
        "end": "07:00",
        "activity": "sleep",
        "state": "sleep",
        "availability": "blocked",
    }
    base.update(over)
    return base


# --------------------------------------------------------------- midnight split
def test_a_crossing_item_is_split_into_two_ordered_segments():
    parts = TimelineItem.split_at_midnight(sleep_item())
    assert len(parts) == 2
    assert (parts[0].start, parts[0].end) == ("23:00", "24:00")
    assert (parts[1].start, parts[1].end) == ("00:00", "07:00")


def test_split_segments_keep_state_and_availability():
    parts = TimelineItem.split_at_midnight(sleep_item())
    for part in parts:
        assert part.state == "sleep"
        assert part.availability == "blocked"
        assert part.activity == "sleep"


def test_split_segments_get_distinct_ids():
    parts = TimelineItem.split_at_midnight(sleep_item())
    assert parts[0].id != parts[1].id
    assert all(part.id.startswith("s1") for part in parts)


def test_a_normal_item_is_not_split():
    parts = TimelineItem.split_at_midnight(
        {"id": "a", "start": "09:00", "end": "10:00", "activity": "class"}
    )
    assert len(parts) == 1
    assert parts[0].id == "a"


def test_an_item_already_ending_at_24_is_not_split():
    parts = TimelineItem.split_at_midnight(
        {"id": "c", "start": "23:00", "end": "24:00", "activity": "read"}
    )
    assert len(parts) == 1


def test_a_zero_length_item_still_raises():
    with pytest.raises(ValueError, match="earlier than end"):
        TimelineItem.split_at_midnight(
            {"id": "b", "start": "10:00", "end": "10:00", "activity": "x"}
        )


def test_split_preserves_order_within_a_plan():
    """A generated 23:00-07:00 sleep must yield a valid continuous day."""

    plan = DailyPlan.from_dict(
        {
            "date": "2026-10-22",
            "persona_id": "p",
            "timeline": [
                sleep_item(),
                {"id": "wake", "start": "07:00", "end": "08:00", "activity": "wake"},
                {"id": "day", "start": "08:00", "end": "23:00", "activity": "day"},
            ],
        }
    )
    starts = [item.start for item in plan.timeline]
    # The morning half sorts to the front of the day, the evening half to the end.
    assert starts == ["00:00", "07:00", "08:00", "23:00"], starts
    assert plan.timeline[0].end == "07:00"
    assert plan.timeline[-1].end == "24:00"
    # The invariant DailyPlan enforces still holds.
    previous = 0
    for item in plan.timeline:
        from decision.virtual_life.core.models import minute_of_day

        assert minute_of_day(item.start) == previous
        previous = minute_of_day(item.end)
    assert previous == 1440


def test_a_gap_in_the_source_timeline_is_still_rejected():
    with pytest.raises(ValueError, match="continuous"):
        DailyPlan.from_dict(
            {
                "date": "2026-10-22",
                "persona_id": "p",
                "timeline": [
                    {"id": "a", "start": "00:00", "end": "01:00", "activity": "a"},
                    {"id": "b", "start": "05:00", "end": "24:00", "activity": "b"},
                ],
            }
        )


# ------------------------------------------------------------------ wording/docs
def test_unanswered_threshold_is_described_as_consecutive():
    from decision.life_config import life_schema

    node = life_schema()["delivery_settings"]["items"]["max_unanswered"]
    assert node["description"] == "连续未回应上限"
    assert "清零" in node["hint"]


def test_cooldown_hint_explains_it_is_independent():
    from decision.life_config import life_schema

    schema = life_schema()
    for group in ("friend_settings", "group_settings"):
        cooldown = schema[group]["items"]["cooldown_minutes"]["hint"]
        idle = schema[group]["items"]["idle_min_minutes"]["hint"]
        assert "独立" in cooldown
        assert "冷却" in idle


def test_page_explains_silence_vs_cooldown():
    page = (Path(__file__).resolve().parents[1] / "pages/dashboard/index.html").read_text(
        encoding="utf-8"
    )
    assert page.count("life-rule-note") == 3, "note markup plus its CSS rule"
    assert "两个独立条件" in page


# ------------------------------------------------- sleep wake-up segment choice
def test_sleep_wakeup_can_land_in_either_half_of_the_night():
    """The old code always picked the longest segment, so 23:00-24:00 was dead."""

    from decision.virtual_life.core.models import minute_of_day
    from decision.virtual_life.core.utils import deterministic_int

    plan = DailyPlan.from_dict(
        {
            "date": "2026-10-22",
            "persona_id": "p",
            "timeline": [
                sleep_item(),
                {"id": "wake", "start": "07:00", "end": "08:00", "activity": "wake"},
                {"id": "day", "start": "08:00", "end": "23:00", "activity": "day"},
            ],
        }
    )
    segments = [item for item in plan.timeline if item.state == "sleep"]
    assert len(segments) == 2
    weights = [minute_of_day(item.end) - minute_of_day(item.start) for item in segments]
    assert weights == [420, 60]

    total = sum(weights)
    draws = [deterministic_int(f"sleep-pick::2026-10-22::u{i}", 0, total - 1) for i in range(400)]
    first = sum(1 for draw in draws if draw < weights[0])
    assert first > 0, "the morning half must be selectable"
    assert len(draws) - first > 0, "the evening half must be selectable"
    # Roughly proportional to segment length.
    assert abs(first / len(draws) - weights[0] / total) < 0.08
