from __future__ import annotations

import random

import pytest

from decision.life_media import (
    KIND_LABELS,
    KIND_TOOLS,
    MEDIA_KINDS,
    SUBJECTS,
    choose_proactive_media,
    clamp_count,
    enabled_kinds,
    media_instruction,
)


def settings(**overrides):
    base = {
        "enable": True,
        "photo_enable": True,
        "video_enable": True,
        "audio_enable": True,
        "photo_probability": 0.0,
        "video_probability": 0.0,
        "audio_probability": 0.0,
        "photo_count_min": 1,
        "photo_count_max": 1,
        "video_count_min": 1,
        "video_count_max": 1,
        "audio_count_min": 1,
        "audio_count_max": 1,
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize(
    "kind,value,expected",
    [
        ("photo", 1, 1),
        ("photo", 0, 1),
        ("photo", 99, 1),
        ("photo", -5, 1),
        ("photo", None, 1),
        ("photo", "abc", 1),
        ("photo", "3", 1),
        ("photo", True, 1),
        ("video", 5, 1),
        ("audio", 2, 1),
    ],
)
def test_clamp_count_inside_default_single_item_range(kind, value, expected):
    # The shipped default range is 1..1, so every requested count collapses to 1.
    assert clamp_count(settings(), kind, value) == expected


@pytest.mark.parametrize("kind", MEDIA_KINDS)
def test_clamp_count_honours_string_number_inside_range(kind):
    config = settings(**{f"{kind}_count_min": 2, f"{kind}_count_max": 5})
    assert clamp_count(config, kind, "3") == 3


def test_clamp_count_respects_range():
    config = settings(photo_count_min=2, photo_count_max=4)
    assert clamp_count(config, "photo", 0) == 2
    assert clamp_count(config, "photo", 3) == 3
    assert clamp_count(config, "photo", 99) == 4


def test_clamp_count_normalizes_inverted_range():
    config = settings(video_count_min=5, video_count_max=2)
    assert clamp_count(config, "video", 3) == 5
    assert clamp_count(config, "video", 99) == 5


def test_clamp_count_zero_range():
    config = settings(audio_count_min=0, audio_count_max=0)
    assert clamp_count(config, "audio", 7) == 0


def test_enabled_kinds_master_switch_off():
    assert enabled_kinds(settings(enable=False)) == []


def test_enabled_kinds_filters_disabled_kinds():
    config = settings(photo_enable=False, audio_enable=False)
    assert enabled_kinds(config) == ["video"]


def test_enabled_kinds_declared_order():
    assert enabled_kinds(settings()) == list(MEDIA_KINDS)


def test_choose_proactive_media_no_probability():
    assert choose_proactive_media(settings()) is None


def test_choose_proactive_media_disabled_kind_never_chosen():
    config = settings(photo_enable=False, photo_probability=1.0)
    assert choose_proactive_media(config) is None


def test_choose_proactive_media_master_switch_off():
    config = settings(photo_probability=1.0, enable=False)
    assert choose_proactive_media(config) is None


def test_choose_proactive_media_single_kind(monkeypatch):
    monkeypatch.setattr(random, "random", lambda: 0.0)
    config = settings(video_probability=0.5, video_count_min=2, video_count_max=3)
    assert choose_proactive_media(config) == ("video", 2, 3)


def test_choose_proactive_media_roll_fails(monkeypatch):
    monkeypatch.setattr(random, "random", lambda: 0.999)
    config = settings(photo_probability=0.5)
    assert choose_proactive_media(config) is None


def test_choose_proactive_media_never_returns_more_than_one_kind(monkeypatch):
    monkeypatch.setattr(random, "random", lambda: 0.0)
    config = settings(
        photo_probability=1.0,
        video_probability=1.0,
        audio_probability=1.0,
        photo_count_min=1,
        photo_count_max=2,
        video_count_min=3,
        video_count_max=4,
        audio_count_min=5,
        audio_count_max=6,
    )
    seen = set()
    for _ in range(200):
        choice = choose_proactive_media(config)
        assert choice is not None
        kind, low, high = choice
        assert kind in MEDIA_KINDS
        assert isinstance(low, int) and isinstance(high, int)
        assert low <= high
        seen.add(kind)
    # Weighted choice must be able to reach more than one kind.
    assert len(seen) > 1


def test_choose_proactive_media_returns_configured_range(monkeypatch):
    monkeypatch.setattr(random, "random", lambda: 0.0)
    config = settings(photo_probability=1.0, photo_count_min=0, photo_count_max=10)
    assert choose_proactive_media(config) == ("photo", 0, 10)


def test_choose_proactive_media_normalizes_inverted_range(monkeypatch):
    monkeypatch.setattr(random, "random", lambda: 0.0)
    config = settings(photo_probability=1.0, photo_count_min=4, photo_count_max=1)
    assert choose_proactive_media(config) == ("photo", 4, 4)


@pytest.mark.parametrize("kind", MEDIA_KINDS)
def test_media_instruction_zero_high_is_empty(kind):
    assert media_instruction(kind, 1, 0) == ""


@pytest.mark.parametrize("kind", MEDIA_KINDS)
def test_media_instruction_single_count(kind):
    text = media_instruction(kind, 2, 2)
    assert KIND_TOOLS[kind] in text
    assert KIND_LABELS[kind] in text
    assert "恰好 2" in text
    assert "~" not in text


@pytest.mark.parametrize("kind", MEDIA_KINDS)
def test_media_instruction_range(kind):
    text = media_instruction(kind, 1, 3)
    assert "1~3" in text
    assert KIND_TOOLS[kind] in text


def test_kind_tool_and_label_maps_are_complete():
    assert set(KIND_TOOLS) == set(MEDIA_KINDS)
    assert set(KIND_LABELS) == set(MEDIA_KINDS)
    assert KIND_TOOLS == {
        "photo": "take_photo",
        "video": "take_video",
        "audio": "send_voice",
    }


def test_subjects_cover_every_supported_target():
    assert set(SUBJECTS) == {"self", "other", "scene", "object", "screenshot"}
