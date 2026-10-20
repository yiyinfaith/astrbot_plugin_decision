from __future__ import annotations

import copy
import math

import pytest

from decision.life_config import life_defaults, life_schema, normalize_life, validate_umo


def test_defaults_roundtrip_and_independent_copies():
    defaults = life_defaults()
    assert normalize_life(defaults) == defaults
    assert not defaults["reply_delay_settings"]["enable"]
    defaults["creative_pool"]["themes"].clear()
    assert life_defaults()["creative_pool"]["themes"]
    assert next(iter(life_schema())) == "integration_settings"


@pytest.mark.parametrize(
    "path,value",
    [
        (("integration_settings", "jev_threshold"), math.nan),
        (("integration_settings", "jev_threshold"), 1.01),
        (("integration_settings", "max_concurrent"), 0),
        (("integration_settings", "candidate_ttl_minutes"), 0),
        (("friend_settings", "idle_min_minutes"), 0),
        (("friend_settings", "daily_hard_max"), 101),
        (("schedule_settings", "generation_retries"), 99),
        (("schedule_settings", "generate_time"), "24:00"),
        (("reply_delay_settings", "enable"), "false"),
        (("media_settings", "video_duration"), True),
        (("media_settings", "image_aspect_ratio"), "0:4"),
        (("image_settings", "image_width"), 9999),
        (("image_settings", "image_theme"), "invalid"),
        (("personal_network_integration", "new_character_probability"), 2),
        (("friend_settings", "session_list"), ["qq:GroupMessage:1"]),
        (("group_settings", "session_list"), ["qq:GroupMessage:"]),
    ],
)
def test_invalid_life_setting_rejected(path, value):
    data = copy.deepcopy(life_defaults())
    data[path[0]][path[1]] = value
    with pytest.raises(ValueError):
        normalize_life(data)


@pytest.mark.parametrize(
    "umo", ["", "qq:FriendMessage:", ":GroupMessage:1", "qq:Unknown:1", "qq:FriendMessage:1\n"]
)
def test_umo_invalid(umo):
    with pytest.raises(ValueError):
        validate_umo(umo)


def test_complete_source_schema_preserved():
    import json
    from pathlib import Path

    source = json.loads(
        (Path(__file__).parents[1] / "decision/virtual_life/schema.json").read_text(encoding="utf8")
    )
    schema = life_schema()
    for key in source:
        assert schema[key] == source[key]
    assert validate_umo("qq:FriendMessage:1")
    assert validate_umo("qq:GroupMessage:1")
