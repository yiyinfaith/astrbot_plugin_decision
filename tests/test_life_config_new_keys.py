from __future__ import annotations

import copy

import pytest

from decision.life_config import life_defaults, life_schema, normalize_life


def test_defaults_roundtrip():
    defaults = life_defaults()
    assert normalize_life(defaults) == defaults


def test_group_order_leads_with_master_switch_and_scope():
    order = list(life_schema())
    assert order[:3] == ["integration_settings", "friend_settings", "group_settings"]
    assert "media_settings" in order
    assert "appearance_settings" in order
    assert "personal_network_integration" in order


def test_no_global_proactive_master_switch():
    """主动发起 is controlled only by the per-kind switches."""

    items = life_schema()["integration_settings"]["items"]
    assert "enable" not in items
    defaults = life_defaults()
    assert "enable" not in defaults["integration_settings"]
    # Both per-kind switches exist and are real booleans.
    assert isinstance(defaults["friend_settings"]["enable"], bool)
    assert isinstance(defaults["group_settings"]["enable"], bool)


def test_media_settings_keys():
    media = life_defaults()["media_settings"]
    for key in (
        "enable",
        "photo_enable",
        "video_enable",
        "audio_enable",
        "image_aspect_ratio",
        "video_aspect_ratio",
        "video_duration",
        "cooldown_seconds",
        "daily_limit",
        "photo_probability",
        "video_probability",
        "audio_probability",
        "photo_count_min",
        "photo_count_max",
        "video_count_min",
        "video_count_max",
        "audio_count_min",
        "audio_count_max",
        "audio_mode",
        "audio_voice",
        "photo_prompt",
        "video_prompt",
        "audio_prompt",
    ):
        assert key in media, key
    assert media["photo_enable"] is True
    assert media["video_enable"] is True
    assert media["audio_enable"] is False


def test_appearance_settings_keys():
    appearance = life_defaults()["appearance_settings"]
    assert set(appearance) == {"description", "use_reference_outfit"}
    assert appearance["description"] == ""
    assert appearance["use_reference_outfit"] is True


def test_personal_network_settings_keys():
    network = life_defaults()["personal_network_integration"]
    for key in (
        "enable",
        "enable_llm_update_tool",
        "enable_llm_query_tool",
        "generation_llm_provider",
        "pronoun_history_messages",
        "pronoun_patterns",
        "context_injection_position",
        "context_max_characters",
        "context_max_relationships",
        "context_max_chars",
        "max_context_chars",
        "record_completed_events",
        "new_character_probability",
        "inject_current_participants",
    ):
        assert key in network, key
    assert network["enable"] is False
    assert network["context_injection_position"] == "system_prompt"


def test_proactive_billing_keys():
    media = life_defaults()["media_settings"]
    assert media["proactive_billing_mode"] == "fixed"
    assert media["proactive_billing_platform"] == ""
    assert media["proactive_billing_user_id"] == ""
    options = life_schema()["media_settings"]["items"]["proactive_billing_mode"]["options"]
    assert options == ["session", "fixed"]


@pytest.mark.parametrize("value", ["invalid", "", "FIXED"])
def test_billing_mode_rejects_unknown_values(value):
    data = copy.deepcopy(life_defaults())
    data["media_settings"]["proactive_billing_mode"] = value
    with pytest.raises(ValueError):
        normalize_life(data)


@pytest.mark.parametrize(
    "key,value",
    [("proactive_billing_user_id", "x" * 201), ("proactive_billing_platform", "y" * 101)],
)
def test_billing_fields_are_length_limited(key, value):
    data = copy.deepcopy(life_defaults())
    data["media_settings"][key] = value
    with pytest.raises(ValueError):
        normalize_life(data)


def test_billing_accepts_a_real_account():
    data = copy.deepcopy(life_defaults())
    data["media_settings"].update(
        proactive_billing_mode="fixed",
        proactive_billing_platform="aiocqhttp",
        proactive_billing_user_id="2980555633",
    )
    result = normalize_life(data)
    assert result["media_settings"]["proactive_billing_user_id"] == "2980555633"
    assert result["media_settings"]["proactive_billing_platform"] == "aiocqhttp"


def test_scope_mode_defaults_to_whitelist():
    """Default is whitelist: an empty list means "nobody", matching the hint."""

    defaults = life_defaults()
    assert defaults["friend_settings"]["session_blacklist"] is False
    assert defaults["group_settings"]["session_blacklist"] is False
    assert defaults["friend_settings"]["session_list"] == []
    assert defaults["group_settings"]["session_list"] == []


def test_group_cooldown_does_not_exceed_its_idle_ceiling():
    """A cooldown above idle_max would make the idle window unreachable."""

    group = life_defaults()["group_settings"]
    assert group["cooldown_minutes"] <= group["idle_max_minutes"]


def test_sleep_skips_quiet_defaults_to_off():
    delivery = life_defaults()["delivery_settings"]
    assert delivery["sleep_skips_quiet"] is False
    assert delivery["sleep_exception_probability"] > 0


@pytest.mark.parametrize(
    "group,key,value",
    [
        ("media_settings", "photo_probability", 1.5),
        ("media_settings", "photo_probability", -0.1),
        ("media_settings", "video_probability", 1.01),
        ("media_settings", "audio_probability", -1.0),
        ("media_settings", "audio_mode", "invalid"),
        ("media_settings", "cooldown_seconds", 3601),
        ("media_settings", "daily_limit", 201),
        ("personal_network_integration", "context_injection_position", "invalid"),
        ("personal_network_integration", "context_max_chars", 499),
        ("personal_network_integration", "context_max_chars", 12001),
        ("personal_network_integration", "max_context_chars", 100),
        ("personal_network_integration", "pronoun_history_messages", 0),
        ("personal_network_integration", "pronoun_history_messages", 101),
        ("personal_network_integration", "context_max_characters", 0),
        ("personal_network_integration", "context_max_relationships", 51),
        ("personal_network_integration", "pronoun_patterns", ["["]),
    ],
)
def test_invalid_new_setting_rejected(group, key, value):
    data = copy.deepcopy(life_defaults())
    data[group][key] = value
    with pytest.raises(ValueError):
        normalize_life(data)


@pytest.mark.parametrize("kind", ["photo", "video", "audio"])
def test_inverted_count_range_rejected(kind):
    data = copy.deepcopy(life_defaults())
    data["media_settings"][f"{kind}_count_min"] = 3
    data["media_settings"][f"{kind}_count_max"] = 1
    with pytest.raises(ValueError):
        normalize_life(data)


@pytest.mark.parametrize("value", [-1, 11])
def test_count_bounds_rejected(value):
    data = copy.deepcopy(life_defaults())
    data["media_settings"]["photo_count_min"] = value
    with pytest.raises(ValueError):
        normalize_life(data)


def test_blacklist_with_empty_list_is_valid():
    data = copy.deepcopy(life_defaults())
    data["friend_settings"]["session_blacklist"] = True
    data["group_settings"]["session_blacklist"] = True
    result = normalize_life(data)
    assert result["friend_settings"]["session_list"] == []
    assert result["friend_settings"]["session_blacklist"] is True


def test_media_boundary_values_are_valid():
    data = copy.deepcopy(life_defaults())
    data["media_settings"].update(
        photo_probability=0.0,
        video_probability=1.0,
        audio_probability=0.5,
        photo_count_min=0,
        photo_count_max=10,
        video_count_min=0,
        video_count_max=0,
        audio_count_min=10,
        audio_count_max=10,
        cooldown_seconds=0,
        daily_limit=0,
    )
    result = normalize_life(data)
    assert result["media_settings"]["video_count_max"] == 0
    assert result["media_settings"]["daily_limit"] == 0
