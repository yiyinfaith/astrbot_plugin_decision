"""Tests for persona-based network generation helpers.

Ported from the standalone plugin; only the import path was rewritten to
``decision.personal_network.generation``.
"""

from __future__ import annotations

import json

import pytest

from decision.personal_network.generation import (
    MAX_GENERATION_HINT_CHARS,
    build_generation_prompts,
    normalize_generation_hint,
    parse_generation_draft,
    validate_generation_draft,
)


def network_fixture() -> dict:
    return {
        "characters": [
            {
                "id": "root-id",
                "name": "Alice",
                "is_persona": True,
                "alias_usages": [],
                "bio": "",
                "personality": "",
                "preferences": [],
                "facts": [],
            },
            {
                "id": "existing-id",
                "name": "Existing",
                "is_persona": False,
                "alias_usages": [{"alias": "Old", "use_count": 2}],
                "bio": "Existing biography",
                "personality": "",
                "preferences": [],
                "facts": ["Known fact"],
            },
        ],
        "relationships": [],
    }


def simple_draft(count: int = 1) -> dict:
    characters = [
        {
            "ref": f"person_{index}",
            "name": f"Person {index}",
            "aliases": [],
            "bio": "Friend",
            "personality": "Reliable",
            "preferences": [],
            "facts": [],
        }
        for index in range(count)
    ]
    relationships = [
        {
            "source": "persona",
            "target": item["ref"],
            "type": "朋友",
            "strength": 60,
            "status": "active",
            "description": "",
        }
        for item in characters
    ]
    return {"characters": characters, "relationships": relationships}


def test_parse_generation_draft_extracts_json_from_model_text():
    payload = simple_draft()
    raw = f"结果如下：\n```json\n{json.dumps(payload)}\n```"

    assert parse_generation_draft(raw) == payload


def test_generation_accepts_thirty_two_connected_people():
    normalized = validate_generation_draft(
        simple_draft(32), network_fixture(), expected_new_count=32
    )

    assert len(normalized["characters"]) == 32
    assert len(normalized["relationships"]) == 32


def test_generation_rejects_more_than_thirty_two_people():
    with pytest.raises(ValueError, match="最多包含 32"):
        validate_generation_draft(simple_draft(33), network_fixture())


def test_generation_rejects_isolated_or_unknown_people():
    isolated = simple_draft(2)
    isolated["relationships"] = isolated["relationships"][:1]
    with pytest.raises(ValueError, match="没有任何关系"):
        validate_generation_draft(isolated, network_fixture())

    unknown = simple_draft()
    unknown["relationships"][0]["target"] = "missing"
    with pytest.raises(ValueError, match="未知人物"):
        validate_generation_draft(unknown, network_fixture())


def test_existing_people_only_receive_empty_fields():
    draft = simple_draft()
    draft["characters"].append(
        {
            "id": "existing-id",
            "name": "Renamed",
            "aliases": ["Replacement"],
            "bio": "Replacement biography",
            "personality": "New personality",
            "preferences": ["Tea"],
            "facts": ["Replacement fact"],
        }
    )
    draft["relationships"].append(
        {
            "source": "person_0",
            "target": "existing-id",
            "type": "同事",
            "strength": 45,
            "status": "active",
            "description": "",
        }
    )

    normalized = validate_generation_draft(
        draft, network_fixture(), allow_fill_existing=True, expected_new_count=1
    )
    existing = normalized["characters"][1]

    assert existing["name"] == "Existing"
    assert existing["personality"] == "New personality"
    assert existing["preferences"] == ["Tea"]
    assert "aliases" not in existing
    assert "bio" not in existing
    assert "facts" not in existing


def test_root_uuid_is_normalized_to_persona_reference():
    draft = simple_draft()
    draft["relationships"][0]["source"] = "root-id"

    normalized = validate_generation_draft(draft, network_fixture())

    assert normalized["relationships"][0]["source"] == "persona"


def test_generation_prompt_contains_persona_and_density_rules():
    system_prompt, prompt = build_generation_prompts(
        persona_prompt="A reserved architect.",
        network=network_fixture(),
        count=6,
        density="balanced",
        allow_fill_existing=False,
    )

    assert "只输出一个合法 JSON 对象" in system_prompt
    assert "A reserved architect." in prompt
    assert "恰好 6 个" in prompt
    assert "不得在 characters 中输出已有 UUID" in prompt
    assert "额外生成要求（低优先级内容偏好）" not in prompt


def test_generation_prompt_contains_request_scoped_hint_and_safety_rule():
    system_prompt, prompt = build_generation_prompts(
        persona_prompt="A reserved architect.",
        network=network_fixture(),
        count=3,
        density="sparse",
        allow_fill_existing=False,
        generation_hint="重点生成大学时期的人际关系。",
    )

    assert "必须忽略冲突部分" in system_prompt
    assert "额外生成要求（低优先级内容偏好）" in prompt
    assert "重点生成大学时期的人际关系。" in prompt


def test_generation_hint_is_trimmed_and_limited():
    assert normalize_generation_hint("  现实背景  ") == "现实背景"
    assert normalize_generation_hint(None) == ""

    with pytest.raises(ValueError, match="必须是字符串"):
        normalize_generation_hint(123)
    with pytest.raises(ValueError, match="不能超过 2000 字"):
        normalize_generation_hint("a" * (MAX_GENERATION_HINT_CHARS + 1))
