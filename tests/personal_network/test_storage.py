"""Tests for personal network persistence and relationship behavior.

Ported from the standalone plugin; only the import path was rewritten to
``decision.personal_network.storage``.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from decision.personal_network.storage import NetworkStorage


@pytest.fixture
def storage(tmp_path: Path):
    """Provide an isolated storage instance.

    Args:
        tmp_path: Pytest temporary directory.

    Yields:
        Initialized network storage.
    """
    instance = NetworkStorage(tmp_path / "network.sqlite3")
    yield instance
    instance.close()


def test_networks_are_isolated_and_roots_are_stable(storage: NetworkStorage):
    alice_root = storage.ensure_network("alice", "Alice")
    bob_root = storage.ensure_network("bob", "Bob")

    assert alice_root != bob_root
    assert storage.ensure_network("alice", "Alice") == alice_root
    assert len(storage.get_network("alice")["characters"]) == 1
    assert len(storage.get_network("bob")["characters"]) == 1


def test_tool_batch_binds_sender_and_creates_directed_relationship(
    storage: NetworkStorage,
):
    sender = {
        "platform": "aiocqhttp",
        "user_id": "10001",
        "session_id": "group-7",
        "nickname": "Group Nick",
        "umo": "aiocqhttp:group:group-7",
    }
    result = storage.upsert_batch(
        "alice",
        [
            {
                "ref": "speaker",
                "name": "Lin",
                "aliases": ["Xiao Lin"],
                "bio": "Alice's classmate",
                "facts": ["Enjoys astronomy"],
                "current_sender": True,
            }
        ],
        [
            {
                "source": "persona",
                "target": "speaker",
                "type": "friend",
                "strength": 72,
                "status": "active",
                "description": "A trusted school friend",
            }
        ],
        [
            {
                "participants": ["persona", "speaker"],
                "type": "相识",
                "summary": "We have been friends since middle school.",
                "occurred_at": "2026-07-20T12:00:00+00:00",
            }
        ],
        sender=sender,
    )

    data = storage.get_network("alice")
    person_id = result["refs"]["speaker"]
    identity = data["identities"][0]
    relationship = data["relationships"][0]
    assert identity["character_id"] == person_id
    assert identity["nicknames"][0]["nickname"] == "Group Nick"
    assert identity["nicknames"][0]["use_count"] == 1
    assert identity["session_id"] == "group-7"
    assert relationship["target_id"] == person_id
    assert relationship["strength"] == 72
    assert data["life_events"][0]["summary"].startswith("We have been")
    assert set(data["life_events"][0]["participant_ids"]) == {
        storage.ensure_network("alice"),
        person_id,
    }


def test_sender_messages_merge_into_one_conversation_session(
    storage: NetworkStorage,
):
    sender = {
        "platform": "test",
        "user_id": "user-1",
        "session_id": "group-1",
        "nickname": "Lin",
        "umo": "test:group:group-1",
    }
    storage.upsert_batch(
        "alice",
        [{"ref": "speaker", "name": "Lin", "current_sender": True}],
        sender=sender,
    )

    first = storage.record_sender_interaction(
        "alice", platform="test", user_id="user-1", session_id="group-1"
    )
    second = storage.record_sender_interaction(
        "alice", platform="test", user_id="user-1", session_id="group-1"
    )
    data = storage.get_network("alice")

    assert first == second
    assert len(data["life_events"]) == 1
    assert data["life_events"][0]["source"] == "conversation"


def test_interaction_stats_and_recent_events_are_injected(storage: NetworkStorage):
    now = datetime.now(UTC)
    storage.upsert_batch(
        "alice",
        [{"ref": "lin", "name": "Lin"}],
        [
            {
                "source": "persona",
                "target": "lin",
                "type": "朋友",
                "strength": 85,
            }
        ],
        [
            {
                "participants": ["persona", "lin"],
                "type": "聚餐",
                "summary": "一起吃了晚饭",
                "occurred_at": (now - timedelta(days=2)).isoformat(),
            },
            {
                "participants": ["persona", "lin"],
                "type": "通话",
                "summary": "讨论了近况",
                "occurred_at": (now - timedelta(days=40)).isoformat(),
            },
        ],
    )

    data = storage.get_network("alice")
    stats = data["relationships"][0]["interaction_stats"]
    context = storage.build_context(
        "alice",
        ["Lin 最近怎么样"],
        max_characters=8,
        max_relationships=16,
        max_chars=6000,
    )

    assert (stats["count_7d"], stats["count_30d"], stats["count_90d"]) == (
        1,
        1,
        2,
    )
    assert "亲密度=85" in context
    assert "近7/30/90天=1/1/2次" in context
    assert "一起吃了晚饭" in context
    assert "讨论了近况" in context


def test_relationship_strength_rejects_negative_values(storage: NetworkStorage):
    with pytest.raises(ValueError, match="between 0 and 100"):
        storage.upsert_batch(
            "alice",
            [{"ref": "lin", "name": "Lin"}],
            [
                {
                    "source": "persona",
                    "target": "lin",
                    "type": "朋友",
                    "strength": -1,
                }
            ],
        )


def test_nicknames_are_lists_sorted_by_usage_frequency(storage: NetworkStorage):
    sender = {
        "platform": "aiocqhttp",
        "user_id": "10001",
        "session_id": "group-7",
        "nickname": "Lin",
        "umo": "aiocqhttp:group:group-7",
    }
    storage.upsert_batch(
        "alice",
        [{"ref": "speaker", "name": "Lin", "current_sender": True}],
        [],
        sender=sender,
    )
    for nickname in ("Xiao Lin", "Lin", "Xiao Lin", "Xiao Lin"):
        assert storage.record_known_sender_nickname(
            "alice",
            platform="aiocqhttp",
            user_id="10001",
            session_id="group-7",
            nickname=nickname,
        )

    identity = storage.get_network("alice")["identities"][0]
    assert [item["nickname"] for item in identity["nicknames"]] == [
        "Xiao Lin",
        "Lin",
    ]
    assert [item["use_count"] for item in identity["nicknames"]] == [3, 2]
    assert not storage.record_known_sender_nickname(
        "alice",
        platform="aiocqhttp",
        user_id="10001",
        session_id="another-group",
        nickname="Other Group Name",
    )


def test_character_aliases_are_editable_and_sorted_by_frequency(
    storage: NetworkStorage,
):
    result = storage.upsert_batch(
        "alice",
        [
            {
                "ref": "lin",
                "name": "Lin",
                "alias_usages": [
                    {"alias": "Xiao Lin", "use_count": 2},
                    {"alias": "Lin Lin", "use_count": 5},
                ],
            }
        ],
        [],
    )
    character_id = result["refs"]["lin"]
    character = next(
        item for item in storage.get_network("alice")["characters"] if item["id"] == character_id
    )
    assert [item["alias"] for item in character["alias_usages"]] == [
        "Lin Lin",
        "Xiao Lin",
    ]
    assert "aliases" not in character
    assert [item["use_count"] for item in character["alias_usages"]] == [5, 2]

    assert storage.record_alias_mentions("alice", "Xiao Lin is here") == 1
    character = next(
        item for item in storage.get_network("alice")["characters"] if item["id"] == character_id
    )
    assert [(item["alias"], item["use_count"]) for item in character["alias_usages"]] == [
        ("Lin Lin", 5),
        ("Xiao Lin", 3),
    ]

    storage.upsert_batch(
        "alice",
        [
            {
                "id": character_id,
                "name": "Lin",
                "alias_usages": [
                    {"alias": "Xiao Lin", "use_count": 9},
                    {"alias": "New Lin", "use_count": 1},
                ],
            }
        ],
        [],
    )
    character = next(
        item for item in storage.get_network("alice")["characters"] if item["id"] == character_id
    )
    assert [item["alias"] for item in character["alias_usages"]] == [
        "Xiao Lin",
        "New Lin",
    ]


def test_relationship_query_supports_aliases_and_overview(storage: NetworkStorage):
    storage.upsert_batch(
        "alice",
        [{"ref": "lin", "name": "Lin", "aliases": ["Xiao Lin"]}],
        [
            {
                "source": "persona",
                "target": "lin",
                "type": "朋友",
                "strength": 80,
                "status": "active",
                "description": "多年好友",
            }
        ],
    )

    overview = storage.query_relationships("alice")
    details = storage.query_relationships("alice", "Xiao Lin")

    assert "1 个人物、1 条关系" in overview
    assert "Lin" in overview
    assert "Xiao Lin（使用 1 次）" in details
    assert "Lin 是 alice 的朋友" in details
    assert "多年好友" in details


def test_batch_rolls_back_when_a_relationship_reference_is_invalid(
    storage: NetworkStorage,
):
    with pytest.raises(ValueError, match="unknown character reference"):
        storage.upsert_batch(
            "alice",
            [{"ref": "new", "name": "Temporary"}],
            [
                {
                    "source": "new",
                    "target": "missing",
                    "type": "friend",
                    "strength": 1,
                }
            ],
        )

    assert [item["name"] for item in storage.get_network("alice")["characters"]] == ["alice"]


def test_same_name_ambiguity_requires_uuid(storage: NetworkStorage):
    first_id = str(uuid.uuid4())
    second_id = str(uuid.uuid4())
    payload = {
        "characters": [
            {"id": first_id, "name": "Alex", "is_persona": False},
            {"id": second_id, "name": "Alex", "is_persona": False},
        ],
        "relationships": [],
        "life_events": [],
        "identities": [],
    }
    storage.replace_from_import("alice", payload)

    with pytest.raises(ValueError, match="ambiguous character name"):
        storage.upsert_batch("alice", [{"ref": "alex", "name": "Alex"}], [])

    result = storage.upsert_batch(
        "alice", [{"id": first_id, "ref": "alex", "name": "Alex", "bio": "First"}], []
    )
    assert result["refs"]["alex"] == first_id


def test_merge_rewires_and_deduplicates_relationships(storage: NetworkStorage):
    root = storage.ensure_network("alice")
    people = storage.upsert_batch(
        "alice",
        [{"ref": "keep", "name": "Lin"}, {"ref": "duplicate", "name": "Lynn"}],
        [
            {
                "source": "persona",
                "target": "keep",
                "type": "friend",
                "strength": 50,
            },
            {
                "source": "persona",
                "target": "duplicate",
                "type": "friend",
                "strength": 60,
            },
            {
                "source": "duplicate",
                "target": "keep",
                "type": "knows",
                "strength": 10,
            },
        ],
        [
            {
                "participants": ["persona", "keep"],
                "type": "见面",
                "summary": "first",
            },
            {
                "participants": ["persona", "duplicate"],
                "type": "通话",
                "summary": "second",
            },
        ],
    )

    storage.merge_characters("alice", people["refs"]["keep"], people["refs"]["duplicate"])
    data = storage.get_network("alice")
    assert {item["id"] for item in data["characters"]} == {
        root,
        people["refs"]["keep"],
    }
    assert len(data["relationships"]) == 1
    assert len(data["life_events"]) == 2
    assert all(
        people["refs"]["duplicate"] not in item["participant_ids"] for item in data["life_events"]
    )
    retained = next(item for item in data["characters"] if not item["is_persona"])
    assert "Lynn" in [item["alias"] for item in retained["alias_usages"]]


def test_delete_character_cascades_relationships_identities_and_life_events(
    storage: NetworkStorage,
):
    sender = {
        "platform": "test",
        "user_id": "user-1",
        "session_id": "group-1",
        "nickname": "Lin",
        "umo": "test:group:group-1",
    }
    created = storage.upsert_batch(
        "alice",
        [{"ref": "lin", "name": "Lin", "current_sender": True}],
        [
            {
                "source": "persona",
                "target": "lin",
                "type": "朋友",
                "strength": 80,
            }
        ],
        [
            {
                "participants": ["persona", "lin"],
                "type": "见面",
                "summary": "一起吃饭",
            }
        ],
        sender=sender,
    )

    storage.delete_character("alice", created["refs"]["lin"])
    data = storage.get_network("alice")

    assert all(item["id"] != created["refs"]["lin"] for item in data["characters"])
    assert data["relationships"] == []
    assert data["identities"] == []
    assert data["life_events"] == []


def test_external_life_event_is_idempotent_and_includes_persona(
    storage: NetworkStorage,
):
    created = storage.upsert_batch(
        "alice",
        [{"ref": "lin", "name": "Lin"}],
        [],
    )

    first = storage.record_external_life_event(
        "alice",
        [created["refs"]["lin"]],
        event_type="virtual_schedule",
        summary="Had lunch together",
        occurred_at="2026-07-28T12:00:00+08:00",
        importance=45,
        emotional_tone="relaxed",
        source="astrbot_plugin_virtual_life",
        source_key="2026-07-28:lunch",
    )
    second = storage.record_external_life_event(
        "alice",
        [created["refs"]["lin"]],
        event_type="virtual_schedule",
        summary="Had lunch and talked about work",
        occurred_at="2026-07-28T12:00:00+08:00",
        importance=50,
        emotional_tone="warm",
        source="astrbot_plugin_virtual_life",
        source_key="2026-07-28:lunch",
    )

    data = storage.get_network("alice")
    assert first["event_id"] == second["event_id"]
    assert first["created"] is True
    assert second["created"] is False
    assert len(data["life_events"]) == 1
    assert data["life_events"][0]["summary"] == "Had lunch and talked about work"
    assert set(data["life_events"][0]["participant_ids"]) == {
        storage._root_id("alice"),
        created["refs"]["lin"],
    }


def test_context_matches_alias_and_includes_one_hop(storage: NetworkStorage):
    storage.upsert_batch(
        "alice",
        [{"ref": "lin", "name": "Lin", "aliases": ["Xiao Lin"], "facts": ["Likes tea"]}],
        [
            {
                "source": "persona",
                "target": "lin",
                "type": "friend",
                "strength": 80,
                "description": "Long-time friend",
            }
        ],
    )

    context = storage.build_context(
        "alice",
        ["How is XIAO LIN?"],
        max_characters=8,
        max_relationships=16,
        max_chars=6000,
    )
    assert "Likes tea" in context
    assert "关系方向固定为：目标人物是主体人物的“关系类型”" in context
    assert "关系事实：Lin 是 alice 的“friend”" in context
    assert (
        storage.build_context(
            "alice",
            ["Nobody relevant"],
            max_characters=8,
            max_relationships=16,
            max_chars=6000,
        )
        == ""
    )


def test_context_uses_first_candidate_text_with_a_character(storage: NetworkStorage):
    storage.upsert_batch(
        "alice",
        [
            {"ref": "lin", "name": "Lin"},
            {"ref": "mei", "name": "Mei"},
        ],
        [],
    )

    context = storage.build_context(
        "alice",
        ["没有人物", "Mei and Lin", "Lin"],
        max_characters=8,
        max_relationships=16,
        max_chars=6000,
    )

    assert "人物：Mei" in context
    assert "人物：Lin" in context


def test_bound_sender_identity_triggers_context_and_keeps_named_people(
    storage: NetworkStorage,
):
    storage.upsert_batch(
        "alice",
        [
            {"ref": "lin", "name": "Lin", "current_sender": True},
            {"ref": "mei", "name": "Mei"},
        ],
        [],
        sender={
            "platform": "test",
            "user_id": "user",
            "session_id": "",
            "nickname": "Lin",
            "umo": "test:friend:user",
        },
    )

    context = storage.build_context(
        "alice",
        ["Mei 今天过得怎么样？"],
        platform="test",
        user_id="user",
        session_id="",
        max_characters=8,
        max_relationships=16,
        max_chars=6000,
    )

    assert "人物：Lin" in context
    assert "人物：Mei" in context


def test_import_merges_without_deleting_local_data(storage: NetworkStorage):
    local = storage.upsert_batch("alice", [{"ref": "local", "name": "Local"}], [])
    imported_id = str(uuid.uuid4())
    payload = {
        "characters": [{"id": imported_id, "name": "Imported", "is_persona": False}],
        "relationships": [],
        "life_events": [],
        "identities": [],
    }

    storage.replace_from_import("alice", payload)
    ids = {item["id"] for item in storage.get_network("alice")["characters"]}
    assert local["refs"]["local"] in ids
    assert imported_id in ids


def test_import_cannot_overwrite_another_personas_uuid(storage: NetworkStorage):
    created = storage.upsert_batch("alice", [{"ref": "person", "name": "Alice Person"}], [])
    malicious = {
        "characters": [
            {
                "id": created["refs"]["person"],
                "name": "Overwritten",
                "is_persona": False,
            }
        ],
        "relationships": [],
        "life_events": [],
        "identities": [],
    }

    with pytest.raises(ValueError, match="another persona"):
        storage.replace_from_import("bob", malicious)
    original = next(
        item
        for item in storage.get_network("alice")["characters"]
        if item["id"] == created["refs"]["person"]
    )
    assert original["name"] == "Alice Person"


def test_import_preserves_nickname_frequency(storage: NetworkStorage, tmp_path: Path):
    sender = {
        "platform": "test",
        "user_id": "user-1",
        "session_id": "group-1",
        "nickname": "Lin",
        "umo": "test:group:group-1",
    }
    storage.upsert_batch(
        "alice",
        [
            {
                "ref": "speaker",
                "name": "Lin",
                "current_sender": True,
                "alias_usages": [{"alias": "Xiao Lin", "use_count": 4}],
            }
        ],
        [],
        sender=sender,
    )
    storage.record_known_sender_nickname(
        "alice",
        platform="test",
        user_id="user-1",
        session_id="group-1",
        nickname="Xiao Lin",
    )
    storage.record_known_sender_nickname(
        "alice",
        platform="test",
        user_id="user-1",
        session_id="group-1",
        nickname="Xiao Lin",
    )
    payload = storage.get_network("alice")

    restored = NetworkStorage(tmp_path / "restored.sqlite3")
    try:
        restored.replace_from_import("alice", payload)
        nicknames = restored.get_network("alice")["identities"][0]["nicknames"]
        assert [(item["nickname"], item["use_count"]) for item in nicknames] == [
            ("Xiao Lin", 2),
            ("Lin", 1),
        ]
        character = next(
            item for item in restored.get_network("alice")["characters"] if not item["is_persona"]
        )
        assert character["alias_usages"][0]["alias"] == "Xiao Lin"
        assert character["alias_usages"][0]["use_count"] == 4
    finally:
        restored.close()


def test_export_shape_is_json_serializable(storage: NetworkStorage):
    storage.upsert_batch("alice", [{"ref": "lin", "name": "Lin"}], [])
    encoded = json.dumps(storage.get_network("alice"), ensure_ascii=False)
    assert "Lin" in encoded


def test_version_three_evidence_migrates_to_life_events(tmp_path: Path):
    database_path = tmp_path / "version-three.sqlite3"
    root_id = str(uuid.uuid5(uuid.NAMESPACE_URL, "astrbot-persona:alice"))
    person_id = str(uuid.uuid4())
    relationship_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()
    connection = sqlite3.connect(database_path)
    connection.executescript(
        """
        CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO schema_meta(key, value) VALUES ('version', '3');
        CREATE TABLE networks (
            persona_id TEXT PRIMARY KEY, enabled INTEGER NOT NULL,
            persona_missing INTEGER NOT NULL, created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE characters (
            id TEXT PRIMARY KEY, persona_id TEXT NOT NULL, name TEXT NOT NULL,
            aliases TEXT NOT NULL, bio TEXT NOT NULL, personality TEXT NOT NULL,
            preferences TEXT NOT NULL, facts TEXT NOT NULL, notes TEXT NOT NULL,
            avatar_filename TEXT, is_persona INTEGER NOT NULL,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE relationships (
            id TEXT PRIMARY KEY, persona_id TEXT NOT NULL, source_id TEXT NOT NULL,
            target_id TEXT NOT NULL, relation_type TEXT NOT NULL,
            strength INTEGER NOT NULL, status TEXT NOT NULL, description TEXT NOT NULL,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE evidence (
            id TEXT PRIMARY KEY, relationship_id TEXT NOT NULL, excerpt TEXT NOT NULL,
            umo TEXT NOT NULL, speaker_id TEXT NOT NULL, created_at TEXT NOT NULL
        );
        """
    )
    connection.execute("INSERT INTO networks VALUES (?, 1, 0, ?, ?)", ("alice", now, now))
    for character_id, name, is_persona in (
        (root_id, "alice", 1),
        (person_id, "Lin", 0),
    ):
        connection.execute(
            """INSERT INTO characters VALUES
               (?, 'alice', ?, '[]', '', '', '[]', '[]', '', NULL, ?, ?, ?)""",
            (character_id, name, is_persona, now, now),
        )
    connection.execute(
        """INSERT INTO relationships VALUES
           (?, 'alice', ?, ?, '朋友', -20, 'active', '', ?, ?)""",
        (relationship_id, root_id, person_id, now, now),
    )
    connection.execute(
        "INSERT INTO evidence VALUES (?, ?, '一起看过电影', '', '', ?)",
        (str(uuid.uuid4()), relationship_id, now),
    )
    connection.commit()
    connection.close()

    migrated = NetworkStorage(database_path)
    try:
        data = migrated.get_network("alice")
        version = migrated._conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'version'"
        ).fetchone()["value"]
        evidence_table = migrated._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'evidence'"
        ).fetchone()

        assert version == "4"
        assert evidence_table is None
        assert data["relationships"][0]["strength"] == 0
        assert data["life_events"][0]["summary"] == "一起看过电影"
        assert set(data["life_events"][0]["participant_ids"]) == {
            root_id,
            person_id,
        }
    finally:
        migrated.close()


def test_generated_batch_can_atomically_create_thirty_two_people(tmp_path: Path):
    storage = NetworkStorage(tmp_path / "network.sqlite3")
    characters = [{"ref": f"person_{index}", "name": f"Person {index}"} for index in range(32)]
    relationships = [
        {
            "source": "persona",
            "target": item["ref"],
            "type": "朋友",
            "strength": 50,
        }
        for item in characters
    ]
    try:
        result = storage.upsert_batch(
            "alice",
            characters,
            relationships,
            max_characters=32,
            max_relationships=128,
        )

        assert len(result["refs"]) == 33
        assert len(result["relationship_ids"]) == 32
        assert len(storage.get_network("alice")["characters"]) == 33
    finally:
        storage.close()
