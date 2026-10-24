"""Service-level tests for the merged Personal Network subpackage.

These tests never import AstrBot: :mod:`tests.personal_network.conftest` installs
minimal doubles first, and the fake owner supplies the ``context`` namespace and
``_setting`` lookup that :class:`PersonalNetworkService` depends on.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from decision.personal_network import (
    VALID_RELATIONSHIP_STATUSES,
    NetworkStorage,
    PersonalNetworkService,
)
from decision.personal_network.storage import SCHEMA_VERSION

from ._astrbot_stub import PluginUploadFile, reset_request

PLUGIN_DATA = "plugin_data"
DECISION_DIR = "astrbot_plugin_decision"
LEGACY_DIR = "astrbot_plugin_personal_network"


class FakeConversationManager:
    """Conversation double returning no active conversation."""

    def __init__(self, persona_id: str | None = None):
        self.persona_id = persona_id

    async def get_curr_conversation_id(self, umo: str) -> str | None:
        return "conversation-1" if self.persona_id else None

    async def get_conversation(self, umo: str, conversation_id: str):
        return SimpleNamespace(persona_id=self.persona_id)


class FakePersonaManager:
    """Persona double backed by a fixed id/name map."""

    def __init__(self, personas: dict[str, str] | None = None):
        self.personas = personas or {"alice": "Alice"}

    async def get_default_persona_v3(self, umo: str | None = None) -> dict[str, str]:
        return {"name": next(iter(self.personas))}

    async def get_all_personas(self) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(persona_id=persona_id, name=name)
            for persona_id, name in self.personas.items()
        ]

    async def get_persona(self, persona_id: str) -> SimpleNamespace:
        return SimpleNamespace(system_prompt=f"prompt for {persona_id}")

    def get_persona_v3_by_id(self, persona_id: str) -> dict[str, str]:
        return {"prompt": f"prompt for {persona_id}"}


class FakeOwner:
    """Minimal ``DecisionPlugin`` double exposing ``context`` and ``_setting``."""

    def __init__(self, life: dict[str, Any] | None = None, persona_id: str | None = None):
        self.context = SimpleNamespace(
            conversation_manager=FakeConversationManager(persona_id),
            persona_manager=FakePersonaManager(),
            get_provider_by_id=lambda provider_id: None,
            get_using_provider=lambda: None,
        )
        self.life = {} if life is None else life
        self.setting_calls: list[tuple[str, Any]] = []

    def _setting(self, key: str, default: Any = None) -> Any:
        self.setting_calls.append((key, default))
        return self.life if key == "life" else default


class FakeToolSet:
    """Tool set double recording removals."""

    def __init__(self, tools: list[str] | None = None):
        self.tools = list(tools or [])
        self.removed: list[str] = []

    def remove_tool(self, name: str) -> None:
        self.removed.append(name)
        if name in self.tools:
            self.tools.remove(name)


class FakeEvent:
    """Message event double covering the attributes the service reads."""

    def __init__(self, message_str: str = "他今天来了", umo: str = "aiocqhttp:FriendMessage:1"):
        self.unified_msg_origin = umo
        self.message_str = message_str

    def get_platform_name(self) -> str:
        return "aiocqhttp"

    def get_platform_id(self) -> str:
        return "aiocqhttp"

    def get_sender_id(self) -> str:
        return "10001"

    def get_sender_name(self) -> str:
        return "Group Nick"

    def get_group_id(self) -> str:
        return ""


def network_settings(**overrides: Any) -> dict[str, Any]:
    """Build a ``life`` payload with the integration group configured."""
    return {"personal_network_integration": dict(overrides)}


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Return the merged plugin data directory used by the service."""
    return tmp_path / PLUGIN_DATA / DECISION_DIR / "personal_network"


@pytest.fixture
def services():
    """Track created services and close their SQLite handles at teardown."""
    created: list[PersonalNetworkService] = []

    def factory(owner: FakeOwner, directory: Path) -> PersonalNetworkService:
        service = PersonalNetworkService(owner, directory)
        created.append(service)
        return service

    yield factory
    for service in created:
        service.storage.close()


def test_construction_creates_database_and_directories(services, data_dir: Path):
    service = services(FakeOwner(), data_dir)

    assert data_dir.is_dir()
    assert service.avatar_dir.is_dir()
    assert service.export_dir.is_dir()
    assert (data_dir / "personal_network.sqlite3").is_file()
    assert service.storage.database_path == data_dir / "personal_network.sqlite3"
    assert SCHEMA_VERSION == 4
    assert VALID_RELATIONSHIP_STATUSES == {"active", "ended", "uncertain"}
    assert isinstance(service.storage, NetworkStorage)


def test_construction_never_creates_the_migration_marker(services, data_dir: Path):
    services(FakeOwner(), data_dir)

    assert not (data_dir / "migrated_from_standalone.json").exists()


async def test_upsert_batch_for_plugin_creates_character_and_returns_ref(services, data_dir: Path):
    service = services(FakeOwner(network_settings(enable=True)), data_dir)

    result = await service.upsert_batch_for_plugin(
        "alice",
        characters=[{"ref": "friend_1", "name": "Lin", "bio": "classmate"}],
        relationships=[
            {
                "source": "persona",
                "target": "friend_1",
                "type": "朋友",
                "strength": 60,
                "status": "active",
                "description": "",
            }
        ],
        source="astrbot_plugin_decision",
    )

    assert result["updated"] is True
    ref = result["refs"]["friend_1"]
    assert result["refs"]["persona"]
    assert result["relationship_ids"]
    network = await service.get_network_for_plugin("alice")
    created = next(item for item in network["characters"] if item["id"] == ref)
    assert created["name"] == "Lin"
    assert created["bio"] == "classmate"


async def test_upsert_batch_for_plugin_reports_value_error_as_reason(services, data_dir: Path):
    service = services(FakeOwner(network_settings(enable=True)), data_dir)

    result = await service.upsert_batch_for_plugin(
        "alice", characters=[{"ref": "broken", "name": ""}]
    )

    assert result["updated"] is False
    assert "name" in result["reason"]


async def test_upsert_batch_for_plugin_rejects_disabled_persona(services, data_dir: Path):
    service = services(FakeOwner(network_settings(enable=True)), data_dir)
    service.storage.set_enabled("alice", False)

    result = await service.upsert_batch_for_plugin(
        "alice", characters=[{"ref": "friend_1", "name": "Lin"}]
    )

    assert result == {"updated": False, "reason": "network disabled"}


async def test_get_network_for_plugin_round_trips_and_requires_persona(services, data_dir: Path):
    service = services(FakeOwner(network_settings(enable=True)), data_dir)
    await service.upsert_batch_for_plugin("alice", characters=[{"ref": "c", "name": "Lin"}])

    network = await service.get_network_for_plugin("alice")

    assert set(network) >= {"network", "characters", "identities", "relationships", "life_events"}
    assert [item["name"] for item in network["characters"] if not item["is_persona"]] == ["Lin"]
    with pytest.raises(ValueError, match="persona_id is required"):
        await service.get_network_for_plugin("   ")


async def test_get_context_for_plugin_wraps_and_truncates(services, data_dir: Path):
    service = services(FakeOwner(network_settings(enable=True)), data_dir)
    await service.upsert_batch_for_plugin(
        "alice",
        characters=[
            {"ref": "c", "name": "Lin", "bio": "classmate"},
            {"ref": "long", "name": "Long", "bio": "x" * 2000},
        ],
    )

    context = await service.get_context_for_plugin("alice")

    assert context.startswith("<personal_network_context>")
    assert "name=Lin" in context
    assert len(context) <= 4000

    truncated = await service.get_context_for_plugin("alice", max_chars=500)
    assert len(truncated) == 500
    assert context.startswith(truncated)

    # Out-of-range and unparsable bounds fall back instead of raising.
    assert len(await service.get_context_for_plugin("alice", max_chars=10)) == 500
    assert len(await service.get_context_for_plugin("alice", max_chars="abc")) <= 4000


async def test_record_life_event_from_plugin_enforces_source_guard(services, data_dir: Path):
    service = services(FakeOwner(network_settings(enable=True)), data_dir)
    result = await service.upsert_batch_for_plugin(
        "alice", characters=[{"ref": "c", "name": "Lin"}]
    )
    character_id = result["refs"]["c"]

    with pytest.raises(ValueError, match="source must identify an AstrBot plugin"):
        await service.record_life_event_from_plugin(
            "alice",
            participant_ids=[character_id],
            event_type="chat",
            summary="met",
            occurred_at="2026-07-14T10:00:00+00:00",
            source="other_plugin",
            source_key="k1",
        )

    recorded = await service.record_life_event_from_plugin(
        "alice",
        participant_ids=[character_id],
        event_type="chat",
        summary="met",
        occurred_at="2026-07-14T10:00:00+00:00",
        source="astrbot_plugin_decision",
        source_key="k1",
    )
    assert recorded["created"] is True
    repeated = await service.record_life_event_from_plugin(
        "alice",
        participant_ids=[character_id],
        event_type="chat",
        summary="met",
        occurred_at="2026-07-14T10:00:00+00:00",
        source="astrbot_plugin_decision",
        source_key="k1",
    )
    assert repeated["event_id"] == recorded["event_id"]
    assert repeated["created"] is False


async def test_upsert_batch_for_plugin_reports_disabled_network(services, data_dir: Path):
    service = services(FakeOwner(network_settings(enable=False)), data_dir)

    result = await service.upsert_batch_for_plugin(
        "alice", characters=[{"ref": "c", "name": "Lin"}]
    )

    assert result == {"updated": False, "reason": "network disabled"}


async def test_get_network_for_plugin_returns_empty_shapes_when_disabled(services, data_dir: Path):
    service = services(FakeOwner(network_settings(enable=True)), data_dir)
    service.storage.set_enabled("alice", False)

    assert await service.get_network_for_plugin("alice") == {
        "network": {},
        "characters": [],
        "identities": [],
        "relationships": [],
        "life_events": [],
    }


def test_config_int_clamps_and_falls_back(services, data_dir: Path):
    default_service = services(FakeOwner(network_settings()), data_dir)
    assert default_service._config_int("pronoun_history_messages", 20, 1, 100) == 20

    service = services(FakeOwner(network_settings(pronoun_history_messages=500)), data_dir)
    assert service._config_int("pronoun_history_messages", 20, 1, 100) == 100

    low = services(FakeOwner(network_settings(pronoun_history_messages=-5)), data_dir)
    assert low._config_int("pronoun_history_messages", 20, 1, 100) == 1

    invalid = services(FakeOwner(network_settings(pronoun_history_messages="abc")), data_dir)
    assert invalid._config_int("pronoun_history_messages", 20, 1, 100) == 20


def test_config_merges_defaults_with_the_life_group(services, data_dir: Path):
    service = services(FakeOwner(network_settings(context_max_chars=9000)), data_dir)

    config = service._config()

    assert config["context_max_chars"] == 9000
    assert config["pronoun_history_messages"] == 20
    assert config["context_injection_position"] == "system_prompt"
    assert service.owner.setting_calls[-1] == ("life", {})


def test_config_tolerates_non_mapping_life_settings(services, data_dir: Path):
    owner = FakeOwner()
    owner.life = "not-a-mapping"
    service = services(owner, data_dir)

    assert service._config()["enable"] is False


async def test_inject_context_strips_previous_context_and_tools(services, data_dir: Path):
    service = services(
        FakeOwner(
            network_settings(enable=True, enable_llm_query_tool=False, enable_llm_update_tool=False)
        ),
        data_dir,
    )
    await service.upsert_batch_for_plugin(
        "alice", characters=[{"ref": "c", "name": "Lin", "bio": "classmate"}]
    )
    event = FakeEvent(message_str="Lin 今天来了")
    tools = FakeToolSet(["query_personal_network", "update_personal_network"])
    req = SimpleNamespace(
        contexts=[
            {
                "role": "system",
                "content": "<personal_network_context>old</personal_network_context>",
            },
            {"role": "user", "content": "Lin 今天来了"},
        ],
        system_prompt="base",
        prompt="Lin 今天来了",
        func_tool=tools,
        extra_user_content_parts=[],
    )

    await service.inject_context(event, req)

    assert tools.removed == ["query_personal_network", "update_personal_network"]
    assert req.contexts == [{"role": "user", "content": "Lin 今天来了"}]
    assert "You have access to update_personal_network" in req.system_prompt
    assert "query_personal_network" not in req.system_prompt
    assert "<personal_network_context>" in req.system_prompt
    assert "人物：Lin" in req.system_prompt
    assert req.extra_user_content_parts == []


async def test_inject_context_backtracks_pronoun_history(services, data_dir: Path):
    service = services(FakeOwner(network_settings(enable=True)), data_dir)
    await service.upsert_batch_for_plugin(
        "alice", characters=[{"ref": "c", "name": "Lin", "bio": "classmate"}]
    )
    req = SimpleNamespace(
        contexts=[{"role": "user", "content": "Lin 昨天来过了"}],
        system_prompt="base",
        prompt="他今天来了",
        func_tool=None,
        extra_user_content_parts=[],
    )

    await service.inject_context(FakeEvent(message_str="他今天来了"), req)

    assert "<personal_network_context>" in req.system_prompt
    assert "人物：Lin" in req.system_prompt


async def test_inject_context_skips_when_nothing_matches(services, data_dir: Path):
    service = services(FakeOwner(network_settings(enable=True)), data_dir)
    await service.upsert_batch_for_plugin(
        "alice", characters=[{"ref": "c", "name": "Lin", "bio": "classmate"}]
    )
    req = SimpleNamespace(
        contexts=[],
        system_prompt="base",
        prompt="今天天气不错",
        func_tool=None,
        extra_user_content_parts=[],
    )

    await service.inject_context(FakeEvent(message_str="今天天气不错"), req)

    assert "<personal_network_context>" not in req.system_prompt


async def test_inject_context_uses_user_content_position_and_temp_part(services, data_dir: Path):
    service = services(
        FakeOwner(network_settings(enable=True, context_injection_position="user_content")),
        data_dir,
    )
    await service.upsert_batch_for_plugin(
        "alice", characters=[{"ref": "c", "name": "Lin", "bio": "classmate"}]
    )
    req = SimpleNamespace(
        contexts=[],
        system_prompt="base",
        prompt="Lin 今天来了",
        func_tool=None,
        extra_user_content_parts=[],
    )

    await service.inject_context(FakeEvent(message_str="Lin 今天来了"), req)

    assert len(req.extra_user_content_parts) == 1
    part = req.extra_user_content_parts[0]
    assert part.text.startswith("<personal_network_context>")
    assert getattr(part, "temporary", False) is True
    assert "<personal_network_context>" not in req.system_prompt


async def test_inject_context_falls_back_without_mark_as_temp(
    services, data_dir: Path, monkeypatch
):
    """AstrBot versions without ``mark_as_temp`` get the plain text part."""
    from decision.personal_network import service as service_module

    monkeypatch.setattr(
        service_module,
        "TextPart",
        lambda text="", **kwargs: SimpleNamespace(text=text, type="text", **kwargs),
    )
    service = services(
        FakeOwner(network_settings(enable=True, context_injection_position="user_content")),
        data_dir,
    )
    await service.upsert_batch_for_plugin(
        "alice", characters=[{"ref": "c", "name": "Lin", "bio": "classmate"}]
    )
    req = SimpleNamespace(
        contexts=[],
        system_prompt="base",
        prompt="Lin 今天来了",
        func_tool=None,
        extra_user_content_parts=[],
    )

    await service.inject_context(FakeEvent(message_str="Lin 今天来了"), req)

    part = req.extra_user_content_parts[0]
    assert part.text.startswith("<personal_network_context>")
    assert not hasattr(part, "mark_as_temp")


async def test_inject_context_respects_disabled_network(services, data_dir: Path):
    service = services(FakeOwner(network_settings(enable=False)), data_dir)
    req = SimpleNamespace(
        contexts=[{"role": "system", "content": "<personal_network_context>old"}],
        system_prompt="base",
        prompt="hi",
        func_tool=None,
        extra_user_content_parts=[],
    )

    await service.inject_context(FakeEvent(), req)

    assert req.contexts == []
    assert req.system_prompt == "base"


async def test_query_tool_reports_disabled_switches(services, data_dir: Path):
    disabled = services(FakeOwner(network_settings(enable=False)), data_dir)
    assert await disabled.query_tool(FakeEvent()) == "人际网络已禁用。"

    query_off = services(
        FakeOwner(network_settings(enable=True, enable_llm_query_tool=False)), data_dir
    )
    assert await query_off.query_tool(FakeEvent()) == "LLM 主动查询人际网络工具已禁用。"


async def test_query_tool_and_command_return_relationship_text(services, data_dir: Path):
    service = services(FakeOwner(network_settings(enable=True)), data_dir)
    await service.upsert_batch_for_plugin(
        "alice",
        characters=[{"ref": "c", "name": "Lin"}],
        relationships=[
            {
                "source": "persona",
                "target": "c",
                "type": "朋友",
                "strength": 60,
                "status": "active",
                "description": "",
            }
        ],
    )

    tool_text = await service.query_tool(FakeEvent(), "Lin")
    command_text = await service.query_command_text(FakeEvent(), "Lin")

    assert "Lin" in tool_text
    assert "Lin" in command_text


async def test_update_tool_returns_json_and_validates(services, data_dir: Path):
    service = services(FakeOwner(network_settings(enable=True)), data_dir)

    payload = json.loads(
        await service.update_tool(FakeEvent(), characters=[{"ref": "c", "name": "Lin"}])
    )

    assert payload["updated"] is True
    assert payload["refs"]["c"]

    empty = json.loads(await service.update_tool(FakeEvent()))
    assert empty == {"updated": False, "reason": "no network updates supplied"}


async def test_update_tool_reports_disabled_network(services, data_dir: Path):
    service = services(FakeOwner(network_settings(enable=False)), data_dir)

    assert json.loads(await service.update_tool(FakeEvent())) == {
        "updated": False,
        "reason": "network disabled",
    }


async def test_on_astrbot_loaded_marks_orphaned_personas(services, data_dir: Path):
    service = services(FakeOwner(network_settings(enable=True)), data_dir)
    service.storage.ensure_network("alice", "Alice")
    service.storage.ensure_network("orphan", "Orphan")

    await service.on_astrbot_loaded()

    networks = {item["persona_id"]: item for item in service.storage.list_networks()}
    assert set(networks) == {"alice", "orphan"}
    assert networks["alice"]["persona_missing"] == 0
    assert networks["orphan"]["persona_missing"] == 1


async def test_terminate_cancels_generation_and_closes_storage(services, data_dir: Path):
    service = services(FakeOwner(network_settings(enable=True)), data_dir)

    async def pending() -> None:
        await asyncio.sleep(60)

    task = asyncio.create_task(pending())
    service._generation_tasks.add(task)

    await service.terminate()

    assert task.cancelled()
    assert service._generation_tasks == set()
    assert service._generation_jobs == {}


# ----------------------------------------------------------------------
# WebUI surface
# ----------------------------------------------------------------------
async def test_api_personas_lists_active_personas(services, data_dir: Path):
    service = services(FakeOwner(network_settings(enable=True)), data_dir)

    response = await service.api_personas()

    assert [item["persona_id"] for item in response["personas"]] == ["alice", "default"]
    assert response["personas"][0]["key"]


async def test_api_network_returns_avatar_data_and_errors(services, data_dir: Path):
    service = services(FakeOwner(network_settings(enable=True)), data_dir)
    await service.upsert_batch_for_plugin("alice", characters=[{"ref": "c", "name": "Lin"}])
    stub = reset_request()
    stub.query = {"persona_id": "alice"}

    response = await service.api_network()

    assert any(item["name"] == "Lin" for item in response["characters"])
    assert all(item["avatar_data"] is None for item in response["characters"])

    stub.query = {}
    assert (await service.api_network())["status_code"] == 500


async def test_api_set_enabled_persists_boolean(services, data_dir: Path):
    service = services(FakeOwner(network_settings(enable=True)), data_dir)
    stub = reset_request()
    stub.json_body = {"persona_id": "alice", "enabled": False}

    assert await service.api_set_enabled() == {"saved": True}
    assert service.storage.is_enabled("alice") is False

    stub.json_body = {"persona_id": "alice", "enabled": "yes"}
    assert "boolean" in (await service.api_set_enabled())["error"]


async def test_api_save_and_delete_character(services, data_dir: Path):
    service = services(FakeOwner(network_settings(enable=True)), data_dir)
    stub = reset_request()
    stub.json_body = {"persona_id": "alice", "ref": "c", "name": "Lin", "bio": "classmate"}

    saved = await service.api_save_character()
    character_id = saved["refs"]["c"]
    assert saved["refs"]["persona"]

    stub.json_body = {"persona_id": "alice", "character_id": character_id}
    assert await service.api_delete_character() == {"deleted": True}
    names = [
        item["name"]
        for item in service.storage.get_network("alice")["characters"]
        if not item["is_persona"]
    ]
    assert names == []

    stub.json_body = {"persona_id": "alice", "character_id": "missing"}
    assert (await service.api_delete_character())["status_code"] == 500


async def test_api_save_relationship_and_delete(services, data_dir: Path):
    service = services(FakeOwner(network_settings(enable=True)), data_dir)
    await service.upsert_batch_for_plugin("alice", characters=[{"ref": "c", "name": "Lin"}])
    network = service.storage.get_network("alice")
    root_id = next(item["id"] for item in network["characters"] if item["is_persona"])
    character_id = next(item["id"] for item in network["characters"] if not item["is_persona"])
    stub = reset_request()
    stub.json_body = {
        "persona_id": "alice",
        "source": root_id,
        "target": character_id,
        "type": "朋友",
        "strength": 60,
        "status": "active",
        "description": "",
    }

    saved = await service.api_save_relationship()

    assert saved["relationship_ids"]
    stub.json_body = {
        "persona_id": "alice",
        "relationship_id": saved["relationship_ids"][0],
    }
    assert await service.api_delete_relationship() == {"deleted": True}
    assert service.storage.get_network("alice")["relationships"] == []


async def test_api_save_and_delete_life_event(services, data_dir: Path):
    service = services(FakeOwner(network_settings(enable=True)), data_dir)
    await service.upsert_batch_for_plugin(
        "alice", characters=[{"ref": "c", "name": "Lin"}, {"ref": "d", "name": "Mei"}]
    )
    network = service.storage.get_network("alice")
    ids = [item["id"] for item in network["characters"] if not item["is_persona"]]
    stub = reset_request()
    stub.json_body = {
        "persona_id": "alice",
        "participants": ids,
        "type": "见面",
        "summary": "一起喝咖啡",
        "occurred_at": "2026-07-14T10:00:00+00:00",
        "importance": 60,
        "emotional_tone": "轻松",
    }

    saved = await service.api_save_life_event()

    assert saved["event_ids"]
    stub.json_body = {"persona_id": "alice", "event_id": saved["event_ids"][0]}
    assert await service.api_delete_life_event() == {"deleted": True}
    assert service.storage.get_network("alice")["life_events"] == []


async def test_api_merge_characters_folds_duplicate_into_target(services, data_dir: Path):
    service = services(FakeOwner(network_settings(enable=True)), data_dir)
    refs = (
        await service.upsert_batch_for_plugin(
            "alice", characters=[{"ref": "a", "name": "Lin"}, {"ref": "b", "name": "Lin B"}]
        )
    )["refs"]
    stub = reset_request()
    stub.json_body = {
        "persona_id": "alice",
        "target_id": refs["a"],
        "duplicate_id": refs["b"],
    }

    assert await service.api_merge_characters() == {"merged": True}
    remaining = {item["id"] for item in service.storage.get_network("alice")["characters"]}
    assert refs["b"] not in remaining
    assert refs["a"] in remaining


async def test_api_generate_network_starts_job(services, data_dir: Path):
    service = services(FakeOwner(network_settings(enable=True)), data_dir)
    service.storage.ensure_network("alice", "Alice")
    stub = reset_request()
    stub.json_body = {"persona_id": "alice", "count": 3, "density": "sparse"}

    response = await service.api_generate_network()

    assert response["job_id"]
    assert response["expected_count"] == 3
    assert response["density"] == "sparse"
    assert response["reused"] is False
    await service.terminate()


async def test_api_generate_network_rejects_bad_count(services, data_dir: Path):
    service = services(FakeOwner(network_settings(enable=True)), data_dir)
    stub = reset_request()
    stub.json_body = {"persona_id": "alice", "count": 0}

    assert "1-" in (await service.api_generate_network())["error"]

    stub.json_body = {"persona_id": "alice", "count": 6}
    assert (await service.api_generate_network())["error"] == "当前人格的关系网络未启用"


async def test_api_avatar_reports_missing_avatar(services, data_dir: Path):
    service = services(FakeOwner(network_settings(enable=True)), data_dir)
    await service.upsert_batch_for_plugin("alice", characters=[{"ref": "c", "name": "Lin"}])
    stub = reset_request()
    stub.method = "GET"
    stub.query = {"persona_id": "alice"}

    response = await service.api_avatar("unknown-character")

    assert response == {"error": "avatar not found", "status_code": 404}


async def test_api_avatar_upload_requires_file(services, data_dir: Path):
    service = services(FakeOwner(network_settings(enable=True)), data_dir)
    stub = reset_request()
    stub.method = "POST"
    stub.query = {"persona_id": "alice"}

    response = await service.api_avatar("any")

    assert response["error"] == "missing avatar file"


def test_avatar_data_and_unlink_are_path_scoped(services, data_dir: Path):
    service = services(FakeOwner(network_settings(enable=True)), data_dir)

    assert service._avatar_data(None) is None
    assert service._avatar_data("../escape.webp") is None
    assert service._avatar_data("missing.webp") is None

    (service.avatar_dir / "ok.webp").write_bytes(b"webp-bytes")
    assert service._avatar_data("ok.webp").startswith("data:image/webp;base64,")
    service._unlink_avatar("ok.webp")
    assert not (service.avatar_dir / "ok.webp").exists()
    service._unlink_avatar("../escape.webp")
    service._unlink_avatar(None)


def test_find_character_persona_resolves_owner(services, data_dir: Path):
    service = services(FakeOwner(network_settings(enable=True)), data_dir)
    result = service.storage.upsert_batch("alice", [{"ref": "c", "name": "Lin"}])

    assert service._find_character_persona(result["refs"]["c"]) == "alice"
    with pytest.raises(ValueError, match="character not found"):
        service._find_character_persona("missing")


def test_validate_import_rejects_wrong_schema_version(services, data_dir: Path):
    service = services(FakeOwner(network_settings(enable=True)), data_dir)

    with pytest.raises(ValueError, match="schema_version 4"):
        service._validate_import({"schema_version": 3})
    with pytest.raises(ValueError, match="export arrays are missing"):
        service._validate_import({"schema_version": 4})


def test_decode_avatar_data_validates_webp(services, data_dir: Path):
    service = services(FakeOwner(network_settings(enable=True)), data_dir)

    with pytest.raises(ValueError, match="WebP data URL"):
        service._decode_avatar_data("data:image/png;base64,AAAA")
    with pytest.raises(ValueError, match="not valid base64"):
        service._decode_avatar_data("data:image/webp;base64,!!!!")
    with pytest.raises(ValueError, match="not a valid image"):
        service._decode_avatar_data("data:image/webp;base64,QUJD")


async def test_persona_prompt_reads_persona_manager(services, data_dir: Path):
    service = services(FakeOwner(network_settings(enable=True)), data_dir)

    assert await service._persona_prompt("alice") == "prompt for alice"


async def test_persona_prompt_raises_without_a_prompt(services, data_dir: Path):
    owner = FakeOwner(network_settings(enable=True))
    owner.context.persona_manager = SimpleNamespace(
        get_persona=lambda persona_id: SimpleNamespace(system_prompt=""),
        get_persona_v3_by_id=lambda persona_id: {},
    )
    service = services(owner, data_dir)

    with pytest.raises(ValueError, match="无法读取所选 AstrBot 人格设定"):
        await service._persona_prompt("alice")


def test_generation_helpers_require_provider(services, data_dir: Path):
    service = services(FakeOwner(network_settings(enable=True)), data_dir)

    with pytest.raises(ValueError, match="没有可用的 LLM Provider"):
        service._generation_provider()
    with pytest.raises(ValueError, match="空结果"):
        service._completion_text(SimpleNamespace())
    assert service._completion_text(SimpleNamespace(completion_text=" text ")) == "text"


def test_prune_generation_jobs_drops_expired_results(services, data_dir: Path):
    service = services(FakeOwner(network_settings(enable=True)), data_dir)
    service._generation_jobs["done"] = {"status": "completed", "updated_at": 0.0}
    service._generation_jobs["live"] = {"status": "running", "updated_at": time.monotonic()}

    service._prune_generation_jobs()

    assert set(service._generation_jobs) == {"live"}


async def test_api_validate_generation_accepts_draft_text(services, data_dir: Path):
    service = services(FakeOwner(network_settings(enable=True)), data_dir)
    stub = reset_request()
    stub.json_body = {
        "persona_id": "alice",
        "allow_fill_existing": False,
        "raw": json.dumps(
            {
                "characters": [{"ref": "c", "name": "Lin"}],
                "relationships": [
                    {
                        "source": "persona",
                        "target": "c",
                        "type": "朋友",
                        "strength": 60,
                        "status": "active",
                        "description": "",
                    }
                ],
            }
        ),
    }

    response = await service.api_validate_generation()

    assert response["valid"] is True
    assert response["draft"]["characters"][0]["name"] == "Lin"

    stub.json_body = {"allow_fill_existing": False, "raw": "{}"}
    invalid = await service.api_validate_generation()
    assert invalid["valid"] is False
    assert "persona_id is required" in invalid["errors"][0]


async def test_api_generation_status_rejects_unknown_job(services, data_dir: Path):
    service = services(FakeOwner(network_settings(enable=True)), data_dir)
    stub = reset_request()
    stub.json_body = {"persona_id": "alice", "job_id": "missing"}

    response = await service.api_generation_status()

    assert "生成任务不存在或已过期" in response["error"]


async def test_api_apply_generation_rejects_bad_draft(services, data_dir: Path):
    service = services(FakeOwner(network_settings(enable=True)), data_dir)
    stub = reset_request()
    stub.json_body = {"persona_id": "alice", "allow_fill_existing": False, "draft": []}

    response = await service.api_apply_generation()

    assert response["applied"] is False
    assert "draft 必须是 JSON 对象" in response["errors"][0]


async def test_api_export_writes_version_four_payload(services, data_dir: Path):
    service = services(FakeOwner(network_settings(enable=True)), data_dir)
    await service.upsert_batch_for_plugin("alice", characters=[{"ref": "c", "name": "Lin"}])
    stub = reset_request()
    stub.query = {"persona_id": "alice"}

    response = await service.api_export()

    payload = json.loads(Path(response["path"]).read_text(encoding="utf-8"))
    assert payload["schema_version"] == 4
    assert payload["persona_id"] == "alice"
    assert response["filename"] == "personal-network-alice.json"


async def test_api_import_preview_and_apply(services, data_dir: Path):
    service = services(FakeOwner(network_settings(enable=True)), data_dir)
    await service.upsert_batch_for_plugin("alice", characters=[{"ref": "c", "name": "Lin"}])
    stub = reset_request()
    stub.query = {"persona_id": "alice"}
    export = await service.api_export()
    raw = Path(export["path"]).read_bytes()

    persona_key = base64.urlsafe_b64encode(b"alice").decode().rstrip("=")
    stub.query = {}
    stub.files_map = {"file": PluginUploadFile(raw)}

    preview = await service.api_import_preview(persona_key)

    assert preview["characters"] == 2
    assert preview["can_apply"] is True
    assert preview["new_characters"] == 0

    stub.json_body = {"token": preview["token"]}
    applied = await service.api_import_apply()

    assert applied["imported"] is True


async def test_api_import_apply_rejects_unknown_token(services, data_dir: Path):
    service = services(FakeOwner(network_settings(enable=True)), data_dir)
    stub = reset_request()
    stub.json_body = {"token": "missing"}

    assert "expired" in (await service.api_import_apply())["error"]


async def test_api_import_preview_expires_and_is_pruned(services, data_dir: Path):
    service = services(FakeOwner(network_settings(enable=True)), data_dir)
    service._import_previews["old"] = (time.monotonic() - 3600, "alice", {})
    stub = reset_request()
    stub.json_body = {"token": "old"}

    response = await service.api_import_apply()

    assert "expired" in response["error"]
    assert "old" not in service._import_previews


async def test_api_import_preview_rejects_invalid_key(services, data_dir: Path):
    service = services(FakeOwner(network_settings(enable=True)), data_dir)
    reset_request()

    response = await service.api_import_preview("!!!not-base64!!!")

    assert response["error"] == "invalid persona key"


def test_decode_persona_key_round_trips():
    key = base64.urlsafe_b64encode("人格".encode()).decode().rstrip("=")

    assert PersonalNetworkService._decode_persona_key(key) == "人格"


# ----------------------------------------------------------------------
# One-time migration
# ----------------------------------------------------------------------
def seed_legacy(tmp_path: Path, *, character: str = "Legacy Person") -> Path:
    """Create a standalone plugin directory with a database and avatar."""
    legacy = tmp_path / PLUGIN_DATA / LEGACY_DIR
    legacy.mkdir(parents=True)
    storage = NetworkStorage(legacy / "personal_network.sqlite3")
    storage.ensure_network("legacy_persona", "Legacy")
    storage.upsert_batch("legacy_persona", [{"ref": "old", "name": character}])
    storage.close()
    (legacy / "avatars").mkdir()
    (legacy / "avatars" / "avatar.webp").write_bytes(b"webp-bytes")
    return legacy


def test_migration_copies_legacy_database_and_avatars(services, data_dir: Path, tmp_path: Path):
    legacy = seed_legacy(tmp_path)

    service = services(FakeOwner(network_settings(enable=True)), data_dir)

    network = service.storage.get_network("legacy_persona")
    assert [item["name"] for item in network["characters"] if not item["is_persona"]] == [
        "Legacy Person"
    ]
    assert (service.avatar_dir / "avatar.webp").read_bytes() == b"webp-bytes"
    marker = json.loads((data_dir / "migrated_from_standalone.json").read_text(encoding="utf-8"))
    assert marker["from"] == str(legacy)
    assert marker["migrated_at"]


def test_migration_is_idempotent(services, data_dir: Path, tmp_path: Path):
    legacy = seed_legacy(tmp_path)
    first = services(FakeOwner(network_settings(enable=True)), data_dir)
    first.storage.close()
    marker_before = (data_dir / "migrated_from_standalone.json").read_text(encoding="utf-8")

    # A later legacy write must never be re-imported once the marker exists.
    storage = NetworkStorage(legacy / "personal_network.sqlite3")
    storage.upsert_batch("legacy_persona", [{"ref": "new", "name": "Too Late"}])
    storage.close()
    (service := services(FakeOwner(network_settings(enable=True)), data_dir))

    names = [
        item["name"]
        for item in service.storage.get_network("legacy_persona")["characters"]
        if not item["is_persona"]
    ]
    assert names == ["Legacy Person"]
    assert (data_dir / "migrated_from_standalone.json").read_text(encoding="utf-8") == marker_before


def test_migration_is_skipped_when_new_database_exists(services, data_dir: Path, tmp_path: Path):
    seed_legacy(tmp_path)
    data_dir.mkdir(parents=True)
    storage = NetworkStorage(data_dir / "personal_network.sqlite3")
    storage.ensure_network("alice", "Alice")
    storage.close()

    services(FakeOwner(network_settings(enable=True)), data_dir)

    assert not (data_dir / "migrated_from_standalone.json").exists()


def test_migration_failure_never_blocks_construction(
    services, data_dir: Path, tmp_path: Path, monkeypatch
):
    seed_legacy(tmp_path)

    def explode(*args: Any, **kwargs: Any) -> None:
        raise OSError("disk on fire")

    monkeypatch.setattr("decision.personal_network.service.shutil.copy2", explode)

    service = services(FakeOwner(network_settings(enable=True)), data_dir)

    assert service.storage.get_network("alice") is not None
    assert not (data_dir / "migrated_from_standalone.json").exists()


def test_migration_without_legacy_directory_is_a_noop(services, data_dir: Path):
    services(FakeOwner(network_settings(enable=True)), data_dir)

    assert not (data_dir / "migrated_from_standalone.json").exists()
