from __future__ import annotations

import asyncio
import json
import os
import uuid
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from .models import DailyPlan, ScheduleRequirement, SessionState

SESSION_KEY_SEPARATOR = "\x00"


def session_key(umo: str, persona_id: str) -> str:
    """Return the durable key for one conversation and one persona.

    UMO values are platform supplied strings; a NUL cannot occur in a valid
    UMO, so it is a safe separator that cannot collide with a user/persona
    value.  The old sessions.json used bare UMO keys and is migrated on load.
    """

    return f"{str(umo)}{SESSION_KEY_SEPARATOR}{str(persona_id or 'default')}"


def split_session_key(value: str) -> tuple[str, str] | None:
    if SESSION_KEY_SEPARATOR not in str(value):
        return None
    umo, persona = str(value).rsplit(SESSION_KEY_SEPARATOR, 1)
    return (umo, persona or "default") if umo else None


class SessionStore(dict[str, SessionState]):
    """Persona-scoped session map with a narrow legacy lookup fallback.

    The fallback keeps older integrations/tests that access ``sessions[umo]``
    working while runtime code uses ``get_for``/``set_for`` exclusively.  It
    only returns a bare-UMO value when exactly one persona state exists, so it
    can never silently choose between two personas.
    """

    def key_for(self, umo: str, persona_id: str) -> str:
        return session_key(umo, persona_id)

    def get_for(self, umo: str, persona_id: str, default=None):
        value = dict.get(self, session_key(umo, persona_id))
        if value is not None:
            return value
        # Compatibility for callers that inserted a pre-migration bare UMO
        # key directly into the map.  It is only accepted when its recorded
        # persona matches the requested scope.
        legacy = dict.get(self, str(umo))
        if legacy is not None and (legacy.persona_id or "default") == str(persona_id or "default"):
            return legacy
        return default

    def set_for(self, umo: str, persona_id: str, value: SessionState):
        self[session_key(umo, persona_id)] = value

    def _legacy_match(self, umo: str):
        prefix = f"{str(umo)}{SESSION_KEY_SEPARATOR}"
        matches = [value for key, value in dict.items(self) if key.startswith(prefix)]
        return matches[0] if len(matches) == 1 else None

    def __getitem__(self, key):
        try:
            return dict.__getitem__(self, key)
        except KeyError:
            value = self._legacy_match(str(key))
            if value is None:
                raise
            return value

    def get(self, key, default=None):
        value = dict.get(self, key)
        if value is not None:
            return value
        return self._legacy_match(str(key)) or default


class JsonRepository:
    def __init__(self, path: Path, default: dict[str, Any]):
        self.path = path
        self.default = default
        self.lock = asyncio.Lock()

    async def load(self) -> dict[str, Any]:
        async with self.lock:
            if not self.path.exists():
                return json.loads(json.dumps(self.default))
            try:
                content = await asyncio.to_thread(self.path.read_text, encoding="utf-8")
                value = json.loads(content)
                return value if isinstance(value, dict) else json.loads(json.dumps(self.default))
            except (OSError, json.JSONDecodeError):
                return json.loads(json.dumps(self.default))

    async def save(self, value: dict[str, Any]) -> None:
        async with self.lock:
            content = json.dumps(value, ensure_ascii=False, indent=2)

            def commit():
                self.path.parent.mkdir(parents=True, exist_ok=True)
                temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
                try:
                    with temporary.open("w", encoding="utf-8") as file:
                        file.write(content)
                        file.flush()
                        os.fsync(file.fileno())
                    os.replace(temporary, self.path)
                finally:
                    temporary.unlink(missing_ok=True)

            # Finish the atomic write before releasing the lock, even during
            # plugin hot reload. A cancelled to_thread still runs in its thread.
            task = asyncio.create_task(asyncio.to_thread(commit))
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                await task
                raise


class PluginStorage:
    def __init__(self, data_dir: Path):
        self.plans_repo = JsonRepository(
            data_dir / "plans.json",
            {"schema_version": 3, "plans": {}, "schedule_requirements": {}},
        )
        self.sessions_repo = JsonRepository(
            data_dir / "sessions.json", {"schema_version": 1, "sessions": {}}
        )
        self.plans: dict[str, DailyPlan] = {}
        self.schedule_requirements: dict[str, ScheduleRequirement] = {}
        self.sessions: SessionStore = SessionStore()

    async def load(self) -> None:
        plans = await self.plans_repo.load()
        sessions = await self.sessions_repo.load()
        self.plans = {}
        for key, value in plans.get("plans", {}).items():
            try:
                self.plans[key] = DailyPlan.from_dict(value)
            except (KeyError, TypeError, ValueError):
                continue
        self.schedule_requirements = {}
        for value in plans.get("schedule_requirements", {}).values():
            try:
                requirement = ScheduleRequirement.from_dict(value)
                self.schedule_requirements[requirement.id] = requirement
            except (TypeError, ValueError):
                continue
        self.sessions = SessionStore()
        for raw_key, value in sessions.get("sessions", {}).items():
            try:
                state = SessionState.from_dict(value)
            except (TypeError, ValueError):
                continue
            # Legacy files keyed states by UMO.  Their stored persona_id is
            # the only safe migration scope; never merge a legacy state into a
            # newly selected persona.
            parsed = split_session_key(str(raw_key))
            if parsed is None:
                umo, persona = str(raw_key), state.persona_id or "default"
            else:
                umo, persona = parsed
            state.persona_id = persona
            self.sessions.set_for(umo, persona, state)

    @staticmethod
    def plan_key(date_str: str, persona_id: str) -> str:
        return f"{date_str}::{persona_id}"

    def get_plan(self, date_str: str, persona_id: str) -> DailyPlan | None:
        return self.plans.get(self.plan_key(date_str, persona_id))

    def get_recent_plans(self, persona_id: str, before: date, days: int) -> list[DailyPlan]:
        result: list[DailyPlan] = []
        for offset in range(1, max(0, days) + 1):
            plan = self.get_plan((before - timedelta(days=offset)).isoformat(), persona_id)
            if plan and plan.status == "ok":
                result.append(plan)
        result.reverse()
        return result

    def active_schedule_requirements(
        self, persona_id: str, target: date
    ) -> list[ScheduleRequirement]:
        target_str = target.isoformat()
        result = [
            item
            for item in self.schedule_requirements.values()
            if item.persona_id == persona_id
            and item.start_date <= target_str <= item.end_date
            and target_str not in item.consumed_dates
        ]
        return sorted(result, key=lambda item: (item.created_at, item.id))

    def consume_schedule_requirements(self, requirement_ids: list[str], target: date) -> int:
        target_str = target.isoformat()
        changed = 0
        for requirement_id in requirement_ids:
            item = self.schedule_requirements.get(requirement_id)
            if (
                item is None
                or not item.start_date <= target_str <= item.end_date
                or target_str in item.consumed_dates
            ):
                continue
            consumed_dates = tuple(sorted((*item.consumed_dates, target_str)))
            expected_days = (
                date.fromisoformat(item.end_date) - date.fromisoformat(item.start_date)
            ).days + 1
            if len(consumed_dates) >= expected_days:
                del self.schedule_requirements[requirement_id]
            else:
                self.schedule_requirements[requirement_id] = replace(
                    item, consumed_dates=consumed_dates
                )
            changed += 1
        return changed

    def prune_schedule_requirements(self, today: date) -> int:
        expired = [
            key
            for key, item in self.schedule_requirements.items()
            if date.fromisoformat(item.end_date) < today
        ]
        for key in expired:
            del self.schedule_requirements[key]
        return len(expired)

    async def save_plans(self) -> None:
        await self.plans_repo.save(
            {
                "schema_version": 3,
                "plans": {key: value.to_dict() for key, value in self.plans.items()},
                "schedule_requirements": {
                    key: value.to_dict() for key, value in self.schedule_requirements.items()
                },
            }
        )

    async def save_sessions(self) -> None:
        await self.sessions_repo.save(
            {
                "schema_version": 1,
                "sessions": {key: value.to_dict() for key, value in self.sessions.items()},
            }
        )
