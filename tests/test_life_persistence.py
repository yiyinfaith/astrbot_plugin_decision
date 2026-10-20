"""Failure and cancellation boundaries for living data outside plugin code."""

from __future__ import annotations

import asyncio
import copy
import json
import threading
from unittest.mock import AsyncMock

import pytest

from decision.virtual_life.core.long_term import LongTermTimelineStore
from decision.virtual_life.core.storage import JsonRepository


@pytest.mark.asyncio
async def test_failed_replace_preserves_file_and_removes_temporary(tmp_path, monkeypatch):
    repo = JsonRepository(tmp_path / "state.json", {})
    await repo.save({"old": True})

    def fail(*args):
        raise OSError("disk unavailable")

    monkeypatch.setattr("decision.virtual_life.core.storage.os.replace", fail)
    with pytest.raises(OSError):
        await repo.save({"old": False})
    assert json.loads(repo.path.read_text()) == {"old": True}
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.asyncio
async def test_cancelled_save_finishes_atomic_commit_before_return(tmp_path, monkeypatch):
    import os

    original = os.replace
    started = threading.Event()
    release = threading.Event()

    def paused_replace(*args):
        started.set()
        assert release.wait(3)
        original(*args)

    monkeypatch.setattr("decision.virtual_life.core.storage.os.replace", paused_replace)
    repo = JsonRepository(tmp_path / "state.json", {})
    task = asyncio.create_task(repo.save({"committed": True}))
    try:
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert json.loads(repo.path.read_text()) == {"committed": True}
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.asyncio
async def test_failed_long_term_approval_keeps_draft_and_old_stages(tmp_path):
    store = LongTermTimelineStore(tmp_path)
    store.stages = [{"id": "old", "persona_id": "alice"}]
    await store.set_draft(
        "alice",
        [{"id": "new", "persona_id": "alice"}],
        source="json",
        admin_umo="qq:FriendMessage:1",
        created_at="now",
        mode="replace_all",
    )
    before = copy.deepcopy((store.stages, store.drafts, store.notification_targets))
    old_file = store.repo.path.read_bytes()
    store.repo.save = AsyncMock(side_effect=OSError("disk unavailable"))
    with pytest.raises(OSError):
        await store.approve_draft("alice", "qq:FriendMessage:1")
    assert (store.stages, store.drafts, store.notification_targets) == before
    assert store.repo.path.read_bytes() == old_file
