"""Persistent cache for media understanding results and the recent media index.

Entries are scoped by conversation UMO so one group's media can never be read from
another group's context. The store is a single small JSON file written atomically,
which keeps it dependency free and easy to inspect.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from .models import MediaIndexEntry, Understanding

# How many recent media rows to remember per conversation.
INDEX_PER_SESSION = 60


class MultimodalCache:
    """TTL cache plus a bounded per-session media index."""

    def __init__(
        self,
        path: Path,
        *,
        ttl_seconds: int = 12 * 3600,
        max_entries: int = 500,
    ) -> None:
        self.path = Path(path)
        self.ttl_seconds = max(0, int(ttl_seconds))
        self.max_entries = max(10, int(max_entries))
        self._lock = asyncio.Lock()
        self._loaded = False
        self._entries: dict[str, Understanding] = {}
        self._index: dict[str, list[MediaIndexEntry]] = {}

    # ------------------------------------------------------------------ io
    async def load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            raw = await asyncio.to_thread(self.path.read_text, "utf-8")
            payload = json.loads(raw)
        except (OSError, ValueError):
            return
        if not isinstance(payload, dict):
            return
        for value in payload.get("entries", []):
            if not isinstance(value, dict):
                continue
            try:
                item = Understanding.from_dict(value)
            except (TypeError, ValueError):
                continue
            if item.key:
                self._entries[item.key] = item
        for umo, rows in (payload.get("index") or {}).items():
            if not isinstance(rows, list):
                continue
            bucket = []
            for value in rows:
                if not isinstance(value, dict):
                    continue
                try:
                    bucket.append(MediaIndexEntry.from_dict(value))
                except (TypeError, ValueError):
                    continue
            if bucket:
                self._index[str(umo)] = bucket
        self._prune(time.time())

    async def save(self) -> None:
        payload = {
            "version": 1,
            "entries": [item.to_dict() for item in self._entries.values()],
            "index": {
                umo: [row.to_dict() for row in rows] for umo, rows in self._index.items()
            },
        }
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        self.path.parent.mkdir(parents=True, exist_ok=True)

        def write() -> None:
            handle = tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=str(self.path.parent),
                prefix=".mm-",
                suffix=".tmp",
                delete=False,
            )
            try:
                with handle as stream:
                    stream.write(data)
                os.replace(handle.name, self.path)
            except BaseException:
                try:
                    os.unlink(handle.name)
                except OSError:
                    pass
                raise

        await asyncio.to_thread(write)

    # --------------------------------------------------------------- prune
    def _prune(self, now: float) -> None:
        if self.ttl_seconds:
            expired = [
                key
                for key, item in self._entries.items()
                if item.expires_at and item.expires_at < now
            ]
            for key in expired:
                self._entries.pop(key, None)
        if len(self._entries) > self.max_entries:
            ordered = sorted(
                self._entries.values(), key=lambda item: item.created_at or 0
            )
            for item in ordered[: len(self._entries) - self.max_entries]:
                self._entries.pop(item.key, None)
        cutoff = now - 7 * 24 * 3600
        for umo, rows in list(self._index.items()):
            kept = [row for row in rows if row.received_at >= cutoff]
            if kept:
                self._index[umo] = kept[-INDEX_PER_SESSION:]
            else:
                self._index.pop(umo, None)

    # --------------------------------------------------------------- reads
    async def get(self, key: str) -> Understanding | None:
        async with self._lock:
            await self.load()
            item = self._entries.get(key)
            if item is None:
                return None
            if item.expires_at and item.expires_at < time.time():
                self._entries.pop(key, None)
                return None
            return item

    async def get_for_session(self, umo: str, keys: list[str]) -> list[Understanding]:
        found = []
        for key in keys:
            item = await self.get(key)
            if item is not None and item.umo == umo:
                found.append(item)
        return found

    async def recent_media(self, umo: str, limit: int = 20) -> list[MediaIndexEntry]:
        async with self._lock:
            await self.load()
            rows = list(self._index.get(umo, []))
        return rows[-limit:][::-1]

    # -------------------------------------------------------------- writes
    async def put(self, item: Understanding) -> None:
        async with self._lock:
            await self.load()
            now = time.time()
            item.created_at = item.created_at or now
            if self.ttl_seconds:
                item.expires_at = item.created_at + self.ttl_seconds
            self._entries[item.key] = item
            self._prune(now)
            await self.save()

    async def remember_media(self, rows: list[MediaIndexEntry]) -> None:
        if not rows:
            return
        async with self._lock:
            await self.load()
            for row in rows:
                bucket = self._index.setdefault(row.umo, [])
                if any(
                    existing.content_key == row.content_key
                    and existing.message_id == row.message_id
                    for existing in bucket
                ):
                    continue
                bucket.append(row)
                del bucket[:-INDEX_PER_SESSION]
            self._prune(time.time())
            await self.save()

    async def clear_session(self, umo: str) -> int:
        async with self._lock:
            await self.load()
            removed = [
                key for key, item in self._entries.items() if item.umo == umo
            ]
            for key in removed:
                self._entries.pop(key, None)
            self._index.pop(umo, None)
            if removed:
                await self.save()
        return len(removed)

    async def stats(self) -> dict[str, Any]:
        async with self._lock:
            await self.load()
            return {
                "entries": len(self._entries),
                "sessions": len(self._index),
                "ttl_seconds": self.ttl_seconds,
                "max_entries": self.max_entries,
                "path": str(self.path),
            }
