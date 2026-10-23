"""Data structures shared by the multi-modal pipeline."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

# A media item as seen in one incoming message.
MEDIA_KINDS = ("image", "video", "audio", "file")


@dataclass(slots=True)
class MediaRef:
    """One media attachment discovered in an incoming message."""

    kind: str
    reference: str
    """Provider-facing reference: a local path, file:/// URI, http(s) URL or base64."""
    urn: str
    """"umo::msg_id::index" identity used for lookups."""
    umo: str = ""
    message_id: str = ""
    name: str = ""
    mime: str = ""
    index: int = 0
    received_at: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def content_key(self) -> str:
        """Stable cache key.

        HTTP references are hashed whole; local files are hashed by their path,
        which is enough because the platform writes a fresh file per message.
        """

        digest = hashlib.sha256(f"{self.kind}|{self.reference}".encode()).hexdigest()[:32]
        return digest


@dataclass(slots=True)
class Understanding:
    """The result of understanding one media item."""

    key: str
    kind: str
    umo: str
    summary: str
    detail: str = ""
    provider_id: str = ""
    model: str = ""
    message_id: str = ""
    name: str = ""
    created_at: float = 0.0
    expires_at: float = 0.0
    error: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.summary)

    def render(self, *, limit: int | None = None, with_detail: bool = False) -> str:
        """Human-readable block for context injection."""

        body = self.summary if not with_detail else (self.detail or self.summary)
        if limit and len(body) > limit:
            body = body[: max(0, limit - 1)].rstrip() + "…"
        label = self.kind
        head = f"[{label}理解]"
        if self.name:
            head = f"[{label}理解: {self.name}]"
        if self.error:
            return f"{head} 失败：{self.error}"
        return f"{head} {body}".strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "kind": self.kind,
            "umo": self.umo,
            "summary": self.summary,
            "detail": self.detail,
            "provider_id": self.provider_id,
            "model": self.model,
            "message_id": self.message_id,
            "name": self.name,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "error": self.error,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Understanding:
        return cls(
            key=str(value.get("key", "")),
            kind=str(value.get("kind", "")),
            umo=str(value.get("umo", "")),
            summary=str(value.get("summary", "")),
            detail=str(value.get("detail", "")),
            provider_id=str(value.get("provider_id", "")),
            model=str(value.get("model", "")),
            message_id=str(value.get("message_id", "")),
            name=str(value.get("name", "")),
            created_at=float(value.get("created_at", 0) or 0),
            expires_at=float(value.get("expires_at", 0) or 0),
            error=str(value.get("error", "")),
            meta=dict(value.get("meta") or {}),
        )


@dataclass(slots=True)
class MediaIndexEntry:
    """Lightweight index row so tools can find recent media by position."""

    umo: str
    message_id: str
    kind: str
    index: int
    name: str
    reference: str
    content_key: str
    received_at: float
    sender_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "umo": self.umo,
            "message_id": self.message_id,
            "kind": self.kind,
            "index": self.index,
            "name": self.name,
            "reference": self.reference,
            "content_key": self.content_key,
            "received_at": self.received_at,
            "sender_id": self.sender_id,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> MediaIndexEntry:
        return cls(
            umo=str(value.get("umo", "")),
            message_id=str(value.get("message_id", "")),
            kind=str(value.get("kind", "")),
            index=int(value.get("index", 0) or 0),
            name=str(value.get("name", "")),
            reference=str(value.get("reference", "")),
            content_key=str(value.get("content_key", "")),
            received_at=float(value.get("received_at", 0) or 0),
            sender_id=str(value.get("sender_id", "")),
        )
