from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class DecisionProviderError(RuntimeError):
    """A transport or remote-provider failure without secret material."""


class DecisionValidationError(DecisionProviderError):
    """The provider returned a response that does not match the request."""


@dataclass(slots=True)
class DecisionResult:
    """Normalized decision answers plus non-sensitive call metadata."""

    answers: dict[str, dict[str, Any]]
    latency_ms: float | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)
    # Which provider answered ("" = AstrBot default) and how many requests
    # were sent for this result, retries and fallbacks included.
    provider_id: str | None = None
    attempts: int = 1
