from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from .models import DecisionResult


class DecisionProvider(ABC):
    """Provider-neutral interface for structured decisions."""

    @abstractmethod
    async def evaluate(
        self,
        *,
        state: str,
        questions: Mapping[str, Mapping[str, Any]],
        model: str | None = None,
    ) -> DecisionResult:
        raise NotImplementedError

    async def close(self) -> None:
        """Release provider resources. Providers without resources can ignore it."""
        return None
