"""Track paid submissions across one analysis, including child asyncio tasks."""

from contextvars import ContextVar
from dataclasses import dataclass


@dataclass
class Submissions:
    count: int = 0


current: ContextVar[Submissions | None] = ContextVar("decision_mm_submissions", default=None)


def mark_submitted() -> None:
    state = current.get()
    if state is not None:
        state.count += 1
