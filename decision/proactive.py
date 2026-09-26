"""State and guardrails for Decision Engine proactive conversation.

This module owns only the decision side of proactive chat. The actual response
is still produced by AstrBot's normal Agent/conversation path; it never
rebuilds or replaces AstrBot's native context.
"""

from __future__ import annotations

import asyncio
import time
from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum


class ProactiveStatus(str, Enum):  # noqa: UP042 - AstrBot supports Python 3.10
    """Four interaction states adapted from AngelHeart."""

    NOT_PRESENT = "not_present"
    SUMMONED = "summoned"
    GETTING_FAMILIAR = "getting_familiar"
    OBSERVATION = "observation"


@dataclass(slots=True)
class ProactiveRecord:
    sender: str
    sender_id: str
    text: str
    is_bot: bool = False
    timestamp: float = field(default_factory=time.monotonic)


@dataclass(slots=True)
class ProactiveSession:
    history: deque[ProactiveRecord]
    reply_times: deque[float] = field(default_factory=deque)
    status: ProactiveStatus = ProactiveStatus.NOT_PRESENT
    status_since: float = field(default_factory=time.monotonic)
    last_activity: float = field(default_factory=time.monotonic)
    last_reply: float = 0.0
    next_analysis: float = 0.0
    consecutive_failures: int = 0
    pending_reply: bool = False


@dataclass(frozen=True, slots=True)
class ProactiveDecision:
    """Normalized result of one Jev proactive judgment."""

    should_reply: bool
    scores: dict[str, float]
    aggregate: float
    forced: bool = False


def normalize_prefixes(value: object, default: Iterable[str] = ("/", "@")) -> list[str]:
    """Return a clean, bounded prefix list suitable for runtime matching."""

    values = value if isinstance(value, (list, tuple, set)) else default
    result: list[str] = []
    for item in values:
        text = str(item).strip()
        if text and text not in result:
            result.append(text)
    return result[:32]


def text_matches_prefix(text: str, prefixes: Iterable[str]) -> bool:
    """Match ordinary prefixes; at-sign is reserved for real At nodes."""

    content = str(text or "").strip()
    return any(prefix != "@" and content.startswith(prefix) for prefix in prefixes)


def bounded_float(value: object, default: float, low: float = 0.0, high: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return min(high, max(low, number))


def parse_noul_scores(
    answers: Mapping[str, Mapping[str, object]], names: Iterable[str]
) -> dict[str, float]:
    """Extract valid numeric Noul answers, ignoring malformed output."""

    scores: dict[str, float] = {}
    for name in names:
        answer = answers.get(name, {})
        value = answer.get("noul") if isinstance(answer, Mapping) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            scores[name] = min(1.0, max(0.0, float(value)))
    return scores


def aggregate_scores(
    scores: Mapping[str, float], weights: Mapping[str, float] | None = None
) -> float:
    """Calculate a normalized weighted mean, returning zero for no valid data."""

    if not scores:
        return 0.0
    weights = weights or {}
    total = 0.0
    weight_sum = 0.0
    for name, value in scores.items():
        weight = max(0.0, float(weights.get(name, 1.0)))
        if weight <= 0:
            continue
        total += min(1.0, max(0.0, float(value))) * weight
        weight_sum += weight
    return total / weight_sum if weight_sum else 0.0


class ProactiveState:
    """Bounded per-session history, state machine, cooldowns and async locks."""

    def __init__(self, max_messages: int = 12, max_sessions: int = 500) -> None:
        self.max_messages = max(2, int(max_messages))
        self.max_sessions = max(1, int(max_sessions))
        # history remains public for compatibility with the previous class.
        self.history: dict[str, deque[ProactiveRecord]] = defaultdict(
            lambda: deque(maxlen=self.max_messages)
        )
        self.sessions: dict[str, ProactiveSession] = {}
        self.reply_times: dict[str, deque[float]] = defaultdict(deque)
        self.last_reply: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._access: dict[str, float] = {}

    def _session(self, session: str) -> ProactiveSession:
        key = str(session)
        current = self.sessions.get(key)
        if current is None:
            current = ProactiveSession(deque(maxlen=self.max_messages))
            self.history[key] = current.history
            self.reply_times[key] = current.reply_times
            self.sessions[key] = current
        self._access[key] = time.monotonic()
        self._prune_sessions()
        return current

    def _prune_sessions(self) -> None:
        if len(self.sessions) <= self.max_sessions:
            return
        removable = sorted(self._access, key=self._access.get)  # type: ignore[arg-type]
        for key in removable:
            if len(self.sessions) <= self.max_sessions:
                break
            session = self.sessions.get(key)
            lock = self._locks.get(key)
            if session and (session.pending_reply or (lock is not None and lock.locked())):
                continue
            self.sessions.pop(key, None)
            self.history.pop(key, None)
            self.reply_times.pop(key, None)
            self.last_reply.pop(key, None)
            self._access.pop(key, None)
            self._locks.pop(key, None)

    def lock_for(self, session: str) -> asyncio.Lock:
        key = str(session)
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        self._session(key)
        return lock

    def add(self, session: str, record: ProactiveRecord) -> None:
        current = self._session(session)
        current.history.append(record)
        current.last_activity = time.monotonic()

    def history_lines(self, session: str, max_chars: int) -> list[str]:
        current = self._session(session)
        lines = []
        for item in current.history:
            role = "bot" if item.is_bot else item.sender
            lines.append(f"[{role}]: {item.text}")
        joined = "\n".join(lines)
        return joined[-max(1, int(max_chars)) :].splitlines() if joined else []

    def status(self, session: str, observation_timeout: float = 600.0) -> ProactiveStatus:
        current = self._session(session)
        now = time.monotonic()
        if current.status == ProactiveStatus.OBSERVATION and now - current.last_activity > max(
            1.0, observation_timeout
        ):
            current.status = ProactiveStatus.NOT_PRESENT
            current.status_since = now
        return current.status

    def observe_message(
        self,
        session: str,
        *,
        text: str,
        sender_id: str,
        summoned: bool = False,
        echo_threshold: int = 3,
        echo_window: float = 30.0,
        dense_threshold: int = 30,
        dense_window: float = 600.0,
        min_participants: int = 2,
        observation_timeout: float = 600.0,
    ) -> ProactiveStatus:
        """Advance the state using local metadata from the current chat."""

        current = self._session(session)
        now = time.monotonic()
        self.status(session, observation_timeout)
        current.last_activity = now
        if summoned:
            return self.set_status(session, ProactiveStatus.SUMMONED)

        records = [
            item
            for item in current.history
            if not item.is_bot and now - item.timestamp <= max(1.0, dense_window)
        ]
        if current.status == ProactiveStatus.NOT_PRESENT:
            recent_echo = [
                item.text.strip()
                for item in records
                if now - item.timestamp <= max(1.0, echo_window) and item.text.strip()
            ]
            repeated = bool(
                recent_echo and max(Counter(recent_echo).values()) >= max(2, int(echo_threshold))
            )
            participants = {item.sender_id for item in records if item.sender_id}
            dense = len(records) >= max(1, int(dense_threshold)) and len(participants) >= max(
                1, int(min_participants)
            )
            if repeated or dense:
                return self.set_status(session, ProactiveStatus.GETTING_FAMILIAR)
        if current.status in {ProactiveStatus.SUMMONED, ProactiveStatus.GETTING_FAMILIAR}:
            return self.set_status(session, ProactiveStatus.OBSERVATION)
        return current.status

    def set_status(self, session: str, status: ProactiveStatus) -> ProactiveStatus:
        current = self._session(session)
        if current.status != status:
            current.status = status
            current.status_since = time.monotonic()
        return status

    def can_analyze(self, session: str, *, now: float | None = None) -> bool:
        current = self._session(session)
        return (time.monotonic() if now is None else now) >= current.next_analysis

    def mark_analysis(self, session: str, *, success: bool, no_reply_cooldown: float) -> None:
        current = self._session(session)
        now = time.monotonic()
        if success:
            current.consecutive_failures = 0
            multiplier = 1.0
        else:
            current.consecutive_failures += 1
            multiplier = min(8.0, 2 ** (current.consecutive_failures - 1))
        current.next_analysis = now + max(0.0, float(no_reply_cooldown)) * multiplier

    def allow_reply(
        self,
        session: str,
        *,
        cooldown_seconds: float,
        window_seconds: float,
        max_replies: int,
    ) -> bool:
        """Reserve one reply slot, retaining the previous public API."""

        current = self._session(session)
        now = time.monotonic()
        if now - current.last_reply < max(0.0, cooldown_seconds):
            return False
        while current.reply_times and now - current.reply_times[0] > max(1.0, window_seconds):
            current.reply_times.popleft()
        if len(current.reply_times) >= max(1, max_replies):
            return False
        current.reply_times.append(now)
        current.last_reply = now
        current.pending_reply = True
        self.last_reply[str(session)] = now
        return True

    def mark_reply_success(self, session: str) -> None:
        current = self._session(session)
        current.pending_reply = False
        current.consecutive_failures = 0
        self.set_status(session, ProactiveStatus.SUMMONED)

    def mark_reply_finished(self, session: str) -> None:
        self._session(session).pending_reply = False
