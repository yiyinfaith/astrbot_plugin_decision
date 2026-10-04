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

GROUP_TARGET_ID = "group"
GROUP_TARGET_NAME = "群聊"
BOT_TARGET_ID = "bot"


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
    # These fields are deliberately adapter-neutral.  The main plugin fills
    # them from AstrBot's At/Reply components, while tests and other adapters
    # can construct records without importing AstrBot message classes.
    reply_to_id: str = ""
    at_targets: tuple[tuple[str, str], ...] = ()
    talking_to: str = GROUP_TARGET_ID
    talking_to_name: str = GROUP_TARGET_NAME


@dataclass(frozen=True, slots=True)
class DialogueInference:
    """A conservative, explainable guess about the current addressee."""

    target_id: str = GROUP_TARGET_ID
    target_name: str = GROUP_TARGET_NAME
    confidence: float = 0.0
    reason: str = "default_group"


DEFAULT_REPLY_STARTERS = (
    "好",
    "嗯",
    "哦",
    "对",
    "是的",
    "谢谢",
    "收到",
    "知道",
    "明白",
    "哈哈",
    "笑死",
)


def _unique_targets(
    targets: Iterable[tuple[str, str]],
) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw_id, raw_name in targets:
        target_id = str(raw_id or "").strip()
        # AstrBot represents @all as an ``AtAll`` component (a subclass of
        # ``At``) and some adapters expose it as the literal target ID
        # ``all``.  It addresses the group, never a concrete participant.
        if not target_id or target_id.casefold() == "all" or target_id in seen:
            continue
        seen.add(target_id)
        target_name = str(raw_name or target_id).strip() or target_id
        result.append((target_id, target_name))
    return result


def _reply_like(text: str, starters: Iterable[str] = DEFAULT_REPLY_STARTERS) -> bool:
    value = str(text or "").strip()
    return bool(value and len(value) <= 20 and any(value.startswith(item) for item in starters))


def infer_dialogue_target(
    current: ProactiveRecord,
    history: Iterable[ProactiveRecord],
    *,
    bot_id: str = "",
    reply_starters: Iterable[str] = DEFAULT_REPLY_STARTERS,
    now: float | None = None,
) -> DialogueInference:
    """Infer who a group message is addressing without an extra Jev call.

    The rules intentionally mirror only the useful, conservative part of
    ContextAware: explicit address signals win; contextual guesses never
    promote a message to the Bot unless the preceding Bot response was aimed
    at this sender.  Sender IDs are the identity anchor, so duplicate or
    changed nicknames do not merge people.
    """

    bot_id = str(bot_id or "").strip()
    bot_target_ids = {bot_id} if bot_id else {BOT_TARGET_ID}
    targets = _unique_targets(current.at_targets)
    bot_targets = [name for target_id, name in targets if target_id in bot_target_ids]
    other_targets = [(target_id, name) for target_id, name in targets if target_id not in bot_target_ids]
    if bot_targets:
        label = "你" if not other_targets else "你和" + "、".join(name for _, name in other_targets)
        return DialogueInference(BOT_TARGET_ID, label, 1.0, "explicit_at_bot")
    if other_targets:
        target_id, target_name = other_targets[0]
        if len(other_targets) > 1:
            target_name = "、".join(name for _, name in other_targets)
        return DialogueInference(target_id, target_name, 1.0, "explicit_at_other")

    history_list = list(history)
    if current.reply_to_id:
        reply_id = str(current.reply_to_id).strip()
        if reply_id in bot_target_ids:
            return DialogueInference(BOT_TARGET_ID, "你", 1.0, "reply_to_bot")
        reply_name = reply_id
        for item in reversed(history_list):
            if str(item.sender_id) == reply_id:
                reply_name = item.sender
                break
        return DialogueInference(reply_id, reply_name, 1.0, "reply_to_other")

    if not history_list:
        return DialogueInference(reason="default_group")
    current_time = time.monotonic() if now is None else float(now)
    recent = [item for item in history_list[-5:] if item.sender_id != current.sender_id]
    if not recent:
        return DialogueInference(reason="default_group")
    last = recent[-1]
    gap = max(0.0, current_time - float(last.timestamp))

    # A short acknowledgement after a Bot response can be a reply to the Bot.
    # If another person was speaking to this sender immediately before the Bot
    # interjected, preserve that human-to-human thread instead.
    if last.is_bot and gap < 20 and last.talking_to == current.sender_id:
        if _reply_like(current.text, reply_starters):
            for item in reversed(history_list[:-1]):
                if current_time - float(item.timestamp) > 90:
                    break
                if item.is_bot or item.sender_id == current.sender_id:
                    continue
                if item.talking_to == current.sender_id and last.timestamp - item.timestamp < 60:
                    return DialogueInference(item.sender_id, item.sender, 0.82, "bot_interrupted")
            return DialogueInference(BOT_TARGET_ID, "你", 0.82, "bot_recently_replied")
        return DialogueInference(reason="after_bot_nonreply")

    # A-B-A: the previous speaker explicitly addressed the current sender.
    if last.talking_to == current.sender_id and gap < 60 and not last.is_bot:
        return DialogueInference(last.sender_id, last.sender, 0.68, "aba_pattern")

    # A quick follow-up to a group-directed message is intentionally left as
    # group-directed.  Timing alone is too weak to claim that the sender is
    # speaking to a particular person.
    return DialogueInference(reason="default_group")


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

    def reconfigure(self, max_messages: int, max_sessions: int) -> None:
        """Apply WebUI history/session limits to the live state immediately."""

        self.max_messages = max(2, int(max_messages))
        self.max_sessions = max(1, int(max_sessions))
        for key, session in list(self.sessions.items()):
            if session.history.maxlen != self.max_messages:
                session.history = deque(session.history, maxlen=self.max_messages)
                self.history[key] = session.history
        self._prune_sessions()

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

    def history_records(self, session: str) -> list[ProactiveRecord]:
        """Return a stable snapshot for local dialogue-flow inference."""

        return list(self._session(session).history)

    def history_lines(
        self,
        session: str,
        max_chars: int,
        *,
        include_dialogue: bool = False,
        max_messages: int | None = None,
    ) -> list[str]:
        current = self._session(session)
        lines = []
        records = list(current.history)
        if max_messages is not None and int(max_messages) > 0:
            records = records[-int(max_messages) :]
        for item in records:
            role = "bot" if item.is_bot else item.sender
            if include_dialogue:
                identity = f"{role} ({item.sender_id})" if item.sender_id else role
                target = item.talking_to_name or item.talking_to or GROUP_TARGET_NAME
                target_identity = (
                    f"{target} ({item.talking_to})"
                    if item.talking_to not in {GROUP_TARGET_ID, BOT_TARGET_ID}
                    and item.talking_to
                    else target
                )
                lines.append(f"[{identity} → {target_identity}]: {item.text}")
            else:
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
        # A zero limit is a useful per-window kill switch.  Treat negative
        # values the same way instead of silently allowing one reply through
        # because of a defensive ``max(1, ...)`` clamp.
        if int(max_replies) <= 0:
            return False
        if now - current.last_reply < max(0.0, cooldown_seconds):
            return False
        while current.reply_times and now - current.reply_times[0] > max(1.0, window_seconds):
            current.reply_times.popleft()
        if len(current.reply_times) >= int(max_replies):
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

    def has_pending_reply(self, session: str) -> bool:
        current = self.sessions.get(str(session))
        return bool(current and current.pending_reply)

    def mark_reply_finished(self, session: str) -> None:
        self._session(session).pending_reply = False

    def cancel_reply(self, session: str) -> None:
        """Release a reply slot when the request was never queued."""

        key = str(session)
        current = self._session(key)
        if not current.pending_reply:
            return
        current.pending_reply = False
        # ``allow_reply`` appends the reservation as the newest timestamp. It
        # is safe to remove only that newest entry; completed replies have
        # already cleared ``pending_reply`` and remain counted for the window.
        if current.reply_times and current.reply_times[-1] == current.last_reply:
            current.reply_times.pop()
        current.last_reply = current.reply_times[-1] if current.reply_times else 0.0
        if current.last_reply:
            self.last_reply[key] = current.last_reply
        else:
            self.last_reply.pop(key, None)
