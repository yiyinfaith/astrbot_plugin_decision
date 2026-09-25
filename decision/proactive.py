from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass(slots=True)
class ProactiveRecord:
    sender: str
    sender_id: str
    text: str
    is_bot: bool = False


class ProactiveState:
    def __init__(self, max_messages: int = 12) -> None:
        self.max_messages = max(2, max_messages)
        self.history: dict[str, deque[ProactiveRecord]] = defaultdict(
            lambda: deque(maxlen=self.max_messages)
        )
        self.reply_times: dict[str, deque[float]] = defaultdict(deque)
        self.last_reply: dict[str, float] = {}

    def add(self, session: str, record: ProactiveRecord) -> None:
        self.history[session].append(record)

    def history_lines(self, session: str, max_chars: int) -> list[str]:
        lines = []
        for item in self.history.get(session, ()):
            role = "bot" if item.is_bot else item.sender
            line = f"[{role}]: {item.text}"
            lines.append(line)
        joined = "\n".join(lines)
        return joined[-max_chars:].splitlines() if joined else []

    def allow_reply(
        self,
        session: str,
        *,
        cooldown_seconds: float,
        window_seconds: float,
        max_replies: int,
    ) -> bool:
        now = time.monotonic()
        if now - self.last_reply.get(session, 0.0) < max(0.0, cooldown_seconds):
            return False
        times = self.reply_times[session]
        while times and now - times[0] > max(1.0, window_seconds):
            times.popleft()
        if len(times) >= max(1, max_replies):
            return False
        times.append(now)
        self.last_reply[session] = now
        return True
