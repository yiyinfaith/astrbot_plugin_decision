"""Per-feature, per-day model call counter for the status page.

Only counts and token totals are stored: no prompt, reply or identity. A
provider response that carries ``usage`` gives real token numbers; otherwise
tokens are estimated from text length and the day is marked as estimated.
"""

from __future__ import annotations

import json
import math
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

KEEP_DAYS = 14


def estimate_tokens(text: Any) -> int:
    """Rough token count: CJK characters count ~0.7, other text ~0.3 per char."""

    value = str(text or "")
    if not value:
        return 0
    cjk = sum(1 for ch in value if "　" <= ch <= "鿿" or "＀" <= ch <= "￯")
    return math.ceil(cjk * 0.7 + (len(value) - cjk) * 0.3)


def response_usage(response: Any) -> tuple[int, int] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    try:
        prompt = int(getattr(usage, "input", None) or 0)
        output = int(getattr(usage, "output", 0) or 0)
    except (TypeError, ValueError):
        return None
    if prompt <= 0 and output <= 0:
        return None
    return prompt, output


class UsageMeter:
    def __init__(self, path: Path | None = None):
        self.path = path
        self._lock = threading.Lock()
        self._days: dict[str, dict[str, dict[str, Any]]] = {}
        self._dirty = False
        self._last_save = 0.0
        if path is not None and path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._days = {
                        str(day): {str(k): dict(v) for k, v in rows.items() if isinstance(v, dict)}
                        for day, rows in data.items()
                        if isinstance(rows, dict)
                    }
            except (OSError, ValueError):
                self._days = {}

    def record(
        self,
        feature: str,
        *,
        response: Any = None,
        prompt: Any = "",
        output: Any = None,
        extra_input: int = 0,
        ok: bool = True,
    ) -> None:
        real = response_usage(response) if response is not None else None
        if real:
            input_tokens, output_tokens, estimated = real[0], real[1], False
        else:
            if output is None and response is not None:
                output = getattr(response, "completion_text", "")
            input_tokens, output_tokens, estimated = (
                estimate_tokens(prompt) + max(0, int(extra_input)),
                estimate_tokens(output),
                True,
            )
        day = datetime.now().date().isoformat()
        with self._lock:
            row = self._days.setdefault(day, {}).setdefault(
                feature,
                {"calls": 0, "failures": 0, "input_tokens": 0, "output_tokens": 0, "estimated": 0},
            )
            row["calls"] += 1
            row["failures"] += 0 if ok else 1
            row["input_tokens"] += input_tokens
            row["output_tokens"] += output_tokens
            row["estimated"] += 1 if estimated else 0
            self._dirty = True
        if time.monotonic() - self._last_save > 60:
            self.save()

    def summary(self, days: int = 7) -> dict[str, Any]:
        today = datetime.now().date()
        wanted = [(today - timedelta(days=offset)).isoformat() for offset in range(days)]
        with self._lock:
            result = {day: json.loads(json.dumps(self._days.get(day, {}))) for day in wanted}
        return {"today": wanted[0], "days": result}

    def save(self) -> None:
        if self.path is None:
            return
        with self._lock:
            if not self._dirty:
                return
            cutoff = (datetime.now().date() - timedelta(days=KEEP_DAYS)).isoformat()
            self._days = {day: rows for day, rows in self._days.items() if day >= cutoff}
            payload = json.dumps(self._days, ensure_ascii=False)
            self._dirty = False
            self._last_save = time.monotonic()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(payload, encoding="utf-8")
            tmp.replace(self.path)
        except OSError:
            with self._lock:
                self._dirty = True


# The plugin installs its own meter at startup; tests and standalone helpers
# record into this in-memory one.
METER = UsageMeter()


def install(meter: UsageMeter) -> None:
    global METER
    METER = meter


def record(feature: str, **kwargs: Any) -> None:
    try:
        METER.record(feature, **kwargs)
    except Exception:
        # Statistics must never break a model call.
        pass
