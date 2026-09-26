"""Small dependency-free token budget helpers for Jev request sizing."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from typing import Any

_ASCII_RUN = re.compile(r"[\x00-\x7f]+")
_ASCII_PART = re.compile(r"[A-Za-z0-9]+|[^A-Za-z0-9]")


def estimate_tokens(text: str) -> int:
    """Estimate model tokens conservatively without a provider tokenizer.

    Jev's tokenizer is not guaranteed to be installed with AstrBot. CJK and
    non-ASCII characters are counted individually; ASCII runs use a roughly
    four-characters-per-token estimate while punctuation/whitespace are kept
    conservative. This intentionally errs high so a request stays below the
    configured context limit.
    """

    value = str(text or "")
    total = 0
    ascii_run_end = 0
    for match in _ASCII_RUN.finditer(value):
        if match.start() > ascii_run_end:
            total += match.start() - ascii_run_end
        run = match.group(0)
        for part in _ASCII_PART.findall(run):
            total += math.ceil(len(part) / 4) if part[0].isalnum() else 1
        ascii_run_end = match.end()
    total += len(value) - ascii_run_end
    return max(1, total) if value else 0


def estimate_request_tokens(state: str, questions: Mapping[str, Mapping[str, Any]]) -> int:
    """Estimate both the state prompt and serialized decision questions."""

    encoded = json.dumps(
        {"state": state, "questions": questions},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return estimate_tokens(encoded)


def truncate_to_token_budget(text: str, budget: int) -> str:
    """Keep the tail of text within a token budget."""

    value = str(text or "")
    limit = max(1, int(budget))
    if estimate_tokens(value) <= limit:
        return value
    low, high = 0, len(value)
    while low < high:
        middle = (low + high + 1) // 2
        candidate = value[-middle:]
        if estimate_tokens(candidate) <= limit:
            low = middle
        else:
            high = middle - 1
    return value[-low:] if low else ""


def fit_request_state(
    state: str,
    questions: Mapping[str, Mapping[str, Any]],
    budget: int,
) -> str:
    """Keep a request's state tail while reserving room for question JSON."""

    value = str(state or "")
    limit = max(1, int(budget))
    if estimate_request_tokens(value, questions) < limit:
        return value
    low, high = 0, len(value)
    while low < high:
        middle = (low + high + 1) // 2
        candidate = value[-middle:]
        if estimate_request_tokens(candidate, questions) < limit:
            low = middle
        else:
            high = middle - 1
    return value[-low:] if low else ""
