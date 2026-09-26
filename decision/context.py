from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from typing import Any


def _clean_text(value: Any, limit: int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    elif isinstance(value, Mapping):
        text = str(value.get("text") or value.get("content") or "")
    elif isinstance(value, list):
        pieces = []
        for item in value:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, Mapping) and item.get("type") in {"text", "input_text"}:
                pieces.append(str(item.get("text") or item.get("content") or ""))
        text = " ".join(pieces)
    else:
        text = str(value)
    text = " ".join(text.split())
    return text if limit is None else text[: max(0, int(limit))]


def context_lines(
    contexts: Iterable[Mapping[str, Any]] | None,
    *,
    max_messages: int,
    max_chars: int,
    current_prompt: str = "",
) -> list[str]:
    """Extract a small text-only context without system prompts or tool schemas."""

    if not contexts or max_messages <= 0 or max_chars <= 0:
        return []
    current = _clean_text(current_prompt, max_chars)
    result: list[str] = []
    for item in list(contexts)[-max_messages:]:
        if not isinstance(item, Mapping):
            continue
        role = str(item.get("role") or "unknown")
        text = _clean_text(item.get("content"), max_chars)
        if not text or (current and text == current and role == "user"):
            continue
        if role == "tool":
            text = _clean_text(text, min(500, max_chars))
        result.append(f"{role}: {text}")
    joined = "\n".join(result)
    if len(joined) > max_chars:
        joined = joined[-max_chars:]
        result = joined.splitlines()
    return result


def short_description(description: Any, max_chars: int | None = None) -> str:
    return (
        _clean_text(description, None if max_chars is None else max(1, max_chars))
        or "(no description)"
    )


def build_decision_state(
    *,
    policy: str,
    current_prompt: str,
    contexts: Iterable[Mapping[str, Any]] | None,
    tools: Iterable[Mapping[str, str]] = (),
    subagents: Iterable[Mapping[str, str]] = (),
    history_max_messages: int = 8,
    history_max_chars: int = 6000,
) -> str:
    sections = [f"[Decision Policy]\n{_clean_text(policy, 4000) or '(default policy)'}"]
    history = context_lines(
        contexts,
        max_messages=history_max_messages,
        max_chars=history_max_chars,
        current_prompt=current_prompt,
    )
    sections.append("[Conversation Context]\n" + ("\n".join(history) if history else "(none)"))
    sections.append(f"[Current User Request]\n{_clean_text(current_prompt, history_max_chars)}")

    tool_lines = [
        f"- {item.get('name', '')}: {item.get('description', '')}"
        for item in tools
        if item.get("name")
    ]
    sections.append("[Available Tools]\n" + ("\n".join(tool_lines) if tool_lines else "(none)"))
    agent_lines = [
        f"- {item.get('name', '')}: {item.get('description', '')}"
        for item in subagents
        if item.get("name")
    ]
    sections.append(
        "[Available SubAgents]\n" + ("\n".join(agent_lines) if agent_lines else "(none)")
    )
    return "\n\n".join(sections)


def tool_summary(tool: Any, max_chars: int | None = None) -> dict[str, str]:
    return {
        "name": str(getattr(tool, "name", "")),
        "description": short_description(getattr(tool, "description", ""), max_chars),
    }


def question_id(prefix: str, name: str, index: int) -> str:
    digest = hashlib.sha1(name.encode("utf-8", "ignore")).hexdigest()[:10]
    safe = re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_")[:32] or "tool"
    return f"{prefix}_{index}_{safe}_{digest}"


def is_handoff_tool(tool: Any) -> bool:
    try:
        from astrbot.core.agent.handoff import HandoffTool

        return isinstance(tool, HandoffTool)
    except Exception:
        return tool.__class__.__name__ == "HandoffTool"


def tool_is_active(tool: Any) -> bool:
    return bool(getattr(tool, "active", True))
