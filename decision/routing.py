from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .context import is_handoff_tool

DECISION_TOOL_NAME = "decision_evaluate"


@dataclass(slots=True)
class RoutingOutcome:
    selected: list[Any]
    handoffs: list[Any]
    decision_tool: Any | None
    recommendation: str | None = None


def choose_tools(
    original_tools: Sequence[Any],
    noul_answers: Mapping[str, Mapping[str, Any]],
    question_to_tool: Mapping[str, str],
    *,
    threshold: float,
    always_keep: set[str],
    decision_tool: Any | None,
) -> RoutingOutcome:
    """Filter only ordinary tools while preserving original order and permission."""

    final: list[Any] = []
    handoffs: list[Any] = []
    seen: set[str] = set()
    for tool in original_tools:
        name = str(getattr(tool, "name", ""))
        if not name or name in seen:
            continue
        seen.add(name)
        if name == DECISION_TOOL_NAME or tool is decision_tool:
            continue
        if is_handoff_tool(tool):
            handoffs.append(tool)
            final.append(tool)
            continue
        keep = name in always_keep
        if not keep:
            matching_ids = [qid for qid, tool_name in question_to_tool.items() if tool_name == name]
            keep = any(
                isinstance(noul_answers.get(qid, {}).get("noul"), (int, float))
                and not isinstance(noul_answers.get(qid, {}).get("noul"), bool)
                and noul_answers[qid]["noul"] >= threshold
                for qid in matching_ids
            )
        if keep:
            final.append(tool)

    if decision_tool is not None and all(
        getattr(tool, "name", None) != DECISION_TOOL_NAME for tool in final
    ):
        final.append(decision_tool)
    return RoutingOutcome(selected=final, handoffs=handoffs, decision_tool=decision_tool)


def recommendation_from_answer(
    answer: Mapping[str, Any] | None,
    allowed_names: set[str],
) -> str | None:
    if not answer:
        return None
    choice = answer.get("choice")
    if not isinstance(choice, str) or choice in {"none", "", "null"}:
        return None
    return choice if choice in allowed_names else None


def add_routing_hint(req: Any, recommendation: str) -> None:
    text = (
        "<decision_routing_hint>\n"
        "Decision Engine recommends considering subagent: "
        f"{recommendation}\n\n"
        "This is advisory only. You may choose another subagent or not delegate. "
        "If you delegate, formulate the actual task yourself and call the appropriate handoff tool.\n"
        "</decision_routing_hint>"
    )
    try:
        from astrbot.core.agent.message import TextPart

        req.extra_user_content_parts.append(TextPart(text=text).mark_as_temp())
    except Exception:
        # This fallback is useful for isolated unit tests and older compatible
        # runtimes; AstrBot 4.28.1 takes the TextPart path above.
        parts = getattr(req, "extra_user_content_parts", None)
        if parts is None:
            req.extra_user_content_parts = []
            parts = req.extra_user_content_parts
        parts.append({"type": "text", "text": text, "_no_save": True})


def decision_tool_parameters() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "state": {
                "type": "string",
                "description": "The smallest relevant state for the decision.",
            },
            "decision_type": {
                "type": "string",
                "enum": ["noul", "choice", "score"],
                "description": "The structured decision primitive to use.",
            },
            "instructions": {
                "type": "string",
                "description": "The yes/no question, choice question, or score rubric question.",
            },
            "choice_options": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["key", "description"],
                },
                "description": "Choice keys and what each key means.",
            },
            "score_levels": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Ordered score levels from lowest to highest.",
            },
        },
        "required": ["state", "decision_type", "instructions"],
    }


def compact_answer(answer: Mapping[str, Any]) -> str:
    return json.dumps(dict(answer), ensure_ascii=False, separators=(",", ":"))
