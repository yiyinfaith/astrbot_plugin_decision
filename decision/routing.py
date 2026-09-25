from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .context import is_handoff_tool

DECISION_TOOL_NAME = "decision_evaluate"
DEFAULT_MAIN_LLM_POST_PROMPT = """<decision_routing_hint>
当前场景需要根据用户请求选择合适的能力。
推荐你优先考虑使用以下 Tools：{tools}
如果确有必要委派，推荐你考虑以下 SubAgents：{subagents}

以上是 Decision Engine 根据当前场景给出的推荐；你仍需自行判断是否调用、调用哪些参数，以及是否委派。
推荐列表可以为空，也可以同时包含多个候选。
</decision_routing_hint>"""
ROUTING_HINT_TEMPLATE = DEFAULT_MAIN_LLM_POST_PROMPT
ROUTING_HINT_MARKER = "<astrbot_plugin_decision_routing>"
ROUTING_HINT_END_MARKER = "</astrbot_plugin_decision_routing>"


@dataclass(slots=True)
class RoutingOutcome:
    selected: list[Any]
    handoffs: list[Any]
    decision_tool: Any | None
    recommended_tools: list[str] = field(default_factory=list)


def choose_tools(
    original_tools: Sequence[Any],
    noul_answers: Mapping[str, Mapping[str, Any]],
    question_to_tool: Mapping[str, str],
    *,
    threshold: float,
    always_keep: set[str],
    decision_tool: Any | None,
    always_keep_recommend: set[str] | None = None,
) -> RoutingOutcome:
    """Route tools while keeping SubAgents available for advisory recommendations.

    Ordinary tools are filtered by their per-tool Noul answer. Handoff tools
    are always retained and are never filtered here. ``always_keep_recommend``
    is intentionally separate from ``always_keep``: a manually preserved tool
    may be retained without being recommended to the main LLM.
    """

    final: list[Any] = []
    handoffs: list[Any] = []
    recommended_tools: list[str] = []
    always_keep_recommend = always_keep_recommend or set()
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
        if name in always_keep and name in always_keep_recommend:
            recommended_tools.append(name)
        elif keep and name not in always_keep:
            matching_ids = [qid for qid, tool_name in question_to_tool.items() if tool_name == name]
            if any(
                isinstance(noul_answers.get(qid, {}).get("noul"), (int, float))
                and not isinstance(noul_answers.get(qid, {}).get("noul"), bool)
                and noul_answers[qid]["noul"] >= threshold
                for qid in matching_ids
            ):
                recommended_tools.append(name)

    if decision_tool is not None and all(
        getattr(tool, "name", None) != DECISION_TOOL_NAME for tool in final
    ):
        final.append(decision_tool)
    return RoutingOutcome(
        selected=final,
        handoffs=handoffs,
        decision_tool=decision_tool,
        recommended_tools=recommended_tools,
    )


def recommendations_from_noul(
    noul_answers: Mapping[str, Mapping[str, Any]],
    question_to_name: Mapping[str, str],
    *,
    threshold: float,
) -> list[str]:
    """Return every candidate whose individual Noul answer passes ``threshold``.

    The function deliberately does not choose a single winner. It can return
    zero, one, or many names while preserving the order of the question map.
    """

    recommendations: list[str] = []
    seen: set[str] = set()
    for question_id, name in question_to_name.items():
        if not name or name in seen:
            continue
        answer = noul_answers.get(question_id, {})
        value = answer.get("noul") if isinstance(answer, Mapping) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= threshold:
            recommendations.append(name)
            seen.add(name)
    return recommendations


def render_routing_prompt(
    template: str,
    *,
    recommended_tools: Sequence[str] = (),
    recommended_subagents: Sequence[str] = (),
) -> str:
    """Render the configurable post-decision prompt without interpreting other braces."""

    tools_text = ", ".join(dict.fromkeys(str(name) for name in recommended_tools if name))
    subagents_text = ", ".join(dict.fromkeys(str(name) for name in recommended_subagents if name))
    # Deliberately use targeted replacement rather than str.format: users may
    # put their own JSON/schema braces in the editable prompt.
    return (
        str(template)
        .replace("{tools}", tools_text or "(无)")
        .replace("{subagents}", subagents_text or "(无)")
    )


def append_system_prompt(req: Any, text: str) -> None:
    """Append one plugin-owned system section while preserving all existing text."""

    text = str(text or "").strip()
    if not text:
        return
    current = str(getattr(req, "system_prompt", "") or "")
    # A request should normally pass this hook once, but this guard keeps a
    # retry or nested invocation from duplicating the same routing section.
    if ROUTING_HINT_MARKER in current:
        return
    section = f"{ROUTING_HINT_MARKER}\n{text}\n{ROUTING_HINT_END_MARKER}"
    req.system_prompt = f"{current}\n\n{section}" if current else section


def add_routing_hint(
    req: Any,
    recommendation: str | None = None,
    *,
    template: str = DEFAULT_MAIN_LLM_POST_PROMPT,
    recommended_tools: Sequence[str] = (),
    recommended_subagents: Sequence[str] = (),
) -> None:
    """Append an advisory hint to the main LLM system prompt.

    ``recommendation`` remains accepted for compatibility with older callers;
    it is treated as a single SubAgent recommendation.
    """

    if recommendation and not recommended_subagents:
        recommended_subagents = (recommendation,)
    text = render_routing_prompt(
        template,
        recommended_tools=recommended_tools,
        recommended_subagents=recommended_subagents,
    )
    append_system_prompt(req, text)


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
