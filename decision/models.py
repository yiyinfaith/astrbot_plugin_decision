from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


class DecisionProviderError(RuntimeError):
    """A transport or remote-provider failure without secret material."""


class DecisionValidationError(DecisionProviderError):
    """The provider returned a response that does not match the request."""


@dataclass(slots=True)
class DecisionResult:
    """Normalized SystemOne response plus non-sensitive call metadata."""

    answers: dict[str, dict[str, Any]]
    latency_ms: float | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


def _is_probability(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 1


def _validate_probability_map(
    value: Any,
    allowed: set[str],
    question_id: str,
) -> None:
    if not isinstance(value, Mapping):
        raise DecisionValidationError(f"SystemOne answer {question_id!r} has invalid probabilities")
    unknown = set(value) - allowed
    if unknown:
        raise DecisionValidationError(
            f"SystemOne answer {question_id!r} contains unknown probability keys"
        )
    if any(not _is_probability(item) for item in value.values()):
        raise DecisionValidationError(
            f"SystemOne answer {question_id!r} contains an invalid probability"
        )


def _validate_answer(question_id: str, question: Mapping[str, Any], answer: Any) -> dict[str, Any]:
    if not isinstance(answer, Mapping):
        raise DecisionValidationError(f"SystemOne answer {question_id!r} is not an object")

    question_type = question.get("type")
    answer_type = answer.get("type")
    if answer_type is not None and answer_type != question_type:
        raise DecisionValidationError(
            f"SystemOne answer {question_id!r} has type {answer_type!r}, expected {question_type!r}"
        )

    result = dict(answer)
    if question_type == "noul":
        value = result.get("noul")
        if not _is_probability(value):
            raise DecisionValidationError(
                f"SystemOne answer {question_id!r} has invalid noul probability"
            )
        return result

    if question_type == "choice":
        criteria = question.get("criteria")
        if not isinstance(criteria, Mapping) or len(criteria) < 2:
            raise DecisionValidationError(f"Choice question {question_id!r} has invalid criteria")
        choice = result.get("choice")
        allowed = {str(key) for key in criteria}
        if not isinstance(choice, str) or choice not in allowed:
            raise DecisionValidationError(
                f"SystemOne choice {choice!r} is not one of the requested options"
            )
        if "confidence" in result and not _is_probability(result["confidence"]):
            raise DecisionValidationError(
                f"SystemOne choice {question_id!r} has invalid confidence"
            )
        if "probabilities" in result:
            _validate_probability_map(result["probabilities"], allowed, question_id)
        return result

    if question_type == "score":
        criteria = question.get("criteria")
        if not isinstance(criteria, list) or len(criteria) < 2:
            raise DecisionValidationError(f"Score question {question_id!r} has invalid criteria")
        score = result.get("score")
        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or score < 0
            or score > len(criteria) - 1
        ):
            raise DecisionValidationError(f"SystemOne score {question_id!r} is out of range")
        if "confidence" in result and not _is_probability(result["confidence"]):
            raise DecisionValidationError(f"SystemOne score {question_id!r} has invalid confidence")
        if "probabilities" in result:
            allowed = {str(index) for index in range(len(criteria))}
            _validate_probability_map(result["probabilities"], allowed, question_id)
        return result

    raise DecisionValidationError(f"Unsupported SystemOne question type: {question_type!r}")


def parse_systemone_response(
    payload: Any,
    questions: Mapping[str, Mapping[str, Any]],
) -> DecisionResult:
    """Validate the useful part of a SystemOne response.

    The provider is allowed to add metadata such as ``usage`` and ``latency_ms``.
    Only answers for questions sent by this plugin are accepted; unknown IDs are
    ignored so a forward-compatible provider extension cannot affect routing.
    """

    if not isinstance(payload, Mapping):
        raise DecisionValidationError("SystemOne response is not a JSON object")
    raw_answers = payload.get("answers")
    if not isinstance(raw_answers, Mapping):
        raise DecisionValidationError("SystemOne response has no answers object")

    answers: dict[str, dict[str, Any]] = {}
    for question_id, question in questions.items():
        if question_id not in raw_answers:
            raise DecisionValidationError(f"SystemOne response omitted answer {question_id!r}")
        answers[question_id] = _validate_answer(
            question_id,
            question,
            raw_answers[question_id],
        )

    usage = payload.get("usage")
    if not isinstance(usage, Mapping):
        usage = {}
    latency = payload.get("latency_ms")
    if not isinstance(latency, (int, float)) or isinstance(latency, bool):
        latency = None
    return DecisionResult(
        answers=answers,
        latency_ms=float(latency) if latency is not None else None,
        usage=dict(usage),
        raw=dict(payload),
    )
