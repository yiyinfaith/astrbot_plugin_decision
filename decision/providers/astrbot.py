"""AstrBot-native decision-provider adapter.

The host owns the JEV transport, credentials, endpoint and model.  This
module only translates the plugin's stable, JSON-shaped question contract to
AstrBot's structured ``DecisionService`` contract and translates validated
answers back.  Keeping this boundary here makes every Decision Engine path
use the same host provider without leaking provider credentials into the
plugin.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from typing import Any

from ..models import DecisionProviderError, DecisionResult


def _native_types() -> dict[str, Any]:
    """Load native decision types lazily for old test doubles and old hosts."""

    try:
        from astrbot.api.provider import (
            DecisionBooleanQuestion,
            DecisionChoiceQuestion,
            DecisionScoreQuestion,
        )
    except (ImportError, AttributeError) as exc:  # pragma: no cover - old host only
        raise DecisionProviderError(
            "当前 AstrBot 未提供原生 JEV Decision Provider，请升级 AstrBot。"
        ) from exc
    return {
        "boolean": DecisionBooleanQuestion,
        "choice": DecisionChoiceQuestion,
        "score": DecisionScoreQuestion,
    }


def _native_question(question: Mapping[str, Any], types: Mapping[str, Any]) -> Any:
    question_type = str(question.get("type", "")).strip().lower()
    instructions = str(question.get("instructions", "")).strip()
    if not instructions:
        raise ValueError("question instructions must not be empty")
    if question_type == "noul":
        return types["boolean"](instructions)
    if question_type == "choice":
        criteria = question.get("criteria")
        if not isinstance(criteria, Mapping) or len(criteria) < 2:
            raise ValueError("choice criteria must contain at least two entries")
        return types["choice"](
            instructions,
            {str(key): str(value) for key, value in criteria.items()},
        )
    if question_type == "score":
        criteria = question.get("criteria")
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
            raise ValueError("score criteria must contain 2 to 10 entries")
        return types["score"](instructions, [str(value) for value in criteria])
    raise ValueError(f"unsupported decision question type: {question_type!r}")


def _answer_to_dict(question: Mapping[str, Any], answer: Any) -> dict[str, Any]:
    """Normalize an AstrBot answer dataclass without depending on its class."""

    question_type = str(question.get("type", "")).strip().lower()
    if question_type == "noul":
        probability = getattr(answer, "probability", None)
        return {"type": "noul", "noul": probability}
    if question_type == "choice":
        return {
            "type": "choice",
            "choice": getattr(answer, "choice", None),
            "probability": getattr(answer, "probability", None),
            "probabilities": dict(getattr(answer, "probabilities", {}) or {}),
            "confidence": getattr(answer, "confidence", None),
        }
    if question_type == "score":
        return {
            "type": "score",
            "score": getattr(answer, "score", None),
            "legend": dict(getattr(answer, "legend", {}) or {}),
            "probabilities": dict(getattr(answer, "probabilities", {}) or {}),
            "confidence": getattr(answer, "confidence", None),
        }
    raise DecisionProviderError(f"unsupported decision question type: {question_type!r}")


class AstrBotDecisionProvider:
    """Use the host's configured native JEV provider through ``Context.decision``."""

    def __init__(
        self,
        context: Any,
        *,
        provider_id: str = "",
        retries: int = 1,
        retry_backoff_sec: float = 0.0,
        retry_logger: Any | None = None,
    ) -> None:
        self.context = context
        self.provider_id = str(provider_id or "").strip()
        self.retries = max(0, int(retries))
        self.retry_backoff_sec = max(0.0, float(retry_backoff_sec))
        self.retry_logger = retry_logger
        self.calls = 0
        self.failures = 0
        self.last_latency_ms: float | None = None

    @property
    def configured(self) -> bool:
        return callable(getattr(self.context, "decision", None)) or getattr(
            self.context, "decision", None
        ) is not None

    async def start(self) -> None:
        """Retained as a lifecycle hook; AstrBot owns provider startup."""

    async def close(self) -> None:
        """Never close the host-owned provider or its client."""

    async def evaluate(
        self,
        *,
        state: str,
        questions: Mapping[str, Mapping[str, Any]],
        model: str | None = None,
    ) -> DecisionResult:
        del model  # The selected native Provider owns the model choice.
        if not questions:
            return DecisionResult(answers={})
        service = getattr(self.context, "decision", None)
        ask = getattr(service, "ask", None)
        if not callable(ask):
            self.failures += 1
            raise DecisionProviderError(
                "当前 AstrBot 未提供原生 JEV DecisionService，请升级 AstrBot。"
            )
        types = _native_types()
        native_questions = {
            str(key): _native_question(question, types)
            for key, question in questions.items()
        }
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            started = time.perf_counter()
            self.calls += 1
            try:
                # AstrBot's DecisionService selects the configured native JEV
                # provider.  An explicit ID is passed only when the operator
                # selected one in the plugin UI; no URL, key or model crosses
                # this boundary.
                native_answers = await ask(
                    native_questions,
                    state,
                    provider_id=self.provider_id or None,
                )
                if not isinstance(native_answers, Mapping):
                    raise DecisionProviderError("AstrBot JEV returned no answer map")
                answers = {
                    key: _answer_to_dict(questions[key], native_answers[key])
                    for key in questions
                    if key in native_answers
                }
                if len(answers) != len(questions):
                    missing = sorted(set(questions) - set(answers))
                    raise DecisionProviderError(
                        f"AstrBot JEV omitted decision answers: {', '.join(missing)}"
                    )
                elapsed = (time.perf_counter() - started) * 1000
                self.last_latency_ms = elapsed
                return DecisionResult(
                    answers=answers,
                    latency_ms=elapsed,
                    raw={"answers": answers},
                )
            except Exception as exc:  # retries are controlled by plugin settings
                last_error = exc
                if attempt >= self.retries:
                    self.failures += 1
                    if isinstance(exc, DecisionProviderError):
                        raise
                    raise DecisionProviderError(
                        f"AstrBot JEV request failed: {type(exc).__name__}"
                    ) from exc
                if self.retry_logger is not None:
                    self.retry_logger(
                        attempt + 1,
                        self.retries,
                        f"AstrBot JEV request failed ({type(exc).__name__})",
                        self.retry_backoff_sec,
                    )
                if self.retry_backoff_sec:
                    await asyncio.sleep(self.retry_backoff_sec)
        self.failures += 1
        raise DecisionProviderError("AstrBot JEV request failed") from last_error


def decision_provider_catalog(context: Any) -> list[dict[str, str]]:
    """Return only native decision providers for the WebUI picker."""

    manager = getattr(context, "provider_manager", None)
    candidates = getattr(manager, "decision_provider_insts", None)
    if candidates is None:
        candidates = getattr(manager, "decision_providers", None)
    if candidates is None:
        return []
    rows: list[dict[str, str]] = []
    for provider in candidates:
        try:
            meta = provider.meta()
            provider_id = str(getattr(meta, "id", "") or "").strip()
            if not provider_id:
                continue
            model = str(getattr(meta, "model", "") or "").strip()
            adapter = str(getattr(meta, "type", "") or "").strip()
            label = provider_id
            details = " · ".join(item for item in (model, adapter) if item)
            if details:
                label = f"{provider_id} · {details}"
            rows.append({"id": provider_id, "name": label, "model": model, "type": adapter})
        except (AttributeError, RuntimeError, TypeError, ValueError):
            continue
    return rows


__all__ = ["AstrBotDecisionProvider", "decision_provider_catalog"]
