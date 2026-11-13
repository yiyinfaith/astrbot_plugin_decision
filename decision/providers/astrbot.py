"""AstrBot-native decision-provider adapter.

The host owns the decision transport, credentials, endpoint and model.  This
module only translates the plugin's stable, JSON-shaped question contract to
AstrBot's structured ``DecisionService`` contract and translates validated
answers back.  Keeping this boundary here makes every Decision Engine path
use the same host provider without leaking provider credentials into the
plugin.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping, Sequence
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
            "当前 AstrBot 未提供原生决策 Provider，请升级 AstrBot。"
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
    """Use the host's configured native decision provider through ``Context.decision``."""

    def __init__(
        self,
        context: Any,
        *,
        provider_id: str = "",
        fallback_provider_ids: Sequence[str] = (),
        retries: int = 1,
        retry_backoff_sec: float = 0.0,
        retry_logger: Any | None = None,
        fallback_logger: Any | None = None,
    ) -> None:
        self.context = context
        self.provider_id = str(provider_id or "").strip()
        self.fallback_provider_ids = normalize_provider_ids(fallback_provider_ids)
        self.retries = max(0, int(retries))
        self.retry_backoff_sec = max(0.0, float(retry_backoff_sec))
        self.retry_logger = retry_logger
        self.fallback_logger = fallback_logger
        self.calls = 0
        self.failures = 0
        self.last_latency_ms: float | None = None
        # Provider ID -> monotonic time until which a failed provider is
        # skipped while another one in the chain is available.
        self._cooldown_until: dict[str, float] = {}
        self._known_ids: tuple[float, set[str]] | None = None

    @property
    def configured(self) -> bool:
        # evaluate() calls DecisionService.ask(); without it, stay fail-open.
        return callable(getattr(getattr(self.context, "decision", None), "ask", None))

    def provider_chain(self) -> list[str]:
        """Primary provider ("" = AstrBot default) followed by distinct fallbacks."""

        chain = [self.provider_id]
        chain.extend(
            provider_id
            for provider_id in self.fallback_provider_ids
            if provider_id != self.provider_id
        )
        return chain

    def reconfigure(self, provider_id: str, fallback_provider_ids: Sequence[str]) -> None:
        """Apply a new primary/fallback selection and forget old failures."""

        self.provider_id = str(provider_id or "").strip()
        self.fallback_provider_ids = normalize_provider_ids(fallback_provider_ids)
        self._cooldown_until.clear()
        self._known_ids = None

    def _ready_chain(self) -> list[str]:
        """The chain to try for one request.

        Providers that AstrBot no longer has are skipped (asking them sends
        nothing and only fails).  Providers that failed within the cooldown
        are skipped too, so an outage does not make every request walk the
        whole chain again.  If nothing is left, only the first provider is
        tried so the request still fails open quickly.
        """

        chain = self.provider_chain()
        if len(chain) == 1:
            return chain
        known = self.known_provider_ids()
        existing = [item for item in chain if not item or not known or item in known]
        candidates = existing or chain[:1]
        now = time.monotonic()
        ready = [item for item in candidates if self._cooldown_until.get(item, 0.0) <= now]
        if ready:
            return ready
        # Everything is cooling down: probe only the one whose cooldown ends first.
        return [min(candidates, key=lambda item: self._cooldown_until.get(item, 0.0))]

    def known_provider_ids(self) -> set[str]:
        """Decision Provider IDs AstrBot currently has (cached for 30 seconds)."""

        now = time.monotonic()
        if self._known_ids is None or now - self._known_ids[0] > 30:
            self._known_ids = (
                now,
                {row["id"] for row in decision_provider_catalog(self.context)},
            )
        return self._known_ids[1]

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
            error = DecisionProviderError(
                "当前 AstrBot 未提供原生决策服务 DecisionService，请升级 AstrBot。"
            )
            error.attempts = 0
            raise error
        try:
            types = _native_types()
            native_questions = {
                str(key): _native_question(question, types)
                for key, question in questions.items()
            }
        except Exception as exc:
            exc.attempts = 0  # nothing was sent
            raise
        chain = self._ready_chain()
        sent = [0]  # requests sent by this evaluation (shared counters race)
        last_error: Exception | None = None
        for position, provider_id in enumerate(chain):
            if position and self.fallback_logger is not None:
                # Name only the original exception type; never its message.
                cause = getattr(last_error, "__cause__", None) or last_error
                self.fallback_logger(
                    chain[position - 1] or "AstrBot 默认决策模型",
                    provider_id,
                    type(cause).__name__,
                )
            try:
                result = await self._ask_with_retries(
                    ask, native_questions, questions, state, provider_id, sent
                )
            except Exception as exc:  # the next provider in the chain takes over
                last_error = exc
                if len(self.provider_chain()) > 1:
                    self._cooldown_until[provider_id] = (
                        time.monotonic() + FALLBACK_COOLDOWN_SECONDS
                    )
                continue
            self._cooldown_until.pop(provider_id, None)
            result.provider_id = provider_id
            result.attempts = sent[0]
            return result
        self.failures += 1
        error = (
            last_error
            if isinstance(last_error, DecisionProviderError)
            else DecisionProviderError(
                f"AstrBot decision request failed: {type(last_error).__name__}"
            )
        )
        error.attempts = sent[0]
        if error is last_error:
            raise error
        raise error from last_error

    async def _ask_with_retries(
        self,
        ask: Any,
        native_questions: Mapping[str, Any],
        questions: Mapping[str, Mapping[str, Any]],
        state: str,
        provider_id: str,
        sent: list[int],
    ) -> DecisionResult:
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            started = time.perf_counter()
            self.calls += 1
            sent[0] += 1
            try:
                # AstrBot's DecisionService selects the configured native decision
                # provider.  An explicit ID is passed only when the operator
                # selected one; no URL, key or model crosses this boundary.
                native_answers = await ask(
                    native_questions,
                    state,
                    provider_id=provider_id or None,
                )
                if not isinstance(native_answers, Mapping):
                    raise DecisionProviderError("AstrBot decision returned no answer map")
                answers = {
                    key: _answer_to_dict(questions[key], native_answers[key])
                    for key in questions
                    if key in native_answers
                }
                if len(answers) != len(questions):
                    missing = sorted(set(questions) - set(answers))
                    raise DecisionProviderError(
                        f"AstrBot decision omitted decision answers: {', '.join(missing)}"
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
                    break
                if self.retry_logger is not None:
                    self.retry_logger(
                        attempt + 1,
                        self.retries,
                        f"{provider_id or 'AstrBot 默认决策模型'}：{type(exc).__name__}",
                        self.retry_backoff_sec,
                    )
                if self.retry_backoff_sec:
                    await asyncio.sleep(self.retry_backoff_sec)
        if isinstance(last_error, DecisionProviderError):
            raise last_error
        raise DecisionProviderError(
            f"AstrBot decision request failed: {type(last_error).__name__}"
        ) from last_error


MAX_FALLBACK_PROVIDERS = 20
# How long a failed provider is skipped while others in the chain are available.
FALLBACK_COOLDOWN_SECONDS = 60.0


def normalize_provider_ids(value: Any) -> list[str]:
    """Ordered, de-duplicated, non-empty provider IDs (at most 20)."""

    if isinstance(value, str) or not isinstance(value, Sequence):
        return []
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        provider_id = item.strip()
        if provider_id and len(provider_id) <= 300 and provider_id not in result:
            result.append(provider_id)
    return result[:MAX_FALLBACK_PROVIDERS]


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


__all__ = [
    "AstrBotDecisionProvider",
    "MAX_FALLBACK_PROVIDERS",
    "decision_provider_catalog",
    "normalize_provider_ids",
]
