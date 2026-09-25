from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping
from typing import Any

import aiohttp

from ..models import DecisionProviderError, DecisionResult, parse_systemone_response
from ..provider import DecisionProvider


class _QuestionLimitError(DecisionProviderError):
    """The remote endpoint rejected a request because it was too large."""


class _MixedQuestionTypeError(DecisionProviderError):
    """The endpoint cannot currently combine Noul with Choice/Score."""


class SystemOneProvider(DecisionProvider):
    """Small async HTTP client for the TypeSafe-compatible SystemOne endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        path: str,
        api_key: str,
        model: str,
        timeout_sec: float = 10,
        retries: int = 1,
        chunk_size: int = 32,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.path = "/" + path.strip("/")
        self.api_key = api_key.strip()
        self.model = model.strip() or "jev-latest"
        self.timeout_sec = max(1.0, float(timeout_sec))
        self.retries = max(0, int(retries))
        self.chunk_size = max(2, int(chunk_size))
        self._session = session
        self._owns_session = session is None
        self.last_latency_ms: float | None = None
        self.calls = 0
        self.failures = 0

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout_sec),
                headers={"Content-Type": "application/json"},
            )
            self._owns_session = True
        return self._session

    def _url(self) -> str:
        return f"{self.base_url}{self.path}"

    async def start(self) -> None:
        """Create the reusable connection pool during plugin initialization."""

        await self._get_session()

    async def evaluate(
        self,
        *,
        state: str,
        questions: Mapping[str, Mapping[str, Any]],
        model: str | None = None,
    ) -> DecisionResult:
        if not questions:
            return DecisionResult(answers={})
        if not self.api_key:
            self.failures += 1
            raise DecisionProviderError("SystemOne API key is not configured")
        if not self.base_url:
            self.failures += 1
            raise DecisionProviderError("SystemOne base URL is not configured")

        question_map = {str(key): dict(value) for key, value in questions.items()}
        try:
            result = await self._evaluate_once(
                state=state,
                questions=question_map,
                model=model or self.model,
            )
        except _QuestionLimitError:
            if len(question_map) <= self.chunk_size:
                self.failures += 1
                raise
            # A normal successful request stays a single fan-out request. Chunking
            # is only a fallback for an endpoint that explicitly rejects size.
            try:
                result = await self._evaluate_in_chunks(
                    state=state,
                    questions=question_map,
                    model=model or self.model,
                )
            except Exception:
                self.failures += 1
                raise
        except _MixedQuestionTypeError:
            question_types = {str(question.get("type", "")) for question in question_map.values()}
            if len(question_types) <= 1:
                self.failures += 1
                raise
            # The current SystemOne service rejects a mixed Noul plus
            # Choice/Score payload with a 400 usage error. Keep each primitive
            # type in one request so a large Noul fan-out remains one request.
            try:
                result = await self._evaluate_by_type(
                    state=state,
                    questions=question_map,
                    model=model or self.model,
                )
            except Exception:
                self.failures += 1
                raise
        except Exception:
            self.failures += 1
            raise
        return result

    async def _evaluate_by_type(
        self,
        *,
        state: str,
        questions: dict[str, dict[str, Any]],
        model: str,
    ) -> DecisionResult:
        groups: dict[str, dict[str, dict[str, Any]]] = {}
        for question_id, question in questions.items():
            question_type = str(question.get("type", "unknown"))
            groups.setdefault(question_type, {})[question_id] = question
        # Keep this fallback sequential. The affected service-side usage bug
        # can also appear when the homogeneous recovery requests overlap.
        results = []
        for group in groups.values():
            results.append(await self._evaluate_once(state=state, questions=group, model=model))
        merged: dict[str, dict[str, Any]] = {}
        usage: dict[str, Any] = {}
        latencies: list[float] = []
        for result in results:
            merged.update(result.answers)
            usage.update(result.usage)
            if result.latency_ms is not None:
                latencies.append(result.latency_ms)
        return DecisionResult(
            answers=merged,
            latency_ms=sum(latencies) if latencies else None,
            usage=usage,
            raw={"answers": merged},
        )

    async def _evaluate_in_chunks(
        self,
        *,
        state: str,
        questions: dict[str, dict[str, Any]],
        model: str,
    ) -> DecisionResult:
        items = list(questions.items())
        chunks = [
            dict(items[i : i + self.chunk_size]) for i in range(0, len(items), self.chunk_size)
        ]
        results = await asyncio.gather(
            *(self._evaluate_once(state=state, questions=chunk, model=model) for chunk in chunks)
        )
        merged: dict[str, dict[str, Any]] = {}
        usage: dict[str, Any] = {}
        latencies: list[float] = []
        raw: dict[str, Any] = {"answers": merged}
        for result in results:
            merged.update(result.answers)
            usage.update(result.usage)
            if result.latency_ms is not None:
                latencies.append(result.latency_ms)
        raw["answers"] = merged
        return DecisionResult(
            answers=merged,
            latency_ms=sum(latencies) if latencies else None,
            usage=usage,
            raw=raw,
        )

    async def _evaluate_once(
        self,
        *,
        state: str,
        questions: dict[str, dict[str, Any]],
        model: str,
    ) -> DecisionResult:
        session = await self._get_session()
        body = {"state": state, "model": model, "questions": questions}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            started = time.perf_counter()
            self.calls += 1
            try:
                async with session.post(
                    self._url(),
                    json=body,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout_sec),
                ) as response:
                    text = await response.text()
                    elapsed = (time.perf_counter() - started) * 1000
                    self.last_latency_ms = elapsed
                    if response.status in {413, 422} or (
                        response.status == 400
                        and "question" in text.lower()
                        and ("limit" in text.lower() or "too many" in text.lower())
                    ):
                        raise _QuestionLimitError(
                            "SystemOne question payload was rejected as too large"
                        )
                    if (
                        response.status == 400
                        and "plugin usage value must be a number" in text.lower()
                    ):
                        raise _MixedQuestionTypeError(
                            "SystemOne rejected mixed decision question types"
                        )
                    if response.status in {401, 403}:
                        raise DecisionProviderError("SystemOne authorization failed")
                    if response.status == 429:
                        raise DecisionProviderError("SystemOne rate limit reached")
                    if response.status >= 500:
                        raise DecisionProviderError(f"SystemOne server error ({response.status})")
                    if response.status < 200 or response.status >= 300:
                        raise DecisionProviderError(f"SystemOne request failed ({response.status})")
                    try:
                        payload = json.loads(text)
                    except (TypeError, json.JSONDecodeError) as exc:
                        raise DecisionProviderError("SystemOne returned non-JSON data") from exc
                    result = parse_systemone_response(payload, questions)
                    if result.latency_ms is None:
                        result.latency_ms = elapsed
                    return result
            except _QuestionLimitError:
                raise
            except (TimeoutError, aiohttp.ClientError, DecisionProviderError) as exc:
                last_error = exc
                retryable = isinstance(exc, (asyncio.TimeoutError, aiohttp.ClientError)) or (
                    isinstance(exc, DecisionProviderError)
                    and any(token in str(exc) for token in ("rate limit", "server error"))
                )
                if not retryable or attempt >= self.retries:
                    raise
                await asyncio.sleep(min(2.0, 0.25 * (2**attempt)))
        raise DecisionProviderError("SystemOne request failed") from last_error

    async def close(self) -> None:
        if self._owns_session and self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None
