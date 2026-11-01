from __future__ import annotations

import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

import pytest

from decision.providers.astrbot import AstrBotDecisionProvider, decision_provider_catalog


@dataclass(frozen=True)
class BooleanQuestion:
    instructions: str


@dataclass(frozen=True)
class ChoiceQuestion:
    instructions: str
    criteria: dict[str, str]


@dataclass(frozen=True)
class ScoreQuestion:
    instructions: str
    criteria: list[str]


def install_native_types(monkeypatch):
    provider_module = ModuleType("astrbot.api.provider")
    provider_module.DecisionBooleanQuestion = BooleanQuestion
    provider_module.DecisionChoiceQuestion = ChoiceQuestion
    provider_module.DecisionScoreQuestion = ScoreQuestion
    monkeypatch.setitem(sys.modules, "astrbot.api.provider", provider_module)


@pytest.mark.asyncio
async def test_native_adapter_routes_all_decision_types_through_astrbot(monkeypatch):
    install_native_types(monkeypatch)
    calls = []

    async def ask(questions, state, *, provider_id=None):
        calls.append((questions, state, provider_id))
        return {
            "yes": SimpleNamespace(probability=0.91),
            "choice": SimpleNamespace(
                choice="keep",
                probability=0.8,
                probabilities={"keep": 0.8, "drop": 0.2},
                confidence=0.9,
            ),
            "score": SimpleNamespace(
                score=1.5,
                legend={"0": "low", "1": "mid", "2": "high"},
                probabilities={"0": 0.1, "1": 0.3, "2": 0.6},
                confidence=0.7,
            ),
        }

    provider = AstrBotDecisionProvider(
        SimpleNamespace(decision=SimpleNamespace(ask=ask)), provider_id="jev-7"
    )
    result = await provider.evaluate(
        state="conversation context",
        questions={
            "yes": {"type": "noul", "instructions": "Is this useful?"},
            "choice": {
                "type": "choice",
                "instructions": "Choose one",
                "criteria": {"keep": "Keep", "drop": "Drop"},
            },
            "score": {
                "type": "score",
                "instructions": "Rate it",
                "criteria": ["low", "mid", "high"],
            },
        },
        model="ignored-model-owned-by-astrbot",
    )

    assert len(calls) == 1
    native_questions, state, selected_provider = calls[0]
    assert type(native_questions["yes"]) is BooleanQuestion
    assert type(native_questions["choice"]) is ChoiceQuestion
    assert type(native_questions["score"]) is ScoreQuestion
    assert state == "conversation context"
    assert selected_provider == "jev-7"
    assert result.answers["yes"] == {"type": "noul", "noul": 0.91}
    assert result.answers["choice"]["choice"] == "keep"
    assert result.answers["score"]["score"] == 1.5


@pytest.mark.asyncio
async def test_native_adapter_retries_without_exposing_provider_error_details(monkeypatch):
    install_native_types(monkeypatch)
    calls = 0
    retry_events = []

    async def ask(questions, state, *, provider_id=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("secret key response body")
        return {"q": SimpleNamespace(probability=0.7)}

    provider = AstrBotDecisionProvider(
        SimpleNamespace(decision=SimpleNamespace(ask=ask)),
        retries=1,
        retry_logger=lambda *args: retry_events.append(args),
    )
    result = await provider.evaluate(
        state="state", questions={"q": {"type": "noul", "instructions": "Question?"}}
    )

    assert calls == 2
    assert result.answers["q"]["noul"] == 0.7
    assert retry_events and "secret key" not in repr(retry_events)
    assert provider.calls == 2
    assert provider.failures == 0


@pytest.mark.asyncio
async def test_native_adapter_fails_closed_on_missing_provider_service(monkeypatch):
    install_native_types(monkeypatch)
    provider = AstrBotDecisionProvider(SimpleNamespace(decision=None), retries=0)
    with pytest.raises(RuntimeError, match="DecisionService"):
        await provider.evaluate(
            state="state", questions={"q": {"type": "noul", "instructions": "Question?"}}
        )
    assert provider.failures == 1


def test_provider_catalog_lists_only_native_decision_providers():
    class NativeProvider:
        def __init__(self, provider_id, model, provider_type):
            self._meta = SimpleNamespace(id=provider_id, model=model, type=provider_type)

        def meta(self):
            return self._meta

    context = SimpleNamespace(
        provider_manager=SimpleNamespace(
            decision_provider_insts=[
                NativeProvider("decision-1", "jev-fast", "system_one_decision")
            ],
            provider_insts=[NativeProvider("chat-1", "chat-model", "openai")],
        )
    )
    assert decision_provider_catalog(context) == [
        {
            "id": "decision-1",
            "name": "decision-1 · jev-fast · system_one_decision",
            "model": "jev-fast",
            "type": "system_one_decision",
        }
    ]
