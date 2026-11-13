"""0.4.15: ordered fallback decision Providers and the Jev -> 决策 renames."""

from __future__ import annotations

import json
import sqlite3
import sys
from types import ModuleType, SimpleNamespace

import pytest

from decision.models import DecisionProviderError
from decision.providers.astrbot import AstrBotDecisionProvider, normalize_provider_ids

QUESTIONS = {"q": {"type": "noul", "instructions": "Question?"}}


@pytest.fixture
def native_types(monkeypatch):
    module = ModuleType("astrbot.api.provider")
    module.DecisionBooleanQuestion = lambda instructions: ("noul", instructions)
    module.DecisionChoiceQuestion = lambda instructions, criteria: ("choice", instructions)
    module.DecisionScoreQuestion = lambda instructions, criteria: ("score", instructions)
    monkeypatch.setitem(sys.modules, "astrbot.api.provider", module)


def provider_with(answers_by_provider, **kwargs):
    """``answers_by_provider`` maps provider ID (None = default) to a probability or an error."""

    calls = []

    async def ask(questions, state, *, provider_id=None):
        calls.append(provider_id)
        outcome = answers_by_provider[provider_id]
        if isinstance(outcome, BaseException):
            raise outcome
        return {"q": SimpleNamespace(probability=outcome)}

    context = SimpleNamespace(decision=SimpleNamespace(ask=ask))
    return AstrBotDecisionProvider(context, **kwargs), calls


async def test_primary_success_never_touches_fallbacks(native_types):
    provider, calls = provider_with(
        {"main": 0.9, "backup": 0.1}, provider_id="main", fallback_provider_ids=["backup"]
    )
    result = await provider.evaluate(state="s", questions=QUESTIONS)
    assert result.answers["q"]["noul"] == 0.9
    assert calls == ["main"]
    assert (result.provider_id, result.attempts) == ("main", 1)


async def test_fallbacks_are_tried_in_order_after_primary_retries(native_types):
    switches = []
    provider, calls = provider_with(
        {
            None: RuntimeError("secret body"),
            "first": RuntimeError("down"),
            "second": 0.4,
            "third": 0.8,
        },
        fallback_provider_ids=["first", "second", "third"],
        retries=1,
        fallback_logger=lambda *args: switches.append(args),
    )
    result = await provider.evaluate(state="s", questions=QUESTIONS)
    # Primary (AstrBot default) and the first fallback each get 1 + retries attempts.
    assert calls == [None, None, "first", "first", "second"]
    assert result.answers["q"]["noul"] == 0.4
    assert provider.failures == 0
    assert (result.provider_id, result.attempts) == ("second", 5)
    assert switches == [
        ("AstrBot 默认决策模型", "first", "RuntimeError"),
        ("first", "second", "RuntimeError"),
    ]
    assert "secret" not in repr(switches)


async def test_all_providers_failing_raises_once_without_details(native_types):
    provider, calls = provider_with(
        {"main": RuntimeError("secret key"), "backup": RuntimeError("secret key")},
        provider_id="main",
        fallback_provider_ids=["backup"],
        retries=0,
    )
    with pytest.raises(DecisionProviderError) as raised:
        await provider.evaluate(state="s", questions=QUESTIONS)
    assert calls == ["main", "backup"]
    assert "secret" not in str(raised.value)
    assert provider.failures == 1
    assert raised.value.attempts == 2


async def test_a_fallback_equal_to_the_primary_is_skipped(native_types):
    provider, calls = provider_with(
        {"main": RuntimeError("down"), "backup": 0.6},
        provider_id="main",
        fallback_provider_ids=["main", "backup", "backup"],
        retries=0,
    )
    await provider.evaluate(state="s", questions=QUESTIONS)
    assert calls == ["main", "backup"]


async def test_missing_answers_from_primary_fall_back(native_types):
    calls = []

    async def ask(questions, state, *, provider_id=None):
        calls.append(provider_id)
        return {} if provider_id == "main" else {"q": SimpleNamespace(probability=0.7)}

    provider = AstrBotDecisionProvider(
        SimpleNamespace(decision=SimpleNamespace(ask=ask)),
        provider_id="main",
        fallback_provider_ids=["backup"],
        retries=0,
    )
    result = await provider.evaluate(state="s", questions=QUESTIONS)
    assert calls == ["main", "backup"] and result.answers["q"]["noul"] == 0.7


async def test_failed_providers_cool_down_while_another_one_answers(native_types, monkeypatch):
    import decision.providers.astrbot as adapter

    clock = [1000.0]
    monkeypatch.setattr(adapter.time, "monotonic", lambda: clock[0])
    outcomes = {"main": RuntimeError("down"), "backup": 0.5}
    provider, calls = provider_with(
        outcomes, provider_id="main", fallback_provider_ids=["backup"], retries=0
    )
    await provider.evaluate(state="s", questions=QUESTIONS)
    await provider.evaluate(state="s", questions=QUESTIONS)
    # The second request skips the failed primary instead of waiting on it.
    assert calls == ["main", "backup", "backup"]

    clock[0] += adapter.FALLBACK_COOLDOWN_SECONDS + 1
    outcomes["main"] = 0.9
    result = await provider.evaluate(state="s", questions=QUESTIONS)
    assert result.provider_id == "main" and calls[-1] == "main"


async def test_when_every_provider_is_cooling_only_the_first_is_probed(native_types, monkeypatch):
    import decision.providers.astrbot as adapter

    monkeypatch.setattr(adapter.time, "monotonic", lambda: 50.0)
    provider, calls = provider_with(
        {"main": RuntimeError("down"), "backup": RuntimeError("down")},
        provider_id="main",
        fallback_provider_ids=["backup"],
        retries=0,
    )
    for _ in range(2):
        with pytest.raises(DecisionProviderError):
            await provider.evaluate(state="s", questions=QUESTIONS)
    assert calls == ["main", "backup", "main"]


async def test_any_provider_error_falls_back(native_types):
    provider, calls = provider_with(
        {"main": ValueError("bad answer"), "backup": 0.5},
        provider_id="main",
        fallback_provider_ids=["backup"],
        retries=0,
    )
    result = await provider.evaluate(state="s", questions=QUESTIONS)
    assert calls == ["main", "backup"] and result.provider_id == "backup"


async def test_providers_missing_from_astrbot_are_skipped_without_requests(native_types):
    class Native:
        def __init__(self, provider_id):
            self.provider_id = provider_id

        def meta(self):
            return SimpleNamespace(id=self.provider_id, model="m", type="t")

    calls = []

    async def ask(questions, state, *, provider_id=None):
        calls.append(provider_id)
        if provider_id == "main":
            raise RuntimeError("down")
        return {"q": SimpleNamespace(probability=0.5)}

    context = SimpleNamespace(
        decision=SimpleNamespace(ask=ask),
        provider_manager=SimpleNamespace(decision_provider_insts=[Native("main"), Native("live")]),
    )
    provider = AstrBotDecisionProvider(
        context, provider_id="main", fallback_provider_ids=["deleted", "live"], retries=2
    )
    result = await provider.evaluate(state="s", questions=QUESTIONS)
    assert calls == ["main", "main", "main", "live"]
    assert (result.provider_id, result.attempts) == ("live", 4)


def test_reconfigure_clears_cooldown():
    provider = AstrBotDecisionProvider(SimpleNamespace(), provider_id="a", fallback_provider_ids=["b"])
    provider._cooldown_until["b"] = 1e18
    provider.reconfigure(" b ", ["a", "a", ""])
    assert (provider.provider_id, provider.fallback_provider_ids) == ("b", ["a"])
    assert provider._cooldown_until == {}


async def test_response_parse_errors_still_fall_back(native_types):
    provider, calls = provider_with(
        {"main": json.JSONDecodeError("bad", "x", 0), "backup": 0.5},
        provider_id="main",
        fallback_provider_ids=["backup"],
        retries=0,
    )
    result = await provider.evaluate(state="s", questions=QUESTIONS)
    assert calls == ["main", "backup"] and result.provider_id == "backup"


def test_provider_ids_are_cleaned_and_capped():
    assert normalize_provider_ids(" a , b") == []
    assert normalize_provider_ids(None) == []
    assert normalize_provider_ids([" a ", "", "a", 3, "b", "x" * 301]) == ["a", "b"]
    assert len(normalize_provider_ids([f"p{i}" for i in range(30)])) == 20


def test_legacy_life_threshold_keys_are_renamed():
    from decision.life_config import life_defaults, normalize_life

    raw = life_defaults()
    integration = raw["integration_settings"]
    for key in (
        "decision_threshold",
        "decision_relevant_threshold",
        "decision_timing_threshold",
        "decision_prompt",
    ):
        integration.pop(key)
    integration.update(
        jev_threshold=0.81, jev_relevant_threshold=0.72, jev_timing_threshold=0.33, jev_prompt="旧策略"
    )
    out = normalize_life(raw)["integration_settings"]
    assert (out["decision_threshold"], out["decision_relevant_threshold"]) == (0.81, 0.72)
    assert out["decision_timing_threshold"] == 0.33 and out["decision_prompt"] == "旧策略"
    assert not any(key.startswith("jev") for key in out)


def test_pre_048_single_legacy_threshold_still_covers_relevance():
    from decision.life_config import life_defaults, normalize_life

    raw = life_defaults()
    integration = raw["integration_settings"]
    integration.pop("decision_threshold")
    integration.pop("decision_relevant_threshold")
    integration["jev_threshold"] = 0.66
    out = normalize_life(raw)["integration_settings"]
    assert out["decision_threshold"] == 0.66 and out["decision_relevant_threshold"] == 0.66


def test_usage_rows_recorded_under_the_old_label_are_merged(tmp_path):
    from decision.usage_meter import UsageMeter

    path = tmp_path / "usage.json"
    row = {"calls": 2, "failures": 1, "input_tokens": 10, "output_tokens": 4, "estimated": 2}
    path.write_text(
        json.dumps({"2026-11-12": {"Jev·能力决策": row, "决策·能力决策": dict(row)}}),
        encoding="utf-8",
    )
    meter = UsageMeter(path)
    meter.save()
    saved = json.loads(path.read_text(encoding="utf-8"))
    if "2026-11-12" in saved:  # older than the 14-day window is pruned on save
        assert list(saved["2026-11-12"]) == ["决策·能力决策"]
    assert meter._days["2026-11-12"]["决策·能力决策"]["calls"] == 4


def test_old_audit_records_are_reworded(tmp_path):
    from decision.life_ledger import LifeLedger

    path = tmp_path / "contact_control.sqlite3"
    LifeLedger(path)
    with sqlite3.connect(path) as db:
        db.execute(
            "INSERT INTO audit(umo,at,sent,data) VALUES (?,?,?,?)",
            ("u", 1.0, 0, json.dumps({"reason": "Jev 未通过，时机 0.2/0.5", "stage": "Jev 判断"}, ensure_ascii=False)),
        )
        db.execute("PRAGMA user_version=0")
        db.execute(
            "INSERT INTO audit(umo,at,sent,data) VALUES (?,?,?,?)",
            (
                "qq:FriendMessage:Jevon",
                2.0,
                0,
                json.dumps(
                    {
                        "umo": "qq:FriendMessage:Jevon",
                        "reason": "Jev 判断失败：AstrBot JEV request failed: X",
                        "intent": "和 Jevon 聊天",
                    },
                    ensure_ascii=False,
                ),
            ),
        )
    ledger = LifeLedger(path)
    first, second = ledger.audits()
    assert first == {"reason": "决策未通过，时机 0.2/0.5", "stage": "决策判断"}
    assert second == {
        "umo": "qq:FriendMessage:Jevon",
        "reason": "决策判断失败：AstrBot decision request failed: X",
        "intent": "和 Jevon 聊天",
    }
    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 1


async def test_when_all_cool_down_the_earliest_recovery_is_probed(native_types, monkeypatch):
    import decision.providers.astrbot as adapter

    clock = [100.0]
    monkeypatch.setattr(adapter.time, "monotonic", lambda: clock[0])
    provider, calls = provider_with(
        {"main": RuntimeError("down"), "backup": 0.5},
        provider_id="main",
        fallback_provider_ids=["backup"],
        retries=0,
    )
    provider._cooldown_until.update(main=200.0, backup=150.0)
    result = await provider.evaluate(state="s", questions=QUESTIONS)
    assert calls == ["backup"] and result.provider_id == "backup"


async def test_errors_before_sending_count_zero_requests(native_types):
    provider = AstrBotDecisionProvider(SimpleNamespace(decision=None))
    with pytest.raises(DecisionProviderError) as raised:
        await provider.evaluate(state="s", questions=QUESTIONS)
    assert raised.value.attempts == 0

    provider, calls = provider_with({None: 0.5})
    with pytest.raises(ValueError) as invalid:
        await provider.evaluate(state="s", questions={"q": {"type": "noul", "instructions": ""}})
    assert invalid.value.attempts == 0 and calls == []


async def test_retry_log_names_the_provider(native_types):
    events = []
    provider, _ = provider_with(
        {"main": RuntimeError("secret"), "backup": 0.5},
        provider_id="main",
        fallback_provider_ids=["backup"],
        retries=1,
        retry_logger=lambda *args: events.append(args),
    )
    await provider.evaluate(state="s", questions=QUESTIONS)
    assert events == [(1, 1, "main：RuntimeError", 0.0)]


def test_usage_meter_ignores_zero_and_multiplies_counts(tmp_path):
    from decision.usage_meter import UsageMeter

    meter = UsageMeter(None)
    meter.record("决策·x", prompt="abc", output="", ok=False, count=0)
    assert meter._days == {}
    meter.record("决策·x", prompt="abc", output="", ok=False, count=3)
    (row,) = next(iter(meter._days.values())).values()
    assert (row["calls"], row["failures"]) == (3, 3)
    assert row["input_tokens"] % 3 == 0 and row["estimated"] == 3
