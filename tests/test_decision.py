from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from decision.context import build_decision_state, context_lines
from decision.context import question_id as make_question_id
from decision.models import DecisionProviderError, DecisionValidationError, parse_systemone_response
from decision.proactive import ProactiveRecord, ProactiveState
from decision.providers.systemone import SystemOneProvider
from decision.routing import (
    add_routing_hint,
    append_system_prompt,
    choose_tools,
    decision_tool_parameters,
    recommendations_from_noul,
    render_routing_prompt,
)


class FakeResponse:
    def __init__(self, status: int = 200, payload=None, text: str | None = None):
        self.status = status
        self._payload = payload
        self._text = text if text is not None else json.dumps(payload)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def text(self):
        return self._text


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.closed = False

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    async def close(self):
        self.closed = True


def answer_payload(questions):
    answers = {}
    for question_id, question in questions.items():
        if question["type"] == "noul":
            answers[question_id] = {"type": "noul", "noul": 0.8}
        elif question["type"] == "choice":
            key = next(iter(question["criteria"]))
            answers[question_id] = {
                "type": "choice",
                "choice": key,
                "confidence": 0.9,
                "probabilities": {name: 0.5 for name in question["criteria"]},
            }
        else:
            answers[question_id] = {
                "type": "score",
                "score": 1.0,
                "confidence": 0.8,
                "probabilities": {"0": 0.1, "1": 0.8, "2": 0.1},
            }
    return {"answers": answers, "usage": {"input_tokens": 10}}


def test_parse_noul_choice_score():
    questions = {
        "n": {"type": "noul", "instructions": "yes?"},
        "c": {"type": "choice", "instructions": "pick", "criteria": {"a": "A", "b": "B"}},
        "s": {"type": "score", "instructions": "rate", "criteria": ["low", "mid", "high"]},
    }
    payload = answer_payload(questions)
    result = parse_systemone_response(payload, questions)
    assert result.answers["n"]["noul"] == 0.8
    assert result.answers["c"]["choice"] == "a"
    assert result.answers["s"]["score"] == 1.0


def test_parse_rejects_invalid_probability_and_choice():
    with pytest.raises(DecisionValidationError):
        parse_systemone_response(
            {"answers": {"n": {"type": "noul", "noul": 1.4}}},
            {"n": {"type": "noul", "instructions": "x"}},
        )


def test_parse_rejects_missing_answers_and_invalid_score():
    with pytest.raises(DecisionValidationError, match="omitted answer"):
        parse_systemone_response(
            {"answers": {}},
            {"n": {"type": "noul", "instructions": "x"}},
        )
    with pytest.raises(DecisionValidationError, match="out of range"):
        parse_systemone_response(
            {"answers": {"s": {"type": "score", "score": 3}}},
            {"s": {"type": "score", "instructions": "x", "criteria": ["low", "high"]}},
        )


def test_parse_rejects_unknown_probability_key():
    with pytest.raises(DecisionValidationError, match="unknown probability"):
        parse_systemone_response(
            {
                "answers": {
                    "c": {
                        "type": "choice",
                        "choice": "a",
                        "probabilities": {"a": 0.8, "b": 0.1, "invented": 0.1},
                    }
                }
            },
            {"c": {"type": "choice", "instructions": "x", "criteria": {"a": "A", "b": "B"}}},
        )


def test_parse_ignores_unknown_answer_ids_but_validates_requested_ids():
    result = parse_systemone_response(
        {
            "answers": {
                "n": {"type": "noul", "noul": 0.4},
                "future_extension": {"type": "noul", "noul": 0.9},
            }
        },
        {"n": {"type": "noul", "instructions": "x"}},
    )
    assert list(result.answers) == ["n"]


def test_score_probability_keys_are_indexed():
    result = parse_systemone_response(
        {
            "answers": {
                "s": {
                    "type": "score",
                    "score": 1,
                    "probabilities": {"0": 0.2, "1": 0.7, "2": 0.1},
                }
            }
        },
        {"s": {"type": "score", "instructions": "x", "criteria": ["low", "mid", "high"]}},
    )
    assert result.answers["s"]["score"] == 1
    with pytest.raises(DecisionValidationError):
        parse_systemone_response(
            {"answers": {"c": {"type": "choice", "choice": "x"}}},
            {"c": {"type": "choice", "instructions": "x", "criteria": {"a": "A", "b": "B"}}},
        )


@pytest.mark.asyncio
async def test_provider_uses_one_request_for_202_nouls():
    questions = {f"q_{index}": {"type": "noul", "instructions": "keep?"} for index in range(202)}
    session = FakeSession([FakeResponse(payload=answer_payload(questions))])
    provider = SystemOneProvider(
        base_url="https://example.invalid",
        path="/v1/systemone",
        api_key="test-key",
        model="jev-latest",
        retries=0,
        session=session,
    )
    result = await provider.evaluate(state="state", questions=questions)
    assert len(result.answers) == 202
    assert len(session.calls) == 1
    body = session.calls[0][1]["json"]
    assert len(body["questions"]) == 202


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response, message",
    [
        (FakeResponse(401, {"error": "no"}), "authorization"),
        (FakeResponse(429, {"error": "rate"}), "rate limit"),
        (FakeResponse(500, {"error": "bad"}), "server error"),
        (FakeResponse(200, text="not json"), "non-JSON"),
    ],
)
async def test_provider_errors_are_sanitized(response, message):
    provider = SystemOneProvider(
        base_url="https://example.invalid",
        path="/v1/systemone",
        api_key="secret-value",
        model="jev-latest",
        retries=0,
        session=FakeSession([response]),
    )
    with pytest.raises(DecisionProviderError, match=message):
        await provider.evaluate(
            state="state",
            questions={"q": {"type": "noul", "instructions": "x"}},
        )


@pytest.mark.asyncio
async def test_provider_timeout():
    provider = SystemOneProvider(
        base_url="https://example.invalid",
        path="/v1/systemone",
        api_key="secret-value",
        model="jev-latest",
        retries=0,
        session=FakeSession([TimeoutError()]),
    )
    with pytest.raises(asyncio.TimeoutError):
        await provider.evaluate(
            state="state",
            questions={"q": {"type": "noul", "instructions": "x"}},
        )


@pytest.mark.asyncio
async def test_provider_requires_key_and_base_url_without_network():
    for kwargs, message in [
        ({"api_key": "", "base_url": "https://example.invalid"}, "API key"),
        ({"api_key": "key", "base_url": ""}, "base URL"),
    ]:
        provider = SystemOneProvider(
            base_url=kwargs["base_url"],
            path="/v1/systemone",
            api_key=kwargs["api_key"],
            model="jev-latest",
            retries=0,
            session=FakeSession([]),
        )
        with pytest.raises(DecisionProviderError, match=message):
            await provider.evaluate(
                state="state",
                questions={"q": {"type": "noul", "instructions": "x"}},
            )


@pytest.mark.asyncio
async def test_provider_retries_server_error_then_succeeds():
    questions = {"q": {"type": "noul", "instructions": "x"}}
    session = FakeSession(
        [
            FakeResponse(500, {"error": "temporary"}),
            FakeResponse(payload=answer_payload(questions)),
        ]
    )
    provider = SystemOneProvider(
        base_url="https://example.invalid",
        path="/v1/systemone",
        api_key="key",
        model="jev-latest",
        retries=1,
        session=session,
    )
    result = await provider.evaluate(state="state", questions=questions)
    assert result.answers["q"]["noul"] == 0.8
    assert len(session.calls) == 2


@pytest.mark.asyncio
async def test_provider_chunks_only_after_explicit_question_limit():
    questions = {f"q_{index}": {"type": "noul", "instructions": "x"} for index in range(5)}
    chunks = [
        {
            f"q_{index}": {"type": "noul", "instructions": "x"}
            for index in range(start, min(start + 2, 5))
        }
        for start in range(0, 5, 2)
    ]
    responses = [FakeResponse(413, {"error": "too many questions"})]
    responses.extend(FakeResponse(payload=answer_payload(chunk)) for chunk in chunks)
    session = FakeSession(responses)
    provider = SystemOneProvider(
        base_url="https://example.invalid",
        path="/v1/systemone",
        api_key="key",
        model="jev-latest",
        retries=0,
        chunk_size=2,
        session=session,
    )
    result = await provider.evaluate(state="state", questions=questions)
    assert len(result.answers) == 5
    assert len(session.calls) == 4


@pytest.mark.asyncio
async def test_provider_splits_known_mixed_type_service_error():
    questions = {
        "n": {"type": "noul", "instructions": "x"},
        "c": {"type": "choice", "instructions": "x", "criteria": {"a": "A", "b": "B"}},
    }
    session = FakeSession(
        [
            FakeResponse(400, {"error": "plugin usage value must be a number"}),
            FakeResponse(payload=answer_payload({"n": questions["n"]})),
            FakeResponse(payload=answer_payload({"c": questions["c"]})),
        ]
    )
    provider = SystemOneProvider(
        base_url="https://example.invalid",
        path="/v1/systemone",
        api_key="key",
        model="jev-latest",
        retries=0,
        session=session,
    )
    result = await provider.evaluate(state="state", questions=questions)
    assert set(result.answers) == {"n", "c"}
    assert len(session.calls) == 3


def test_provider_error_does_not_echo_api_key():
    key = "secret-key-value"
    provider = SystemOneProvider(
        base_url="https://example.invalid",
        path="/v1/systemone",
        api_key=key,
        model="jev-latest",
        retries=0,
        session=FakeSession([FakeResponse(401, {"error": key})]),
    )
    with pytest.raises(DecisionProviderError) as exc_info:
        asyncio.run(
            provider.evaluate(
                state="state",
                questions={"q": {"type": "noul", "instructions": "x"}},
            )
        )
    assert key not in str(exc_info.value)


def test_routing_always_keep_handoff_and_decision_tool():
    ordinary = SimpleNamespace(name="ordinary", description="ordinary", active=True)
    keep = SimpleNamespace(name="keep", description="keep", active=True)
    handoff = type("HandoffTool", (), {"name": "transfer_to_search", "description": "search"})()
    decision = SimpleNamespace(name="decision_evaluate", description="decision")
    original = [ordinary, keep, handoff]
    outcome = choose_tools(
        original,
        {"q": {"noul": 0.1}},
        {"q": "ordinary"},
        threshold=0.2,
        always_keep={"keep"},
        decision_tool=decision,
        always_keep_recommend={"keep"},
    )
    assert [tool.name for tool in outcome.selected] == [
        "keep",
        "transfer_to_search",
        "decision_evaluate",
    ]
    assert [tool.name for tool in outcome.handoffs] == ["transfer_to_search"]
    assert outcome.recommended_tools == ["keep"]


def test_each_noul_can_recommend_zero_one_or_many_subagents():
    answers = {
        "a": {"noul": 0.1},
        "b": {"noul": 0.8},
        "c": {"noul": 0.9},
    }
    mapping = {"a": "agent_a", "b": "agent_b", "c": "agent_c"}
    assert recommendations_from_noul(answers, mapping, threshold=0.2) == ["agent_b", "agent_c"]
    assert (
        recommendations_from_noul({key: {"noul": 0.1} for key in mapping}, mapping, threshold=0.2)
        == []
    )


def test_filter_switches_keep_all_or_filter_each_candidate_independently():
    ordinary = SimpleNamespace(name="ordinary", description="ordinary", active=True)
    dropped = SimpleNamespace(name="dropped", description="dropped", active=True)
    handoff_a = type("HandoffTool", (), {"name": "agent_a", "description": "a"})()
    handoff_b = type("HandoffTool", (), {"name": "agent_b", "description": "b"})()
    decision = SimpleNamespace(name="decision_evaluate", description="decision")
    answers = {
        "ordinary_q": {"noul": 0.8},
        "dropped_q": {"noul": 0.1},
        "agent_a_q": {"noul": 0.8},
        "agent_b_q": {"noul": 0.1},
    }
    ordinary_questions = {"ordinary_q": "ordinary", "dropped_q": "dropped"}
    handoff_questions = {"agent_a_q": "agent_a", "agent_b_q": "agent_b"}

    keep_all = choose_tools(
        [ordinary, dropped, handoff_a, handoff_b],
        answers,
        ordinary_questions,
        threshold=0.2,
        always_keep=set(),
        decision_tool=decision,
        question_to_handoff=handoff_questions,
        filter_ordinary=False,
        filter_handoffs=False,
    )
    assert [tool.name for tool in keep_all.selected] == [
        "ordinary",
        "dropped",
        "agent_a",
        "agent_b",
        "decision_evaluate",
    ]
    assert keep_all.recommended_tools == ["ordinary"]

    filter_tools_only = choose_tools(
        [ordinary, dropped, handoff_a, handoff_b],
        answers,
        ordinary_questions,
        threshold=0.2,
        always_keep=set(),
        decision_tool=decision,
        question_to_handoff=handoff_questions,
        filter_ordinary=True,
        filter_handoffs=False,
    )
    assert [tool.name for tool in filter_tools_only.selected] == [
        "ordinary",
        "agent_a",
        "agent_b",
        "decision_evaluate",
    ]
    assert filter_tools_only.recommended_tools == ["ordinary"]

    filter_both = choose_tools(
        [ordinary, dropped, handoff_a, handoff_b],
        answers,
        ordinary_questions,
        threshold=0.2,
        always_keep=set(),
        decision_tool=decision,
        question_to_handoff=handoff_questions,
        filter_ordinary=True,
        filter_handoffs=True,
    )
    assert [tool.name for tool in filter_both.selected] == [
        "ordinary",
        "agent_a",
        "decision_evaluate",
    ]
    assert filter_both.recommended_tools == ["ordinary"]


def test_always_keep_without_recommendation_stays_silent_even_when_jev_selects_it():
    keep = SimpleNamespace(name="keep", description="keep", active=True)
    decision = SimpleNamespace(name="decision_evaluate", description="decision")
    outcome = choose_tools(
        [keep],
        {"keep_q": {"noul": 0.9}},
        {"keep_q": "keep"},
        threshold=0.2,
        always_keep={"keep"},
        decision_tool=decision,
        always_keep_recommend=set(),
    )
    assert [tool.name for tool in outcome.selected] == ["keep", "decision_evaluate"]
    assert outcome.recommended_tools == []


def test_recommendation_is_appended_to_system_prompt_and_preserves_other_plugins():
    req = SimpleNamespace(
        system_prompt="AstrBot persona\n\nOther plugin prompt", extra_user_content_parts=[]
    )
    add_routing_hint(
        req,
        recommended_tools=["search", "calendar"],
        recommended_subagents=["transfer_to_search", "transfer_to_code"],
    )
    hint = req.system_prompt
    assert "当前场景需要根据用户请求选择合适的能力" in hint
    assert "search, calendar" in hint
    assert "transfer_to_search, transfer_to_code" in hint
    assert "AstrBot persona" in hint
    assert "Other plugin prompt" in hint
    assert req.extra_user_content_parts == []


def test_custom_routing_template_replaces_only_supported_placeholders():
    rendered = render_routing_prompt(
        'tools={tools}; agents={subagents}; json={"keep": true}',
        recommended_tools=["search"],
        recommended_subagents=[],
    )
    assert rendered == 'tools=search; agents=(无); json={"keep": true}'


def test_append_system_prompt_is_idempotent_for_this_plugin_section():
    req = SimpleNamespace(system_prompt="persona")
    append_system_prompt(req, "recommendation")
    first = req.system_prompt
    append_system_prompt(req, "recommendation")
    assert req.system_prompt == first


def test_state_excludes_system_prompt_and_tool_schema():
    state = build_decision_state(
        policy="short policy",
        current_prompt="find cats",
        contexts=[{"role": "user", "content": "older"}],
        tools=[{"name": "search", "description": "search the web"}],
        subagents=[],
    )
    assert "short policy" in state
    assert "find cats" in state
    assert "search the web" in state
    assert "system_prompt" not in state
    assert "parameters" not in state


def test_public_schema_hides_custom_page_prompts_and_page_switches():
    schema = json.loads(
        (Path(__file__).resolve().parents[1] / "_conf_schema.json").read_text(encoding="utf-8")
    )
    assert schema["jev_pre_prompt"]["invisible"] is True
    assert schema["main_llm_post_prompt"]["invisible"] is True
    assert schema["tool_filter_enabled"]["invisible"] is True
    assert schema["tool_filter_enabled"]["default"] is True
    assert schema["subagent_filter_enabled"]["invisible"] is True
    assert schema["subagent_filter_enabled"]["default"] is False
    assert schema["tools_subagents_decision_enabled"]["default"] is True
    assert "不影响主动对话" in schema["tools_subagents_decision_enabled"]["description"]
    assert "decision_policy" not in schema
    assert "subagent_recommendation_enabled" not in schema
    assert "enable" not in schema


def test_context_limits_history_and_truncates_tool_results():
    lines = context_lines(
        [
            {"role": "user", "content": "old"},
            {"role": "tool", "content": "x" * 1000},
            {"role": "user", "content": "current"},
        ],
        max_messages=3,
        max_chars=700,
        current_prompt="current",
    )
    assert not any(line.endswith("current") for line in lines)
    assert len(next(line for line in lines if line.startswith("tool:"))) <= 506


def test_question_id_is_stable_and_collision_resistant():
    assert make_question_id("tool", "web-search", 0) == make_question_id("tool", "web-search", 0)
    assert make_question_id("tool", "web-search", 0) != make_question_id("tool", "web search", 0)


def test_decision_tool_schema_has_valid_array_items():
    schema = decision_tool_parameters()
    assert schema["properties"]["choice_options"]["items"]["type"] == "object"
    assert schema["properties"]["score_levels"]["items"] == {"type": "string"}


def test_choose_tools_fail_open_shape_preserves_handoffs_and_decision_tool():
    ordinary = SimpleNamespace(name="ordinary", description="ordinary", active=True)
    handoff = type("HandoffTool", (), {"name": "transfer_to_search", "description": "search"})()
    decision = SimpleNamespace(name="decision_evaluate", description="decision")
    outcome = choose_tools(
        [ordinary, handoff],
        {},
        {"ordinary": "missing-answer"},
        threshold=0.2,
        always_keep=set(),
        decision_tool=decision,
    )
    assert [tool.name for tool in outcome.selected] == ["transfer_to_search", "decision_evaluate"]


def test_proactive_history_records_bot_and_user_roles():
    state = ProactiveState(max_messages=3)
    state.add("g", ProactiveRecord("Alice", "1", "hello"))
    state.add("g", ProactiveRecord("bot", "0", "reply", True))
    assert state.history_lines("g", 200) == ["[Alice]: hello", "[bot]: reply"]


def test_proactive_cooldown_and_window():
    state = ProactiveState()
    assert state.allow_reply("g", cooldown_seconds=60, window_seconds=600, max_replies=2)
    assert not state.allow_reply("g", cooldown_seconds=60, window_seconds=600, max_replies=2)
