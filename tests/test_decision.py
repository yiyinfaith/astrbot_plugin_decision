from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from decision.context import (
    build_decision_state,
    context_lines,
    is_mcp_tool,
    is_skill_tool,
    tool_summary,
)
from decision.context import question_id as make_question_id
from decision.models import DecisionProviderError, DecisionValidationError, parse_systemone_response
from decision.proactive import (
    DialogueInference,
    ProactiveRecord,
    ProactiveState,
    ProactiveStatus,
    aggregate_scores,
    infer_dialogue_target,
    normalize_prefixes,
    parse_noul_scores,
    text_matches_prefix,
)
from decision.providers.systemone import SystemOneProvider
from decision.routing import (
    add_routing_hint,
    append_system_prompt,
    choose_tools,
    decision_tool_parameters,
    recommendations_from_noul,
    render_routing_prompt,
)
from decision.skills import filter_skills_prompt
from decision.tokens import estimate_request_tokens, estimate_tokens, fit_request_state


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
        retry_backoff_sec=0,
        session=session,
    )
    result = await provider.evaluate(state="state", questions=questions)
    assert result.answers["q"]["noul"] == 0.8
    assert len(session.calls) == 2


@pytest.mark.asyncio
async def test_provider_retry_is_immediate_and_default_timeout_is_five_seconds(monkeypatch):
    questions = {"q": {"type": "noul", "instructions": "x"}}
    session = FakeSession(
        [
            FakeResponse(500, {"error": "temporary"}),
            FakeResponse(payload=answer_payload(questions)),
        ]
    )
    sleeps = []

    async def remember_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", remember_sleep)
    provider = SystemOneProvider(
        base_url="https://example.invalid",
        path="/v1/systemone",
        api_key="key",
        model="jev-latest",
        session=session,
        retries=1,
    )
    assert provider.timeout_sec == 5.0
    await provider.evaluate(state="state", questions=questions)
    assert sleeps == [0]


@pytest.mark.asyncio
async def test_provider_uses_configured_retry_backoff(monkeypatch):
    questions = {"q": {"type": "noul", "instructions": "x"}}
    session = FakeSession(
        [
            FakeResponse(500, {"error": "temporary"}),
            FakeResponse(payload=answer_payload(questions)),
        ]
    )
    sleeps = []

    async def remember_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", remember_sleep)
    provider = SystemOneProvider(
        base_url="https://example.invalid",
        path="/v1/systemone",
        api_key="key",
        model="jev-latest",
        retries=1,
        retry_backoff_sec=1.5,
        session=session,
    )
    await provider.evaluate(state="state", questions=questions)
    assert sleeps == [1.5]


@pytest.mark.asyncio
async def test_provider_notifies_retry_logger():
    questions = {"q": {"type": "noul", "instructions": "x"}}
    session = FakeSession(
        [
            FakeResponse(500, {"error": "temporary"}),
            FakeResponse(payload=answer_payload(questions)),
        ]
    )
    retries = []
    provider = SystemOneProvider(
        base_url="https://example.invalid",
        path="/v1/systemone",
        api_key="key",
        model="jev-latest",
        retries=1,
        retry_logger=lambda *values: retries.append(values),
        session=session,
    )
    await provider.evaluate(state="state", questions=questions)
    assert retries == [(1, 1, "SystemOne server error (500)", 0.0)]


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
async def test_provider_adapts_when_server_limit_is_smaller_than_nominal_chunk_size():
    questions = {f"q_{index}": {"type": "noul", "instructions": "x"} for index in range(5)}
    # The first request and every payload larger than two questions is rejected.
    # The provider must split rejected chunks again rather than retrying the same
    # default-sized body forever.
    responses = [FakeResponse(413, {"error": "too many questions"})]
    responses.extend(
        [
            FakeResponse(413, {"error": "too many questions"}),
            FakeResponse(payload=answer_payload({key: questions[key] for key in ("q_0", "q_1")})),
            FakeResponse(413, {"error": "too many questions"}),
            FakeResponse(payload=answer_payload({"q_2": questions["q_2"]})),
            FakeResponse(payload=answer_payload({key: questions[key] for key in ("q_3", "q_4")})),
        ]
    )
    session = FakeSession(responses)
    provider = SystemOneProvider(
        base_url="https://example.invalid",
        path="/v1/systemone",
        api_key="key",
        model="jev-latest",
        retries=0,
        chunk_size=32,
        session=session,
    )
    result = await provider.evaluate(state="state", questions=questions)
    assert set(result.answers) == set(questions)


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
    decision = SimpleNamespace(name="jev_decide", description="decision")
    original = [ordinary, keep, handoff]
    outcome = choose_tools(
        original,
        {"q": {"noul": 0.1}},
        {"q": "ordinary"},
        threshold=0.2,
        always_keep={"keep", "jev_decide"},
        decision_tool=decision,
        always_keep_recommend={"keep"},
    )
    assert [tool.name for tool in outcome.selected] == [
        "keep",
        "transfer_to_search",
        "jev_decide",
    ]
    assert [tool.name for tool in outcome.handoffs] == ["transfer_to_search"]
    assert outcome.recommended_tools == ["keep"]


def test_routing_manual_subagent_keep_and_recommend_are_independent():
    agent_keep = type("HandoffTool", (), {"name": "agent_keep", "description": "keep"})()
    agent_recommend = type(
        "HandoffTool", (), {"name": "agent_recommend", "description": "recommend"}
    )()
    agent_jev = type("HandoffTool", (), {"name": "agent_jev", "description": "jev"})()
    answers = {
        "keep_q": {"noul": 0.1},
        "recommend_q": {"noul": 0.1},
        "jev_q": {"noul": 0.9},
    }
    outcome = choose_tools(
        [agent_keep, agent_recommend, agent_jev],
        answers,
        {},
        threshold=0.65,
        always_keep=set(),
        decision_tool=None,
        always_keep_handoffs={"agent_keep", "agent_recommend"},
        always_keep_handoffs_recommend={"agent_recommend"},
        question_to_handoff={
            "keep_q": "agent_keep",
            "recommend_q": "agent_recommend",
            "jev_q": "agent_jev",
        },
        filter_handoffs=True,
    )
    assert [tool.name for tool in outcome.selected] == [
        "agent_keep",
        "agent_recommend",
        "agent_jev",
    ]
    assert outcome.recommended_subagents == ["agent_recommend", "agent_jev"]


def test_routing_never_readds_inactive_always_kept_tool():
    inactive = SimpleNamespace(name="inactive", description="offline", active=False)
    outcome = choose_tools(
        [inactive],
        {},
        {},
        threshold=0.2,
        always_keep={"inactive"},
        decision_tool=None,
        filter_ordinary=False,
    )
    assert outcome.selected == []


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
    decision = SimpleNamespace(name="jev_decide", description="decision")
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
        always_keep={"jev_decide"},
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
        "jev_decide",
    ]
    assert keep_all.recommended_tools == ["ordinary"]

    filter_tools_only = choose_tools(
        [ordinary, dropped, handoff_a, handoff_b],
        answers,
        ordinary_questions,
        threshold=0.2,
        always_keep={"jev_decide"},
        decision_tool=decision,
        question_to_handoff=handoff_questions,
        filter_ordinary=True,
        filter_handoffs=False,
    )
    assert [tool.name for tool in filter_tools_only.selected] == [
        "ordinary",
        "agent_a",
        "agent_b",
        "jev_decide",
    ]
    assert filter_tools_only.recommended_tools == ["ordinary"]

    filter_both = choose_tools(
        [ordinary, dropped, handoff_a, handoff_b],
        answers,
        ordinary_questions,
        threshold=0.2,
        always_keep={"jev_decide"},
        decision_tool=decision,
        question_to_handoff=handoff_questions,
        filter_ordinary=True,
        filter_handoffs=True,
    )
    assert [tool.name for tool in filter_both.selected] == [
        "ordinary",
        "agent_a",
        "jev_decide",
    ]
    assert filter_both.recommended_tools == ["ordinary"]


def test_jev_selection_recommends_kept_tool_but_manual_only_keep_stays_silent():
    keep = SimpleNamespace(name="keep", description="keep", active=True)
    decision = SimpleNamespace(name="jev_decide", description="decision")
    outcome = choose_tools(
        [keep],
        {"keep_q": {"noul": 0.9}},
        {"keep_q": "keep"},
        threshold=0.2,
        always_keep={"keep", "jev_decide"},
        decision_tool=decision,
        always_keep_recommend=set(),
    )
    assert [tool.name for tool in outcome.selected] == ["keep", "jev_decide"]
    assert outcome.recommended_tools == ["keep"]
    manual_only = choose_tools(
        [keep],
        {},
        {},
        threshold=0.2,
        always_keep={"keep", "jev_decide"},
        decision_tool=decision,
        always_keep_recommend=set(),
    )
    assert manual_only.recommended_tools == []

    opt_out = choose_tools(
        [keep],
        {},
        {},
        threshold=0.2,
        always_keep=set(),
        decision_tool=decision,
        filter_ordinary=False,
    )
    assert [tool.name for tool in opt_out.selected] == ["keep"]


def test_builtin_jev_tool_can_be_manually_recommended_from_page_settings():
    decision = SimpleNamespace(name="jev_decide", description="decision")
    outcome = choose_tools(
        [],
        {},
        {},
        threshold=0.2,
        always_keep={"jev_decide"},
        decision_tool=decision,
        always_keep_recommend={"jev_decide"},
    )
    assert [tool.name for tool in outcome.selected] == ["jev_decide"]
    assert outcome.recommended_tools == ["jev_decide"]

    existing = choose_tools(
        [decision],
        {},
        {},
        threshold=0.2,
        always_keep={"jev_decide"},
        decision_tool=decision,
        always_keep_recommend={"jev_decide"},
    )
    assert existing.recommended_tools == ["jev_decide"]


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


def test_routing_template_replaces_mcp_placeholder():
    rendered = render_routing_prompt(
        "tools={tools}; mcp={mcps}; agents={subagents}",
        recommended_tools=[],
        recommended_mcp=["filesystem_read"],
        recommended_subagents=[],
    )
    assert rendered == "tools=(无); mcp=filesystem_read; agents=(无)"


def test_mcp_tool_detection_uses_server_identity():
    mcp_tool = SimpleNamespace(
        name="filesystem_read", mcp_server_name="filesystem", active=True
    )
    ordinary = SimpleNamespace(name="filesystem_read", active=True)
    assert is_mcp_tool(mcp_tool)
    assert not is_mcp_tool(ordinary)


def test_mcp_filter_keep_and_recommend_are_independent():
    selected = SimpleNamespace(
        name="filesystem_read", description="read files", mcp_server_name="filesystem", active=True
    )
    dropped = SimpleNamespace(
        name="filesystem_write", description="write files", mcp_server_name="filesystem", active=True
    )
    outcome = choose_tools(
        [selected, dropped],
        {"selected_q": {"noul": 0.9}, "dropped_q": {"noul": 0.1}},
        {},
        threshold=0.65,
        always_keep=set(),
        decision_tool=None,
        always_keep_mcp={"filesystem_write"},
        always_keep_mcp_recommend={"filesystem_write"},
        question_to_mcp={"selected_q": "filesystem_read", "dropped_q": "filesystem_write"},
        filter_mcp=True,
    )
    assert [tool.name for tool in outcome.selected] == ["filesystem_read", "filesystem_write"]
    assert outcome.recommended_mcp == ["filesystem_read", "filesystem_write"]


def test_mcp_filter_off_keeps_unselected_and_still_recommends_only_selected():
    mcp_tool = SimpleNamespace(
        name="filesystem_read", description="read files", mcp_server_name="filesystem", active=True
    )
    outcome = choose_tools(
        [mcp_tool],
        {"mcp_q": {"noul": 0.1}},
        {},
        threshold=0.65,
        always_keep=set(),
        decision_tool=None,
        question_to_mcp={"mcp_q": "filesystem_read"},
        filter_mcp=False,
    )
    assert [tool.name for tool in outcome.selected] == ["filesystem_read"]
    assert outcome.recommended_mcp == []


def test_skill_routing_supports_filter_and_manual_keep_recommend_independently():
    selected = SimpleNamespace(name="selected_skill", description="selected", active=True)
    manual = SimpleNamespace(name="manual_skill", description="manual", active=True)
    dropped = SimpleNamespace(name="dropped_skill", description="dropped", active=True)
    outcome = choose_tools(
        [],
        {
            "selected_q": {"noul": 0.9},
            "dropped_q": {"noul": 0.1},
        },
        {},
        threshold=0.65,
        always_keep=set(),
        decision_tool=None,
        skills=[selected, manual, dropped],
        always_keep_skills={"manual_skill"},
        always_keep_skills_recommend={"manual_skill"},
        question_to_skill={"selected_q": "selected_skill", "dropped_q": "dropped_skill"},
        filter_skills=True,
    )
    assert [item.name for item in outcome.selected_skills] == ["selected_skill", "manual_skill"]
    assert outcome.recommended_skills == ["selected_skill", "manual_skill"]


def test_skill_filter_off_keeps_all_but_manual_recommendations_need_the_checkbox():
    skills = [
        SimpleNamespace(name="a", description="a", active=True),
        SimpleNamespace(name="b", description="b", active=True),
    ]
    outcome = choose_tools(
        [],
        {"a_q": {"noul": 0.9}, "b_q": {"noul": 0.1}},
        {},
        threshold=0.65,
        always_keep=set(),
        decision_tool=None,
        skills=skills,
        always_keep_skills={"a"},
        always_keep_skills_recommend=set(),
        question_to_skill={"a_q": "a", "b_q": "b"},
        filter_skills=False,
    )
    assert [item.name for item in outcome.selected_skills] == ["a", "b"]
    assert outcome.recommended_skills == ["a"]


def test_skill_detection_is_narrow_and_skill_prompt_filter_preserves_other_sections():
    class SkillInfo:
        name = "x"
        skill_name = "x"
        source_type = "local_only"
        path = "/skills/x/SKILL.md"
        active = True

    skill = SkillInfo()
    # A real SkillInfo-like class is accepted; a plain function tool is not.
    assert is_skill_tool(skill)
    assert not is_skill_tool(SimpleNamespace(name="tool", description="tool"))
    prompt = (
        "persona\n## Skills\n\n### Available skills\n\n"
        "- **keep**: keep it\n  File: `/skills/keep/SKILL.md`\n"
        "- **drop**: drop it\n  File: `/skills/drop/SKILL.md`\n"
        "\n### Skill rules\n\n1. rule\n## Other\nrest"
    )
    filtered = filter_skills_prompt(prompt, ["keep"])
    assert "**keep**" in filtered
    assert "**drop**" not in filtered
    assert "## Other\nrest" in filtered


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


def test_decision_state_has_a_separate_skill_section():
    state = build_decision_state(
        policy="policy",
        current_prompt="make a spreadsheet",
        contexts=[],
        skills=[{"name": "spreadsheet_skill", "description": "spreadsheet help"}],
    )
    assert "[Available Skills]" in state
    assert "spreadsheet_skill: spreadsheet help" in state


def test_public_schema_contains_only_global_settings_and_master_switches():
    schema = json.loads(
        (Path(__file__).resolve().parents[1] / "_conf_schema.json").read_text(encoding="utf-8")
    )
    assert set(schema) == {
        "provider",
        "base_url",
        "systemone_path",
        "api_key",
        "model",
        "timeout_sec",
        "retries",
        "retry_backoff_seconds",
        "model_context_tokens",
        "history_max_messages",
        "history_max_chars",
        "tools_subagents_decision_enabled",
        "proactive_reply_enabled",
        "conversation_flow_enabled",
    }
    assert all("[" not in item.get("description", "") for item in schema.values())
    assert schema["tools_subagents_decision_enabled"]["default"] is True
    assert schema["tools_subagents_decision_enabled"]["description"] == "Tool/MCP/Skill/SubAgent 决策"
    assert "不影响主动对话" in schema["tools_subagents_decision_enabled"]["hint"]
    assert schema["proactive_reply_enabled"]["default"] is False
    assert "详细配置在插件 WebUI" in schema["proactive_reply_enabled"]["hint"]
    assert schema["retry_backoff_seconds"]["default"] == 0.0
    assert "默认 1" in schema["retries"]["hint"]
    assert schema["model_context_tokens"]["default"] == 32000
    assert schema["timeout_sec"]["default"] == 5.0
    detailed_fields = {
        "jev_pre_prompt",
        "main_llm_post_prompt",
        "tool_filter_enabled",
        "mcp_filter_enabled",
        "subagent_filter_enabled",
        "tool_noul_threshold",
        "always_keep_tools",
        "always_keep_recommend_tools",
        "always_keep_mcp",
        "always_keep_recommend_mcp",
        "always_keep_subagents",
        "always_keep_recommend_subagents",
        "proactive_whitelist",
        "direct_reply_prefixes",
        "force_reply_when_summoned",
        "proactive_score_threshold",
        "cooldown_seconds",
    }
    assert not detailed_fields.intersection(schema)


def test_builtin_jev_tool_is_default_keep_only_in_settings_page():
    page = (Path(__file__).resolve().parents[1] / "pages" / "settings" / "index.html").read_text(
        encoding="utf-8"
    )
    assert "<title>jev决策综合插件设置</title>" in page
    assert "<h1>jev决策综合插件设置</h1>" in page
    assert "始终推荐给主 LLM" in page
    assert "tool.builtin" in page
    assert "builtin" in page
    assert "#ba96ff" in page
    assert "#6ce0bf" in page
    assert "origin_display" in page
    assert "Vue.createApp" not in page
    assert "createApp" in page
    assert "https://glass.goose.cc.cd/liquid-glass.js" in page
    assert "setTabs" in page
    assert "background: linear-gradient" in page
    assert "只负责结构化判断的 Decision Model" in page
    assert "当前场景需要根据用户请求选择合适的能力" in page
    assert "payloadOf" in page
    assert "waitForBridge" in page
    assert "fallbackBridge" not in page
    assert "JSON.parse(JSON.stringify(body))" in page
    assert "TOOL_THRESHOLD_DEFAULT = .65" in page
    assert "数值越高越严格" in page
    assert "always_keep_subagents" in page
    assert "always_keep_recommend_subagents" in page
    assert "tool_decision_enabled" in page
    assert "subagent_decision_enabled" in page
    assert "tools_scope_blacklist" in page
    assert "tools_scope" in page
    assert "锁定内置顺序" not in page
    assert "始终按内置顺序执行" in page
    assert "routing-rules-table" in page
    assert "四类能力的 Jev 判断会合并为一次请求" in page
    assert "mcp_decision_enabled" in page
    assert "mcp_filter_enabled" in page
    assert "always_keep_mcp" in page
    assert "always_keep_skills" in page
    assert "always_keep_recommend_skills" in page
    assert "skill_decision_enabled" in page
    assert "skill_filter_enabled" in page
    assert "Tool/MCP/Skill/SubAgent" in page
    assert "<h2>输入增强</h2>" in page
    assert "conversation_flow_analysis_enabled" in page
    assert "conversation_flow_window" in page
    for field in (
        "outputList('summary','quotes_files')",
        "outputConfig.error.custom_msg",
        "outputConfig.block.block_reread",
        "outputConfig.at.at_str",
        "outputConfig.clean.punctuation",
        "outputList('clean','lead')",
        "outputList('replace','words')",
        "outputConfig.typo.tone_error_rate",
        "outputConfig.tts.character_id",
        "outputConfig.t2i.pillowmd_style_dir",
        "outputConfig.reply.threshold",
        "outputConfig.forward.threshold",
        "outputList('recall','keywords')",
        "outputConfig.split.delay_scope_str",
        "outputList('split','tail_punc')",
    ):
        assert field in page


def test_proactive_details_are_page_only_and_excluded_from_public_schema():
    schema = json.loads(
        (Path(__file__).resolve().parents[1] / "_conf_schema.json").read_text(encoding="utf-8")
    )
    page = (Path(__file__).resolve().parents[1] / "pages" / "settings" / "index.html").read_text(
        encoding="utf-8"
    )
    for field in (
        "proactive_whitelist",
        "direct_reply_prefixes",
        "force_reply_when_summoned",
        "conversation_flow_analysis_enabled",
        "conversation_flow_window",
        "proactive_score_threshold",
        "observation_timeout_seconds",
    ):
        assert field in page
        assert field not in schema
    descriptions = " ".join(item.get("description", "") for item in schema.values())
    for excluded in ("工具调用提示", "安抚机制", "AI人格设定", "接管astrbot原生上下文"):
        assert excluded not in descriptions


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


def test_token_budget_helpers_keep_requests_strictly_under_limit():
    questions = {"q": {"type": "noul", "instructions": "判断"}}
    state = "旧消息\n" * 200 + "当前消息"
    assert estimate_tokens("中文") >= 2
    fitted = fit_request_state(state, questions, 64)
    assert estimate_request_tokens(fitted, questions) < 64
    assert fitted.endswith("当前消息")


def test_tool_summary_does_not_apply_a_per_tool_description_cap():
    tool = SimpleNamespace(name="large", description="x" * 1000)
    assert len(tool_summary(tool)["description"]) == 1000


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
    decision = SimpleNamespace(name="jev_decide", description="decision")
    outcome = choose_tools(
        [ordinary, handoff],
        {},
        {"ordinary": "missing-answer"},
        threshold=0.2,
        always_keep={"jev_decide"},
        decision_tool=decision,
    )
    assert [tool.name for tool in outcome.selected] == ["transfer_to_search", "jev_decide"]


def test_proactive_history_records_bot_and_user_roles():
    state = ProactiveState(max_messages=3)
    state.add("g", ProactiveRecord("Alice", "1", "hello"))
    state.add("g", ProactiveRecord("bot", "0", "reply", True))
    assert state.history_lines("g", 200) == ["[Alice]: hello", "[bot]: reply"]


def test_dialogue_flow_prefers_explicit_bot_and_other_targets():
    bot = ProactiveRecord(
        "Alice", "u1", "@bot hi", at_targets=(("bot-id", "Bot"),), timestamp=100
    )
    result = infer_dialogue_target(bot, [], bot_id="bot-id", now=100)
    assert result == DialogueInference("bot", "你", 1.0, "explicit_at_bot")

    other = ProactiveRecord(
        "Alice", "u1", "@Bob hi", at_targets=(("u2", "Bob"),), timestamp=100
    )
    result = infer_dialogue_target(other, [], bot_id="bot-id", now=100)
    assert result.target_id == "u2"
    assert result.target_name == "Bob"
    assert result.reason == "explicit_at_other"

    # AstrBot's AtAll is represented as a target named ``all`` by some
    # adapters; it addresses the group and must not become a user target.
    group_mention = ProactiveRecord(
        "Alice", "u1", "@all hi", at_targets=(("all", "全体成员"),), timestamp=100
    )
    result = infer_dialogue_target(group_mention, [], bot_id="bot-id", now=100)
    assert result == DialogueInference(reason="default_group")


def test_dialogue_flow_uses_reply_bot_aba_and_conservative_group_fallback():
    reply = ProactiveRecord("Alice", "u1", "thanks", reply_to_id="bot-id", timestamp=100)
    assert infer_dialogue_target(reply, [], bot_id="bot-id", now=100).target_id == "bot"

    recent_bot = ProactiveRecord(
        "bot", "bot-id", "answer", True, timestamp=90, talking_to="u1", talking_to_name="Alice"
    )
    short_ack = ProactiveRecord("Alice", "u1", "好的", timestamp=100)
    assert infer_dialogue_target(short_ack, [recent_bot], bot_id="bot-id", now=100).reason == "bot_recently_replied"

    previous = ProactiveRecord(
        "Bob", "u2", "问 Alice", timestamp=90, talking_to="u1", talking_to_name="Alice"
    )
    follow = ProactiveRecord("Alice", "u1", "我来了", timestamp=100)
    result = infer_dialogue_target(follow, [previous], bot_id="bot-id", now=100)
    assert result.target_id == "u2"
    assert result.reason == "aba_pattern"

    unrelated = ProactiveRecord("Alice", "u1", "随便说说", timestamp=100)
    result = infer_dialogue_target(
        unrelated,
        [ProactiveRecord("Bob", "u2", "群里说话", timestamp=70)],
        bot_id="bot-id",
        now=100,
    )
    assert result.target_id == "group"
    assert result.reason == "default_group"


def test_proactive_history_can_render_identity_aware_dialogue_flow():
    state = ProactiveState(max_messages=4)
    state.add(
        "g",
        ProactiveRecord(
            "Alice", "u1", "hello", talking_to="u2", talking_to_name="Bob"
        ),
    )
    assert state.history_lines("g", 200, include_dialogue=True) == [
        "[Alice (u1) → Bob (u2)]: hello"
    ]


def test_proactive_reconfigure_applies_live_history_and_session_limits():
    state = ProactiveState(max_messages=5, max_sessions=3)
    for index in range(5):
        state.add("g", ProactiveRecord("user", str(index), str(index)))
    state.reconfigure(max_messages=2, max_sessions=1)
    assert state.history_lines("g", 200) == ["[user]: 3", "[user]: 4"]
    state.add("other", ProactiveRecord("user", "x", "x"))
    state.add("third", ProactiveRecord("user", "y", "y"))
    assert len(state.sessions) <= 1


def test_proactive_cooldown_and_window():
    state = ProactiveState()
    assert state.allow_reply("g", cooldown_seconds=60, window_seconds=600, max_replies=2)
    assert not state.allow_reply("g", cooldown_seconds=60, window_seconds=600, max_replies=2)
    assert not state.allow_reply("disabled", cooldown_seconds=0, window_seconds=600, max_replies=0)


def test_proactive_cancel_reply_releases_unqueued_slot():
    state = ProactiveState()
    assert state.allow_reply("g", cooldown_seconds=60, window_seconds=600, max_replies=1)
    state.cancel_reply("g")
    assert not state.has_pending_reply("g")
    assert state.allow_reply("g", cooldown_seconds=60, window_seconds=600, max_replies=1)


def test_proactive_analysis_cooldown_does_not_drop_new_history():
    state = ProactiveState(max_messages=4)
    state.add("g", ProactiveRecord("Alice", "1", "first"))
    state.mark_analysis("g", success=True, no_reply_cooldown=60)
    assert not state.can_analyze("g")
    state.add("g", ProactiveRecord("Bob", "2", "during cooldown"))
    assert state.history_lines("g", 200)[-1] == "[Bob]: during cooldown"


def test_direct_prefix_matching_reserves_at_for_real_message_components():
    assert normalize_prefixes(["/", "@", "", "/"]) == ["/", "@"]
    assert text_matches_prefix("/hello", ["/", "@"])
    assert not text_matches_prefix("hello @bot", ["/", "@"])
    assert not text_matches_prefix("@bot hello", ["@"])


def test_proactive_scores_are_strict_and_weighted():
    scores = parse_noul_scores(
        {
            "good": {"noul": 0.8},
            "bad": {"noul": "0.9"},
            "clamped": {"noul": 2},
            "bool": {"noul": True},
        },
        ["good", "bad", "clamped", "bool"],
    )
    assert scores == {"good": 0.8, "clamped": 1.0}
    assert aggregate_scores(scores, {"good": 3, "clamped": 1}) == pytest.approx(0.85)


def test_proactive_state_tracks_echo_status_and_per_session_locks():
    state = ProactiveState(max_messages=5)
    state.add("g", ProactiveRecord("Alice", "1", "same"))
    state.add("g", ProactiveRecord("Bob", "2", "same"))
    state.add("g", ProactiveRecord("Carol", "3", "same"))
    status = state.observe_message(
        "g",
        text="same",
        sender_id="4",
        echo_threshold=3,
        echo_window=60,
        dense_threshold=99,
    )
    assert status == ProactiveStatus.GETTING_FAMILIAR
    assert state.lock_for("g") is state.lock_for("g")
    assert state.lock_for("g") is not state.lock_for("other")


def test_proactive_state_session_limit_does_not_grow_without_bound():
    state = ProactiveState(max_sessions=2)
    state.add("one", ProactiveRecord("a", "1", "x"))
    state.add("two", ProactiveRecord("b", "2", "x"))
    state.add("three", ProactiveRecord("c", "3", "x"))
    assert len(state.sessions) <= 2


def test_normal_llm_response_does_not_create_proactive_state():
    state = ProactiveState()
    assert not state.has_pending_reply("ordinary")
    assert "ordinary" not in state.sessions
