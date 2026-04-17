"""Unit tests for the ennoia agent loop in the benchmark harness.

The agent loop is benchmark-specific (uses the OpenAI SDK directly rather
than ennoia's adapter) and handles the discover/search/get_full dispatch.
These tests stub the SDK + the underlying ennoia pipeline so we exercise
the loop's control flow without any API calls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest
from openai import APIConnectionError, APITimeoutError

from benchmark.config import AGENT_SYSTEM_PROMPT, MAX_AGENT_ITERATIONS
from benchmark.data.loader import Contract, Question
from benchmark.pipelines import ennoia_pipeline
from benchmark.pipelines.ennoia_pipeline import EnnoiaPipeline
from ennoia.index.result import SearchHit, SearchResult


@dataclass
class _FakeToolCall:
    id: str
    name: str
    arguments: str

    @property
    def function(self) -> _FakeToolCall:
        return self

    @property
    def type(self) -> str:
        return "function"


@dataclass
class _FakeAssistantMessage:
    content: str | None
    tool_calls: list[_FakeToolCall] | None


@dataclass
class _FakeChoice:
    message: _FakeAssistantMessage


@dataclass
class _FakeUsage:
    prompt_tokens: int
    completion_tokens: int


@dataclass
class _FakeResponse:
    choices: list[_FakeChoice]
    usage: _FakeUsage


class _FakeCompletions:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _FakeResponse:
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("FakeCompletions ran out of queued responses")
        return self._responses.pop(0)


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions = completions


class _FakeClient:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self.chat = _FakeChat(_FakeCompletions(responses))


def _make_response(
    *,
    text: str | None = None,
    tool_calls: list[tuple[str, str, dict[str, Any]]] | None = None,
    prompt_tokens: int = 100,
    completion_tokens: int = 20,
) -> _FakeResponse:
    calls = (
        [
            _FakeToolCall(id=call_id, name=name, arguments=json.dumps(args))
            for call_id, name, args in tool_calls
        ]
        if tool_calls
        else None
    )
    msg = _FakeAssistantMessage(content=text, tool_calls=calls)
    return _FakeResponse(
        choices=[_FakeChoice(message=msg)],
        usage=_FakeUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


class _FakeEnnoiaCore:
    """Minimal stand-in for ``ennoia.Pipeline`` — the agent only touches
    ``schemas()``, ``asearch``, and ``aretrieve``."""

    def __init__(
        self,
        schemas_payload: list[Any],
        search_hits: dict[str, list[SearchHit]] | None = None,
        records: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._schemas = schemas_payload
        self._search_hits = search_hits or {}
        self._records = records or {}
        self.search_calls: list[dict[str, Any]] = []
        self.retrieve_calls: list[str] = []

    def schemas(self) -> list[Any]:
        return self._schemas

    async def asearch(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
    ) -> SearchResult:
        self.search_calls.append({"query": query, "filters": filters, "top_k": top_k})
        return SearchResult(hits=self._search_hits.get(query, []))

    async def aretrieve(self, source_id: str) -> dict[str, Any] | None:
        self.retrieve_calls.append(source_id)
        return self._records.get(source_id)


def _make_pipeline(
    *,
    search_hits: dict[str, list[SearchHit]] | None = None,
    records: dict[str, dict[str, Any]] | None = None,
    responses: list[_FakeResponse],
    max_iterations: int = MAX_AGENT_ITERATIONS,
) -> tuple[EnnoiaPipeline, _FakeClient, _FakeEnnoiaCore]:
    pipe = EnnoiaPipeline.__new__(EnnoiaPipeline)
    core = _FakeEnnoiaCore(
        schemas_payload=[],
        search_hits=search_hits,
        records=records,
    )
    pipe._pipeline = core  # type: ignore[attr-defined]
    pipe._gen_model = "fake-nano"  # type: ignore[attr-defined]
    pipe._max_iterations = max_iterations  # type: ignore[attr-defined]
    client = _FakeClient(responses)
    pipe._client_factory = lambda: client  # type: ignore[attr-defined]
    return pipe, client, core


QUESTION: Question = {
    "question_id": "q-1",
    "contract_id": "contract-1",
    "question": "Does this contract include a non-compete clause?",
    "category": "Non-Compete",
    "gold_answers": ["Non-compete of 5 years..."],
    "has_answer": True,
}


async def test_agent_emits_discover_search_then_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hits = [
        SearchHit(
            source_id="contract-1",
            score=0.9,
            structural={"contract_type": "License"},
            semantic={"ClauseMention": "non-compete for 5 years in NA"},
        ),
        SearchHit(
            source_id="contract-2",
            score=0.6,
            structural={"contract_type": "NDA"},
            semantic={"ClauseMention": "mutual confidentiality"},
        ),
    ]
    responses = [
        _make_response(tool_calls=[("c1", "get_search_schema", {})]),
        _make_response(
            tool_calls=[
                (
                    "c2",
                    "search",
                    {
                        "query": "non-compete",
                        "filter": {"clauses_present__contains": "competition_exclusivity"},
                        "limit": 5,
                    },
                )
            ]
        ),
        _make_response(
            text="Yes — 5-year non-compete in NA.",
            prompt_tokens=50,
            completion_tokens=15,
        ),
    ]
    pipe, client, core = _make_pipeline(
        search_hits={"non-compete": hits},
        responses=responses,
    )

    run = await pipe.answer(QUESTION)

    assert run.answer == "Yes — 5-year non-compete in NA."
    assert run.retrieved_source_ids == ["contract-1", "contract-2"]
    # Usage summed across three turns.
    assert run.prompt_tokens == 100 + 100 + 50
    assert run.completion_tokens == 20 + 20 + 15
    # Discovered first, searched next — order preserved in trace.
    assert [entry["tool"] for entry in run.trace] == ["get_search_schema", "search"]
    # Underlying pipeline saw the filter the agent chose.
    assert core.search_calls == [
        {
            "query": "non-compete",
            "filters": {"clauses_present__contains": "competition_exclusivity"},
            "top_k": 5,
        }
    ]
    # Three chat completions calls in total (discover, search, final answer).
    assert len(client.chat.completions.calls) == 3
    # Last call had tool_choice="auto" because we had margin left.
    assert client.chat.completions.calls[-1]["tool_choice"] == "auto"


async def test_agent_dispatches_get_full_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hits = [
        SearchHit(source_id="contract-7", score=0.8, structural={}, semantic={"o": "gist"}),
    ]
    records = {"contract-7": {"contract_type": "Lease", "parties": ["Acme", "Beta"]}}
    responses = [
        _make_response(tool_calls=[("c1", "get_search_schema", {})]),
        _make_response(tool_calls=[("c2", "search", {"query": "parties"})]),
        _make_response(tool_calls=[("c3", "get_full", {"document_id": "contract-7"})]),
        _make_response(text="Acme and Beta are the parties."),
    ]
    pipe, _client, core = _make_pipeline(
        search_hits={"parties": hits},
        records=records,
        responses=responses,
    )

    run = await pipe.answer(QUESTION)

    assert core.retrieve_calls == ["contract-7"]
    assert run.retrieved_source_ids == ["contract-7"]
    assert run.answer == "Acme and Beta are the parties."


async def test_agent_forces_final_turn_when_iterations_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hits = [SearchHit(source_id="contract-x", score=0.5, structural={}, semantic={})]
    # Fill every iteration with tool calls so the loop hits the is_last
    # branch (tool_choice="none") instead of exiting via a text-only reply.
    responses: list[_FakeResponse] = []
    for _ in range(MAX_AGENT_ITERATIONS - 1):
        responses.append(_make_response(tool_calls=[("c", "search", {"query": "x"})]))
    # Final iteration is forced tool_choice="none" — we reply with text so
    # the loop exits normally.
    responses.append(_make_response(text="Partial answer despite exhaustion."))
    pipe, client, _core = _make_pipeline(
        search_hits={"x": hits},
        responses=responses,
    )

    run = await pipe.answer(QUESTION)

    assert run.answer == "Partial answer despite exhaustion."
    assert len(client.chat.completions.calls) == MAX_AGENT_ITERATIONS
    # The last call must have been made with tool_choice="none".
    assert client.chat.completions.calls[-1]["tool_choice"] == "none"


async def test_agent_defaults_to_not_found_on_empty_final_text() -> None:
    responses = [_make_response(text="")]
    pipe, _client, _core = _make_pipeline(responses=responses)
    run = await pipe.answer(QUESTION)
    assert run.answer == "NOT_FOUND"


def test_agent_prompt_mandates_get_full_on_candidate_hit() -> None:
    # Regression guard for the B1 fix: snippet-only answers starve the
    # generator of extractive context, so the prompt must direct the agent to
    # call get_full on the top candidate before answering. The previous
    # "escape hatch only" phrasing produced 70% abstention even when recall
    # succeeded.
    prompt = AGENT_SYSTEM_PROMPT
    assert "get_full" in prompt
    # Must explicitly instruct the agent to call get_full on a matching hit.
    assert "call `get_full" in prompt or "call get_full" in prompt
    # And must still preserve the NOT_FOUND escape for weak/no candidates.
    assert "NOT_FOUND" in prompt


CONTRACT: Contract = {
    "source_id": "contract-1",
    "title": "contract-1",
    "text": "Body of contract 1.",
}


async def test_aindex_retries_transient_api_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Fail twice with retryable OpenAI errors, then succeed on attempt 3.
    calls = {"n": 0}

    async def flaky_aindex(*, text: str, source_id: str) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise APIConnectionError(request=None)  # type: ignore[arg-type]
        if calls["n"] == 2:
            raise APITimeoutError(request=None)  # type: ignore[arg-type]
        return None

    pipe, _client, core = _make_pipeline(responses=[])
    core.aindex = flaky_aindex  # type: ignore[attr-defined, method-assign]
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(ennoia_pipeline.asyncio, "sleep", fake_sleep)

    await pipe._aindex_with_retry(CONTRACT)

    assert calls["n"] == 3
    # Linear backoff: sleep = initial * attempt_index, so the first
    # failure (attempt_index=0) triggers an immediate retry and the
    # second (attempt_index=1) sleeps for the base backoff.
    assert sleeps == [
        ennoia_pipeline._INDEX_RETRY_INITIAL_BACKOFF_SEC * 0,
        ennoia_pipeline._INDEX_RETRY_INITIAL_BACKOFF_SEC * 1,
    ]


async def test_aindex_raises_after_exhausting_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def always_timeout(*, text: str, source_id: str) -> None:
        raise APITimeoutError(request=None)  # type: ignore[arg-type]

    pipe, _client, core = _make_pipeline(responses=[])
    core.aindex = always_timeout  # type: ignore[attr-defined, method-assign]

    async def fake_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr(ennoia_pipeline.asyncio, "sleep", fake_sleep)

    with pytest.raises(APITimeoutError):
        await pipe._aindex_with_retry(CONTRACT)


async def test_aindex_does_not_retry_non_transient_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Non-transient exception (e.g. validation error inside the extraction
    # pipeline) must surface on the first attempt — retrying it wastes
    # tokens and masks a real bug.
    calls = {"n": 0}

    async def boom(*, text: str, source_id: str) -> None:
        calls["n"] += 1
        raise ValueError("schema validation failed")

    pipe, _client, core = _make_pipeline(responses=[])
    core.aindex = boom  # type: ignore[attr-defined, method-assign]

    with pytest.raises(ValueError, match="schema validation failed"):
        await pipe._aindex_with_retry(CONTRACT)
    assert calls["n"] == 1


async def test_index_corpus_records_only_final_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Two contracts: one recovers on retry, the other stays broken.
    call_counts: dict[str, int] = {}

    async def mixed_aindex(*, text: str, source_id: str) -> None:
        call_counts[source_id] = call_counts.get(source_id, 0) + 1
        if source_id == "flaky" and call_counts[source_id] < 2:
            raise APIConnectionError(request=None)  # type: ignore[arg-type]
        if source_id == "broken":
            raise APITimeoutError(request=None)  # type: ignore[arg-type]
        return None

    pipe, _client, core = _make_pipeline(responses=[])
    core.aindex = mixed_aindex  # type: ignore[attr-defined, method-assign]

    async def fake_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr(ennoia_pipeline.asyncio, "sleep", fake_sleep)

    contracts: list[Contract] = [
        {"source_id": "ok", "title": "ok", "text": "text"},
        {"source_id": "flaky", "title": "flaky", "text": "text"},
        {"source_id": "broken", "title": "broken", "text": "text"},
    ]
    await pipe.index_corpus(contracts)

    # ok succeeds on attempt 1; flaky recovers on attempt 2; broken
    # exhausts the retry budget (``_INDEX_RETRY_ATTEMPTS`` total calls).
    assert call_counts == {
        "ok": 1,
        "flaky": 2,
        "broken": ennoia_pipeline._INDEX_RETRY_ATTEMPTS,
    }
    captured = capsys.readouterr().out
    assert "1/3 contracts failed to index" in captured
    assert "broken" in captured
    assert "flaky" not in captured.split("contracts failed to index")[-1]


async def test_search_tool_error_is_returned_to_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # When asearch raises, we want the error surfaced back to the agent as
    # a tool-result so it can retry / back off, not propagate out of the
    # pipeline.
    class _FailingCore(_FakeEnnoiaCore):
        async def asearch(
            self,
            query: str,
            filters: dict[str, Any] | None = None,
            top_k: int = 10,
        ) -> SearchResult:
            raise RuntimeError("simulated filter validation failure")

    responses = [
        _make_response(tool_calls=[("c1", "search", {"query": "anything"})]),
        _make_response(text="NOT_FOUND"),
    ]
    pipe, client, _core = _make_pipeline(responses=responses)
    pipe._pipeline = _FailingCore(schemas_payload=[])  # type: ignore[attr-defined]

    run = await pipe.answer(QUESTION)
    assert run.answer == "NOT_FOUND"
    # Tool result message payload contains the error string, not an exception.
    tool_msg = next(m for m in client.chat.completions.calls[-1]["messages"] if m["role"] == "tool")
    assert "simulated filter validation failure" in tool_msg["content"]
