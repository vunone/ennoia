"""Unit tests for the ennoia agent loop in the product benchmark harness.

Stubs the OpenAI-shaped client and the underlying ennoia pipeline so the
agent loop's control flow runs without any API calls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest
from openai import APIConnectionError, APITimeoutError

from benchmark.config import AGENT_SYSTEM_PROMPT, MAX_AGENT_ITERATIONS, RETRIEVAL_TOP_K
from benchmark.data.prep import Product
from benchmark.pipelines import ennoia_pipeline as epkg
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
        self.closed = False

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *_: Any) -> None:
        self.closed = True


def _make_response(
    *,
    text: str | None = None,
    tool_calls: list[tuple[str, str, dict[str, Any]]] | None = None,
    prompt_tokens: int = 100,
    completion_tokens: int = 20,
) -> _FakeResponse:
    calls = (
        [_FakeToolCall(id=cid, name=n, arguments=json.dumps(a)) for cid, n, a in tool_calls]
        if tool_calls
        else None
    )
    return _FakeResponse(
        choices=[_FakeChoice(message=_FakeAssistantMessage(content=text, tool_calls=calls))],
        usage=_FakeUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


class _FakeEnnoiaCore:
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
    core = _FakeEnnoiaCore(schemas_payload=[], search_hits=search_hits, records=records)
    pipe._pipeline = core  # type: ignore[attr-defined]
    pipe._gen_model = "fake-model"  # type: ignore[attr-defined]
    pipe._threads = 1  # type: ignore[attr-defined]
    pipe._max_iterations = max_iterations  # type: ignore[attr-defined]
    client = _FakeClient(responses)
    pipe._client_factory = lambda: client  # type: ignore[attr-defined]
    return pipe, client, core


_QUERY = "aluminium laptop stand with rgb lights"


async def test_agent_discover_search_getfull_then_answer() -> None:
    hits = [
        SearchHit(
            source_id="p-1",
            score=0.9,
            structural={"brand": "ACME"},
            semantic={"ProductSummary": "aluminium laptop stand"},
        ),
        SearchHit(
            source_id="p-2",
            score=0.6,
            structural={"brand": "BetaCo"},
            semantic={"ProductSummary": "plastic stand"},
        ),
    ]
    responses = [
        _make_response(tool_calls=[("c1", "get_search_schema", {})]),
        _make_response(
            tool_calls=[
                (
                    "c2",
                    "search",
                    {"query": "aluminium stand", "filter": {"brand__eq": "ACME"}},
                )
            ]
        ),
        _make_response(tool_calls=[("c3", "get_full", {"document_id": "p-1"})]),
        _make_response(text='{"docid":"p-1","reason":"aluminium stand by ACME"}'),
    ]
    pipe, client, core = _make_pipeline(
        search_hits={"aluminium stand": hits},
        records={"p-1": {"brand": "ACME"}},
        responses=responses,
    )

    run = await pipe.answer(_QUERY)
    assert run.answer == '{"docid":"p-1","reason":"aluminium stand by ACME"}'
    assert run.retrieved_source_ids == ["p-1", "p-2"]
    assert [t["tool"] for t in run.trace] == ["get_search_schema", "search", "get_full"]
    assert core.retrieve_calls == ["p-1"]
    assert core.search_calls == [
        {
            "query": "aluminium stand",
            "filters": {"brand__eq": "ACME"},
            "top_k": RETRIEVAL_TOP_K,
        }
    ]
    assert len(client.chat.completions.calls) == 4


async def test_agent_aggregates_retrieved_over_multiple_searches() -> None:
    responses = [
        _make_response(tool_calls=[("c1", "search", {"query": "first"})]),
        _make_response(tool_calls=[("c2", "search", {"query": "second"})]),
        _make_response(text="done"),
    ]
    hits = {
        "first": [SearchHit(source_id="a", score=0.8, structural={}, semantic={})],
        "second": [
            SearchHit(source_id="a", score=0.8, structural={}, semantic={}),  # dup dedups
            SearchHit(source_id="b", score=0.6, structural={}, semantic={}),
        ],
    }
    pipe, _client, _core = _make_pipeline(search_hits=hits, responses=responses)
    run = await pipe.answer(_QUERY)
    assert run.retrieved_source_ids == ["a", "b"]


async def test_agent_forces_extra_turn_when_all_iters_emit_tool_calls() -> None:
    """All ``MAX_AGENT_ITERATIONS`` turns emit tool calls → the for/else
    branch kicks in with a forced ``tool_choice="none"`` turn."""
    hits = {"x": [SearchHit(source_id="z", score=0.5, structural={}, semantic={})]}
    responses: list[_FakeResponse] = [
        _make_response(tool_calls=[("c", "search", {"query": "x"})])
        for _ in range(MAX_AGENT_ITERATIONS)
    ]
    # Extra response consumed by the forced tool_choice="none" turn.
    responses.append(_make_response(text="forced reply"))
    pipe, client, _core = _make_pipeline(search_hits=hits, responses=responses)
    run = await pipe.answer(_QUERY)
    assert run.answer == "forced reply"
    # One extra call beyond MAX_AGENT_ITERATIONS (the forced turn).
    assert len(client.chat.completions.calls) == MAX_AGENT_ITERATIONS + 1
    assert client.chat.completions.calls[-1]["tool_choice"] == "none"


async def test_agent_forces_final_turn_on_iteration_exhaustion() -> None:
    hits = {"x": [SearchHit(source_id="z", score=0.5, structural={}, semantic={})]}
    responses: list[_FakeResponse] = [
        _make_response(tool_calls=[("c", "search", {"query": "x"})])
        for _ in range(MAX_AGENT_ITERATIONS - 1)
    ]
    responses.append(_make_response(text="partial final answer"))
    pipe, client, _core = _make_pipeline(search_hits=hits, responses=responses)
    run = await pipe.answer(_QUERY)
    assert run.answer == "partial final answer"
    assert client.chat.completions.calls[-1]["tool_choice"] == "none"


async def test_agent_defaults_to_not_found_on_empty_final_text() -> None:
    pipe, _client, _core = _make_pipeline(responses=[_make_response(text="")])
    run = await pipe.answer(_QUERY)
    assert run.answer == "NOT_FOUND"


def test_agent_prompt_includes_required_tools_and_notfound() -> None:
    assert "get_search_schema" in AGENT_SYSTEM_PROMPT
    assert "search(" in AGENT_SYSTEM_PROMPT
    assert "get_full" in AGENT_SYSTEM_PROMPT
    assert "NOT_FOUND" in AGENT_SYSTEM_PROMPT
    assert "docid" in AGENT_SYSTEM_PROMPT


_PRODUCT: Product = {
    "docid": "p-1",
    "title": "Aluminium Stand",
    "text": "body",
    "bullet_points": ["aluminium", "foldable"],
    "brand": "ACME",
    "color": "silver",
    "price_usd": 75,
}


def test_product_text_includes_all_salient_fields() -> None:
    text = epkg._product_text(_PRODUCT)
    assert "Aluminium Stand" in text
    assert "Brand: ACME" in text
    assert "Color: silver" in text
    # Price line is the anchor ProductMeta.price_usd reads from.
    assert "Price: $75" in text
    assert "aluminium" in text


def test_product_text_skips_unknown_brand_and_color() -> None:
    product: Product = {
        "docid": "x",
        "title": "T",
        "text": "",
        "bullet_points": [],
        "brand": "unknown",
        "color": "unknown",
        "price_usd": 20,
    }
    text = epkg._product_text(product)
    assert "Brand:" not in text
    assert "Color:" not in text
    # Price ALWAYS appears — there is no "unknown" price; ProductMeta must
    # always extract one, so the anchor must be in the text regardless.
    assert "Price: $20" in text


async def test_aindex_retries_on_transient(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr(epkg.asyncio, "sleep", fake_sleep)

    calls = {"n": 0}

    async def flaky(*, text: str, source_id: str) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise APIConnectionError(request=None)  # type: ignore[arg-type]
        if calls["n"] == 2:
            raise APITimeoutError(request=None)  # type: ignore[arg-type]
        return None

    pipe, _client, core = _make_pipeline(responses=[])
    core.aindex = flaky  # type: ignore[attr-defined, method-assign]
    await pipe._aindex_with_retry(_PRODUCT)
    assert calls["n"] == 3


async def test_aindex_raises_after_exhaustion(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr(epkg.asyncio, "sleep", fake_sleep)

    async def always_timeout(*, text: str, source_id: str) -> None:
        raise APITimeoutError(request=None)  # type: ignore[arg-type]

    pipe, _client, core = _make_pipeline(responses=[])
    core.aindex = always_timeout  # type: ignore[attr-defined, method-assign]
    with pytest.raises(APITimeoutError):
        await pipe._aindex_with_retry(_PRODUCT)


async def test_index_corpus_records_failures(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def fake_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr(epkg.asyncio, "sleep", fake_sleep)

    async def mixed(*, text: str, source_id: str) -> None:
        if source_id == "broken":
            raise APITimeoutError(request=None)  # type: ignore[arg-type]
        return None

    pipe, _client, core = _make_pipeline(responses=[])
    core.aindex = mixed  # type: ignore[attr-defined, method-assign]

    products: list[Product] = [
        {**_PRODUCT, "docid": "ok"},  # type: ignore[typeddict-item]
        {**_PRODUCT, "docid": "broken"},  # type: ignore[typeddict-item]
    ]
    await pipe.index_corpus(products)
    captured = capsys.readouterr().out
    assert "1/2 products failed" in captured
    assert "broken" in captured


async def test_search_tool_error_returned_as_tool_message() -> None:
    class _FailingCore(_FakeEnnoiaCore):
        async def asearch(
            self, query: str, filters: dict[str, Any] | None = None, top_k: int = 10
        ) -> SearchResult:
            raise RuntimeError("simulated filter validation failure")

    responses = [
        _make_response(tool_calls=[("c1", "search", {"query": "q"})]),
        _make_response(text="NOT_FOUND"),
    ]
    pipe, client, _core = _make_pipeline(responses=responses)
    pipe._pipeline = _FailingCore(schemas_payload=[])  # type: ignore[attr-defined]
    run = await pipe.answer(_QUERY)
    assert run.answer == "NOT_FOUND"
    tool_msg = next(m for m in client.chat.completions.calls[-1]["messages"] if m["role"] == "tool")
    assert "simulated filter validation failure" in tool_msg["content"]


async def test_get_full_requires_document_id() -> None:
    responses = [
        _make_response(tool_calls=[("c1", "get_full", {"document_id": ""})]),
        _make_response(text="NOT_FOUND"),
    ]
    pipe, client, _core = _make_pipeline(responses=responses)
    await pipe.answer(_QUERY)
    tool_msg = next(m for m in client.chat.completions.calls[-1]["messages"] if m["role"] == "tool")
    assert "document_id is required" in tool_msg["content"]


async def test_unknown_tool_returns_error_payload() -> None:
    responses = [
        _make_response(tool_calls=[("c1", "nonexistent", {})]),
        _make_response(text="NOT_FOUND"),
    ]
    pipe, client, _core = _make_pipeline(responses=responses)
    await pipe.answer(_QUERY)
    tool_msg = next(m for m in client.chat.completions.calls[-1]["messages"] if m["role"] == "tool")
    assert "unknown tool" in tool_msg["content"]


async def test_get_full_returns_record() -> None:
    responses = [
        _make_response(tool_calls=[("c1", "get_full", {"document_id": "p-1"})]),
        _make_response(text="ok"),
    ]
    pipe, client, core = _make_pipeline(
        records={"p-1": {"brand": "ACME", "price_usd": 75}},
        responses=responses,
    )
    await pipe.answer(_QUERY)
    tool_msg = next(m for m in client.chat.completions.calls[-1]["messages"] if m["role"] == "tool")
    assert "p-1" in tool_msg["content"]
    assert "ACME" in tool_msg["content"]
    assert core.retrieve_calls == ["p-1"]


def test_assistant_message_dict_preserves_tool_calls() -> None:
    msg = _FakeAssistantMessage(
        content=None,
        tool_calls=[_FakeToolCall(id="c1", name="search", arguments='{"query":"x"}')],
    )
    out = epkg._assistant_message_dict(msg)
    assert out["role"] == "assistant"
    assert out["tool_calls"][0]["id"] == "c1"
    assert out["tool_calls"][0]["function"]["name"] == "search"


def test_assistant_message_dict_without_tool_calls() -> None:
    msg = _FakeAssistantMessage(content="final", tool_calls=None)
    out = epkg._assistant_message_dict(msg)
    assert out["tool_calls"] == []
    assert out["content"] == "final"


def test_default_openrouter_client_is_async_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    client = epkg._default_openrouter_client()
    assert client.__class__.__name__ == "AsyncOpenAI"


def test_ennoia_pipeline_init_wires_product_schemas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class _FakeCore:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(epkg, "EnnoiaCorePipeline", _FakeCore)
    pipe = EnnoiaPipeline(gen_model="g", embed_model="text-embedding-3-small", threads=2)
    assert pipe._gen_model == "g"
    assert pipe._threads == 2
    assert captured["concurrency"] == 2
    schema_names = [s.__name__ for s in captured["schemas"]]
    assert schema_names == ["ProductMeta", "ProductSummary"]


async def test_index_corpus_succeeds_without_failures_prints_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def ok_aindex(*, text: str, source_id: str) -> None:
        return None

    pipe, _client, core = _make_pipeline(responses=[])
    core.aindex = ok_aindex  # type: ignore[attr-defined, method-assign]
    await pipe.index_corpus([{**_PRODUCT, "docid": f"p{i}"} for i in range(3)])  # type: ignore[typeddict-item]
    captured = capsys.readouterr().out
    assert "failed to index" not in captured


async def test_index_corpus_prints_overflow_when_many_failures(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def fake_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr(epkg.asyncio, "sleep", fake_sleep)

    async def always_fail(*, text: str, source_id: str) -> None:
        raise APITimeoutError(request=None)  # type: ignore[arg-type]

    pipe, _client, core = _make_pipeline(responses=[])
    core.aindex = always_fail  # type: ignore[attr-defined, method-assign]
    products = [{**_PRODUCT, "docid": f"p{i}"} for i in range(15)]  # type: ignore[typeddict-item]
    await pipe.index_corpus(products)
    out = capsys.readouterr().out
    # Exactly 5 overflow entries ("... and 5 more") beyond the 10 printed.
    assert "and 5 more" in out


async def test_agent_response_without_usage_still_aggregates() -> None:
    """A response lacking ``usage`` must not crash the loop."""

    @dataclass
    class _NoUsageResponse:
        choices: list[_FakeChoice]

    responses = [
        _NoUsageResponse(
            choices=[_FakeChoice(message=_FakeAssistantMessage(content="ok", tool_calls=None))]
        ),
    ]
    pipe, _client, _core = _make_pipeline(responses=responses)  # type: ignore[arg-type]
    run = await pipe.answer(_QUERY)
    assert run.answer == "ok"
    assert run.prompt_tokens == 0
    assert run.completion_tokens == 0


async def test_forced_turn_without_usage_still_aggregates() -> None:
    """Forced-final turn must handle response without ``usage`` too."""

    @dataclass
    class _NoUsageResponse:
        choices: list[_FakeChoice]

    responses: list[Any] = [
        _make_response(tool_calls=[("c", "search", {"query": "q"})])
        for _ in range(MAX_AGENT_ITERATIONS)
    ]
    responses.append(
        _NoUsageResponse(
            choices=[_FakeChoice(message=_FakeAssistantMessage(content="forced", tool_calls=None))]
        )
    )
    pipe, _client, _core = _make_pipeline(
        search_hits={"q": [SearchHit(source_id="z", score=0.1, structural={}, semantic={})]},
        responses=responses,
    )
    run = await pipe.answer(_QUERY)
    assert run.answer == "forced"


async def test_invalid_tool_json_parses_as_empty_args() -> None:
    @dataclass
    class _BadToolCall:
        id: str
        name: str
        arguments: str

        @property
        def function(self) -> _BadToolCall:
            return self

    msg = _FakeAssistantMessage(
        content=None,
        tool_calls=[_BadToolCall(id="c1", name="get_search_schema", arguments="not{json")],
    )
    responses: list[_FakeResponse] = [
        _FakeResponse(choices=[_FakeChoice(message=msg)], usage=_FakeUsage(10, 5)),
        _make_response(text="ok"),
    ]
    pipe, _client, _core = _make_pipeline(responses=responses)
    run = await pipe.answer(_QUERY)
    assert run.answer == "ok"
