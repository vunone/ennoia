"""FastMCP server smoke tests driven by the in-process ``fastmcp.Client``."""

from __future__ import annotations

from typing import Any

import pytest
from fastmcp import Client

from ennoia import BaseSemantic, BaseStructure, Pipeline
from ennoia.server import ServerContext, no_auth
from ennoia.server.mcp import create_mcp
from ennoia.testing import MockEmbeddingAdapter, MockLLMAdapter, MockStore


class _Doc(BaseStructure):
    """Extract doc metadata."""

    cat: str


class Summary(BaseSemantic):
    """Summarise doc."""


class Facts(BaseSemantic):
    """Facts of doc."""


def _pipeline(
    store: MockStore | None = None,
    *,
    semantics: tuple[type[BaseSemantic], ...] = (Summary,),
) -> Pipeline:
    return Pipeline(
        schemas=[_Doc, *semantics],
        store=(store or MockStore()),
        llm=MockLLMAdapter(
            json_responses=lambda _: {"cat": "legal", "extraction_confidence": 0.9},
            text_responses=lambda _: "a summary",
        ),
        embedding=MockEmbeddingAdapter(dim=4),
    )


def _server(pipeline: Pipeline | None = None) -> tuple[Pipeline, Any]:
    pipe = pipeline or _pipeline()
    ctx = ServerContext(pipeline=pipe, auth=no_auth())
    return pipe, create_mcp(ctx)


def _unwrap(payload: Any) -> Any:
    """FastMCP wraps list/None tool returns under a ``result`` envelope."""
    if isinstance(payload, dict) and "result" in payload:
        return payload["result"]
    return payload


async def test_list_tools_shows_three_readonly_tools() -> None:
    _, mcp = _server()
    async with Client(mcp) as client:
        tools = await client.list_tools()
    names = sorted(t.name for t in tools)
    assert names == ["discover_schema", "retrieve", "search"]


async def test_discover_schema_returns_payload() -> None:
    _, mcp = _server()
    async with Client(mcp) as client:
        result = await client.call_tool("discover_schema", {})
    payload = result.structured_content
    assert payload is not None
    assert "structural_fields" in payload
    assert any(f["name"] == "cat" for f in payload["structural_fields"])


async def test_discover_search_retrieve_agent_flow() -> None:
    store = MockStore()
    pipe = _pipeline(store)
    await pipe.aindex(text="some text", source_id="doc_1")

    _, mcp = _server(pipe)
    async with Client(mcp) as client:
        # Single `search` call — filter and semantic ranking together.
        search_result = await client.call_tool(
            "search",
            {"query": "anything", "filter": {"cat": "legal"}},
        )
        hits = _unwrap(search_result.structured_content)
        assert isinstance(hits, list)
        assert len(hits) == 1
        assert hits[0]["source_id"] == "doc_1"
        assert hits[0]["structural"].get("cat") == "legal"

        # Full record pulled separately.
        retrieved = await client.call_tool("retrieve", {"id": "doc_1"})
        record = _unwrap(retrieved.structured_content)
        assert isinstance(record, dict)
        assert record.get("cat") == "legal"


async def test_search_without_filter_returns_all_docs() -> None:
    store = MockStore()
    pipe = _pipeline(store)
    await pipe.aindex(text="doc one", source_id="doc_1")
    await pipe.aindex(text="doc two", source_id="doc_2")

    _, mcp = _server(pipe)
    async with Client(mcp) as client:
        result = await client.call_tool("search", {"query": "anything"})
    hits = _unwrap(result.structured_content)
    assert {h["source_id"] for h in hits} == {"doc_1", "doc_2"}


async def test_search_filter_excludes_nonmatching_docs() -> None:
    store = MockStore()
    pipe = _pipeline(store)
    await pipe.aindex(text="doc one", source_id="doc_1")

    _, mcp = _server(pipe)
    async with Client(mcp) as client:
        result = await client.call_tool(
            "search",
            {"query": "anything", "filter": {"cat": "not-a-category"}},
        )
    hits = _unwrap(result.structured_content)
    assert hits == []


async def test_search_limit_caps_hits() -> None:
    store = MockStore()
    pipe = _pipeline(store)
    await pipe.aindex(text="doc one", source_id="doc_1")
    await pipe.aindex(text="doc two", source_id="doc_2")

    _, mcp = _server(pipe)
    async with Client(mcp) as client:
        result = await client.call_tool("search", {"query": "anything", "limit": 1})
    hits = _unwrap(result.structured_content)
    assert len(hits) == 1


async def test_search_index_pins_ranking_to_one_semantic_index() -> None:
    store = MockStore()
    pipe = _pipeline(store, semantics=(Summary, Facts))
    await pipe.aindex(text="doc one", source_id="doc_1")

    _, mcp = _server(pipe)
    async with Client(mcp) as client:
        result = await client.call_tool(
            "search",
            {"query": "anything", "index": "Summary"},
        )
    hits = _unwrap(result.structured_content)
    assert hits, "expected at least one hit"
    for hit in hits:
        assert set(hit["semantic"].keys()) == {"Summary"}


async def test_retrieve_missing_returns_none() -> None:
    _, mcp = _server()
    async with Client(mcp) as client:
        result = await client.call_tool("retrieve", {"id": "nope"})
    payload = result.structured_content
    assert payload is None or payload == {"result": None}


async def test_search_filter_validation_error_propagates() -> None:
    _, mcp = _server()
    async with Client(mcp) as client:
        with pytest.raises(Exception):  # noqa: BLE001 — any tool error is acceptable
            await client.call_tool(
                "search",
                {"query": "x", "filter": {"unknown_field": "x"}},
            )


async def test_server_exposes_instructions() -> None:
    """The FastMCP server ships LLM-facing instructions describing tool usage."""
    _, mcp = _server()
    instructions = mcp.instructions or ""
    assert instructions, "FastMCP instructions must not be empty"
    # Key phrases that guide the LLM through the canonical flow.
    assert "discover_schema" in instructions
    assert "filter" in instructions
    assert "retrieve" in instructions


async def test_tool_descriptions_are_llm_facing() -> None:
    """Each tool exposes a multi-line docstring rather than a terse one-liner."""
    _, mcp = _server()
    async with Client(mcp) as client:
        tools = await client.list_tools()
    by_name = {t.name: t for t in tools}
    for name in ("discover_schema", "search", "retrieve"):
        description = by_name[name].description or ""
        assert description.strip(), f"{name} must expose a description"
    search_desc = by_name["search"].description or ""
    assert "filter" in search_desc
    assert "index" in search_desc
