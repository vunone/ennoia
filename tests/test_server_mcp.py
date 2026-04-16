"""FastMCP server smoke tests driven by the in-process ``fastmcp.Client``."""

from __future__ import annotations

import pytest
from fastmcp import Client

from ennoia import BaseSemantic, BaseStructure, Pipeline
from ennoia.server import ServerContext, no_auth
from ennoia.server.mcp import create_mcp
from ennoia.testing import MockEmbeddingAdapter, MockLLMAdapter, MockStore


class _Doc(BaseStructure):
    """Extract doc metadata."""

    cat: str


class _Summary(BaseSemantic):
    """Summarise doc."""


def _pipeline(store: MockStore | None = None) -> Pipeline:
    return Pipeline(
        schemas=[_Doc, _Summary],
        store=(store or MockStore()),
        llm=MockLLMAdapter(
            json_responses=lambda _: {"cat": "legal", "_confidence": 0.9},
            text_responses=lambda _: "a summary",
        ),
        embedding=MockEmbeddingAdapter(dim=4),
    )


def _server(pipeline: Pipeline | None = None) -> tuple[Pipeline, object]:
    pipe = pipeline or _pipeline()
    ctx = ServerContext(pipeline=pipe, auth=no_auth())
    return pipe, create_mcp(ctx)


async def test_list_tools_shows_four_readonly_tools() -> None:
    _, mcp = _server()
    async with Client(mcp) as client:
        tools = await client.list_tools()
    names = sorted(t.name for t in tools)
    assert names == ["discover_schema", "filter", "retrieve", "search"]


async def test_discover_schema_returns_payload() -> None:
    _, mcp = _server()
    async with Client(mcp) as client:
        result = await client.call_tool("discover_schema", {})
    payload = result.structured_content
    assert payload is not None
    assert "structural_fields" in payload
    assert any(f["name"] == "cat" for f in payload["structural_fields"])


async def test_filter_discover_search_retrieve_agent_flow() -> None:
    store = MockStore()
    pipe = _pipeline(store)
    await pipe.aindex(text="some text", source_id="doc_1")

    _, mcp = _server(pipe)
    async with Client(mcp) as client:
        # Step 1: filter to get candidate ids
        filter_result = await client.call_tool("filter", {"filters": {"cat": "legal"}})
        ids_payload = filter_result.structured_content
        # Tools returning lists surface them wrapped under 'result' by fastmcp.
        ids = ids_payload["result"] if isinstance(ids_payload, dict) else ids_payload
        assert ids == ["doc_1"]

        # Step 2: search with filter_ids
        search_result = await client.call_tool(
            "search",
            {"query": "anything", "filter_ids": ["doc_1"], "top_k": 3},
        )
        hits_payload = search_result.structured_content
        hits = (
            hits_payload["result"]
            if isinstance(hits_payload, dict) and "result" in hits_payload
            else hits_payload
        )
        assert isinstance(hits, list)

        # Step 3: retrieve full record — fastmcp wraps dict returns under "result".
        retrieved = await client.call_tool("retrieve", {"id": "doc_1"})
        record_payload = retrieved.structured_content
        record = (
            record_payload["result"]
            if isinstance(record_payload, dict) and "result" in record_payload
            else record_payload
        )
        assert isinstance(record, dict)
        assert record.get("cat") == "legal"


async def test_retrieve_missing_returns_none() -> None:
    _, mcp = _server()
    async with Client(mcp) as client:
        result = await client.call_tool("retrieve", {"id": "nope"})
    payload = result.structured_content
    # FastMCP wraps ``None`` returns under a structured envelope or returns None directly.
    assert payload is None or payload == {"result": None}


async def test_filter_validation_error_propagates() -> None:
    _, mcp = _server()
    async with Client(mcp) as client:
        with pytest.raises(Exception):  # noqa: BLE001 — any tool error is acceptable
            await client.call_tool("filter", {"filters": {"unknown_field": "x"}})
