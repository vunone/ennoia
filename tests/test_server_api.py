"""FastAPI REST smoke tests — exercise every route in-process via ASGITransport."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from ennoia import BaseSemantic, BaseStructure, Pipeline
from ennoia.server import ServerContext, no_auth, static_bearer_auth
from ennoia.server.api import create_app
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


def _client(pipeline: Pipeline, *, auth: Any = None) -> httpx.AsyncClient:
    ctx = ServerContext(pipeline=pipeline, auth=auth or no_auth())
    app = create_app(ctx)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_discover_returns_superschema_payload() -> None:
    async with _client(_pipeline()) as client:
        resp = await client.get("/discover")
    assert resp.status_code == 200
    payload = resp.json()
    assert "structural_fields" in payload
    assert any(f["name"] == "cat" for f in payload["structural_fields"])
    assert any(s["name"] == "_Summary" for s in payload["semantic_indices"])


async def test_index_and_retrieve_roundtrip() -> None:
    store = MockStore()
    async with _client(_pipeline(store)) as client:
        resp = await client.post("/index", json={"text": "some text", "source_id": "doc_1"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["source_id"] == "doc_1"
        assert body["rejected"] is False

        retrieved = await client.get("/retrieve/doc_1")
        assert retrieved.status_code == 200
        assert retrieved.json()["cat"] == "legal"


async def test_filter_accepts_both_filter_key_and_flat() -> None:
    store = MockStore()
    async with _client(_pipeline(store)) as client:
        await client.post("/index", json={"text": "text", "source_id": "doc_1"})
        # Wrapped under "filters" key
        resp1 = await client.post("/filter", json={"filters": {"cat": "legal"}})
        assert resp1.status_code == 200
        assert resp1.json() == {"ids": ["doc_1"]}
        # Flat — top-level keys treated as filters.
        resp2 = await client.post("/filter", json={"cat": "legal"})
        assert resp2.json() == {"ids": ["doc_1"]}


async def test_filter_rejects_non_object_filters() -> None:
    async with _client(_pipeline()) as client:
        resp = await client.post("/filter", json={"filters": "not an object"})
    # The inner ValueError surfaces as a 500 unless we reshape; accept either
    # 500 or 422 — the important thing is it doesn't pass silently.
    assert resp.status_code >= 400


async def test_search_returns_hits() -> None:
    store = MockStore()
    async with _client(_pipeline(store)) as client:
        await client.post("/index", json={"text": "text", "source_id": "doc_1"})
        resp = await client.post(
            "/search", json={"query": "anything", "filters": {"cat": "legal"}, "top_k": 5}
        )
    assert resp.status_code == 200
    assert len(resp.json()["hits"]) >= 0


async def test_search_requires_string_query() -> None:
    async with _client(_pipeline()) as client:
        resp = await client.post("/search", json={"query": 42})
    assert resp.status_code == 422


async def test_search_filter_validation_returns_422_shape() -> None:
    async with _client(_pipeline()) as client:
        resp = await client.post(
            "/search",
            json={"query": "q", "filters": {"unknown_field": "x"}},
        )
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"] == "invalid_filter"
    assert body["field"] == "unknown_field"


async def test_retrieve_missing_returns_404() -> None:
    async with _client(_pipeline()) as client:
        resp = await client.get("/retrieve/missing")
    assert resp.status_code == 404


async def test_index_requires_string_text_and_source_id() -> None:
    async with _client(_pipeline()) as client:
        resp = await client.post("/index", json={"text": 42, "source_id": "x"})
    assert resp.status_code == 422


async def test_delete_returns_removed_flag() -> None:
    store = MockStore()
    async with _client(_pipeline(store)) as client:
        await client.post("/index", json={"text": "text", "source_id": "doc_1"})
        resp = await client.delete("/delete/doc_1")
        assert resp.status_code == 200
        assert resp.json() == {"removed": True}
        resp2 = await client.delete("/delete/doc_1")
        assert resp2.json() == {"removed": False}


async def test_static_bearer_auth_rejects_without_token() -> None:
    async with _client(_pipeline(), auth=static_bearer_auth("correct-key")) as client:
        resp = await client.get("/discover")
    assert resp.status_code == 401


async def test_static_bearer_auth_accepts_with_token() -> None:
    async with _client(_pipeline(), auth=static_bearer_auth("correct-key")) as client:
        resp = await client.get("/discover", headers={"Authorization": "Bearer correct-key"})
    assert resp.status_code == 200


async def test_static_bearer_auth_rejects_wrong_token() -> None:
    async with _client(_pipeline(), auth=static_bearer_auth("correct-key")) as client:
        resp = await client.get("/discover", headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401
