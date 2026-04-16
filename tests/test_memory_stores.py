"""In-memory store behavior (no LLM / embedding required)."""

from __future__ import annotations

from datetime import date

import pytest

np = pytest.importorskip("numpy")

from ennoia.store import InMemoryStructuredStore, InMemoryVectorStore  # noqa: E402


async def test_structured_equality_filter():
    store = InMemoryStructuredStore()
    await store.upsert("a", {"category": "legal"})
    await store.upsert("b", {"category": "medical"})
    assert await store.filter({"category": "legal"}) == ["a"]


async def test_structured_in_operator():
    store = InMemoryStructuredStore()
    await store.upsert("a", {"category": "legal"})
    await store.upsert("b", {"category": "medical"})
    await store.upsert("c", {"category": "financial"})
    result = set(await store.filter({"category__in": ["legal", "medical"]}))
    assert result == {"a", "b"}


async def test_structured_date_comparison_coerces_strings():
    store = InMemoryStructuredStore()
    await store.upsert("old", {"doc_date": date(2019, 1, 1)})
    await store.upsert("new", {"doc_date": date(2024, 6, 1)})
    assert await store.filter({"doc_date__gte": "2020-01-01"}) == ["new"]


async def test_structured_empty_query_returns_all():
    store = InMemoryStructuredStore()
    await store.upsert("a", {"x": 1})
    await store.upsert("b", {"x": 2})
    assert set(await store.filter({})) == {"a", "b"}


async def test_structured_get_returns_copy():
    store = InMemoryStructuredStore()
    await store.upsert("a", {"x": 1})
    record = await store.get("a")
    assert record == {"x": 1}
    record["x"] = 999
    assert await store.get("a") == {"x": 1}


async def test_vector_search_returns_top_k_by_similarity():
    store = InMemoryVectorStore()
    await store.upsert("a", [1.0, 0.0], {"source_id": "a"})
    await store.upsert("b", [0.0, 1.0], {"source_id": "b"})
    await store.upsert("c", [0.9, 0.1], {"source_id": "c"})

    hits = await store.search(query_vector=[1.0, 0.0], top_k=2)
    ids = [h[0] for h in hits]
    assert ids[0] == "a"
    assert "c" in ids


async def test_vector_search_respects_restrict_to():
    store = InMemoryVectorStore()
    await store.upsert("a:1", [1.0, 0.0], {"source_id": "a"})
    await store.upsert("b:1", [1.0, 0.0], {"source_id": "b"})

    hits = await store.search(query_vector=[1.0, 0.0], top_k=10, restrict_to=["b"])
    assert [h[0] for h in hits] == ["b:1"]


async def test_vector_search_empty_store_returns_empty():
    store = InMemoryVectorStore()
    assert await store.search(query_vector=[1.0, 0.0], top_k=5) == []
