"""In-memory store behavior (no LLM / embedding required)."""

from __future__ import annotations

from datetime import date

import pytest

np = pytest.importorskip("numpy")

from ennoia.store import InMemoryStructuredStore, InMemoryVectorStore  # noqa: E402


def test_structured_equality_filter():
    store = InMemoryStructuredStore()
    store.upsert("a", {"category": "legal"})
    store.upsert("b", {"category": "medical"})
    assert store.filter({"category": "legal"}) == ["a"]


def test_structured_in_operator():
    store = InMemoryStructuredStore()
    store.upsert("a", {"category": "legal"})
    store.upsert("b", {"category": "medical"})
    store.upsert("c", {"category": "financial"})
    result = set(store.filter({"category__in": ["legal", "medical"]}))
    assert result == {"a", "b"}


def test_structured_date_comparison_coerces_strings():
    store = InMemoryStructuredStore()
    store.upsert("old", {"doc_date": date(2019, 1, 1)})
    store.upsert("new", {"doc_date": date(2024, 6, 1)})
    assert store.filter({"doc_date__gte": "2020-01-01"}) == ["new"]


def test_structured_empty_query_returns_all():
    store = InMemoryStructuredStore()
    store.upsert("a", {"x": 1})
    store.upsert("b", {"x": 2})
    assert set(store.filter({})) == {"a", "b"}


def test_structured_get_returns_copy():
    store = InMemoryStructuredStore()
    store.upsert("a", {"x": 1})
    record = store.get("a")
    assert record == {"x": 1}
    record["x"] = 999
    assert store.get("a") == {"x": 1}


def test_vector_search_returns_top_k_by_similarity():
    store = InMemoryVectorStore()
    store.upsert("a", [1.0, 0.0], {"source_id": "a"})
    store.upsert("b", [0.0, 1.0], {"source_id": "b"})
    store.upsert("c", [0.9, 0.1], {"source_id": "c"})

    hits = store.search(query_vector=[1.0, 0.0], top_k=2)
    ids = [h[0] for h in hits]
    assert ids[0] == "a"
    assert "c" in ids


def test_vector_search_respects_restrict_to():
    store = InMemoryVectorStore()
    store.upsert("a:1", [1.0, 0.0], {"source_id": "a"})
    store.upsert("b:1", [1.0, 0.0], {"source_id": "b"})

    hits = store.search(query_vector=[1.0, 0.0], top_k=10, restrict_to=["b"])
    assert [h[0] for h in hits] == ["b:1"]


def test_vector_search_empty_store_returns_empty():
    store = InMemoryVectorStore()
    assert store.search(query_vector=[1.0, 0.0], top_k=5) == []
