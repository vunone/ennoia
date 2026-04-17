"""Pipeline Stage 3 API: afilter, aretrieve, adelete and their sync mirrors."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ennoia import BaseSemantic, BaseStructure, Pipeline, Store
from ennoia.adapters.embedding import EmbeddingAdapter
from ennoia.adapters.llm import LLMAdapter
from ennoia.store import HybridStore, InMemoryStructuredStore, InMemoryVectorStore


class _Doc(BaseStructure):
    """Extract doc metadata."""

    cat: str


class _Summary(BaseSemantic):
    """Summarise doc."""


class _LLM(LLMAdapter):
    async def complete_json(self, prompt: str) -> dict[str, Any]:
        raise AssertionError("unused in filter/retrieve/delete tests")

    async def complete_text(self, prompt: str) -> str:
        raise AssertionError("unused in filter/retrieve/delete tests")


class _Embedding(EmbeddingAdapter):
    async def embed(self, text: str) -> list[float]:
        return [1.0, 0.0]


def _pipeline(store: Store | HybridStore) -> Pipeline:
    return Pipeline(schemas=[_Doc, _Summary], store=store, llm=_LLM(), embedding=_Embedding())


# ---------------------------------------------------------------------------
# Composite Store path
# ---------------------------------------------------------------------------


def test_afilter_returns_candidate_ids_from_structured_store() -> None:
    store = Store(vector=InMemoryVectorStore(), structured=InMemoryStructuredStore())
    asyncio.run(store.structured.upsert("a", {"cat": "legal"}))
    asyncio.run(store.structured.upsert("b", {"cat": "medical"}))
    pipeline = _pipeline(store)

    ids = pipeline.filter({"cat": "legal"})
    assert ids == ["a"]


def test_afilter_with_none_returns_all_ids() -> None:
    store = Store(vector=InMemoryVectorStore(), structured=InMemoryStructuredStore())
    asyncio.run(store.structured.upsert("a", {"cat": "legal"}))
    asyncio.run(store.structured.upsert("b", {"cat": "medical"}))
    pipeline = _pipeline(store)

    ids = sorted(pipeline.filter())
    assert ids == ["a", "b"]


def test_aretrieve_returns_flat_record_or_none() -> None:
    store = Store(vector=InMemoryVectorStore(), structured=InMemoryStructuredStore())
    asyncio.run(store.structured.upsert("doc_1", {"cat": "legal"}))
    pipeline = _pipeline(store)

    assert pipeline.retrieve("doc_1") == {"cat": "legal"}
    assert pipeline.retrieve("missing") is None


def test_adelete_fans_out_to_structured_and_vector() -> None:
    store = Store(vector=InMemoryVectorStore(), structured=InMemoryStructuredStore())
    asyncio.run(store.structured.upsert("doc_1", {"cat": "legal"}))
    asyncio.run(
        store.vector.upsert("doc_1:Summary", [1.0, 0.0], {"source_id": "doc_1", "index": "Summary"})
    )
    asyncio.run(
        store.vector.upsert("doc_1:Other", [0.0, 1.0], {"source_id": "doc_1", "index": "Other"})
    )
    pipeline = _pipeline(store)

    assert pipeline.delete("doc_1") is True
    assert asyncio.run(store.structured.get("doc_1")) is None
    assert asyncio.run(store.vector.search([1.0, 0.0], top_k=5)) == []


def test_adelete_returns_false_when_source_absent() -> None:
    store = Store(vector=InMemoryVectorStore(), structured=InMemoryStructuredStore())
    pipeline = _pipeline(store)
    assert pipeline.delete("missing") is False


def test_adelete_returns_true_when_only_vector_side_matched() -> None:
    store = Store(vector=InMemoryVectorStore(), structured=InMemoryStructuredStore())
    asyncio.run(store.vector.upsert("doc_1:S", [1.0, 0.0], {"source_id": "doc_1", "index": "S"}))
    pipeline = _pipeline(store)
    assert pipeline.delete("doc_1") is True


# ---------------------------------------------------------------------------
# HybridStore path
# ---------------------------------------------------------------------------


class _HybridSpy(HybridStore):
    def __init__(self) -> None:
        self.filter_calls: list[dict[str, Any]] = []
        self.get_calls: list[str] = []
        self.delete_calls: list[str] = []
        self._records: dict[str, dict[str, Any]] = {"doc_1": {"cat": "legal"}}

    async def upsert(
        self,
        source_id: str,
        data: dict[str, Any],
        entries: list[Any],
    ) -> None:
        raise AssertionError("unused")

    async def hybrid_search(
        self,
        filters: dict[str, Any],
        query_vector: list[float],
        top_k: int,
        *,
        index: str | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        raise AssertionError("unused")

    async def get(self, source_id: str) -> dict[str, Any] | None:
        self.get_calls.append(source_id)
        return self._records.get(source_id)

    async def filter(self, filters: dict[str, Any]) -> list[str]:
        self.filter_calls.append(dict(filters))
        return ["doc_1"] if filters.get("cat") == "legal" else []

    async def delete(self, source_id: str) -> bool:
        self.delete_calls.append(source_id)
        return self._records.pop(source_id, None) is not None


def test_afilter_delegates_to_hybrid_filter() -> None:
    hybrid = _HybridSpy()
    pipeline = _pipeline(hybrid)
    assert pipeline.filter({"cat": "legal"}) == ["doc_1"]
    assert hybrid.filter_calls == [{"cat": "legal"}]


def test_aretrieve_delegates_to_hybrid_get() -> None:
    hybrid = _HybridSpy()
    pipeline = _pipeline(hybrid)
    assert pipeline.retrieve("doc_1") == {"cat": "legal"}
    assert hybrid.get_calls == ["doc_1"]


def test_adelete_delegates_to_hybrid_delete() -> None:
    hybrid = _HybridSpy()
    pipeline = _pipeline(hybrid)
    assert pipeline.delete("doc_1") is True
    assert pipeline.delete("doc_1") is False  # second call: already gone
    assert hybrid.delete_calls == ["doc_1", "doc_1"]


# ---------------------------------------------------------------------------
# Filter validation
# ---------------------------------------------------------------------------


def test_afilter_validates_filters_against_superschema() -> None:
    from ennoia.index.exceptions import FilterValidationError

    store = Store(vector=InMemoryVectorStore(), structured=InMemoryStructuredStore())
    pipeline = _pipeline(store)
    with pytest.raises(FilterValidationError):
        pipeline.filter({"unknown_field": "x"})
