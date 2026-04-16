"""Pipeline.asearch two-phase flow: ``filter_ids=`` and ``index=`` arguments."""

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


class _Holding(BaseSemantic):
    """Holding text."""


class _Facts(BaseSemantic):
    """Facts text."""


class _LLM(LLMAdapter):
    async def complete_json(self, prompt: str) -> dict[str, Any]:
        raise AssertionError("unused")

    async def complete_text(self, prompt: str) -> str:
        raise AssertionError("unused")


class _Embedding(EmbeddingAdapter):
    async def embed(self, text: str) -> list[float]:
        return [1.0, 0.0]


def _seeded_store() -> Store:
    store = Store(vector=InMemoryVectorStore(), structured=InMemoryStructuredStore())
    asyncio.run(store.structured.upsert("doc_1", {"cat": "legal"}))
    asyncio.run(store.structured.upsert("doc_2", {"cat": "legal"}))
    asyncio.run(
        store.vector.upsert(
            "doc_1:_Holding",
            [1.0, 0.0],
            {"source_id": "doc_1", "index": "_Holding", "text": "h1"},
        )
    )
    asyncio.run(
        store.vector.upsert(
            "doc_1:_Facts",
            [0.95, 0.05],
            {"source_id": "doc_1", "index": "_Facts", "text": "f1"},
        )
    )
    asyncio.run(
        store.vector.upsert(
            "doc_2:_Holding",
            [0.9, 0.1],
            {"source_id": "doc_2", "index": "_Holding", "text": "h2"},
        )
    )
    return store


def test_search_with_filter_ids_skips_structured_filter() -> None:
    store = _seeded_store()
    pipeline = Pipeline(
        schemas=[_Doc, _Holding, _Facts], store=store, llm=_LLM(), embedding=_Embedding()
    )
    # Even though ``cat=medical`` would match no records, we pass pre-resolved ids.
    hits = pipeline.search(query="q", filter_ids=["doc_2"])
    assert [h.source_id for h in hits.hits] == ["doc_2"]


def test_search_with_index_filters_to_single_semantic_index() -> None:
    store = _seeded_store()
    pipeline = Pipeline(
        schemas=[_Doc, _Holding, _Facts], store=store, llm=_LLM(), embedding=_Embedding()
    )
    hits = pipeline.search(query="q", index="_Holding", top_k=5)
    # Three stored vectors, two of which are on ``_Holding`` (doc_1 + doc_2).
    sources = {h.source_id for h in hits.hits}
    assert sources == {"doc_1", "doc_2"}
    for hit in hits.hits:
        assert "_Holding" in hit.semantic


def test_search_rejects_both_filters_and_filter_ids() -> None:
    store = _seeded_store()
    pipeline = Pipeline(
        schemas=[_Doc, _Holding, _Facts], store=store, llm=_LLM(), embedding=_Embedding()
    )
    with pytest.raises(ValueError, match="Pass either filters= or filter_ids="):
        pipeline.search(query="q", filters={"cat": "legal"}, filter_ids=["doc_1"])


def test_search_empty_filter_ids_returns_no_hits() -> None:
    store = _seeded_store()
    pipeline = Pipeline(
        schemas=[_Doc, _Holding, _Facts], store=store, llm=_LLM(), embedding=_Embedding()
    )
    assert pipeline.search(query="q", filter_ids=[]).hits == []


# ---------------------------------------------------------------------------
# HybridStore path — filter_ids post-filters the native hit set
# ---------------------------------------------------------------------------


class _HybridHits(HybridStore):
    def __init__(self, hits: list[tuple[str, float, dict[str, Any]]]) -> None:
        self._hits = hits
        self.last_top_k = 0
        self.last_index: str | None = None

    async def upsert(
        self,
        source_id: str,
        data: dict[str, Any],
        vectors: dict[str, list[float]],
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
        self.last_top_k = top_k
        self.last_index = index
        return self._hits

    async def get(self, source_id: str) -> dict[str, Any] | None:
        return None


def test_hybrid_search_with_filter_ids_post_filters_hits() -> None:
    hybrid = _HybridHits(
        [
            ("doc_1:S", 0.9, {"source_id": "doc_1", "index": "S", "text": "one"}),
            ("doc_2:S", 0.8, {"source_id": "doc_2", "index": "S", "text": "two"}),
            ("doc_3:S", 0.7, {"source_id": "doc_3", "index": "S", "text": "three"}),
        ]
    )
    pipeline = Pipeline(schemas=[_Doc, _Holding], store=hybrid, llm=_LLM(), embedding=_Embedding())
    hits = pipeline.search(query="q", filter_ids=["doc_2"], top_k=5)
    assert [h.source_id for h in hits.hits] == ["doc_2"]
    # Query expanded top_k 4x to improve post-filter yield.
    assert hybrid.last_top_k == 5 * 4


def test_hybrid_search_empty_filter_ids_returns_empty() -> None:
    hybrid = _HybridHits([])
    pipeline = Pipeline(schemas=[_Doc, _Holding], store=hybrid, llm=_LLM(), embedding=_Embedding())
    assert pipeline.search(query="q", filter_ids=[]).hits == []


def test_hybrid_search_passes_index_down() -> None:
    hybrid = _HybridHits([])
    pipeline = Pipeline(schemas=[_Doc, _Holding], store=hybrid, llm=_LLM(), embedding=_Embedding())
    pipeline.search(query="q", index="_Holding")
    assert hybrid.last_index == "_Holding"
