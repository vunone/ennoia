"""Pipeline edge cases — search score dedup, HybridStore path, empty semantic skip."""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
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


class _OtherSummary(BaseSemantic):
    """Summarise other angle."""


class _LLM(LLMAdapter):
    def __init__(self, json_map: dict[str, dict[str, Any]], text_map: dict[str, str]) -> None:
        self._json = json_map
        self._text = text_map

    async def complete_json(self, prompt: str) -> dict[str, Any]:
        for key, value in self._json.items():
            if key in prompt:
                return value
        raise AssertionError(f"Unexpected prompt: {prompt[:80]}")

    async def complete_text(self, prompt: str) -> str:
        for key, value in self._text.items():
            if key in prompt:
                return value
        raise AssertionError(f"Unexpected prompt: {prompt[:80]}")


class _Embedding(EmbeddingAdapter):
    async def embed(self, text: str) -> list[float]:
        # Deterministic distinct vectors per text so the two semantic
        # indices hash to different vectors yet share a source.
        if "other" in text.lower():
            return [0.1, 1.0]
        return [1.0, 0.0]


def _store() -> Store:
    return Store(vector=InMemoryVectorStore(), structured=InMemoryStructuredStore())


# ---------------------------------------------------------------------------
# Score dedup — same source_id present twice in raw_hits
# ---------------------------------------------------------------------------


def test_search_keeps_best_score_per_source_id() -> None:
    store = _store()
    # Two vectors pointing at the same source_id with different scores.
    asyncio.run(
        store.vector.upsert(
            "doc_1:Summary",
            [1.0, 0.0],
            {"source_id": "doc_1", "index": "Summary", "text": "first"},
        )
    )
    asyncio.run(
        store.vector.upsert(
            "doc_1:OtherSummary",
            [0.9, 0.1],
            {"source_id": "doc_1", "index": "OtherSummary", "text": "second"},
        )
    )
    asyncio.run(store.structured.upsert("doc_1", {"cat": "legal"}))

    pipeline = Pipeline(
        schemas=[_Doc, _Summary, _OtherSummary],
        store=store,
        llm=_LLM({}, {}),
        embedding=_Embedding(),
    )

    hits = pipeline.search(query="anything")
    # Dedup to one hit — higher score wins.
    assert len(hits) == 1
    assert hits.hits[0].source_id == "doc_1"
    assert hits.hits[0].score == pytest.approx(1.0)


def test_search_updates_score_when_second_hit_is_higher() -> None:
    # cosine_search sorts descending, so raw_hits arrive with the best score
    # first — the score-update branch is defensive for backends that don't
    # guarantee ordering. Use a fake HybridStore to deliver unsorted hits.
    class _UnsortedHybrid(HybridStore):
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
            # Same source, lower score first — forces the score-update branch.
            return [
                (
                    "doc_1:A",
                    0.3,
                    {"source_id": "doc_1", "index": "A", "text": "low"},
                ),
                (
                    "doc_1:B",
                    0.9,
                    {"source_id": "doc_1", "index": "B", "text": "high"},
                ),
            ]

        async def get(self, source_id: str) -> dict[str, Any] | None:
            return None

    pipeline = Pipeline(
        schemas=[_Doc, _Summary],
        store=_UnsortedHybrid(),
        llm=_LLM({}, {}),
        embedding=_Embedding(),
    )
    hits = pipeline.search(query="q")
    assert len(hits) == 1
    assert hits.hits[0].score == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Empty semantic answer — skip the embed + upsert
# ---------------------------------------------------------------------------


def test_empty_semantic_answer_skips_vector_upsert() -> None:
    store = _store()
    pipeline = Pipeline(
        schemas=[_Doc, _Summary],
        store=store,
        llm=_LLM(
            json_map={"Extract doc metadata.": {"cat": "legal", "_confidence": 0.9}},
            text_map={"Summarise doc.": ""},
        ),
        embedding=_Embedding(),
    )

    result = pipeline.index(text="body", source_id="doc_1")
    assert result.semantic == {"_Summary": ""}
    # Structured record persisted, but no vector since the answer was empty.
    assert asyncio.run(store.structured.get("doc_1")) == {"cat": "legal"}
    assert asyncio.run(store.vector.search([1.0, 0.0], top_k=5)) == []


# ---------------------------------------------------------------------------
# HybridStore path — asearch goes through plan_search's hybrid branch
# ---------------------------------------------------------------------------


class _FakeHybridStore(HybridStore):
    def __init__(self) -> None:
        self._hits: list[tuple[str, float, dict[str, Any]]] = []
        self.upserts: list[tuple[str, dict[str, Any], dict[str, list[float]]]] = []

    def set_hits(self, hits: list[tuple[str, float, dict[str, Any]]]) -> None:
        self._hits = hits

    async def upsert(
        self,
        source_id: str,
        data: dict[str, Any],
        vectors: dict[str, list[float]],
    ) -> None:
        self.upserts.append((source_id, dict(data), {k: list(v) for k, v in vectors.items()}))

    async def hybrid_search(
        self,
        filters: dict[str, Any],
        query_vector: list[float],
        top_k: int,
        *,
        index: str | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        return self._hits

    async def get(self, source_id: str) -> dict[str, Any] | None:
        return None


def test_search_against_hybrid_store_skips_structured_get() -> None:
    # The structured-store lookup is a composite-store concern; a HybridStore
    # returns fully-formed rows from ``hybrid_search`` directly.
    hybrid = _FakeHybridStore()
    hybrid.set_hits(
        [
            (
                "doc_1:Summary",
                0.95,
                {"source_id": "doc_1", "index": "Summary", "text": "hello"},
            )
        ]
    )
    pipeline = Pipeline(
        schemas=[_Doc, _Summary],
        store=hybrid,
        llm=_LLM({}, {}),
        embedding=_Embedding(),
    )
    hits = pipeline.search(query="q")
    assert len(hits) == 1
    assert hits.hits[0].source_id == "doc_1"
    # structural_record is the empty dict initialised at line 142 — no get() call
    # happened because the structured store branch was skipped.
    assert hits.hits[0].structural == {}
    assert hits.hits[0].semantic == {"Summary": "hello"}


def test_persist_to_hybrid_store_calls_upsert_once() -> None:
    # Stage 3: the HybridStore persistence path flattens structural fields
    # and embeds semantic answers into a single ``upsert(source_id, data, vectors)``
    # call, with ``vectors`` keyed by semantic index name.
    store = _FakeHybridStore()
    pipeline = Pipeline(
        schemas=[_Doc, _Summary],
        store=store,
        llm=_LLM(
            json_map={"Extract doc metadata.": {"cat": "legal", "_confidence": 0.9}},
            text_map={"Summarise doc.": "a summary"},
        ),
        embedding=_Embedding(),
    )
    pipeline.index(text="body", source_id="doc_1")
    assert len(store.upserts) == 1
    source_id, data, vectors = store.upserts[0]
    assert source_id == "doc_1"
    assert data["cat"] == "legal"
    # Semantic index name keys the vector dict — matches HybridStore.upsert ABC.
    assert list(vectors.keys()) == ["_Summary"]
    assert len(vectors["_Summary"]) > 0


def test_persist_to_hybrid_store_skips_vectors_when_semantic_empty() -> None:
    store = _FakeHybridStore()
    pipeline = Pipeline(
        schemas=[_Doc, _Summary],
        store=store,
        llm=_LLM(
            json_map={"Extract doc metadata.": {"cat": "legal", "_confidence": 0.9}},
            text_map={"Summarise doc.": ""},  # empty semantic answer
        ),
        embedding=_Embedding(),
    )
    pipeline.index(text="body", source_id="doc_1")
    assert len(store.upserts) == 1
    _, _, vectors = store.upserts[0]
    assert vectors == {}


# Ensure numpy is imported — the embedding path relies on it via cosine_search.
assert np.__version__
