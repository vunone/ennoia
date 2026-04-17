"""End-to-end Pipeline tests for BaseCollection extraction.

Covers the full flow against both store flavours: composite (Structured +
Vector) and HybridStore (MockStore). Also exercises ``extend()`` emission from
a collection entity into a structural schema.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ennoia import BaseCollection, BaseStructure, Pipeline
from ennoia.store import InMemoryStructuredStore, InMemoryVectorStore, Store
from ennoia.testing import MockEmbeddingAdapter, MockLLMAdapter, MockStore


class Party(BaseCollection):
    """Extract every contract party."""

    name: str
    year: int

    def get_unique(self) -> str:
        return self.name

    def template(self) -> str:
        return f"{self.name} ({self.year})"


class PartyWithExtend(BaseCollection):
    """Extract every contract party; emit Spotlight on FAANG."""

    name: str
    year: int

    def get_unique(self) -> str:
        return self.name

    def template(self) -> str:
        return f"{self.name} ({self.year})"

    class Schema:
        extensions: ClassVar[list[type]] = []

    def extend(self) -> list[type]:
        return list(PartyWithExtend.Schema.extensions)


class Spotlight(BaseStructure):
    """Mark documents that mention a FAANG party."""

    has_faang: bool


# Wire Spotlight into the extensions post-hoc so the class is defined.
PartyWithExtend.Schema.extensions = [Spotlight]


def _composite_store() -> Store:
    return Store(structured=InMemoryStructuredStore(), vector=InMemoryVectorStore())


def _llm_two_parties_then_done() -> MockLLMAdapter:
    return MockLLMAdapter(
        json_responses=[
            {
                "entities_list": [
                    {"name": "Acme", "year": 2024, "_confidence": 0.9},
                    {"name": "Beta", "year": 2023, "_confidence": 0.8},
                ],
                "is_done": True,
            }
        ]
    )


def test_collection_indexes_into_composite_store_with_one_vector_per_entity() -> None:
    store = _composite_store()
    pipeline = Pipeline(
        schemas=[Party],
        store=store,
        llm=_llm_two_parties_then_done(),
        embedding=MockEmbeddingAdapter(dim=8),
    )
    result = pipeline.index(text="body", source_id="doc_1")

    # IndexResult exposes the collection + per-item confidences.
    assert [p.name for p in result.collections["Party"]] == ["Acme", "Beta"]
    assert result.collection_confidences["Party"] == [0.9, 0.8]
    # Flat confidences carries the mean so existing consumers keep reading one value.
    assert result.confidences["Party"] == (0.9 + 0.8) / 2

    # Two vectors landed in the vector store, both under Party's schema name.
    import asyncio

    hits = asyncio.run(store.vector.search(query_vector=[1.0] + [0.0] * 7, top_k=10))
    index_names = sorted({meta["index"] for _, _, meta in hits})
    assert index_names == ["Party"]
    assert len(hits) == 2


def test_collection_entries_searchable_by_index_through_pipeline() -> None:
    store = _composite_store()
    pipeline = Pipeline(
        schemas=[Party],
        store=store,
        llm=_llm_two_parties_then_done(),
        embedding=MockEmbeddingAdapter(dim=8),
    )
    pipeline.index(text="body", source_id="doc_1")

    results = pipeline.search(query="acme", top_k=5, index="Party")
    assert len(results) == 1
    hit = results.hits[0]
    assert hit.source_id == "doc_1"
    # Hit text is the entity's template() output, not the raw answer.
    assert hit.semantic["Party"] in {"Acme (2024)", "Beta (2023)"}


def test_collection_indexes_into_hybrid_store() -> None:
    store = MockStore()
    pipeline = Pipeline(
        schemas=[Party],
        store=store,
        llm=_llm_two_parties_then_done(),
        embedding=MockEmbeddingAdapter(dim=8),
    )
    pipeline.index(text="body", source_id="doc_1")

    # Two rows landed under the same source_id.
    rows = [row for row in store._rows.values() if row.source_id == "doc_1"]
    assert len(rows) == 2
    assert all(row.index_name == "Party" for row in rows)
    # Structural data is copied onto every row.
    assert all(row.data == {} for row in rows)  # no BaseStructure here → empty superschema

    # Search distincts on source_id.
    results = pipeline.search(query="acme", top_k=5, index="Party")
    assert len(results) == 1
    assert results.hits[0].source_id == "doc_1"


def test_reindex_drops_stale_collection_entries_on_composite_store() -> None:
    store = _composite_store()
    llm = MockLLMAdapter(
        json_responses=[
            {
                "entities_list": [
                    {"name": "Acme", "year": 2024, "_confidence": 0.9},
                    {"name": "Beta", "year": 2023, "_confidence": 0.8},
                ],
                "is_done": True,
            },
            {
                "entities_list": [
                    {"name": "Acme", "year": 2024, "_confidence": 0.9},
                ],
                "is_done": True,
            },
        ]
    )
    pipeline = Pipeline(
        schemas=[Party],
        store=store,
        llm=llm,
        embedding=MockEmbeddingAdapter(dim=8),
    )
    pipeline.index(text="body", source_id="doc_1")
    pipeline.index(text="body", source_id="doc_1")

    import asyncio

    hits = asyncio.run(store.vector.search(query_vector=[1.0] + [0.0] * 7, top_k=10))
    # Only one row remains after re-index — stale Beta is gone.
    assert len(hits) == 1
    assert hits[0][2]["text"] == "Acme (2024)"


def test_collection_extend_emits_structural_child() -> None:
    store = _composite_store()
    llm = MockLLMAdapter(
        json_responses=[
            # Collection: one Party entity that triggers Spotlight via extend().
            {
                "entities_list": [
                    {"name": "Meta", "year": 2024, "_confidence": 0.9},
                ],
                "is_done": True,
            },
            # Spotlight extraction triggered by Party.extend().
            {"has_faang": True, "_confidence": 0.95},
        ]
    )
    pipeline = Pipeline(
        schemas=[PartyWithExtend],
        store=store,
        llm=llm,
        embedding=MockEmbeddingAdapter(dim=8),
    )
    result = pipeline.index(text="body", source_id="doc_1")

    assert [p.name for p in result.collections["PartyWithExtend"]] == ["Meta"]
    assert result.structural["Spotlight"].has_faang is True


def test_collection_extend_can_emit_another_collection() -> None:
    # A collection entity's extend() emits a second collection, exercising the
    # BaseCollection → BaseCollection routing branch in the executor.
    class Nested(BaseCollection):
        """Nested collection."""

        note: str

        def get_unique(self) -> str:
            return self.note

        def template(self) -> str:
            return self.note

    class Root(BaseCollection):
        """Root collection that triggers Nested."""

        name: str

        def get_unique(self) -> str:
            return self.name

        def template(self) -> str:
            return self.name

        class Schema:
            extensions: ClassVar[list[type]] = [Nested]

        def extend(self) -> list[type]:
            return [Nested]

    llm = MockLLMAdapter(
        json_responses=[
            # Root collection
            {
                "entities_list": [{"name": "r", "_confidence": 0.9}],
                "is_done": True,
            },
            # Nested collection (triggered via Root.extend)
            {
                "entities_list": [{"note": "n", "_confidence": 0.9}],
                "is_done": True,
            },
        ]
    )
    store = _composite_store()
    pipeline = Pipeline(
        schemas=[Root],
        store=store,
        llm=llm,
        embedding=MockEmbeddingAdapter(dim=8),
    )
    result = pipeline.index(text="body", source_id="doc_1")
    assert [c.name for c in result.collections["Root"]] == ["r"]
    assert [c.note for c in result.collections["Nested"]] == ["n"]


def test_collection_deduplicates_when_emitted_twice_across_entities() -> None:
    # Two parent entities each emit the same child collection class; executor
    # must run it once. Exercises ``seen_collection`` short-circuit at the top
    # of ``_run_collection_layer``.
    class Child(BaseCollection):
        """Child collection."""

        tag: str

        def get_unique(self) -> str:
            return self.tag

        def template(self) -> str:
            return self.tag

    class Parent(BaseCollection):
        """Parent that emits Child per entity."""

        label: str

        def get_unique(self) -> str:
            return self.label

        def template(self) -> str:
            return self.label

        class Schema:
            extensions: ClassVar[list[type]] = [Child]

        def extend(self) -> list[type]:
            return [Child]

    llm = MockLLMAdapter(
        json_responses=[
            {
                "entities_list": [
                    {"label": "a", "_confidence": 0.9},
                    {"label": "b", "_confidence": 0.9},
                ],
                "is_done": True,
            },
            # Child collection runs only once.
            {"entities_list": [{"tag": "t", "_confidence": 0.9}], "is_done": True},
        ]
    )
    store = _composite_store()
    pipeline = Pipeline(
        schemas=[Parent],
        store=store,
        llm=llm,
        embedding=MockEmbeddingAdapter(dim=8),
    )
    result = pipeline.index(text="body", source_id="doc_1")
    assert [c.tag for c in result.collections["Child"]] == ["t"]
    # Only 2 LLM calls total: one for Parent, one for Child (dedup).
    assert len(llm.json_calls) == 2


def test_collection_entity_with_empty_template_is_not_persisted() -> None:
    # An entity whose template() returns "" must not produce a vector row —
    # treating it the same as an empty BaseSemantic answer.
    class Silent(BaseCollection):
        """Silent collection."""

        label: str

        def get_unique(self) -> str:
            return self.label

        def template(self) -> str:
            # Deliberately empty — simulates an entity with nothing to embed.
            return ""

    llm = MockLLMAdapter(
        json_responses=[{"entities_list": [{"label": "a", "_confidence": 0.9}], "is_done": True}]
    )
    store = _composite_store()
    pipeline = Pipeline(
        schemas=[Silent],
        store=store,
        llm=llm,
        embedding=MockEmbeddingAdapter(dim=8),
    )
    result = pipeline.index(text="body", source_id="doc_1")
    assert [s.label for s in result.collections["Silent"]] == ["a"]
    # No vectors landed.
    import asyncio

    hits = asyncio.run(store.vector.search(query_vector=[1.0] + [0.0] * 7, top_k=10))
    assert hits == []


def test_collection_index_result_summary_includes_entities() -> None:
    store = _composite_store()
    pipeline = Pipeline(
        schemas=[Party],
        store=store,
        llm=_llm_two_parties_then_done(),
        embedding=MockEmbeddingAdapter(dim=8),
    )
    result = pipeline.index(text="body", source_id="doc_1")
    summary: dict[str, Any] = result.summary()
    assert summary["collections"]["Party"] == [
        {"name": "Acme", "year": 2024},
        {"name": "Beta", "year": 2023},
    ]
    assert summary["collection_confidences"]["Party"] == [0.9, 0.8]
