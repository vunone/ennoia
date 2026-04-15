"""Public Pipeline — orchestrates the extraction DAG and retrieval."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from ennoia.events import Emitter, IndexEvent, NullEmitter, SearchEvent
from ennoia.index.dag import validate_schemas
from ennoia.index.exceptions import RejectException
from ennoia.index.executor import execute_layers
from ennoia.index.extractor import CONFIDENCE_KEY
from ennoia.index.query import plan_search
from ennoia.index.result import IndexResult, SearchHit, SearchResult
from ennoia.index.validation import validate_filters
from ennoia.schema.base import BaseSemantic, BaseStructure
from ennoia.utils.ids import extract_source_id, make_semantic_vector_id

if TYPE_CHECKING:
    from ennoia.adapters.embedding.protocols import EmbeddingAdapter
    from ennoia.adapters.llm.protocols import LLMAdapter
    from ennoia.store.base import HybridStore
    from ennoia.store.composite import Store

__all__ = ["Pipeline"]


class Pipeline:
    def __init__(
        self,
        schemas: list[type[BaseStructure] | type[BaseSemantic]] | None = None,
        semantics: list[type[BaseSemantic]] | None = None,
        *,
        store: Store | HybridStore,
        llm: LLMAdapter,
        embedding: EmbeddingAdapter,
        events: Emitter | None = None,
    ) -> None:
        schemas = list(schemas or [])
        semantics = list(semantics or [])

        # ``schemas=`` accepts both structural and semantic classes so users can
        # write a single combined list per ``docs/quickstart.md``. Split them here.
        structural: list[type[BaseStructure]] = []
        extracted_semantics: list[type[BaseSemantic]] = list(semantics)
        for cls in schemas:
            if issubclass(cls, BaseSemantic):
                extracted_semantics.append(cls)
            else:
                structural.append(cls)

        validate_schemas([*structural, *extracted_semantics])
        self._structural = structural
        self._semantics = extracted_semantics
        self.store = store
        self.llm = llm
        self.embedding = embedding
        self.events = events or NullEmitter()

    def index(self, text: str, source_id: str) -> IndexResult:
        return asyncio.run(self.aindex(text=text, source_id=source_id))

    def search(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
    ) -> SearchResult:
        return asyncio.run(self.asearch(query=query, filters=filters, top_k=top_k))

    async def aindex(self, text: str, source_id: str) -> IndexResult:
        start = time.perf_counter()
        try:
            batch = await execute_layers(
                seed_structural=list(self._structural),
                seed_semantic=list(self._semantics),
                text=text,
                llm=self.llm,
            )
        except RejectException:
            rejected = IndexResult(source_id=source_id, rejected=True)
            self.events.emit(
                IndexEvent(
                    source_id=source_id,
                    schemas_extracted=[],
                    rejected=True,
                    duration_ms=(time.perf_counter() - start) * 1000,
                )
            )
            return rejected

        result = IndexResult(
            source_id=source_id,
            structural=batch.structural,
            semantic=batch.semantic,
            confidences=batch.confidences,
        )
        self._persist(result)
        self.events.emit(
            IndexEvent(
                source_id=source_id,
                schemas_extracted=[*result.structural.keys(), *result.semantic.keys()],
                rejected=False,
                duration_ms=(time.perf_counter() - start) * 1000,
            )
        )
        return result

    async def asearch(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
    ) -> SearchResult:
        start = time.perf_counter()
        validate_filters(filters, self._structural)
        query_vector = self.embedding.embed_query(query)
        raw_hits = plan_search(self.store, filters, query_vector, top_k)

        from ennoia.store.composite import Store

        structured_store = self.store.structured if isinstance(self.store, Store) else None

        grouped: dict[str, SearchHit] = {}
        for vector_id, score, metadata in raw_hits:
            source_id = extract_source_id(metadata, vector_id)
            if source_id in grouped:
                # Keep the best score per source document.
                if score > grouped[source_id].score:
                    grouped[source_id].score = score
                continue

            structural_record: dict[str, Any] = {}
            if structured_store is not None:
                structural_record = structured_store.get(source_id) or {}

            semantic_text = metadata.get("text", "")
            index_name = metadata.get("index", "")
            grouped[source_id] = SearchHit(
                source_id=source_id,
                score=score,
                structural=structural_record,
                semantic={index_name: semantic_text} if index_name else {},
            )

        hits_sorted = sorted(grouped.values(), key=lambda h: h.score, reverse=True)[:top_k]
        self.events.emit(
            SearchEvent(
                query=query,
                filters=dict(filters or {}),
                hit_count=len(hits_sorted),
                duration_ms=(time.perf_counter() - start) * 1000,
            )
        )
        return SearchResult(hits=hits_sorted)

    def _persist(self, result: IndexResult) -> None:
        from ennoia.store.composite import Store

        if not isinstance(self.store, Store):
            # HybridStore persistence is Stage 3 territory; bail gracefully.
            raise NotImplementedError(
                "Persistence for HybridStore implementations arrives in a later stage."
            )

        flat: dict[str, Any] = {}
        for instance in result.structural.values():
            # ``_confidence`` rides on the instance via extra='allow'; strip before persist.
            dumped = instance.model_dump(mode="json")
            dumped.pop(CONFIDENCE_KEY, None)
            flat.update(dumped)
        self.store.structured.upsert(result.source_id, flat)

        for name, semantic_text in result.semantic.items():
            if not semantic_text:
                # Empty answer is a signal from the LLM, not an invitation to
                # embed the full document; skip the upsert entirely.
                continue
            vector = self.embedding.embed_document(semantic_text)
            self.store.vector.upsert(
                vector_id=make_semantic_vector_id(result.source_id, name),
                vector=vector,
                metadata={
                    "source_id": result.source_id,
                    "index": name,
                    "text": semantic_text,
                },
            )
