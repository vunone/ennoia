"""Public Pipeline — orchestrates the extraction DAG and retrieval."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from ennoia.events import Emitter, IndexEvent, NullEmitter, SearchEvent
from ennoia.index.dag import validate_schemas
from ennoia.index.exceptions import RejectException
from ennoia.index.executor import execute_layers
from ennoia.index.extractor import CONFIDENCE_KEY
from ennoia.index.query import plan_search
from ennoia.index.result import IndexResult, SearchHit, SearchResult
from ennoia.index.validation import validate_filters
from ennoia.schema.base import (
    BaseCollection,
    BaseSemantic,
    BaseStructure,
    get_schema_namespace,
)
from ennoia.schema.manifest import build_manifest
from ennoia.schema.merging import Superschema, build_superschema
from ennoia.store.base import HybridStore, VectorEntry
from ennoia.utils.ids import extract_source_id, make_semantic_vector_id

if TYPE_CHECKING:
    from ennoia.adapters.embedding.base import EmbeddingAdapter
    from ennoia.adapters.llm.base import LLMAdapter
    from ennoia.store.composite import Store

__all__ = ["Pipeline"]


class Pipeline:
    def __init__(
        self,
        schemas: (
            Sequence[type[BaseStructure] | type[BaseSemantic] | type[BaseCollection]] | None
        ) = None,
        semantics: Sequence[type[BaseSemantic]] | None = None,
        *,
        store: Store | HybridStore,
        llm: LLMAdapter,
        embedding: EmbeddingAdapter,
        events: Emitter | None = None,
        concurrency: int | None = None,
    ) -> None:
        schemas = list(schemas or [])
        semantics = list(semantics or [])

        # ``schemas=`` accepts structural, semantic, and collection classes so
        # users can write a single combined list per ``docs/quickstart.md``. Split them here.
        structural: list[type[BaseStructure]] = []
        extracted_semantics: list[type[BaseSemantic]] = list(semantics)
        collections: list[type[BaseCollection]] = []
        for cls in schemas:
            if issubclass(cls, BaseCollection):
                collections.append(cls)
            elif issubclass(cls, BaseSemantic):
                extracted_semantics.append(cls)
            else:
                structural.append(cls)

        validate_schemas([*structural, *extracted_semantics, *collections])
        self._structural = structural
        self._semantics = extracted_semantics
        self._collections = collections
        # Resolve the emission manifest (transitive closure of Schema.extensions)
        # and collapse it into the unified superschema once, at init. Both are
        # cached on the Pipeline so _persist, asearch, and discovery all read
        # the same source of truth.
        self._manifest = build_manifest([*structural, *extracted_semantics, *collections])
        self._superschema: Superschema = build_superschema(self._manifest)
        self.store = store
        self.llm = llm
        self.embedding = embedding
        self.events = events or NullEmitter()
        # ``concurrency`` is a hard cap on simultaneous LLM extractions and
        # embedding calls. ``None`` means no cap (unbounded ``asyncio.gather``);
        # ``1`` serialises every LLM/embedding call — the CLI's ``--no-threads``
        # mode for resource-constrained local Ollama setups.
        if concurrency is not None and concurrency < 1:
            raise ValueError(f"concurrency must be >= 1 or None, got {concurrency}")
        self._concurrency = concurrency

    def schemas(
        self,
    ) -> list[type[BaseStructure] | type[BaseSemantic] | type[BaseCollection]]:
        """Return the root schemas (structural + semantic + collection) passed at construction.

        Stable public accessor used by the server layer (:mod:`ennoia.server`)
        to render the discovery payload without reaching into private state.
        """
        return [*self._structural, *self._semantics, *self._collections]

    def _make_semaphore(self) -> asyncio.Semaphore | None:
        # Semaphores bind to the running event loop, so we instantiate them
        # inside the async methods rather than at __init__ time.
        if self._concurrency is None:
            return None
        return asyncio.Semaphore(self._concurrency)

    def index(self, text: str, source_id: str) -> IndexResult:
        return asyncio.run(self.aindex(text=text, source_id=source_id))

    def search(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        *,
        filter_ids: list[str] | None = None,
        index: str | None = None,
    ) -> SearchResult:
        return asyncio.run(
            self.asearch(
                query=query,
                filters=filters,
                top_k=top_k,
                filter_ids=filter_ids,
                index=index,
            )
        )

    def filter(self, filters: dict[str, Any] | None = None) -> list[str]:
        return asyncio.run(self.afilter(filters=filters))

    def retrieve(self, source_id: str) -> dict[str, Any] | None:
        return asyncio.run(self.aretrieve(source_id=source_id))

    def delete(self, source_id: str) -> bool:
        return asyncio.run(self.adelete(source_id=source_id))

    async def aindex(self, text: str, source_id: str) -> IndexResult:
        start = time.perf_counter()
        sem = self._make_semaphore()
        try:
            batch = await execute_layers(
                seed_structural=list(self._structural),
                seed_semantic=list(self._semantics),
                seed_collection=list(self._collections),
                text=text,
                llm=self.llm,
                semaphore=sem,
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
            collections=batch.collections,
            confidences=batch.confidences,
            collection_confidences=batch.collection_confidences,
        )
        await self._apersist(result, sem)
        self.events.emit(
            IndexEvent(
                source_id=source_id,
                schemas_extracted=[
                    *result.structural.keys(),
                    *result.semantic.keys(),
                    *result.collections.keys(),
                ],
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
        *,
        filter_ids: list[str] | None = None,
        index: str | None = None,
    ) -> SearchResult:
        """Run vector search, optionally restricted by filters or pre-computed ids.

        ``filters`` and ``filter_ids`` are mutually exclusive:

        - ``filters=...`` is the classic one-shot flow — validate filters, run
          the structured filter on this call, then vector search.
        - ``filter_ids=[...]`` is the two-phase MCP flow — the caller has
          already run :meth:`afilter` and passes surviving ids directly;
          structured filtering is skipped.

        ``index`` optionally restricts vector search to a single semantic index.
        """
        if filters and filter_ids is not None:
            raise ValueError(
                "Pass either filters= or filter_ids=, not both."
                " Use afilter() to resolve filters to ids, then pass those ids back in."
            )
        start = time.perf_counter()
        if filter_ids is None:
            validate_filters(filters, self._superschema)
        query_vector = await self.embedding.embed_query(query)
        raw_hits = await plan_search(
            self.store,
            filters,
            query_vector,
            top_k,
            candidate_ids=filter_ids,
            index=index,
        )

        from ennoia.store.composite import Store

        # ``self.store`` is typed Store | HybridStore; use isinstance to pick
        # the right ``get`` source of truth for filling in each hit's
        # structured record.
        record_source: Store | HybridStore = self.store

        grouped: dict[str, SearchHit] = {}
        for vector_id, score, metadata in raw_hits:
            source_id = extract_source_id(metadata, vector_id)
            if source_id in grouped:
                # Keep the best score per source document.
                if score > grouped[source_id].score:
                    grouped[source_id].score = score
                continue

            structural_record: dict[str, Any] = {}
            if isinstance(record_source, Store):
                structural_record = await record_source.structured.get(source_id) or {}
            else:
                structural_record = await record_source.get(source_id) or {}

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

    async def afilter(self, filters: dict[str, Any] | None = None) -> list[str]:
        """Resolve ``filters`` to the list of matching ``source_id``s.

        The first half of the MCP two-phase flow (``filter → search``). The
        returned list is meant to be passed back in via
        :meth:`asearch(filter_ids=...)` on a subsequent call.
        """
        from ennoia.store.composite import Store

        validate_filters(filters, self._superschema)
        if isinstance(self.store, Store):
            return await self.store.structured.filter(filters or {})
        return await self.store.filter(filters or {})

    async def aretrieve(self, source_id: str) -> dict[str, Any] | None:
        """Return the full structured record for ``source_id``, or None if absent."""
        from ennoia.store.composite import Store

        if isinstance(self.store, Store):
            return await self.store.structured.get(source_id)
        return await self.store.get(source_id)

    async def adelete(self, source_id: str) -> bool:
        """Remove every structured + vector trace of ``source_id``.

        Returns ``True`` if anything was removed on either side. Composite
        stores fan the delete out to both halves; hybrid stores delegate to a
        single backend call.
        """
        from ennoia.store.composite import Store

        if isinstance(self.store, Store):
            structured_removed = await self.store.structured.delete(source_id)
            vectors_removed = await self.store.vector.delete_by_source(source_id)
            return structured_removed or vectors_removed > 0
        return await self.store.delete(source_id)

    async def _apersist(self, result: IndexResult, semaphore: asyncio.Semaphore | None) -> None:
        from ennoia.store.composite import Store

        # Seed every superschema field with None so documents that didn't
        # activate all possible emission branches still produce a uniform
        # structured record (crucial for tabular backends).
        flat: dict[str, Any] = {name: None for name in self._superschema.all_field_names()}
        for instance in result.structural.values():
            # ``_confidence`` rides on the instance via extra='allow'; strip before persist.
            dumped = instance.model_dump(mode="json")
            dumped.pop(CONFIDENCE_KEY, None)
            ns = get_schema_namespace(type(instance))
            if ns is not None:
                dumped = {f"{ns}__{k}": v for k, v in dumped.items()}
            flat.update(dumped)

        # Collect the pending ``(index_name, text, unique)`` tuples in a stable
        # order: semantics first (by schema name), then each collection's entities
        # in the order they were extracted. Empty semantic answers are a signal
        # from the LLM, not an invitation to embed the full document, so skip them.
        pending: list[tuple[str, str, str | None]] = []
        for name, text in result.semantic.items():
            if text:
                pending.append((name, text, None))
        for name, entities in result.collections.items():
            for entity in entities:
                template_text = entity.template()
                if not template_text:
                    continue
                pending.append((name, template_text, entity.get_unique()))

        if isinstance(self.store, HybridStore):
            # Native hybrid: embed (if any) then one unified upsert.
            entries: list[VectorEntry] = []
            if pending:
                texts = [text for _, text, _ in pending]
                embedded = await self._embed_documents(texts, semaphore)
                for (index_name, text, unique), vector in zip(pending, embedded, strict=True):
                    entries.append(
                        VectorEntry(
                            index_name=index_name,
                            vector=vector,
                            text=text,
                            unique=unique,
                        )
                    )
            await self.store.upsert(
                source_id=result.source_id,
                data=flat,
                entries=entries,
            )
            return

        # Composite path: persist structured record once, then individual vectors.
        assert isinstance(self.store, Store)
        await self.store.structured.upsert(result.source_id, flat)

        # Clear stale vectors for this source before re-writing. Collections use
        # user-defined (often random) ``unique`` keys, so we cannot reliably
        # overwrite by vector_id — old entries would accumulate forever.
        # ``delete_by_source`` is implemented by every concrete VectorStore.
        await self.store.vector.delete_by_source(result.source_id)

        if not pending:
            return

        texts = [text for _, text, _ in pending]
        vectors = await self._embed_documents(texts, semaphore)
        # Vector upserts run sequentially: every concrete VectorStore writes the
        # same backing file/connection per call, so concurrent writes would race.
        # Embedding is the slow step we parallelise above.
        for (index_name, semantic_text, unique), vector in zip(pending, vectors, strict=True):
            metadata: dict[str, Any] = {
                "source_id": result.source_id,
                "index": index_name,
                "text": semantic_text,
            }
            if unique is not None:
                metadata["unique"] = unique
            await self.store.vector.upsert(
                vector_id=make_semantic_vector_id(result.source_id, index_name, unique),
                vector=vector,
                metadata=metadata,
            )

    async def _embed_documents(
        self, texts: list[str], semaphore: asyncio.Semaphore | None
    ) -> list[list[float]]:
        # ``embed_batch`` issues one round-trip for backends with native
        # list-input APIs (OpenAI) or a single ``encode`` call for
        # sentence-transformers; both honour the semaphore as a single unit
        # of work. The semaphore primarily guards local-Ollama users — for
        # them embedding usually runs on CPU and any concurrency would
        # contend with the LLM thread.
        if semaphore is None:
            return await self.embedding.embed_batch(texts)
        async with semaphore:
            return await self.embedding.embed_batch(texts)
