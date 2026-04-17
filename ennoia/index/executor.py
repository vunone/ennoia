"""Layer-wise parallel executor for structural + collection + semantic extraction.

Stage 1 ran every schema serially. Stage 2 processes each structural layer with
``asyncio.gather`` so independent branches complete concurrently while
``extend()`` still sees a fully-populated parent before queueing children.

Phase order:

1. Structural DAG — layer-wise BFS driven by ``extend()``. Each layer runs in
   parallel; children are queued for the next layer after every parent's
   ``extend()`` has been consulted.
2. Collection phase — every queued :class:`BaseCollection` runs in parallel
   (each one internally iterates against the LLM). ``extend()`` is called per
   extracted entity after its collection finishes; emissions re-enter the
   structural DAG, semantic queue, or collection queue as appropriate.
3. Semantic phase — a single parallel layer of :class:`BaseSemantic` answers.

De-duplication is schema-class-based so a schema queued twice runs once.
Collections emitting structural children cause the DAG to re-drain before the
semantic layer starts so ``extend()`` still sees fully populated parents in
every downstream layer.

A ``semaphore`` argument caps how many extractions are in flight at once;
``None`` means no cap. Pipeline injects the cap from its ``concurrency``
constructor argument so CLI users running local Ollama can serialise calls
via ``--no-threads``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import TYPE_CHECKING, TypeVar

from ennoia.index.exceptions import SchemaError
from ennoia.index.extractor import (
    CONFIDENCE_KEY,
    extract_collection,
    extract_semantic,
    extract_structural,
)
from ennoia.schema.base import (
    BaseCollection,
    BaseSemantic,
    BaseStructure,
    get_schema_extensions,
)

if TYPE_CHECKING:
    from ennoia.adapters.llm.base import LLMAdapter

_T = TypeVar("_T")

__all__ = ["ExtractionBatch", "execute_layers"]


class ExtractionBatch:
    """Accumulates extractions across parallel layers.

    The executor records structural instances + their confidences in the order
    they resolve within a layer, then calls ``extend()`` sequentially on the
    layer results so the queueing order for the next layer is deterministic
    even when extractions finish out-of-order. Collections land in
    :attr:`collections` with per-entity confidences in
    :attr:`collection_confidences`; semantics stay in :attr:`semantic` as today.
    """

    def __init__(self) -> None:
        self.structural: dict[str, BaseStructure] = {}
        self.semantic: dict[str, str] = {}
        self.collections: dict[str, list[BaseCollection]] = {}
        self.collection_confidences: dict[str, list[float]] = {}
        self.confidences: dict[str, float] = {}


def _render_parent_context(parent: BaseStructure | BaseCollection) -> str:
    dumped = parent.model_dump(mode="json")
    dumped.pop(CONFIDENCE_KEY, None)
    import json as _json

    return f"Parent extraction from {type(parent).__name__}: " + _json.dumps(
        dumped, default=str, sort_keys=True
    )


async def execute_layers(
    seed_structural: list[type[BaseStructure]],
    seed_semantic: list[type[BaseSemantic]],
    text: str,
    llm: LLMAdapter,
    semaphore: asyncio.Semaphore | None = None,
    *,
    seed_collection: list[type[BaseCollection]] | None = None,
) -> ExtractionBatch:
    """Run seeds through the layered DAG and return accumulated results.

    ``seed_collection`` is keyword-only with a default of ``[]`` so existing
    callers that pre-date :class:`BaseCollection` keep working; the Pipeline
    always passes it explicitly.
    """
    seed_collection = list(seed_collection or [])
    batch = ExtractionBatch()

    seen_structural: set[type[BaseStructure]] = set()
    seen_semantic: set[type[BaseSemantic]] = set()
    seen_collection: set[type[BaseCollection]] = set()

    pending_structural: list[tuple[type[BaseStructure], BaseStructure | BaseCollection | None]] = [
        (cls, None) for cls in seed_structural
    ]
    semantic_queue: list[tuple[type[BaseSemantic], BaseStructure | BaseCollection | None]] = [
        (cls, None) for cls in seed_semantic
    ]
    collection_queue: list[tuple[type[BaseCollection], BaseStructure | BaseCollection | None]] = [
        (cls, None) for cls in seed_collection
    ]

    # Drain structurals first; collections can re-feed new structurals (and
    # new collections), in which case we re-enter this loop until both queues
    # are empty.
    while pending_structural or collection_queue:
        await _drain_structural_dag(
            pending_structural,
            seen_structural,
            batch,
            semantic_queue,
            collection_queue,
            text,
            llm,
            semaphore,
        )
        pending_structural = []

        if collection_queue:
            # Swap to a local batch so any new collections emitted by extend()
            # land in ``collection_queue`` and get picked up on the next outer iteration.
            layer_queue = collection_queue
            collection_queue = []
            new_structural = await _run_collection_layer(
                layer_queue,
                seen_collection,
                batch,
                semantic_queue,
                collection_queue,
                text,
                llm,
                semaphore,
            )
            pending_structural.extend(new_structural)

    if semantic_queue:
        filtered: list[tuple[type[BaseSemantic], BaseStructure | BaseCollection | None]] = []
        for schema, parent in semantic_queue:
            if schema in seen_semantic:
                continue
            seen_semantic.add(schema)
            filtered.append((schema, parent))

        async def _run_sem(
            schema: type[BaseSemantic],
            parent: BaseStructure | BaseCollection | None,
        ) -> tuple[type[BaseSemantic], str, float]:
            # Semantic prompts don't consume parent context today — text is primary.
            _ = parent
            answer, confidence = await _under_semaphore(
                semaphore, extract_semantic(schema=schema, text=text, llm=llm)
            )
            return schema, answer, confidence

        sem_results = await asyncio.gather(*(_run_sem(s, p) for s, p in filtered))
        for schema, answer, confidence in sem_results:
            batch.semantic[schema.__name__] = answer
            batch.confidences[schema.__name__] = confidence

    return batch


async def _drain_structural_dag(
    pending_structural: list[tuple[type[BaseStructure], BaseStructure | BaseCollection | None]],
    seen_structural: set[type[BaseStructure]],
    batch: ExtractionBatch,
    semantic_queue: list[tuple[type[BaseSemantic], BaseStructure | BaseCollection | None]],
    collection_queue: list[tuple[type[BaseCollection], BaseStructure | BaseCollection | None]],
    text: str,
    llm: LLMAdapter,
    semaphore: asyncio.Semaphore | None,
) -> None:
    while pending_structural:
        layer: list[tuple[type[BaseStructure], BaseStructure | BaseCollection | None]] = []
        next_layer_candidates: list[
            tuple[type[BaseStructure], BaseStructure | BaseCollection | None]
        ] = []
        for schema, parent in pending_structural:
            if schema in seen_structural:
                continue
            seen_structural.add(schema)
            layer.append((schema, parent))

        async def _run(
            schema: type[BaseStructure],
            parent: BaseStructure | BaseCollection | None,
        ) -> tuple[type[BaseStructure], BaseStructure, float]:
            context_additions = [_render_parent_context(parent)] if parent is not None else []
            instance, confidence = await _under_semaphore(
                semaphore,
                extract_structural(
                    schema=schema,
                    text=text,
                    context_additions=context_additions,
                    llm=llm,
                ),
            )
            return schema, instance, confidence

        results = await asyncio.gather(*(_run(schema, parent) for schema, parent in layer))

        ordered = {schema: (instance, confidence) for schema, instance, confidence in results}
        for schema, _parent in layer:
            instance, confidence = ordered[schema]
            batch.structural[schema.__name__] = instance
            batch.confidences[schema.__name__] = confidence

            _route_children(
                parent_instance=instance,
                children=instance.extend(),
                seen_structural=seen_structural,
                next_structural=next_layer_candidates,
                semantic_queue=semantic_queue,
                collection_queue=collection_queue,
            )

        pending_structural.clear()
        pending_structural.extend(next_layer_candidates)


async def _run_collection_layer(
    layer_queue: list[tuple[type[BaseCollection], BaseStructure | BaseCollection | None]],
    seen_collection: set[type[BaseCollection]],
    batch: ExtractionBatch,
    semantic_queue: list[tuple[type[BaseSemantic], BaseStructure | BaseCollection | None]],
    collection_queue: list[tuple[type[BaseCollection], BaseStructure | BaseCollection | None]],
    text: str,
    llm: LLMAdapter,
    semaphore: asyncio.Semaphore | None,
) -> list[tuple[type[BaseStructure], BaseStructure | BaseCollection | None]]:
    """Run every queued collection in parallel; return new structural work to drain.

    Newly emitted collections (from per-entity ``extend()`` calls) are appended
    to the shared ``collection_queue`` so the outer loop picks them up on the
    next iteration.
    """
    layer: list[tuple[type[BaseCollection], BaseStructure | BaseCollection | None]] = []
    for schema, parent in layer_queue:
        if schema in seen_collection:
            continue
        seen_collection.add(schema)
        layer.append((schema, parent))

    async def _run(
        schema: type[BaseCollection],
        parent: BaseStructure | BaseCollection | None,
    ) -> tuple[type[BaseCollection], list[BaseCollection], list[float]]:
        context_additions = [_render_parent_context(parent)] if parent is not None else []
        instances, confidences = await _under_semaphore(
            semaphore,
            extract_collection(
                schema=schema,
                text=text,
                context_additions=context_additions,
                llm=llm,
            ),
        )
        return schema, instances, confidences

    results = await asyncio.gather(*(_run(schema, parent) for schema, parent in layer))

    new_structural: list[tuple[type[BaseStructure], BaseStructure | BaseCollection | None]] = []
    ordered = {schema: (instances, confs) for schema, instances, confs in results}
    for schema, _parent in layer:
        instances, confs = ordered[schema]
        batch.collections[schema.__name__] = instances
        batch.collection_confidences[schema.__name__] = confs
        # Summary confidence in the flat map — the mean, so existing consumers
        # keep reading one value per index regardless of whether it's semantic
        # or collection.
        batch.confidences[schema.__name__] = sum(confs) / len(confs) if confs else 1.0

        # ``seen_structural_for_layer`` is scoped to this collection's extend()
        # calls so a structural child emitted by this collection gets queued
        # (dedup against the global ``seen_structural`` happens in ``_drain_structural_dag``).
        seen_structural_for_layer: set[type[BaseStructure]] = set()
        for entity in instances:
            _route_children(
                parent_instance=entity,
                children=entity.extend(),
                seen_structural=seen_structural_for_layer,
                next_structural=new_structural,
                semantic_queue=semantic_queue,
                collection_queue=collection_queue,
                declared_lookup=type(entity),
            )

    return new_structural


def _route_children(
    *,
    parent_instance: BaseStructure | BaseCollection,
    children: list[type[BaseStructure] | type[BaseSemantic] | type[BaseCollection]],
    seen_structural: set[type[BaseStructure]],
    next_structural: list[tuple[type[BaseStructure], BaseStructure | BaseCollection | None]],
    semantic_queue: list[tuple[type[BaseSemantic], BaseStructure | BaseCollection | None]],
    collection_queue: list[tuple[type[BaseCollection], BaseStructure | BaseCollection | None]],
    declared_lookup: type | None = None,
) -> None:
    """Validate and partition a parent's ``extend()`` return value into queues."""
    source_cls = declared_lookup or type(parent_instance)
    declared = set(get_schema_extensions(source_cls))
    for child in children:
        if child not in declared:
            raise SchemaError(
                f"{source_cls.__name__}.extend() returned "
                f"{child.__name__!r}, which is not declared in "
                f"Schema.extensions "
                f"{sorted(c.__name__ for c in declared)}."
            )
        if issubclass(child, BaseCollection):
            collection_queue.append((child, parent_instance))
        elif issubclass(child, BaseSemantic):
            semantic_queue.append((child, parent_instance))
        elif child not in seen_structural:
            next_structural.append((child, parent_instance))


async def _under_semaphore(semaphore: asyncio.Semaphore | None, coro: Awaitable[_T]) -> _T:
    """Await ``coro`` while holding ``semaphore`` (if provided).

    Centralised so the structural, collection, and semantic paths share the
    same cap. With ``semaphore=None`` the call is a straight ``await`` — zero
    overhead for the unbounded path.
    """
    if semaphore is None:
        return await coro
    async with semaphore:
        return await coro
