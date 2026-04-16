"""Layer-wise parallel executor for structural + semantic extraction.

Stage 1 ran every schema serially. Stage 2 processes each layer with
``asyncio.gather`` so independent branches complete concurrently while
``extend()`` still sees a fully-populated parent before queueing children.

Layers are built dynamically from the runtime DAG:

1. Seed structural schemas form layer 0.
2. After each layer completes, every extracted instance's ``extend()`` return
   value is partitioned into structural (next layer) and semantic (appended to
   the semantic queue) contributions.
3. Once all structural layers drain, all semantics run in a single parallel
   layer.

De-duplication is schema-class-based so a schema queued twice (by two parent
branches) runs once.

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
from ennoia.index.extractor import CONFIDENCE_KEY, extract_semantic, extract_structural
from ennoia.schema.base import BaseSemantic, BaseStructure, get_schema_extensions

if TYPE_CHECKING:
    from ennoia.adapters.llm.base import LLMAdapter

_T = TypeVar("_T")

__all__ = ["ExtractionBatch", "execute_layers"]


class ExtractionBatch:
    """Accumulates extractions across parallel layers.

    The executor records structural instances + their confidences in the order
    they resolve within a layer, then calls ``extend()`` sequentially on the
    layer results so the queueing order for the next layer is deterministic
    even when extractions finish out-of-order.
    """

    def __init__(self) -> None:
        self.structural: dict[str, BaseStructure] = {}
        self.semantic: dict[str, str] = {}
        self.confidences: dict[str, float] = {}


def _render_parent_context(parent: BaseStructure) -> str:
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
) -> ExtractionBatch:
    """Run seeds through the layer-wise parallel DAG and return accumulated results."""
    batch = ExtractionBatch()

    seen_structural: set[type[BaseStructure]] = set()
    seen_semantic: set[type[BaseSemantic]] = set()

    # ``pending_context`` maps each queued schema to the parent whose ``extend()``
    # introduced it; the root layer has no parent. A list lets a schema be
    # enqueued under multiple parents (first-write-wins: the first to reach the
    # runner provides the context).
    pending_structural: list[tuple[type[BaseStructure], BaseStructure | None]] = [
        (cls, None) for cls in seed_structural
    ]
    semantic_queue: list[tuple[type[BaseSemantic], BaseStructure | None]] = [
        (cls, None) for cls in seed_semantic
    ]

    while pending_structural:
        layer: list[tuple[type[BaseStructure], BaseStructure | None]] = []
        next_layer_candidates: list[tuple[type[BaseStructure], BaseStructure | None]] = []
        for schema, parent in pending_structural:
            if schema in seen_structural:
                continue
            seen_structural.add(schema)
            layer.append((schema, parent))
        # ``next_layer_candidates`` receives new work from this layer's extend()
        # calls; it's repopulated before being assigned to ``pending_structural``
        # for the next iteration.

        async def _run(
            schema: type[BaseStructure], parent: BaseStructure | None
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

        # Record in declared-layer order for determinism, then fan out children.
        ordered = {schema: (instance, confidence) for schema, instance, confidence in results}
        for schema, _parent in layer:
            instance, confidence = ordered[schema]
            batch.structural[schema.__name__] = instance
            batch.confidences[schema.__name__] = confidence

            declared = set(get_schema_extensions(type(instance)))
            for child in instance.extend():
                if child not in declared:
                    raise SchemaError(
                        f"{type(instance).__name__}.extend() returned "
                        f"{child.__name__!r}, which is not declared in "
                        f"Schema.extensions "
                        f"{sorted(c.__name__ for c in declared)}."
                    )
                if issubclass(child, BaseSemantic):
                    # ``seen_semantic`` is empty during the structural phase;
                    # dedup happens in the semantic loop below.
                    semantic_queue.append((child, instance))
                elif child not in seen_structural:
                    next_layer_candidates.append((child, instance))

        pending_structural = next_layer_candidates

    if semantic_queue:

        async def _run_sem(
            schema: type[BaseSemantic], parent: BaseStructure | None
        ) -> tuple[type[BaseSemantic], str, float]:
            # Semantic prompts don't consume parent context today — text is primary.
            _ = parent
            answer, confidence = await _under_semaphore(
                semaphore, extract_semantic(schema=schema, text=text, llm=llm)
            )
            return schema, answer, confidence

        filtered: list[tuple[type[BaseSemantic], BaseStructure | None]] = []
        for schema, parent in semantic_queue:
            if schema in seen_semantic:
                continue
            seen_semantic.add(schema)
            filtered.append((schema, parent))

        sem_results = await asyncio.gather(*(_run_sem(s, p) for s, p in filtered))
        for schema, answer, confidence in sem_results:
            batch.semantic[schema.__name__] = answer
            batch.confidences[schema.__name__] = confidence

    return batch


async def _under_semaphore(semaphore: asyncio.Semaphore | None, coro: Awaitable[_T]) -> _T:
    """Await ``coro`` while holding ``semaphore`` (if provided).

    Centralised so the structural and semantic paths share the same cap.
    With ``semaphore=None`` the call is a straight ``await`` — zero overhead
    for the unbounded path.
    """
    if semaphore is None:
        return await coro
    async with semaphore:
        return await coro
