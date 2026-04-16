"""Query planner — decides filter-then-search vs hybrid single-roundtrip."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ennoia.store.base import HybridStore
    from ennoia.store.composite import Store

__all__ = ["plan_search"]


async def plan_search(
    store: Store | HybridStore,
    filters: dict[str, Any] | None,
    query_vector: list[float],
    top_k: int,
) -> list[tuple[str, float, dict[str, Any]]]:
    """Return a list of (vector_id, score, metadata) hits in score order.

    For a composite `Store`: filter on the structured side first, then run
    vector search restricted to the surviving `source_id`s. For a
    `HybridStore`, delegate to `hybrid_search` and let the backend
    execute both in one roundtrip.
    """
    from ennoia.store.composite import Store

    filters = filters or {}

    if isinstance(store, Store):
        candidate_ids = await store.structured.filter(filters)
        if not candidate_ids:
            return []
        return await store.vector.search(query_vector, top_k=top_k, restrict_to=candidate_ids)

    # HybridStore path — exists for symmetry; no Stage 1 implementations.
    return await store.hybrid_search(filters=filters, query_vector=query_vector, top_k=top_k)
