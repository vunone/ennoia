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
    *,
    candidate_ids: list[str] | None = None,
    index: str | None = None,
) -> list[tuple[str, float, dict[str, Any]]]:
    """Return a list of (vector_id, score, metadata) hits in score order.

    For a composite :class:`~ennoia.store.composite.Store`: when ``candidate_ids``
    is supplied the structured filter phase is skipped entirely (this is the
    MCP ``filter → search(filter_ids=...)`` flow); otherwise ``filters`` are
    applied to the structured side and the surviving ``source_id``s become the
    restrict-list for the vector search. For a ``HybridStore``, the same
    ``candidate_ids`` path maps to a post-filter on the native hit set.

    ``index`` optionally restricts vector search to one semantic index name.
    """
    from ennoia.store.composite import Store

    filters = filters or {}

    if isinstance(store, Store):
        if candidate_ids is not None:
            if not candidate_ids:
                return []
            return await store.vector.search(
                query_vector, top_k=top_k, restrict_to=candidate_ids, index=index
            )
        candidate_ids = await store.structured.filter(filters)
        if not candidate_ids:
            return []
        return await store.vector.search(
            query_vector, top_k=top_k, restrict_to=candidate_ids, index=index
        )

    # HybridStore path — delegate to the backend's single-roundtrip search.
    if candidate_ids is not None:
        if not candidate_ids:
            return []
        # No native filter_ids knob on the hybrid ABC; post-filter the hit set.
        # Request ``top_k * 4`` candidates to keep the post-filter yield high
        # for sparse overlaps; cheap for every hybrid backend.
        raw = await store.hybrid_search(
            filters={}, query_vector=query_vector, top_k=top_k * 4, index=index
        )
        allowed = set(candidate_ids)
        return [(vid, score, meta) for vid, score, meta in raw if meta.get("source_id") in allowed][
            :top_k
        ]
    return await store.hybrid_search(
        filters=filters, query_vector=query_vector, top_k=top_k, index=index
    )
