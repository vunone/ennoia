"""Shared NumPy cosine-similarity search used by every NumPy-backed vector store."""

from __future__ import annotations

from typing import Any

from ennoia.utils.imports import require_module

__all__ = ["cosine_search"]


def cosine_search(
    entries: dict[str, tuple[list[float], dict[str, Any]]],
    query_vector: list[float],
    top_k: int,
    restrict_to: list[str] | None = None,
    *,
    index: str | None = None,
) -> list[tuple[str, float, dict[str, Any]]]:
    """Return the ``top_k`` most similar entries to ``query_vector``.

    ``entries`` is a mapping from vector id to ``(vector, metadata)``. The
    restrict-list filters by ``metadata['source_id']`` — the canonical filter
    hook used by the two-phase query planner. When ``index`` is set, only
    entries whose metadata records the same semantic index name participate.
    """
    if not entries:
        return []

    np = require_module("numpy", "sentence-transformers")

    allowed: set[str] | None = set(restrict_to) if restrict_to is not None else None

    query_arr = np.asarray(query_vector, dtype=float)
    q_norm = float(np.linalg.norm(query_arr))
    if q_norm == 0.0:
        return []

    scored: list[tuple[str, float, dict[str, Any]]] = []
    for vector_id, (vec, metadata) in entries.items():
        if allowed is not None and metadata.get("source_id") not in allowed:
            continue
        if index is not None and metadata.get("index") != index:
            continue
        vec_arr = np.asarray(vec, dtype=float)
        v_norm = float(np.linalg.norm(vec_arr))
        if v_norm == 0.0:
            continue
        score = float(np.dot(query_arr, vec_arr) / (q_norm * v_norm))
        scored.append((vector_id, score, dict(metadata)))

    scored.sort(key=lambda t: t[1], reverse=True)
    return scored[:top_k]
