"""Conventions for vector identifiers and source-id extraction."""

from __future__ import annotations

from typing import Any

__all__ = [
    "SEMANTIC_VECTOR_ID_SEP",
    "extract_source_id",
    "make_semantic_vector_id",
    "parse_semantic_vector_id",
]


SEMANTIC_VECTOR_ID_SEP = ":"


def make_semantic_vector_id(
    source_id: str,
    index_name: str,
    unique: str | None = None,
) -> str:
    """Build a canonical vector id.

    Without ``unique``: ``{source_id}:{index_name}`` — the shape used for
    :class:`BaseSemantic` answers (one per (doc, index)). With ``unique``:
    ``{source_id}:{index_name}:{unique}`` — the shape used for
    :class:`BaseCollection` entries, where many rows share a
    ``(source_id, index_name)`` pair and ``unique`` is the per-entity key
    returned by ``get_unique()``.
    """
    if unique is None:
        return f"{source_id}{SEMANTIC_VECTOR_ID_SEP}{index_name}"
    return f"{source_id}{SEMANTIC_VECTOR_ID_SEP}{index_name}{SEMANTIC_VECTOR_ID_SEP}{unique}"


def parse_semantic_vector_id(vector_id: str) -> tuple[str, str, str | None]:
    """Inverse of :func:`make_semantic_vector_id`.

    Returns ``(source_id, index_name, unique)``. ``unique`` is ``None`` for
    the 2-part semantic form. When no separator is present the whole id is
    treated as the source id with empty index name — matching the behaviour
    of a plain structured id.
    """
    parts = vector_id.split(SEMANTIC_VECTOR_ID_SEP, 2)
    if len(parts) == 1:
        return parts[0], "", None
    if len(parts) == 2:
        return parts[0], parts[1], None
    return parts[0], parts[1], parts[2]


def extract_source_id(metadata: dict[str, Any], vector_id: str) -> str:
    """Prefer ``metadata['source_id']``; fall back to parsing ``vector_id``."""
    candidate = metadata.get("source_id")
    if isinstance(candidate, str):
        return candidate
    return parse_semantic_vector_id(vector_id)[0]
