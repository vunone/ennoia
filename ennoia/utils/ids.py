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


def make_semantic_vector_id(source_id: str, index_name: str) -> str:
    """Build the canonical ``{source_id}:{index_name}`` vector id."""
    return f"{source_id}{SEMANTIC_VECTOR_ID_SEP}{index_name}"


def parse_semantic_vector_id(vector_id: str) -> tuple[str, str]:
    """Inverse of :func:`make_semantic_vector_id`.

    When no separator is present the whole id is treated as the source id and
    the index name is empty — matching the behaviour of a plain structured id.
    """
    source_id, sep, index_name = vector_id.partition(SEMANTIC_VECTOR_ID_SEP)
    if not sep:
        return vector_id, ""
    return source_id, index_name


def extract_source_id(metadata: dict[str, Any], vector_id: str) -> str:
    """Prefer ``metadata['source_id']``; fall back to parsing ``vector_id``."""
    candidate = metadata.get("source_id")
    if isinstance(candidate, str):
        return candidate
    return parse_semantic_vector_id(vector_id)[0]
