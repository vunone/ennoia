"""In-memory vector store — NumPy cosine similarity."""

from __future__ import annotations

from typing import Any

from ennoia.store.base import VectorStore
from ennoia.store.vector._numpy import cosine_search

__all__ = ["InMemoryVectorStore"]


class InMemoryVectorStore(VectorStore):
    def __init__(self) -> None:
        self._entries: dict[str, tuple[list[float], dict[str, Any]]] = {}

    def upsert(
        self,
        vector_id: str,
        vector: list[float],
        metadata: dict[str, Any],
    ) -> None:
        self._entries[vector_id] = (list(vector), dict(metadata))

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        restrict_to: list[str] | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        return cosine_search(self._entries, query_vector, top_k, restrict_to)
