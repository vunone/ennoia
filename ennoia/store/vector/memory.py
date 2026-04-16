"""In-memory vector store — NumPy cosine similarity."""

from __future__ import annotations

from typing import Any

from ennoia.store.base import VectorStore, validate_collection_name
from ennoia.store.vector._numpy import cosine_search

__all__ = ["InMemoryVectorStore"]


class InMemoryVectorStore(VectorStore):
    def __init__(self, *, collection: str = "documents") -> None:
        self.collection = validate_collection_name(collection)
        self._entries: dict[str, tuple[list[float], dict[str, Any]]] = {}

    async def upsert(
        self,
        vector_id: str,
        vector: list[float],
        metadata: dict[str, Any],
    ) -> None:
        self._entries[vector_id] = (list(vector), dict(metadata))

    async def search(
        self,
        query_vector: list[float],
        top_k: int,
        restrict_to: list[str] | None = None,
        *,
        index: str | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        return cosine_search(self._entries, query_vector, top_k, restrict_to, index=index)

    async def delete_by_source(self, source_id: str) -> int:
        doomed = [
            vid for vid, (_, meta) in self._entries.items() if meta.get("source_id") == source_id
        ]
        for vid in doomed:
            del self._entries[vid]
        return len(doomed)
