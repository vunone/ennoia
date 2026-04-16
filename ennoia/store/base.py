"""Abstract base classes every store backend must implement.

These ABCs replace the former ``typing.Protocol`` definitions. Inheritance
(rather than structural typing) is used so that concrete backends have a
single, discoverable contract and so that instantiating a partially
implemented backend fails loudly with ``TypeError``. Shared helper logic
(filter parsing/evaluation, vector-id conventions, lazy-import shims) lives in
:mod:`ennoia.utils`, not here, because backends with very different storage
models (in-memory vs SQL vs managed engines) can reuse the helpers a la
carte without being forced into a common implementation shape.

Every method is async. Even backends that perform no I/O (in-memory stores)
expose async methods so the pipeline doesn't have to branch on backend type;
trivial implementations just don't ``await`` anything internally.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

__all__ = ["HybridStore", "StructuredStore", "VectorStore"]


class StructuredStore(ABC):
    """Persists structural extractions and answers metadata filters."""

    @abstractmethod
    async def upsert(self, source_id: str, data: dict[str, Any]) -> None: ...

    @abstractmethod
    async def filter(self, query: dict[str, Any]) -> list[str]: ...

    @abstractmethod
    async def get(self, source_id: str) -> dict[str, Any] | None: ...


class VectorStore(ABC):
    """Persists semantic embeddings and answers similarity queries."""

    @abstractmethod
    async def upsert(
        self,
        vector_id: str,
        vector: list[float],
        metadata: dict[str, Any],
    ) -> None: ...

    @abstractmethod
    async def search(
        self,
        query_vector: list[float],
        top_k: int,
        restrict_to: list[str] | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]: ...


class HybridStore(ABC):
    """A single backend that natively does both structured filter and vector search."""

    @abstractmethod
    async def upsert(
        self,
        source_id: str,
        data: dict[str, Any],
        vectors: dict[str, list[float]],
    ) -> None: ...

    @abstractmethod
    async def hybrid_search(
        self,
        filters: dict[str, Any],
        query_vector: list[float],
        top_k: int,
    ) -> list[tuple[str, float, dict[str, Any]]]: ...

    @abstractmethod
    async def get(self, source_id: str) -> dict[str, Any] | None: ...
