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

Deletion (:meth:`StructuredStore.delete`, :meth:`VectorStore.delete_by_source`,
:meth:`HybridStore.delete`) and :meth:`HybridStore.filter` are declared as
concrete methods that raise :class:`NotImplementedError` rather than as
``@abstractmethod`` stubs. This preserves backward compatibility with
third-party Stage-2 store subclasses that were written before those contracts
existed: subclasses remain instantiable, and calls fail loudly with an
``override me`` message rather than at class-construction time.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

__all__ = [
    "HybridStore",
    "StructuredStore",
    "VectorEntry",
    "VectorStore",
    "validate_collection_name",
]


_COLLECTION_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def validate_collection_name(name: str) -> str:
    """Validate a collection/table name against a conservative identifier rule.

    Collection names are interpolated into SQL ``CREATE TABLE`` / ``INSERT`` /
    ``SELECT`` statements in the SQLite and pgvector backends (``aiosqlite``
    and ``asyncpg`` do not parameterise identifiers). Restricting the grammar
    to ASCII word characters keeps every backend's identifier handling
    identical and eliminates the SQL-injection surface at the constructor
    boundary rather than sprinkling it across each query.
    """
    if not _COLLECTION_NAME_RE.match(name):
        raise ValueError(
            f"collection must match {_COLLECTION_NAME_RE.pattern} (ASCII letters, "
            f"digits, underscores; leading non-digit), got {name!r}"
        )
    return name


@dataclass(frozen=True)
class VectorEntry:
    """One vector row to write: a (text, vector) pair tagged with its index.

    A ``BaseSemantic`` schema produces one entry per document with ``unique=None``.
    A ``BaseCollection`` produces N entries per document sharing an ``index_name``,
    each with its own ``unique`` dedup key returned by ``get_unique()``. Stores
    treat the ``(source_id, index_name, unique)`` triple as the row identity.
    """

    index_name: str
    vector: list[float]
    text: str
    unique: str | None = None


class StructuredStore(ABC):
    """Persists structural extractions and answers metadata filters."""

    @abstractmethod
    async def upsert(self, source_id: str, data: dict[str, Any]) -> None: ...

    @abstractmethod
    async def filter(self, query: dict[str, Any]) -> list[str]: ...

    @abstractmethod
    async def get(self, source_id: str) -> dict[str, Any] | None: ...

    async def delete(self, source_id: str) -> bool:
        """Remove ``source_id``'s record. Returns True iff something was deleted.

        Concrete backends override this. The default raises so Stage-2 subclasses
        that predate the contract fail loudly at call time rather than silently
        succeeding.
        """
        raise NotImplementedError(f"{type(self).__name__} must override delete(source_id)")


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
        *,
        index: str | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]: ...

    async def delete_by_source(self, source_id: str) -> int:
        """Remove every vector whose metadata points to ``source_id``.

        Returns the number of vectors removed. Default raises — see
        :meth:`StructuredStore.delete` for the rationale.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override delete_by_source(source_id)"
        )


class HybridStore(ABC):
    """A single backend that natively does both structured filter and vector search.

    Storage model: one row per :class:`VectorEntry`, with the document's
    structural ``data`` **denormalized (copied)** onto every row. A document
    with a single ``BaseSemantic`` answer is one row; a ``BaseCollection`` with
    N entities produces N rows sharing the same ``data`` payload. Row identity
    is the triple ``(source_id, index_name, unique)``. The pipeline collapses
    multi-row hits to one per ``source_id`` at search time.
    """

    @abstractmethod
    async def upsert(
        self,
        source_id: str,
        data: dict[str, Any],
        entries: list[VectorEntry],
    ) -> None: ...

    @abstractmethod
    async def hybrid_search(
        self,
        filters: dict[str, Any],
        query_vector: list[float],
        top_k: int,
        *,
        index: str | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]: ...

    @abstractmethod
    async def get(self, source_id: str) -> dict[str, Any] | None: ...

    async def filter(self, filters: dict[str, Any]) -> list[str]:
        """Return matching ``source_id``s without running vector search.

        Powers :meth:`ennoia.index.pipeline.Pipeline.afilter` — the first step of
        the agent's two-phase MCP flow (``filter → search(filter_ids=...)``).
        """
        raise NotImplementedError(f"{type(self).__name__} must override filter(filters)")

    async def delete(self, source_id: str) -> bool:
        """Remove every row with the given ``source_id`` (structural + vectors)."""
        raise NotImplementedError(f"{type(self).__name__} must override delete(source_id)")
