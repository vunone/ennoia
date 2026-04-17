"""Stage 3: delete / delete_by_source coverage across every built-in store backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ennoia.store.base import HybridStore, StructuredStore, VectorStore
from ennoia.store.structured.memory import InMemoryStructuredStore
from ennoia.store.structured.parquet import ParquetStructuredStore
from ennoia.store.structured.sqlite import SQLiteStructuredStore
from ennoia.store.vector.filesystem import FilesystemVectorStore
from ennoia.store.vector.memory import InMemoryVectorStore

# ---------------------------------------------------------------------------
# StructuredStore.delete
# ---------------------------------------------------------------------------


async def test_in_memory_structured_delete() -> None:
    store = InMemoryStructuredStore()
    await store.upsert("a", {"cat": "legal"})
    assert await store.delete("a") is True
    assert await store.delete("a") is False
    assert await store.get("a") is None


async def test_parquet_structured_delete(tmp_path: Path) -> None:
    store = ParquetStructuredStore(tmp_path)
    await store.upsert("a", {"cat": "legal"})
    await store.upsert("b", {"cat": "medical"})
    assert await store.delete("a") is True
    assert await store.delete("missing") is False
    remaining = await store.filter({})
    assert remaining == ["b"]


async def test_sqlite_structured_delete(tmp_path: Path) -> None:
    store = SQLiteStructuredStore(tmp_path / "store.sqlite")
    await store.upsert("a", {"cat": "legal"})
    await store.upsert("b", {"cat": "medical"})
    assert await store.delete("a") is True
    assert await store.delete("missing") is False
    assert await store.filter({}) == ["b"]
    await store.close()


# ---------------------------------------------------------------------------
# VectorStore.delete_by_source
# ---------------------------------------------------------------------------


async def test_in_memory_vector_delete_by_source() -> None:
    store = InMemoryVectorStore()
    await store.upsert("a:X", [1.0, 0.0], {"source_id": "a", "index": "X"})
    await store.upsert("a:Y", [0.0, 1.0], {"source_id": "a", "index": "Y"})
    await store.upsert("b:X", [1.0, 1.0], {"source_id": "b", "index": "X"})
    removed = await store.delete_by_source("a")
    assert removed == 2
    hits = await store.search([1.0, 0.0], top_k=5)
    assert all(meta["source_id"] == "b" for _, _, meta in hits)


async def test_in_memory_vector_delete_by_source_missing_returns_zero() -> None:
    store = InMemoryVectorStore()
    assert await store.delete_by_source("missing") == 0


async def test_filesystem_vector_delete_by_source(tmp_path: Path) -> None:
    store = FilesystemVectorStore(tmp_path)
    await store.upsert("a:X", [1.0, 0.0], {"source_id": "a", "index": "X"})
    await store.upsert("b:X", [0.0, 1.0], {"source_id": "b", "index": "X"})
    removed = await store.delete_by_source("a")
    assert removed == 1
    # Round-trip through disk — reopen and confirm ``a`` is gone.
    reopened = FilesystemVectorStore(tmp_path)
    hits = await reopened.search([1.0, 0.0], top_k=5)
    assert [vid for vid, _, _ in hits] == ["b:X"]


async def test_filesystem_vector_delete_by_source_missing_returns_zero(tmp_path: Path) -> None:
    store = FilesystemVectorStore(tmp_path)
    await store.upsert("a:X", [1.0, 0.0], {"source_id": "a", "index": "X"})
    assert await store.delete_by_source("missing") == 0


# ---------------------------------------------------------------------------
# ABC default-raise behavior for third-party Stage-2 subclasses
# ---------------------------------------------------------------------------


async def test_structured_store_default_delete_raises() -> None:
    class _Minimal(StructuredStore):
        async def upsert(self, source_id: str, data: dict[str, Any]) -> None: ...
        async def filter(self, query: dict[str, Any]) -> list[str]:
            return []

        async def get(self, source_id: str) -> dict[str, Any] | None:
            return None

    with pytest.raises(NotImplementedError, match="must override delete"):
        await _Minimal().delete("x")


async def test_vector_store_default_delete_by_source_raises() -> None:
    class _Minimal(VectorStore):
        async def upsert(
            self, vector_id: str, vector: list[float], metadata: dict[str, Any]
        ) -> None: ...
        async def search(
            self,
            query_vector: list[float],
            top_k: int,
            restrict_to: list[str] | None = None,
            *,
            index: str | None = None,
        ) -> list[tuple[str, float, dict[str, Any]]]:
            return []

    with pytest.raises(NotImplementedError, match="must override delete_by_source"):
        await _Minimal().delete_by_source("x")


async def test_hybrid_store_default_filter_and_delete_raise() -> None:
    class _Minimal(HybridStore):
        async def upsert(
            self,
            source_id: str,
            data: dict[str, Any],
            entries: list[Any],
        ) -> None: ...
        async def hybrid_search(
            self,
            filters: dict[str, Any],
            query_vector: list[float],
            top_k: int,
            *,
            index: str | None = None,
        ) -> list[tuple[str, float, dict[str, Any]]]:
            return []

        async def get(self, source_id: str) -> dict[str, Any] | None:
            return None

    store = _Minimal()
    with pytest.raises(NotImplementedError, match="must override filter"):
        await store.filter({})
    with pytest.raises(NotImplementedError, match="must override delete"):
        await store.delete("x")
