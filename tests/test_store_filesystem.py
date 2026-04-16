"""Parquet structured + NumPy filesystem vector round-trip + persistence-across-reopen."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

from ennoia.store.composite import Store
from ennoia.store.structured.parquet import ParquetStructuredStore
from ennoia.store.vector.filesystem import FilesystemVectorStore


async def test_parquet_roundtrip_and_reload(tmp_path: Path) -> None:
    store = ParquetStructuredStore(tmp_path)
    await store.upsert("a", {"cat": "legal", "n": 5})
    await store.upsert("b", {"cat": "medical", "n": 3})
    assert await store.get("a") == {"cat": "legal", "n": 5}
    assert sorted(await store.filter({"cat": "legal"})) == ["a"]

    # Reopen to exercise the Parquet read path.
    reopened = ParquetStructuredStore(tmp_path)
    assert await reopened.get("b") == {"cat": "medical", "n": 3}


async def test_filesystem_vector_store_persists(tmp_path: Path) -> None:
    store = FilesystemVectorStore(tmp_path)
    await store.upsert("a:Summary", [0.1, 0.2, 0.3], {"source_id": "a", "index": "Summary"})
    await store.upsert("b:Summary", [0.9, 0.1, 0.0], {"source_id": "b", "index": "Summary"})
    hits = await store.search([0.1, 0.2, 0.3], top_k=2)
    assert hits[0][0] == "a:Summary"

    reopened = FilesystemVectorStore(tmp_path)
    hits2 = await reopened.search([0.1, 0.2, 0.3], top_k=2)
    assert [h[0] for h in hits2] == [h[0] for h in hits]


async def test_store_from_path_builds_both(tmp_path: Path) -> None:
    store = Store.from_path(tmp_path)
    await store.structured.upsert("doc_1", {"cat": "legal"})
    await store.vector.upsert("doc_1:Summary", [1.0, 0.0], {"source_id": "doc_1"})
    assert await store.structured.get("doc_1") == {"cat": "legal"}
    assert (await store.vector.search([1.0, 0.0], top_k=1))[0][0] == "doc_1:Summary"

    # Files live in the expected subdirectories.
    assert (tmp_path / "structured" / "structured.parquet").exists()
    assert (tmp_path / "vectors" / "vectors.npy").exists()
    assert (tmp_path / "vectors" / "ids.json").exists()


# ---------------------------------------------------------------------------
# Edge: parquet _load with an empty __data__ entry — skip silently
# ---------------------------------------------------------------------------


async def test_parquet_load_skips_empty_data_rows(tmp_path: Path) -> None:
    import pandas as pd

    target = tmp_path / "structured.parquet"
    # Manually write a parquet with one row whose ``__data__`` is empty; the
    # loader must skip it without failing the whole load.
    pd.DataFrame(
        [
            {"__source_id__": "good", "__data__": '{"cat": "legal"}'},
            {"__source_id__": "empty", "__data__": ""},
        ]
    ).to_parquet(target, index=False)
    store = ParquetStructuredStore(tmp_path)
    assert await store.get("good") == {"cat": "legal"}
    assert await store.get("empty") is None


# ---------------------------------------------------------------------------
# Edge: filesystem vector store flush with empty entries removes the .npy file
# ---------------------------------------------------------------------------


async def test_filesystem_vector_flush_with_no_entries_removes_vectors_npy(
    tmp_path: Path,
) -> None:
    store = FilesystemVectorStore(tmp_path)
    await store.upsert("a", [1.0, 0.0], {"source_id": "a"})
    vec_file = tmp_path / "vectors.npy"
    assert vec_file.exists()

    # Bypass public API to drain entries; then _flush must unlink the .npy.
    store._entries.clear()
    store._flush()
    assert not vec_file.exists()


def test_filesystem_vector_flush_with_no_entries_noop_when_file_absent(
    tmp_path: Path,
) -> None:
    # Constructing an empty store and flushing shouldn't crash even though
    # vectors.npy doesn't exist.
    store = FilesystemVectorStore(tmp_path)
    store._flush()
    assert not (tmp_path / "vectors.npy").exists()


# ---------------------------------------------------------------------------
# Edge: cosine_search with zero-magnitude vectors
# ---------------------------------------------------------------------------


def test_cosine_search_with_zero_query_returns_empty(tmp_path: Path) -> None:
    from ennoia.store.vector._numpy import cosine_search

    entries = {"a": ([1.0, 0.0], {"source_id": "a"})}
    assert cosine_search(entries, [0.0, 0.0], top_k=5) == []


def test_cosine_search_skips_zero_magnitude_entries(tmp_path: Path) -> None:
    from ennoia.store.vector._numpy import cosine_search

    entries = {
        "zero": ([0.0, 0.0], {"source_id": "zero"}),
        "real": ([1.0, 0.0], {"source_id": "real"}),
    }
    hits = cosine_search(entries, [1.0, 0.0], top_k=5)
    assert [h[0] for h in hits] == ["real"]


def test_cosine_search_empty_entries_short_circuits() -> None:
    from ennoia.store.vector._numpy import cosine_search

    assert cosine_search({}, [1.0, 0.0], top_k=5) == []
