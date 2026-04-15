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


def test_parquet_roundtrip_and_reload(tmp_path: Path) -> None:
    store = ParquetStructuredStore(tmp_path)
    store.upsert("a", {"cat": "legal", "n": 5})
    store.upsert("b", {"cat": "medical", "n": 3})
    assert store.get("a") == {"cat": "legal", "n": 5}
    assert sorted(store.filter({"cat": "legal"})) == ["a"]

    # Reopen to exercise the Parquet read path.
    reopened = ParquetStructuredStore(tmp_path)
    assert reopened.get("b") == {"cat": "medical", "n": 3}


def test_filesystem_vector_store_persists(tmp_path: Path) -> None:
    store = FilesystemVectorStore(tmp_path)
    store.upsert("a:Summary", [0.1, 0.2, 0.3], {"source_id": "a", "index": "Summary"})
    store.upsert("b:Summary", [0.9, 0.1, 0.0], {"source_id": "b", "index": "Summary"})
    hits = store.search([0.1, 0.2, 0.3], top_k=2)
    assert hits[0][0] == "a:Summary"

    reopened = FilesystemVectorStore(tmp_path)
    hits2 = reopened.search([0.1, 0.2, 0.3], top_k=2)
    assert [h[0] for h in hits2] == [h[0] for h in hits]


def test_store_from_path_builds_both(tmp_path: Path) -> None:
    store = Store.from_path(tmp_path)
    store.structured.upsert("doc_1", {"cat": "legal"})
    store.vector.upsert("doc_1:Summary", [1.0, 0.0], {"source_id": "doc_1"})
    assert store.structured.get("doc_1") == {"cat": "legal"}
    assert store.vector.search([1.0, 0.0], top_k=1)[0][0] == "doc_1:Summary"

    # Files live in the expected subdirectories.
    assert (tmp_path / "structured" / "structured.parquet").exists()
    assert (tmp_path / "vectors" / "vectors.npy").exists()
    assert (tmp_path / "vectors" / "ids.json").exists()
