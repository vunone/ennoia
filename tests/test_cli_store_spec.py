"""Tests for ``ennoia.cli.factories.parse_store_spec`` — prefix dispatch + errors."""

from __future__ import annotations

from pathlib import Path

import pytest
import typer

from ennoia.cli.factories import parse_store_spec
from ennoia.store.composite import Store
from ennoia.store.hybrid.pgvector import PgVectorHybridStore
from ennoia.store.hybrid.qdrant import QdrantHybridStore


def test_plain_path_builds_filesystem_store(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    pytest.importorskip("pandas")
    store = parse_store_spec(str(tmp_path / "idx"), collection="mine")
    assert isinstance(store, Store)
    # Collection name becomes the subdirectory.
    assert (tmp_path / "idx" / "mine").is_dir()


def test_file_prefix_builds_filesystem_store(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    pytest.importorskip("pandas")
    store = parse_store_spec(f"file:{tmp_path / 'idx'}", collection="mine")
    assert isinstance(store, Store)
    assert (tmp_path / "idx" / "mine").is_dir()


def test_file_prefix_without_value_errors() -> None:
    with pytest.raises(typer.BadParameter, match="missing the value after"):
        parse_store_spec("file:")


def test_qdrant_prefix_builds_hybrid_store() -> None:
    store = parse_store_spec(
        "qdrant:my_coll",
        qdrant_url="http://qdrant.example:6333",
        qdrant_api_key="sekret",
    )
    assert isinstance(store, QdrantHybridStore)
    assert store.collection == "my_coll"


def test_qdrant_prefix_requires_url() -> None:
    with pytest.raises(typer.BadParameter, match="qdrant:"):
        parse_store_spec("qdrant:my_coll")


def test_qdrant_prefix_without_collection_errors() -> None:
    with pytest.raises(typer.BadParameter, match="missing the value after"):
        parse_store_spec("qdrant:", qdrant_url="http://x")


def test_pgvector_prefix_builds_hybrid_store() -> None:
    store = parse_store_spec("pgvector:my_coll", pg_dsn="postgresql://x/db")
    assert isinstance(store, PgVectorHybridStore)
    assert store.collection == "my_coll"


def test_pgvector_prefix_requires_dsn() -> None:
    with pytest.raises(typer.BadParameter, match="pgvector:"):
        parse_store_spec("pgvector:my_coll")


def test_pgvector_prefix_without_collection_errors() -> None:
    with pytest.raises(typer.BadParameter, match="missing the value after"):
        parse_store_spec("pgvector:", pg_dsn="postgresql://x")
