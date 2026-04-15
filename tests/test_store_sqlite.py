"""SQLite structured store — scalar SQL path + Python-fallback for list/substring ops."""

from __future__ import annotations

from pathlib import Path

import pytest

from ennoia.store.structured.sqlite import SQLiteStructuredStore


@pytest.fixture()
def store(tmp_path: Path) -> SQLiteStructuredStore:
    s = SQLiteStructuredStore(tmp_path / "ennoia.db")
    s.upsert("a", {"cat": "legal", "n": 5, "tags": ["x", "y"], "note": None})
    s.upsert("b", {"cat": "medical", "n": 3, "tags": ["y"], "note": "hi"})
    s.upsert("c", {"cat": "legal", "n": 8, "tags": ["x"], "note": "hello"})
    return s


def test_upsert_overwrites_existing_row(store: SQLiteStructuredStore) -> None:
    store.upsert("a", {"cat": "financial", "n": 1, "tags": []})
    assert store.get("a") == {"cat": "financial", "n": 1, "tags": []}


def test_scalar_filters_use_sql_path(store: SQLiteStructuredStore) -> None:
    assert sorted(store.filter({"cat": "legal"})) == ["a", "c"]
    assert store.filter({"n__gt": 4}) == ["a", "c"]
    assert sorted(store.filter({"cat__in": ["legal", "medical"]})) == ["a", "b", "c"]


def test_list_operator_uses_python_fallback(store: SQLiteStructuredStore) -> None:
    assert sorted(store.filter({"tags__contains": "x"})) == ["a", "c"]
    assert store.filter({"tags__contains_all": "x,y"}) == ["a"]


def test_is_null_filters_at_sql_layer(store: SQLiteStructuredStore) -> None:
    assert store.filter({"note__is_null": True}) == ["a"]
    assert sorted(store.filter({"note__is_null": False})) == ["b", "c"]


def test_missing_document_returns_none(store: SQLiteStructuredStore) -> None:
    assert store.get("missing") is None


def test_empty_query_returns_every_id(store: SQLiteStructuredStore) -> None:
    assert sorted(store.filter({})) == ["a", "b", "c"]


def test_database_persists_across_reopen(tmp_path: Path) -> None:
    path = tmp_path / "ennoia.db"
    first = SQLiteStructuredStore(path)
    first.upsert("doc_1", {"cat": "legal"})
    first.close()

    second = SQLiteStructuredStore(path)
    assert second.get("doc_1") == {"cat": "legal"}
