"""SQLite structured store — scalar SQL path + Python-fallback for list/substring ops."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from ennoia.store.structured.sqlite import SQLiteStructuredStore


@pytest.fixture()
async def store(tmp_path: Path) -> SQLiteStructuredStore:
    s = SQLiteStructuredStore(tmp_path / "ennoia.db")
    await s.upsert("a", {"cat": "legal", "n": 5, "tags": ["x", "y"], "note": None})
    await s.upsert("b", {"cat": "medical", "n": 3, "tags": ["y"], "note": "hi"})
    await s.upsert("c", {"cat": "legal", "n": 8, "tags": ["x"], "note": "hello"})
    return s


async def test_upsert_overwrites_existing_row(store: SQLiteStructuredStore) -> None:
    await store.upsert("a", {"cat": "financial", "n": 1, "tags": []})
    assert await store.get("a") == {"cat": "financial", "n": 1, "tags": []}


async def test_scalar_filters_use_sql_path(store: SQLiteStructuredStore) -> None:
    assert sorted(await store.filter({"cat": "legal"})) == ["a", "c"]
    assert await store.filter({"n__gt": 4}) == ["a", "c"]
    assert sorted(await store.filter({"cat__in": ["legal", "medical"]})) == ["a", "b", "c"]


async def test_list_operator_uses_python_fallback(store: SQLiteStructuredStore) -> None:
    assert sorted(await store.filter({"tags__contains": "x"})) == ["a", "c"]
    assert await store.filter({"tags__contains_all": "x,y"}) == ["a"]


async def test_is_null_filters_at_sql_layer(store: SQLiteStructuredStore) -> None:
    assert await store.filter({"note__is_null": True}) == ["a"]
    assert sorted(await store.filter({"note__is_null": False})) == ["b", "c"]


async def test_missing_document_returns_none(store: SQLiteStructuredStore) -> None:
    assert await store.get("missing") is None


async def test_empty_query_returns_every_id(store: SQLiteStructuredStore) -> None:
    assert sorted(await store.filter({})) == ["a", "b", "c"]


async def test_database_persists_across_reopen(tmp_path: Path) -> None:
    path = tmp_path / "ennoia.db"
    first = SQLiteStructuredStore(path)
    await first.upsert("doc_1", {"cat": "legal"})
    await first.close()

    second = SQLiteStructuredStore(path)
    assert await second.get("doc_1") == {"cat": "legal"}
    await second.close()


async def test_in_filter_comma_string_is_split(store: SQLiteStructuredStore) -> None:
    # ``field__in`` with a single comma-separated string is accepted for CLI
    # ergonomics — exercises the non-list branch of ``_compile_scalar``.
    assert sorted(await store.filter({"cat__in": "legal,medical"})) == ["a", "b", "c"]


async def test_in_filter_empty_list_yields_no_rows(store: SQLiteStructuredStore) -> None:
    # Empty candidates must compile to an always-false SQL predicate.
    assert await store.filter({"cat__in": []}) == []


async def test_close_on_fresh_store_is_noop(tmp_path: Path) -> None:
    """``close`` on a store that never opened a connection must not raise."""
    s = SQLiteStructuredStore(tmp_path / "fresh.db")
    # ``_conn`` is lazy; no I/O has happened yet — close must short-circuit.
    await s.close()


async def test_collections_are_isolated_in_one_file(tmp_path: Path) -> None:
    # Two stores pointing at the same SQLite file with different
    # collections must not see each other's rows — each collection is
    # its own SQL table.
    path = tmp_path / "shared.db"
    invoices = SQLiteStructuredStore(path, collection="invoices")
    emails = SQLiteStructuredStore(path, collection="emails")
    await invoices.upsert("doc_1", {"amount": 100})
    await emails.upsert("doc_1", {"subject": "hi"})
    assert await invoices.get("doc_1") == {"amount": 100}
    assert await emails.get("doc_1") == {"subject": "hi"}
    assert await invoices.filter({}) == ["doc_1"]
    assert await emails.filter({}) == ["doc_1"]
    await invoices.close()
    await emails.close()


def test_invalid_collection_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="collection"):
        SQLiteStructuredStore(tmp_path / "x.db", collection="bad name!")


def test_survives_multiple_event_loops(tmp_path: Path) -> None:
    """Two ``asyncio.run`` calls against one store must both succeed.

    aiosqlite connections bind to the loop that opened them; the sync
    Pipeline API creates a fresh loop per call. ``_get_conn`` lazily
    reopens whenever the running loop differs — regressing that behaviour
    reintroduces ``RuntimeError: Event loop is closed`` on the second call.
    """
    path = tmp_path / "loops.db"
    s = SQLiteStructuredStore(path)
    asyncio.run(s.upsert("a", {"cat": "legal"}))
    # Fresh event loop — the cached connection from the first loop is dead.
    assert asyncio.run(s.get("a")) == {"cat": "legal"}
    asyncio.run(s.upsert("b", {"cat": "medical"}))
    assert sorted(asyncio.run(s.filter({}))) == ["a", "b"]
