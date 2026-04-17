"""PgVectorHybridStore unit tests against an in-memory fake asyncpg Connection."""

from __future__ import annotations

import pytest

from ennoia.store.base import VectorEntry
from ennoia.store.hybrid.pgvector import PgVectorHybridStore
from tests._pgvector_fake import FakeConnection, FakeRecord


def _store(connection: FakeConnection | None = None) -> PgVectorHybridStore:
    return PgVectorHybridStore(
        dsn="postgresql://ignored",
        connection=(connection or FakeConnection()),  # type: ignore[arg-type]
    )


def _entries(*specs: tuple[str, list[float], str, str | None]) -> list[VectorEntry]:
    return [
        VectorEntry(index_name=name, vector=vec, text=text, unique=unique)
        for name, vec, text, unique in specs
    ]


def test_invalid_collection_raises() -> None:
    with pytest.raises(ValueError, match="collection"):
        PgVectorHybridStore(dsn="postgresql://x", collection="bad name!")


async def test_upsert_creates_extension_table_with_vector_column() -> None:
    conn = FakeConnection()
    store = _store(conn)
    await store.upsert(
        "doc_1",
        {"cat": "legal"},
        _entries(
            ("Holding", [1.0, 0.0, 0.0], "holding-text", None),
            ("Facts", [0.0, 1.0, 0.0], "facts-text", None),
        ),
    )
    executes = [sql for sql, _ in conn.executes]
    assert any("CREATE EXTENSION IF NOT EXISTS vector" in e for e in executes)
    table_sql = next(e for e in executes if e.startswith("CREATE TABLE"))
    # Row-per-entry layout with a single vector column sized by the first sample.
    assert "vector_id   TEXT PRIMARY KEY" in table_sql
    assert "index_name  TEXT NOT NULL" in table_sql
    assert "vector vector(3)" in table_sql


async def test_collection_name_threads_through_all_sql() -> None:
    # Two pgvector stores with different collections must not cross-talk.
    conn_a = FakeConnection()
    conn_b = FakeConnection()
    store_a = PgVectorHybridStore(
        dsn="postgresql://x",
        collection="invoices",
        connection=conn_a,  # type: ignore[arg-type]
    )
    store_b = PgVectorHybridStore(
        dsn="postgresql://x",
        collection="emails",
        connection=conn_b,  # type: ignore[arg-type]
    )
    await store_a.upsert("doc_1", {"cat": "A"}, _entries(("H", [1.0, 0.0], "h", None)))
    await store_b.upsert("doc_1", {"cat": "B"}, _entries(("H", [0.0, 1.0], "h", None)))
    assert any("CREATE TABLE IF NOT EXISTS invoices" in sql for sql, _ in conn_a.executes)
    assert any("CREATE TABLE IF NOT EXISTS emails" in sql for sql, _ in conn_b.executes)
    assert not any("emails" in sql for sql, _ in conn_a.executes)
    assert not any("invoices" in sql for sql, _ in conn_b.executes)


async def test_upsert_replaces_prior_rows_for_source() -> None:
    # Re-indexing a document must drop any stale rows first so N → M cardinality
    # changes (collection shrinking, semantic answer going empty) are correct.
    conn = FakeConnection()
    store = _store(conn)
    await store.upsert(
        "doc_1",
        {"cat": "legal"},
        _entries(
            ("Parties", [1.0, 0.0], "Alice", "alice"),
            ("Parties", [0.0, 1.0], "Bob", "bob"),
        ),
    )
    # Re-index with just one entity.
    await store.upsert(
        "doc_1",
        {"cat": "legal"},
        _entries(("Parties", [1.0, 0.0], "Alice", "alice")),
    )
    # Two DELETE-by-source statements fired (one per upsert); final row count is 1.
    deletes = [sql for sql, _ in conn.executes if "DELETE FROM" in sql]
    assert len(deletes) == 2
    assert len(conn.rows) == 1


async def test_upsert_insert_shape_row_per_entry() -> None:
    conn = FakeConnection()
    store = _store(conn)
    await store.upsert(
        "doc_1",
        {"cat": "legal"},
        _entries(("Parties", [1.0, 0.0], "Alice", "hash_alice")),
    )
    insert_sql, insert_params = next(
        (sql, params) for sql, params in conn.executes if "INSERT INTO" in sql and len(params) == 7
    )
    # 7 params: vector_id, source_id, index_name, unique_key, text, data, vector.
    assert "(vector_id, source_id, index_name, unique_key, text, data, vector)" in insert_sql
    assert insert_params[0] == "doc_1:Parties:hash_alice"
    assert insert_params[1] == "doc_1"
    assert insert_params[2] == "Parties"
    assert insert_params[3] == "hash_alice"
    assert insert_params[4] == "Alice"
    assert insert_params[6] == "[1.0, 0.0]"


async def test_hybrid_search_issues_distance_ordered_select() -> None:
    conn = FakeConnection()
    conn.prime_fetch(
        "SELECT vector_id, source_id",
        [
            FakeRecord(
                vector_id="doc_1:H",
                source_id="doc_1",
                index_name="H",
                unique_key=None,
                text="hello",
                data={"cat": "legal"},
                distance=0.1,
            ),
            FakeRecord(
                vector_id="doc_2:H",
                source_id="doc_2",
                index_name="H",
                unique_key=None,
                text="world",
                data={"cat": "legal"},
                distance=0.5,
            ),
        ],
    )
    store = _store(conn)
    await store.upsert(
        "doc_1",
        {"cat": "legal"},
        _entries(("H", [1.0, 0.0], "hello", None)),
    )

    hits = await store.hybrid_search({"cat": "legal"}, [1.0, 0.0], top_k=5)
    # Distance 0.1 → score 0.9.
    assert hits[0][0] == "doc_1:H"
    assert hits[0][1] == pytest.approx(0.9)
    assert hits[0][2]["text"] == "hello"
    assert hits[0][2]["index"] == "H"
    search_sql = next(sql for sql, _ in conn.fetches if "SELECT vector_id" in sql)
    assert "<=>" in search_sql
    assert "ORDER BY" in search_sql
    assert "LIMIT" in search_sql


async def test_hybrid_search_restricted_by_index_name() -> None:
    conn = FakeConnection()
    conn.prime_fetch("SELECT vector_id, source_id", [])
    store = _store(conn)
    await store.upsert(
        "doc_1",
        {"cat": "legal"},
        _entries(("Parties", [1.0, 0.0], "Alice", "a")),
    )

    await store.hybrid_search({}, [1.0, 0.0], top_k=5, index="Parties")
    search_sql, params = next((sql, p) for sql, p in conn.fetches if "SELECT vector_id" in sql)
    assert "index_name = " in search_sql
    assert "Parties" in params


async def test_hybrid_search_without_any_upsert_returns_empty() -> None:
    store = _store()
    hits = await store.hybrid_search({}, [1.0, 0.0], top_k=5)
    assert hits == []


async def test_filter_translates_and_returns_distinct_source_ids() -> None:
    conn = FakeConnection()
    conn.prime_fetch(
        "SELECT DISTINCT source_id",
        [FakeRecord(source_id="doc_1"), FakeRecord(source_id="doc_2")],
    )
    store = _store(conn)
    # Touch the store so ``_table_created`` becomes True; otherwise filter short-circuits.
    await store.upsert("doc_1", {"cat": "legal"}, _entries(("H", [1.0, 0.0], "h", None)))
    ids = await store.filter({"cat": "legal"})
    assert ids == ["doc_1", "doc_2"]


async def test_get_returns_payload_or_none() -> None:
    conn = FakeConnection()
    store = _store(conn)
    await store.upsert("doc_1", {"cat": "legal"}, _entries(("H", [1.0, 0.0], "h", None)))
    data = await store.get("doc_1")
    assert data == {"cat": "legal"}
    assert await store.get("missing") is None


async def test_delete_returns_true_when_rows_existed() -> None:
    conn = FakeConnection()
    store = _store(conn)
    await store.upsert(
        "doc_1",
        {"cat": "legal"},
        _entries(
            ("H", [1.0, 0.0], "h", None),
            ("P", [0.0, 1.0], "p", "u1"),
        ),
    )
    assert await store.delete("doc_1") is True
    assert await store.delete("doc_1") is False


async def test_close_closes_connection() -> None:
    conn = FakeConnection()
    store = _store(conn)
    await store.close()
    assert conn.closed is True
    # Second close is a no-op.
    await store.close()


async def test_lazy_connection_from_dsn(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncpg  # pyright: ignore[reportMissingImports]
    import pgvector.asyncpg  # pyright: ignore[reportMissingImports]

    conn = FakeConnection()

    async def _connect(_dsn: str) -> FakeConnection:
        return conn

    async def _register(_conn: FakeConnection) -> None:
        pass

    monkeypatch.setattr(asyncpg, "connect", _connect)
    monkeypatch.setattr(pgvector.asyncpg, "register_vector", _register)
    store = PgVectorHybridStore(dsn="postgresql://fake")
    await store.upsert("doc_1", {"cat": "legal"}, _entries(("H", [1.0, 0.0], "h", None)))
    assert conn.executes


async def test_empty_entries_still_persists_structural_row() -> None:
    # A document with no semantics / no collection entries still needs to be
    # queryable via ``filter``/``get``. The adapter inserts a sentinel row.
    conn = FakeConnection()
    store = _store(conn)
    await store.upsert("doc_1", {"cat": "legal"}, [])
    assert await store.get("doc_1") == {"cat": "legal"}


async def test_uncreated_table_short_circuits_read_methods() -> None:
    # All read methods guard against being called before any upsert has
    # materialised the table — returning empty / None instead of issuing SQL.
    store = _store()
    assert await store.filter({}) == []
    assert await store.get("doc_1") is None
    assert await store.delete("doc_1") is False


async def test_alter_table_adds_vector_column_when_first_upsert_was_empty() -> None:
    # The first upsert had no entries (sentinel row, no vector column), so the
    # second upsert with vectors must issue ``ALTER TABLE ... ADD COLUMN vector``.
    conn = FakeConnection()
    store = _store(conn)
    await store.upsert("doc_1", {"cat": "legal"}, [])
    await store.upsert(
        "doc_2",
        {"cat": "medical"},
        _entries(("H", [1.0, 0.0], "h", None)),
    )
    alters = [sql for sql, _ in conn.executes if "ALTER TABLE" in sql]
    assert alters
    assert "vector vector(2)" in alters[0]


async def test_hybrid_search_hydrates_unique_key_when_present() -> None:
    conn = FakeConnection()
    conn.prime_fetch(
        "SELECT vector_id, source_id",
        [
            FakeRecord(
                vector_id="doc_1:Parties:alice",
                source_id="doc_1",
                index_name="Parties",
                unique_key="alice",
                text="Alice",
                data={"cat": "legal"},
                distance=0.1,
            ),
        ],
    )
    store = _store(conn)
    await store.upsert(
        "doc_1",
        {"cat": "legal"},
        _entries(("Parties", [1.0, 0.0], "Alice", "alice")),
    )
    hits = await store.hybrid_search({"cat": "legal"}, [1.0, 0.0], top_k=5)
    assert hits[0][2]["unique"] == "alice"
