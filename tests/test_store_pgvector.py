"""PgVectorHybridStore unit tests against an in-memory fake asyncpg Connection."""

from __future__ import annotations

import pytest

from ennoia.store.hybrid.pgvector import PgVectorHybridStore
from tests._pgvector_fake import FakeConnection, FakeRecord


def _store(connection: FakeConnection | None = None) -> PgVectorHybridStore:
    return PgVectorHybridStore(
        dsn="postgresql://ignored",
        connection=(connection or FakeConnection()),  # type: ignore[arg-type]
    )


def test_invalid_collection_raises() -> None:
    with pytest.raises(ValueError, match="collection"):
        PgVectorHybridStore(dsn="postgresql://x", collection="bad name!")


async def test_upsert_creates_extension_table_and_columns() -> None:
    conn = FakeConnection()
    store = _store(conn)
    await store.upsert(
        "doc_1",
        {"cat": "legal"},
        {"Holding": [1.0, 0.0, 0.0], "Facts": [0.0, 1.0, 0.0]},
    )
    executes = [sql for sql, _ in conn.executes]
    assert any("CREATE EXTENSION IF NOT EXISTS vector" in e for e in executes)
    assert any("CREATE TABLE IF NOT EXISTS documents" in e for e in executes)
    # Table creation SQL contains both vector columns.
    table_sql = next(e for e in executes if e.startswith("CREATE TABLE"))
    assert '"Holding_vector" vector(3)' in table_sql
    assert '"Facts_vector" vector(3)' in table_sql


async def test_collection_name_threads_through_all_sql() -> None:
    # Two pgvector stores with different collections must not cross-talk in SQL.
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
    await store_a.upsert("doc_1", {"cat": "A"}, {"H": [1.0, 0.0]})
    await store_b.upsert("doc_1", {"cat": "B"}, {"H": [0.0, 1.0]})
    assert any("CREATE TABLE IF NOT EXISTS invoices" in sql for sql, _ in conn_a.executes)
    assert any("CREATE TABLE IF NOT EXISTS emails" in sql for sql, _ in conn_b.executes)
    assert not any("emails" in sql for sql, _ in conn_a.executes)
    assert not any("invoices" in sql for sql, _ in conn_b.executes)


async def test_upsert_with_new_index_alters_table() -> None:
    conn = FakeConnection()
    store = _store(conn)
    await store.upsert("doc_1", {"cat": "legal"}, {"Holding": [1.0, 0.0]})
    # Second upsert introduces a new index — must issue ALTER TABLE ADD COLUMN.
    await store.upsert("doc_2", {"cat": "medical"}, {"Facts": [0.0, 1.0]})
    alters = [e for e, _ in conn.executes if "ALTER TABLE" in e]
    assert alters and '"Facts_vector"' in alters[0]


async def test_upsert_insert_shape() -> None:
    conn = FakeConnection()
    store = _store(conn)
    await store.upsert("doc_1", {"cat": "legal"}, {"H": [1.0, 0.0]})
    insert_sql, insert_params = next(
        (sql, params) for sql, params in conn.executes if "INSERT INTO" in sql
    )
    assert "ON CONFLICT (source_id) DO UPDATE SET" in insert_sql
    # Params: source_id, json data, vector literal.
    assert insert_params[0] == "doc_1"
    assert insert_params[2] == "[1.0, 0.0]"


async def test_hybrid_search_issues_distance_ordered_select() -> None:
    conn = FakeConnection()
    conn.prime_fetch(
        "SELECT source_id, data",
        [
            FakeRecord(source_id="doc_1", data={"cat": "legal"}, distance=0.1),
            FakeRecord(source_id="doc_2", data={"cat": "legal"}, distance=0.5),
        ],
    )
    store = _store(conn)
    await store.upsert("doc_1", {"cat": "legal"}, {"H": [1.0, 0.0]})

    hits = await store.hybrid_search({"cat": "legal"}, [1.0, 0.0], top_k=5)
    # Distance 0.1 -> score 0.9; distance 0.5 -> score 0.5.
    assert hits[0][0] == "doc_1:H"
    assert hits[0][1] == pytest.approx(0.9)
    # SQL contains the cosine-distance operator and ORDER BY.
    search_sql = next(sql for sql, _ in conn.fetches if "SELECT source_id, data" in sql)
    assert "<=>" in search_sql
    assert "ORDER BY" in search_sql
    assert "LIMIT" in search_sql


async def test_hybrid_search_without_any_index_returns_empty() -> None:
    # Fresh store, no upserts yet — no known index to search against.
    store = _store()
    hits = await store.hybrid_search({}, [1.0, 0.0], top_k=5)
    assert hits == []


async def test_filter_translates_and_returns_source_ids() -> None:
    conn = FakeConnection()
    conn.prime_fetch(
        "SELECT source_id FROM",
        [FakeRecord(source_id="doc_1"), FakeRecord(source_id="doc_2")],
    )
    store = _store(conn)
    ids = await store.filter({"cat": "legal"})
    assert ids == ["doc_1", "doc_2"]
    assert any("WHERE" in sql for sql, _ in conn.fetches)


async def test_get_returns_payload_or_none() -> None:
    conn = FakeConnection()
    store = _store(conn)
    await store.upsert("doc_1", {"cat": "legal"}, {"H": [1.0, 0.0]})
    data = await store.get("doc_1")
    assert data == {"cat": "legal"}
    assert await store.get("missing") is None


async def test_delete_returns_true_when_row_existed() -> None:
    conn = FakeConnection()
    store = _store(conn)
    await store.upsert("doc_1", {"cat": "legal"}, {"H": [1.0, 0.0]})
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
    await store.upsert("doc_1", {"cat": "legal"}, {"H": [1.0, 0.0]})
    assert conn.executes  # connection was used


async def test_validate_index_rejects_non_identifier() -> None:
    # The adapter sanitises semantic index names — a class named with a
    # problematic character should raise instead of emitting bad SQL.
    conn = FakeConnection()
    store = _store(conn)
    with pytest.raises(ValueError, match="safe SQL identifier"):
        await store.upsert("doc_1", {"cat": "legal"}, {"bad name!": [1.0]})
