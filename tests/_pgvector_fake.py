"""In-memory fake for a subset of the asyncpg ``Connection`` interface.

Records every SQL call so tests can assert the emitted shape, and answers
``fetch`` / ``fetchrow`` / ``execute`` from a simple Python dict model so the
adapter's end-to-end flow can be exercised without a live Postgres.

The fake mirrors the row-per-entry schema used by
:class:`~ennoia.store.hybrid.pgvector.PgVectorHybridStore`: one logical row per
``vector_id``, with multiple rows sharing the same ``source_id`` when a
document produced multiple embeddings.
"""

from __future__ import annotations

import json
from typing import Any


class FakeRecord(dict[str, Any]):
    """Row class that mimics asyncpg's ``Record`` dict-access interface."""


class FakeConnection:
    def __init__(self) -> None:
        self.executes: list[tuple[str, tuple[Any, ...]]] = []
        self.fetches: list[tuple[str, tuple[Any, ...]]] = []
        # Rows keyed by vector_id so one source_id may own many rows — matching
        # the row-per-entry schema used by the adapter.
        self.rows: dict[str, FakeRecord] = {}
        self.closed = False
        self._fetch_hooks: list[tuple[str, list[FakeRecord]]] = []
        self._transaction_depth = 0

    async def close(self) -> None:
        self.closed = True

    def transaction(self) -> _FakeTxn:
        return _FakeTxn(self)

    async def execute(self, sql: str, *args: Any) -> str:
        self.executes.append((sql, args))
        stripped = sql.strip().upper()
        if stripped.startswith("INSERT "):
            # Two INSERT shapes exist in the adapter:
            # - No-vectors sentinel: 4 args (vector_id, source_id, index_name, data).
            # - Normal row:          7 args (..., unique_key, text, data, vector).
            if len(args) == 4:
                vector_id, source_id, index_name, data_json = args
                unique_key: str | None = None
                text_val: str | None = None
                vector_literal: str | None = None
            else:
                (
                    vector_id,
                    source_id,
                    index_name,
                    unique_key,
                    text_val,
                    data_json,
                    vector_literal,
                ) = args
            data = json.loads(data_json)
            self.rows[vector_id] = FakeRecord(
                vector_id=vector_id,
                source_id=source_id,
                index_name=index_name,
                unique_key=unique_key,
                text=text_val,
                data=data,
                vector=vector_literal,
            )
            return "INSERT 1"
        if stripped.startswith("DELETE "):
            source_id = args[0]
            to_drop = [vid for vid, row in self.rows.items() if row["source_id"] == source_id]
            for vid in to_drop:
                self.rows.pop(vid, None)
            return f"DELETE {len(to_drop)}"
        return "OK"

    async def fetch(self, sql: str, *args: Any) -> list[FakeRecord]:
        self.fetches.append((sql, args))
        for prefix, rows in self._fetch_hooks:
            if sql.strip().startswith(prefix):
                return rows
        if "SELECT DISTINCT source_id" in sql:
            seen: list[str] = []
            seen_set: set[str] = set()
            for row in self.rows.values():
                sid = row["source_id"]
                if sid in seen_set:
                    continue
                seen.append(sid)
                seen_set.add(sid)
            return [FakeRecord(source_id=sid) for sid in seen]
        return []

    async def fetchrow(self, sql: str, *args: Any) -> FakeRecord | None:
        self.fetches.append((sql, args))
        if "SELECT data FROM" in sql:
            source_id = args[0]
            for row in self.rows.values():
                if row["source_id"] == source_id:
                    return FakeRecord(data=json.dumps(row.get("data", {})))
            return None
        return None

    def prime_fetch(self, prefix: str, rows: list[FakeRecord]) -> None:
        self._fetch_hooks.append((prefix, rows))


class _FakeTxn:
    """Minimal async-context-manager shim for ``conn.transaction()`` usage."""

    def __init__(self, conn: FakeConnection) -> None:
        self._conn = conn

    async def __aenter__(self) -> _FakeTxn:
        self._conn._transaction_depth += 1
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        self._conn._transaction_depth -= 1
