"""In-memory fake for a subset of the asyncpg ``Connection`` interface.

Records every SQL call so tests can assert the emitted shape. Answers
``fetch`` / ``fetchrow`` / ``execute`` from a simple Python dict model so the
adapter's end-to-end flow can be exercised without a live Postgres.
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
        # Shallow model: one dict of rows keyed by source_id.
        self.rows: dict[str, FakeRecord] = {}
        self.closed = False
        # Test hooks: callers can pre-programme responses for specific SQL prefixes.
        self._fetch_hooks: list[tuple[str, list[FakeRecord]]] = []

    async def close(self) -> None:
        self.closed = True

    async def execute(self, sql: str, *args: Any) -> str:
        self.executes.append((sql, args))
        stripped = sql.strip().upper()
        if stripped.startswith("INSERT "):
            source_id, data_json = args[0], args[1]
            data = json.loads(data_json)
            row = self.rows.setdefault(source_id, FakeRecord(source_id=source_id))
            row["data"] = data
            # Any vector arg positions after $2 are named after the columns in
            # the SQL — we don't model vectors, just record the upsert happened.
            return "INSERT 1"
        if stripped.startswith("DELETE "):
            source_id = args[0]
            existed = source_id in self.rows
            self.rows.pop(source_id, None)
            return f"DELETE {1 if existed else 0}"
        return "OK"

    async def fetch(self, sql: str, *args: Any) -> list[FakeRecord]:
        self.fetches.append((sql, args))
        for prefix, rows in self._fetch_hooks:
            if sql.strip().startswith(prefix):
                return rows
        return []

    async def fetchrow(self, sql: str, *args: Any) -> FakeRecord | None:
        self.fetches.append((sql, args))
        if "SELECT data FROM" in sql:
            source_id = args[0]
            row = self.rows.get(source_id)
            if row is None:
                return None
            return FakeRecord(data=json.dumps(row.get("data", {})))
        return None

    def prime_fetch(self, prefix: str, rows: list[FakeRecord]) -> None:
        self._fetch_hooks.append((prefix, rows))
