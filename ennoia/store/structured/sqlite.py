"""SQLite-backed structured store.

Persists each document as a single JSON blob; scalar comparisons compile to
``json_extract`` queries for server-side filtering, while list/substring
operators fall back to Python (reusing :func:`apply_filters`). The split
keeps the common path efficient without sacrificing correctness for the
harder operators — a conscious v0.1.0 trade-off; a fully pushed-down
implementation is tracked for a later iteration.

Uses :mod:`aiosqlite` so calls don't block the event loop. ``aiosqlite``
runs the underlying ``sqlite3`` connection on a dedicated background thread
and exposes async ``execute`` / ``commit`` — the connection itself is bound
to the asyncio loop that opened it via the ``run_coroutine_threadsafe``
calls inside the library. The sync ``Pipeline.index`` / ``Pipeline.search``
wrappers create a fresh loop on every call (``asyncio.run``), so we cannot
cache a connection across loops; ``_get_conn`` lazily reopens whenever the
running loop differs from the one that owns the cached connection. The same
invariant governs every async adapter in this package — see
:class:`ennoia.adapters.llm.ollama.OllamaAdapter`.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from ennoia.store.base import StructuredStore, validate_collection_name
from ennoia.utils.filters import (
    apply_filters,
    coerce_filter_value,
    parse_bool,
    split_filter_key,
)

if TYPE_CHECKING:
    import aiosqlite

__all__ = ["SQLiteStructuredStore"]


_SCALAR_OPS = {"eq", "gt", "gte", "lt", "lte", "in", "is_null"}
_SQL_COMPARATOR = {
    "eq": "=",
    "gt": ">",
    "gte": ">=",
    "lt": "<",
    "lte": "<=",
}


class SQLiteStructuredStore(StructuredStore):
    """Structured store backed by a single SQLite file.

    The table layout is deliberately minimal: ``<collection>(source_id PRIMARY KEY, data JSON)``.
    All field semantics are driven by the JSON payload; there is no per-field column.
    Multiple collections can coexist in one database file — use different
    ``collection=`` names for each pipeline.
    """

    def __init__(self, path: str | Path, *, collection: str = "documents") -> None:
        self.path = Path(path)
        self.collection = validate_collection_name(collection)
        self._conn: aiosqlite.Connection | None = None
        self._conn_loop: asyncio.AbstractEventLoop | None = None

    async def _get_conn(self) -> aiosqlite.Connection:
        import aiosqlite

        current = asyncio.get_running_loop()
        if self._conn is not None and self._conn_loop is current:
            return self._conn
        # Different loop (or first use): open a fresh connection bound to it.
        # The previous connection (if any) is leaked rather than awaited-closed
        # because its loop is already gone — closing would raise.
        self._conn = await aiosqlite.connect(self.path)
        self._conn_loop = current
        await self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.collection} (
                source_id TEXT PRIMARY KEY,
                data TEXT NOT NULL
            )
            """
        )
        await self._conn.commit()
        return self._conn

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
            self._conn_loop = None

    async def upsert(self, source_id: str, data: dict[str, Any]) -> None:
        conn = await self._get_conn()
        payload = json.dumps(data, default=str)
        await conn.execute(
            f"""
            INSERT INTO {self.collection}(source_id, data) VALUES (?, ?)
            ON CONFLICT(source_id) DO UPDATE SET data = excluded.data
            """,
            (source_id, payload),
        )
        await conn.commit()

    async def get(self, source_id: str) -> dict[str, Any] | None:
        conn = await self._get_conn()
        async with conn.execute(
            f"SELECT data FROM {self.collection} WHERE source_id = ?",
            (source_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return dict(json.loads(row[0]))

    async def delete(self, source_id: str) -> bool:
        conn = await self._get_conn()
        cursor = await conn.execute(
            f"DELETE FROM {self.collection} WHERE source_id = ?",
            (source_id,),
        )
        deleted = cursor.rowcount > 0
        await cursor.close()
        await conn.commit()
        return deleted

    async def filter(self, query: dict[str, Any]) -> list[str]:
        conn = await self._get_conn()
        if not query:
            async with conn.execute(f"SELECT source_id FROM {self.collection}") as cursor:
                return [row[0] async for row in cursor]

        sql_conditions: list[tuple[str, list[Any]]] = []
        python_conditions: dict[str, Any] = {}

        for key, value in query.items():
            field, op = split_filter_key(key)
            if op not in _SCALAR_OPS:
                python_conditions[key] = value
                continue

            sql_conditions.append(_compile_scalar(field, op, value))

        if sql_conditions:
            where = " AND ".join(clause for clause, _ in sql_conditions)
            params: list[Any] = [p for _, bag in sql_conditions for p in bag]
            sql = f"SELECT source_id, data FROM {self.collection} WHERE {where}"
        else:
            sql = f"SELECT source_id, data FROM {self.collection}"
            params = []

        async with conn.execute(sql, params) as cursor:
            rows = [row async for row in cursor]
        if not python_conditions:
            return [row[0] for row in rows]

        decoded = [(row[0], dict(json.loads(row[1]))) for row in rows]
        return apply_filters(decoded, python_conditions)


def _compile_scalar(field: str, op: str, value: Any) -> tuple[str, list[Any]]:
    """Compile a scalar condition to a SQL fragment + parameter list.

    Caller restricts ``op`` to :data:`_SCALAR_OPS`, so every branch below is
    exhaustive — every scalar operator maps to either ``is_null``, ``in``, or a
    comparator in :data:`_SQL_COMPARATOR`.
    """
    json_path = f"json_extract(data, '$.{field}')"

    if op == "is_null":
        expected_null = parse_bool(value)
        return (
            f"({json_path} IS NULL)" if expected_null else f"({json_path} IS NOT NULL)",
            [],
        )

    if op == "in":
        candidates: list[Any]
        if isinstance(value, (list, tuple)):
            candidates = list(cast("list[Any] | tuple[Any, ...]", value))
        else:
            candidates = [v.strip() for v in str(value).split(",")]
        if not candidates:
            return ("0 = 1", [])
        placeholders = ",".join("?" * len(candidates))
        return (f"{json_path} IN ({placeholders})", candidates)

    comparator = _SQL_COMPARATOR[op]

    # SQLite's json_extract returns JSON-typed primitives; string/number
    # comparisons work natively. Dates arrive as ISO strings on both sides
    # because :meth:`upsert` serialises via ``json.dumps(default=str)``.
    _, coerced = coerce_filter_value(value, value)
    return (f"{json_path} {comparator} ?", [coerced])
