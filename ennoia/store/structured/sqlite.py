"""SQLite-backed structured store.

Persists each document as a single JSON blob; scalar comparisons compile to
``json_extract`` queries for server-side filtering, while list/substring
operators fall back to Python (reusing :func:`apply_filters`). The split
keeps the common path efficient without sacrificing correctness for the
harder operators — a conscious v0.1.0 trade-off; a fully pushed-down
implementation is tracked for a later iteration.

Depends only on stdlib ``sqlite3``; no optional extra is required.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, cast

from ennoia.store.base import StructuredStore
from ennoia.utils.filters import (
    KNOWN_OPERATORS,
    apply_filters,
    coerce_filter_value,
    parse_bool,
    split_filter_key,
)

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

    The table layout is deliberately minimal: ``documents(source_id PRIMARY KEY, data JSON)``.
    All field semantics are driven by the JSON payload; there is no per-field column.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                source_id TEXT PRIMARY KEY,
                data TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def upsert(self, source_id: str, data: dict[str, Any]) -> None:
        payload = json.dumps(data, default=str)
        self._conn.execute(
            """
            INSERT INTO documents(source_id, data) VALUES (?, ?)
            ON CONFLICT(source_id) DO UPDATE SET data = excluded.data
            """,
            (source_id, payload),
        )
        self._conn.commit()

    def get(self, source_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT data FROM documents WHERE source_id = ?",
            (source_id,),
        ).fetchone()
        if row is None:
            return None
        return dict(json.loads(row[0]))

    def filter(self, query: dict[str, Any]) -> list[str]:
        if not query:
            return [row[0] for row in self._conn.execute("SELECT source_id FROM documents")]

        sql_conditions: list[tuple[str, list[Any]]] = []
        python_conditions: dict[str, Any] = {}

        for key, value in query.items():
            field, op = split_filter_key(key)
            if op not in KNOWN_OPERATORS:
                raise ValueError(f"Unknown filter operator: {op}")
            if op not in _SCALAR_OPS:
                python_conditions[key] = value
                continue

            condition = _compile_scalar(field, op, value)
            if condition is None:
                python_conditions[key] = value
            else:
                sql_conditions.append(condition)

        if sql_conditions:
            where = " AND ".join(clause for clause, _ in sql_conditions)
            params: list[Any] = [p for _, bag in sql_conditions for p in bag]
            sql = f"SELECT source_id, data FROM documents WHERE {where}"
        else:
            sql = "SELECT source_id, data FROM documents"
            params = []

        rows = list(self._conn.execute(sql, params))
        if not python_conditions:
            return [row[0] for row in rows]

        decoded = [(row[0], dict(json.loads(row[1]))) for row in rows]
        return apply_filters(decoded, python_conditions)


def _compile_scalar(field: str, op: str, value: Any) -> tuple[str, list[Any]] | None:
    """Compile a scalar condition to a SQL fragment + parameter list.

    Returns ``None`` when the condition is better handled in Python (e.g. a
    date-typed value that needs to be compared as a string lexicographically —
    :func:`coerce_filter_value` does that in-memory).
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

    comparator = _SQL_COMPARATOR.get(op)
    if comparator is None:
        return None

    # SQLite's json_extract returns JSON-typed primitives; string/number
    # comparisons work natively. Dates arrive as ISO strings on both sides
    # because :meth:`upsert` serialises via ``json.dumps(default=str)``.
    _, coerced = coerce_filter_value(value, value)
    return (f"{json_path} {comparator} ?", [coerced])
