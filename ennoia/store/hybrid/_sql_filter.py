"""Translate Ennoia's unified filter syntax to a parameterised SQL WHERE clause.

Assumes the structured payload lives in a ``jsonb`` column named ``data``.
Produces positional parameters (``$1``, ``$2``, ...) compatible with asyncpg.

Operators cover the full 11-operator spec; there is no residual.

- ``eq``, ``in`` — string-coerced equality and membership via ``data->>'field'``.
- ``gt/gte/lt/lte`` — numeric comparisons use ``(data->>'field')::numeric``;
  date/datetime comparisons use ``(data->>'field')::timestamp``.
- ``contains`` — substring on strings (``ILIKE`` → positional param with ``%`` wrap);
  element-of on lists (``data->'field' ? $n`` — jsonb "contains key" operator).
- ``startswith`` — anchored ``LIKE`` with trailing ``%``.
- ``contains_all`` / ``contains_any`` — jsonb ``@>`` (contains) and ``?|``
  (any-of-keys) for list values.
- ``is_null`` — ``data->>'field'`` IS NULL; negated for the ``false`` form.

Keeping the SQL narrow to a single jsonb column avoids a schema migration
per new field — the Ennoia pipeline already persists everything as JSON.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

from ennoia.utils.filters import parse_bool, split_filter_key

__all__ = ["build_where"]


def build_where(filters: dict[str, Any] | None) -> tuple[str, list[Any]]:
    """Compile ``filters`` to ``(where_sql, params)``.

    ``where_sql`` is the body of a ``WHERE`` clause (no leading keyword). When
    ``filters`` is empty the returned SQL is ``"TRUE"`` so callers can
    unconditionally wrap it.
    """
    if not filters:
        return "TRUE", []

    clauses: list[str] = []
    params: list[Any] = []

    def take(value: Any) -> str:
        """Append ``value`` to params and return the matching ``$N`` placeholder."""
        params.append(value)
        return f"${len(params)}"

    for key, value in filters.items():
        field, op = split_filter_key(key)
        ident = _sql_ident(field)

        if op == "eq":
            clauses.append(f"(data->>'{ident}') = {take(str(value))}::text")
        elif op == "in":
            values = _as_list(value)
            if not values:
                clauses.append("FALSE")
                continue
            placeholders = [f"{take(str(v))}::text" for v in values]
            clauses.append(f"(data->>'{ident}') IN ({', '.join(placeholders)})")
        elif op in {"gt", "gte", "lt", "lte"}:
            comparator = {"gt": ">", "gte": ">=", "lt": "<", "lte": "<="}[op]
            cast, coerced = _sql_cast_for(value)
            clauses.append(f"((data->>'{ident}'){cast}) {comparator} {take(coerced)}{cast}")
        elif op == "contains":
            text_ph = take(str(value))
            like_ph = take(f"%{value}%")
            clauses.append(
                f"(jsonb_typeof(data->'{ident}') = 'array'"
                f" AND data->'{ident}' ? {text_ph})"
                f" OR (data->>'{ident}') ILIKE {like_ph}"
            )
        elif op == "startswith":
            clauses.append(f"(data->>'{ident}') LIKE {take(f'{value}%')}")
        elif op == "contains_all":
            values = _as_list(value)
            if not values:
                clauses.append("TRUE")
                continue
            clauses.append(f"data->'{ident}' @> {take(json.dumps(values))}::jsonb")
        elif op == "contains_any":
            values = _as_list(value)
            if not values:
                clauses.append("FALSE")
                continue
            clauses.append(f"data->'{ident}' ?| {take([str(v) for v in values])}::text[]")
        elif op == "is_null":
            expect_null = parse_bool(value)
            null_expr = f"(data->>'{ident}') IS NULL"
            clauses.append(null_expr if expect_null else f"NOT ({null_expr})")
        else:  # pragma: no cover — split_filter_key only emits known ops
            raise ValueError(f"Unknown operator: {op}")

    return " AND ".join(f"({c})" for c in clauses), params


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return list(value)  # type: ignore[arg-type]
    if isinstance(value, tuple):
        return list(value)  # type: ignore[arg-type]
    if isinstance(value, str):
        return [v.strip() for v in value.split(",")]
    return [value]


def _sql_cast_for(value: Any) -> tuple[str, Any]:
    """Decide whether to compare as numeric or timestamp and coerce the value."""
    if isinstance(value, bool):
        return "::int", 1 if value else 0
    if isinstance(value, (int, float)):
        return "::numeric", value
    if isinstance(value, datetime):
        return "::timestamp", value.isoformat()
    if isinstance(value, date):
        return "::date", value.isoformat()
    if isinstance(value, str):
        # Try ISO parsing first — DB schemas with ``date`` fields store ISO strings.
        try:
            datetime.fromisoformat(value)
            return "::timestamp", value
        except ValueError:
            pass
        try:
            # Plain number as string.
            float(value)
            return "::numeric", value
        except ValueError:
            return "::text", value
    return "::text", str(value)


def _sql_ident(name: str) -> str:
    """Escape a jsonb key name for use inside single quotes.

    Field names come from Pydantic schemas (validated identifiers), so the
    only character that could break quoting is a single quote itself — which
    is not a valid Python identifier. We still double-escape defensively.
    """
    return name.replace("'", "''")
