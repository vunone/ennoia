"""Filter parsing and evaluation shared by structured store backends.

The operator set and the ``field__op`` key convention are canonical per
``docs/filters.md``. In-memory and other non-SQL structured stores can implement
``filter()`` by delegating to :func:`apply_filters`; SQL/engine-native stores
translate these primitives into their own query language instead.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, datetime
from typing import Any, cast

__all__ = [
    "KNOWN_OPERATORS",
    "apply_filters",
    "coerce_filter_value",
    "evaluate_condition",
    "parse_bool",
    "split_filter_key",
]


KNOWN_OPERATORS: frozenset[str] = frozenset(
    {
        "eq",
        "in",
        "gt",
        "gte",
        "lt",
        "lte",
        "contains",
        "startswith",
        "contains_all",
        "contains_any",
        "is_null",
    }
)


def split_filter_key(key: str) -> tuple[str, str]:
    """Return ``(field, operator)``. Operator defaults to ``eq`` when no known suffix is present.

    Raises ``ValueError`` when the resulting field name would be empty
    (``""`` or ``"__<op>"``) — fail fast rather than silently matching nothing.
    """
    # Match the longest suffix first so ``contains_all`` wins over ``all``.
    for op in sorted(KNOWN_OPERATORS, key=len, reverse=True):
        suffix = f"__{op}"
        if key.endswith(suffix):
            base = key[: -len(suffix)]
            if not base:
                raise ValueError(f"Empty field name in filter key: {key!r}")
            return base, op
    if not key:
        raise ValueError(f"Empty field name in filter key: {key!r}")
    return key, "eq"


def coerce_filter_value(field_value: Any, filter_value: Any) -> tuple[Any, Any]:
    """Coerce ``filter_value`` to match ``field_value``'s type for fair comparison.

    CLI ``--filter`` values arrive as strings; this helper promotes them to the
    record field's native type so operators like ``gt``/``lt`` don't raise
    ``TypeError`` when comparing e.g. ``float`` with ``str``. Branch order
    matters: ``bool`` before ``int`` (``isinstance(True, int)`` is ``True``),
    ``datetime`` before ``date`` (``datetime`` subclasses ``date``).
    """
    if not isinstance(filter_value, str):
        return field_value, filter_value
    if isinstance(field_value, bool):
        filter_value = parse_bool(filter_value)
    elif isinstance(field_value, datetime):
        filter_value = datetime.fromisoformat(filter_value)
    elif isinstance(field_value, date):
        filter_value = date.fromisoformat(filter_value)
    elif isinstance(field_value, int):
        filter_value = int(filter_value)
    elif isinstance(field_value, float):
        filter_value = float(filter_value)
    return field_value, filter_value


def parse_bool(value: Any) -> bool:
    """Parse a boolean value from CLI/JSON-shaped inputs."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse boolean from {value!r}")


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, str):
        return [v.strip() for v in value.split(",")]
    if isinstance(value, (bytes, bytearray, Mapping)) or not isinstance(value, Iterable):
        raise ValueError(
            f"Filter requires an iterable value, got {type(cast(Any, value)).__name__}"
        )
    return list(cast(Iterable[Any], value))


def evaluate_condition(record: dict[str, Any], field: str, op: str, value: Any) -> bool:
    """Evaluate a single filter condition against a record."""
    if op == "is_null":
        # ``is_null`` is the only operator that meaningfully fires on absent fields.
        expected_null = parse_bool(value)
        present = field in record and record[field] is not None
        return present is not expected_null

    if field not in record:
        return False
    record_value = record[field]

    if op == "in":
        candidates = _as_list(value)
        return any(record_value == coerce_filter_value(record_value, c)[1] for c in candidates)

    if op in {"contains", "startswith"}:
        # ``contains`` on lists accepts a scalar element; on strings, a substring.
        if isinstance(record_value, list):
            if op != "contains":
                return False
            return value in record_value
        if not isinstance(record_value, str) or not isinstance(value, str):
            return False
        return value in record_value if op == "contains" else record_value.startswith(value)

    if op in {"contains_all", "contains_any"}:
        if not isinstance(record_value, list):
            return False
        candidates = _as_list(value)
        if op == "contains_all":
            return all(c in record_value for c in candidates)
        return any(c in record_value for c in candidates)

    record_value, coerced = coerce_filter_value(record_value, value)

    if op == "eq":
        return bool(record_value == coerced)
    # Ordering operators are undefined against None; treat like a missing field.
    if (record_value is None or coerced is None) and op in {"gt", "gte", "lt", "lte"}:
        return False
    if op == "gt":
        return bool(record_value > coerced)
    if op == "gte":
        return bool(record_value >= coerced)
    if op == "lt":
        return bool(record_value < coerced)
    if op == "lte":
        return bool(record_value <= coerced)
    raise ValueError(f"Unknown filter operator: {op}")


def apply_filters(
    records: Iterable[tuple[str, dict[str, Any]]],
    query: dict[str, Any],
) -> list[str]:
    """Return the ids of records satisfying every ``field__op`` entry in ``query``.

    An empty ``query`` returns every id in iteration order.
    """
    if not query:
        return [source_id for source_id, _ in records]

    parsed = [(*split_filter_key(k), v) for k, v in query.items()]

    matched: list[str] = []
    for source_id, record in records:
        if all(evaluate_condition(record, f, op, v) for f, op, v in parsed):
            matched.append(source_id)
    return matched
