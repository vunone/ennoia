"""Translate Ennoia's unified filter syntax to a Qdrant ``Filter`` clause.

Qdrant natively covers the bulk of the 11 Ennoia operators via
``MatchValue`` / ``MatchAny`` / ``Range`` / ``IsNullCondition``. Operators that
don't map cleanly (``startswith``, string ``contains``, ``contains_all``) are
returned in the ``residual`` dict so the caller can post-filter hits with
:func:`~ennoia.utils.filters.apply_filters`.

The split keeps the common path one round-trip to Qdrant while staying
semantically faithful to ``docs/filters.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ennoia.utils.filters import parse_bool, split_filter_key

if TYPE_CHECKING:  # pragma: no cover
    from qdrant_client.models import Filter

__all__ = ["translate_filter"]


# Operators that always push down to a native Qdrant condition.
_NATIVE_OPS: frozenset[str] = frozenset(
    {"eq", "in", "gt", "gte", "lt", "lte", "is_null", "contains_any"}
)


def translate_filter(
    filters: dict[str, Any] | None,
    *,
    list_fields: frozenset[str] = frozenset(),
) -> tuple[Filter | None, dict[str, Any]]:
    """Compile ``filters`` into ``(qdrant_filter, residual)``.

    ``list_fields`` is the set of payload keys whose values are stored as
    lists. For those, the ``contains`` operator is treated as a list-element
    match (native) rather than a substring match (residual).
    """
    from qdrant_client.models import (
        Condition,
        DatetimeRange,
        FieldCondition,
        Filter,
        IsNullCondition,
        MatchAny,
        MatchValue,
        PayloadField,
        Range,
    )

    if not filters:
        return None, {}

    must: list[Condition] = []
    must_not: list[Condition] = []
    residual: dict[str, Any] = {}

    for key, value in filters.items():
        field, op = split_filter_key(key)

        if op == "contains" and field in list_fields:
            # List-contains: element equality on a keyword list field.
            must.append(FieldCondition(key=field, match=MatchValue(value=value)))
            continue

        if op not in _NATIVE_OPS:
            residual[key] = value
            continue

        if op == "eq":
            must.append(FieldCondition(key=field, match=MatchValue(value=value)))
        elif op == "in":
            must.append(FieldCondition(key=field, match=MatchAny(any=_as_list(value))))
        elif op in {"gt", "gte", "lt", "lte"}:
            range_cls = DatetimeRange if _looks_like_datetime(value) else Range
            must.append(FieldCondition(key=field, range=range_cls(**{op: value})))
        elif op == "is_null":
            clause = IsNullCondition(is_null=PayloadField(key=field))
            if parse_bool(value):
                must.append(clause)
            else:
                must_not.append(clause)
        else:
            # op is the last native op ("contains_any") — the `op not in _NATIVE_OPS`
            # early-continue above already filtered out non-native operators.
            must.append(FieldCondition(key=field, match=MatchAny(any=_as_list(value))))

    if not must and not must_not:
        return None, residual

    qfilter = Filter(must=must or None, must_not=must_not or None)
    return qfilter, residual


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return list(value)  # type: ignore[arg-type]
    if isinstance(value, tuple):
        return list(value)  # type: ignore[arg-type]
    if isinstance(value, str):
        return [v.strip() for v in value.split(",")]
    return [value]


def _looks_like_datetime(value: Any) -> bool:
    """Route range comparisons against dates/datetimes to ``DatetimeRange``."""
    from datetime import date, datetime

    if isinstance(value, (date, datetime)):
        return True
    if isinstance(value, str):
        try:
            datetime.fromisoformat(value)
        except ValueError:
            return False
        return True
    return False
