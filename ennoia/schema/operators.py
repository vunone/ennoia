"""Operator inference from Pydantic field annotations.

The mapping of Python / Pydantic types to filter operators is canonical per
``docs/filters.md``. Every interface (SDK, CLI, MCP, REST) derives the filter
contract from the same :func:`infer_operators` + :func:`describe_field`
helpers, so behavior is consistent across surfaces.
"""

from __future__ import annotations

from datetime import date, datetime
from types import UnionType
from typing import Any, Literal, Union, cast, get_args, get_origin

from pydantic.fields import FieldInfo

__all__ = [
    "ENNOIA_FIELD_METADATA_KEY",
    "FieldDescription",
    "describe_field",
    "field_metadata",
    "infer_operators",
    "is_filterable",
    "type_label",
    "unwrap_optional",
]

ENNOIA_FIELD_METADATA_KEY = "ennoia"

_STRING_OPERATORS: tuple[str, ...] = ("eq", "contains", "startswith")
_NUMERIC_OPERATORS: tuple[str, ...] = ("eq", "gt", "gte", "lt", "lte")
_DATE_OPERATORS: tuple[str, ...] = ("eq", "gt", "gte", "lt", "lte")
_LITERAL_OPERATORS: tuple[str, ...] = ("eq", "in")
_LIST_OPERATORS: tuple[str, ...] = ("contains", "contains_all", "contains_any")
_BOOL_OPERATORS: tuple[str, ...] = ("eq",)


FieldDescription = dict[str, Any]


def unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    """If ``annotation`` is ``Optional[T]`` / ``T | None`` return ``(T, True)``."""
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        args = [a for a in get_args(annotation) if a is not type(None)]
        nullable = len(args) != len(get_args(annotation))
        if len(args) == 1:
            return args[0], nullable
        # Unsupported union shape (e.g. ``int | str``) — leave as-is but flag nullability.
        return annotation, nullable
    return annotation, False


def infer_operators(annotation: Any) -> list[str]:
    """Return the inferred operator list for a Pydantic field annotation.

    Follows ``docs/filters.md §Operator Inference``. Unknown types fall back
    to ``["eq"]`` — the only operator guaranteed to be universally defined.
    """
    inner, nullable = unwrap_optional(annotation)
    operators = list(_operators_for_non_optional(inner))
    if nullable and "is_null" not in operators:
        operators.append("is_null")
    return operators


def _operators_for_non_optional(annotation: Any) -> tuple[str, ...]:
    if annotation is bool:
        return _BOOL_OPERATORS
    if annotation is str:
        return _STRING_OPERATORS
    if annotation in (int, float):
        return _NUMERIC_OPERATORS
    if annotation is date or annotation is datetime:
        return _DATE_OPERATORS

    origin = get_origin(annotation)
    if origin is Literal:
        return _LITERAL_OPERATORS
    if origin in (list, tuple, set, frozenset):
        return _LIST_OPERATORS

    return ("eq",)


def field_metadata(field_info: FieldInfo) -> dict[str, Any]:
    """Return the ennoia-specific metadata dict stored on a pydantic FieldInfo."""
    extra = field_info.json_schema_extra
    if isinstance(extra, dict):
        extra_map = cast(dict[str, Any], extra)
        meta = extra_map.get(ENNOIA_FIELD_METADATA_KEY)
        if isinstance(meta, dict):
            # Explicit ``dict[str, Any]`` because FieldInfo types ``json_schema_extra``
            # loosely; downstream callers rely on string keys.
            meta_map = cast(dict[str, Any], meta)
            return {str(k): v for k, v in meta_map.items()}
    return {}


def is_filterable(field_info: FieldInfo) -> bool:
    meta = field_metadata(field_info)
    return bool(meta.get("filterable", True))


def type_label(annotation: Any) -> tuple[str, dict[str, Any]]:
    """Return ``(type_label, extras)`` for the discovery payload."""
    inner, _nullable = unwrap_optional(annotation)
    if inner is bool:
        return "bool", {}
    if inner is str:
        return "str", {}
    if inner is int:
        return "int", {}
    if inner is float:
        return "float", {}
    if inner is date:
        return "date", {}
    if inner is datetime:
        return "datetime", {}

    origin = get_origin(inner)
    if origin is Literal:
        return "enum", {"options": list(get_args(inner))}
    if origin in (list, tuple, set, frozenset):
        args = get_args(inner)
        if args:
            item_label, _ = type_label(args[0])
            return "list", {"item_type": item_label}
        return "list", {}

    return getattr(inner, "__name__", str(inner)), {}


def describe_field(name: str, field_info: FieldInfo) -> FieldDescription | None:
    """Return the discovery record for a filterable field, or ``None`` if excluded.

    The resulting shape matches ``docs/filters.md §Schema Discovery``.
    """
    if not is_filterable(field_info):
        return None

    annotation = field_info.annotation
    _, nullable = unwrap_optional(annotation)
    label, extras = type_label(annotation)

    override = field_metadata(field_info).get("operators")
    if isinstance(override, (list, tuple)):
        override_seq = cast("list[Any] | tuple[Any, ...]", override)
        operators = [str(op) for op in override_seq]
    else:
        operators = infer_operators(annotation)

    record: FieldDescription = {"name": name, "type": label}
    record.update(extras)
    if nullable:
        record["nullable"] = True
        if "is_null" not in operators:
            operators = [*operators, "is_null"]
    record["operators"] = operators
    return record
