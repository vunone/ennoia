"""Ennoia-flavored ``Field`` wrapper around ``pydantic.Field``.

Adds two ennoia-specific keyword arguments that steer filter behavior:

- ``operators``: restrict the filter operators exposed for this field (overrides
  the type-based inference documented in ``docs/filters.md``).
- ``filterable``: when ``False``, the field is excluded from the filter contract
  entirely (``describe_schema`` omits it, and ``validate_filters`` rejects any
  query that references it).

Both values are stored in ``json_schema_extra`` under the ``ennoia`` namespace,
so downstream code (filter validator, discovery emitter) reads them via a
single accessor (:func:`ennoia.schema.operators.field_metadata`).
"""

from __future__ import annotations

from typing import Any, cast

from pydantic import Field as _PydanticField

from ennoia.schema.operators import ENNOIA_FIELD_METADATA_KEY

__all__ = ["Field"]


def Field(  # noqa: N802 — mirrors ``pydantic.Field``'s public name.
    default: Any = ...,
    *,
    operators: list[str] | None = None,
    filterable: bool = True,
    **kwargs: Any,
) -> Any:
    """Construct a Pydantic FieldInfo with ennoia-aware metadata.

    Any keyword accepted by ``pydantic.Field`` passes through unchanged. The
    ``operators`` / ``filterable`` kwargs are stored under ``json_schema_extra``
    keyed by :data:`ENNOIA_FIELD_METADATA_KEY`, leaving the public JSON Schema
    clean for LLM prompts.
    """
    extra = kwargs.pop("json_schema_extra", None)
    merged_extra: dict[str, Any] = (
        dict(cast(dict[str, Any], extra)) if isinstance(extra, dict) else {}
    )
    stored_meta = merged_extra.get(ENNOIA_FIELD_METADATA_KEY, {})
    ennoia_meta: dict[str, Any] = (
        dict(cast(dict[str, Any], stored_meta)) if isinstance(stored_meta, dict) else {}
    )

    if operators is not None:
        ennoia_meta["operators"] = list(operators)
    # ``filterable`` defaults to True so we only persist when explicitly False —
    # keeps the stored payload minimal for the common case.
    if filterable is False:
        ennoia_meta["filterable"] = False

    if ennoia_meta:
        merged_extra[ENNOIA_FIELD_METADATA_KEY] = ennoia_meta
        kwargs["json_schema_extra"] = merged_extra
    elif extra is not None:
        kwargs["json_schema_extra"] = merged_extra

    return _PydanticField(default, **kwargs)
