"""Filter validation — reject unknown fields / operators before execution.

See ``docs/filters.md §Filter Validation`` for the canonical error shape.
Validation happens once per ``search()`` call, before any I/O hits the store,
so an agent using the SDK/CLI/MCP/REST receives a uniformly shaped error
regardless of which structured backend is mounted.
"""

from __future__ import annotations

from typing import Any

from ennoia.index.exceptions import FilterValidationError
from ennoia.schema.base import BaseStructure
from ennoia.schema.operators import describe_field, is_filterable
from ennoia.utils.filters import split_filter_key

__all__ = ["build_filter_contract", "validate_filters"]


def build_filter_contract(
    schemas: list[type[BaseStructure]],
) -> dict[str, dict[str, Any]]:
    """Collapse every structural schema's fields into ``{name: descriptor}``.

    Later schemas override earlier ones on name collision (mirrors the behavior
    of :func:`ennoia.schema.describe`). Non-filterable fields are excluded
    entirely, so a ``validate_filters`` lookup on them reports *unknown field*
    — the correct signal for agents that should never reference them.
    """
    contract: dict[str, dict[str, Any]] = {}
    for schema in schemas:
        for name, info in schema.model_fields.items():
            if not is_filterable(info):
                contract.pop(name, None)
                continue
            record = describe_field(name, info)
            if record is not None:
                contract[name] = record
    return contract


def validate_filters(
    filters: dict[str, Any] | None,
    schemas: list[type[BaseStructure]],
) -> None:
    """Validate ``filters`` against the combined contract of ``schemas``.

    Raises :class:`FilterValidationError` on the first offending entry.
    """
    if not filters:
        return

    contract = build_filter_contract(schemas)

    for key in filters:
        field_name, operator = split_filter_key(key)

        if field_name not in contract:
            raise FilterValidationError(
                field=field_name,
                operator=operator,
                message=(
                    f"Field {field_name!r} is not a filterable field on the configured schemas."
                ),
            )

        supported = contract[field_name].get("operators", [])
        if operator not in supported:
            type_label = contract[field_name].get("type", "unknown")
            raise FilterValidationError(
                field=field_name,
                operator=operator,
                supported=supported,
                message=(
                    f"Field {field_name!r} (type: {type_label}) does not support operator "
                    f"{operator!r}. Supported operators: {', '.join(supported)}."
                ),
            )
