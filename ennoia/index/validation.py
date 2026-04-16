"""Filter validation — reject unknown fields / operators before execution.

See ``docs/filters.md §Filter Validation`` for the canonical error shape.
Validation happens once per ``search()`` call, before any I/O hits the store,
so an agent using the SDK/CLI/MCP/REST receives a uniformly shaped error
regardless of which structured backend is mounted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ennoia.index.exceptions import FilterValidationError
from ennoia.utils.filters import split_filter_key

if TYPE_CHECKING:
    from ennoia.schema.merging import Superschema

__all__ = ["validate_filters"]


def validate_filters(
    filters: dict[str, Any] | None,
    superschema: Superschema,
) -> None:
    """Validate ``filters`` against the pipeline's unified superschema.

    The superschema is the single source of truth for the field space — it
    contains every reachable structural field (flat or namespaced) with its
    merged operator list. Raises :class:`FilterValidationError` on the first
    offending entry.
    """
    if not filters:
        return

    for key in filters:
        field_name, operator = split_filter_key(key)

        if field_name not in superschema.fields:
            raise FilterValidationError(
                field=field_name,
                operator=operator,
                message=(
                    f"Field {field_name!r} is not a filterable field on the configured schemas."
                ),
            )

        supported = superschema.fields[field_name].operators
        if operator not in supported:
            type_label = superschema.fields[field_name].type_label
            raise FilterValidationError(
                field=field_name,
                operator=operator,
                supported=supported,
                message=(
                    f"Field {field_name!r} (type: {type_label}) does not support operator "
                    f"{operator!r}. Supported operators: {', '.join(supported)}."
                ),
            )
