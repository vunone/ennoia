"""User-facing schema declaration layer."""

from __future__ import annotations

from typing import Any

from ennoia.schema.base import BaseSemantic, BaseStructure
from ennoia.schema.fields import Field

__all__ = [
    "BaseSemantic",
    "BaseStructure",
    "Field",
    "describe",
]


def describe(
    schemas: list[type[BaseStructure] | type[BaseSemantic]],
) -> dict[str, Any]:
    """Return the combined discovery payload for a set of schemas.

    Shape follows ``docs/filters.md §Schema Discovery``:

    - ``structural_fields``: flattened list of per-field records from every
      ``BaseStructure`` subclass (later schemas override earlier ones on name
      collision — keeps the output deterministic for overlapping fields).
    - ``semantic_indices``: one entry per ``BaseSemantic`` subclass with its
      docstring as the agent-readable description.
    """
    structural_fields: list[dict[str, Any]] = []
    seen_field_names: set[str] = set()
    semantic_indices: list[dict[str, Any]] = []

    for cls in schemas:
        if issubclass(cls, BaseSemantic):
            semantic_indices.append(
                {
                    "name": cls.__name__,
                    "description": cls.extract_prompt(),
                }
            )
        else:
            for field_record in cls.describe_schema()["fields"]:
                if field_record["name"] in seen_field_names:
                    # Replace the earlier record — last declaration wins.
                    structural_fields = [
                        f for f in structural_fields if f["name"] != field_record["name"]
                    ]
                structural_fields.append(field_record)
                seen_field_names.add(field_record["name"])

    return {
        "structural_fields": structural_fields,
        "semantic_indices": semantic_indices,
    }
