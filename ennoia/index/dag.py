"""Schema validation and initial execution layer.

In Stage 1 the DAG is trivial — every declared structural schema is
extracted in order, and `BaseStructure.extend()` returns at runtime
whatever child schemas should follow. The runtime tree is built in
`Pipeline.aindex` from those `extend()` results, not from a compile-time
inspection of decorated dependencies.

`build_dag` is kept as the single layer container so future stages can
reintroduce parallelism (e.g. `asyncio.gather` over a layer) without
changing the Pipeline's call site.
"""

from __future__ import annotations

from ennoia.schema.base import BaseCollection, BaseSemantic, BaseStructure

__all__ = ["build_dag", "validate_schemas"]


def validate_schemas(schemas: list[type]) -> None:
    for schema in schemas:
        if not (
            issubclass(schema, BaseStructure)
            or issubclass(schema, BaseSemantic)
            or issubclass(schema, BaseCollection)
        ):
            raise TypeError(
                f"{schema.__name__} must inherit from BaseStructure, BaseSemantic, "
                "or BaseCollection."
            )


def build_dag(schemas: list[type]) -> list[list[type]]:
    """Return a single execution layer containing the declared schemas.

    Stage 1 has no compile-time dependency mechanism. Inter-schema
    relationships are expressed at runtime through `extend()`.
    """
    validate_schemas(schemas)
    return [list(schemas)]
