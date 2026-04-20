"""Identify top-level DAG schemas in a module.

A schema class declaring a non-empty ``Schema.extensions`` is a composite
that pulls other schemas in through ``extend()`` at runtime. It is, by
construction, a DAG root — no other schema reaches it from above.

When *any* class in the input list has extensions, only those classes
are treated as roots (the rest are leaves reached transitively). When
*no* class has extensions, the list is a flat set of independent
extractors and every class is a root.

The same rule powers two call sites: the ``ennoia craft`` post-step that
appends the ``ennoia_schema`` entrypoint variable, and the CLI
``load_schemas`` fallback that runs when the user's module does not
declare the variable.
"""

from __future__ import annotations

from ennoia.schema.base import (
    BaseCollection,
    BaseSemantic,
    BaseStructure,
    get_schema_extensions,
)

__all__ = ["identify_roots"]

SchemaClass = type[BaseStructure] | type[BaseSemantic] | type[BaseCollection]


def identify_roots(classes: list[SchemaClass]) -> list[SchemaClass]:
    """Return the subset of ``classes`` that are DAG roots.

    Rule:

    - If any class has a non-empty ``Schema.extensions``, roots are every
      class that does.
    - Otherwise, every class is a root.

    Declaration order is preserved.
    """
    with_extensions = [cls for cls in classes if get_schema_extensions(cls)]
    if with_extensions:
        return with_extensions
    return list(classes)
