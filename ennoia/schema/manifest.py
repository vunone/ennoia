"""Emission-manifest graph — transitive closure of ``Schema.extensions``.

The manifest is computed once at ``Pipeline.__init__`` from the root
schemas and consumed by :mod:`ennoia.schema.merging` to build the
superschema. Walking ``Schema.extensions`` instead of inferring from
``extend()`` keeps the field space known ahead of time and lets the
framework reject undeclared emissions deterministically.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ennoia.index.exceptions import SchemaError
from ennoia.schema.base import (
    BaseCollection,
    BaseSemantic,
    BaseStructure,
    get_schema_extensions,
    get_schema_namespace,
)
from ennoia.utils.filters import KNOWN_OPERATORS

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ["ManifestNode", "SchemaManifest", "build_manifest"]


@dataclass(frozen=True)
class ManifestNode:
    """One resolved node in the emission graph.

    ``namespace`` applies only to ``cls``'s own fields and does not
    propagate to descendants. ``depth`` is the shortest distance from any
    root, used by the merging layer to tie-break description conflicts.
    """

    cls: type[BaseStructure] | type[BaseSemantic] | type[BaseCollection]
    namespace: str | None
    depth: int


@dataclass(frozen=True)
class SchemaManifest:
    """Resolved manifest — deduplicated BFS order from all roots."""

    roots: tuple[type[BaseStructure] | type[BaseSemantic] | type[BaseCollection], ...]
    nodes: tuple[ManifestNode, ...]

    def structurals(self) -> tuple[ManifestNode, ...]:
        return tuple(
            n
            for n in self.nodes
            if issubclass(n.cls, BaseStructure) and not issubclass(n.cls, BaseCollection)
        )

    def semantics(self) -> tuple[ManifestNode, ...]:
        return tuple(n for n in self.nodes if issubclass(n.cls, BaseSemantic))

    def collections(self) -> tuple[ManifestNode, ...]:
        return tuple(n for n in self.nodes if issubclass(n.cls, BaseCollection))


def _validate_namespace(ns: str, cls: type) -> None:
    if not ns.isidentifier():
        raise SchemaError(
            f"{cls.__name__}.Schema.namespace {ns!r} is not a valid Python identifier."
        )
    if "__" in ns:
        raise SchemaError(
            f"{cls.__name__}.Schema.namespace {ns!r} contains '__', which is reserved "
            "as the namespace/operator delimiter."
        )
    if ns in KNOWN_OPERATORS:
        raise SchemaError(
            f"{cls.__name__}.Schema.namespace {ns!r} collides with a filter operator. "
            f"Reserved names: {sorted(KNOWN_OPERATORS)}."
        )


def _validate_field_names(cls: type[BaseStructure], namespace: str | None) -> None:
    """Reject any field whose emitted key would break filter-operator parsing.

    Two failure modes:

    - Direct collision: the field name itself is a reserved operator
      (e.g. a field literally named ``eq``). A filter key ``eq`` would be
      interpreted as operator ``eq`` against an empty field.
    - Suffix collision: the emitted key (``{ns}__{field}`` when namespaced,
      else just ``{field}``) ends with ``__{op}``. Example: field ``foo__eq``
      at top level — the filter key ``foo__eq`` then splits as
      ``(foo, eq)`` instead of ``(foo__eq, eq)``. Rename to avoid.
    """
    for field_name in cls.model_fields:
        if field_name in KNOWN_OPERATORS:
            raise SchemaError(
                f"{cls.__name__}.{field_name}: field name {field_name!r} collides with "
                f"a reserved filter operator. Rename the field. "
                f"Reserved names: {sorted(KNOWN_OPERATORS)}."
            )
        emitted = f"{namespace}__{field_name}" if namespace is not None else field_name
        for op in KNOWN_OPERATORS:
            if emitted.endswith(f"__{op}"):
                where = f" under namespace {namespace!r}" if namespace is not None else ""
                raise SchemaError(
                    f"{cls.__name__}.{field_name}{where} would emit filter key "
                    f"{emitted!r}, which ends with a reserved operator suffix "
                    f"'__{op}'. Rename the field."
                )


def _validate_extension_entries(cls: type[BaseStructure], extensions: Iterable[type]) -> None:
    for ext in extensions:
        if not issubclass(ext, BaseStructure | BaseSemantic):
            raise SchemaError(
                f"{cls.__name__}.Schema.extensions contains {ext.__name__!r}, which is "
                "not a BaseStructure or BaseSemantic subclass."
            )


def _validate_extension_entries_for_collection(
    cls: type[BaseCollection], extensions: Iterable[type]
) -> None:
    for ext in extensions:
        if not issubclass(ext, BaseStructure | BaseSemantic | BaseCollection):
            raise SchemaError(
                f"{cls.__name__}.Schema.extensions contains {ext.__name__!r}, which is "
                "not a BaseStructure, BaseSemantic, or BaseCollection subclass."
            )


def build_manifest(
    roots: list[type[BaseStructure] | type[BaseSemantic] | type[BaseCollection]],
) -> SchemaManifest:
    """Resolve the transitive emission graph from the given roots.

    - Dedupes by class identity; a schema reachable through multiple paths
      appears once, at its shallowest observed depth.
    - Validates namespaces and structural field names against
      :data:`KNOWN_OPERATORS` as each node is visited.
    - Cycles raise ``SchemaError`` with the offending path.
    - ``BaseSemantic`` subclasses are terminal — the walk never reads
      ``Schema.*`` off them.
    - ``BaseCollection`` subclasses are walked for ``Schema.extensions`` like
      structural nodes but do not contribute tabular fields to the superschema.
    """
    if not roots:
        return SchemaManifest(roots=(), nodes=())

    nodes: list[ManifestNode] = []
    seen: set[type] = set()

    # Queue entries: (cls, depth, ancestor_path). The ancestor path is
    # per-branch so a diamond (A -> B -> D and A -> C -> D) does not false-trip
    # cycle detection — only a repeat on the *current* path is a cycle.
    queue: deque[tuple[type, int, tuple[type, ...]]] = deque((root, 0, ()) for root in roots)

    while queue:
        cls, depth, path = queue.popleft()

        if cls in path:
            cycle = [c.__name__ for c in (*path, cls)]
            raise SchemaError(f"Cycle detected in Schema.extensions: {' -> '.join(cycle)}.")

        if cls in seen:
            continue
        seen.add(cls)

        if not issubclass(cls, BaseStructure | BaseSemantic | BaseCollection):
            raise SchemaError(
                f"{cls.__name__!r} is not a BaseStructure, BaseSemantic, or "
                "BaseCollection subclass."
            )

        namespace = get_schema_namespace(cls)
        if namespace is not None:
            _validate_namespace(namespace, cls)

        if issubclass(cls, BaseCollection):
            extensions = get_schema_extensions(cls)
            _validate_extension_entries_for_collection(cls, extensions)
            nodes.append(ManifestNode(cls=cls, namespace=None, depth=depth))
            for ext in extensions:
                queue.append((ext, depth + 1, (*path, cls)))
        elif issubclass(cls, BaseStructure):
            _validate_field_names(cls, namespace)
            extensions = get_schema_extensions(cls)
            _validate_extension_entries(cls, extensions)
            nodes.append(ManifestNode(cls=cls, namespace=namespace, depth=depth))
            for ext in extensions:
                queue.append((ext, depth + 1, (*path, cls)))
        else:
            # BaseSemantic leaf — no extensions, no namespace semantics.
            nodes.append(ManifestNode(cls=cls, namespace=None, depth=depth))

    return SchemaManifest(roots=tuple(roots), nodes=tuple(nodes))
