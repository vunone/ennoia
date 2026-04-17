"""Superschema construction — merge manifest fields into a unified space.

Given a :class:`~ennoia.schema.manifest.SchemaManifest`, collapse every
reachable structural schema's fields into a single ``{name: SuperField}``
map, applying namespace prefixes and merging multi-source fields per the
spec rules in ``docs/schemas.md §Field Merging`` and
``docs/schemas.md §Type Compatibility``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Literal, cast, get_args, get_origin

from ennoia.index.exceptions import SchemaError, SchemaWarning
from ennoia.schema.base import BaseCollection, BaseSemantic, BaseStructure
from ennoia.schema.manifest import SchemaManifest
from ennoia.schema.operators import (
    field_metadata,
    infer_operators,
    is_filterable,
    type_label,
    unwrap_optional,
)

__all__ = [
    "SuperField",
    "Superschema",
    "build_superschema",
    "merge_field_types",
]


@dataclass
class SuperField:
    """Resolved entry in the unified superschema.

    Shape mirrors ``docs/filters.md §Schema Discovery`` with two additions:
    ``sources`` (classes contributing this field) and
    ``has_divergent_descriptions`` (only emitted when multi-source).
    """

    name: str
    annotation: Any
    type_label: str
    operators: list[str]
    sources: list[str]
    options: list[Any] | None = None
    item_type: str | None = None
    description: str | None = None
    nullable: bool = False
    has_divergent_descriptions: bool = False


@dataclass
class Superschema:
    """Unified field space and semantic-index list."""

    fields: dict[str, SuperField]
    semantic_indices: list[dict[str, str]]
    manifest: SchemaManifest

    def all_field_names(self) -> set[str]:
        return set(self.fields)

    def to_discovery_payload(self) -> dict[str, Any]:
        """Render the SDK/MCP discovery JSON payload.

        Fields are sorted by name for deterministic output. Optional keys
        (``options``, ``item_type``, ``nullable``, ``has_divergent_descriptions``)
        are only present when meaningful.
        """
        structural: list[dict[str, Any]] = []
        for sf in sorted(self.fields.values(), key=lambda f: f.name):
            record: dict[str, Any] = {
                "name": sf.name,
                "type": sf.type_label,
                "operators": sf.operators,
                "sources": sf.sources,
                "description": sf.description,
            }
            if sf.options is not None:
                record["options"] = sf.options
            if sf.item_type is not None:
                record["item_type"] = sf.item_type
            if sf.nullable:
                record["nullable"] = True
            if len(sf.sources) > 1:
                record["has_divergent_descriptions"] = sf.has_divergent_descriptions
            structural.append(record)
        return {
            "structural_fields": structural,
            "semantic_indices": list(self.semantic_indices),
        }


def merge_field_types(a: Any, b: Any) -> Any:
    """Return the merged annotation for two field types, or raise ``SchemaError``.

    Handles identity, Literal union, Optional absorption, and list recursion
    per ``docs/schemas.md §Type Compatibility``. Other pairings are rejected.
    """
    if a == b:
        return a

    a_inner, a_null = unwrap_optional(a)
    b_inner, b_null = unwrap_optional(b)
    any_nullable = a_null or b_null

    if a_inner == b_inner:
        return _reapply_optional(a_inner, any_nullable)

    a_origin = get_origin(a_inner)
    b_origin = get_origin(b_inner)

    # Literal ∪ Literal: union of values, preserving first-seen order.
    if a_origin is Literal and b_origin is Literal:
        a_vals = list(get_args(a_inner))
        b_vals = list(get_args(b_inner))
        merged_vals = list(dict.fromkeys([*a_vals, *b_vals]))
        # Python typing requires Literal[...] args to be a tuple of values.
        merged_literal = cast(Any, Literal).__getitem__(tuple(merged_vals))
        return _reapply_optional(merged_literal, any_nullable)

    # Bare ``list`` (no type args) is not mergeable — the element type is
    # unknown so we cannot decide inner-type compatibility.
    if a_inner is list or b_inner is list:
        raise SchemaError(f"Cannot merge unparameterized list types: {a!r} vs {b!r}.")

    # list[T] + list[U] — recurse on inner type. Enforce same container origin.
    if a_origin is list and b_origin is list:
        a_args, b_args = get_args(a_inner), get_args(b_inner)
        if not a_args or not b_args:
            raise SchemaError(f"Cannot merge unparameterized list types: {a!r} vs {b!r}.")
        merged_item = merge_field_types(a_args[0], b_args[0])
        merged_list = list[merged_item]  # type: ignore[valid-type]
        return _reapply_optional(merged_list, any_nullable)

    raise SchemaError(f"Incompatible field types: {a!r} vs {b!r}.")


def _reapply_optional(annotation: Any, nullable: bool) -> Any:
    if not nullable:
        return annotation
    # ``Optional[T]`` is ``Union[T, None]``; use PEP 604 union for 3.11+.
    return annotation | None


def build_superschema(manifest: SchemaManifest) -> Superschema:
    """Walk the manifest in BFS order and construct the unified superschema.

    - Structural nodes contribute filterable fields under a namespace prefix
      (when declared) or flat.
    - Multi-source flat fields merge per :func:`merge_field_types`.
    - Descriptions: first-declared wins; divergence emits ``SchemaWarning``.
    - Semantic nodes surface as ``{name, description}`` entries separately.
    """
    fields: dict[str, SuperField] = {}
    semantic_indices: list[dict[str, str]] = []
    # Track the original Pydantic FieldInfo per emitted key so we can
    # re-derive operators when merging types.
    first_info: dict[str, Any] = {}
    # Track description sources for divergence detection.
    descriptions_seen: dict[str, set[str | None]] = {}

    for node in manifest.nodes:
        cls = node.cls
        if issubclass(cls, BaseCollection):
            semantic_indices.append(
                {
                    "name": cls.__name__,
                    "description": cls.extract_prompt(),
                    "kind": "collection",
                }
            )
            continue
        if issubclass(cls, BaseSemantic):
            semantic_indices.append(
                {
                    "name": cls.__name__,
                    "description": cls.extract_prompt(),
                    "kind": "semantic",
                }
            )
            continue

        assert issubclass(cls, BaseStructure)
        ns = node.namespace
        for field_name, info in cls.model_fields.items():
            if not is_filterable(info):
                continue
            emitted = f"{ns}__{field_name}" if ns is not None else field_name
            annotation = info.annotation
            description = info.description

            if emitted not in fields:
                fields[emitted] = _superfield_from_single_source(
                    name=emitted,
                    annotation=annotation,
                    info=info,
                    description=description,
                    source_cls=cls.__name__,
                )
                first_info[emitted] = info
                descriptions_seen[emitted] = {description}
            else:
                # Second+ source: merge types and append source.
                existing = fields[emitted]
                try:
                    merged_annotation = merge_field_types(existing.annotation, annotation)
                except SchemaError as err:
                    raise SchemaError(
                        f"Field {emitted!r} has incompatible types across sources "
                        f"{existing.sources + [cls.__name__]}: {err}"
                    ) from err
                fields[emitted] = _rebuild_superfield_after_merge(
                    existing=existing,
                    merged_annotation=merged_annotation,
                    info=first_info[emitted],
                    new_source=cls.__name__,
                )
                descriptions_seen[emitted].add(description)

    # Divergent descriptions: warn + flag on each affected field.
    for name, seen in descriptions_seen.items():
        non_empty = {d for d in seen if d}
        if len(fields[name].sources) > 1 and len(non_empty) > 1:
            fields[name].has_divergent_descriptions = True
            warnings.warn(
                f"Field {name!r} has divergent descriptions across sources "
                f"{fields[name].sources}. First-declared wins: "
                f"{fields[name].description!r}.",
                SchemaWarning,
                stacklevel=2,
            )

    return Superschema(fields=fields, semantic_indices=semantic_indices, manifest=manifest)


def _superfield_from_single_source(
    *,
    name: str,
    annotation: Any,
    info: Any,
    description: str | None,
    source_cls: str,
) -> SuperField:
    _, nullable = unwrap_optional(annotation)
    label, extras = type_label(annotation)

    override = field_metadata(info).get("operators")
    if isinstance(override, list | tuple):
        override_seq = cast("list[Any] | tuple[Any, ...]", override)
        operators = [str(op) for op in override_seq]
    else:
        operators = infer_operators(annotation)

    if nullable and "is_null" not in operators:
        operators = [*operators, "is_null"]

    return SuperField(
        name=name,
        annotation=annotation,
        type_label=label,
        operators=operators,
        options=extras.get("options"),
        item_type=extras.get("item_type"),
        sources=[source_cls],
        description=description,
        nullable=nullable,
    )


def _rebuild_superfield_after_merge(
    *,
    existing: SuperField,
    merged_annotation: Any,
    info: Any,
    new_source: str,
) -> SuperField:
    """Recompute label/operators/options from the merged annotation.

    Description stays as the first-declared (``existing.description``).
    If the original source declared an explicit operator override via
    ``Field(operators=[...])``, it is preserved — overrides are author intent
    and should not be silently broadened by the merge.
    """
    _, nullable = unwrap_optional(merged_annotation)
    label, extras = type_label(merged_annotation)

    override = field_metadata(info).get("operators")
    if isinstance(override, list | tuple):
        override_seq = cast("list[Any] | tuple[Any, ...]", override)
        operators = [str(op) for op in override_seq]
    else:
        operators = infer_operators(merged_annotation)
    if nullable and "is_null" not in operators:
        operators = [*operators, "is_null"]

    return SuperField(
        name=existing.name,
        annotation=merged_annotation,
        type_label=label,
        operators=operators,
        options=extras.get("options"),
        item_type=extras.get("item_type"),
        sources=[*existing.sources, new_source],
        description=existing.description,
        nullable=nullable,
        has_divergent_descriptions=existing.has_divergent_descriptions,
    )
