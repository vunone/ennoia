"""Emission-manifest construction: BFS walk, cycle detection, validation."""

from __future__ import annotations

from typing import ClassVar

import pytest

from ennoia import BaseSemantic, BaseStructure
from ennoia.index.exceptions import SchemaError
from ennoia.schema.base import get_schema_extensions, get_schema_namespace
from ennoia.schema.manifest import build_manifest


class Leaf(BaseStructure):
    """Leaf schema with no extensions."""

    name: str


class _DefaultSchemaProbe(BaseStructure):
    """Schema without explicit inner Schema inherits safe defaults."""

    value: str


def test_defaults_when_schema_class_absent() -> None:
    assert get_schema_namespace(_DefaultSchemaProbe) is None
    assert get_schema_extensions(_DefaultSchemaProbe) == []


class _WithBareSchema(BaseStructure):
    """User declares a bare Schema class with no base."""

    value: str

    class Schema:
        namespace = "ns"
        extensions: ClassVar[list[type]] = [Leaf]


def test_bare_schema_class_is_read_without_base_inheritance() -> None:
    # The user's Schema does NOT extend the framework default; the accessors
    # read via getattr with defaults, so partial declarations work too.
    assert get_schema_namespace(_WithBareSchema) == "ns"
    assert get_schema_extensions(_WithBareSchema) == [Leaf]


class _PartialSchemaOnlyExtensions(BaseStructure):
    """User declares extensions but omits namespace."""

    value: str

    class Schema:
        extensions: ClassVar[list[type]] = [Leaf]


def test_partial_schema_falls_back_to_defaults_for_missing_attrs() -> None:
    assert get_schema_namespace(_PartialSchemaOnlyExtensions) is None
    assert get_schema_extensions(_PartialSchemaOnlyExtensions) == [Leaf]


def test_empty_roots_returns_empty_manifest() -> None:
    manifest = build_manifest([])
    assert manifest.roots == ()
    assert manifest.nodes == ()


def test_single_root_structural_leaf() -> None:
    manifest = build_manifest([Leaf])
    assert manifest.roots == (Leaf,)
    assert len(manifest.nodes) == 1
    node = manifest.nodes[0]
    assert node.cls is Leaf
    assert node.namespace is None
    assert node.depth == 0


class _Summary(BaseSemantic):
    """What is the summary?"""


def test_single_root_semantic_leaf() -> None:
    manifest = build_manifest([_Summary])
    assert len(manifest.nodes) == 1
    assert manifest.nodes[0].cls is _Summary
    assert manifest.structurals() == ()
    assert manifest.semantics() == (manifest.nodes[0],)


class _Child(BaseStructure):
    """A child schema with a single field."""

    detail: str


class _Parent(BaseStructure):
    """Parent extends to a single child."""

    label: str

    class Schema:
        extensions: ClassVar[list[type]] = [_Child]


def test_multi_level_manifest_preserves_depth() -> None:
    manifest = build_manifest([_Parent])
    depths = {n.cls.__name__: n.depth for n in manifest.nodes}
    assert depths == {"_Parent": 0, "_Child": 1}


class _Shared(BaseStructure):
    """Diamond bottom."""

    key: str


class _LeftBranch(BaseStructure):
    """Diamond left arm."""

    left_field: str

    class Schema:
        extensions: ClassVar[list[type]] = [_Shared]


class _RightBranch(BaseStructure):
    """Diamond right arm."""

    right_field: str

    class Schema:
        extensions: ClassVar[list[type]] = [_Shared]


class _Top(BaseStructure):
    """Diamond top."""

    top_field: str

    class Schema:
        extensions: ClassVar[list[type]] = [_LeftBranch, _RightBranch]


def test_diamond_dedupes_at_shallowest_depth() -> None:
    manifest = build_manifest([_Top])
    classes = [n.cls.__name__ for n in manifest.nodes]
    assert classes.count("_Shared") == 1
    shared_node = next(n for n in manifest.nodes if n.cls is _Shared)
    # LeftBranch and RightBranch both at depth 1; _Shared appears at depth 2.
    assert shared_node.depth == 2


def test_self_loop_raises_cycle() -> None:
    # Construct a self-referential extensions list at runtime — the class
    # body cannot reference itself.
    class SelfLoop(BaseStructure):
        """Self-referential — illegal."""

        value: str

        class Schema:
            extensions: ClassVar[list[type]] = []

    SelfLoop.Schema.extensions = [SelfLoop]
    with pytest.raises(SchemaError, match="Cycle"):
        build_manifest([SelfLoop])


def test_mutual_cycle_raises_with_path_in_message() -> None:
    class NodeA(BaseStructure):
        """A."""

        a: str

        class Schema:
            extensions: ClassVar[list[type]] = []

    class NodeB(BaseStructure):
        """B."""

        b: str

        class Schema:
            extensions: ClassVar[list[type]] = [NodeA]

    NodeA.Schema.extensions = [NodeB]

    with pytest.raises(SchemaError) as exc:
        build_manifest([NodeA])
    msg = str(exc.value)
    assert "NodeA" in msg and "NodeB" in msg


class _SemLeafForExt(BaseSemantic):
    """Question?"""


class _HasSemExtension(BaseStructure):
    """Structural parent pulling in a semantic child."""

    header: str

    class Schema:
        extensions: ClassVar[list[type]] = [_SemLeafForExt]


def test_semantic_terminal_leaf_ignored_for_recursion() -> None:
    manifest = build_manifest([_HasSemExtension])
    names = {n.cls.__name__ for n in manifest.nodes}
    assert names == {"_HasSemExtension", "_SemLeafForExt"}
    sem_node = next(n for n in manifest.nodes if n.cls is _SemLeafForExt)
    assert sem_node.namespace is None


def test_extension_entry_must_be_base_subclass() -> None:
    class NotASchema:
        pass

    class Bad(BaseStructure):
        """Bad."""

        x: str

        class Schema:
            extensions: ClassVar[list[type]] = []

    Bad.Schema.extensions = [NotASchema]
    with pytest.raises(SchemaError, match="not a BaseStructure or BaseSemantic"):
        build_manifest([Bad])


def test_field_name_collides_with_operator_raises() -> None:
    class CollidingField(BaseStructure):
        """Field name 'eq' collides with the reserved filter operator."""

        eq: str

    with pytest.raises(SchemaError, match="collides with a reserved filter operator"):
        build_manifest([CollidingField])


def test_field_suffix_collides_with_operator_raises() -> None:
    class SuffixCollision(BaseStructure):
        """Field 'foo__eq' would be parsed as (foo, eq) in filters."""

        foo__eq: str

    with pytest.raises(SchemaError, match="reserved operator suffix"):
        build_manifest([SuffixCollision])


def test_namespaced_non_colliding_fields_succeed() -> None:
    # Sanity: a namespace + ordinary field name combine without issue and
    # the suffix-collision check does not fire on similar-looking names.
    class OkNamespaced(BaseStructure):
        """Ordinary fields under a namespace succeed."""

        my_eq: str = ""  # emits 'n__my_eq' — last 4 chars 'y_eq', not '__eq'
        court_type: str = ""

        class Schema:
            namespace = "n"

    build_manifest([OkNamespaced])


def test_invalid_namespace_contains_double_underscore() -> None:
    class Bad(BaseStructure):
        """Bad namespace."""

        x: str

        class Schema:
            namespace = "bad__ns"

    with pytest.raises(SchemaError, match="contains '__'"):
        build_manifest([Bad])


def test_invalid_namespace_not_identifier() -> None:
    class Bad2(BaseStructure):
        """Bad namespace."""

        x: str

        class Schema:
            namespace = "1bad"

    with pytest.raises(SchemaError, match="not a valid Python identifier"):
        build_manifest([Bad2])


def test_namespace_equal_to_reserved_operator_raises() -> None:
    class Bad3(BaseStructure):
        """Namespace matching operator name."""

        x: str

        class Schema:
            namespace = "in"

    with pytest.raises(SchemaError, match="collides with a filter operator"):
        build_manifest([Bad3])


def test_mixed_roots_structurals_and_semantics() -> None:
    class StructRoot(BaseStructure):
        """Struct."""

        x: str

    class SemRoot(BaseSemantic):
        """Sem?"""

    manifest = build_manifest([StructRoot, SemRoot])
    assert len(manifest.structurals()) == 1
    assert len(manifest.semantics()) == 1


def test_non_schema_root_raises() -> None:
    # Passing a class that inherits from neither base directly as a root
    # trips the root-validation branch in ``build_manifest``.
    class NotASchema:
        pass

    with pytest.raises(
        SchemaError,
        match="not a BaseStructure, BaseSemantic, or BaseCollection subclass",
    ):
        build_manifest([NotASchema])  # type: ignore[list-item]


def test_collection_extensions_must_be_schema_subclass() -> None:
    # A BaseCollection with an extensions entry that is not a valid base
    # schema subclass should raise at manifest build time.
    from ennoia.schema.base import BaseCollection

    class _NotASchema:
        pass

    class Coll(BaseCollection):
        """Docstring."""

        name: str

        class Schema:
            extensions = [_NotASchema]

    with pytest.raises(SchemaError, match="not a BaseStructure, BaseSemantic"):
        build_manifest([Coll])
