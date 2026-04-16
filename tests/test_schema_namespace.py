"""Namespaced field merging behavior."""

from __future__ import annotations

from typing import ClassVar, Literal

from ennoia import BaseStructure
from ennoia.schema.manifest import build_manifest
from ennoia.schema.merging import build_superschema


class _WashingtonDetails(BaseStructure):
    """Washington-specific fields."""

    court_type: Literal["appellate", "supreme", "district"] = "district"
    case_number: str = ""

    class Schema:
        namespace = "wa"


class _FlatRoot(BaseStructure):
    """Flat root pulling in a namespaced child."""

    jurisdiction: Literal["WA", "NY", "TX"]

    class Schema:
        extensions: ClassVar[list[type]] = [_WashingtonDetails]


def test_flat_merge_default() -> None:
    class OnlyFlat(BaseStructure):
        """No namespace — fields appear at top level."""

        foo: str

    ss = build_superschema(build_manifest([OnlyFlat]))
    assert "foo" in ss.fields
    assert ss.fields["foo"].sources == ["OnlyFlat"]


def test_namespaced_merge_prefixes_fields() -> None:
    ss = build_superschema(build_manifest([_FlatRoot]))
    assert "jurisdiction" in ss.fields
    assert "wa__court_type" in ss.fields
    assert "wa__case_number" in ss.fields
    # Unprefixed field names must NOT appear when the source was namespaced.
    assert "court_type" not in ss.fields
    assert "case_number" not in ss.fields


def test_namespace_does_not_inherit_into_descendants() -> None:
    class Grandchild(BaseStructure):
        """No namespace — fields flat."""

        grand_field: str

    class NamespacedParent(BaseStructure):
        """Namespace applies only to its own fields."""

        own_field: str

        class Schema:
            namespace = "p"
            extensions: ClassVar[list[type]] = [Grandchild]

    ss = build_superschema(build_manifest([NamespacedParent]))
    # Parent's own field is namespaced.
    assert "p__own_field" in ss.fields
    # Grandchild's field is NOT under "p__" — namespace doesn't propagate.
    assert "grand_field" in ss.fields
    assert "p__grand_field" not in ss.fields


def test_namespaced_field_never_collides_with_flat() -> None:
    class FlatWithName(BaseStructure):
        """Top-level flat field 'court_type'."""

        court_type: str

    class NsWithSameName(BaseStructure):
        """Same field name under a namespace."""

        court_type: str = ""

        class Schema:
            namespace = "wa"

    class Root(BaseStructure):
        """Root pulling in both."""

        header: str

        class Schema:
            extensions: ClassVar[list[type]] = [FlatWithName, NsWithSameName]

    ss = build_superschema(build_manifest([Root]))
    # Both exist side by side.
    assert "court_type" in ss.fields
    assert "wa__court_type" in ss.fields
    assert ss.fields["court_type"].sources == ["FlatWithName"]
    assert ss.fields["wa__court_type"].sources == ["NsWithSameName"]


def test_two_namespaces_same_field_name_both_present() -> None:
    class WA(BaseStructure):
        """Washington."""

        court: str = ""

        class Schema:
            namespace = "wa"

    class NY(BaseStructure):
        """New York."""

        court: str = ""

        class Schema:
            namespace = "ny"

    class DualRoot(BaseStructure):
        """Root pulling in both jurisdictions."""

        header: str

        class Schema:
            extensions: ClassVar[list[type]] = [WA, NY]

    ss = build_superschema(build_manifest([DualRoot]))
    assert "wa__court" in ss.fields
    assert "ny__court" in ss.fields
