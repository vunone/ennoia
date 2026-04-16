"""Superschema construction + type-compatibility merging."""

from __future__ import annotations

import warnings
from typing import ClassVar, Literal

import pytest
from pydantic import Field as PydField

from ennoia import BaseSemantic, BaseStructure
from ennoia.index.exceptions import SchemaError, SchemaWarning
from ennoia.schema.manifest import build_manifest
from ennoia.schema.merging import Superschema, build_superschema, merge_field_types

# ---------------------------------------------------------------------------
# merge_field_types: pure type-compatibility algorithm
# ---------------------------------------------------------------------------


def test_merge_identical_primitives() -> None:
    assert merge_field_types(str, str) is str
    assert merge_field_types(int, int) is int
    assert merge_field_types(bool, bool) is bool


def test_merge_incompatible_primitives_raises() -> None:
    with pytest.raises(SchemaError, match="Incompatible"):
        merge_field_types(str, int)


def test_merge_literal_unions_values_and_dedupes() -> None:
    merged = merge_field_types(Literal["A", "B"], Literal["B", "C"])
    assert list(merged.__args__) == ["A", "B", "C"]  # type: ignore[attr-defined]


def test_merge_literal_against_str_raises() -> None:
    with pytest.raises(SchemaError):
        merge_field_types(Literal["A"], str)


def test_merge_optional_absorbs_plain() -> None:
    merged = merge_field_types(str | None, str)
    assert merged == (str | None)


def test_merge_plain_absorbs_optional() -> None:
    merged = merge_field_types(str, str | None)
    assert merged == (str | None)


def test_merge_optional_optional_remains_optional() -> None:
    merged = merge_field_types(str | None, str | None)
    assert merged == (str | None)


def test_merge_list_inner_types_recursively() -> None:
    merged = merge_field_types(list[Literal["A"]], list[Literal["B"]])
    # list[Literal["A", "B"]]
    inner = merged.__args__[0]  # type: ignore[attr-defined]
    assert list(inner.__args__) == ["A", "B"]


def test_merge_list_inner_incompatible_raises() -> None:
    with pytest.raises(SchemaError, match="Incompatible"):
        merge_field_types(list[str], list[int])


def test_merge_deep_composition() -> None:
    a = list[Literal["A", "B"]] | None
    b = list[Literal["B", "C"]]
    merged = merge_field_types(a, b)
    # Optional[list[Literal["A", "B", "C"]]]
    assert type(None) in merged.__args__  # type: ignore[attr-defined]
    # Strip Optional and check inner Literal values
    non_none = [t for t in merged.__args__ if t is not type(None)]  # type: ignore[attr-defined]
    assert len(non_none) == 1
    inner_literal = non_none[0].__args__[0]
    assert list(inner_literal.__args__) == ["A", "B", "C"]


def test_merge_unparameterized_list_raises() -> None:
    with pytest.raises(SchemaError, match="unparameterized"):
        merge_field_types(list, list[str])


# ---------------------------------------------------------------------------
# build_superschema: end-to-end merging from a manifest
# ---------------------------------------------------------------------------


class _OneField(BaseStructure):
    """One filterable field."""

    category: Literal["a", "b"]


def test_single_source_field_superschema() -> None:
    manifest = build_manifest([_OneField])
    ss = build_superschema(manifest)
    assert set(ss.fields) == {"category"}
    sf = ss.fields["category"]
    assert sf.sources == ["_OneField"]
    assert sf.type_label == "enum"
    assert sf.options == ["a", "b"]
    assert sf.operators == ["eq", "in"]


class _OldCaseFormat(BaseStructure):
    """Historical case format."""

    citation: Literal["federal", "state"] = "federal"


class _NewCaseFormat(BaseStructure):
    """Modern case format."""

    citation: Literal["state", "international"] = "state"


class _CaseRoot(BaseStructure):
    """Root pulling in both case-format schemas."""

    kind: str

    class Schema:
        extensions: ClassVar[list[type]] = [_OldCaseFormat, _NewCaseFormat]


def test_multi_source_literal_merges_with_both_sources() -> None:
    manifest = build_manifest([_CaseRoot])
    ss = build_superschema(manifest)
    sf = ss.fields["citation"]
    # Two sources merged.
    assert sf.sources == ["_OldCaseFormat", "_NewCaseFormat"]
    # Literal union of values.
    assert sf.type_label == "enum"
    assert set(sf.options or []) == {"federal", "state", "international"}


class _StrField(BaseStructure):
    """String citation."""

    citation: str


class _IntField(BaseStructure):
    """Integer citation — incompatible with str above."""

    citation: int


class _IncompatibleRoot(BaseStructure):
    """Root pulling in two incompatible children."""

    header: str

    class Schema:
        extensions: ClassVar[list[type]] = [_StrField, _IntField]


def test_superschema_incompatible_types_raises_with_source_names() -> None:
    manifest = build_manifest([_IncompatibleRoot])
    with pytest.raises(SchemaError) as exc:
        build_superschema(manifest)
    msg = str(exc.value)
    assert "citation" in msg
    assert "_StrField" in msg and "_IntField" in msg


class _DescA(BaseStructure):
    """First source with a description."""

    title: str = PydField(default="", description="Case title as originally filed.")


class _DescB(BaseStructure):
    """Second source with a divergent description."""

    title: str = PydField(default="", description="Case title in modern style.")


class _DescRoot(BaseStructure):
    """Root pulling in both description sources."""

    label: str

    class Schema:
        extensions: ClassVar[list[type]] = [_DescA, _DescB]


def test_divergent_descriptions_warn_and_flag() -> None:
    manifest = build_manifest([_DescRoot])
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        ss = build_superschema(manifest)
    assert any(issubclass(w.category, SchemaWarning) for w in captured)
    sf = ss.fields["title"]
    assert sf.has_divergent_descriptions is True
    # First-declared wins.
    assert sf.description == "Case title as originally filed."


class _Equal1(BaseStructure):
    """Matching description."""

    label: str = PydField(default="", description="Same description.")


class _Equal2(BaseStructure):
    """Matching description."""

    label: str = PydField(default="", description="Same description.")


class _EqualRoot(BaseStructure):
    """Two sources with matching description — no warning."""

    header: str

    class Schema:
        extensions: ClassVar[list[type]] = [_Equal1, _Equal2]


def test_matching_descriptions_no_warning() -> None:
    manifest = build_manifest([_EqualRoot])
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        ss = build_superschema(manifest)
    assert not any(issubclass(w.category, SchemaWarning) for w in captured)
    assert ss.fields["label"].has_divergent_descriptions is False


class _SemanticHolding(BaseSemantic):
    """What is the holding?"""


class _WithSemantic(BaseStructure):
    """Structural that pulls in a semantic extension."""

    case_id: str

    class Schema:
        extensions: ClassVar[list[type]] = [_SemanticHolding]


def test_semantic_indices_surface_in_superschema() -> None:
    manifest = build_manifest([_WithSemantic])
    ss = build_superschema(manifest)
    names = [idx["name"] for idx in ss.semantic_indices]
    assert names == ["_SemanticHolding"]
    assert ss.semantic_indices[0]["description"] == "What is the holding?"


# ---------------------------------------------------------------------------
# Discovery payload shape matches spec
# ---------------------------------------------------------------------------


def test_discovery_payload_shape_matches_spec_example() -> None:
    manifest = build_manifest([_CaseRoot])
    ss = build_superschema(manifest)
    payload = ss.to_discovery_payload()
    assert set(payload) == {"structural_fields", "semantic_indices"}
    citation = next(f for f in payload["structural_fields"] if f["name"] == "citation")
    assert citation["type"] == "enum"
    assert "options" in citation
    assert citation["sources"] == ["_OldCaseFormat", "_NewCaseFormat"]
    # Multi-source → key present (value may be False when no divergence).
    assert "has_divergent_descriptions" in citation


def test_single_source_discovery_omits_multi_source_keys() -> None:
    manifest = build_manifest([_OneField])
    ss = build_superschema(manifest)
    payload = ss.to_discovery_payload()
    field = payload["structural_fields"][0]
    assert "has_divergent_descriptions" not in field


def test_superschema_field_names_getter() -> None:
    manifest = build_manifest([_CaseRoot])
    ss = build_superschema(manifest)
    assert ss.all_field_names() == {"kind", "citation"}


def test_empty_manifest_produces_empty_superschema() -> None:
    manifest = build_manifest([])
    ss = build_superschema(manifest)
    assert ss.fields == {}
    assert ss.semantic_indices == []
    assert isinstance(ss, Superschema)
