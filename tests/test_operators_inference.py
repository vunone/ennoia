"""Operator inference + schema discovery — FILTER_SPECS.md inference table."""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Literal, Optional

import pytest

from ennoia import BaseSemantic, BaseStructure, Field, describe
from ennoia.schema.operators import describe_field, infer_operators


@pytest.mark.parametrize(
    "annotation,expected",
    [
        (bool, ["eq"]),
        (str, ["eq", "contains", "startswith"]),
        (int, ["eq", "gt", "gte", "lt", "lte"]),
        (float, ["eq", "gt", "gte", "lt", "lte"]),
        (date, ["eq", "gt", "gte", "lt", "lte"]),
        (datetime, ["eq", "gt", "gte", "lt", "lte"]),
        (Literal["a", "b"], ["eq", "in"]),
        (list[str], ["contains", "contains_all", "contains_any"]),
    ],
)
def test_infer_operators_for_core_types(annotation: object, expected: list[str]) -> None:
    assert infer_operators(annotation) == expected


def test_optional_adds_is_null_to_inner_operators() -> None:
    assert infer_operators(Optional[str]) == [
        "eq",
        "contains",
        "startswith",
        "is_null",
    ]
    assert infer_operators(int | None) == ["eq", "gt", "gte", "lt", "lte", "is_null"]


class CaseDocument(BaseStructure):
    """Extract case metadata."""

    jurisdiction: Literal["WA", "NY", "TX"]
    date_decided: date
    court_level: Literal["supreme", "appellate", "district"]
    is_overruled: bool
    tags: list[str]
    overruled_by: Optional[str] = None


class Holding(BaseSemantic):
    """What is the core holding of this case?"""


def test_describe_schema_matches_filter_specs_reference() -> None:
    payload = describe([CaseDocument, Holding])

    by_name = {f["name"]: f for f in payload["structural_fields"]}
    assert by_name["jurisdiction"]["type"] == "enum"
    assert by_name["jurisdiction"]["options"] == ["WA", "NY", "TX"]
    assert by_name["jurisdiction"]["operators"] == ["eq", "in"]

    assert by_name["tags"]["type"] == "list"
    assert by_name["tags"]["item_type"] == "str"
    assert by_name["tags"]["operators"] == ["contains", "contains_all", "contains_any"]

    assert by_name["overruled_by"]["nullable"] is True
    assert "is_null" in by_name["overruled_by"]["operators"]

    assert payload["semantic_indices"] == [
        {
            "name": "Holding",
            "description": "What is the core holding of this case?",
            "kind": "semantic",
        }
    ]


class OverrideDoc(BaseStructure):
    """Override-driven field surface."""

    title: Annotated[str, Field(description="Title", operators=["eq", "contains"])] = ""
    note: Annotated[str, Field(filterable=False)] = ""


def test_field_override_restricts_operators() -> None:
    info = OverrideDoc.model_fields["title"]
    record = describe_field("title", info)
    assert record is not None
    assert record["operators"] == ["eq", "contains"]


def test_field_override_excludes_from_discovery() -> None:
    payload = describe([OverrideDoc])
    names = [f["name"] for f in payload["structural_fields"]]
    assert "note" not in names
    assert "title" in names


# ---------------------------------------------------------------------------
# type_label coverage — primitive + container + fallback branches
# ---------------------------------------------------------------------------

from ennoia.schema.operators import (  # noqa: E402
    field_metadata,
    type_label,
    unwrap_optional,
)


def test_type_label_primitives_render_correct_names() -> None:
    assert type_label(int) == ("int", {})
    assert type_label(float) == ("float", {})
    assert type_label(datetime) == ("datetime", {})
    assert type_label(bool) == ("bool", {})
    assert type_label(str) == ("str", {})
    assert type_label(date) == ("date", {})


def test_type_label_bare_list_returns_empty_extras() -> None:
    assert type_label(list) == ("list", {})


def test_type_label_list_with_item_type_extras() -> None:
    label, extras = type_label(list[str])
    assert label == "list"
    assert extras == {"item_type": "str"}


def test_type_label_unknown_type_falls_back_to_name() -> None:
    # A type that doesn't match any branch falls through to the generic name
    # lookup — ``bytes`` has a ``__name__`` attribute.
    assert type_label(bytes) == ("bytes", {})


# ---------------------------------------------------------------------------
# unwrap_optional — unsupported union shape path
# ---------------------------------------------------------------------------


def test_unwrap_optional_on_unsupported_union_returns_annotation_as_is() -> None:
    # A union with 2+ non-None members (``int | str``) can't be narrowed;
    # the helper returns the original annotation and ``nullable=False``.
    inner, nullable = unwrap_optional(int | str)
    assert inner == (int | str)
    assert nullable is False


def test_unwrap_optional_on_multi_arg_union_with_none_flags_nullable() -> None:
    # ``int | str | None`` — ambiguous non-None shape, but the None presence
    # still flags ``nullable=True``.
    inner, nullable = unwrap_optional(int | str | None)
    assert nullable is True
    # inner is the unchanged annotation because the shape isn't ``T | None``.
    assert inner == (int | str | None)


# ---------------------------------------------------------------------------
# field_metadata — non-dict ``ennoia`` metadata is ignored
# ---------------------------------------------------------------------------


def test_field_metadata_ignores_non_dict_ennoia_entry() -> None:
    # Construct a FieldInfo whose json_schema_extra has ``ennoia`` set to a
    # non-dict value — the helper must return an empty dict, not crash.
    from pydantic import Field as PydField
    from pydantic.fields import FieldInfo

    info = FieldInfo.from_annotation(str)
    info.json_schema_extra = {"ennoia": "not-a-dict"}
    assert field_metadata(info) == {}

    # Sanity: dict-typed metadata is still read.
    info2 = FieldInfo.from_annotated_attribute(
        str, PydField(json_schema_extra={"ennoia": {"filterable": False}})
    )
    assert field_metadata(info2) == {"filterable": False}


# ---------------------------------------------------------------------------
# describe_field on nullable field — exercises the is_null + nullable block
# ---------------------------------------------------------------------------


class _WithNullable(BaseStructure):
    """Schema carrying a nullable field."""

    overruled_by: Optional[str] = None
    plain_int: int = 0
    plain_float: float = 0.0


def test_describe_schema_handles_nullable_field() -> None:
    payload = _WithNullable.describe_schema()
    by_name = {f["name"]: f for f in payload["fields"]}
    assert by_name["overruled_by"]["nullable"] is True
    assert "is_null" in by_name["overruled_by"]["operators"]
    # Non-nullable primitives still render the right type labels.
    assert by_name["plain_int"]["type"] == "int"
    assert by_name["plain_float"]["type"] == "float"


def test_describe_field_nullable_inferred_operators_without_duplicate_is_null() -> None:
    from ennoia.schema.operators import describe_field

    info = _WithNullable.model_fields["overruled_by"]
    record = describe_field("overruled_by", info)
    assert record is not None
    # ``is_null`` appears exactly once even though both infer_operators and the
    # nullable-branch would add it.
    assert record["operators"].count("is_null") == 1


class _NullableWithOverride(BaseStructure):
    """Nullable field with explicit operators override lacking is_null."""

    code: Annotated[Optional[str], Field(default=None, operators=["eq", "contains"])]


def test_describe_field_nullable_override_appends_is_null() -> None:
    from ennoia.schema.operators import describe_field

    info = _NullableWithOverride.model_fields["code"]
    record = describe_field("code", info)
    assert record is not None
    # Override starts without is_null → nullable branch appends it.
    assert record["operators"] == ["eq", "contains", "is_null"]
    assert record["nullable"] is True


def test_type_label_generic_alias_list_with_empty_args_returns_label_only() -> None:
    # ``types.GenericAlias(list, ())`` has origin ``list`` but no args — the
    # helper must return ``("list", {})`` with no ``item_type`` extras.
    import types

    from ennoia.schema.operators import type_label as tl

    weird = types.GenericAlias(list, ())
    assert tl(weird) == ("list", {})
