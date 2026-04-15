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
        {"name": "Holding", "description": "What is the core holding of this case?"}
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
