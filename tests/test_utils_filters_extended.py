"""Stage 2 operator coverage — contains, startswith, contains_all/any, is_null."""

from __future__ import annotations

from typing import Any

import pytest

from ennoia.utils.filters import apply_filters, evaluate_condition


@pytest.mark.parametrize(
    "record,field,op,value,expected",
    [
        ({"title": "In re Smith"}, "title", "startswith", "In re", True),
        ({"title": "Smith v. Jones"}, "title", "startswith", "In re", False),
        ({"title": "bankruptcy filing"}, "title", "contains", "bankruptcy", True),
        ({"title": "ruling"}, "title", "contains", "bankruptcy", False),
        ({"tags": ["tort", "negligence"]}, "tags", "contains", "tort", True),
        ({"tags": ["tort"]}, "tags", "contains_all", "tort,negligence", False),
        (
            {"tags": ["tort", "negligence", "civil"]},
            "tags",
            "contains_all",
            "tort,negligence",
            True,
        ),
        ({"tags": ["civil"]}, "tags", "contains_any", "tort,contract", False),
        ({"tags": ["contract", "civil"]}, "tags", "contains_any", "tort,contract", True),
        ({"overruled_by": None}, "overruled_by", "is_null", True, True),
        ({"overruled_by": "doc_9"}, "overruled_by", "is_null", True, False),
        ({"overruled_by": "doc_9"}, "overruled_by", "is_null", False, True),
    ],
)
def test_evaluate_condition_extended(
    record: dict[str, Any], field: str, op: str, value: Any, expected: bool
) -> None:
    assert evaluate_condition(record, field, op, value) is expected


def test_apply_filters_combines_extended_operators() -> None:
    rows = [
        ("a", {"title": "In re Smith", "tags": ["tort", "negligence"]}),
        ("b", {"title": "In re Jones", "tags": ["contract"]}),
        ("c", {"title": "Jane Doe", "tags": ["tort"]}),
    ]
    result = apply_filters(rows, {"title__startswith": "In re", "tags__contains": "tort"})
    assert result == ["a"]


def test_is_null_treats_missing_field_as_null() -> None:
    # A filter ``field__is_null=true`` should match a record lacking the field entirely.
    assert evaluate_condition({}, "maybe", "is_null", True) is True
    assert evaluate_condition({}, "maybe", "is_null", False) is False
