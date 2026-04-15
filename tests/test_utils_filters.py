"""Unit tests for ennoia.utils.filters — the shared structured filter engine."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pytest

from ennoia.utils.filters import (
    KNOWN_OPERATORS,
    apply_filters,
    coerce_filter_value,
    evaluate_condition,
    split_filter_key,
)

# ---------------------------------------------------------------------------
# KNOWN_OPERATORS
# ---------------------------------------------------------------------------


def test_known_operators_match_full_spec():
    assert (
        frozenset(
            {
                "eq",
                "in",
                "gt",
                "gte",
                "lt",
                "lte",
                "contains",
                "startswith",
                "contains_all",
                "contains_any",
                "is_null",
            }
        )
        == KNOWN_OPERATORS
    )


# ---------------------------------------------------------------------------
# split_filter_key
# ---------------------------------------------------------------------------


def test_split_filter_key_defaults_to_eq():
    assert split_filter_key("category") == ("category", "eq")


@pytest.mark.parametrize("op", sorted(KNOWN_OPERATORS))
def test_split_filter_key_recognises_every_known_suffix(op: str):
    assert split_filter_key(f"doc_date__{op}") == ("doc_date", op)


def test_split_filter_key_ignores_unknown_suffix():
    # Unknown suffix must stay part of the field name per FILTER_SPECS.md.
    assert split_filter_key("weird__nope") == ("weird__nope", "eq")


def test_split_filter_key_rpartition_handles_double_underscore_fields():
    assert split_filter_key("parent__child__gt") == ("parent__child", "gt")


@pytest.mark.parametrize("key", ["", "__eq", "__in", "__lte"])
def test_split_filter_key_rejects_empty_field_name(key: str):
    with pytest.raises(ValueError, match="Empty field name"):
        split_filter_key(key)


def test_split_filter_key_accepts_double_underscore_standalone():
    # "__" has no known operator suffix, so the whole thing is the field name.
    assert split_filter_key("__") == ("__", "eq")


# ---------------------------------------------------------------------------
# coerce_filter_value
# ---------------------------------------------------------------------------


def test_coerce_date_string_to_date():
    field, filt = coerce_filter_value(date(2024, 1, 1), "2020-01-01")
    assert field == date(2024, 1, 1)
    assert filt == date(2020, 1, 1)


def test_coerce_datetime_string_to_datetime():
    field, filt = coerce_filter_value(datetime(2024, 1, 1, 12, 0), "2020-01-01T00:00:00")
    assert field == datetime(2024, 1, 1, 12, 0)
    assert filt == datetime(2020, 1, 1, 0, 0)


def test_coerce_datetime_field_is_not_treated_as_date():
    # datetime is a subclass of date — the first branch must exclude it so the
    # value is parsed with the time component preserved, not truncated.
    field, filt = coerce_filter_value(datetime(2024, 1, 1, 12, 0), "2020-01-01T05:30:00")
    assert filt == datetime(2020, 1, 1, 5, 30)


def test_coerce_preserves_already_typed_values():
    assert coerce_filter_value(date(2024, 1, 1), date(2020, 1, 1)) == (
        date(2024, 1, 1),
        date(2020, 1, 1),
    )
    assert coerce_filter_value(datetime(2024, 1, 1, 12, 0), datetime(2020, 1, 1, 0, 0)) == (
        datetime(2024, 1, 1, 12, 0),
        datetime(2020, 1, 1, 0, 0),
    )


@pytest.mark.parametrize(
    ("field_value", "filter_value"),
    [
        (5, 3),
        ("a", "b"),
        (None, None),
        (None, 5),
        (5, None),
        (True, False),
        (3.14, "3.14"),  # numeric fields don't auto-parse strings
    ],
)
def test_coerce_passes_through_non_date_values(field_value: Any, filter_value: Any):
    assert coerce_filter_value(field_value, filter_value) == (field_value, filter_value)


def test_coerce_raises_on_malformed_iso_string():
    # Document: invalid ISO strings surface the underlying ValueError rather
    # than silently passing through. Callers (Stage 2 validation layer) are
    # expected to sanitise before reaching this helper.
    with pytest.raises(ValueError):
        coerce_filter_value(date(2024, 1, 1), "not-a-date")
    with pytest.raises(ValueError):
        coerce_filter_value(datetime(2024, 1, 1), "not-a-datetime")


# ---------------------------------------------------------------------------
# evaluate_condition — base behaviour
# ---------------------------------------------------------------------------


def test_evaluate_condition_missing_field_is_false():
    assert evaluate_condition({"a": 1}, "missing", "eq", 1) is False


@pytest.mark.parametrize(
    ("op", "value", "expected"),
    [
        ("eq", 5, True),
        ("eq", 4, False),
        ("gt", 4, True),
        ("gt", 5, False),
        ("gte", 5, True),
        ("gte", 6, False),
        ("lt", 6, True),
        ("lt", 5, False),
        ("lte", 5, True),
        ("lte", 4, False),
    ],
)
def test_evaluate_condition_numeric_operators(op: str, value: int, expected: bool):
    assert evaluate_condition({"x": 5}, "x", op, value) is expected


def test_evaluate_condition_rejects_unknown_operator():
    with pytest.raises(ValueError, match="Unknown filter operator"):
        evaluate_condition({"x": 1}, "x", "bogus", 1)


# ---------------------------------------------------------------------------
# evaluate_condition — date/datetime coercion through ordering ops
# ---------------------------------------------------------------------------


def test_evaluate_condition_date_field_with_string_value():
    record = {"d": date(2024, 6, 1)}
    assert evaluate_condition(record, "d", "gte", "2024-01-01") is True
    assert evaluate_condition(record, "d", "lt", "2024-01-01") is False


def test_evaluate_condition_datetime_field_with_string_value():
    record = {"d": datetime(2024, 6, 1, 12, 0)}
    assert evaluate_condition(record, "d", "gt", "2024-06-01T00:00:00") is True


# ---------------------------------------------------------------------------
# evaluate_condition — "in" operator
# ---------------------------------------------------------------------------


def test_evaluate_condition_in_with_list():
    assert evaluate_condition({"c": "legal"}, "c", "in", ["legal", "medical"]) is True
    assert evaluate_condition({"c": "other"}, "c", "in", ["legal", "medical"]) is False


def test_evaluate_condition_in_splits_comma_strings():
    # String values on both sides are comma-split for CLI ergonomics.
    assert evaluate_condition({"c": "legal"}, "c", "in", "legal, medical") is True
    assert evaluate_condition({"c": "other"}, "c", "in", "legal, medical") is False


def test_evaluate_condition_in_coerces_date_strings():
    # Date field against ISO-string list must match after per-candidate coercion.
    record = {"d": date(2024, 1, 1)}
    assert evaluate_condition(record, "d", "in", ["2024-01-01", "2025-01-01"]) is True
    assert evaluate_condition(record, "d", "in", ["2020-01-01"]) is False


def test_evaluate_condition_in_coerces_datetime_comma_string():
    record = {"d": datetime(2024, 1, 1, 0, 0)}
    assert evaluate_condition(record, "d", "in", "2024-01-01T00:00:00, 2025-01-01T00:00:00") is True


def test_evaluate_condition_in_with_tuple_and_generator():
    assert evaluate_condition({"x": 2}, "x", "in", (1, 2, 3)) is True
    assert evaluate_condition({"x": 2}, "x", "in", (i for i in [1, 2, 3])) is True


def test_evaluate_condition_in_with_numeric_list():
    assert evaluate_condition({"x": 5}, "x", "in", [1, 5, 9]) is True
    assert evaluate_condition({"x": 4}, "x", "in", [1, 5, 9]) is False


@pytest.mark.parametrize("bad_value", [5, None, 3.14, True, {"a": 1}, b"bytes"])
def test_evaluate_condition_in_rejects_non_iterable_or_ambiguous(bad_value: Any):
    with pytest.raises(ValueError, match="requires an iterable"):
        evaluate_condition({"x": 1}, "x", "in", bad_value)


# ---------------------------------------------------------------------------
# evaluate_condition — None handling on ordering operators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", ["gt", "gte", "lt", "lte"])
def test_evaluate_condition_none_record_value_short_circuits_ordering_ops(op: str):
    assert evaluate_condition({"x": None}, "x", op, 5) is False


@pytest.mark.parametrize("op", ["gt", "gte", "lt", "lte"])
def test_evaluate_condition_none_filter_value_short_circuits_ordering_ops(op: str):
    assert evaluate_condition({"x": 5}, "x", op, None) is False


def test_evaluate_condition_eq_handles_none_symmetrically():
    assert evaluate_condition({"x": None}, "x", "eq", None) is True
    assert evaluate_condition({"x": None}, "x", "eq", 5) is False
    assert evaluate_condition({"x": 5}, "x", "eq", None) is False


# ---------------------------------------------------------------------------
# apply_filters
# ---------------------------------------------------------------------------


def test_apply_filters_empty_query_returns_all_ids_in_order():
    records = [("a", {"x": 1}), ("b", {"x": 2})]
    assert apply_filters(records, {}) == ["a", "b"]


def test_apply_filters_composes_conditions_with_and():
    records = [
        ("a", {"cat": "legal", "year": 2024}),
        ("b", {"cat": "legal", "year": 2019}),
        ("c", {"cat": "medical", "year": 2024}),
    ]
    assert apply_filters(records, {"cat": "legal", "year__gte": 2020}) == ["a"]


def test_apply_filters_skips_records_missing_fields():
    records = [("a", {"x": 1}), ("b", {})]
    assert apply_filters(records, {"x": 1}) == ["a"]


def test_apply_filters_consumes_generator_exactly_once():
    def gen():
        yield ("a", {"x": 1})
        yield ("b", {"x": 2})

    assert apply_filters(gen(), {"x__gte": 1}) == ["a", "b"]


def test_apply_filters_propagates_split_filter_key_errors():
    with pytest.raises(ValueError, match="Empty field name"):
        apply_filters([("a", {"x": 1})], {"__eq": 1})


def test_apply_filters_handles_mixed_missing_and_present_fields():
    records = [
        ("a", {"cat": "legal", "year": 2024}),
        ("b", {"cat": "legal"}),  # missing year
        ("c", {"year": 2024}),  # missing cat
    ]
    assert apply_filters(records, {"cat": "legal", "year__gte": 2020}) == ["a"]
