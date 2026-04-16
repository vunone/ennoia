"""Translation from Ennoia's unified filter syntax to a parameterised WHERE clause."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from ennoia.store.hybrid._sql_filter import build_where


def test_empty_filters_return_true_sentinel() -> None:
    assert build_where(None) == ("TRUE", [])
    assert build_where({}) == ("TRUE", [])


def test_eq_produces_text_compare() -> None:
    sql, params = build_where({"jurisdiction": "WA"})
    assert "(data->>'jurisdiction') = $1::text" in sql
    assert params == ["WA"]


def test_in_produces_positional_placeholders() -> None:
    sql, params = build_where({"jurisdiction__in": ["WA", "NY"]})
    assert "IN ($1::text, $2::text)" in sql
    assert params == ["WA", "NY"]


def test_in_with_csv_string_splits() -> None:
    sql, params = build_where({"jurisdiction__in": "WA,NY,TX"})
    assert params == ["WA", "NY", "TX"]


def test_in_empty_short_circuits_to_false() -> None:
    sql, params = build_where({"x__in": []})
    assert "FALSE" in sql
    assert params == []


def test_gt_on_int_uses_numeric_cast() -> None:
    sql, params = build_where({"age__gt": 18})
    assert "::numeric" in sql
    assert params == [18]


def test_gt_on_float_uses_numeric_cast() -> None:
    sql, params = build_where({"score__gte": 0.8})
    assert "::numeric" in sql
    assert params == [0.8]


def test_gte_on_date_object_uses_date_cast() -> None:
    sql, params = build_where({"date__gte": date(2020, 1, 1)})
    assert "::date" in sql
    assert params == ["2020-01-01"]


def test_lt_on_datetime_object_uses_timestamp_cast() -> None:
    sql, params = build_where({"when__lt": datetime(2020, 1, 1, 12, 0)})
    assert "::timestamp" in sql
    assert params == ["2020-01-01T12:00:00"]


def test_range_on_iso_string_uses_timestamp_cast() -> None:
    sql, _ = build_where({"when__gte": "2020-01-01"})
    assert "::timestamp" in sql


def test_range_on_numeric_string_uses_numeric_cast() -> None:
    sql, _ = build_where({"age__gt": "42"})
    assert "::numeric" in sql


def test_range_on_free_form_string_uses_text_cast() -> None:
    sql, _ = build_where({"x__gt": "apple"})
    assert "::text" in sql


def test_range_on_bool_uses_int_cast() -> None:
    sql, params = build_where({"flag__gt": True})
    assert "::int" in sql
    assert params == [1]


def test_contains_emits_both_array_and_ilike_clauses() -> None:
    sql, params = build_where({"title__contains": "bankruptcy"})
    assert "jsonb_typeof" in sql
    assert "ILIKE" in sql
    assert params == ["bankruptcy", "%bankruptcy%"]


def test_startswith_anchors_like() -> None:
    sql, params = build_where({"title__startswith": "In re"})
    assert "LIKE $1" in sql
    assert params == ["In re%"]


def test_contains_all_uses_jsonb_contains_operator() -> None:
    sql, params = build_where({"tags__contains_all": ["tort", "negligence"]})
    assert "@>" in sql
    assert "::jsonb" in sql
    assert params == ['["tort", "negligence"]']


def test_contains_all_empty_short_circuits_to_true() -> None:
    sql, _ = build_where({"x__contains_all": []})
    assert "TRUE" in sql


def test_contains_any_uses_jsonb_any_of_keys() -> None:
    sql, params = build_where({"tags__contains_any": ["tort", "contract"]})
    assert "?|" in sql
    assert "::text[]" in sql
    assert params == [["tort", "contract"]]


def test_contains_any_empty_short_circuits_to_false() -> None:
    sql, _ = build_where({"x__contains_any": []})
    assert "FALSE" in sql


def test_is_null_true_emits_is_null() -> None:
    sql, _ = build_where({"overruled_by__is_null": "true"})
    assert "IS NULL" in sql
    assert "NOT" not in sql


def test_is_null_false_emits_not_is_null() -> None:
    sql, _ = build_where({"overruled_by__is_null": "false"})
    assert "NOT" in sql and "IS NULL" in sql


def test_multiple_filters_compose_with_and() -> None:
    sql, params = build_where({"jurisdiction": "WA", "age__gt": 18})
    assert " AND " in sql
    assert params == ["WA", 18]


def test_invalid_bool_raises() -> None:
    with pytest.raises(ValueError):
        build_where({"x__is_null": "maybe"})


def test_tuple_values_normalised_to_list() -> None:
    sql, params = build_where({"x__in": ("a", "b")})
    assert params == ["a", "b"]
    # also cover contains_all via tuple
    sql2, params2 = build_where({"tags__contains_all": ("a",)})
    assert "::jsonb" in sql2
    assert params2 == ['["a"]']


def test_single_non_iter_value_coerces_to_singleton() -> None:
    _, params = build_where({"x__in": 7})
    assert params == ["7"]


def test_range_on_unknown_type_falls_back_to_text() -> None:
    # Passing a non-scalar object (e.g. None) should route through the final
    # ``text`` fallback rather than crash.
    sql, params = build_where({"x__gt": None})
    assert "::text" in sql
    assert params == ["None"]
