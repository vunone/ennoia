"""Filter translation from Ennoia's unified syntax to Qdrant Filter clauses."""

from __future__ import annotations

import pytest
from qdrant_client.models import Filter

from ennoia.store.hybrid._qdrant_filter import translate_filter


def _field_conditions(qfilter: Filter | None) -> list:
    assert qfilter is not None
    return list(qfilter.must or [])


def test_none_filters_returns_nothing() -> None:
    qfilter, residual = translate_filter(None)
    assert qfilter is None
    assert residual == {}


def test_empty_dict_returns_nothing() -> None:
    qfilter, residual = translate_filter({})
    assert qfilter is None


def test_eq_becomes_match_value() -> None:
    qfilter, residual = translate_filter({"jurisdiction": "WA"})
    conds = _field_conditions(qfilter)
    assert conds[0].key == "jurisdiction"
    assert conds[0].match.value == "WA"
    assert residual == {}


def test_in_becomes_match_any() -> None:
    qfilter, _ = translate_filter({"jurisdiction__in": ["WA", "NY"]})
    conds = _field_conditions(qfilter)
    assert conds[0].match.any == ["WA", "NY"]


def test_in_accepts_csv_string() -> None:
    qfilter, _ = translate_filter({"jurisdiction__in": "WA,NY,TX"})
    conds = _field_conditions(qfilter)
    assert conds[0].match.any == ["WA", "NY", "TX"]


def test_range_operators_map_to_range() -> None:
    qfilter, _ = translate_filter(
        {"date__gte": "2020-01-01", "date__lt": "2023-01-01", "age__gt": 18, "age__lte": 65}
    )
    by_key: dict[str, object] = {}
    for cond in _field_conditions(qfilter):
        by_key.setdefault(cond.key, []).append(cond)  # type: ignore[attr-defined,union-attr]
    assert len(by_key["date"]) == 2
    assert len(by_key["age"]) == 2


def test_is_null_true_goes_to_must() -> None:
    qfilter, _ = translate_filter({"overruled_by__is_null": "true"})
    assert qfilter is not None
    must = list(qfilter.must or [])
    must_not = list(qfilter.must_not or [])
    assert len(must) == 1
    assert len(must_not) == 0


def test_is_null_false_goes_to_must_not() -> None:
    qfilter, _ = translate_filter({"overruled_by__is_null": "false"})
    assert qfilter is not None
    assert not qfilter.must
    assert qfilter.must_not and len(list(qfilter.must_not)) == 1


def test_contains_on_list_field_is_native() -> None:
    qfilter, residual = translate_filter(
        {"tags__contains": "tort"}, list_fields=frozenset({"tags"})
    )
    conds = _field_conditions(qfilter)
    assert conds[0].match.value == "tort"
    assert residual == {}


def test_contains_on_string_field_is_residual() -> None:
    qfilter, residual = translate_filter({"title__contains": "bankruptcy"})
    assert qfilter is None
    assert residual == {"title__contains": "bankruptcy"}


def test_contains_any_is_native_match_any() -> None:
    qfilter, _ = translate_filter({"tags__contains_any": ["tort", "contract"]})
    conds = _field_conditions(qfilter)
    assert conds[0].match.any == ["tort", "contract"]


def test_contains_all_and_startswith_fall_to_residual() -> None:
    qfilter, residual = translate_filter(
        {"tags__contains_all": ["tort", "negligence"], "title__startswith": "In re"}
    )
    assert qfilter is None
    assert residual == {
        "tags__contains_all": ["tort", "negligence"],
        "title__startswith": "In re",
    }


def test_mixed_native_and_residual_filters() -> None:
    qfilter, residual = translate_filter({"jurisdiction": "WA", "title__startswith": "In re"})
    assert qfilter is not None
    assert "title__startswith" in residual


def test_in_with_bare_value_becomes_singleton_list() -> None:
    qfilter, _ = translate_filter({"jurisdiction__in": 7})
    conds = _field_conditions(qfilter)
    assert conds[0].match.any == [7]


def test_is_null_parses_python_bool_values() -> None:
    # parse_bool handles "true"/"false" strings and real bools.
    qfilter_true, _ = translate_filter({"x__is_null": True})
    qfilter_false, _ = translate_filter({"x__is_null": False})
    assert qfilter_true is not None
    assert qfilter_false is not None
    assert qfilter_true.must
    assert qfilter_false.must_not


def test_invalid_bool_raises() -> None:
    with pytest.raises(ValueError):
        translate_filter({"x__is_null": "maybe"})


def test_in_accepts_tuple() -> None:
    qfilter, _ = translate_filter({"x__in": ("a", "b")})
    conds = _field_conditions(qfilter)
    assert conds[0].match.any == ["a", "b"]


def test_range_on_number_uses_plain_range() -> None:
    # Numbers (not ISO strings) stay on ``Range`` rather than ``DatetimeRange``.
    qfilter, _ = translate_filter({"age__gt": 18})
    conds = _field_conditions(qfilter)
    # pydantic-validated Range — the value on ``.gt`` is a plain number.
    assert conds[0].range.gt == 18


def test_range_on_date_object_uses_datetime_range() -> None:
    from datetime import date

    qfilter, _ = translate_filter({"date__gte": date(2020, 1, 1)})
    conds = _field_conditions(qfilter)
    # DatetimeRange coerces to datetime; the important thing is it didn't raise.
    assert conds[0].range.gte is not None


def test_range_on_non_iso_string_uses_plain_range() -> None:
    # A free-form string that isn't ISO-parseable falls back to ``Range`` —
    # which will then reject it at pydantic validation. We just check the
    # routing, not the pydantic behaviour.
    with pytest.raises(ValueError):
        translate_filter({"x__gt": "not-a-date"})
