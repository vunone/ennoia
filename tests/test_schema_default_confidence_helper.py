"""``get_schema_default_confidence`` helper unit tests."""

from __future__ import annotations

import pytest

from ennoia.schema.base import get_schema_default_confidence


class _NoInnerSchema:
    pass


class _Valid:
    class Schema:
        default_confidence = 0.4


class _IntDefault:
    class Schema:
        default_confidence = 1  # must be coerced to 1.0


class _OutOfRange:
    class Schema:
        default_confidence = 1.5


class _NegativeOutOfRange:
    class Schema:
        default_confidence = -0.1


class _NonNumeric:
    class Schema:
        default_confidence = "high"  # type: ignore[assignment]


class _BoolValue:
    class Schema:
        default_confidence = True  # type: ignore[assignment]


class _UnsetAttribute:
    class Schema:
        pass


def test_returns_one_when_no_inner_schema() -> None:
    assert get_schema_default_confidence(_NoInnerSchema) == 1.0


def test_returns_one_when_attribute_missing() -> None:
    assert get_schema_default_confidence(_UnsetAttribute) == 1.0


def test_returns_declared_float_value() -> None:
    assert get_schema_default_confidence(_Valid) == 0.4


def test_coerces_int_to_float() -> None:
    result = get_schema_default_confidence(_IntDefault)
    assert result == 1.0
    assert isinstance(result, float)


def test_raises_when_value_above_one() -> None:
    with pytest.raises(ValueError, match="default_confidence"):
        get_schema_default_confidence(_OutOfRange)


def test_raises_when_value_below_zero() -> None:
    with pytest.raises(ValueError, match="default_confidence"):
        get_schema_default_confidence(_NegativeOutOfRange)


def test_raises_when_value_non_numeric() -> None:
    with pytest.raises(TypeError, match="default_confidence"):
        get_schema_default_confidence(_NonNumeric)


def test_raises_when_value_is_bool() -> None:
    with pytest.raises(TypeError, match="default_confidence"):
        get_schema_default_confidence(_BoolValue)
