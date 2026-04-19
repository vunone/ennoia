"""``BaseStructure.confidence`` / ``BaseCollection.confidence`` property."""

from __future__ import annotations

import pytest

from ennoia import BaseCollection, BaseStructure


class _Plain(BaseStructure):
    """Plain structural with the default (1.0) confidence fallback."""

    label: str


class _Custom(BaseStructure):
    """Structural with a custom default_confidence."""

    label: str

    class Schema:
        default_confidence = 0.3


class _PlainCollection(BaseCollection):
    """Collection with the default (1.0) confidence fallback."""

    name: str


class _CustomCollection(BaseCollection):
    """Collection with a custom default_confidence."""

    name: str

    class Schema:
        default_confidence = 0.25


# ---------------------------------------------------------------------------
# BaseStructure.confidence
# ---------------------------------------------------------------------------


def test_structure_returns_extra_when_valid() -> None:
    instance = _Plain.model_validate({"label": "x", "extraction_confidence": 0.72})
    assert instance.confidence == 0.72


def test_structure_extra_at_boundary_accepted() -> None:
    # 0.0 and 1.0 are both valid.
    low = _Plain.model_validate({"label": "x", "extraction_confidence": 0.0})
    high = _Plain.model_validate({"label": "x", "extraction_confidence": 1.0})
    assert low.confidence == 0.0
    assert high.confidence == 1.0


def test_structure_missing_extra_falls_back_to_default() -> None:
    instance = _Plain.model_validate({"label": "x"})
    assert instance.confidence == 1.0


def test_structure_extra_out_of_range_falls_back() -> None:
    # The extractor normally strips these; direct construction still degrades
    # gracefully — the property ignores invalid extras.
    instance = _Plain.model_validate({"label": "x", "extraction_confidence": 1.5})
    assert instance.confidence == 1.0


def test_structure_extra_non_numeric_falls_back() -> None:
    instance = _Plain.model_validate({"label": "x", "extraction_confidence": "high"})
    assert instance.confidence == 1.0


def test_structure_extra_bool_is_ignored() -> None:
    # ``bool`` is a subclass of ``int`` in Python; the property must treat it
    # as invalid so ``True`` doesn't sneak through as ``1.0``.
    instance = _Plain.model_validate({"label": "x", "extraction_confidence": True})
    assert instance.confidence == 1.0


def test_structure_custom_default_confidence() -> None:
    instance = _Custom.model_validate({"label": "x"})
    assert instance.confidence == 0.3


def test_structure_custom_default_overridden_by_valid_extra() -> None:
    instance = _Custom.model_validate({"label": "x", "extraction_confidence": 0.9})
    assert instance.confidence == 0.9


def test_structure_pydantic_extra_none_uses_default() -> None:
    # A freshly constructed instance with no extras has ``__pydantic_extra__``
    # as an empty dict (not ``None``) under ``extra='allow'``, but we guard
    # against the ``None`` case anyway — simulate it explicitly.
    instance = _Plain(label="x")
    object.__setattr__(instance, "__pydantic_extra__", None)
    assert instance.confidence == 1.0


# ---------------------------------------------------------------------------
# BaseCollection.confidence
# ---------------------------------------------------------------------------


def test_collection_returns_extra_when_valid() -> None:
    instance = _PlainCollection.model_validate({"name": "a", "extraction_confidence": 0.66})
    assert instance.confidence == 0.66


def test_collection_missing_extra_falls_back_to_default() -> None:
    instance = _PlainCollection.model_validate({"name": "a"})
    assert instance.confidence == 1.0


def test_collection_custom_default_confidence() -> None:
    instance = _CustomCollection.model_validate({"name": "a"})
    assert instance.confidence == 0.25


def test_collection_invalid_extra_falls_back() -> None:
    instance = _CustomCollection.model_validate({"name": "a", "extraction_confidence": -0.5})
    assert instance.confidence == 0.25


# ---------------------------------------------------------------------------
# default_confidence validation surfaces via the property
# ---------------------------------------------------------------------------


class _BrokenDefault(BaseStructure):
    """Structural with an invalid default_confidence."""

    label: str

    class Schema:
        default_confidence = 1.5  # out of [0.0, 1.0]


def test_invalid_default_confidence_raises_through_property() -> None:
    instance = _BrokenDefault.model_validate({"label": "x"})
    with pytest.raises(ValueError, match="default_confidence"):
        _ = instance.confidence
