"""DAG layer construction and schema validation."""

from __future__ import annotations

import pytest

from ennoia import BaseSemantic, BaseStructure
from ennoia.index.dag import build_dag, validate_schemas


class A(BaseStructure):
    """A."""

    x: int


class B(BaseStructure):
    """B."""

    x: int


class S(BaseSemantic):
    """S."""


def test_build_dag_returns_single_layer():
    layers = build_dag([A, B, S])
    assert len(layers) == 1
    assert set(layers[0]) == {A, B, S}


def test_build_dag_empty_input():
    assert build_dag([]) == [[]]


def test_validate_schemas_accepts_structural_and_semantic():
    validate_schemas([A, S])


def test_validate_schemas_rejects_other_types():
    class NotASchema:
        pass

    with pytest.raises(TypeError, match="must inherit"):
        validate_schemas([NotASchema])  # type: ignore[list-item]
