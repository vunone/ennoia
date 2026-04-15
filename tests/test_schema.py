"""Schema declaration layer."""

from __future__ import annotations

from typing import Literal

import pytest

from ennoia import BaseSemantic, BaseStructure, Field


class Meta(BaseStructure):
    """Extract document metadata."""

    category: Literal["legal", "medical"]
    title: str = Field(description="The document title.")


class Summary(BaseSemantic):
    """What is the main topic?"""


def test_structural_extract_prompt_is_docstring():
    assert Meta.extract_prompt() == "Extract document metadata."


def test_semantic_extract_prompt_is_docstring():
    assert Summary.extract_prompt() == "What is the main topic?"


def test_structural_json_schema_delegates_to_pydantic():
    schema = Meta.json_schema()
    assert schema == Meta.model_json_schema()
    assert schema["properties"]["category"]["enum"] == ["legal", "medical"]
    assert schema["properties"]["title"]["description"] == "The document title."
    assert set(schema["required"]) == {"category", "title"}


def test_extend_default_returns_empty():
    assert Meta(category="legal", title="X").extend() == []


def test_missing_docstring_raises():
    class NoDoc(BaseStructure):
        field: str

    NoDoc.__doc__ = None
    with pytest.raises(ValueError, match="docstring"):
        NoDoc.extract_prompt()
