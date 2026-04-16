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


# ---------------------------------------------------------------------------
# get_schema_namespace / get_schema_extensions / describe_schema
# ---------------------------------------------------------------------------


def test_get_schema_extensions_returns_empty_when_schema_attr_missing():
    from ennoia.schema.base import get_schema_extensions

    # BaseSemantic doesn't declare an inner ``Schema`` class, so extensions
    # fall through to the default empty list.
    assert get_schema_extensions(Summary) == []


def test_get_schema_extensions_returns_empty_when_extensions_attr_missing():
    from ennoia.schema.base import get_schema_extensions

    class HasOnlyNamespace(BaseStructure):
        """Schema with namespace but no extensions attr."""

        v: str

        class Schema:
            namespace = "ns"

    assert get_schema_extensions(HasOnlyNamespace) == []


def test_describe_schema_emits_fields_and_name():
    payload = Meta.describe_schema()
    assert payload["name"] == "Meta"
    names = [f["name"] for f in payload["fields"]]
    assert "category" in names
    assert "title" in names
    category = next(f for f in payload["fields"] if f["name"] == "category")
    assert category["type"] == "enum"
    assert category["options"] == ["legal", "medical"]


def test_describe_schema_omits_non_filterable_fields():
    class WithHiddenField(BaseStructure):
        """Has a hidden field."""

        visible: str = Field(description="v")
        hidden: str = Field(default="", filterable=False)

    payload = WithHiddenField.describe_schema()
    assert [f["name"] for f in payload["fields"]] == ["visible"]
