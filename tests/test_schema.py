"""Schema declaration layer."""

from __future__ import annotations

from typing import ClassVar, Literal

import pytest

from ennoia import BaseCollection, BaseSemantic, BaseStructure, Field


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


# ---------------------------------------------------------------------------
# BaseCollection
# ---------------------------------------------------------------------------


class Party(BaseCollection):
    """Extract every contract party mentioned in the document."""

    company_name: str
    year: int


def test_collection_extract_prompt_is_docstring():
    assert Party.extract_prompt() == "Extract every contract party mentioned in the document."


def test_collection_kind_marker():
    assert Party.__ennoia_kind__ == "collection"


def test_collection_json_schema_delegates_to_pydantic():
    schema = Party.json_schema()
    assert schema == Party.model_json_schema()
    assert set(schema["required"]) == {"company_name", "year"}


def test_collection_template_defaults_to_model_dump():
    instance = Party(company_name="Acme", year=2024)
    assert instance.template() == str({"company_name": "Acme", "year": 2024})


def test_collection_get_unique_default_is_random_per_call():
    a = Party(company_name="Acme", year=2024).get_unique()
    b = Party(company_name="Acme", year=2024).get_unique()
    # Random token default: two identical extractions are NOT deduped by default.
    assert a != b
    assert isinstance(a, str) and a


def test_collection_is_valid_default_is_noop():
    # Default returns None without raising — the loop keeps every validated entity.
    assert Party(company_name="Acme", year=2024).is_valid() is None


def test_collection_extend_default_is_empty():
    assert Party(company_name="Acme", year=2024).extend() == []


def test_collection_schema_defaults():
    # No inner Schema → helpers return defaults.
    from ennoia.schema.base import (
        get_schema_extensions,
        get_schema_max_iterations,
        get_schema_namespace,
    )

    assert get_schema_extensions(Party) == []
    assert get_schema_max_iterations(Party) is None
    assert get_schema_namespace(Party) is None


def test_collection_max_iterations_read_from_inner_schema():
    class Capped(BaseCollection):
        """Capped extraction."""

        name: str

        class Schema:
            max_iterations: ClassVar[int | None] = 3

    from ennoia.schema.base import get_schema_max_iterations

    assert get_schema_max_iterations(Capped) == 3


def test_collection_max_iterations_rejects_non_int():
    class Bad(BaseCollection):
        """Bad max_iterations type."""

        name: str

        class Schema:
            # Deliberately wrong type to trip the runtime check.
            max_iterations: ClassVar[object] = "ten"  # type: ignore[assignment]

    from ennoia.schema.base import get_schema_max_iterations

    with pytest.raises(TypeError, match="max_iterations"):
        get_schema_max_iterations(Bad)


def test_get_schema_max_iterations_returns_none_for_class_without_inner_schema():
    # BaseSemantic (and other non-collection classes) have no inner Schema →
    # helper returns None rather than raising.
    from ennoia.schema.base import get_schema_max_iterations

    assert get_schema_max_iterations(Summary) is None


def test_collection_max_iterations_rejects_zero_or_negative():
    class Zero(BaseCollection):
        """Non-positive max_iterations."""

        name: str

        class Schema:
            max_iterations: ClassVar[int | None] = 0

    from ennoia.schema.base import get_schema_max_iterations

    with pytest.raises(ValueError, match=">= 1"):
        get_schema_max_iterations(Zero)
