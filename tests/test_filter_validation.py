"""Filter validation — ``validate_filters`` + ``FilterValidationError``."""

from __future__ import annotations

from datetime import date
from typing import Annotated, Literal

import pytest

pytest.importorskip("numpy")

from ennoia import BaseStructure, Field, FilterValidationError, Pipeline, Store
from ennoia.adapters.embedding import EmbeddingAdapter
from ennoia.adapters.llm import LLMAdapter
from ennoia.index.validation import validate_filters
from ennoia.schema.manifest import build_manifest
from ennoia.schema.merging import build_superschema
from ennoia.store import InMemoryStructuredStore, InMemoryVectorStore


class Doc(BaseStructure):
    """Extract doc metadata."""

    jurisdiction: Literal["WA", "NY"]
    date_decided: date
    tags: list[str]
    title: Annotated[str, Field(operators=["eq"])] = ""


def _superschema_for(*cls_list: type[BaseStructure]):
    return build_superschema(build_manifest(list(cls_list)))


def test_superschema_exposes_every_filterable_field() -> None:
    ss = _superschema_for(Doc)
    assert set(ss.fields) == {"jurisdiction", "date_decided", "tags", "title"}


def test_unknown_field_raises_with_right_shape() -> None:
    ss = _superschema_for(Doc)
    with pytest.raises(FilterValidationError) as exc:
        validate_filters({"unknown": "x"}, ss)
    assert exc.value.to_dict()["field"] == "unknown"
    assert exc.value.to_dict()["error"] == "invalid_filter"


def test_unsupported_operator_raises_with_supported_list() -> None:
    ss = _superschema_for(Doc)
    with pytest.raises(FilterValidationError) as exc:
        validate_filters({"jurisdiction__gt": "WA"}, ss)
    err = exc.value
    assert err.field == "jurisdiction"
    assert err.operator == "gt"
    assert "eq" in err.supported and "in" in err.supported


def test_field_override_tightens_operators() -> None:
    ss = _superschema_for(Doc)
    with pytest.raises(FilterValidationError):
        validate_filters({"title__contains": "x"}, ss)


def test_empty_filters_is_noop() -> None:
    ss = _superschema_for(Doc)
    validate_filters(None, ss)
    validate_filters({}, ss)


def test_nested_dict_value_rejected_with_flat_form_hint() -> None:
    ss = _superschema_for(Doc)
    with pytest.raises(FilterValidationError) as exc:
        validate_filters({"date_decided": {"gt": "2020-01-01"}}, ss)
    payload = exc.value.to_dict()
    assert payload["field"] == "date_decided"
    assert payload["operator"] == "eq"
    assert "flat" in payload["message"]
    assert "date_decided__eq" in payload["message"]


def test_nested_dict_value_rejected_even_when_operator_suffix_present() -> None:
    ss = _superschema_for(Doc)
    with pytest.raises(FilterValidationError) as exc:
        validate_filters({"date_decided__gt": {"any": "2020-01-01"}}, ss)
    assert exc.value.operator == "gt"
    assert "flat" in exc.value.message


def test_list_value_is_allowed_for_in_operator() -> None:
    ss = _superschema_for(Doc)
    validate_filters({"jurisdiction__in": ["WA", "NY"]}, ss)


def test_pipeline_search_raises_validation_error() -> None:
    class FakeLLM(LLMAdapter):
        async def complete_json(self, prompt: str) -> dict[str, object]:
            return {}

        async def complete_text(self, prompt: str) -> str:
            return ""

    class FakeEmbedding(EmbeddingAdapter):
        async def embed(self, text: str) -> list[float]:
            return [1.0]

    pipeline = Pipeline(
        schemas=[Doc],
        store=Store(vector=InMemoryVectorStore(), structured=InMemoryStructuredStore()),
        llm=FakeLLM(),
        embedding=FakeEmbedding(),
    )
    with pytest.raises(FilterValidationError):
        pipeline.search(query="x", filters={"jurisdiction__gt": "WA"})
