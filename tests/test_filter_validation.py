"""Filter validation — ``validate_filters`` + ``FilterValidationError``."""

from __future__ import annotations

from datetime import date
from typing import Annotated, Literal

import pytest

pytest.importorskip("numpy")

from ennoia import BaseStructure, Field, FilterValidationError, Pipeline, Store
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


def test_pipeline_search_raises_validation_error() -> None:
    class FakeLLM:
        async def complete_json(self, prompt: str) -> dict[str, object]:
            return {}

        async def complete_text(self, prompt: str) -> str:
            return ""

    class FakeEmbedding:
        def embed_document(self, text: str) -> list[float]:
            return [1.0]

        def embed_query(self, text: str) -> list[float]:
            return [1.0]

    pipeline = Pipeline(
        schemas=[Doc],
        store=Store(vector=InMemoryVectorStore(), structured=InMemoryStructuredStore()),
        llm=FakeLLM(),
        embedding=FakeEmbedding(),
    )
    with pytest.raises(FilterValidationError):
        pipeline.search(query="x", filters={"jurisdiction__gt": "WA"})
