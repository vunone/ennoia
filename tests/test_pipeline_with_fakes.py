"""End-to-end pipeline test using hand-rolled LLM / embedding fakes."""

from __future__ import annotations

import asyncio
from datetime import date
from typing import Any, Literal

import pytest

pytest.importorskip("numpy")

from ennoia import BaseSemantic, BaseStructure, Pipeline, RejectException, Store
from ennoia.adapters.embedding import EmbeddingAdapter
from ennoia.adapters.llm import LLMAdapter
from ennoia.store import InMemoryStructuredStore, InMemoryVectorStore


class DocMeta(BaseStructure):
    """Extract basic metadata."""

    category: Literal["legal", "medical"]
    doc_date: date


class Summary(BaseSemantic):
    """What is the main topic?"""


class FakeLLM(LLMAdapter):
    def __init__(
        self,
        json_responses: dict[str, dict[str, Any]],
        text_default: str = "A topic.",
        reject_if: str | None = None,
    ) -> None:
        self._json_responses = json_responses
        self._text_default = text_default
        self._reject_if = reject_if

    async def complete_json(self, prompt: str) -> dict[str, Any]:
        if self._reject_if is not None and self._reject_if in prompt:
            raise RejectException("Out of scope.")
        for key, value in self._json_responses.items():
            if key in prompt:
                return value
        raise AssertionError("FakeLLM got an unexpected prompt.")

    async def complete_text(self, prompt: str) -> str:
        return self._text_default


class FakeEmbedding(EmbeddingAdapter):
    """Deterministic embedding — rotates a simple 4-dim vector per call content."""

    async def embed(self, text: str) -> list[float]:
        seed = abs(hash(text)) % 1000 / 1000.0
        return [1.0, seed, 0.0, 0.0]


def _store() -> Store:
    return Store(vector=InMemoryVectorStore(), structured=InMemoryStructuredStore())


def test_index_then_search_roundtrips():
    llm = FakeLLM(
        json_responses={
            "Extract basic metadata.": {"category": "legal", "doc_date": "2024-01-02"},
        }
    )
    pipeline = Pipeline(
        schemas=[DocMeta, Summary],
        store=_store(),
        llm=llm,
        embedding=FakeEmbedding(),
    )

    result = pipeline.index(text="Some legal text.", source_id="doc_001")
    assert not result.rejected
    assert result.structural["DocMeta"].category == "legal"
    assert result.semantic["Summary"] == "A topic."

    hits = pipeline.search(query="legal topic", filters={"category": "legal"})
    assert len(hits) == 1
    assert hits.hits[0].source_id == "doc_001"
    assert hits.hits[0].structural["category"] == "legal"


def test_filter_miss_returns_empty():
    llm = FakeLLM(
        json_responses={
            "Extract basic metadata.": {"category": "legal", "doc_date": "2024-01-02"},
        }
    )
    pipeline = Pipeline(
        schemas=[DocMeta, Summary],
        store=_store(),
        llm=llm,
        embedding=FakeEmbedding(),
    )

    pipeline.index(text="Some legal text.", source_id="doc_001")
    hits = pipeline.search(query="legal", filters={"category": "medical"})
    assert len(hits) == 0


def test_reject_exception_skips_store_writes():
    llm = FakeLLM(json_responses={}, reject_if="Extract basic metadata.")
    store = _store()
    pipeline = Pipeline(
        schemas=[DocMeta, Summary],
        store=store,
        llm=llm,
        embedding=FakeEmbedding(),
    )

    result = pipeline.index(text="Some medical text.", source_id="doc_reject")
    assert result.rejected is True
    assert asyncio.run(store.structured.get("doc_reject")) is None
    assert asyncio.run(store.vector.search([1.0, 0.0, 0.0, 0.0], top_k=5)) == []


class ChildDetail(BaseStructure):
    """Extract child detail."""

    detail: str


class ParentMeta(BaseStructure):
    """Extract parent metadata."""

    needs_child: bool

    class Schema:
        extensions = [ChildDetail]

    def extend(self) -> list[type[BaseStructure] | type[BaseSemantic]]:
        if self.needs_child:
            return [ChildDetail]
        return []


def test_extend_queues_child_schema_at_runtime():
    llm = FakeLLM(
        json_responses={
            "Extract parent metadata.": {"needs_child": True},
            "Extract child detail.": {"detail": "deep"},
        }
    )
    pipeline = Pipeline(
        schemas=[ParentMeta],
        store=_store(),
        llm=llm,
        embedding=FakeEmbedding(),
    )

    result = pipeline.index(text="Anything.", source_id="doc_nested")
    assert "ParentMeta" in result.structural
    assert "ChildDetail" in result.structural
    assert result.structural["ChildDetail"].detail == "deep"


def test_extend_returning_empty_skips_child():
    llm = FakeLLM(
        json_responses={
            "Extract parent metadata.": {"needs_child": False},
        }
    )
    pipeline = Pipeline(
        schemas=[ParentMeta],
        store=_store(),
        llm=llm,
        embedding=FakeEmbedding(),
    )

    result = pipeline.index(text="Anything.", source_id="doc_no_nest")
    assert "ParentMeta" in result.structural
    assert "ChildDetail" not in result.structural
