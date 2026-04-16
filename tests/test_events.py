"""Event bus — subscribe / emit / error-isolation + pipeline dispatch."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("numpy")

from ennoia import BaseStructure, Emitter, IndexEvent, Pipeline, SearchEvent, Store
from ennoia.adapters.embedding import EmbeddingAdapter
from ennoia.adapters.llm import LLMAdapter
from ennoia.store import InMemoryStructuredStore, InMemoryVectorStore


class Doc(BaseStructure):
    """Extract doc."""

    value: str


class FakeLLM(LLMAdapter):
    async def complete_json(self, prompt: str) -> dict[str, Any]:
        return {"value": "x", "_confidence": 0.9}

    async def complete_text(self, prompt: str) -> str:
        return ""


class FakeEmbedding(EmbeddingAdapter):
    async def embed(self, text: str) -> list[float]:
        return [1.0, 0.0]


def test_subscribed_handler_receives_events() -> None:
    bus = Emitter()
    received: list[object] = []
    bus.subscribe(IndexEvent, received.append)
    bus.subscribe(SearchEvent, received.append)

    pipeline = Pipeline(
        schemas=[Doc],
        store=Store(vector=InMemoryVectorStore(), structured=InMemoryStructuredStore()),
        llm=FakeLLM(),
        embedding=FakeEmbedding(),
        events=bus,
    )
    pipeline.index(text="x", source_id="d1")
    pipeline.search(query="x", filters=None)

    assert any(isinstance(e, IndexEvent) for e in received)
    assert any(isinstance(e, SearchEvent) for e in received)


def test_handler_errors_are_swallowed(caplog: pytest.LogCaptureFixture) -> None:
    bus = Emitter()

    def boom(_event: object) -> None:
        raise RuntimeError("handler blew up")

    bus.subscribe(IndexEvent, boom)

    pipeline = Pipeline(
        schemas=[Doc],
        store=Store(vector=InMemoryVectorStore(), structured=InMemoryStructuredStore()),
        llm=FakeLLM(),
        embedding=FakeEmbedding(),
        events=bus,
    )
    # No raise → handler errors do not break the pipeline.
    with caplog.at_level("ERROR"):
        pipeline.index(text="x", source_id="d2")
    assert any("Event handler raised" in r.message for r in caplog.records)


def test_default_emitter_is_noop() -> None:
    pipeline = Pipeline(
        schemas=[Doc],
        store=Store(vector=InMemoryVectorStore(), structured=InMemoryStructuredStore()),
        llm=FakeLLM(),
        embedding=FakeEmbedding(),
    )
    # Default NullEmitter swallows every event — no exception either way.
    pipeline.index(text="x", source_id="d3")
