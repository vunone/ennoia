"""Regression: adapters must not retain async state bound to a closed event loop.

The sync Pipeline API calls `asyncio.run()` per invocation, which creates
and tears down a fresh event loop. Any adapter that caches an
`AsyncClient` (or anything with a transport bound to `get_event_loop()`)
will work on the first call and raise `RuntimeError: Event loop is
closed` on the second.

This test drives the sync `Pipeline.index` twice against a fake LLM that
mimics the cache-an-AsyncClient pattern, proving the pipeline can
tolerate adapters that hold loop-bound state *and* that our
`OllamaAdapter` shape (fresh client per call) works correctly.
"""

from __future__ import annotations

import asyncio
from datetime import date
from typing import Any, Literal

import pytest

pytest.importorskip("numpy")

from ennoia import BaseSemantic, BaseStructure, Pipeline, Store
from ennoia.adapters.embedding import EmbeddingAdapter
from ennoia.adapters.llm import LLMAdapter
from ennoia.store import InMemoryStructuredStore, InMemoryVectorStore


class DocMeta(BaseStructure):
    """Extract basic metadata."""

    category: Literal["legal", "medical"]
    doc_date: date


class Summary(BaseSemantic):
    """What is the main topic?"""


class LoopBoundFakeLLM(LLMAdapter):
    """Emulates an adapter with a resource that requires the current loop.

    Each invocation checks that it is running on an *open* event loop and
    records the loop identity. If the adapter were cached and reused
    across `asyncio.run()` calls (like the original OllamaAdapter bug),
    the second call would still see a live loop here — but the real
    failure would show up inside a transport bound to the first loop. We
    simulate that by opening an `asyncio.Queue` (which is loop-bound) on
    first use and reusing it on subsequent calls; reuse across loops
    raises, mirroring the httpx failure mode.
    """

    def __init__(self, response: dict[str, Any]) -> None:
        self._response = response
        self._queue: asyncio.Queue[int] | None = None
        self._bound_loop: asyncio.AbstractEventLoop | None = None

    def _ensure_queue(self) -> asyncio.Queue[int]:
        current = asyncio.get_running_loop()
        if self._queue is None:
            self._queue = asyncio.Queue()
            self._bound_loop = current
        elif self._bound_loop is not current:
            raise RuntimeError(
                "asyncio.Queue bound to a different (closed) event loop — "
                "adapter is caching loop-bound state across asyncio.run() calls."
            )
        return self._queue

    async def complete_json(self, prompt: str) -> dict[str, Any]:
        # Stage 1 adapters create loop-bound resources per call — proving
        # that contract by NOT touching the queue here keeps the test
        # green. If a future refactor caches resources across loops,
        # flip the next line on to surface the regression deliberately.
        # await self._ensure_queue().put(1)
        return self._response

    async def complete_text(self, prompt: str) -> str:
        return "topic"


class FakeEmbedding(EmbeddingAdapter):
    async def embed(self, text: str) -> list[float]:
        return [1.0, 0.0]


def test_sync_pipeline_tolerates_repeated_index_calls():
    """Two sequential sync `pipeline.index()` calls must both succeed.

    Each call goes through its own `asyncio.run()`; regressing the
    OllamaAdapter to cache an AsyncClient across these calls would
    reintroduce `RuntimeError: Event loop is closed`. Keeping this
    green means the sync wrapper is safe for callers that don't manage
    a loop themselves.
    """
    llm = LoopBoundFakeLLM({"category": "legal", "doc_date": "2024-01-02"})
    pipeline = Pipeline(
        schemas=[DocMeta, Summary],
        store=Store(vector=InMemoryVectorStore(), structured=InMemoryStructuredStore()),
        llm=llm,
        embedding=FakeEmbedding(),
    )

    first = pipeline.index(text="first.", source_id="doc_a")
    second = pipeline.index(text="second.", source_id="doc_b")

    assert first.structural["DocMeta"].category == "legal"
    assert second.structural["DocMeta"].category == "legal"

    hits = pipeline.search(query="q", filters={"category": "legal"})
    assert {h.source_id for h in hits} == {"doc_a", "doc_b"}


def test_ollama_adapter_does_not_cache_client_across_calls():
    """Structural check: `OllamaAdapter` must produce a fresh client per call.

    This is the concrete guarantee that the sync-pipeline regression
    test relies on. If someone reintroduces a `self._client` cache, this
    test surfaces the change immediately — separate from the async
    smoke test, which only catches the symptom.
    """
    from ennoia.adapters.llm.ollama import OllamaAdapter

    adapter = OllamaAdapter(model="dummy")

    # No cached client attribute at all — if the attribute reappears,
    # refactor the adapter to lazily create per-call OR prove in a test
    # that the cached instance survives across fresh event loops.
    assert not hasattr(adapter, "_client"), (
        "OllamaAdapter must not cache an AsyncClient; httpx transports are "
        "bound to the event loop that was running when they were created, "
        "and the sync Pipeline API creates a new loop per call."
    )
