"""``Pipeline(concurrency=...)`` caps in-flight LLM and embedding calls.

The CLI ``--no-threads`` flag passes ``concurrency=1`` to serialise calls
for local-Ollama users on resource-constrained machines. These tests pin
that contract so a regression in the executor / pipeline gather wiring
surfaces immediately.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

pytest.importorskip("numpy")

from ennoia import BaseSemantic, BaseStructure, Pipeline, Store
from ennoia.adapters.embedding import EmbeddingAdapter
from ennoia.adapters.llm import LLMAdapter
from ennoia.store import InMemoryStructuredStore, InMemoryVectorStore


class _A(BaseStructure):
    """Schema A."""

    value: str


class _B(BaseStructure):
    """Schema B."""

    value: str


class _SummaryA(BaseSemantic):
    """Summarise A."""


class _SummaryB(BaseSemantic):
    """Summarise B."""


class _BatchTrackingEmbedding(EmbeddingAdapter):
    """Embedding adapter that records whether ``embed_batch`` ran under a semaphore.

    The pipeline acquires the semaphore around the ``embed_batch`` call so
    embedding work is rate-limited together with LLM extractions.
    """

    def __init__(self) -> None:
        self.batch_calls = 0

    async def embed(self, text: str) -> list[float]:
        return [1.0, 0.0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.batch_calls += 1
        return [[1.0, 0.0] for _ in texts]


class _TimingLLM(LLMAdapter):
    """Records concurrency — how many LLM calls overlap in time."""

    def __init__(self) -> None:
        self.in_flight = 0
        self.peak_in_flight = 0
        self.lock = asyncio.Lock()

    async def _track(self) -> None:
        async with self.lock:
            self.in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        await asyncio.sleep(0.02)
        async with self.lock:
            self.in_flight -= 1

    async def complete_json(self, prompt: str) -> dict[str, Any]:
        await self._track()
        return {"value": "x", "extraction_confidence": 0.9}

    async def complete_text(self, prompt: str) -> str:
        await self._track()
        return "answer"


class _Embedding(EmbeddingAdapter):
    async def embed(self, text: str) -> list[float]:
        return [1.0, 0.0]


def _store() -> Store:
    return Store(vector=InMemoryVectorStore(), structured=InMemoryStructuredStore())


def test_concurrency_one_serialises_all_llm_calls() -> None:
    """``concurrency=1`` (i.e. ``--no-threads``) caps the executor's gather to 1.

    With two independent structural schemas, the unbounded executor
    overlaps both LLM calls (peak=2). ``concurrency=1`` must serialise
    them (peak=1).
    """
    llm = _TimingLLM()
    pipeline = Pipeline(
        schemas=[_A, _B],
        store=_store(),
        llm=llm,
        embedding=_Embedding(),
        concurrency=1,
    )
    result = pipeline.index(text="body", source_id="d1")
    assert "_A" in result.structural and "_B" in result.structural
    assert llm.peak_in_flight == 1


def test_concurrency_none_leaves_executor_unbounded() -> None:
    """The default (no concurrency arg) keeps the existing parallel behaviour."""
    llm = _TimingLLM()
    pipeline = Pipeline(
        schemas=[_A, _B],
        store=_store(),
        llm=llm,
        embedding=_Embedding(),
    )
    pipeline.index(text="body", source_id="d2")
    assert llm.peak_in_flight == 2


def test_concurrency_one_routes_embed_batch_through_semaphore() -> None:
    """``concurrency=1`` makes ``_apersist`` acquire the semaphore around ``embed_batch``.

    Covers the ``async with semaphore`` branch of ``_embed_documents``; without
    a semaphore, that branch is skipped entirely.
    """
    embedding = _BatchTrackingEmbedding()
    pipeline = Pipeline(
        schemas=[_A, _SummaryA, _SummaryB],
        store=_store(),
        llm=_TimingLLM(),
        embedding=embedding,
        concurrency=1,
    )
    pipeline.index(text="body", source_id="bd1")
    assert embedding.batch_calls == 1


def test_concurrency_zero_or_negative_rejected() -> None:
    """``concurrency`` is a positive cap (>=1); zero/negative are bugs."""
    with pytest.raises(ValueError, match="concurrency must be"):
        Pipeline(
            schemas=[_A],
            store=_store(),
            llm=_TimingLLM(),
            embedding=_Embedding(),
            concurrency=0,
        )
