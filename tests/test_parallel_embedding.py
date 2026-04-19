"""Embedding I/O is async — multiple semantic fields per doc embed in parallel."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

pytest.importorskip("numpy")

from ennoia import BaseSemantic, BaseStructure, Pipeline, Store
from ennoia.adapters.embedding import EmbeddingAdapter
from ennoia.adapters.llm import LLMAdapter
from ennoia.store import InMemoryStructuredStore, InMemoryVectorStore


class _Doc(BaseStructure):
    """Extract doc."""

    cat: str


class _SummaryA(BaseSemantic):
    """Summarise angle A."""


class _SummaryB(BaseSemantic):
    """Summarise angle B."""


class _LLM(LLMAdapter):
    async def complete_json(self, prompt: str) -> dict[str, Any]:
        return {"cat": "legal", "extraction_confidence": 0.9}

    async def complete_text(self, prompt: str) -> str:
        # Distinct text per semantic so two embeds happen.
        if "angle A" in prompt:
            return "answer A"
        return "answer B"


class _TimingEmbedding(EmbeddingAdapter):
    """Records concurrency — how many embed calls overlap in time."""

    def __init__(self) -> None:
        self.in_flight = 0
        self.peak_in_flight = 0
        self.lock = asyncio.Lock()
        # Track whether ``embed_batch`` was used (it should be — single batch
        # call for the whole document) versus N parallel ``embed`` calls.
        self.batch_calls = 0
        self.embed_calls = 0

    async def embed(self, text: str) -> list[float]:
        async with self.lock:
            self.in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
            self.embed_calls += 1
        await asyncio.sleep(0.02)
        async with self.lock:
            self.in_flight -= 1
        return [1.0, 0.0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        async with self.lock:
            self.in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
            self.batch_calls += 1
        await asyncio.sleep(0.02)
        async with self.lock:
            self.in_flight -= 1
        return [[1.0, 0.0] for _ in texts]


def test_persist_uses_embed_batch_for_multiple_semantic_fields() -> None:
    """Two semantic schemas → one ``embed_batch`` call, not two ``embed`` calls.

    The pipeline forwards the full batch into ``embed_batch`` so backends
    with a native list-input API (OpenAI) round-trip once instead of N times.
    """
    embedding = _TimingEmbedding()
    pipeline = Pipeline(
        schemas=[_Doc, _SummaryA, _SummaryB],
        store=Store(vector=InMemoryVectorStore(), structured=InMemoryStructuredStore()),
        llm=_LLM(),
        embedding=embedding,
    )
    pipeline.index(text="body", source_id="d1")
    # Both semantic answers funnelled through one ``embed_batch`` call.
    assert embedding.batch_calls == 1
    assert embedding.embed_calls == 0
