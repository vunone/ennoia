"""Layer-wise parallel extraction — assert independent schemas run concurrently."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

pytest.importorskip("numpy")

from ennoia import BaseStructure, Pipeline, Store
from ennoia.adapters.embedding import EmbeddingAdapter
from ennoia.adapters.llm import LLMAdapter
from ennoia.store import InMemoryStructuredStore, InMemoryVectorStore


class A(BaseStructure):
    """Schema A."""

    value: str


class B(BaseStructure):
    """Schema B."""

    value: str


class TimingLLM(LLMAdapter):
    """LLM fake that records concurrency — how many calls overlap in time."""

    def __init__(self) -> None:
        self.in_flight = 0
        self.peak_in_flight = 0
        self.lock = asyncio.Lock()

    async def complete_json(self, prompt: str) -> dict[str, Any]:
        async with self.lock:
            self.in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        # Yield so the other task can start before we decrement.
        await asyncio.sleep(0.02)
        async with self.lock:
            self.in_flight -= 1
        return {"value": "x", "extraction_confidence": 0.9}

    async def complete_text(self, prompt: str) -> str:
        return "topic"


class FakeEmbedding(EmbeddingAdapter):
    async def embed(self, text: str) -> list[float]:
        return [1.0]


def test_independent_structural_schemas_run_in_parallel() -> None:
    llm = TimingLLM()
    pipeline = Pipeline(
        schemas=[A, B],
        store=Store(vector=InMemoryVectorStore(), structured=InMemoryStructuredStore()),
        llm=llm,
        embedding=FakeEmbedding(),
    )
    result = pipeline.index(text="body", source_id="doc_1")
    assert "A" in result.structural and "B" in result.structural
    # Two independent schemas gathered in one layer — must overlap at some point.
    assert llm.peak_in_flight == 2
