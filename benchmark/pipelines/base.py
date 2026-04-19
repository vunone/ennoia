"""Common types shared by both pipeline wrappers.

Both ennoia and langchain wrappers reduce to the same shape:
``index_corpus(products) -> None`` and ``answer(query) -> PipelineRun``.
Each pipeline runs end-to-end (retrieval + answer generation) internally,
so the runner can judge + measure precision without knowing whether the
pipeline used a shared template or an agent loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from benchmark.data.prep import Product


@dataclass(slots=True)
class PipelineRun:
    retrieved_source_ids: list[str]
    answer: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    trace: list[dict[str, Any]] = field(default_factory=list)


class Pipeline(Protocol):
    name: str

    async def index_corpus(self, products: list[Product]) -> None: ...
    async def answer(self, query: str) -> PipelineRun: ...
