"""Common types shared by both pipeline wrappers.

Both ennoia and langchain wrappers reduce to the same shape:
``index_corpus(contracts) -> None`` and ``answer(question) -> PipelineRun``.
Each pipeline runs end-to-end (retrieval + answer generation) internally,
so the runner can judge + measure recall without knowing whether the
pipeline used a shared template or an agent loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from benchmark.data.loader import Contract, Question


@dataclass(slots=True)
class PipelineRun:
    retrieved_source_ids: list[str]
    answer: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    trace: list[dict[str, Any]] = field(default_factory=list)


class Pipeline(Protocol):
    name: str

    async def index_corpus(self, contracts: list[Contract]) -> None: ...
    async def answer(self, question: Question) -> PipelineRun: ...
