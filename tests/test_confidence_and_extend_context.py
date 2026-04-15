"""Confidence plumbing + extend() parent-context injection."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("numpy")

from ennoia import BaseSemantic, BaseStructure, Pipeline, Store
from ennoia.store import InMemoryStructuredStore, InMemoryVectorStore


class Parent(BaseStructure):
    """Extract parent fields."""

    label: str

    def extend(self) -> list[type[BaseStructure] | type[BaseSemantic]]:
        # Read _confidence set by the extractor; branch when above threshold.
        if getattr(self, "_confidence", 0.0) >= 0.8:
            return [Child]
        return []


class Child(BaseStructure):
    """Extract child based on parent."""

    detail: str


class RecordingLLM:
    def __init__(self, responses: dict[str, dict[str, Any]]) -> None:
        self._responses = responses
        self.prompts: list[str] = []

    async def complete_json(self, prompt: str) -> dict[str, Any]:
        self.prompts.append(prompt)
        for key, value in self._responses.items():
            if key in prompt:
                return value
        raise AssertionError(f"No response for prompt starting: {prompt[:80]}")

    async def complete_text(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "topic"


class FakeEmbedding:
    def embed_document(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


def _pipeline(llm: RecordingLLM) -> Pipeline:
    return Pipeline(
        schemas=[Parent],
        store=Store(vector=InMemoryVectorStore(), structured=InMemoryStructuredStore()),
        llm=llm,
        embedding=FakeEmbedding(),
    )


def test_high_confidence_branches_into_child() -> None:
    llm = RecordingLLM(
        {
            "Extract parent fields.": {"label": "parent-value", "_confidence": 0.95},
            "Extract child based on parent.": {"detail": "child-value", "_confidence": 0.9},
        }
    )
    pipeline = _pipeline(llm)
    result = pipeline.index(text="body", source_id="d1")
    assert result.confidences == {"Parent": 0.95, "Child": 0.9}
    assert "Child" in result.structural

    child_prompt = next(p for p in llm.prompts if "Extract child based on parent." in p)
    assert "Parent extraction from Parent" in child_prompt
    assert "parent-value" in child_prompt


def test_low_confidence_skips_child() -> None:
    llm = RecordingLLM({"Extract parent fields.": {"label": "parent-value", "_confidence": 0.4}})
    pipeline = _pipeline(llm)
    result = pipeline.index(text="body", source_id="d2")
    assert "Child" not in result.structural
    assert result.confidences == {"Parent": 0.4}


def test_confidence_stripped_from_store_payload() -> None:
    llm = RecordingLLM({"Extract parent fields.": {"label": "x", "_confidence": 0.5}})
    pipeline = _pipeline(llm)
    pipeline.index(text="body", source_id="d3")
    stored = pipeline.store.structured.get("d3")
    assert stored == {"label": "x"}


def test_missing_confidence_defaults_to_one() -> None:
    llm = RecordingLLM({"Extract parent fields.": {"label": "x"}})
    pipeline = _pipeline(llm)
    result = pipeline.index(text="body", source_id="d4")
    assert result.confidences["Parent"] == 1.0
