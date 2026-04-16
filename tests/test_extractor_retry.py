"""Extractor retry / fallback paths and confidence-parsing edge cases."""

from __future__ import annotations

from typing import Any

import pytest

from ennoia import BaseSemantic, BaseStructure
from ennoia.adapters.llm.base import LLMAdapter
from ennoia.index.exceptions import ExtractionError
from ennoia.index.extractor import (
    CONFIDENCE_KEY,
    augment_json_schema_with_confidence,
    extract_semantic,
    extract_structural,
)


class _Meta(BaseStructure):
    """Extract metadata."""

    value: int


class _Summary(BaseSemantic):
    """Summarise."""


class _ScriptedLLM(LLMAdapter):
    """Serves scripted JSON / text responses in call order."""

    def __init__(
        self,
        json_responses: list[Any] | None = None,
        text_responses: list[str] | None = None,
    ) -> None:
        self._json = list(json_responses or [])
        self._text = list(text_responses or [])
        self.json_calls: list[str] = []
        self.text_calls: list[str] = []

    async def complete_json(self, prompt: str) -> dict[str, Any]:
        self.json_calls.append(prompt)
        payload = self._json.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return dict(payload)

    async def complete_text(self, prompt: str) -> str:
        self.text_calls.append(prompt)
        return self._text.pop(0)


# ---------------------------------------------------------------------------
# augment_json_schema_with_confidence — required-list idempotency
# ---------------------------------------------------------------------------


def test_augment_idempotent_on_required_when_already_present() -> None:
    # If the incoming schema already lists ``_confidence`` as required, the
    # helper must not duplicate the entry.
    schema = {
        "properties": {"value": {"type": "integer"}},
        "required": ["value", CONFIDENCE_KEY],
    }
    augmented = augment_json_schema_with_confidence(schema)
    assert augmented["required"].count(CONFIDENCE_KEY) == 1


# ---------------------------------------------------------------------------
# extract_structural — retry on first ValidationError
# ---------------------------------------------------------------------------


async def test_structural_retries_once_on_validation_error() -> None:
    llm = _ScriptedLLM(
        json_responses=[
            {"value": "not-an-int", "_confidence": 0.5},  # fails Pydantic validation
            {"value": 7, "_confidence": 0.9},  # retry succeeds
        ]
    )
    instance, confidence = await extract_structural(
        schema=_Meta, text="body", context_additions=[], llm=llm
    )
    assert instance.value == 7
    assert confidence == 0.9
    # Second prompt carries the validation-error text so the LLM can self-correct.
    assert len(llm.json_calls) == 2
    assert "failed validation" in llm.json_calls[1]


async def test_structural_raises_when_retry_also_fails() -> None:
    llm = _ScriptedLLM(
        json_responses=[
            {"value": "bad", "_confidence": 0.5},
            {"value": "still-bad", "_confidence": 0.5},
        ]
    )
    with pytest.raises(ExtractionError, match="after retry"):
        await extract_structural(schema=_Meta, text="body", context_additions=[], llm=llm)


# ---------------------------------------------------------------------------
# _split_confidence (via extract_structural) — non-numeric confidence
# ---------------------------------------------------------------------------


async def test_structural_non_numeric_confidence_defaults_to_one() -> None:
    llm = _ScriptedLLM(json_responses=[{"value": 1, "_confidence": "high"}])
    _, confidence = await extract_structural(
        schema=_Meta, text="body", context_additions=[], llm=llm
    )
    assert confidence == 1.0


async def test_structural_clamps_confidence_to_unit_interval() -> None:
    llm = _ScriptedLLM(json_responses=[{"value": 1, "_confidence": 5.0}])
    _, confidence = await extract_structural(
        schema=_Meta, text="body", context_additions=[], llm=llm
    )
    assert confidence == 1.0


# ---------------------------------------------------------------------------
# extract_semantic — confidence tag present / missing
# ---------------------------------------------------------------------------


async def test_semantic_strips_trailing_confidence_tag() -> None:
    llm = _ScriptedLLM(text_responses=["The answer is 42. <confidence>0.8</confidence>"])
    answer, confidence = await extract_semantic(schema=_Summary, text="body", llm=llm)
    assert answer == "The answer is 42."
    assert confidence == 0.8


async def test_semantic_defaults_to_one_when_tag_missing() -> None:
    llm = _ScriptedLLM(text_responses=["No confidence tag here."])
    answer, confidence = await extract_semantic(schema=_Summary, text="body", llm=llm)
    assert answer == "No confidence tag here."
    assert confidence == 1.0


async def test_semantic_clamps_confidence_to_unit_interval() -> None:
    llm = _ScriptedLLM(text_responses=["answer <confidence>5.5</confidence>"])
    _, confidence = await extract_semantic(schema=_Summary, text="body", llm=llm)
    assert confidence == 1.0


async def test_semantic_parses_confidence_without_decimal_point() -> None:
    llm = _ScriptedLLM(text_responses=["answer <confidence>1</confidence>"])
    _, confidence = await extract_semantic(schema=_Summary, text="body", llm=llm)
    assert confidence == 1.0
