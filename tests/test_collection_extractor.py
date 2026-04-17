"""Unit tests for the BaseCollection iteration loop (``extract_collection``)."""

from __future__ import annotations

import json
from typing import Any, ClassVar

import pytest

from ennoia import BaseCollection, RejectException, SkipItem
from ennoia.adapters.llm.base import LLMAdapter
from ennoia.index.extractor import (
    build_collection_prompt,
    build_collection_schema,
    extract_collection,
)


class _ScriptedLLM(LLMAdapter):
    """Feed ``extract_collection`` a list of JSON responses in order."""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = list(responses)
        self.json_calls: list[str] = []

    async def complete_json(self, prompt: str) -> dict[str, Any]:
        self.json_calls.append(prompt)
        if not self._responses:
            raise AssertionError("scripted LLM ran out of responses")
        return self._responses.pop(0)

    async def complete_text(self, prompt: str) -> str:
        raise AssertionError("complete_text should not be used by the collection path")


class Party(BaseCollection):
    """Extract every party mentioned."""

    name: str
    year: int

    def get_unique(self) -> str:
        # Deterministic dedup key for tests (the production default is random).
        return f"{self.name}|{self.year}"


class CappedParty(BaseCollection):
    """Capped extraction."""

    name: str

    def get_unique(self) -> str:
        return self.name

    class Schema:
        max_iterations: ClassVar[int | None] = 2


# ---------------------------------------------------------------------------
# Wrapper schema shape
# ---------------------------------------------------------------------------


def test_build_collection_schema_wraps_items_with_entities_list_and_is_done():
    schema = build_collection_schema(Party)
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"entities_list", "is_done"}
    items = schema["properties"]["entities_list"]["items"]
    # _confidence is appended per-item, not at top level.
    assert "_confidence" in items["properties"]
    assert "_confidence" in items["required"]
    # Top-level does NOT have _confidence.
    assert "_confidence" not in schema["properties"]


def test_build_collection_prompt_includes_previously_extracted_block():
    prior = [Party(name="Acme", year=2024), Party(name="Beta", year=2023)]
    prompt = build_collection_prompt(Party, "doc body", None, prior)
    # The opening tag appears as a section header (freshly on a new line).
    assert "\n<PreviouslyExtracted>\n" in prompt
    assert "Acme" in prompt
    assert "Beta" in prompt
    # Empty prior list omits the block entirely (the role text mentions the
    # tag in prose, but no section header with the tag on its own line exists).
    empty_prompt = build_collection_prompt(Party, "doc body", None, [])
    assert "\n<PreviouslyExtracted>\n" not in empty_prompt


def test_build_collection_prompt_additional_context_appears():
    prompt = build_collection_prompt(Party, "doc body", ["Parent says X"], [])
    assert "# Additional Context" in prompt
    assert "Parent says X" in prompt


# ---------------------------------------------------------------------------
# Iteration loop termination conditions
# ---------------------------------------------------------------------------


async def test_is_done_true_terminates_loop():
    llm = _ScriptedLLM(
        [
            {
                "entities_list": [
                    {"name": "Acme", "year": 2024, "_confidence": 0.9},
                ],
                "is_done": True,
            }
        ]
    )
    collected, confs = await extract_collection(Party, "body", [], llm)
    assert [p.name for p in collected] == ["Acme"]
    assert confs == [0.9]
    assert len(llm.json_calls) == 1


async def test_empty_entities_list_terminates_loop():
    llm = _ScriptedLLM([{"entities_list": [], "is_done": False}])
    collected, confs = await extract_collection(Party, "body", [], llm)
    assert collected == []
    assert confs == []
    assert len(llm.json_calls) == 1


async def test_no_new_unique_items_terminates_loop():
    # First call returns Acme; second call re-emits Acme (dup). Loop stops.
    llm = _ScriptedLLM(
        [
            {
                "entities_list": [{"name": "Acme", "year": 2024, "_confidence": 0.9}],
                "is_done": False,
            },
            {
                "entities_list": [{"name": "Acme", "year": 2024, "_confidence": 0.9}],
                "is_done": False,
            },
        ]
    )
    collected, _ = await extract_collection(Party, "body", [], llm)
    assert [p.name for p in collected] == ["Acme"]
    assert len(llm.json_calls) == 2


async def test_max_iterations_caps_loop():
    llm = _ScriptedLLM(
        [
            {
                "entities_list": [{"name": "Acme", "_confidence": 0.9}],
                "is_done": False,
            },
            {
                "entities_list": [{"name": "Beta", "_confidence": 0.9}],
                "is_done": False,
            },
            # Third iteration should NOT happen because max_iterations=2.
            {
                "entities_list": [{"name": "Gamma", "_confidence": 0.9}],
                "is_done": False,
            },
        ]
    )
    collected, _ = await extract_collection(CappedParty, "body", [], llm)
    assert [p.name for p in collected] == ["Acme", "Beta"]
    assert len(llm.json_calls) == 2


async def test_iteration_accumulates_across_calls():
    llm = _ScriptedLLM(
        [
            {
                "entities_list": [{"name": "Acme", "year": 2024, "_confidence": 0.9}],
                "is_done": False,
            },
            {
                "entities_list": [{"name": "Beta", "year": 2023, "_confidence": 0.8}],
                "is_done": True,
            },
        ]
    )
    collected, confs = await extract_collection(Party, "body", [], llm)
    assert [p.name for p in collected] == ["Acme", "Beta"]
    assert confs == [0.9, 0.8]
    # Second prompt MUST contain the first extraction via <PreviouslyExtracted>.
    assert "Acme" in llm.json_calls[1]


# ---------------------------------------------------------------------------
# Per-item failures
# ---------------------------------------------------------------------------


async def test_malformed_item_is_silently_dropped():
    llm = _ScriptedLLM(
        [
            {
                "entities_list": [
                    {"name": "Acme", "year": 2024, "_confidence": 0.9},
                    {"name": "Bad", "year": "not-an-int", "_confidence": 0.5},
                ],
                "is_done": True,
            }
        ]
    )
    collected, _ = await extract_collection(Party, "body", [], llm)
    assert [p.name for p in collected] == ["Acme"]


async def test_skipitem_drops_just_that_entity():
    class Discriminating(BaseCollection):
        """Reject blank names."""

        name: str

        def get_unique(self) -> str:
            return self.name

        def is_valid(self) -> None:
            if not self.name.strip():
                raise SkipItem("blank")

    llm = _ScriptedLLM(
        [
            {
                "entities_list": [
                    {"name": "Acme", "_confidence": 0.9},
                    {"name": "   ", "_confidence": 0.3},
                    {"name": "Beta", "_confidence": 0.8},
                ],
                "is_done": True,
            }
        ]
    )
    collected, _ = await extract_collection(Discriminating, "body", [], llm)
    assert [d.name for d in collected] == ["Acme", "Beta"]


async def test_rejectexception_from_is_valid_propagates():
    class Strict(BaseCollection):
        """Strict collection."""

        name: str

        def get_unique(self) -> str:
            return self.name

        def is_valid(self) -> None:
            raise RejectException("always reject")

    llm = _ScriptedLLM(
        [
            {
                "entities_list": [{"name": "Acme", "_confidence": 0.9}],
                "is_done": True,
            }
        ]
    )
    with pytest.raises(RejectException):
        await extract_collection(Strict, "body", [], llm)


# ---------------------------------------------------------------------------
# Context pass-through
# ---------------------------------------------------------------------------


async def test_context_additions_flow_into_prompt():
    llm = _ScriptedLLM([{"entities_list": [], "is_done": False}])
    await extract_collection(Party, "body", ["parent context"], llm)
    assert "parent context" in llm.json_calls[0]


# ---------------------------------------------------------------------------
# Confidence handling
# ---------------------------------------------------------------------------


async def test_missing_per_item_confidence_defaults_to_one():
    llm = _ScriptedLLM(
        [
            {
                "entities_list": [{"name": "Acme", "year": 2024}],  # no _confidence
                "is_done": True,
            }
        ]
    )
    _, confs = await extract_collection(Party, "body", [], llm)
    assert confs == [1.0]


async def test_non_dict_item_entries_are_ignored():
    llm = _ScriptedLLM(
        [
            {
                "entities_list": [
                    "this should be a dict not a string",
                    {"name": "Acme", "year": 2024, "_confidence": 0.9},
                ],
                "is_done": True,
            }
        ]
    )
    collected, _ = await extract_collection(Party, "body", [], llm)
    assert [p.name for p in collected] == ["Acme"]


# ---------------------------------------------------------------------------
# JSON schema round-trip sanity
# ---------------------------------------------------------------------------


def test_build_collection_schema_is_json_serialisable():
    # Prompt rendering does json.dumps on it; regressions would break the LLM call.
    schema = build_collection_schema(Party)
    json.dumps(schema, indent=2)
