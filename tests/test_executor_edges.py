"""Executor edge cases — undeclared extend(), semantic extensions, dedup, diamonds."""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from ennoia import BaseSemantic, BaseStructure
from ennoia.adapters.llm.base import LLMAdapter
from ennoia.index.exceptions import SchemaError
from ennoia.index.executor import execute_layers


class _LLM(LLMAdapter):
    def __init__(self, json_responses: dict[str, dict[str, Any]]) -> None:
        self._json = json_responses

    async def complete_json(self, prompt: str) -> dict[str, Any]:
        for key, value in self._json.items():
            if key in prompt:
                return value
        raise AssertionError(f"Unexpected prompt: {prompt[:80]}")

    async def complete_text(self, prompt: str) -> str:
        return "text-answer <confidence>0.9</confidence>"


# ---------------------------------------------------------------------------
# Undeclared extend() target
# ---------------------------------------------------------------------------


class _Undeclared(BaseStructure):
    """Undeclared child schema."""

    x: str


class _ParentEmittingUndeclared(BaseStructure):
    """Parent whose extend() returns an undeclared child."""

    label: str

    class Schema:
        # Deliberately empty — ``extend()`` below violates the contract.
        extensions: ClassVar[list[type]] = []

    def extend(self) -> list[type[BaseStructure] | type[BaseSemantic]]:
        return [_Undeclared]


async def test_extend_returning_undeclared_class_raises_schema_error() -> None:
    llm = _LLM({"Parent whose extend": {"label": "x", "_confidence": 0.9}})
    with pytest.raises(SchemaError, match="not declared in Schema.extensions"):
        await execute_layers(
            seed_structural=[_ParentEmittingUndeclared],
            seed_semantic=[],
            text="body",
            llm=llm,
        )


# ---------------------------------------------------------------------------
# Semantic extension queued from extend()
# ---------------------------------------------------------------------------


class _SemanticChild(BaseSemantic):
    """Summarise child section?"""


class _ParentWithSemanticChild(BaseStructure):
    """Parent pulling in a semantic child via extend()."""

    label: str

    class Schema:
        extensions: ClassVar[list[type]] = [_SemanticChild]

    def extend(self) -> list[type[BaseStructure] | type[BaseSemantic]]:
        return [_SemanticChild]


async def test_extend_queues_semantic_child() -> None:
    llm = _LLM({"Parent pulling in a semantic child": {"label": "x", "_confidence": 0.9}})
    batch = await execute_layers(
        seed_structural=[_ParentWithSemanticChild],
        seed_semantic=[],
        text="body",
        llm=llm,
    )
    assert "_SemanticChild" in batch.semantic


# ---------------------------------------------------------------------------
# Semantic dedup — same semantic queued by two parents
# ---------------------------------------------------------------------------


class _SharedSem(BaseSemantic):
    """Summarise shared concern?"""


class _ParentA(BaseStructure):
    """Parent A pulls in shared semantic."""

    v: str

    class Schema:
        extensions: ClassVar[list[type]] = [_SharedSem]

    def extend(self) -> list[type[BaseStructure] | type[BaseSemantic]]:
        return [_SharedSem]


class _ParentB(BaseStructure):
    """Parent B pulls in shared semantic."""

    v: str

    class Schema:
        extensions: ClassVar[list[type]] = [_SharedSem]

    def extend(self) -> list[type[BaseStructure] | type[BaseSemantic]]:
        return [_SharedSem]


async def test_semantic_queued_by_two_parents_runs_once() -> None:
    llm = _LLM(
        {
            "Parent A pulls in shared semantic.": {"v": "a", "_confidence": 0.9},
            "Parent B pulls in shared semantic.": {"v": "b", "_confidence": 0.9},
        }
    )
    batch = await execute_layers(
        seed_structural=[_ParentA, _ParentB],
        seed_semantic=[],
        text="body",
        llm=llm,
    )
    # Single semantic entry — the second parent's request is deduped.
    assert list(batch.semantic.keys()) == ["_SharedSem"]


# ---------------------------------------------------------------------------
# Structural diamond — same structural child queued by two parents
# ---------------------------------------------------------------------------


class _DiamondLeaf(BaseStructure):
    """Shared leaf node."""

    w: str


class _DiamondLeft(BaseStructure):
    """Left arm emits the shared leaf."""

    left: str

    class Schema:
        extensions: ClassVar[list[type]] = [_DiamondLeaf]

    def extend(self) -> list[type[BaseStructure] | type[BaseSemantic]]:
        return [_DiamondLeaf]


class _DiamondRight(BaseStructure):
    """Right arm emits the shared leaf."""

    right: str

    class Schema:
        extensions: ClassVar[list[type]] = [_DiamondLeaf]

    def extend(self) -> list[type[BaseStructure] | type[BaseSemantic]]:
        return [_DiamondLeaf]


async def test_structural_diamond_dedupes_shared_leaf() -> None:
    llm = _LLM(
        {
            "Left arm emits the shared leaf.": {"left": "L", "_confidence": 0.9},
            "Right arm emits the shared leaf.": {"right": "R", "_confidence": 0.9},
            "Shared leaf node.": {"w": "shared", "_confidence": 0.9},
        }
    )
    batch = await execute_layers(
        seed_structural=[_DiamondLeft, _DiamondRight],
        seed_semantic=[],
        text="body",
        llm=llm,
    )
    # Leaf ran exactly once despite being declared by both parents.
    assert sorted(batch.structural.keys()) == [
        "_DiamondLeaf",
        "_DiamondLeft",
        "_DiamondRight",
    ]


# ---------------------------------------------------------------------------
# Already-seen structural child returned by extend() is ignored
# ---------------------------------------------------------------------------


class _AlreadySeen(BaseStructure):
    """Leaf declared on both sides — only runs once."""

    z: str


class _ReEmitter(BaseStructure):
    """Schema that re-emits its sibling on extend()."""

    y: str

    class Schema:
        extensions: ClassVar[list[type]] = [_AlreadySeen]

    def extend(self) -> list[type[BaseStructure] | type[BaseSemantic]]:
        return [_AlreadySeen]


async def test_extend_targeting_already_processed_schema_is_skipped() -> None:
    # Seed includes both the re-emitter and the leaf — the leaf is therefore
    # already in ``seen_structural`` by the time the re-emitter calls extend().
    llm = _LLM(
        {
            "Schema that re-emits its sibling on extend()": {"y": "y", "_confidence": 0.9},
            "Leaf declared on both sides": {"z": "z", "_confidence": 0.9},
        }
    )
    batch = await execute_layers(
        seed_structural=[_ReEmitter, _AlreadySeen],
        seed_semantic=[],
        text="body",
        llm=llm,
    )
    assert sorted(batch.structural.keys()) == ["_AlreadySeen", "_ReEmitter"]
