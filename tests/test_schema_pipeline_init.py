"""End-to-end Pipeline behavior with Schema.extensions."""

from __future__ import annotations

import warnings
from typing import Any, ClassVar, Literal

import pytest

pytest.importorskip("numpy")

from pydantic import Field as PydField

from ennoia import BaseSemantic, BaseStructure, Pipeline, Store
from ennoia.adapters.embedding import EmbeddingAdapter
from ennoia.adapters.llm import LLMAdapter
from ennoia.index.exceptions import SchemaError, SchemaWarning
from ennoia.store import InMemoryStructuredStore, InMemoryVectorStore


class _FakeEmbedding(EmbeddingAdapter):
    async def embed(self, text: str) -> list[float]:
        return [1.0, 0.0]


class _RecordingLLM(LLMAdapter):
    def __init__(self, json_responses: dict[str, dict[str, Any]]) -> None:
        self._json_responses = json_responses

    async def complete_json(self, prompt: str) -> dict[str, Any]:
        for key, value in self._json_responses.items():
            if key in prompt:
                return value
        raise AssertionError(f"No response for prompt starting: {prompt[:80]}")

    async def complete_text(self, prompt: str) -> str:
        return "topic"


def _store() -> Store:
    return Store(vector=InMemoryVectorStore(), structured=InMemoryStructuredStore())


# ---------------------------------------------------------------------------
# Superschema exposed on the Pipeline after init
# ---------------------------------------------------------------------------


class _Header(BaseStructure):
    """Simple root."""

    label: str


def test_pipeline_caches_superschema_on_init() -> None:
    pipeline = Pipeline(
        schemas=[_Header],
        store=_store(),
        llm=_RecordingLLM({}),
        embedding=_FakeEmbedding(),
    )
    assert pipeline._superschema.all_field_names() == {"label"}


# ---------------------------------------------------------------------------
# Init-time SchemaError on incompatible merge
# ---------------------------------------------------------------------------


class _BadA(BaseStructure):
    """String citation."""

    citation: str


class _BadB(BaseStructure):
    """Integer citation — incompatible."""

    citation: int


class _BadRoot(BaseStructure):
    """Root pulling in both incompatible children."""

    header: str

    class Schema:
        extensions: ClassVar[list[type]] = [_BadA, _BadB]


def test_pipeline_init_fails_on_incompatible_types() -> None:
    with pytest.raises(SchemaError, match="citation"):
        Pipeline(
            schemas=[_BadRoot],
            store=_store(),
            llm=_RecordingLLM({}),
            embedding=_FakeEmbedding(),
        )


# ---------------------------------------------------------------------------
# Divergent descriptions emit SchemaWarning at init
# ---------------------------------------------------------------------------


class _DescA(BaseStructure):
    """First description source."""

    field_x: str = PydField(default="", description="First description.")


class _DescB(BaseStructure):
    """Second description source."""

    field_x: str = PydField(default="", description="Second description.")


class _DescRoot(BaseStructure):
    """Root combining two description sources."""

    label: str

    class Schema:
        extensions: ClassVar[list[type]] = [_DescA, _DescB]


def test_pipeline_init_warns_on_divergent_descriptions() -> None:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        Pipeline(
            schemas=[_DescRoot],
            store=_store(),
            llm=_RecordingLLM({}),
            embedding=_FakeEmbedding(),
        )
    assert any(issubclass(w.category, SchemaWarning) for w in captured)


# ---------------------------------------------------------------------------
# Runtime extend() validation hook in executor
# ---------------------------------------------------------------------------


class _DeclaredChild(BaseStructure):
    """Declared child."""

    detail: str


class _UndeclaredChild(BaseStructure):
    """Not declared in Schema.extensions — should error at runtime."""

    other: str


class _DeclaresDeclared(BaseStructure):
    """Declares only _DeclaredChild but returns _UndeclaredChild from extend()."""

    trigger: bool

    class Schema:
        extensions: ClassVar[list[type]] = [_DeclaredChild]

    def extend(self) -> list[type[BaseStructure] | type[BaseSemantic]]:
        return [_UndeclaredChild]  # illegal


def test_runtime_extend_undeclared_raises() -> None:
    pipeline = Pipeline(
        schemas=[_DeclaresDeclared],
        store=_store(),
        llm=_RecordingLLM(
            {
                "Declares only": {"trigger": True},
            }
        ),
        embedding=_FakeEmbedding(),
    )
    with pytest.raises(SchemaError, match="not declared in Schema.extensions"):
        pipeline.index(text="anything", source_id="x1")


class _OkParent(BaseStructure):
    """Well-formed parent; returns only declared child."""

    trigger: bool

    class Schema:
        extensions: ClassVar[list[type]] = [_DeclaredChild]

    def extend(self) -> list[type[BaseStructure] | type[BaseSemantic]]:
        return [_DeclaredChild] if self.trigger else []


def test_runtime_extend_declared_succeeds() -> None:
    pipeline = Pipeline(
        schemas=[_OkParent],
        store=_store(),
        llm=_RecordingLLM(
            {
                "Well-formed parent;": {"trigger": True},
                "Declared child.": {"detail": "d"},
            }
        ),
        embedding=_FakeEmbedding(),
    )
    result = pipeline.index(text="x", source_id="x2")
    assert "_DeclaredChild" in result.structural


# ---------------------------------------------------------------------------
# Persist null-fills inactive branches
# ---------------------------------------------------------------------------


class _BranchA(BaseStructure):
    """Branch A."""

    a_value: str


class _BranchB(BaseStructure):
    """Branch B — may not fire."""

    b_value: str


class _BranchRoot(BaseStructure):
    """Root that extends to A but not B depending on a flag."""

    pick_a: bool

    class Schema:
        extensions: ClassVar[list[type]] = [_BranchA, _BranchB]

    def extend(self) -> list[type[BaseStructure] | type[BaseSemantic]]:
        return [_BranchA] if self.pick_a else [_BranchB]


def test_persist_null_fills_inactive_branches() -> None:
    import asyncio

    pipeline = Pipeline(
        schemas=[_BranchRoot],
        store=_store(),
        llm=_RecordingLLM(
            {
                "Root that extends": {"pick_a": True},
                "Branch A.": {"a_value": "AA"},
            }
        ),
        embedding=_FakeEmbedding(),
    )
    pipeline.index(text="x", source_id="only-a")
    stored = asyncio.run(pipeline.store.structured.get("only-a"))
    assert stored is not None
    assert stored["pick_a"] is True
    assert stored["a_value"] == "AA"
    # Branch B did not fire; its field is present with None, not missing.
    assert "b_value" in stored and stored["b_value"] is None


# ---------------------------------------------------------------------------
# Persist applies namespace prefix to emitted keys
# ---------------------------------------------------------------------------


class _NamespacedChild(BaseStructure):
    """Washington-specific details — fields prefixed with 'wa__'."""

    court_type: Literal["appellate", "supreme"] = "appellate"

    class Schema:
        namespace = "wa"


class _NsRoot(BaseStructure):
    """Root pulling in a namespaced child."""

    jurisdiction: Literal["WA"]

    class Schema:
        extensions: ClassVar[list[type]] = [_NamespacedChild]

    def extend(self) -> list[type[BaseStructure] | type[BaseSemantic]]:
        return [_NamespacedChild]


def test_persist_namespaced_fields_written_with_prefix() -> None:
    import asyncio

    pipeline = Pipeline(
        schemas=[_NsRoot],
        store=_store(),
        llm=_RecordingLLM(
            {
                "Root pulling in": {"jurisdiction": "WA"},
                "Washington-specific details": {"court_type": "supreme"},
            }
        ),
        embedding=_FakeEmbedding(),
    )
    pipeline.index(text="x", source_id="wa-1")
    stored = asyncio.run(pipeline.store.structured.get("wa-1"))
    assert stored is not None
    assert stored["jurisdiction"] == "WA"
    # Namespaced key, not flat.
    assert stored["wa__court_type"] == "supreme"
    assert "court_type" not in stored


# ---------------------------------------------------------------------------
# Search filter uses the superschema (reaches namespaced + merged fields)
# ---------------------------------------------------------------------------


def test_search_filter_over_namespaced_field_validates() -> None:
    pipeline = Pipeline(
        schemas=[_NsRoot],
        store=_store(),
        llm=_RecordingLLM(
            {
                "Root pulling in": {"jurisdiction": "WA"},
                "Washington-specific details": {"court_type": "supreme"},
            }
        ),
        embedding=_FakeEmbedding(),
    )
    pipeline.index(text="x", source_id="wa-1")
    # Should not raise — 'wa__court_type' is a legitimate superschema field.
    pipeline.search(query="q", filters={"wa__court_type": "supreme"})
