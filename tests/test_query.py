"""Query planner — composite-store filter-then-search + HybridStore delegation."""

from __future__ import annotations

from typing import Any

import pytest

from ennoia.index.query import plan_search
from ennoia.store import HybridStore, InMemoryStructuredStore, InMemoryVectorStore, Store


class _RecordingVectorStore(InMemoryVectorStore):
    """InMemoryVectorStore that records the ``restrict_to`` arg it received."""

    def __init__(self) -> None:
        super().__init__()
        self.last_restrict_to: list[str] | None = None

    async def search(
        self,
        query_vector: list[float],
        top_k: int,
        restrict_to: list[str] | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        self.last_restrict_to = restrict_to
        return await super().search(query_vector, top_k, restrict_to)


async def test_plan_search_composite_store_passes_restrict_to() -> None:
    vector = _RecordingVectorStore()
    structured = InMemoryStructuredStore()
    await structured.upsert("a", {"cat": "legal"})
    await structured.upsert("b", {"cat": "medical"})
    await vector.upsert("a:S", [1.0, 0.0], {"source_id": "a"})
    await vector.upsert("b:S", [0.9, 0.1], {"source_id": "b"})

    store = Store(vector=vector, structured=structured)
    await plan_search(store, {"cat": "legal"}, [1.0, 0.0], top_k=5)
    assert vector.last_restrict_to == ["a"]


async def test_plan_search_returns_empty_when_filter_eliminates_candidates() -> None:
    vector = _RecordingVectorStore()
    structured = InMemoryStructuredStore()
    await structured.upsert("a", {"cat": "legal"})
    store = Store(vector=vector, structured=structured)
    assert await plan_search(store, {"cat": "non-existent"}, [1.0], top_k=5) == []
    # Vector store short-circuited — never called.
    assert vector.last_restrict_to is None


async def test_plan_search_normalises_none_filters_to_empty_dict() -> None:
    vector = _RecordingVectorStore()
    structured = InMemoryStructuredStore()
    await structured.upsert("a", {"cat": "legal"})
    await vector.upsert("a:S", [1.0, 0.0], {"source_id": "a"})
    store = Store(vector=vector, structured=structured)

    hits = await plan_search(store, None, [1.0, 0.0], top_k=5)
    assert [h[0] for h in hits] == ["a:S"]
    assert vector.last_restrict_to == ["a"]


# ---------------------------------------------------------------------------
# HybridStore path
# ---------------------------------------------------------------------------


class _FakeHybrid(HybridStore):
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def upsert(
        self,
        source_id: str,
        data: dict[str, Any],
        vectors: dict[str, list[float]],
    ) -> None:
        pytest.fail("upsert should not be called from plan_search")

    async def hybrid_search(
        self,
        filters: dict[str, Any],
        query_vector: list[float],
        top_k: int,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        self.calls.append(
            {"filters": dict(filters), "query_vector": list(query_vector), "top_k": top_k}
        )
        return [("hit-1", 0.42, {"source_id": "hit-1"})]

    async def get(self, source_id: str) -> dict[str, Any] | None:
        return None


async def test_plan_search_hybrid_store_delegates_to_hybrid_search() -> None:
    hybrid = _FakeHybrid()
    hits = await plan_search(hybrid, {"cat": "legal"}, [0.1, 0.2], top_k=3)
    assert hits == [("hit-1", 0.42, {"source_id": "hit-1"})]
    assert hybrid.calls == [{"filters": {"cat": "legal"}, "query_vector": [0.1, 0.2], "top_k": 3}]
