"""Store contract lives in ABCs: concrete stores inherit, bare ABCs can't instantiate."""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")

from ennoia.store import (  # noqa: E402
    HybridStore,
    InMemoryStructuredStore,
    InMemoryVectorStore,
    StructuredStore,
    VectorStore,
)


def test_in_memory_structured_is_structured_store():
    assert isinstance(InMemoryStructuredStore(), StructuredStore)


def test_in_memory_vector_is_vector_store():
    assert isinstance(InMemoryVectorStore(), VectorStore)


def test_instantiating_structured_abc_raises():
    with pytest.raises(TypeError):
        StructuredStore()  # type: ignore[abstract]


def test_instantiating_vector_abc_raises():
    with pytest.raises(TypeError):
        VectorStore()  # type: ignore[abstract]


def test_instantiating_hybrid_abc_raises():
    with pytest.raises(TypeError):
        HybridStore()  # type: ignore[abstract]


def test_partial_structured_impl_raises():
    class Partial(StructuredStore):  # missing filter() and get()
        async def upsert(self, source_id, data):  # type: ignore[no-untyped-def]
            pass

    with pytest.raises(TypeError):
        Partial()  # type: ignore[abstract]
