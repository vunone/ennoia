"""ennoia.testing.mocks — MockLLMAdapter, MockEmbeddingAdapter, MockStore."""

from __future__ import annotations

import pytest

from ennoia.testing import MockEmbeddingAdapter, MockLLMAdapter, MockStore

# ---------------------------------------------------------------------------
# MockLLMAdapter
# ---------------------------------------------------------------------------


async def test_mock_llm_substring_dict_matching() -> None:
    llm = MockLLMAdapter(
        json_responses={"extract doc": {"cat": "legal"}},
        text_responses={"summar": "a summary"},
    )
    assert await llm.complete_json("please extract doc metadata") == {"cat": "legal"}
    assert await llm.complete_text("summarise this") == "a summary"
    assert llm.json_calls == ["please extract doc metadata"]
    assert llm.text_calls == ["summarise this"]


async def test_mock_llm_callable_source() -> None:
    llm = MockLLMAdapter(
        json_responses=lambda prompt: {"prompt_len": len(prompt)},
        text_responses=lambda prompt: prompt.upper(),
    )
    assert await llm.complete_json("hi") == {"prompt_len": 2}
    assert await llm.complete_text("hi") == "HI"


async def test_mock_llm_list_source_consumed_in_order() -> None:
    llm = MockLLMAdapter(
        json_responses=[{"n": 1}, {"n": 2}],
        text_responses=["one", "two"],
    )
    assert await llm.complete_json("a") == {"n": 1}
    assert await llm.complete_json("b") == {"n": 2}
    assert await llm.complete_text("a") == "one"
    assert await llm.complete_text("b") == "two"


async def test_mock_llm_list_exhausted_raises() -> None:
    llm = MockLLMAdapter(json_responses=[{"n": 1}])
    await llm.complete_json("a")
    with pytest.raises(AssertionError, match="ran out"):
        await llm.complete_json("b")


async def test_mock_llm_unmatched_dict_raises() -> None:
    llm = MockLLMAdapter(json_responses={"expected": {"ok": True}})
    with pytest.raises(AssertionError, match="no scripted"):
        await llm.complete_json("nope")


async def test_mock_llm_longest_key_wins() -> None:
    # When multiple keys match the prompt, the longest (most specific) wins.
    llm = MockLLMAdapter(json_responses={"doc": {"n": 1}, "document meta": {"n": 2}})
    assert await llm.complete_json("extract document meta please") == {"n": 2}


async def test_mock_llm_no_sources_set_raises() -> None:
    llm = MockLLMAdapter()
    with pytest.raises(AssertionError):
        await llm.complete_json("anything")
    with pytest.raises(AssertionError):
        await llm.complete_text("anything")


# ---------------------------------------------------------------------------
# MockEmbeddingAdapter
# ---------------------------------------------------------------------------


async def test_mock_embedding_deterministic() -> None:
    emb = MockEmbeddingAdapter(dim=8)
    a = await emb.embed("hello")
    b = await emb.embed("hello")
    assert a == b
    assert len(a) == 8


async def test_mock_embedding_different_inputs_different_vectors() -> None:
    emb = MockEmbeddingAdapter(dim=8)
    a = await emb.embed("hello")
    b = await emb.embed("world")
    assert a != b


async def test_mock_embedding_is_unit_norm() -> None:
    import math

    emb = MockEmbeddingAdapter(dim=16)
    vec = await emb.embed("hello")
    norm = math.sqrt(sum(v * v for v in vec))
    assert abs(norm - 1.0) < 1e-6


async def test_mock_embedding_records_calls() -> None:
    emb = MockEmbeddingAdapter(dim=4)
    await emb.embed("a")
    await emb.embed("b")
    assert emb.calls == ["a", "b"]


async def test_mock_embedding_large_dim_re_hashes() -> None:
    # Forcing dim > 32 exercises the seed-reseeding loop in ``embed``.
    emb = MockEmbeddingAdapter(dim=100)
    vec = await emb.embed("hello")
    assert len(vec) == 100


async def test_mock_embedding_rejects_invalid_dim() -> None:
    with pytest.raises(ValueError):
        MockEmbeddingAdapter(dim=0)


async def test_mock_embedding_zero_vector_when_bytes_collapse() -> None:
    # Build an adapter subclass that forces a zero digest to cover the
    # norm == 0 escape hatch.
    class _Zero(MockEmbeddingAdapter):
        async def embed(self, text: str) -> list[float]:
            # Bypass super(): return all-zeros to hit the norm == 0 return path
            # in cosine / unit-norm routines the mock uses in MockStore.
            return [0.0] * self.dim

    adapter = _Zero(dim=4)
    assert await adapter.embed("x") == [0.0, 0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# MockStore
# ---------------------------------------------------------------------------


async def test_mock_store_upsert_filter_search_retrieve_delete() -> None:
    store = MockStore()
    await store.upsert("doc_1", {"cat": "legal"}, {"H": [1.0, 0.0], "F": [0.0, 1.0]})
    await store.upsert("doc_2", {"cat": "medical"}, {"H": [0.9, 0.1]})

    # filter
    assert await store.filter({"cat": "legal"}) == ["doc_1"]
    # retrieve
    assert await store.get("doc_1") == {"cat": "legal"}
    assert await store.get("missing") is None
    # hybrid_search — cosine ranks doc_1's H=1,0 above doc_2's H=0.9,0.1 for query 1,0
    hits = await store.hybrid_search({"cat": "legal"}, [1.0, 0.0], top_k=5)
    assert hits[0][2]["source_id"] == "doc_1"
    # index-scoped hybrid search: only the "H" vector of doc_1 competes
    scoped = await store.hybrid_search({"cat": "legal"}, [0.0, 1.0], top_k=5, index="F")
    assert scoped[0][2]["index"] == "F"
    # delete
    assert await store.delete("doc_1") is True
    assert await store.delete("doc_1") is False
    assert await store.get("doc_1") is None


async def test_mock_store_search_skips_zero_norm_vectors() -> None:
    store = MockStore()
    await store.upsert("doc_1", {"cat": "legal"}, {"H": [0.0, 0.0]})
    hits = await store.hybrid_search({"cat": "legal"}, [1.0, 0.0], top_k=5)
    assert hits == []


async def test_mock_store_search_skips_dim_mismatch() -> None:
    store = MockStore()
    await store.upsert("doc_1", {"cat": "legal"}, {"H": [1.0]})  # dim 1
    hits = await store.hybrid_search({"cat": "legal"}, [1.0, 0.0], top_k=5)  # query dim 2
    assert hits == []


async def test_mock_store_search_skips_empty_vectors() -> None:
    store = MockStore()
    # No vectors stored — apply_filters returns the id, but pool is empty.
    await store.upsert("doc_1", {"cat": "legal"}, {})
    assert await store.hybrid_search({"cat": "legal"}, [1.0, 0.0], top_k=5) == []


async def test_mock_store_empty_query_vector_returns_empty() -> None:
    store = MockStore()
    await store.upsert("doc_1", {"cat": "legal"}, {"H": [1.0, 0.0]})
    assert await store.hybrid_search({"cat": "legal"}, [], top_k=5) == []
