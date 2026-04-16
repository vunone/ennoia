"""Exercise the pytest fixtures exposed via the pytest11 entry point."""

from __future__ import annotations

from ennoia.testing import MockEmbeddingAdapter, MockLLMAdapter, MockStore


def test_mock_store_fixture(mock_store: MockStore) -> None:
    assert isinstance(mock_store, MockStore)


def test_mock_llm_fixture(mock_llm: MockLLMAdapter) -> None:
    assert isinstance(mock_llm, MockLLMAdapter)
    # Fresh fixture — no scripted responses configured.
    assert mock_llm.json_calls == []


def test_mock_embedding_fixture(mock_embedding: MockEmbeddingAdapter) -> None:
    assert isinstance(mock_embedding, MockEmbeddingAdapter)
    assert mock_embedding.dim == 8
