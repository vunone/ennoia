"""Adapter contracts live in ABCs: concrete adapters inherit, bare ABCs can't instantiate."""

from __future__ import annotations

from typing import Any

import pytest

from ennoia.adapters.embedding.base import EmbeddingAdapter
from ennoia.adapters.llm.base import LLMAdapter, parse_json_object
from ennoia.index.exceptions import ExtractionError


def test_instantiating_llm_abc_raises() -> None:
    with pytest.raises(TypeError):
        LLMAdapter()  # type: ignore[abstract]


def test_instantiating_embedding_abc_raises() -> None:
    with pytest.raises(TypeError):
        EmbeddingAdapter()  # type: ignore[abstract]


def test_partial_llm_impl_raises() -> None:
    class Partial(LLMAdapter):  # missing complete_text
        async def complete_json(self, prompt: str) -> dict[str, Any]:
            return {}

    with pytest.raises(TypeError):
        Partial()  # type: ignore[abstract]


def test_partial_embedding_impl_raises() -> None:
    class Partial(EmbeddingAdapter):
        pass

    with pytest.raises(TypeError):
        Partial()  # type: ignore[abstract]


async def test_embedding_delegators_route_through_embed() -> None:
    """`embed_document`, `embed_query`, and `embed_batch` all delegate to `embed`."""
    calls: list[str] = []

    class Stub(EmbeddingAdapter):
        async def embed(self, text: str) -> list[float]:
            calls.append(text)
            return [1.0]

    stub = Stub()
    assert await stub.embed_document("doc") == [1.0]
    assert await stub.embed_query("query") == [1.0]
    # Default embed_batch fans out to embed() in parallel.
    assert await stub.embed_batch(["a", "b"]) == [[1.0], [1.0]]
    assert calls == ["doc", "query", "a", "b"]


def test_concrete_llm_adapters_are_llm_adapter_instances() -> None:
    pytest.importorskip("ollama")
    from ennoia.adapters.llm.ollama import OllamaAdapter

    assert isinstance(OllamaAdapter(model="dummy"), LLMAdapter)


def test_concrete_openai_llm_adapter_is_llm_adapter_instance() -> None:
    pytest.importorskip("openai")
    from ennoia.adapters.llm.openai import OpenAIAdapter

    assert isinstance(OpenAIAdapter(model="gpt-4o-mini", api_key="k"), LLMAdapter)


def test_concrete_anthropic_llm_adapter_is_llm_adapter_instance() -> None:
    pytest.importorskip("anthropic")
    from ennoia.adapters.llm.anthropic import AnthropicAdapter

    assert isinstance(AnthropicAdapter(model="claude-3-5-sonnet", api_key="k"), LLMAdapter)


def test_concrete_openai_embedding_is_embedding_adapter_instance() -> None:
    pytest.importorskip("openai")
    from ennoia.adapters.embedding.openai import OpenAIEmbedding

    adapter = OpenAIEmbedding(model="text-embedding-3-small", api_key="k")
    assert isinstance(adapter, EmbeddingAdapter)


def test_parse_json_object_happy_path() -> None:
    assert parse_json_object('{"a": 1}', "Test") == {"a": 1}


def test_parse_json_object_raises_on_invalid_json() -> None:
    with pytest.raises(ExtractionError, match="non-JSON"):
        parse_json_object("not json", "Test")


def test_parse_json_object_raises_on_non_dict() -> None:
    with pytest.raises(ExtractionError, match="Expected a JSON object"):
        parse_json_object("[1, 2, 3]", "Test")
