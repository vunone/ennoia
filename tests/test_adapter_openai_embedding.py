"""OpenAIEmbedding unit tests — mock the AsyncOpenAI client via _new_client."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

import ennoia.adapters.embedding.openai as openai_embedding_module
from ennoia.adapters.embedding.openai import OpenAIEmbedding


@dataclass
class _Datum:
    embedding: list[float]


@dataclass
class _Resp:
    data: list[_Datum]


class _Embeddings:
    """Async fake matching ``AsyncOpenAI().embeddings``: ``await create(...)``."""

    def __init__(self, vector: list[float]) -> None:
        self._vector = vector
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _Resp:
        self.calls.append(kwargs)
        # When ``input`` is a list, return one Datum per text — matches the
        # real SDK contract for batch embedding requests.
        if isinstance(kwargs.get("input"), list):
            return _Resp(data=[_Datum(embedding=self._vector) for _ in kwargs["input"]])
        return _Resp(data=[_Datum(embedding=self._vector)])


class _Client:
    def __init__(self, vector: list[float]) -> None:
        self.embeddings = _Embeddings(vector)


async def test_embed_document_and_query_share_path(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = OpenAIEmbedding(model="text-embedding-3-small", api_key="k")
    client = _Client([0.1, 0.2, 0.3])
    monkeypatch.setattr(adapter, "_new_client", lambda: client)

    assert await adapter.embed_document("a") == [0.1, 0.2, 0.3]
    assert await adapter.embed_query("b") == [0.1, 0.2, 0.3]
    assert client.embeddings.calls[0]["model"] == "text-embedding-3-small"


async def test_embed_batch_issues_single_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """``embed_batch`` MUST hit the SDK once with ``input=[...]`` — not N times."""
    adapter = OpenAIEmbedding(model="text-embedding-3-small", api_key="k")
    client = _Client([0.4, 0.5, 0.6])
    monkeypatch.setattr(adapter, "_new_client", lambda: client)

    vectors = await adapter.embed_batch(["a", "b", "c"])
    assert vectors == [[0.4, 0.5, 0.6]] * 3
    # One call for the full batch — saves N-1 round trips for multi-semantic docs.
    assert len(client.embeddings.calls) == 1
    assert client.embeddings.calls[0]["input"] == ["a", "b", "c"]


async def test_embed_batch_empty_short_circuits(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = OpenAIEmbedding(model="text-embedding-3-small", api_key="k")
    client = _Client([0.0])
    monkeypatch.setattr(adapter, "_new_client", lambda: client)

    assert await adapter.embed_batch([]) == []
    # No round-trip for an empty batch.
    assert client.embeddings.calls == []


def test_api_key_falls_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    adapter = OpenAIEmbedding(model="text-embedding-3-small")
    assert adapter.api_key == "from-env"


class _AsyncOpenAISpy:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def _patch_openai_embedding_module(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = SimpleNamespace(AsyncOpenAI=_AsyncOpenAISpy)
    monkeypatch.setattr(openai_embedding_module, "require_module", lambda *_args: fake_module)


def test_new_client_passes_all_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_openai_embedding_module(monkeypatch)
    adapter = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key="sk-test",
        base_url="https://proxy.local",
        timeout=5.0,
    )
    client = adapter._new_client()
    assert isinstance(client, _AsyncOpenAISpy)
    assert client.kwargs == {
        "api_key": "sk-test",
        "base_url": "https://proxy.local",
        "timeout": 5.0,
    }


def test_new_client_omits_unset_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_openai_embedding_module(monkeypatch)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    adapter = OpenAIEmbedding(model="text-embedding-3-small")
    client = adapter._new_client()
    assert client.kwargs == {"timeout": 60.0}


def test_new_client_uses_async_openai_constructor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Loop-lifecycle invariant: the adapter must instantiate the *async* client."""
    _patch_openai_embedding_module(monkeypatch)
    adapter = OpenAIEmbedding(model="text-embedding-3-small", api_key="k")
    client = adapter._new_client()
    # If this regresses to ``module.OpenAI``, the sync pipeline wrappers will
    # call a coroutine on a sync client and explode at runtime.
    assert isinstance(client, _AsyncOpenAISpy)
