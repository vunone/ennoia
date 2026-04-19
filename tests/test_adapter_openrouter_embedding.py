"""OpenRouterEmbedding unit tests — mock the AsyncOpenAI client via _new_client."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

import ennoia.adapters.embedding.openrouter as openrouter_embedding_module
from ennoia.adapters.embedding.openrouter import OpenRouterEmbedding


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
        if isinstance(kwargs.get("input"), list):
            return _Resp(data=[_Datum(embedding=self._vector) for _ in kwargs["input"]])
        return _Resp(data=[_Datum(embedding=self._vector)])


class _Client:
    def __init__(self, vector: list[float]) -> None:
        self.embeddings = _Embeddings(vector)
        self.closed = False

    async def __aenter__(self) -> _Client:
        return self

    async def __aexit__(self, *_: Any) -> None:
        self.closed = True


async def test_embed_document_and_query_share_path(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = OpenRouterEmbedding(model="openai/text-embedding-3-small", api_key="k")
    client = _Client([0.1, 0.2, 0.3])
    monkeypatch.setattr(adapter, "_new_client", lambda: client)

    assert await adapter.embed_document("a") == [0.1, 0.2, 0.3]
    assert await adapter.embed_query("b") == [0.1, 0.2, 0.3]
    assert client.embeddings.calls[0]["model"] == "openai/text-embedding-3-small"


async def test_embed_batch_issues_single_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """``embed_batch`` MUST hit the SDK once with ``input=[...]`` — not N times."""
    adapter = OpenRouterEmbedding(model="openai/text-embedding-3-small", api_key="k")
    client = _Client([0.4, 0.5, 0.6])
    monkeypatch.setattr(adapter, "_new_client", lambda: client)

    vectors = await adapter.embed_batch(["a", "b", "c"])
    assert vectors == [[0.4, 0.5, 0.6]] * 3
    assert len(client.embeddings.calls) == 1
    assert client.embeddings.calls[0]["input"] == ["a", "b", "c"]


async def test_embed_batch_empty_short_circuits(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = OpenRouterEmbedding(model="m", api_key="k")
    client = _Client([0.0])
    monkeypatch.setattr(adapter, "_new_client", lambda: client)

    assert await adapter.embed_batch([]) == []
    assert client.embeddings.calls == []


def test_api_key_falls_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-env")
    adapter = OpenRouterEmbedding(model="m")
    assert adapter.api_key == "from-env"


def test_explicit_api_key_wins_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-env")
    adapter = OpenRouterEmbedding(model="m", api_key="explicit")
    assert adapter.api_key == "explicit"


def test_default_base_url_is_openrouter() -> None:
    adapter = OpenRouterEmbedding(model="m", api_key="k")
    assert adapter.base_url == "https://openrouter.ai/api/v1"


def test_explicit_base_url_overrides_default() -> None:
    adapter = OpenRouterEmbedding(model="m", api_key="k", base_url="https://proxy.local/v1")
    assert adapter.base_url == "https://proxy.local/v1"


class _AsyncOpenAISpy:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def _patch_openai_embedding_module(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = SimpleNamespace(AsyncOpenAI=_AsyncOpenAISpy)
    monkeypatch.setattr(openrouter_embedding_module, "require_module", lambda *_args: fake_module)


def test_new_client_passes_all_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_openai_embedding_module(monkeypatch)
    adapter = OpenRouterEmbedding(
        model="m",
        api_key="sk-or-test",
        base_url="https://proxy.local/v1",
        timeout=5.0,
    )
    client = adapter._new_client()
    assert isinstance(client, _AsyncOpenAISpy)
    assert client.kwargs == {
        "api_key": "sk-or-test",
        "base_url": "https://proxy.local/v1",
        "timeout": 5.0,
    }


def test_new_client_applies_default_base_url_and_omits_missing_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_openai_embedding_module(monkeypatch)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    adapter = OpenRouterEmbedding(model="m")
    client = adapter._new_client()
    assert client.kwargs == {
        "base_url": "https://openrouter.ai/api/v1",
        "timeout": 60.0,
    }


def test_new_client_uses_async_openai_constructor(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_openai_embedding_module(monkeypatch)
    adapter = OpenRouterEmbedding(model="m", api_key="k")
    client = adapter._new_client()
    assert isinstance(client, _AsyncOpenAISpy)


def test_require_module_uses_openrouter_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, str] = {}

    def _fake_require(name: str, extra: str) -> Any:
        seen["name"] = name
        seen["extra"] = extra
        return SimpleNamespace(AsyncOpenAI=_AsyncOpenAISpy)

    monkeypatch.setattr(openrouter_embedding_module, "require_module", _fake_require)
    adapter = OpenRouterEmbedding(model="m", api_key="k")
    adapter._new_client()
    assert seen == {"name": "openai", "extra": "openrouter"}
