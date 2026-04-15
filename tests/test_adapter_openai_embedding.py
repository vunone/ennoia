"""OpenAIEmbedding unit tests — mock the SDK client via _new_client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

pytest.importorskip("openai")

from ennoia.adapters.embedding.openai import OpenAIEmbedding


@dataclass
class _Datum:
    embedding: list[float]


@dataclass
class _Resp:
    data: list[_Datum]


class _Embeddings:
    def __init__(self, vector: list[float]) -> None:
        self._vector = vector
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _Resp:
        self.calls.append(kwargs)
        return _Resp(data=[_Datum(embedding=self._vector)])


class _Client:
    def __init__(self, vector: list[float]) -> None:
        self.embeddings = _Embeddings(vector)


def test_embed_document_and_query_share_path(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = OpenAIEmbedding(model="text-embedding-3-small", api_key="k")
    client = _Client([0.1, 0.2, 0.3])
    monkeypatch.setattr(adapter, "_new_client", lambda: client)

    assert adapter.embed_document("a") == [0.1, 0.2, 0.3]
    assert adapter.embed_query("b") == [0.1, 0.2, 0.3]
    assert client.embeddings.calls[0]["model"] == "text-embedding-3-small"


def test_api_key_falls_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    adapter = OpenAIEmbedding(model="text-embedding-3-small")
    assert adapter.api_key == "from-env"
