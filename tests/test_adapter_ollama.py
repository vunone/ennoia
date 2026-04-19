"""OllamaAdapter unit tests — mock the SDK client via a fake ollama module."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

import ennoia.adapters.llm.ollama as ollama_module
from ennoia.adapters.llm.ollama import OllamaAdapter
from ennoia.index.exceptions import ExtractionError


class _FakeInnerHttpx:
    """Stand-in for the httpx.AsyncClient that ollama.AsyncClient composes.

    The real ``ollama.AsyncClient`` doesn't support ``async with`` — the
    httpx transport lives at ``self._client`` and is closed via
    ``await self._client.aclose()``. The adapter mirrors this, so the fake
    exposes the same shape.
    """

    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class _FakeAsyncClient:
    """Captures init kwargs and scripts ``chat`` responses."""

    last_kwargs: dict[str, Any] | None = None
    script: dict[str, Any] | None = None

    def __init__(self, **kwargs: Any) -> None:
        type(self).last_kwargs = kwargs
        self.calls: list[dict[str, Any]] = []
        self._client = _FakeInnerHttpx()

    async def chat(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return type(self).script or {"message": {"content": ""}}


def _patch_ollama_module(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeAsyncClient.last_kwargs = None
    _FakeAsyncClient.script = None
    fake_module = SimpleNamespace(AsyncClient=_FakeAsyncClient)
    monkeypatch.setattr(ollama_module, "require_module", lambda *_args: fake_module)


def test_new_client_passes_host_and_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_ollama_module(monkeypatch)
    adapter = OllamaAdapter(model="qwen3:0.6b", host="http://ollama.local:11434", timeout=15.0)
    client = adapter._new_client()
    assert isinstance(client, _FakeAsyncClient)
    assert _FakeAsyncClient.last_kwargs == {
        "host": "http://ollama.local:11434",
        "timeout": 15.0,
    }


async def test_complete_json_parses_content(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_ollama_module(monkeypatch)
    _FakeAsyncClient.script = {"message": {"content": json.dumps({"category": "legal"})}}
    adapter = OllamaAdapter(model="qwen3:0.6b")
    result = await adapter.complete_json("prompt")
    assert result == {"category": "legal"}


async def test_complete_json_uses_json_format_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_ollama_module(monkeypatch)
    _FakeAsyncClient.script = {"message": {"content": "{}"}}
    adapter = OllamaAdapter(model="qwen3:0.6b")

    captured_client: dict[str, _FakeAsyncClient] = {}

    original_new = adapter._new_client

    def capturing_new() -> _FakeAsyncClient:
        c = original_new()
        captured_client["c"] = c
        return c

    monkeypatch.setattr(adapter, "_new_client", capturing_new)
    await adapter.complete_json("prompt-text")
    call = captured_client["c"].calls[0]
    assert call["format"] == "json"
    assert call["model"] == "qwen3:0.6b"


async def test_complete_json_raises_on_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_ollama_module(monkeypatch)
    _FakeAsyncClient.script = {"message": {"content": "not json"}}
    adapter = OllamaAdapter(model="qwen3:0.6b")
    with pytest.raises(ExtractionError):
        await adapter.complete_json("prompt")


async def test_complete_text_returns_content_as_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_ollama_module(monkeypatch)
    _FakeAsyncClient.script = {"message": {"content": "hello there"}}
    adapter = OllamaAdapter(model="qwen3:0.6b")
    assert await adapter.complete_text("prompt") == "hello there"


async def test_complete_text_closes_inner_httpx_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every call must close the composed httpx client; otherwise the
    transport leaks and httpx's GC finalizer eventually fires
    ``aclose()`` on a dead event loop."""
    _patch_ollama_module(monkeypatch)
    _FakeAsyncClient.script = {"message": {"content": "ok"}}
    adapter = OllamaAdapter(model="qwen3:0.6b")

    captured: dict[str, _FakeAsyncClient] = {}
    original_new = adapter._new_client

    def capturing_new() -> _FakeAsyncClient:
        c = original_new()
        captured["c"] = c
        return c

    monkeypatch.setattr(adapter, "_new_client", capturing_new)
    await adapter.complete_text("prompt")
    assert captured["c"]._client.closed is True


async def test_aclose_ollama_client_tolerates_missing_inner() -> None:
    """If the SDK shape changes and ``_client`` is absent, the helper must
    not raise — the adapter should still return its parsed response."""
    from ennoia.adapters.llm.ollama import _aclose_ollama_client

    class _ShapelessClient:
        pass

    await _aclose_ollama_client(_ShapelessClient())  # must not raise
