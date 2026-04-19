"""OpenRouterAdapter unit tests — mock the SDK client via _new_client."""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

import ennoia.adapters.llm.openrouter as openrouter_module
from ennoia.adapters.llm.openrouter import OpenRouterAdapter
from ennoia.index.exceptions import ExtractionError


@dataclass
class _Msg:
    content: str


@dataclass
class _Choice:
    message: _Msg


@dataclass
class _Resp:
    choices: list[_Choice]


class _FakeCompletions:
    def __init__(self, payload: str) -> None:
        self._payload = payload
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _Resp:
        self.calls.append(kwargs)
        return _Resp(choices=[_Choice(message=_Msg(content=self._payload))])


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions = completions


class _FakeClient:
    def __init__(self, payload: str) -> None:
        self.chat = _FakeChat(_FakeCompletions(payload))
        self.closed = False

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *_: Any) -> None:
        self.closed = True


async def test_complete_json_round_trips(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = OpenRouterAdapter(model="meta-llama/llama-3.1-8b-instruct", api_key="sk-or-test")
    client = _FakeClient(json.dumps({"category": "legal"}))
    monkeypatch.setattr(adapter, "_new_client", lambda: client)
    assert await adapter.complete_json("prompt") == {"category": "legal"}
    call = client.chat.completions.calls[0]
    assert call["response_format"] == {"type": "json_object"}
    assert call["model"] == "meta-llama/llama-3.1-8b-instruct"


async def test_complete_json_raises_on_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = OpenRouterAdapter(model="m", api_key="k")
    monkeypatch.setattr(adapter, "_new_client", lambda: _FakeClient("not json"))
    with pytest.raises(ExtractionError, match="OpenRouter"):
        await adapter.complete_json("prompt")


async def test_complete_text_returns_plain_string(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = OpenRouterAdapter(model="m", api_key="k")
    monkeypatch.setattr(adapter, "_new_client", lambda: _FakeClient("hello"))
    assert await adapter.complete_text("prompt") == "hello"


def test_api_key_falls_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-env")
    adapter = OpenRouterAdapter(model="m")
    assert adapter.api_key == "from-env"


def test_explicit_api_key_wins_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-env")
    adapter = OpenRouterAdapter(model="m", api_key="explicit")
    assert adapter.api_key == "explicit"


def test_default_base_url_is_openrouter() -> None:
    adapter = OpenRouterAdapter(model="m", api_key="k")
    assert adapter.base_url == "https://openrouter.ai/api/v1"


def test_explicit_base_url_overrides_default() -> None:
    adapter = OpenRouterAdapter(model="m", api_key="k", base_url="https://proxy.local/v1")
    assert adapter.base_url == "https://proxy.local/v1"


class _AsyncOpenAISpy:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def _patch_openai_module(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = SimpleNamespace(AsyncOpenAI=_AsyncOpenAISpy)
    monkeypatch.setattr(openrouter_module, "require_module", lambda *_args: fake_module)


def test_new_client_passes_all_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_openai_module(monkeypatch)
    adapter = OpenRouterAdapter(
        model="m",
        api_key="sk-or-test",
        base_url="https://proxy.local/v1",
        timeout=12.0,
    )
    client = adapter._new_client()
    assert isinstance(client, _AsyncOpenAISpy)
    assert client.kwargs == {
        "api_key": "sk-or-test",
        "base_url": "https://proxy.local/v1",
        "timeout": 12.0,
    }


def test_new_client_applies_default_base_url_and_omits_missing_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_openai_module(monkeypatch)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    adapter = OpenRouterAdapter(model="m")
    client = adapter._new_client()
    # No api_key when unset; base_url always flows through so the SDK
    # doesn't silently hit OpenAI's endpoint.
    assert client.kwargs == {
        "base_url": "https://openrouter.ai/api/v1",
        "timeout": 60.0,
    }


def test_new_client_uses_async_openai_constructor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Loop-lifecycle invariant: the adapter must instantiate the *async* client."""
    _patch_openai_module(monkeypatch)
    adapter = OpenRouterAdapter(model="m", api_key="k")
    client = adapter._new_client()
    assert isinstance(client, _AsyncOpenAISpy)


def test_require_module_uses_openrouter_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """Import errors must point users at ``pip install ennoia[openrouter]``."""
    seen: dict[str, str] = {}

    def _fake_require(name: str, extra: str) -> Any:
        seen["name"] = name
        seen["extra"] = extra
        return SimpleNamespace(AsyncOpenAI=_AsyncOpenAISpy)

    monkeypatch.setattr(openrouter_module, "require_module", _fake_require)
    adapter = OpenRouterAdapter(model="m", api_key="k")
    adapter._new_client()
    assert seen == {"name": "openai", "extra": "openrouter"}
