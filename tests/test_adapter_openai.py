"""OpenAIAdapter unit tests — mock the SDK client via _new_client."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

pytest.importorskip("openai")

from ennoia.adapters.llm.openai import OpenAIAdapter
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


async def test_complete_json_round_trips(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = OpenAIAdapter(model="gpt-x", api_key="sk-test")
    client = _FakeClient(json.dumps({"category": "legal"}))
    monkeypatch.setattr(adapter, "_new_client", lambda: client)
    assert await adapter.complete_json("prompt") == {"category": "legal"}
    call = client.chat.completions.calls[0]
    assert call["response_format"] == {"type": "json_object"}
    assert call["model"] == "gpt-x"


async def test_complete_json_raises_on_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = OpenAIAdapter(model="gpt-x", api_key="sk-test")
    monkeypatch.setattr(adapter, "_new_client", lambda: _FakeClient("not json"))
    with pytest.raises(ExtractionError):
        await adapter.complete_json("prompt")


async def test_complete_text_returns_plain_string(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = OpenAIAdapter(model="gpt-x", api_key="sk-test")
    monkeypatch.setattr(adapter, "_new_client", lambda: _FakeClient("hello"))
    assert await adapter.complete_text("prompt") == "hello"


def test_api_key_falls_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    adapter = OpenAIAdapter(model="gpt-x")
    assert adapter.api_key == "from-env"
