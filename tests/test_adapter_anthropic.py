"""AnthropicAdapter unit tests — mock the SDK client via _new_client."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

import ennoia.adapters.llm.anthropic as anthropic_module
from ennoia.adapters.llm.anthropic import AnthropicAdapter
from ennoia.index.exceptions import ExtractionError


@dataclass
class _ToolUseBlock:
    type: str = "tool_use"
    name: str = "emit_extraction"
    input: dict[str, Any] = field(default_factory=dict)


@dataclass
class _TextBlock:
    type: str = "text"
    text: str = ""


@dataclass
class _Response:
    content: list[Any]


class _Messages:
    def __init__(self, response: _Response) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _Response:
        self.calls.append(kwargs)
        return self._response


class _Client:
    def __init__(self, response: _Response) -> None:
        self.messages = _Messages(response)


async def test_complete_json_uses_tool_use_block(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = AnthropicAdapter(model="claude-x", api_key="k")
    block = _ToolUseBlock(input={"category": "legal", "_confidence": 0.9})
    client = _Client(_Response(content=[_TextBlock(text="prelude"), block]))
    monkeypatch.setattr(adapter, "_new_client", lambda: client)

    result = await adapter.complete_json("prompt")
    assert result == {"category": "legal", "_confidence": 0.9}

    kwargs = client.messages.calls[0]
    assert kwargs["tool_choice"]["name"] == "emit_extraction"
    assert kwargs["tools"][0]["name"] == "emit_extraction"


async def test_complete_json_raises_when_no_tool_block(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = AnthropicAdapter(model="claude-x", api_key="k")
    client = _Client(_Response(content=[_TextBlock(text="no tool")]))
    monkeypatch.setattr(adapter, "_new_client", lambda: client)
    with pytest.raises(ExtractionError):
        await adapter.complete_json("prompt")


async def test_complete_text_concatenates_text_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = AnthropicAdapter(model="claude-x", api_key="k")
    client = _Client(_Response(content=[_TextBlock(text="hello "), _TextBlock(text="world")]))
    monkeypatch.setattr(adapter, "_new_client", lambda: client)
    assert await adapter.complete_text("prompt") == "hello world"


def test_api_key_falls_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    adapter = AnthropicAdapter(model="claude-x")
    assert adapter.api_key == "from-env"


async def test_complete_json_raises_on_non_dict_tool_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A tool_use block whose ``input`` is not a dict must be skipped — the code
    # falls through to the no-tool-block raise.
    adapter = AnthropicAdapter(model="claude-x", api_key="k")
    block = _ToolUseBlock()
    block.input = "not a dict"  # type: ignore[assignment]
    client = _Client(_Response(content=[block]))
    monkeypatch.setattr(adapter, "_new_client", lambda: client)
    with pytest.raises(ExtractionError):
        await adapter.complete_json("prompt")


async def test_complete_text_ignores_non_text_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = AnthropicAdapter(model="claude-x", api_key="k")
    client = _Client(
        _Response(content=[_ToolUseBlock(input={"k": 1}), _TextBlock(text="only-this")])
    )
    monkeypatch.setattr(adapter, "_new_client", lambda: client)
    assert await adapter.complete_text("prompt") == "only-this"


class _AsyncAnthropicSpy:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def _patch_anthropic_module(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = SimpleNamespace(AsyncAnthropic=_AsyncAnthropicSpy)
    monkeypatch.setattr(anthropic_module, "require_module", lambda *_args: fake_module)


def test_new_client_passes_api_key_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_anthropic_module(monkeypatch)
    adapter = AnthropicAdapter(model="claude-x", api_key="k", timeout=30.0)
    client = adapter._new_client()
    assert isinstance(client, _AsyncAnthropicSpy)
    assert client.kwargs == {"api_key": "k", "timeout": 30.0}


def test_new_client_omits_api_key_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_anthropic_module(monkeypatch)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    adapter = AnthropicAdapter(model="claude-x")
    client = adapter._new_client()
    assert client.kwargs == {"timeout": 60.0}
