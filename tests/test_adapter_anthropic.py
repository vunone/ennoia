"""AnthropicAdapter unit tests — mock the SDK client via _new_client."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

pytest.importorskip("anthropic")

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
