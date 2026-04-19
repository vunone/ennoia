"""Anthropic Claude LLM adapter.

Requires the ``anthropic`` extra: ``pip install ennoia[anthropic]``.

Structured output uses forced tool-use: the model is given a single tool
named ``emit_extraction`` whose input schema is a permissive JSON object, and
``tool_choice={"type": "tool", "name": ...}`` forces the call. Parsing the
tool input returns the extraction payload. This is Anthropic's documented
way to guarantee machine-readable output.

Fresh :class:`AsyncAnthropic` per call (same event-loop-binding reason as the
Ollama + OpenAI adapters).
"""

from __future__ import annotations

import os
from typing import Any, cast

from ennoia.adapters.llm.base import LLMAdapter
from ennoia.index.exceptions import ExtractionError
from ennoia.utils.imports import require_module

__all__ = ["AnthropicAdapter"]

_EMIT_TOOL = {
    "name": "emit_extraction",
    "description": "Return the extracted JSON object matching the schema in the prompt.",
    "input_schema": {"type": "object"},
}


class AnthropicAdapter(LLMAdapter):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        max_tokens: int = 4096,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.max_tokens = max_tokens
        self.timeout = timeout

    def _new_client(self) -> Any:
        module = require_module("anthropic", "anthropic")
        kwargs: dict[str, Any] = {"timeout": self.timeout}
        if self.api_key is not None:
            kwargs["api_key"] = self.api_key
        return module.AsyncAnthropic(**kwargs)

    async def complete_json(self, prompt: str) -> dict[str, Any]:
        async with self._new_client() as client:
            response = await client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
                tools=[_EMIT_TOOL],
                tool_choice={"type": "tool", "name": _EMIT_TOOL["name"]},
            )
        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                input_value = getattr(block, "input", None)
                if isinstance(input_value, dict):
                    return cast(dict[str, Any], input_value)
        raise ExtractionError(
            f"Anthropic response did not include a {_EMIT_TOOL['name']!r} tool_use block."
        )

    async def complete_text(self, prompt: str) -> str:
        async with self._new_client() as client:
            response = await client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
        parts: list[str] = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                parts.append(str(getattr(block, "text", "")))
        return "".join(parts)
