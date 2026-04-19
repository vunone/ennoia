"""OpenAI LLM adapter.

Requires the ``openai`` extra: ``pip install ennoia[openai]``.

Fresh :class:`AsyncOpenAI` per call because httpx transports are event-loop
bound; the same Stage 1 invariant that governs :class:`OllamaAdapter` applies
here. ``api_key`` defaults to ``OPENAI_API_KEY`` via the SDK's own resolution
if unset, so environments that export the var work without passing the key.
"""

from __future__ import annotations

import os
from typing import Any

from ennoia.adapters.llm.base import LLMAdapter, parse_json_object
from ennoia.utils.imports import require_module

__all__ = ["OpenAIAdapter"]


class OpenAIAdapter(LLMAdapter):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url
        self.timeout = timeout

    def _new_client(self) -> Any:
        module = require_module("openai", "openai")
        kwargs: dict[str, Any] = {"timeout": self.timeout}
        if self.api_key is not None:
            kwargs["api_key"] = self.api_key
        if self.base_url is not None:
            kwargs["base_url"] = self.base_url
        return module.AsyncOpenAI(**kwargs)

    async def complete_json(self, prompt: str) -> dict[str, Any]:
        async with self._new_client() as client:
            response = await client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or ""
            return parse_json_object(content, "OpenAI")

    async def complete_text(self, prompt: str) -> str:
        async with self._new_client() as client:
            response = await client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            return str(response.choices[0].message.content or "")
