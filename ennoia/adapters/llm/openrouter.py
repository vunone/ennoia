"""OpenRouter LLM adapter.

Requires the ``openrouter`` extra (an alias for ``openai``): ``pip install
ennoia[openrouter]``. OpenRouter exposes an OpenAI-compatible REST surface,
so the official ``openai`` Python SDK is the canonical client — pointed at
``https://openrouter.ai/api/v1``.

Fresh :class:`AsyncOpenAI` per call because httpx transports are event-loop
bound; the same Stage 1 invariant that governs :class:`OpenAIAdapter` and
:class:`OllamaAdapter` applies here. ``api_key`` defaults to
``OPENROUTER_API_KEY`` so environments that export the var work without
passing the key.
"""

from __future__ import annotations

import os
from typing import Any

from ennoia.adapters.llm.base import LLMAdapter, parse_json_object
from ennoia.utils.imports import require_module

__all__ = ["OpenRouterAdapter"]

_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterAdapter(LLMAdapter):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.base_url = base_url or _DEFAULT_BASE_URL
        self.timeout = timeout

    def _new_client(self) -> Any:
        module = require_module("openai", "openrouter")
        kwargs: dict[str, Any] = {"timeout": self.timeout, "base_url": self.base_url}
        if self.api_key is not None:
            kwargs["api_key"] = self.api_key
        return module.AsyncOpenAI(**kwargs)

    async def complete_json(self, prompt: str) -> dict[str, Any]:
        client = self._new_client()
        response = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or ""
        return parse_json_object(content, "OpenRouter")

    async def complete_text(self, prompt: str) -> str:
        client = self._new_client()
        response = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return str(response.choices[0].message.content or "")
