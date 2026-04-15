"""OpenAI LLM adapter.

Requires the ``openai`` extra: ``pip install ennoia[openai]``.

Fresh :class:`AsyncOpenAI` per call because httpx transports are event-loop
bound; the same Stage 1 invariant that governs :class:`OllamaAdapter` applies
here. ``api_key`` defaults to ``OPENAI_API_KEY`` via the SDK's own resolution
if unset, so environments that export the var work without passing the key.
"""

from __future__ import annotations

import json
import os
from typing import Any, cast

from ennoia.index.exceptions import ExtractionError
from ennoia.utils.imports import require_module

__all__ = ["OpenAIAdapter"]


class OpenAIAdapter:
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
        client = self._new_client()
        response = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or ""
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as err:
            raise ExtractionError(
                f"OpenAI returned non-JSON content despite response_format=json_object: {content!r}"
            ) from err
        if not isinstance(parsed, dict):
            raise ExtractionError(
                f"Expected a JSON object from OpenAI, got {type(parsed).__name__}."
            )
        return cast(dict[str, Any], parsed)

    async def complete_text(self, prompt: str) -> str:
        client = self._new_client()
        response = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return str(response.choices[0].message.content or "")
