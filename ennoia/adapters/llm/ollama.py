"""Ollama LLM adapter — local inference via ollama-python."""

from __future__ import annotations

import json
from typing import Any, cast

from ennoia.index.exceptions import ExtractionError
from ennoia.utils.imports import require_module

__all__ = ["OllamaAdapter"]


class OllamaAdapter:
    """Adapter for a locally running Ollama instance.

    Requires the `ollama` extra: `pip install ennoia[ollama]`.
    """

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.host = host
        self.timeout = timeout

    def _new_client(self) -> Any:
        # A fresh AsyncClient is instantiated per call because httpx's
        # transport binds to the running event loop; reusing one across
        # separate asyncio.run() invocations raises "Event loop is closed".
        module = require_module("ollama", "ollama")
        return module.AsyncClient(host=self.host, timeout=self.timeout)

    async def complete_json(self, prompt: str) -> dict[str, Any]:
        client = self._new_client()
        response = await client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            format="json",
        )
        content = response["message"]["content"]
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as err:
            raise ExtractionError(
                f"Ollama returned non-JSON content despite format=json: {content!r}"
            ) from err
        if not isinstance(parsed, dict):
            raise ExtractionError(
                f"Expected a JSON object from Ollama, got {type(parsed).__name__}."
            )
        # json.loads returns dict[Any, Any]; our contract is dict[str, Any].
        return cast(dict[str, Any], parsed)

    async def complete_text(self, prompt: str) -> str:
        client = self._new_client()
        response = await client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return str(response["message"]["content"])
