"""Ollama LLM adapter — local inference via ollama-python."""

from __future__ import annotations

from typing import Any

from ennoia.adapters.llm.base import LLMAdapter, parse_json_object
from ennoia.utils.imports import require_module

__all__ = ["OllamaAdapter"]


async def _aclose_ollama_client(client: Any) -> None:
    """Close the httpx transport backing an ``ollama.AsyncClient``.

    Unlike :class:`openai.AsyncOpenAI` / :class:`anthropic.AsyncAnthropic`,
    ``ollama.AsyncClient`` does not expose ``__aenter__`` / ``close`` —
    the httpx client is composed as ``client._client`` and must be closed
    directly. Without this the transport leaks and httpx's GC finalizer
    schedules ``aclose()`` on an already-closed event loop (producing
    ``RuntimeError: Event loop is closed`` at process exit).
    """
    inner = getattr(client, "_client", None)
    if inner is not None:
        await inner.aclose()


class OllamaAdapter(LLMAdapter):
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
        try:
            response = await client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                format="json",
            )
        finally:
            await _aclose_ollama_client(client)
        content = response["message"]["content"]
        return parse_json_object(content, "Ollama")

    async def complete_text(self, prompt: str) -> str:
        client = self._new_client()
        try:
            response = await client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
        finally:
            await _aclose_ollama_client(client)
        return str(response["message"]["content"])
