"""OpenRouter embedding adapter.

Requires the ``openrouter`` extra (an alias for ``openai``): ``pip install
ennoia[openrouter]``. OpenRouter's embeddings endpoint is OpenAI-compatible,
so the :class:`openai.AsyncOpenAI` client — pointed at
``https://openrouter.ai/api/v1`` — is the canonical client here. ``embed_batch``
is overridden to issue a single ``embeddings.create(input=[...])`` round-trip
when the pipeline embeds many semantic fields per document.

Fresh client per call: ``httpx`` transports inside the SDK bind to the
running event loop, and the sync ``Pipeline.index`` / ``Pipeline.search``
wrappers create a fresh loop on every call. Caching the client across
``asyncio.run()`` boundaries reintroduces ``RuntimeError: Event loop is
closed`` — the same invariant that governs every async adapter in this
package (see :class:`ennoia.adapters.llm.openrouter.OpenRouterAdapter`).

``api_key`` falls back to ``OPENROUTER_API_KEY``.
"""

from __future__ import annotations

import os
from typing import Any

from ennoia.adapters.embedding.base import EmbeddingAdapter
from ennoia.utils.imports import require_module

__all__ = ["OpenRouterEmbedding"]

_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterEmbedding(EmbeddingAdapter):
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

    async def embed(self, text: str) -> list[float]:
        async with self._new_client() as client:
            response = await client.embeddings.create(model=self.model, input=text)
            return [float(x) for x in response.data[0].embedding]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        async with self._new_client() as client:
            response = await client.embeddings.create(model=self.model, input=texts)
            return [[float(x) for x in datum.embedding] for datum in response.data]
