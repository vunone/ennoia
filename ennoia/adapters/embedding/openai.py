"""OpenAI embedding adapter.

Requires the ``openai`` extra. Uses :class:`openai.AsyncOpenAI` so calls
overlap with the rest of the pipeline; ``embed_batch`` is overridden to
issue a single ``embeddings.create(input=[...])`` round-trip when the
pipeline embeds many semantic fields per document.

Fresh client per call: ``httpx`` transports inside the SDK bind to the
running event loop, and the sync ``Pipeline.index`` / ``Pipeline.search``
wrappers create a fresh loop on every call. Caching the client across
``asyncio.run()`` boundaries reintroduces ``RuntimeError: Event loop is
closed`` — the same invariant that governs every async adapter in this
package (see :class:`ennoia.adapters.llm.openai.OpenAIAdapter`).

``api_key`` falls back to ``OPENAI_API_KEY`` via the SDK's own resolution.
"""

from __future__ import annotations

import os
from typing import Any

from ennoia.adapters.embedding.base import EmbeddingAdapter
from ennoia.utils.imports import require_module

__all__ = ["OpenAIEmbedding"]


class OpenAIEmbedding(EmbeddingAdapter):
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

    async def embed(self, text: str) -> list[float]:
        client = self._new_client()
        response = await client.embeddings.create(model=self.model, input=text)
        return [float(x) for x in response.data[0].embedding]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._new_client()
        response = await client.embeddings.create(model=self.model, input=texts)
        return [[float(x) for x in datum.embedding] for datum in response.data]
