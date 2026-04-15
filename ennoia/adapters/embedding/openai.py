"""OpenAI embedding adapter.

Requires the ``openai`` extra. Matches the synchronous
:class:`~ennoia.adapters.embedding.protocols.EmbeddingAdapter` protocol —
under the hood we call the sync ``OpenAI`` client because the pipeline's
embedding calls don't need to interleave with other IO (they run once per
indexed document or once per query).

``api_key`` falls back to ``OPENAI_API_KEY`` via the SDK's own resolution.
"""

from __future__ import annotations

import os
from typing import Any

from ennoia.utils.imports import require_module

__all__ = ["OpenAIEmbedding"]


class OpenAIEmbedding:
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
        return module.OpenAI(**kwargs)

    def _embed(self, text: str) -> list[float]:
        client = self._new_client()
        response = client.embeddings.create(model=self.model, input=text)
        return [float(x) for x in response.data[0].embedding]

    def embed_document(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
