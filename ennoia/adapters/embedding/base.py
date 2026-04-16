"""Abstract base class for embedding adapters.

These ABCs replace the former ``typing.Protocol`` definitions. Inheritance
(rather than structural typing) is used so that concrete adapters have a
single, discoverable contract and so that instantiating a partially
implemented adapter fails loudly with ``TypeError``.

``embed`` is the single abstract hook; ``embed_document`` and ``embed_query``
are concrete async delegators so backends with symmetric doc/query embedding —
the common case — only override one method. ``embed_batch`` is a concrete
delegator too: backends with native list-input APIs (OpenAI's
``embeddings.create(input=[...])``) override it for a single round-trip;
others get a parallel-gather fallback for free.

All methods are async — embedding is I/O (network round-trip for hosted
backends) or CPU work that should be wrapped in :func:`asyncio.to_thread`
for sync libraries (sentence-transformers). Adapters with asymmetric flows
(e.g. Voyage's ``input_type``) simply override ``embed_document`` /
``embed_query`` directly; those methods are not ``@final``.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

__all__ = ["EmbeddingAdapter"]


class EmbeddingAdapter(ABC):
    """Minimal contract every embedding backend implements."""

    @abstractmethod
    async def embed(self, text: str) -> list[float]: ...

    async def embed_document(self, text: str) -> list[float]:
        return await self.embed(text)

    async def embed_query(self, text: str) -> list[float]:
        return await self.embed(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed many texts. Default: gather single-text calls in parallel.

        Backends with a native list-input API (OpenAI) override this to issue
        a single round-trip; sentence-transformers overrides it to forward the
        batch into a single ``encode`` call (vectorised over the full batch).
        """
        return list(await asyncio.gather(*(self.embed(t) for t in texts)))
