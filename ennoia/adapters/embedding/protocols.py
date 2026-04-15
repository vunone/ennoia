"""EmbeddingAdapter protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = ["EmbeddingAdapter"]


@runtime_checkable
class EmbeddingAdapter(Protocol):
    def embed_document(self, text: str) -> list[float]: ...

    def embed_query(self, text: str) -> list[float]: ...
