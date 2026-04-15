"""sentence-transformers embedding adapter."""

from __future__ import annotations

from typing import Any

from ennoia.utils.imports import require_module

__all__ = ["SentenceTransformerEmbedding"]


class SentenceTransformerEmbedding:
    """Embedding adapter backed by a sentence-transformers model.

    Requires the `sentence-transformers` extra:
    `pip install ennoia[sentence-transformers]`.
    """

    def __init__(self, model: str = "all-MiniLM-L6-v2", device: str | None = None) -> None:
        self.model_name = model
        self.device = device
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            module = require_module("sentence_transformers", "sentence-transformers")
            self._model = module.SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def _encode(self, text: str) -> list[float]:
        model = self._get_model()
        vector = model.encode(text, convert_to_numpy=True, normalize_embeddings=False)
        return [float(x) for x in vector.tolist()]

    def embed_document(self, text: str) -> list[float]:
        return self._encode(text)

    def embed_query(self, text: str) -> list[float]:
        return self._encode(text)
