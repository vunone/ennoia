"""sentence-transformers embedding adapter."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from ennoia.adapters.embedding.base import EmbeddingAdapter
from ennoia.utils.imports import require_module

__all__ = ["SentenceTransformerEmbedding"]


def _silence_model_loading() -> None:
    """Suppress HF Hub / transformers load-time noise.

    Runs before the first import of ``sentence_transformers``. Four
    sources to silence: transformers' INFO/WARNING logs (including the
    BertModel LOAD REPORT), huggingface_hub's unauthenticated-request
    hint, huggingface_hub's download progress bars, and the tokenizers
    parallelism warning. ``setdefault`` so the caller can still override
    any of these by exporting the env var upstream.
    """
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for logger_name in ("transformers", "huggingface_hub", "sentence_transformers"):
        logging.getLogger(logger_name).setLevel(logging.ERROR)


class SentenceTransformerEmbedding(EmbeddingAdapter):
    """Embedding adapter backed by a sentence-transformers model.

    Requires the `sentence-transformers` extra:
    `pip install ennoia[sentence-transformers]`.

    ``encode`` is CPU-bound (and releases the GIL inside torch); both
    ``embed`` and ``embed_batch`` dispatch it via :func:`asyncio.to_thread`
    so other I/O can progress while the model runs.
    """

    def __init__(self, model: str = "all-MiniLM-L6-v2", device: str | None = None) -> None:
        self.model_name = model
        self.device = device
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            _silence_model_loading()
            module = require_module("sentence_transformers", "sentence-transformers")
            self._model = module.SentenceTransformer(self.model_name, device=self.device)
        return self._model

    async def embed(self, text: str) -> list[float]:
        model = self._get_model()
        vector = await asyncio.to_thread(
            model.encode, text, convert_to_numpy=True, normalize_embeddings=False
        )
        return [float(x) for x in vector.tolist()]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model()
        # ``encode`` accepts a list and returns a 2-D ndarray — one threadpool
        # hop and one model invocation regardless of batch size.
        matrix = await asyncio.to_thread(
            model.encode, texts, convert_to_numpy=True, normalize_embeddings=False
        )
        return [[float(x) for x in row] for row in matrix.tolist()]
