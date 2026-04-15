"""Filesystem-backed vector store — NumPy arrays + JSON sidecars.

Persists three files under ``<path>/``:

- ``vectors.npy``: 2D float array, one row per vector, written with :func:`numpy.save`.
- ``ids.json``: ordered list of vector ids aligned with the rows of ``vectors.npy``.
- ``metadata.json``: mapping vector-id → metadata dict.

Upserts rewrite all three files — the same read-modify-write idiom as
:class:`ennoia.store.structured.parquet.ParquetStructuredStore`. Search
delegates to the shared :func:`ennoia.store.vector._numpy.cosine_search`.

Requires the ``filesystem`` or ``sentence-transformers`` extra (either pulls
in ``numpy``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ennoia.store.base import VectorStore
from ennoia.store.vector._numpy import cosine_search
from ennoia.utils.imports import require_module

__all__ = ["FilesystemVectorStore"]


class FilesystemVectorStore(VectorStore):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._vectors_file = self.path / "vectors.npy"
        self._ids_file = self.path / "ids.json"
        self._metadata_file = self.path / "metadata.json"
        self._entries: dict[str, tuple[list[float], dict[str, Any]]] = {}
        self._load()

    def _load(self) -> None:
        if not (self._vectors_file.exists() and self._ids_file.exists()):
            return
        np = require_module("numpy", "sentence-transformers")
        ids = json.loads(self._ids_file.read_text())
        vectors = np.load(self._vectors_file).tolist()
        metadata_all: dict[str, dict[str, Any]] = (
            json.loads(self._metadata_file.read_text()) if self._metadata_file.exists() else {}
        )
        for vid, vec in zip(ids, vectors, strict=False):
            self._entries[vid] = (list(vec), dict(metadata_all.get(vid, {})))

    def _flush(self) -> None:
        np = require_module("numpy", "sentence-transformers")
        ids = list(self._entries.keys())
        vectors = [vec for vec, _ in self._entries.values()]
        metadata = {vid: meta for vid, (_, meta) in self._entries.items()}
        if ids:
            np.save(self._vectors_file, np.asarray(vectors, dtype=float))
        else:
            if self._vectors_file.exists():
                self._vectors_file.unlink()
        self._ids_file.write_text(json.dumps(ids))
        self._metadata_file.write_text(json.dumps(metadata, default=str))

    def upsert(
        self,
        vector_id: str,
        vector: list[float],
        metadata: dict[str, Any],
    ) -> None:
        self._entries[vector_id] = (list(vector), dict(metadata))
        self._flush()

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        restrict_to: list[str] | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        return cosine_search(self._entries, query_vector, top_k, restrict_to)
