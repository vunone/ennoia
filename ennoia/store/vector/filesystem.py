"""Filesystem-backed vector store — NumPy arrays + JSON sidecars.

Persists three files under ``<path>/``, named after the store's
``collection`` (default ``"documents"``):

- ``{collection}.npy``: 2D float array, one row per vector, written with
  :func:`numpy.save`.
- ``{collection}_ids.json``: ordered list of vector ids aligned with the
  rows of the ``.npy`` array.
- ``{collection}_metadata.json``: mapping vector-id → metadata dict.

Multiple collections can share a directory by instantiating with different
``collection=`` names — each writes to its own prefixed set of files.

Upserts rewrite all three files — the same read-modify-write idiom as
:class:`ennoia.store.structured.parquet.ParquetStructuredStore`. Search
delegates to the shared :func:`ennoia.store.vector._numpy.cosine_search`.

``numpy`` and the standard ``Path.read_text`` / ``Path.write_text`` calls
are sync; the upsert path dispatches the disk I/O via
:func:`asyncio.to_thread` so the event loop stays responsive.

Requires the ``filesystem`` or ``sentence-transformers`` extra (either pulls
in ``numpy``).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from ennoia.store.base import VectorStore, validate_collection_name
from ennoia.store.vector._numpy import cosine_search
from ennoia.utils.imports import require_module

__all__ = ["FilesystemVectorStore"]


class FilesystemVectorStore(VectorStore):
    def __init__(self, path: str | Path, *, collection: str = "documents") -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.collection = validate_collection_name(collection)
        self._vectors_file = self.path / f"{self.collection}.npy"
        self._ids_file = self.path / f"{self.collection}_ids.json"
        self._metadata_file = self.path / f"{self.collection}_metadata.json"
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

    async def upsert(
        self,
        vector_id: str,
        vector: list[float],
        metadata: dict[str, Any],
    ) -> None:
        self._entries[vector_id] = (list(vector), dict(metadata))
        await asyncio.to_thread(self._flush)

    async def search(
        self,
        query_vector: list[float],
        top_k: int,
        restrict_to: list[str] | None = None,
        *,
        index: str | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        return cosine_search(self._entries, query_vector, top_k, restrict_to, index=index)

    async def delete_by_source(self, source_id: str) -> int:
        doomed = [
            vid for vid, (_, meta) in self._entries.items() if meta.get("source_id") == source_id
        ]
        for vid in doomed:
            del self._entries[vid]
        if doomed:
            await asyncio.to_thread(self._flush)
        return len(doomed)
