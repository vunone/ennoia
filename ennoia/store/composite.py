"""Composite Store — pairs a structured store with a vector store."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ennoia.store.base import StructuredStore, VectorStore

__all__ = ["Store"]


@dataclass
class Store:
    """Bundle a vector and structured store for two-phase retrieval.

    The pipeline reads ``.vector`` and ``.structured`` directly; the query
    planner executes a structured filter first, then vector search restricted
    to the surviving source ids.
    """

    vector: VectorStore
    structured: StructuredStore

    @classmethod
    def from_path(cls, path: str | Path) -> Store:
        """Build a filesystem-backed Store under ``path``.

        Uses :class:`~ennoia.store.structured.parquet.ParquetStructuredStore`
        + :class:`~ennoia.store.vector.filesystem.FilesystemVectorStore`, the
        default drive-by for ``ennoia index --store ./my_index`` per
        ``docs/cli.md``.
        """
        from ennoia.store.structured.parquet import ParquetStructuredStore
        from ennoia.store.vector.filesystem import FilesystemVectorStore

        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        return cls(
            vector=FilesystemVectorStore(root / "vectors"),
            structured=ParquetStructuredStore(root / "structured"),
        )
