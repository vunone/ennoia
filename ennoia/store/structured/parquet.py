"""Parquet-backed structured store.

One ``.parquet`` file under ``<path>/<collection>.parquet``. Every
``upsert`` rewrites the whole file (simple + correct for the dev/ops scale
this store targets). Filtering loads the records into memory and delegates
to :func:`apply_filters` so operator semantics stay identical to every
other backend.

``pyarrow`` / ``pandas`` are sync libraries with no async API; the read /
write paths are wrapped in :func:`asyncio.to_thread` so the event loop
stays responsive while parquet I/O is in flight.

Multiple collections can share a directory by instantiating with different
``collection=`` names — each writes to its own ``{collection}.parquet``.

Requires the ``filesystem`` extra (``pyarrow`` + ``pandas``).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from ennoia.store.base import StructuredStore, validate_collection_name
from ennoia.utils.filters import apply_filters
from ennoia.utils.imports import require_module

__all__ = ["ParquetStructuredStore"]


class ParquetStructuredStore(StructuredStore):
    def __init__(self, path: str | Path, *, collection: str = "documents") -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.collection = validate_collection_name(collection)
        self._file = self.path / f"{self.collection}.parquet"
        self._records: dict[str, dict[str, Any]] = {}
        # Eager load is sync because ``__init__`` cannot be async; the
        # file lives on the local disk and the dataset targeted by this
        # backend is dev/ops scale.
        self._load()

    def _load(self) -> None:
        if not self._file.exists():
            return
        pd = require_module("pandas", "filesystem")
        df = pd.read_parquet(self._file)
        for row in df.to_dict(orient="records"):
            source_id = row.pop("__source_id__")
            raw = row.get("__data__")
            if isinstance(raw, str) and raw:
                self._records[str(source_id)] = dict(json.loads(raw))

    def _flush_sync(self) -> None:
        pd = require_module("pandas", "filesystem")
        rows = [
            {"__source_id__": sid, "__data__": json.dumps(data, default=str)}
            for sid, data in self._records.items()
        ]
        pd.DataFrame(rows).to_parquet(self._file, index=False)

    async def upsert(self, source_id: str, data: dict[str, Any]) -> None:
        self._records[source_id] = dict(data)
        await asyncio.to_thread(self._flush_sync)

    async def get(self, source_id: str) -> dict[str, Any] | None:
        record = self._records.get(source_id)
        return dict(record) if record is not None else None

    async def filter(self, query: dict[str, Any]) -> list[str]:
        return apply_filters(self._records.items(), query)

    async def delete(self, source_id: str) -> bool:
        if self._records.pop(source_id, None) is None:
            return False
        await asyncio.to_thread(self._flush_sync)
        return True
