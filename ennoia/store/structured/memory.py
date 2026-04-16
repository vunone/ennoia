"""In-memory structured store — dict-of-dicts backed."""

from __future__ import annotations

from typing import Any

from ennoia.store.base import StructuredStore, validate_collection_name
from ennoia.utils.filters import apply_filters

__all__ = ["InMemoryStructuredStore"]


class InMemoryStructuredStore(StructuredStore):
    def __init__(self, *, collection: str = "documents") -> None:
        self.collection = validate_collection_name(collection)
        self._records: dict[str, dict[str, Any]] = {}

    async def upsert(self, source_id: str, data: dict[str, Any]) -> None:
        self._records[source_id] = dict(data)

    async def filter(self, query: dict[str, Any]) -> list[str]:
        return apply_filters(self._records.items(), query)

    async def get(self, source_id: str) -> dict[str, Any] | None:
        record = self._records.get(source_id)
        return dict(record) if record is not None else None

    async def delete(self, source_id: str) -> bool:
        return self._records.pop(source_id, None) is not None
