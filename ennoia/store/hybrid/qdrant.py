"""Qdrant hybrid store — one Qdrant collection holds structured payload + vectors.

Per-point layout::

    id:      UUIDv5 of ``{source_id}:{index_name}:{unique}`` (stable across re-indexes)
    vector:  a single unnamed vector per point — one point per
             :class:`~ennoia.store.base.VectorEntry` (``BaseSemantic`` answer or
             ``BaseCollection`` entity), not one point per document.
    payload: the full structural ``data`` (denormalized across the document's
             points) plus ``source_id``, ``index_name``, ``unique``, ``text``.

A document with a single ``BaseSemantic`` yields one point; a
``BaseCollection`` with N entities yields N points sharing the same ``data``.
``hybrid_search`` issues a single native query; the pipeline collapses
multi-point hits to one :class:`~ennoia.index.result.SearchHit` per
``source_id`` at the boundary.

Requires the ``qdrant`` extra.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from ennoia.store.base import HybridStore, VectorEntry
from ennoia.store.hybrid._qdrant_filter import translate_filter
from ennoia.utils.filters import apply_filters
from ennoia.utils.ids import make_semantic_vector_id
from ennoia.utils.imports import require_module

if TYPE_CHECKING:  # pragma: no cover
    from qdrant_client import AsyncQdrantClient

__all__ = ["QdrantHybridStore"]


class QdrantHybridStore(HybridStore):
    def __init__(
        self,
        collection: str,
        *,
        url: str | None = None,
        host: str | None = None,
        port: int | None = None,
        api_key: str | None = None,
        distance: str = "Cosine",
        list_payload_fields: frozenset[str] = frozenset(),
        client: AsyncQdrantClient | None = None,
    ) -> None:
        self.collection = collection
        self._distance = distance
        self._list_fields = list_payload_fields
        self._client: AsyncQdrantClient | None = client
        self._client_ctor_kwargs: dict[str, Any] = {}
        if client is None:
            if url is not None:
                self._client_ctor_kwargs["url"] = url
            if host is not None:
                self._client_ctor_kwargs["host"] = host
            if port is not None:
                self._client_ctor_kwargs["port"] = port
            if api_key is not None:
                self._client_ctor_kwargs["api_key"] = api_key
        self._collection_ensured = False

    async def _get_client(self) -> AsyncQdrantClient:
        client = self._client
        if client is None:
            qdrant = require_module("qdrant_client", "qdrant")
            client = qdrant.AsyncQdrantClient(**self._client_ctor_kwargs)
            self._client = client
        return client

    async def _ensure_collection(self, sample_dim: int) -> None:
        if self._collection_ensured:
            return
        models = require_module("qdrant_client.models", "qdrant")
        client = await self._get_client()
        existing = await client.collection_exists(self.collection)
        if not existing:
            await client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=sample_dim,
                    distance=getattr(models.Distance, self._distance.upper()),
                ),
            )
        self._collection_ensured = True

    async def upsert(
        self,
        source_id: str,
        data: dict[str, Any],
        entries: list[VectorEntry],
    ) -> None:
        models = require_module("qdrant_client.models", "qdrant")
        client = await self._get_client()

        # Remove any prior points for this document so N → M cardinality
        # changes (e.g., a collection shrinking on re-index) are correct.
        if self._collection_ensured:
            await client.delete(
                collection_name=self.collection,
                points_selector=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source_id",
                            match=models.MatchValue(value=source_id),
                        )
                    ]
                ),
            )

        if not entries:
            return

        await self._ensure_collection(len(entries[0].vector))
        points: list[Any] = []
        for entry in entries:
            payload: dict[str, Any] = {
                **data,
                "source_id": source_id,
                "index_name": entry.index_name,
                "text": entry.text,
            }
            if entry.unique is not None:
                payload["unique"] = entry.unique
            points.append(
                models.PointStruct(
                    id=_point_id(source_id, entry.index_name, entry.unique),
                    vector=list(entry.vector),
                    payload=payload,
                )
            )
        await client.upsert(collection_name=self.collection, points=points)

    async def hybrid_search(
        self,
        filters: dict[str, Any],
        query_vector: list[float],
        top_k: int,
        *,
        index: str | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        if not self._collection_ensured:
            return []

        models = require_module("qdrant_client.models", "qdrant")
        client = await self._get_client()
        qfilter, residual = translate_filter(filters, list_fields=self._list_fields)

        if index is not None:
            index_cond = models.FieldCondition(
                key="index_name",
                match=models.MatchValue(value=index),
            )
            if qfilter is None:
                qfilter = models.Filter(must=[index_cond])
            else:
                existing_must = list(qfilter.must or [])
                qfilter = models.Filter(
                    must=[*existing_must, index_cond],
                    must_not=qfilter.must_not,
                    should=qfilter.should,
                )

        hits = await client.query_points(
            collection_name=self.collection,
            query=list(query_vector),
            query_filter=qfilter,
            limit=top_k * 4 if residual else top_k,
            with_payload=True,
        )
        rows = [_unwrap_hit(point) for point in hits.points]
        if residual:
            records = [(vid, meta) for vid, _, meta in rows]
            allowed = set(apply_filters(records, residual))
            rows = [r for r in rows if r[0] in allowed][:top_k]
        return rows[:top_k]

    async def get(self, source_id: str) -> dict[str, Any] | None:
        if not self._collection_ensured:
            return None
        models = require_module("qdrant_client.models", "qdrant")
        client = await self._get_client()
        batch, _ = await client.scroll(
            collection_name=self.collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source_id",
                        match=models.MatchValue(value=source_id),
                    )
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if not batch:
            return None
        payload = dict(batch[0].payload or {})
        # Strip row-level metadata so the return shape is the pure structural record.
        for key in ("source_id", "index_name", "unique", "text"):
            payload.pop(key, None)
        return payload

    async def filter(self, filters: dict[str, Any]) -> list[str]:
        if not self._collection_ensured:
            return []
        client = await self._get_client()
        qfilter, residual = translate_filter(filters, list_fields=self._list_fields)
        seen: list[str] = []
        seen_set: set[str] = set()
        next_offset: Any = None
        while True:
            batch, next_offset = await client.scroll(
                collection_name=self.collection,
                scroll_filter=qfilter,
                limit=512,
                offset=next_offset,
                with_payload=True,
                with_vectors=False,
            )
            if residual:
                records = [
                    (str(r.payload["source_id"]), dict(r.payload)) for r in batch if r.payload
                ]
                for sid in apply_filters(records, residual):
                    if sid not in seen_set:
                        seen.append(sid)
                        seen_set.add(sid)
            else:
                for r in batch:
                    if r.payload and "source_id" in r.payload:  # pragma: no branch
                        # All rows written by the adapter include source_id, so the
                        # false arm is unreachable in practice — guard kept for
                        # defence against foreign writers.
                        sid = str(r.payload["source_id"])
                        if sid not in seen_set:
                            seen.append(sid)
                            seen_set.add(sid)
            if next_offset is None:  # pragma: no branch — single-batch fake never paginates
                break
        return seen

    async def delete(self, source_id: str) -> bool:
        if not self._collection_ensured:
            return False
        models = require_module("qdrant_client.models", "qdrant")
        client = await self._get_client()
        # Check for existence first so we can return the boolean contract honestly.
        batch, _ = await client.scroll(
            collection_name=self.collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source_id",
                        match=models.MatchValue(value=source_id),
                    )
                ]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        if not batch:
            return False
        await client.delete(
            collection_name=self.collection,
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source_id",
                        match=models.MatchValue(value=source_id),
                    )
                ]
            ),
        )
        return True


def _point_id(source_id: str, index_name: str, unique: str | None) -> str:
    """Stable UUIDv5 derived from the row's identity triple."""
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            make_semantic_vector_id(source_id, index_name, unique),
        )
    )


def _unwrap_hit(point: Any) -> tuple[str, float, dict[str, Any]]:
    payload = dict(point.payload or {})
    source_id = str(payload.get("source_id", point.id))
    index_name = str(payload.get("index_name", ""))
    unique = payload.get("unique")
    virtual_vector_id = make_semantic_vector_id(
        source_id,
        index_name,
        unique if isinstance(unique, str) else None,
    )
    payload_out: dict[str, Any] = dict(payload)
    payload_out["source_id"] = source_id
    payload_out["index"] = index_name
    payload_out["text"] = payload.get("text", "")
    return virtual_vector_id, float(point.score), payload_out
