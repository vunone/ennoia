"""Qdrant-backed pure vector store.

Uses a single unnamed vector slot per point. Point ids are Ennoia's canonical
``{source_id}:{index}`` keys; the canonical ``source_id`` and ``index`` are
also mirrored into the payload so the restrict-list / index-filter branches
in :class:`~ennoia.index.pipeline.Pipeline.asearch` work against the metadata
that comes back from ``search``.

For an integrated structured-filter + vector-search backend prefer
:class:`ennoia.store.hybrid.qdrant.QdrantHybridStore`.

Requires the ``qdrant`` extra (``qdrant-client``).
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from ennoia.store.base import VectorStore
from ennoia.utils.imports import require_module

if TYPE_CHECKING:  # pragma: no cover
    from qdrant_client import AsyncQdrantClient

__all__ = ["QdrantVectorStore"]


class QdrantVectorStore(VectorStore):
    def __init__(
        self,
        collection: str,
        *,
        url: str | None = None,
        host: str | None = None,
        port: int | None = None,
        api_key: str | None = None,
        vector_size: int | None = None,
        distance: str = "Cosine",
        client: AsyncQdrantClient | None = None,
    ) -> None:
        self.collection = collection
        self._vector_size = vector_size
        self._distance = distance
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
        self._ensured = False

    async def _get_client(self) -> AsyncQdrantClient:
        client = self._client
        if client is None:
            qdrant = require_module("qdrant_client", "qdrant")
            client = qdrant.AsyncQdrantClient(**self._client_ctor_kwargs)
            self._client = client
        return client

    async def _ensure_collection(self, vector_size: int) -> None:
        if self._ensured:
            return
        models = require_module("qdrant_client.models", "qdrant")
        client = await self._get_client()
        existing = await client.collection_exists(self.collection)
        if not existing:
            await client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=getattr(models.Distance, self._distance.upper()),
                ),
            )
        self._ensured = True

    async def upsert(
        self,
        vector_id: str,
        vector: list[float],
        metadata: dict[str, Any],
    ) -> None:
        await self._ensure_collection(len(vector))
        models = require_module("qdrant_client.models", "qdrant")
        client = await self._get_client()
        payload = {**metadata, "_ennoia_vector_id": vector_id}
        await client.upsert(
            collection_name=self.collection,
            points=[
                models.PointStruct(
                    id=_stable_point_id(vector_id),
                    vector=list(vector),
                    payload=payload,
                )
            ],
        )

    async def search(
        self,
        query_vector: list[float],
        top_k: int,
        restrict_to: list[str] | None = None,
        *,
        index: str | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        models = require_module("qdrant_client.models", "qdrant")
        client = await self._get_client()
        must: list[Any] = []
        if restrict_to is not None:
            must.append(
                models.FieldCondition(key="source_id", match=models.MatchAny(any=list(restrict_to)))
            )
        if index is not None:
            must.append(models.FieldCondition(key="index", match=models.MatchValue(value=index)))
        query_filter = models.Filter(must=must) if must else None
        hits = await client.query_points(
            collection_name=self.collection,
            query=list(query_vector),
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        return [_unwrap_hit(point) for point in hits.points]

    async def delete_by_source(self, source_id: str) -> int:
        models = require_module("qdrant_client.models", "qdrant")
        client = await self._get_client()
        selector = models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(key="source_id", match=models.MatchValue(value=source_id))
                ]
            )
        )
        # Qdrant's delete API returns an ``UpdateResult``; we report the
        # match-count by counting via a scroll first so the caller can act on
        # "nothing was removed" the same way composite stores do.
        scroll, _ = await client.scroll(
            collection_name=self.collection,
            scroll_filter=selector.filter,
            with_payload=False,
            with_vectors=False,
            limit=10_000,
        )
        removed = len(scroll)
        if removed:
            await client.delete(collection_name=self.collection, points_selector=selector)
        return removed


def _stable_point_id(vector_id: str) -> str:
    """Return a deterministic UUIDv5 string so repeated upserts overwrite."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, vector_id))


def _unwrap_hit(point: Any) -> tuple[str, float, dict[str, Any]]:
    payload = dict(point.payload or {})
    vector_id = str(payload.pop("_ennoia_vector_id", point.id))
    score = float(point.score)
    return vector_id, score, payload
