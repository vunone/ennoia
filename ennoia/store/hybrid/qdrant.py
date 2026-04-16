"""Qdrant hybrid store — one Qdrant collection holds structured payload + vectors.

Per-point layout::

    id:      UUIDv5 of ``source_id`` (stable across upserts of the same document)
    vector:  dict[str, list[float]] — one named vector per Ennoia semantic index.
             Set at collection-creation time from the first ``upsert`` call.
    payload: the flattened structural record from the superschema, plus
             ``source_id`` mirrored as a payload key for filter matches.

The single-point layout (one per ``source_id``, not one per (source_id, index))
lets ``hybrid_search`` issue one native query for structured filter + vector
similarity simultaneously. Hits carry a virtual per-index vector_id built from
``{source_id}:{index}`` on the way out so the pipeline's ``SearchHit.semantic``
dict is keyed consistently with the composite-store path.

Requires the ``qdrant`` extra.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from ennoia.store.base import HybridStore
from ennoia.store.hybrid._qdrant_filter import translate_filter
from ennoia.utils.filters import apply_filters
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
        self._ensured_named_vectors: tuple[str, ...] | None = None

    async def _get_client(self) -> AsyncQdrantClient:
        client = self._client
        if client is None:
            qdrant = require_module("qdrant_client", "qdrant")
            client = qdrant.AsyncQdrantClient(**self._client_ctor_kwargs)
            self._client = client
        return client

    async def _ensure_collection(self, vectors: dict[str, list[float]]) -> None:
        if self._ensured_named_vectors is not None:
            return
        models = require_module("qdrant_client.models", "qdrant")
        client = await self._get_client()
        existing = await client.collection_exists(self.collection)
        if not existing:
            vectors_config = {
                name: models.VectorParams(
                    size=len(vec),
                    distance=getattr(models.Distance, self._distance.upper()),
                )
                for name, vec in vectors.items()
            }
            await client.create_collection(
                collection_name=self.collection,
                vectors_config=vectors_config,
            )
        self._ensured_named_vectors = tuple(sorted(vectors.keys()))

    async def upsert(
        self,
        source_id: str,
        data: dict[str, Any],
        vectors: dict[str, list[float]],
    ) -> None:
        if not vectors:
            # Qdrant requires a vector on each point; when there's no semantic
            # content we can only persist the structured half by reading the
            # named-vector spec from the collection and padding with zeros.
            await self._upsert_structured_only(source_id, data)
            return

        await self._ensure_collection(vectors)
        models = require_module("qdrant_client.models", "qdrant")
        client = await self._get_client()
        payload = {**data, "source_id": source_id}
        await client.upsert(
            collection_name=self.collection,
            points=[
                models.PointStruct(
                    id=_point_id(source_id),
                    vector={name: list(vec) for name, vec in vectors.items()},
                    payload=payload,
                )
            ],
        )

    async def _upsert_structured_only(self, source_id: str, data: dict[str, Any]) -> None:
        # Only reachable once the collection exists — ``_ensure_collection``
        # needs at least one vector sample to derive dimensions.
        if self._ensured_named_vectors is None:
            raise RuntimeError(
                "QdrantHybridStore cannot upsert a document with no semantic vectors "
                "before the collection's named-vector spec has been established. "
                "Either provide at least one vector on the first upsert, or call "
                "``set_payload`` directly."
            )
        client = await self._get_client()
        await client.set_payload(
            collection_name=self.collection,
            payload={**data, "source_id": source_id},
            points=[_point_id(source_id)],
        )

    async def hybrid_search(
        self,
        filters: dict[str, Any],
        query_vector: list[float],
        top_k: int,
        *,
        index: str | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        client = await self._get_client()
        qfilter, residual = translate_filter(filters, list_fields=self._list_fields)
        using = index or (self._ensured_named_vectors[0] if self._ensured_named_vectors else None)
        hits = await client.query_points(
            collection_name=self.collection,
            query=list(query_vector),
            using=using,
            query_filter=qfilter,
            limit=top_k * 4 if residual else top_k,
            with_payload=True,
        )
        rows = [_unwrap_hit(point, using) for point in hits.points]
        if residual:
            # Post-filter the hit payloads in Python — correctness stays tied
            # to the canonical :func:`apply_filters` evaluator.
            records = [(vid, meta) for vid, _, meta in rows]
            allowed = set(apply_filters(records, residual))
            rows = [r for r in rows if r[0] in allowed][:top_k]
        return rows[:top_k]

    async def get(self, source_id: str) -> dict[str, Any] | None:
        client = await self._get_client()
        records = await client.retrieve(
            collection_name=self.collection,
            ids=[_point_id(source_id)],
            with_payload=True,
            with_vectors=False,
        )
        if not records:
            return None
        payload = dict(records[0].payload or {})
        payload.pop("source_id", None)
        return payload

    async def filter(self, filters: dict[str, Any]) -> list[str]:
        client = await self._get_client()
        qfilter, residual = translate_filter(filters, list_fields=self._list_fields)
        ids: list[str] = []
        next_offset: Any = None
        while True:
            batch, next_offset = await client.scroll(
                collection_name=self.collection,
                scroll_filter=qfilter,
                limit=512,
                offset=next_offset,
                with_payload=bool(residual),
                with_vectors=False,
            )
            if residual:
                records = [
                    (str(r.payload["source_id"]), dict(r.payload)) for r in batch if r.payload
                ]
                ids.extend(apply_filters(records, residual))
            else:
                ids.extend(
                    str(r.payload["source_id"])
                    for r in batch
                    if r.payload and "source_id" in r.payload
                )
            if next_offset is None:  # pragma: no branch — single-batch fake never paginates
                break
        return ids

    async def delete(self, source_id: str) -> bool:
        models = require_module("qdrant_client.models", "qdrant")
        client = await self._get_client()
        existing = await client.retrieve(
            collection_name=self.collection,
            ids=[_point_id(source_id)],
            with_payload=False,
            with_vectors=False,
        )
        if not existing:
            return False
        await client.delete(
            collection_name=self.collection,
            points_selector=models.PointIdsList(points=[_point_id(source_id)]),
        )
        return True


def _point_id(source_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, source_id))


def _unwrap_hit(point: Any, using: str | None) -> tuple[str, float, dict[str, Any]]:
    payload = dict(point.payload or {})
    source_id = str(payload.get("source_id", point.id))
    virtual_vector_id = f"{source_id}:{using}" if using else source_id
    # Hydrate the metadata the pipeline expects: ``source_id`` + ``index``.
    # ``text`` is not persisted in the hybrid payload (structural only), so the
    # SearchHit's ``semantic`` dict will be empty for hybrid hits — callers
    # interested in the actual text can retrieve() or use a separate vector store.
    payload_out: dict[str, Any] = dict(payload)
    payload_out.setdefault("source_id", source_id)
    if using is not None:  # pragma: no branch — hybrid_search always resolves a using name
        payload_out["index"] = using
    return virtual_vector_id, float(point.score), payload_out
