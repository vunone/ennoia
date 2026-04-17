"""In-memory fake for ``qdrant_client.AsyncQdrantClient``.

Implements only the call surface the Ennoia adapters use. Kept here, not
under ``ennoia/``, because it's a test double — shipping it in the package
would suggest it's a supported public API.
"""

from __future__ import annotations

from typing import Any


class _Record:
    def __init__(
        self,
        point_id: str,
        payload: dict[str, Any],
        vector: dict[str, list[float]] | list[float],
    ) -> None:
        self.id = point_id
        self.payload = payload
        self.vector = vector
        self.score: float = 0.0


class FakeAsyncQdrantClient:
    def __init__(self) -> None:
        self._collections: dict[str, dict[str, Any]] = {}
        self._points: dict[str, dict[str, _Record]] = {}

    async def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self._collections

    async def create_collection(
        self,
        collection_name: str,
        vectors_config: Any,
    ) -> None:
        self._collections[collection_name] = {"vectors_config": vectors_config}
        self._points[collection_name] = {}

    async def upsert(self, collection_name: str, points: list[Any]) -> None:
        for point in points:
            record = _Record(
                point_id=str(point.id),
                payload=dict(point.payload or {}),
                vector=point.vector,
            )
            self._points[collection_name][str(point.id)] = record

    async def set_payload(
        self,
        collection_name: str,
        payload: dict[str, Any],
        points: list[str],
    ) -> None:
        for pid in points:
            record = self._points[collection_name].get(str(pid))
            if record is not None:
                record.payload.update(payload)

    async def retrieve(
        self,
        collection_name: str,
        ids: list[str],
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> list[_Record]:
        out: list[_Record] = []
        for pid in ids:
            record = self._points[collection_name].get(str(pid))
            if record is not None:
                out.append(record)
        return out

    async def scroll(
        self,
        collection_name: str,
        scroll_filter: Any = None,
        limit: int = 512,
        offset: Any = None,
        with_payload: bool = False,
        with_vectors: bool = False,
    ) -> tuple[list[_Record], Any]:
        records = list(self._points[collection_name].values())
        if scroll_filter is not None:
            records = [r for r in records if _matches_filter(r.payload, scroll_filter)]
        return records, None

    async def query_points(
        self,
        collection_name: str,
        query: list[float],
        using: str | None = None,
        query_filter: Any = None,
        limit: int = 10,
        with_payload: bool = True,
    ) -> Any:
        candidates = list(self._points[collection_name].values())
        if query_filter is not None:
            candidates = [r for r in candidates if _matches_filter(r.payload, query_filter)]
        # Deterministic scoring: dot-product of the query against the chosen
        # named vector (or unnamed).
        scored: list[_Record] = []
        for record in candidates:
            if isinstance(record.vector, dict):
                vec = record.vector.get(using) if using else next(iter(record.vector.values()))
            else:
                vec = record.vector
            if vec is None:
                continue
            record.score = float(sum(a * b for a, b in zip(query, vec, strict=False)))
            scored.append(record)
        scored.sort(key=lambda r: r.score, reverse=True)
        return _QueryResponse(points=scored[:limit])

    async def delete(
        self,
        collection_name: str,
        points_selector: Any,
    ) -> None:
        store = self._points[collection_name]
        selector = points_selector
        if hasattr(selector, "points"):  # PointIdsList
            for pid in selector.points:
                store.pop(str(pid), None)
        elif hasattr(selector, "filter"):  # FilterSelector
            doomed = [
                pid for pid, rec in store.items() if _matches_filter(rec.payload, selector.filter)
            ]
            for pid in doomed:
                store.pop(pid, None)
        elif hasattr(selector, "must") or hasattr(selector, "must_not"):  # raw Filter
            doomed = [pid for pid, rec in store.items() if _matches_filter(rec.payload, selector)]
            for pid in doomed:
                store.pop(pid, None)


class _QueryResponse:
    def __init__(self, points: list[_Record]) -> None:
        self.points = points


def _matches_filter(payload: dict[str, Any], qfilter: Any) -> bool:
    """Evaluate a Qdrant ``Filter`` object against a payload dict."""
    must = getattr(qfilter, "must", None) or []
    must_not = getattr(qfilter, "must_not", None) or []
    if any(not _matches_condition(payload, c) for c in must):
        return False
    return not any(_matches_condition(payload, c) for c in must_not)


def _matches_condition(payload: dict[str, Any], condition: Any) -> bool:
    if hasattr(condition, "is_null") and condition.is_null is not None:
        key = condition.is_null.key
        return payload.get(key) is None
    if not hasattr(condition, "key"):
        return True
    key = condition.key
    match = getattr(condition, "match", None)
    range_obj = getattr(condition, "range", None)
    value = payload.get(key)
    if match is not None:
        if hasattr(match, "value"):
            if isinstance(value, list):
                return match.value in value
            return value == match.value
        if hasattr(match, "any"):
            candidates = list(match.any)
            if isinstance(value, list):
                return any(c in value for c in candidates)
            return value in candidates
    if range_obj is not None:
        return _matches_range(value, range_obj)
    return True


def _matches_range(value: Any, range_obj: Any) -> bool:
    if value is None:
        return False
    gt = getattr(range_obj, "gt", None)
    gte = getattr(range_obj, "gte", None)
    lt = getattr(range_obj, "lt", None)
    lte = getattr(range_obj, "lte", None)
    if gt is not None and not (value > gt):
        return False
    if gte is not None and not (value >= gte):
        return False
    if lt is not None and not (value < lt):
        return False
    return not (lte is not None and not (value <= lte))
