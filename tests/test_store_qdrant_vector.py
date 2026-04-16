"""QdrantVectorStore unit tests using an in-memory fake Qdrant client."""

from __future__ import annotations

import pytest

from ennoia.store.vector.qdrant import QdrantVectorStore
from tests._qdrant_fake import FakeAsyncQdrantClient


def _store() -> tuple[QdrantVectorStore, FakeAsyncQdrantClient]:
    fake = FakeAsyncQdrantClient()
    store = QdrantVectorStore(collection="test", client=fake)  # type: ignore[arg-type]
    return store, fake


async def test_upsert_creates_collection_and_stores_point() -> None:
    store, fake = _store()
    await store.upsert(
        "doc_1:Summary",
        [1.0, 0.0, 0.0],
        {"source_id": "doc_1", "index": "Summary", "text": "hello"},
    )
    assert await fake.collection_exists("test") is True
    # one point written
    assert len(fake._points["test"]) == 1


async def test_search_returns_top_k_ordered_by_score() -> None:
    store, _fake = _store()
    await store.upsert("a:S", [1.0, 0.0], {"source_id": "a", "index": "S"})
    await store.upsert("b:S", [0.0, 1.0], {"source_id": "b", "index": "S"})
    await store.upsert("c:S", [0.9, 0.1], {"source_id": "c", "index": "S"})

    hits = await store.search([1.0, 0.0], top_k=2)
    vids = [vid for vid, _, _ in hits]
    assert vids[0] == "a:S"
    assert vids[1] == "c:S"


async def test_search_respects_restrict_to() -> None:
    store, _ = _store()
    await store.upsert("a:S", [1.0, 0.0], {"source_id": "a", "index": "S"})
    await store.upsert("b:S", [0.9, 0.1], {"source_id": "b", "index": "S"})
    hits = await store.search([1.0, 0.0], top_k=5, restrict_to=["b"])
    assert [vid for vid, _, _ in hits] == ["b:S"]


async def test_search_respects_index_filter() -> None:
    store, _ = _store()
    await store.upsert("a:S", [1.0, 0.0], {"source_id": "a", "index": "S"})
    await store.upsert("a:T", [1.0, 0.0], {"source_id": "a", "index": "T"})
    hits = await store.search([1.0, 0.0], top_k=5, index="T")
    assert [meta["index"] for _, _, meta in hits] == ["T"]


async def test_delete_by_source_removes_all_matching_vectors() -> None:
    store, fake = _store()
    await store.upsert("a:S", [1.0, 0.0], {"source_id": "a", "index": "S"})
    await store.upsert("a:T", [1.0, 0.0], {"source_id": "a", "index": "T"})
    await store.upsert("b:S", [0.0, 1.0], {"source_id": "b", "index": "S"})

    removed = await store.delete_by_source("a")
    assert removed == 2
    remaining = list(fake._points["test"].values())
    assert len(remaining) == 1
    assert remaining[0].payload["source_id"] == "b"


async def test_delete_by_source_missing_returns_zero() -> None:
    store, _ = _store()
    await store.upsert("a:S", [1.0, 0.0], {"source_id": "a", "index": "S"})
    assert await store.delete_by_source("missing") == 0


async def test_client_ctor_kwargs_captured_for_each_option() -> None:
    # Exercise each optional connection kwarg independently — covers the
    # per-kwarg ``is not None`` branches in __init__.
    store_url = QdrantVectorStore(collection="t", url="http://x")
    store_host = QdrantVectorStore(collection="t", host="localhost")
    store_port = QdrantVectorStore(collection="t", port=6333)
    store_key = QdrantVectorStore(collection="t", api_key="sekret")
    assert store_url._client_ctor_kwargs == {"url": "http://x"}
    assert store_host._client_ctor_kwargs == {"host": "localhost"}
    assert store_port._client_ctor_kwargs == {"port": 6333}
    assert store_key._client_ctor_kwargs == {"api_key": "sekret"}


async def test_lazy_client_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    import qdrant_client  # pyright: ignore[reportMissingImports]

    captured: dict[str, object] = {}

    class _Ctor:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)
            self._fake = FakeAsyncQdrantClient()

        def __getattr__(self, name: str) -> object:
            return getattr(self._fake, name)

    monkeypatch.setattr(qdrant_client, "AsyncQdrantClient", _Ctor)
    store = QdrantVectorStore(collection="t", url="http://x")
    await store.upsert("a:S", [1.0, 0.0], {"source_id": "a", "index": "S"})
    assert captured == {"url": "http://x"}


async def test_upsert_skips_create_when_collection_exists() -> None:
    fake = FakeAsyncQdrantClient()
    # Pre-create the collection so the adapter's _ensure_collection sees it.
    await fake.create_collection(collection_name="test", vectors_config=None)
    store = QdrantVectorStore(collection="test", client=fake)  # type: ignore[arg-type]
    # This upsert should NOT try to create again.
    await store.upsert("a:S", [1.0, 0.0], {"source_id": "a", "index": "S"})
    assert len(fake._points["test"]) == 1
