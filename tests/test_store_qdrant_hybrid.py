"""QdrantHybridStore unit tests with an in-memory fake Qdrant client."""

from __future__ import annotations

import pytest

from ennoia.store.hybrid.qdrant import QdrantHybridStore
from tests._qdrant_fake import FakeAsyncQdrantClient


def _store() -> tuple[QdrantHybridStore, FakeAsyncQdrantClient]:
    fake = FakeAsyncQdrantClient()
    store = QdrantHybridStore(collection="hyb", client=fake)  # type: ignore[arg-type]
    return store, fake


async def test_upsert_creates_collection_with_named_vectors() -> None:
    store, fake = _store()
    await store.upsert(
        "doc_1",
        {"cat": "legal"},
        {"Holding": [1.0, 0.0, 0.0], "Facts": [0.0, 1.0, 0.0]},
    )
    assert await fake.collection_exists("hyb")
    points = fake._points["hyb"]
    assert len(points) == 1
    point = next(iter(points.values()))
    assert set(point.vector.keys()) == {"Holding", "Facts"}
    assert point.payload["source_id"] == "doc_1"


async def test_upsert_with_empty_vectors_before_collection_raises() -> None:
    store, _ = _store()
    with pytest.raises(RuntimeError, match="named-vector spec"):
        await store.upsert("doc_x", {"cat": "legal"}, {})


async def test_upsert_with_empty_vectors_after_collection_updates_payload() -> None:
    store, fake = _store()
    # First upsert creates the collection + first point.
    await store.upsert("doc_1", {"cat": "legal"}, {"Holding": [1.0, 0.0]})
    # Second upsert: no vectors; should fall back to set_payload on existing point.
    await store.upsert("doc_1", {"cat": "legal", "note": "added"}, {})
    record = fake._points["hyb"][next(iter(fake._points["hyb"]))]
    assert record.payload["note"] == "added"


async def test_hybrid_search_returns_scored_hits_with_index_metadata() -> None:
    store, _ = _store()
    await store.upsert("doc_1", {"cat": "legal"}, {"Holding": [1.0, 0.0]})
    await store.upsert("doc_2", {"cat": "legal"}, {"Holding": [0.5, 0.5]})

    hits = await store.hybrid_search({"cat": "legal"}, [1.0, 0.0], top_k=5)
    assert len(hits) == 2
    # Each hit's metadata has the virtual index name we queried on.
    for _, _, meta in hits:
        assert meta["index"] == "Holding"


async def test_hybrid_search_respects_index_override() -> None:
    store, _ = _store()
    await store.upsert(
        "doc_1",
        {"cat": "legal"},
        {"Holding": [1.0, 0.0], "Facts": [0.0, 1.0]},
    )
    # Query against the Facts slot — the top hit's "index" metadata reflects that.
    hits = await store.hybrid_search({"cat": "legal"}, [0.0, 1.0], top_k=5, index="Facts")
    assert hits[0][2]["index"] == "Facts"


async def test_hybrid_search_residual_post_filters_native_hits() -> None:
    # ``startswith`` isn't native; it should survive the translator and the hybrid
    # store must post-filter candidate payloads.
    store, _ = _store()
    await store.upsert("doc_1", {"cat": "legal", "title": "In re Acme"}, {"H": [1.0, 0.0]})
    await store.upsert("doc_2", {"cat": "legal", "title": "Unrelated"}, {"H": [0.9, 0.1]})

    hits = await store.hybrid_search(
        {"cat": "legal", "title__startswith": "In re"}, [1.0, 0.0], top_k=5
    )
    vids = [meta["source_id"] for _, _, meta in hits]
    assert vids == ["doc_1"]


async def test_filter_returns_source_ids() -> None:
    store, _ = _store()
    await store.upsert("doc_1", {"cat": "legal"}, {"H": [1.0, 0.0]})
    await store.upsert("doc_2", {"cat": "medical"}, {"H": [0.0, 1.0]})
    ids = await store.filter({"cat": "legal"})
    assert ids == ["doc_1"]


async def test_filter_with_residual_post_filters_in_python() -> None:
    store, _ = _store()
    await store.upsert("doc_1", {"cat": "legal", "title": "In re Acme"}, {"H": [1.0, 0.0]})
    await store.upsert("doc_2", {"cat": "legal", "title": "Unrelated"}, {"H": [0.0, 1.0]})
    ids = await store.filter({"cat": "legal", "title__startswith": "In re"})
    assert ids == ["doc_1"]


async def test_get_returns_payload_without_source_id_echo() -> None:
    store, _ = _store()
    await store.upsert("doc_1", {"cat": "legal"}, {"H": [1.0, 0.0]})
    data = await store.get("doc_1")
    assert data == {"cat": "legal"}


async def test_get_returns_none_for_missing() -> None:
    store, _ = _store()
    # Pre-create the collection so the lookup has something to scan.
    await store.upsert("doc_1", {"cat": "legal"}, {"H": [1.0, 0.0]})
    assert await store.get("missing") is None


async def test_delete_removes_point_and_reports_bool() -> None:
    store, fake = _store()
    await store.upsert("doc_1", {"cat": "legal"}, {"H": [1.0, 0.0]})
    assert await store.delete("doc_1") is True
    assert not fake._points["hyb"]
    # Second delete: already gone.
    assert await store.delete("doc_1") is False


async def test_client_ctor_kwargs_captured_for_each_option() -> None:
    store_url = QdrantHybridStore(collection="h", url="http://x")
    store_host = QdrantHybridStore(collection="h", host="localhost")
    store_port = QdrantHybridStore(collection="h", port=1)
    store_key = QdrantHybridStore(collection="h", api_key="k")
    assert store_url._client_ctor_kwargs == {"url": "http://x"}
    assert store_host._client_ctor_kwargs == {"host": "localhost"}
    assert store_port._client_ctor_kwargs == {"port": 1}
    assert store_key._client_ctor_kwargs == {"api_key": "k"}


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
    store = QdrantHybridStore(collection="h", url="http://x")
    await store.upsert("doc_1", {"cat": "legal"}, {"H": [1.0, 0.0]})
    assert captured == {"url": "http://x"}


async def test_upsert_skips_create_when_collection_exists() -> None:
    fake = FakeAsyncQdrantClient()
    await fake.create_collection(collection_name="hyb", vectors_config=None)
    store = QdrantHybridStore(collection="hyb", client=fake)  # type: ignore[arg-type]
    # This upsert should NOT try to create again.
    await store.upsert("doc_1", {"cat": "legal"}, {"H": [1.0, 0.0]})
    assert len(fake._points["hyb"]) == 1
