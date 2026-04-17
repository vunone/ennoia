"""QdrantHybridStore unit tests with an in-memory fake Qdrant client."""

from __future__ import annotations

import pytest

from ennoia.store.base import VectorEntry
from ennoia.store.hybrid.qdrant import QdrantHybridStore
from tests._qdrant_fake import FakeAsyncQdrantClient


def _store() -> tuple[QdrantHybridStore, FakeAsyncQdrantClient]:
    fake = FakeAsyncQdrantClient()
    store = QdrantHybridStore(collection="hyb", client=fake)  # type: ignore[arg-type]
    return store, fake


def _entries(*specs: tuple[str, list[float], str, str | None]) -> list[VectorEntry]:
    return [
        VectorEntry(index_name=name, vector=vec, text=text, unique=unique)
        for name, vec, text, unique in specs
    ]


async def test_upsert_creates_one_point_per_entry() -> None:
    store, fake = _store()
    await store.upsert(
        "doc_1",
        {"cat": "legal"},
        _entries(
            ("Holding", [1.0, 0.0, 0.0], "holding-text", None),
            ("Facts", [0.0, 1.0, 0.0], "facts-text", None),
        ),
    )
    assert await fake.collection_exists("hyb")
    points = fake._points["hyb"]
    # One point per entry, not one point per document.
    assert len(points) == 2
    payloads = [p.payload for p in points.values()]
    index_names = sorted(p["index_name"] for p in payloads)
    assert index_names == ["Facts", "Holding"]
    for p in payloads:
        assert p["source_id"] == "doc_1"
        assert p["cat"] == "legal"


async def test_collection_entity_becomes_its_own_point() -> None:
    store, fake = _store()
    await store.upsert(
        "doc_1",
        {"cat": "legal"},
        _entries(
            ("Parties", [1.0, 0.0], "Alice", "alice_key"),
            ("Parties", [0.0, 1.0], "Bob", "bob_key"),
            ("Parties", [0.5, 0.5], "Carol", "carol_key"),
        ),
    )
    points = list(fake._points["hyb"].values())
    assert len(points) == 3
    payloads_by_unique = {p.payload["unique"]: p.payload for p in points}
    assert set(payloads_by_unique) == {"alice_key", "bob_key", "carol_key"}
    assert all(p["index_name"] == "Parties" for p in payloads_by_unique.values())


async def test_upsert_with_no_entries_is_noop_on_fresh_collection() -> None:
    store, fake = _store()
    # Empty entries + never-created collection: the adapter just returns; we
    # don't require the hybrid store to materialise structural-only docs in
    # Qdrant, since Qdrant requires a vector per point.
    await store.upsert("doc_x", {"cat": "legal"}, [])
    assert not await fake.collection_exists("hyb")


async def test_reindex_replaces_prior_points_for_source() -> None:
    store, fake = _store()
    await store.upsert(
        "doc_1",
        {"cat": "legal"},
        _entries(
            ("Parties", [1.0, 0.0], "Alice", "a"),
            ("Parties", [0.0, 1.0], "Bob", "b"),
        ),
    )
    assert len(fake._points["hyb"]) == 2
    await store.upsert(
        "doc_1",
        {"cat": "legal"},
        _entries(("Parties", [1.0, 0.0], "Alice", "a")),
    )
    # Stale Bob must be gone.
    remaining = [p.payload["unique"] for p in fake._points["hyb"].values()]
    assert remaining == ["a"]


async def test_hybrid_search_returns_scored_hits_with_metadata() -> None:
    store, _ = _store()
    await store.upsert(
        "doc_1",
        {"cat": "legal"},
        _entries(("Holding", [1.0, 0.0], "hello", None)),
    )
    await store.upsert(
        "doc_2",
        {"cat": "legal"},
        _entries(("Holding", [0.5, 0.5], "world", None)),
    )

    hits = await store.hybrid_search({"cat": "legal"}, [1.0, 0.0], top_k=5)
    assert len(hits) == 2
    for _, _, meta in hits:
        assert meta["index"] == "Holding"
        assert "text" in meta


async def test_hybrid_search_respects_index_override() -> None:
    store, _ = _store()
    await store.upsert(
        "doc_1",
        {"cat": "legal"},
        _entries(
            ("Holding", [1.0, 0.0], "h-text", None),
            ("Facts", [0.0, 1.0], "f-text", None),
        ),
    )
    hits = await store.hybrid_search({"cat": "legal"}, [0.0, 1.0], top_k=5, index="Facts")
    assert hits
    assert all(meta["index"] == "Facts" for _, _, meta in hits)


async def test_hybrid_search_residual_post_filters_native_hits() -> None:
    # ``startswith`` isn't native; it should survive the translator and the hybrid
    # store must post-filter candidate payloads.
    store, _ = _store()
    await store.upsert(
        "doc_1",
        {"cat": "legal", "title": "In re Acme"},
        _entries(("H", [1.0, 0.0], "h", None)),
    )
    await store.upsert(
        "doc_2",
        {"cat": "legal", "title": "Unrelated"},
        _entries(("H", [0.9, 0.1], "h", None)),
    )
    hits = await store.hybrid_search(
        {"cat": "legal", "title__startswith": "In re"}, [1.0, 0.0], top_k=5
    )
    vids = [meta["source_id"] for _, _, meta in hits]
    assert vids == ["doc_1"]


async def test_filter_returns_distinct_source_ids() -> None:
    store, _ = _store()
    await store.upsert(
        "doc_1",
        {"cat": "legal"},
        _entries(
            ("Parties", [1.0, 0.0], "Alice", "a"),
            ("Parties", [0.0, 1.0], "Bob", "b"),
        ),
    )
    await store.upsert(
        "doc_2",
        {"cat": "medical"},
        _entries(("Parties", [0.5, 0.5], "Eve", "e")),
    )
    ids = await store.filter({"cat": "legal"})
    # Distinct source_ids — multiple rows for doc_1 collapse.
    assert ids == ["doc_1"]


async def test_filter_with_residual_post_filters_in_python() -> None:
    store, _ = _store()
    await store.upsert(
        "doc_1",
        {"cat": "legal", "title": "In re Acme"},
        _entries(("H", [1.0, 0.0], "h", None)),
    )
    await store.upsert(
        "doc_2",
        {"cat": "legal", "title": "Unrelated"},
        _entries(("H", [0.0, 1.0], "h", None)),
    )
    ids = await store.filter({"cat": "legal", "title__startswith": "In re"})
    assert ids == ["doc_1"]


async def test_get_returns_payload_without_row_metadata() -> None:
    store, _ = _store()
    await store.upsert(
        "doc_1",
        {"cat": "legal"},
        _entries(("H", [1.0, 0.0], "h", None)),
    )
    data = await store.get("doc_1")
    # Strip all row-level keys from the return — callers want pure structural data.
    assert data == {"cat": "legal"}


async def test_get_returns_none_for_missing() -> None:
    store, _ = _store()
    # Pre-create the collection so the lookup has something to scan.
    await store.upsert("doc_1", {"cat": "legal"}, _entries(("H", [1.0, 0.0], "h", None)))
    assert await store.get("missing") is None


async def test_delete_removes_all_points_for_source() -> None:
    store, fake = _store()
    await store.upsert(
        "doc_1",
        {"cat": "legal"},
        _entries(
            ("Parties", [1.0, 0.0], "Alice", "a"),
            ("Parties", [0.0, 1.0], "Bob", "b"),
        ),
    )
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
    await store.upsert("doc_1", {"cat": "legal"}, _entries(("H", [1.0, 0.0], "h", None)))
    assert captured == {"url": "http://x"}


async def test_upsert_skips_create_when_collection_exists() -> None:
    fake = FakeAsyncQdrantClient()
    await fake.create_collection(collection_name="hyb", vectors_config=None)
    store = QdrantHybridStore(collection="hyb", client=fake)  # type: ignore[arg-type]
    # This upsert should NOT try to create again.
    await store.upsert("doc_1", {"cat": "legal"}, _entries(("H", [1.0, 0.0], "h", None)))
    assert len(fake._points["hyb"]) == 1


async def test_read_methods_short_circuit_before_first_upsert() -> None:
    # Every read method checks ``_collection_ensured`` first so unqualified
    # calls against a cold client don't error on a missing collection.
    store, _ = _store()
    assert await store.hybrid_search({}, [1.0, 0.0], top_k=5) == []
    assert await store.get("doc_1") is None
    assert await store.filter({}) == []
    assert await store.delete("doc_1") is False


async def test_filter_residual_deduplicates_multi_row_source_ids() -> None:
    # When a residual filter matches multiple rows of the same document, the
    # result must still report that source_id exactly once.
    store, _ = _store()
    await store.upsert(
        "doc_1",
        {"cat": "legal", "title": "In re Acme"},
        _entries(
            ("Parties", [1.0, 0.0], "Alice", "a"),
            ("Parties", [0.0, 1.0], "Bob", "b"),
        ),
    )
    ids = await store.filter({"cat": "legal", "title__startswith": "In re"})
    assert ids == ["doc_1"]


async def test_hybrid_search_index_only_filter_with_no_filters() -> None:
    # Index-only restriction exercises the branch where ``qfilter`` starts as
    # ``None`` and is built from the single FieldCondition.
    store, _ = _store()
    await store.upsert(
        "doc_1",
        {"cat": "legal"},
        _entries(
            ("A", [1.0, 0.0], "a", None),
            ("B", [0.0, 1.0], "b", None),
        ),
    )
    hits = await store.hybrid_search({}, [1.0, 0.0], top_k=5, index="B")
    assert [meta["index"] for _, _, meta in hits] == ["B"]
