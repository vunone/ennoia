# Stores

A `Store` pairs a structured backend with a vector backend. The pipeline
runs the two retrieval phases against them:

1. `await store.structured.filter(filters)` narrows the candidate set.
2. `await store.vector.search(query_vector, top_k, restrict_to=candidates)`
   ranks within it.

Every backend implements one of three ABCs in `ennoia.store.base`. All
methods are async — even in-memory backends, which makes the pipeline a
single `await` path regardless of backing storage.

- `StructuredStore`: `async upsert` / `async filter` / `async get`
- `VectorStore`: `async upsert` / `async search`
- `HybridStore`: `async upsert` / `async hybrid_search` / `async get` /
  `async filter` (powers the two-phase MCP flow)

## Built-in backends

| Kind | Class | Extra | Notes |
|---|---|---|---|
| Structured | `InMemoryStructuredStore` | — | Dev / testing |
| Structured | `SQLiteStructuredStore` | — (`aiosqlite` is a base dep) | One table per `collection`, JSON payload, scalar ops push down to SQL |
| Structured | `ParquetStructuredStore` | `filesystem` | `{collection}.parquet`, read-modify-write upserts (pandas dispatched via `asyncio.to_thread`) |
| Vector | `InMemoryVectorStore` | `sentence-transformers` (for numpy) | Dev / testing |
| Vector | `FilesystemVectorStore` | `filesystem` / `sentence-transformers` | `{collection}.npy` + `{collection}_ids.json` + `{collection}_metadata.json` (numpy dispatched via `asyncio.to_thread`) |
| Vector | `QdrantVectorStore` | `qdrant` | Qdrant collection, one unnamed vector slot per point, UUIDv5 stable point ids |
| Hybrid | `QdrantHybridStore` | `qdrant` | One Qdrant collection holds structured payload + named vectors; filter + vector search in a single native query |
| Hybrid | `PgVectorHybridStore` | `pgvector` | PostgreSQL + pgvector, JSONB payload + per-index vector columns; every operator pushes down to SQL via `asyncpg` |

`SQLiteStructuredStore` opens its `aiosqlite` connection lazily and rebinds
to the running event loop on every call — the sync `Pipeline.index` /
`Pipeline.search` wrappers create a fresh loop per invocation, so a cached
connection from a closed loop would otherwise raise.

## `Store.from_path`

The CLI default. Builds a `ParquetStructuredStore` + `FilesystemVectorStore`
under `<path>/<collection>/`. The `collection` kwarg (default
`"documents"`) becomes the namespace subdirectory, so multiple pipelines
can share one project root without colliding:

```python
from ennoia.store import Store

invoices = Store.from_path("./my_index", collection="invoices")
emails = Store.from_path("./my_index", collection="emails")
```

Layout:

```
my_index/
├── invoices/
│   ├── structured/documents.parquet
│   └── vectors/{documents.npy, documents_ids.json, documents_metadata.json}
└── emails/
    ├── structured/documents.parquet
    └── vectors/{documents.npy, documents_ids.json, documents_metadata.json}
```

Collection names must match `^[A-Za-z_][A-Za-z0-9_]*$` — the same
identifier grammar enforced by the SQL-backed stores — so filesystem
layouts, SQLite tables, and pgvector tables all use the same collection
name without translation.

## SQLite specifics

Scalar operators (`eq`, `gt`, `gte`, `lt`, `lte`, `in`, `is_null`)
compile to `json_extract(data, '$.<field>') ...` SQL fragments and run
server-side. List and substring operators (`contains`, `contains_all`,
`contains_any`, `startswith`) load the matching rows and reuse the same
Python `apply_filters` utility every other store uses — the trade-off is
correctness + shared semantics today, full pushdown in a later iteration.

## Custom backends

Subclass the ABC and plug the instance into `Store(vector=..., structured=...)`:

```python
from elasticsearch import AsyncElasticsearch

from ennoia.store.base import StructuredStore


class ElasticsearchStructuredStore(StructuredStore):
    def __init__(self, host: str, index_name: str):
        self.client = AsyncElasticsearch(host)
        self.index_name = index_name

    async def upsert(self, source_id: str, data: dict) -> None:
        await self.client.index(index=self.index_name, id=source_id, document=data)

    async def filter(self, query: dict) -> list[str]:
        ...

    async def get(self, source_id: str) -> dict | None:
        ...
```

Backends without a native async client should wrap each blocking call in
`await asyncio.to_thread(...)` — the parquet and filesystem-vector stores
are the canonical examples.

## Qdrant

Install the `qdrant` extra:

```bash
pip install "ennoia[qdrant]"
```

Two backends ship against Qdrant:

- `QdrantVectorStore(collection, *, url=..., host=..., port=..., api_key=..., vector_size=None, distance="Cosine", client=None)`
  — a pure `VectorStore`. One unnamed vector slot per point, UUIDv5 point
  ids so repeated upserts overwrite cleanly. Pair it with any
  `StructuredStore` via the composite `Store(...)`.
- `QdrantHybridStore(collection, *, url=..., host=..., port=..., api_key=..., distance="Cosine", list_payload_fields=frozenset(), client=None)`
  — a `HybridStore`. One point per `source_id` with named vectors per
  semantic index; `hybrid_search` issues a single native query that
  combines the structured filter with the vector similarity search.
  `list_payload_fields` marks payload keys that should be treated as
  lists at filter-translation time (for `contains_all` / `contains_any`).

```python
from ennoia.store.hybrid.qdrant import QdrantHybridStore

store = QdrantHybridStore(
    collection="cases",
    url="http://localhost:6333",
    list_payload_fields=frozenset({"tags"}),
)
```

## pgvector

Install the `pgvector` extra:

```bash
pip install "ennoia[pgvector]"
```

`PgVectorHybridStore(dsn, *, collection="documents", connection=None)` is
a `HybridStore` backed by a single PostgreSQL table named after
`collection`, with a JSONB payload column and one `vector(N)` column per
semantic index (materialised lazily on first sight of each index). Every
filter operator compiles to a parameterised `asyncpg` query via
`ennoia.store.hybrid._sql_filter`, so there is no Python post-filter
residual.

```python
from ennoia.store.hybrid.pgvector import PgVectorHybridStore

store = PgVectorHybridStore(
    dsn="postgresql://user:pass@localhost:5432/ennoia",
    collection="cases",
)
```

Both hybrid backends plug directly into `Pipeline(store=...)` — no
composite `Store(...)` wrapping required.

## Custom hybrid stores

See [Cookbook — Custom adapters and stores](cookbook/custom-adapter.md#custom-hybrid-store)
for a walkthrough of the `HybridStore` contract and a pattern for
post-filtering residual operators via the canonical `apply_filters`
evaluator.
