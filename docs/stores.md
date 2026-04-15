# Stores

A `Store` pairs a structured backend with a vector backend. The pipeline
runs the two retrieval phases against them:

1. `store.structured.filter(filters)` narrows the candidate set.
2. `store.vector.search(query_vector, top_k, restrict_to=candidates)`
   ranks within it.

Every backend implements one of three ABCs in `ennoia.store.base`:

- `StructuredStore`: `upsert` / `filter` / `get`
- `VectorStore`: `upsert` / `search`
- `HybridStore`: `upsert` / `hybrid_search` / `get` (Stage 3)

## Built-in backends

| Kind | Class | Extra | Notes |
|---|---|---|---|
| Structured | `InMemoryStructuredStore` | — | Dev / testing |
| Structured | `SQLiteStructuredStore` | — (stdlib) | One table, JSON payload, scalar ops push down to SQL |
| Structured | `ParquetStructuredStore` | `filesystem` | One `.parquet` file, read-modify-write upserts |
| Vector | `InMemoryVectorStore` | `sentence-transformers` (for numpy) | Dev / testing |
| Vector | `FilesystemVectorStore` | `filesystem` / `sentence-transformers` | `vectors.npy` + `ids.json` + `metadata.json` |

## `Store.from_path`

The CLI default. Builds a `ParquetStructuredStore` + `FilesystemVectorStore`
under one directory:

```python
from ennoia.store import Store
store = Store.from_path("./my_index")
```

Layout:

```
my_index/
├── structured/
│   └── structured.parquet
└── vectors/
    ├── vectors.npy
    ├── ids.json
    └── metadata.json
```

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
from elasticsearch import Elasticsearch

from ennoia.store.base import StructuredStore


class ElasticsearchStructuredStore(StructuredStore):
    def __init__(self, host: str, index_name: str):
        self.client = Elasticsearch(host)
        self.index_name = index_name

    def upsert(self, source_id: str, data: dict) -> None:
        self.client.index(index=self.index_name, id=source_id, document=data)

    def filter(self, query: dict) -> list[str]:
        ...

    def get(self, source_id: str) -> dict | None:
        ...
```

Hybrid stores (single roundtrip for filter + vector search) land in
Stage 3 alongside the Qdrant integration.
