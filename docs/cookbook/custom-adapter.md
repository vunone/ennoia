# Cookbook: custom adapters and stores

Ennoia's integration points are a small set of abstract base classes.
Subclass one, implement the abstract method(s), pass the instance into
`Pipeline(...)` — no framework changes required.

| Contract | ABC path | Abstract methods |
|---|---|---|
| LLM | `ennoia/adapters/llm/base.py` | `complete_json`, `complete_text` |
| Embedding | `ennoia/adapters/embedding/base.py` | `embed` (plus optional `embed_batch` override) |
| Structured store | `ennoia/store/base.py` | `upsert`, `filter`, `get` (`delete` recommended) |
| Vector store | `ennoia/store/base.py` | `upsert`, `search` (`delete_by_source` recommended) |
| Hybrid store | `ennoia/store/base.py` | `upsert`, `hybrid_search`, `get` (`filter`, `delete` recommended) |

All methods are async. Sync libraries should wrap blocking calls in
`await asyncio.to_thread(...)` — the sentence-transformers embedding
adapter and the parquet structured store are the canonical examples.

## Custom LLM adapter

Implements the two abstract methods from
[`LLMAdapter`](https://github.com/vunone/ennoia/blob/main/ennoia/adapters/llm/base.py):

```python
from typing import Any

import httpx

from ennoia.adapters.llm.base import LLMAdapter, parse_json_object


class VllmAdapter(LLMAdapter):
    """Talk to a self-hosted vLLM OpenAI-compatible endpoint."""

    def __init__(self, base_url: str, model: str, api_key: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    async def complete_json(self, prompt: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=120) as client:  # fresh client per call
            response = await client.post(
                f"{self.base_url}/v1/chat/completions",
                headers=self._headers(),
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},
                },
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        return parse_json_object(content, source="vllm")

    async def complete_text(self, prompt: str) -> str:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{self.base_url}/v1/chat/completions",
                headers=self._headers(),
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
```

Two things worth calling out:

- **Fresh `httpx.AsyncClient` per call.** httpx transports are event-loop
  bound, and the sync `Pipeline.index` / `Pipeline.search` wrappers
  re-enter `asyncio.run()` on every call — caching a client across calls
  surfaces "Event loop is closed" on some Python runtimes. Every built-in
  adapter follows this convention; see the [adapter notes](../adapters.md#built-in-llm-adapters).
- **Parse through `parse_json_object`**, the shared helper colocated with
  the `LLMAdapter` ABC. It raises `ExtractionError` on malformed JSON so
  the pipeline's retry / error reporting paths stay consistent.

## Custom embedding adapter

Override `embed` — that's the single abstract hook. `embed_document`,
`embed_query`, and `embed_batch` are inherited and route through `embed`.
Override them only when you have a reason to.

```python
import asyncio

from ennoia.adapters.embedding.base import EmbeddingAdapter


class VoyageEmbedding(EmbeddingAdapter):
    """Voyage uses different ``input_type`` for document vs query."""

    def __init__(self, model: str, api_key: str) -> None:
        from voyageai import AsyncClient  # local import keeps the dep optional

        self.client = AsyncClient(api_key=api_key)
        self.model = model

    async def embed(self, text: str) -> list[float]:
        # Symmetric default — Voyage will still work, just not optimally.
        return (await self.embed_document(text))

    async def embed_document(self, text: str) -> list[float]:
        result = await self.client.embed([text], model=self.model, input_type="document")
        return list(result.embeddings[0])

    async def embed_query(self, text: str) -> list[float]:
        result = await self.client.embed([text], model=self.model, input_type="query")
        return list(result.embeddings[0])

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Single round-trip for the whole batch — beats the default
        # asyncio.gather fallback over N single calls.
        result = await self.client.embed(texts, model=self.model, input_type="document")
        return [list(v) for v in result.embeddings]
```

If your library is sync, dispatch the call via `asyncio.to_thread`:

```python
class LocalEmbedding(EmbeddingAdapter):
    def __init__(self, model) -> None:
        self.model = model

    async def embed(self, text: str) -> list[float]:
        vec = await asyncio.to_thread(self.model.encode, text)
        return list(vec)
```

## Custom structured store

Implement `upsert`, `filter`, `get`, and (recommended) `delete`:

```python
from typing import Any

from elasticsearch import AsyncElasticsearch

from ennoia.store.base import StructuredStore


class ElasticsearchStructuredStore(StructuredStore):
    def __init__(self, host: str, index_name: str) -> None:
        self.client = AsyncElasticsearch(host)
        self.index_name = index_name

    async def upsert(self, source_id: str, data: dict[str, Any]) -> None:
        await self.client.index(index=self.index_name, id=source_id, document=data)

    async def filter(self, query: dict[str, Any]) -> list[str]:
        body = self._compile(query)  # your operator → ES DSL translation
        response = await self.client.search(index=self.index_name, body=body)
        return [hit["_id"] for hit in response["hits"]["hits"]]

    async def get(self, source_id: str) -> dict[str, Any] | None:
        try:
            doc = await self.client.get(index=self.index_name, id=source_id)
        except Exception:  # NotFoundError in the real client
            return None
        return dict(doc["_source"])

    async def delete(self, source_id: str) -> bool:
        result = await self.client.delete(index=self.index_name, id=source_id, ignore=[404])
        return result["result"] == "deleted"
```

For operator semantics, lean on `ennoia.utils.filters.apply_filters` —
every built-in store falls back to it for list / substring operators
that don't push down cleanly. That keeps correctness tied to one
evaluator.

## Custom hybrid store

`HybridStore` is the Stage 3 one-shot interface: structured filter +
vector search in a single round-trip.

```python
from typing import Any

from ennoia.store.base import HybridStore
from ennoia.utils.filters import apply_filters


class MyHybridStore(HybridStore):
    async def upsert(
        self,
        source_id: str,
        data: dict[str, Any],
        vectors: dict[str, list[float]],
    ) -> None:
        # Persist the structured payload + named vectors in one write.
        ...

    async def hybrid_search(
        self,
        filters: dict[str, Any],
        query_vector: list[float],
        top_k: int,
        *,
        index: str | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        # One native query that filters + vector-ranks. Return
        # (virtual_vector_id, score, metadata) triples; metadata must
        # include "source_id" and "index" so the pipeline can hydrate
        # SearchHit correctly.
        ...

    async def get(self, source_id: str) -> dict[str, Any] | None:
        ...

    async def filter(self, filters: dict[str, Any]) -> list[str]:
        # Powers ``Pipeline.afilter`` and the REST ``POST /filter`` endpoint.
        # If your backend doesn't push down every operator natively,
        # post-filter the candidate rows through ``apply_filters``.
        ...

    async def delete(self, source_id: str) -> bool:
        ...
```

The [`QdrantHybridStore`](https://github.com/vunone/ennoia/blob/main/ennoia/store/hybrid/qdrant.py)
and [`PgVectorHybridStore`](https://github.com/vunone/ennoia/blob/main/ennoia/store/hybrid/pgvector.py)
implementations are useful references — one pushes down some operators
and post-filters the residual in Python; the other translates every
operator to SQL.

## Plugging it in

A pipeline mixes and matches any compliant adapter / store:

```python
from ennoia import Pipeline, Store
from ennoia.adapters.embedding.sentence_transformers import SentenceTransformerEmbedding

pipeline = Pipeline(
    schemas=[...],
    store=Store(
        vector=QdrantVectorStore(collection="docs", url="http://localhost:6333"),
        structured=ElasticsearchStructuredStore(host="localhost:9200", index_name="docs"),
    ),
    llm=VllmAdapter(base_url="http://localhost:8000", model="llama3.1-70b"),
    embedding=SentenceTransformerEmbedding(model="all-MiniLM-L6-v2"),
)
```

Or with a hybrid store (the pipeline accepts `store=HybridStore` directly):

```python
pipeline = Pipeline(
    schemas=[...],
    store=MyHybridStore(...),
    llm=VllmAdapter(...),
    embedding=SentenceTransformerEmbedding(...),
)
```

## Optional dependencies

If your adapter / store depends on a third-party package, follow the
ennoia convention for optional-dependency errors:

```python
from ennoia.utils.imports import require_module


voyage = require_module("voyageai", "voyage")
```

Pair that with an `extras` entry in your own `pyproject.toml`. Users get
a uniform `ImportError: voyageai is required. Install with 'pip install
yourpkg[voyage]'` instead of a confusing `ModuleNotFoundError`.
