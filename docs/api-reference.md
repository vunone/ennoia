# API reference

Generated from docstrings via
[mkdocstrings](https://mkdocstrings.github.io). For narrative guides see
[Concepts](concepts.md) and the [Quickstart](quickstart.md).

## Pipeline

::: ennoia.Pipeline
    options:
      members:
        - __init__
        - schemas
        - index
        - aindex
        - search
        - asearch
        - filter
        - afilter
        - retrieve
        - aretrieve
        - delete
        - adelete

## Results

::: ennoia.IndexResult

::: ennoia.SearchHit

::: ennoia.SearchResult

## Exceptions

::: ennoia.ExtractionError

::: ennoia.FilterValidationError

::: ennoia.RejectException

::: ennoia.SkipItem

## Schemas

::: ennoia.BaseStructure

::: ennoia.BaseSemantic

::: ennoia.BaseCollection

::: ennoia.Field

::: ennoia.describe

## Stores

### ABCs

::: ennoia.store.StructuredStore

::: ennoia.store.VectorStore

::: ennoia.store.HybridStore

### Composite

::: ennoia.Store

### Built-in backends

::: ennoia.store.InMemoryStructuredStore

::: ennoia.store.InMemoryVectorStore

::: ennoia.store.structured.sqlite.SQLiteStructuredStore

::: ennoia.store.structured.parquet.ParquetStructuredStore

::: ennoia.store.vector.filesystem.FilesystemVectorStore

::: ennoia.store.vector.qdrant.QdrantVectorStore

::: ennoia.store.hybrid.qdrant.QdrantHybridStore

::: ennoia.store.hybrid.pgvector.PgVectorHybridStore

## Adapters

### LLM

::: ennoia.adapters.llm.LLMAdapter

::: ennoia.adapters.llm.base.parse_json_object

### Embedding

::: ennoia.adapters.embedding.EmbeddingAdapter

## Events

::: ennoia.Emitter

::: ennoia.ExtractionEvent

::: ennoia.IndexEvent

::: ennoia.SearchEvent

## Server

::: ennoia.server.AuthHook

::: ennoia.server.ServerContext

::: ennoia.server.static_bearer_auth

::: ennoia.server.env_bearer_auth

::: ennoia.server.no_auth

::: ennoia.server.api.create_app

::: ennoia.server.mcp.create_mcp

## Testing

::: ennoia.testing.MockLLMAdapter

::: ennoia.testing.MockEmbeddingAdapter

::: ennoia.testing.MockStore
