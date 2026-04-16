# CLI

Install the `cli` extra to get the `ennoia` console script:

```bash
pip install "ennoia[cli,ollama,sentence-transformers]"
```

## `ennoia try` — iterate on schemas

Run a single extraction pass against one document and print the fields,
confidences, and `extend()` chain. Nothing is written to a store.

```bash
ennoia try ./sample.txt --schema my_schemas.py
```

```
Schema: CaseDocument
  jurisdiction: 'WA'        (confidence: 0.95)
  court_level:  'appellate' (confidence: 0.87)
  date_decided: '2023-03-15'(confidence: 0.92)
  is_overruled: False       (confidence: 0.78)
  -> extend(): WashingtonAppellateSchema

Schema: Holding
  'The court reversed the lower court's finding, holding that...'  (confidence: 0.91)
```

## `ennoia index` — index a folder

Walks a directory and indexes each file through the pipeline. The
`--store` flag accepts a prefix-qualified spec — filesystem path by
default, `qdrant:<collection>` or `pgvector:<collection>` when pointing
at a production vector backend. See [Store backends](#store-backends) below.

```bash
ennoia index ./docs \
  --schema my_schemas.py \
  --store ./my_index \
  --collection invoices \
  --llm ollama:qwen3:0.6b \
  --embedding sentence-transformers:all-MiniLM-L6-v2
```

Multiple pipelines can sit under one project root by passing different
`--collection` values — each gets its own `<root>/<collection>/` subtree.

By default, independent schemas in a layer extract in parallel via
`asyncio.gather`. On a resource-constrained machine running local
Ollama, pass `--no-threads` to serialise LLM and embedding calls
(equivalent to `Pipeline(..., concurrency=1)`):

```bash
ennoia index ./docs --schema my_schemas.py --store ./my_index --no-threads
```

## `ennoia search` — hybrid search

```bash
ennoia search "employer duty to accommodate disability" \
  --schema my_schemas.py \
  --store ./my_index \
  --filter "jurisdiction=WA" \
  --filter "date_decided__gt=2020-01-01" \
  --top-k 5
```

Filters go through the same validator the SDK uses. An unsupported
operator exits with code `2` and an error JSON:

```json
{
  "error": "invalid_filter",
  "field": "jurisdiction",
  "operator": "gt",
  "message": "Field 'jurisdiction' (type: enum) does not support operator 'gt'. Supported operators: eq, in."
}
```

## `ennoia api` — REST server

Boot a FastAPI server against any supported store. Full CRUD surface —
`/discover`, `/filter`, `/search`, `/retrieve`, `/index`, `/delete`.

```bash
ennoia api --store qdrant:cases \
  --qdrant-url http://localhost:6333 \
  --schema my_schemas.py \
  --api-key "$ENNOIA_API_KEY"
```

Requires the `server` extra. See [Servers — REST + MCP](serve.md) for
every flag, the endpoint table, and auth configuration.

## `ennoia mcp` — MCP tool server

Read-only agent surface — `discover_schema`, `filter`, `search`,
`retrieve`. Supports `sse`, `stdio`, and `http` transports.

```bash
ennoia mcp --store qdrant:cases \
  --qdrant-url http://localhost:6333 \
  --schema my_schemas.py \
  --transport sse \
  --api-key "$ENNOIA_API_KEY"
```

Requires the `server` extra. See [Servers — REST + MCP](serve.md) and
the [MCP agent cookbook](cookbook/mcp-agent.md).

## Store backends

`--store` accepts three forms across `index`, `search`, `api`, and `mcp`:

| Form | Backend | Required extra flags |
|---|---|---|
| `<path>` or `file:<path>` | Filesystem (Parquet + NumPy) | `--collection` (default `documents`) |
| `qdrant:<collection>` | `QdrantHybridStore` | `--qdrant-url` (or `ENNOIA_QDRANT_URL`); `--qdrant-api-key` / `ENNOIA_QDRANT_API_KEY` optional |
| `pgvector:<collection>` | `PgVectorHybridStore` | `--pg-dsn` (or `ENNOIA_PG_DSN`) |

Collection names must match `^[A-Za-z_][A-Za-z0-9_]*$` so the same name
is valid as a subdirectory, SQLite table, Qdrant collection, and
PostgreSQL table.

## Adapter URIs

Every `--llm` / `--embedding` argument is a URI of the form
`prefix:model`. Current prefixes:

| Flag | Prefix | Example |
|---|---|---|
| `--llm` | `ollama` | `ollama:qwen3:0.6b` |
| `--llm` | `openai` | `openai:gpt-4o-mini` |
| `--llm` | `anthropic` | `anthropic:claude-sonnet-4-20250514` |
| `--embedding` | `sentence-transformers` | `sentence-transformers:all-MiniLM-L6-v2` |
| `--embedding` | `openai-embedding` | `openai-embedding:text-embedding-3-small` |

API keys for OpenAI / Anthropic are read from `OPENAI_API_KEY` /
`ANTHROPIC_API_KEY` if not passed explicitly.
