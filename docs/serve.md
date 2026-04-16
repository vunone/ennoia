# Servers — REST + MCP

Ennoia ships two remote interfaces. Both are built once at startup from a
shared `ServerContext` (a configured `Pipeline` + an `AuthHook`) and
delegate every request back to the pipeline, so the REST, MCP, CLI, and
SDK surfaces all speak the same filter language and return the same
payload shapes.

- **REST (`ennoia api`)** — full CRUD against the index. FastAPI + uvicorn.
- **MCP (`ennoia mcp`)** — read-only tools for agents. FastMCP.

Install the `server` extra:

```bash
pip install "ennoia[server,ollama,sentence-transformers,filesystem]"
```

## `ennoia api` — REST server

Boot a FastAPI server against any supported store. Every route sits
behind a bearer-token auth hook by default.

```bash
ennoia api \
  --store ./my_index \
  --collection cases \
  --schema my_schemas.py \
  --host 127.0.0.1 \
  --port 8080 \
  --llm ollama:qwen3:0.6b \
  --embedding sentence-transformers:all-MiniLM-L6-v2 \
  --api-key "$ENNOIA_API_KEY"
```

| Flag | Default | Purpose |
|---|---|---|
| `--store` | required | Store spec — path, `qdrant:<collection>`, or `pgvector:<collection>`. See [CLI — Store backends](cli.md#store-backends). |
| `--collection` | `documents` | Collection name for filesystem stores; ignored for hybrid backends (collection is in `--store`). |
| `--qdrant-url` | `$ENNOIA_QDRANT_URL` | Required when `--store qdrant:...`. |
| `--qdrant-api-key` | `$ENNOIA_QDRANT_API_KEY` | Optional. |
| `--pg-dsn` | `$ENNOIA_PG_DSN` | Required when `--store pgvector:...`. |
| `--schema` | required | Python module declaring the same schemas used at index time. |
| `--host` | `127.0.0.1` | Bind address. |
| `--port` | `8080` | Listen port. |
| `--llm` | `ollama:qwen3:0.6b` | LLM adapter URI (see [CLI — Adapter URIs](cli.md#adapter-uris)). |
| `--embedding` | `sentence-transformers:all-MiniLM-L6-v2` | Embedding adapter URI. |
| `--api-key` | `$ENNOIA_API_KEY` | Expected bearer token. Required unless `--no-auth` is passed. |
| `--no-auth` | off | Accept unauthenticated requests. For local development only. |

### Endpoints

| Method + path | Purpose |
|---|---|
| `GET /discover` | Returns the superschema discovery payload — same shape as `ennoia.describe(schemas)` and the MCP `discover_schema` tool. |
| `POST /filter` | Body: `{"filters": {...}}` (or the filter dict at the top level). Returns `{"ids": [...]}`. |
| `POST /search` | Body: `{"query", "filters", "filter_ids", "index", "top_k"}`. Returns `{"hits": [...]}`. |
| `GET /retrieve/{source_id}` | Full structured record for one `source_id`, or 404. |
| `POST /index` | Body: `{"text", "source_id"}`. Runs the pipeline and returns `{"source_id", "rejected", "schemas_extracted"}`. |
| `DELETE /delete/{source_id}` | Remove every trace of `source_id`. Returns `{"removed": bool}`. |

Filter validation failures return HTTP 422 with the error body described
in [Filter validation errors](filters.md#validation-errors).

### curl example

```bash
# Discover available fields and indices.
curl -s http://127.0.0.1:8080/discover \
  -H "Authorization: Bearer $ENNOIA_API_KEY"

# Two-phase agent flow: filter → search restricted to filtered ids.
IDS=$(curl -s -X POST http://127.0.0.1:8080/filter \
  -H "Authorization: Bearer $ENNOIA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"jurisdiction": "WA", "date_decided__gte": "2020-01-01"}' \
  | jq -r '.ids | @json')

curl -s -X POST http://127.0.0.1:8080/search \
  -H "Authorization: Bearer $ENNOIA_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"employer duty to accommodate disability\",
       \"filter_ids\": $IDS,
       \"top_k\": 5}"
```

## `ennoia mcp` — MCP tool server

Read-only surface. Agents can discover schemas, filter, search, and
retrieve — but cannot index or delete. Supported transports:

- `sse` — Server-Sent Events. Default for remote agents.
- `http` — streamable-HTTP transport.
- `stdio` — JSON over stdin/stdout for local agents.

```bash
ennoia mcp \
  --store ./my_index \
  --schema my_schemas.py \
  --transport sse \
  --host 127.0.0.1 \
  --port 8090 \
  --llm ollama:qwen3:0.6b \
  --embedding sentence-transformers:all-MiniLM-L6-v2 \
  --api-key "$ENNOIA_API_KEY"
```

Flags mirror `ennoia api` — `--store`, `--collection`, `--qdrant-url`,
`--qdrant-api-key`, `--pg-dsn`, `--schema`, `--host`, `--port`, `--llm`,
`--embedding`, `--api-key`, `--no-auth` — plus `--transport`. Default
port is `8090`.

### Tools

| Tool | Signature | Purpose |
|---|---|---|
| `discover_schema()` | `() -> dict` | Unified discovery payload — structural fields, semantic indices. |
| `filter(filters)` | `(filters: dict \| None) -> list[str]` | Source ids matching a structured filter. Pass the list back into `search` via `filter_ids`. |
| `search(query, top_k, filter_ids, index)` | `(query: str, top_k=5, filter_ids=None, index=None) -> list[dict]` | Vector search. `filter_ids` restricts to a pre-filtered set; `index` picks a single semantic schema. |
| `retrieve(id)` | `(id: str) -> dict \| None` | Full structured record for a `source_id`. |

The canonical agent flow (per `.ref/USAGE.md §5`) is **discover → filter
→ search(filter_ids=…) → retrieve**. See the [MCP agent cookbook](cookbook/mcp-agent.md)
for a full worked example.

### Python client example

```python
import asyncio
import os

from fastmcp import Client


async def main() -> None:
    headers = {"Authorization": f"Bearer {os.environ['ENNOIA_API_KEY']}"}
    async with Client(
        "http://127.0.0.1:8090/sse",
        transport="sse",
        headers=headers,
    ) as client:
        schema = await client.call_tool("discover_schema", {})
        print(schema)

        ids = await client.call_tool(
            "filter",
            {"filters": {"jurisdiction": "WA", "date_decided__gte": "2020-01-01"}},
        )

        hits = await client.call_tool(
            "search",
            {
                "query": "employer duty to accommodate disability",
                "filter_ids": ids,
                "top_k": 5,
            },
        )
        for hit in hits:
            print(hit["source_id"], hit["score"])


asyncio.run(main())
```

## Authentication

Both servers share the same `AuthHook` protocol:

```python
class AuthHook(Protocol):
    async def __call__(self, token: str | None) -> bool: ...
```

Built-in hooks live in `ennoia.server`:

- `static_bearer_auth(api_key)` — compare the bearer token against a
  fixed string.
- `env_bearer_auth(var_name="ENNOIA_API_KEY")` — read the expected token
  from an env var; returns `None` when unset so callers can decide
  whether to fail-closed or fall back to `no_auth`.
- `no_auth()` — accept every request. Local dev only; CLI opt-in is
  `--no-auth`.

The CLI helpers (`ennoia api` / `ennoia mcp`) refuse to start without
either `--api-key` (or the `ENNOIA_API_KEY` env var) **or** an explicit
`--no-auth`, so you never accidentally expose an unauthenticated server.

### Custom auth

For OAuth, JWT, per-tenant policies, or anything else, build your own
`ServerContext` and call `create_app` / `create_mcp` directly:

```python
from ennoia import Pipeline, Store
from ennoia.adapters.embedding.sentence_transformers import SentenceTransformerEmbedding
from ennoia.adapters.llm.ollama import OllamaAdapter
from ennoia.server import ServerContext
from ennoia.server.api import create_app


async def my_auth(token: str | None) -> bool:
    # Validate against your IdP, check scopes, etc.
    return await my_idp.verify(token)


pipeline = Pipeline(
    schemas=[...],
    store=Store.from_path("./my_index"),
    llm=OllamaAdapter(model="qwen3:0.6b"),
    embedding=SentenceTransformerEmbedding(model="all-MiniLM-L6-v2"),
)
app = create_app(ServerContext(pipeline=pipeline, auth=my_auth))
# hand `app` to uvicorn, gunicorn, or any ASGI runner.
```

## Observability

Events emitted by the pipeline (`ExtractionEvent`, `IndexEvent`,
`SearchEvent`) flow through the same `Emitter` regardless of surface —
remote calls are instrumented for free. See [Concepts — Observability](concepts.md#observability).
