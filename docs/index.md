# Ennoia

Ennoia introduces **Declarative Document Indexing Schemas (DDI Schemas)**
for RAG — a new pre-indexing approach where LLM-powered extraction is
defined through schemas and executed *before* documents enter any store,
replacing naive chunk-and-embed with structured, queryable indices.

> Traditional RAG is like feeding your documents through a shredder and
> then trying to answer questions by pulling out strips of paper one by
> one.
>
> Ennoia is like reading each document first, taking structured notes on
> what matters, and then searching your notes — while keeping the
> originals on the shelf.

## Install

```bash
pip install "ennoia[ollama,sentence-transformers,cli]"
```

Available extras: `ollama`, `openai`, `anthropic`, `sentence-transformers`,
`filesystem` (Parquet + NumPy stores), `cli` (`ennoia` CLI), `qdrant`,
`pgvector`, `server` (REST + MCP), `all` (everything above).

## Where to go next

- [Quickstart](quickstart.md) — five-minute SDK walkthrough.
- [Concepts](concepts.md) — why pre-indexing, how the two-phase flow works.
- [Schema authoring](schemas.md) — `BaseStructure`, `BaseSemantic`,
  `extend()`, the superschema manifest.
- [Filter language](filters.md) — the unified operator table shared by
  SDK, CLI, REST, and MCP.
- [CLI](cli.md) — `ennoia try | index | search | api | mcp`.
- [Adapters](adapters.md) — built-in LLM and embedding backends plus
  the ABCs for custom ones.
- [Stores](stores.md) — in-memory, SQLite, filesystem, Qdrant,
  pgvector.
- [Servers](serve.md) — run Ennoia as a REST API or MCP tool server.
- [Testing](testing.md) — `ennoia.testing` mocks and pytest fixtures.
- [API reference](api-reference.md) — rendered from docstrings.
- [Cookbook: MCP agent loop](cookbook/mcp-agent.md) — a worked agent
  discover → filter → search → retrieve example.
- [Cookbook: Custom adapters and stores](cookbook/custom-adapter.md) —
  implementation patterns for your own LLM, embedding, or hybrid store.

## License

Apache 2.0. See [LICENSE.txt](https://github.com/vunone/ennoia/blob/main/LICENSE.txt)
and [NOTICE](https://github.com/vunone/ennoia/blob/main/NOTICE).
