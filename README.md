# Ennoia

[![CI](https://github.com/vunone/ennoia/actions/workflows/ci.yml/badge.svg)](https://github.com/vunone/ennoia/actions/workflows/ci.yml)
[![coverage](https://codecov.io/gh/vunone/ennoia/branch/main/graph/badge.svg)](https://codecov.io/gh/vunone/ennoia)
[![PyPI](https://img.shields.io/pypi/v/ennoia.svg)](https://pypi.org/project/ennoia/)
[![Python](https://img.shields.io/pypi/pyversions/ennoia.svg)](https://pypi.org/project/ennoia/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/vunone/ennoia/blob/main/LICENSE.txt)
[![types: pyright strict](https://img.shields.io/badge/types-pyright%20strict-informational.svg)](https://microsoft.github.io/pyright/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

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
`filesystem` (Parquet + NumPy stores), `cli` (`ennoia` CLI),
`qdrant` (Qdrant vector + hybrid stores),
`pgvector` (PostgreSQL + pgvector hybrid store),
`server` (FastAPI REST + FastMCP),
`docs` (mkdocs-material site),
`benchmark` (CUAD comparison harness — see [benchmark/README.md](benchmark/README.md)),
`all` (everything above).

## Quick start (SDK)

```python
from datetime import date
from typing import Literal

from ennoia import BaseSemantic, BaseStructure, Pipeline, Store
from ennoia.adapters.embedding.sentence_transformers import SentenceTransformerEmbedding
from ennoia.adapters.llm.ollama import OllamaAdapter
from ennoia.store import InMemoryStructuredStore, InMemoryVectorStore


# DDI Schema #1 — structured extraction. Field types drive filter
# operators automatically (Literal → eq/in, date → range ops); the
# docstring is the LLM prompt.
class DocMeta(BaseStructure):
    """Extract basic document metadata."""

    category: Literal["legal", "medical", "financial"]
    doc_date: date


# DDI Schema #2 — semantic extraction. The docstring is the question the
# LLM answers; the answer is embedded for vector search.
class Summary(BaseSemantic):
    """What is the main topic of this document?"""


# DDI Schema #3 — collection extraction. The LLM iterates until it has
# captured every entity; each entity is embedded as its own searchable row.
from ennoia import BaseCollection


class Party(BaseCollection):
    """Extract every party mentioned in the document."""

    company_name: str
    participation_year: int

    def template(self) -> str:
        return f"{self.company_name} ({self.participation_year})"


# Configure the pipeline: schemas + a two-phase store (structured filter
# → vector search) + LLM and embedding adapters.
pipeline = Pipeline(
    schemas=[DocMeta, Summary, Party],
    store=Store(vector=InMemoryVectorStore(), structured=InMemoryStructuredStore()),
    llm=OllamaAdapter(model="qwen3:0.6b"),
    embedding=SentenceTransformerEmbedding(model="all-MiniLM-L6-v2"),
)

# Pre-indexing: every schema runs against the document once, before writing
# structured fields to the structured store and embedded answers to the
# vector store — before any query touches them.
pipeline.index(text="The court held that...", source_id="doc_001")

# Hybrid search: `filters` narrows candidates via the structured store,
# then vector similarity ranks within that subset.
results = pipeline.search(
    query="court holdings on liability",
    filters={"category": "legal"},
    top_k=5,
)
```

See [docs/quickstart.md](https://github.com/vunone/ennoia/blob/main/docs/quickstart.md) for the full walkthrough.

## Quick start (CLI)

```bash
# Iterate on a schema against a single document
ennoia try ./sample.txt --schema my_schemas.py

# Index a folder into a filesystem-backed store
ennoia index ./docs \
  --schema my_schemas.py \
  --store ./my_index \
  --collection cases \
  --llm ollama:qwen3:0.6b \
  --embedding sentence-transformers:all-MiniLM-L6-v2

# …or into a production Qdrant / pgvector backend
ennoia index ./docs \
  --schema my_schemas.py \
  --store qdrant:cases \
  --qdrant-url http://localhost:6333 \
  --llm openai:gpt-4o-mini \
  --embedding openai-embedding:text-embedding-3-small

# Hybrid search
ennoia search "employer duty to accommodate disability" \
  --schema my_schemas.py \
  --store ./my_index \
  --collection cases \
  --filter "jurisdiction=WA" \
  --filter "date_decided__gte=2020-01-01" \
  --top-k 5
```

See [docs/cli.md](https://github.com/vunone/ennoia/blob/main/docs/cli.md).

## Serve an index (REST + MCP)

Stage 3 ships two remote interfaces. Both accept the same `--store`
prefix scheme (filesystem path, `qdrant:<collection>`, or
`pgvector:<collection>`) as `ennoia index`:

```bash
# REST — full CRUD for application integration.
export ENNOIA_API_KEY=sekret
ennoia api --store ./my_index --schema my_schemas.py --port 8080

# MCP — read-only tools (discover_schema, filter, search, retrieve) for agents,
# pointed at a production Qdrant collection.
export ENNOIA_QDRANT_URL=http://localhost:6333
ennoia mcp --store qdrant:cases --schema my_schemas.py --transport sse --port 8090
```

Agents consume the MCP flow `discover_schema → filter → search(filter_ids=...) → retrieve`
out of the box. See [docs/serve.md](https://github.com/vunone/ennoia/blob/main/docs/serve.md).

## Benchmarks

A reproducible CUAD legal-QA benchmark pits ennoia DDI+RAG against a textbook
langchain shred-embed RAG baseline using identical models (`gpt-5.4-nano` for
generation, `text-embedding-3-small` for embeddings, `gpt-5.4` as judge):

![CUAD benchmark](benchmark/results/chart_latest.png)

See [benchmark/README.md](benchmark/README.md) for methodology, the one-command
reproduction, and the cookbook walkthrough at
[docs/cookbook/cuad-benchmark.md](https://github.com/vunone/ennoia/blob/main/docs/cookbook/cuad-benchmark.md).

## Documentation

- [Concepts](https://github.com/vunone/ennoia/blob/main/docs/concepts.md)
- [Quickstart](https://github.com/vunone/ennoia/blob/main/docs/quickstart.md)
- [Schema authoring](https://github.com/vunone/ennoia/blob/main/docs/schemas.md)
- [Filter language](https://github.com/vunone/ennoia/blob/main/docs/filters.md)
- [CLI reference](https://github.com/vunone/ennoia/blob/main/docs/cli.md)
- [Adapters](https://github.com/vunone/ennoia/blob/main/docs/adapters.md)
- [Stores](https://github.com/vunone/ennoia/blob/main/docs/stores.md)
- [Serve (REST + MCP)](https://github.com/vunone/ennoia/blob/main/docs/serve.md)
- [Testing utilities](https://github.com/vunone/ennoia/blob/main/docs/testing.md)

## License

Apache 2.0. See [LICENSE.txt](https://github.com/vunone/ennoia/blob/main/LICENSE.txt) and [NOTICE](https://github.com/vunone/ennoia/blob/main/NOTICE).
