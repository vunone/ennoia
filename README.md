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
`filesystem` (Parquet + NumPy stores), `cli` (`ennoia` CLI), `all`
(everything above).

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


# Configure the pipeline: schemas + a two-phase store (structured filter
# → vector search) + LLM and embedding adapters.
pipeline = Pipeline(
    schemas=[DocMeta, Summary],
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
  --llm ollama:qwen3:0.6b \
  --embedding sentence-transformers:all-MiniLM-L6-v2

# Hybrid search
ennoia search "employer duty to accommodate disability" \
  --schema my_schemas.py \
  --store ./my_index \
  --filter "jurisdiction=WA" \
  --filter "date_decided__gte=2020-01-01" \
  --top-k 5
```

See [docs/cli.md](https://github.com/vunone/ennoia/blob/main/docs/cli.md).

## Documentation

- [Concepts](https://github.com/vunone/ennoia/blob/main/docs/concepts.md)
- [Quickstart](https://github.com/vunone/ennoia/blob/main/docs/quickstart.md)
- [Schema authoring](https://github.com/vunone/ennoia/blob/main/docs/schemas.md)
- [Filter language](https://github.com/vunone/ennoia/blob/main/docs/filters.md)
- [CLI reference](https://github.com/vunone/ennoia/blob/main/docs/cli.md)
- [Adapters](https://github.com/vunone/ennoia/blob/main/docs/adapters.md)
- [Stores](https://github.com/vunone/ennoia/blob/main/docs/stores.md)

## License

Apache 2.0. See [LICENSE.txt](https://github.com/vunone/ennoia/blob/main/LICENSE.txt) and [NOTICE](https://github.com/vunone/ennoia/blob/main/NOTICE).
