# Ennoia

[![CI](https://github.com/ennoia-ai/ennoia/actions/workflows/ci.yml/badge.svg)](https://github.com/ennoia-ai/ennoia/actions/workflows/ci.yml)
[![coverage](https://codecov.io/gh/ennoia-ai/ennoia/branch/main/graph/badge.svg)](https://codecov.io/gh/ennoia-ai/ennoia)
[![PyPI](https://img.shields.io/pypi/v/ennoia.svg)](https://pypi.org/project/ennoia/)
[![Python](https://img.shields.io/pypi/pyversions/ennoia.svg)](https://pypi.org/project/ennoia/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE.txt)
[![types: pyright strict](https://img.shields.io/badge/types-pyright%20strict-informational.svg)](https://microsoft.github.io/pyright/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A framework for LLM-powered document pre-indexing and hybrid retrieval.

Ennoia treats indexing as a first-class problem. Instead of embedding raw
text and hoping vector similarity recovers relevance, you declare
extraction schemas (Pydantic models for structured metadata, marker
classes for semantic summaries), and Ennoia runs an LLM-driven DAG over
each document to produce rich, filterable indices. At query time,
retrieval runs in two phases: structured filters narrow the candidate set,
then vector search ranks within it.

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


class DocMeta(BaseStructure):
    """Extract basic document metadata."""

    category: Literal["legal", "medical", "financial"]
    doc_date: date


class Summary(BaseSemantic):
    """What is the main topic of this document?"""


pipeline = Pipeline(
    schemas=[DocMeta, Summary],
    store=Store(vector=InMemoryVectorStore(), structured=InMemoryStructuredStore()),
    llm=OllamaAdapter(model="qwen3:0.6b"),
    embedding=SentenceTransformerEmbedding(model="all-MiniLM-L6-v2"),
)

pipeline.index(text="The court held that...", source_id="doc_001")
results = pipeline.search(
    query="court holdings on liability",
    filters={"category": "legal"},
    top_k=5,
)
```

See [docs/quickstart.md](docs/quickstart.md) for the full walkthrough.

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

See [docs/cli.md](docs/cli.md).

## Documentation

- [Concepts](docs/concepts.md)
- [Quickstart](docs/quickstart.md)
- [Schema authoring](docs/schemas.md)
- [Filter language](docs/filters.md)
- [CLI reference](docs/cli.md)
- [Adapters](docs/adapters.md)
- [Stores](docs/stores.md)

## License

Apache 2.0. See [LICENSE.txt](LICENSE.txt) and [NOTICE](NOTICE).
