# Quickstart

Five-minute walkthrough: install Ennoia, declare a schema, index a
document, run a hybrid search. Uses a local Ollama instance and
sentence-transformers — no keys, no containers.

## 1. Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) running locally with a small model pulled:

  ```bash
  ollama pull qwen3:0.6b
  ```

## 2. Install

```bash
pip install "ennoia[ollama,sentence-transformers]"
```

## 3. Declare schemas + index

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

pipeline.index(
    text="The court held that employers must accommodate disabilities.",
    source_id="doc_001",
)
```

## 4. Search

```python
results = pipeline.search(
    query="court holdings on liability",
    filters={"category": "legal"},
    top_k=5,
)
for hit in results:
    print(hit.source_id, hit.score, hit.structural)
```

## Next steps

- [Schema authoring](schemas.md) — `extend()`, operator overrides,
  confidence.
- [Filter language](filters.md) — operator table, inference, validation.
- [CLI](cli.md) — index folders of files without writing Python.
- [Stores](stores.md) — swap the in-memory stores for SQLite or
  filesystem-backed Parquet + NumPy.
