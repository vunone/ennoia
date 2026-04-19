# Ennoia

[![CI](https://github.com/vunone/ennoia/actions/workflows/ci.yml/badge.svg)](https://github.com/vunone/ennoia/actions/workflows/ci.yml)
[![coverage](https://codecov.io/gh/vunone/ennoia/branch/main/graph/badge.svg)](https://codecov.io/gh/vunone/ennoia)
[![PyPI](https://img.shields.io/pypi/v/ennoia.svg)](https://pypi.org/project/ennoia/)
[![Python](https://img.shields.io/pypi/pyversions/ennoia.svg)](https://pypi.org/project/ennoia/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/vunone/ennoia/blob/main/LICENSE.txt)
[![types: pyright strict](https://img.shields.io/badge/types-pyright%20strict-informational.svg)](https://microsoft.github.io/pyright/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Ennoia is a Python library for indexing documents with LLM-extracted structure. You declare typed **extractors** — Python classes that describe what to pull out of each document — and Ennoia runs the extraction, persists the results, and gives you hybrid filter + vector search on the output.

**Status:** alpha (`0.3.x`). API is converging but may still shift between minor versions. Benchmarks: TBD.

## Install

```bash
pip install "ennoia[ollama,sentence-transformers]"
```

The quickstart below needs nothing else. Other extras plug in when you need them:

| Extra | Adds |
|---|---|
| `openai`, `anthropic`, `openrouter` | Hosted LLM adapters |
| `ollama` | Local LLM adapter |
| `sentence-transformers` | Local embedding adapter |
| `filesystem` | Parquet + NumPy persistent stores |
| `qdrant` | Qdrant vector + hybrid store |
| `pgvector` | PostgreSQL + pgvector hybrid store |
| `cli` | `ennoia` command |
| `server` | FastAPI REST + FastMCP |
| `all` | Everything above |

## Showcase

Three extractor kinds, one `extend()` branch, a hybrid search — runnable end-to-end with a local Ollama:

```python
from datetime import date
from typing import Annotated, ClassVar, Literal

from ennoia import (
    BaseCollection, BaseSemantic, BaseStructure,
    Field, Pipeline, Store,
)
from ennoia.adapters.embedding.sentence_transformers import SentenceTransformerEmbedding
from ennoia.adapters.llm.ollama import OllamaAdapter
from ennoia.store import InMemoryStructuredStore, InMemoryVectorStore


class PaymentTerms(BaseStructure):
    """Extract the payment terms of a high-value contract."""

    net_days: Annotated[int, Field(description="Net payment window in days.")]
    late_fee_pct: Annotated[float, Field(description="Late fee as a decimal, e.g. 0.015 for 1.5%.")]


class ContractMeta(BaseStructure):
    """Extract the header of a master services agreement."""

    governing_law: Annotated[
        Literal["Delaware", "New York", "California"],
        Field(description="US state whose law governs the agreement."),
    ]
    effective_date: Annotated[date, Field(description="Date the agreement takes effect.")]
    contract_value: Annotated[
        int, Field(description="Total fees in USD across the initial term; 0 if not stated."),
    ]

    def extend(self) -> list[type[BaseStructure]]:
        # High-value agreements branch into a deeper extractor.
        return [PaymentTerms] if self.contract_value >= 100_000 else []

    class Schema(BaseStructure.Schema):
        extensions: ClassVar[list[type]] = [PaymentTerms]


class ContractSummary(BaseSemantic):
    """What services does the provider deliver under this agreement?"""


class Party(BaseCollection):
    """Extract every party named in the agreement."""

    legal_name: Annotated[str, Field(description="Full legal name of the party.")]
    role: Annotated[
        Literal["provider", "client"],
        Field(description="Whether this party provides or receives the services."),
    ]

    def get_unique(self) -> str:
        return self.legal_name.casefold()

    def template(self) -> str:
        return f"{self.legal_name} ({self.role})"


pipeline = Pipeline(
    schemas=[ContractMeta, ContractSummary, Party],
    store=Store(vector=InMemoryVectorStore(), structured=InMemoryStructuredStore()),
    llm=OllamaAdapter(model="qwen3:0.6b"),
    embedding=SentenceTransformerEmbedding(model="all-MiniLM-L6-v2"),
)

pipeline.index(text=AGREEMENT_TEXT, source_id="msa-acme-globex")

hits = pipeline.search(
    query="payment obligations and late fees",
    filters={"governing_law": "Delaware"},
    top_k=3,
)
```

Two patterns to note:

- **Class docstring is the extractor-level prompt.** The LLM sees `ContractMeta.__doc__` as the task description.
- **`Field(description=...)` is the per-field prompt.** Field descriptions ride on the JSON schema the extractor sends to the LLM, which makes every field's intent visible next to its type.

A runnable version of this script lives at [examples/04_extend_branching.py](https://github.com/vunone/ennoia/blob/main/examples/04_extend_branching.py). For a smaller first step, [examples/01_getting_started.py](https://github.com/vunone/ennoia/blob/main/examples/01_getting_started.py) uses one structural and one semantic extractor only.

## When to reach for it

- You need **hybrid filter + vector search** on the same corpus — not one or the other.
- Your documents carry **structure** (dates, parties, categories, clauses) that plain chunking discards.
- You want the **LLM pre-processing step to be typed and visible in your diff**, not buried in a prompt string at runtime.

## CLI

```bash
# Index a folder into a filesystem-backed store.
ennoia index ./contracts \
  --schema my_extractors.py \
  --store ./contracts_index \
  --collection contracts \
  --llm ollama:qwen3:0.6b \
  --embedding sentence-transformers:all-MiniLM-L6-v2

# Hybrid search.
ennoia search "payment obligations and late fees" \
  --schema my_extractors.py \
  --store ./contracts_index \
  --collection contracts \
  --filter "governing_law=Delaware" \
  --filter "effective_date__gte=2025-01-01" \
  --top-k 5
```

`ennoia try <document> --schema <file>` runs a single extraction without writing to any store — useful while iterating on extractors. `ennoia api` and `ennoia mcp` serve an existing index over REST or MCP respectively.

## Documentation

- [Getting started](https://github.com/vunone/ennoia/blob/main/docs/quickstart.md) — 10-minute walkthrough
- [Concepts](https://github.com/vunone/ennoia/blob/main/docs/concepts.md) — extractor kinds, the extraction DAG, two-phase retrieval
- [Extractors](https://github.com/vunone/ennoia/blob/main/docs/extractors.md) — authoring reference
- [Filters](https://github.com/vunone/ennoia/blob/main/docs/filters.md) — filter language
- [Stores](https://github.com/vunone/ennoia/blob/main/docs/stores.md) — in-memory, filesystem, Qdrant, pgvector
- [Adapters](https://github.com/vunone/ennoia/blob/main/docs/adapters.md) — LLM + embedding backends
- [CLI](https://github.com/vunone/ennoia/blob/main/docs/cli.md) — command reference
- [Serve](https://github.com/vunone/ennoia/blob/main/docs/serve.md) — REST + MCP
- [Testing](https://github.com/vunone/ennoia/blob/main/docs/testing.md) — mocks and fixtures
- [Runnable examples](https://github.com/vunone/ennoia/tree/main/examples) — every concept as a standalone script

## License

Apache 2.0. See [LICENSE.txt](https://github.com/vunone/ennoia/blob/main/LICENSE.txt) and [NOTICE](https://github.com/vunone/ennoia/blob/main/NOTICE).

## Contributors

- [Maks P.](https://maks.vun.one)

Sponsored by: [Achiv :: Smart Market Research](https://achiv.com)
