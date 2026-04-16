# Testing

Ennoia ships an `ennoia.testing` package so downstream suites can
unit-test schemas, pipelines, and agent logic without an LLM backend,
an embedding model, or a real store.

Everything in `ennoia.testing` depends only on the ennoia core —
`pip install ennoia[dev]` is enough, no Ollama / OpenAI /
sentence-transformers install required.

## Mocks

### `MockLLMAdapter`

Scripted, deterministic `LLMAdapter` subclass. `json_responses` and
`text_responses` can each be:

- A **mapping** from a substring of the prompt to the response. The
  longest matching key wins (specific > generic).
- A **callable** `(prompt: str) -> response` for dynamic logic.
- A **list**, consumed in order (one response per call).

Unmatched prompts raise `AssertionError` so miswired tests fail loudly.
Every call is recorded in `json_calls` / `text_calls` for assertions.

```python
from ennoia.testing import MockLLMAdapter

llm = MockLLMAdapter(
    json_responses={
        "Extract case metadata": {
            "jurisdiction": "WA",
            "date_decided": "2023-03-15",
            "_confidence": 0.95,
        }
    },
    text_responses={
        "What is the core legal holding": "The court held that...",
    },
)
```

### `MockEmbeddingAdapter`

Deterministic hash-seeded embeddings. Each call returns a unit-norm
vector of length `dim` derived from SHA-256 of the input text —
identical input yields byte-identical output across processes and
platforms. Similarity search in tests produces meaningful (if not
semantically faithful) ordering.

```python
from ennoia.testing import MockEmbeddingAdapter

embedding = MockEmbeddingAdapter(dim=8)
```

### `MockStore`

In-memory `HybridStore` implementation. Implements the full contract —
`upsert`, `hybrid_search`, `get`, `filter`, `delete` — so a pipeline can
index, filter, search, and retrieve end-to-end against it in a single
test. Filters run through the canonical `ennoia.utils.filters.apply_filters`
evaluator, so behaviour stays identical to the real stores.

```python
from ennoia.testing import MockStore

store = MockStore()
```

## Pytest fixtures

`ennoia.testing.fixtures` is exposed through the `pytest11` entry point
in `pyproject.toml`. Installing `ennoia` registers the fixtures
automatically — no `conftest.py` imports needed.

| Fixture | Returns |
|---|---|
| `mock_store` | A fresh `MockStore`. |
| `mock_llm` | A `MockLLMAdapter` with no scripted responses. |
| `mock_embedding` | An 8-dim `MockEmbeddingAdapter`. |

## End-to-end example

A user test that indexes two documents, filters them, and verifies the
scored order:

```python
from datetime import date
from typing import Literal

import pytest

from ennoia import BaseSemantic, BaseStructure, Pipeline


class DocMeta(BaseStructure):
    """Extract basic document metadata."""

    category: Literal["legal", "medical"]
    doc_date: date


class Summary(BaseSemantic):
    """What is the main topic of this document?"""


@pytest.mark.asyncio
async def test_hybrid_search(mock_store, mock_llm, mock_embedding):
    mock_llm._json = {
        "Extract basic document metadata": {
            "category": "legal",
            "doc_date": "2023-03-15",
            "_confidence": 0.95,
        }
    }
    mock_llm._text = {
        "What is the main topic": "Employer duty to accommodate disability.",
    }

    pipeline = Pipeline(
        schemas=[DocMeta, Summary],
        store=mock_store,
        llm=mock_llm,
        embedding=mock_embedding,
    )

    await pipeline.aindex(
        text="The court held that employers must accommodate disabilities.",
        source_id="doc_001",
    )

    result = await pipeline.asearch(
        query="accommodation",
        filters={"category": "legal"},
        top_k=5,
    )
    assert [hit.source_id for hit in result.hits] == ["doc_001"]
```

Because `MockEmbeddingAdapter` is deterministic and `MockStore` is
in-memory, the assertion above is stable across runs and platforms.

## Unit-testing schemas without a pipeline

For pure schema / `extend()` logic, instantiate the Pydantic model
directly — no pipeline required:

```python
from mymodule.schemas import CaseDocument


def test_extend_branches_on_jurisdiction():
    instance = CaseDocument(jurisdiction="WA")
    assert [c.__name__ for c in instance.extend()] == ["WashingtonDetails"]
```

Schemas are plain `BaseModel`s — all the usual Pydantic testing
techniques apply.
