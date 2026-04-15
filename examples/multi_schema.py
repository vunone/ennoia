"""Multiple structural + semantic schemas in one indexing pass.

Proves the pipeline walks more than one schema for a single document
without any inter-schema dependencies. Nested extractions (parent
schema queues a child via `extend()`) are exercised in
`tests/test_pipeline_with_fakes.py`.
"""

from __future__ import annotations

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


class Provenance(BaseStructure):
    """Extract where the document came from and who authored it."""

    author: str
    source: str


class Summary(BaseSemantic):
    """What is the main topic of this document?"""


class KeyQuestion(BaseSemantic):
    """What is the central legal or factual question this document answers?"""


def main() -> None:
    pipeline = Pipeline(
        schemas=[DocMeta, Provenance, Summary, KeyQuestion],
        store=Store(
            vector=InMemoryVectorStore(),
            structured=InMemoryStructuredStore(),
        ),
        llm=OllamaAdapter(model="qwen3:0.6b"),
        embedding=SentenceTransformerEmbedding(model="all-MiniLM-L6-v2"),
    )

    document = (
        "Memo by Justice A. Lin, published by the WA Court of Appeals on "
        "March 15, 2023. The court considered whether employers must "
        "accommodate commuting-related disabilities under the WLAD. The "
        "court held that such accommodation is required when reasonable."
    )

    result = pipeline.index(text=document, source_id="doc_042")
    print("Indexed:", result.summary())

    hits = pipeline.search(
        query="disability accommodation",
        filters={"category": "legal"},
        top_k=5,
    )
    print(f"\n{len(hits)} hit(s):")
    for hit in hits:
        print(f"  [{hit.source_id}] score={hit.score:.3f}")


if __name__ == "__main__":
    main()
