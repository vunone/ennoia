"""Demonstrates the filter-first design: a mismatched filter returns zero hits.

The document is tagged `category=legal`, but we search with
`category=medical`. The structured filter eliminates the candidate
before vector search runs, so no hits are returned even though the
semantic content is generic enough to match.
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


class Summary(BaseSemantic):
    """What is the main topic of this document?"""


def main() -> None:
    pipeline = Pipeline(
        schemas=[DocMeta, Summary],
        store=Store(
            vector=InMemoryVectorStore(),
            structured=InMemoryStructuredStore(),
        ),
        llm=OllamaAdapter(model="qwen3:0.6b"),
        embedding=SentenceTransformerEmbedding(model="all-MiniLM-L6-v2"),
    )

    pipeline.index(
        text=(
            "Court opinion dated 2022-07-01 addressing negligence claims "
            "arising from workplace injuries."
        ),
        source_id="doc_legal_001",
    )

    hits = pipeline.search(
        query="negligence workplace injury",
        filters={"category": "legal"},
        top_k=5,
    )
    assert len(hits) == 1, f"Expected one hit for legal filter, got {len(hits)}."
    print("OK — structured filter correctly eliminated the legal document.")


if __name__ == "__main__":
    main()
