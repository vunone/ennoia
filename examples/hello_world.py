"""Hello World — the Stage 1 acceptance demo from .ref/USAGE.md §1.

Prerequisites:
    - Ollama running locally with a small model pulled:
        ollama pull qwen3:0.6b
    - Install with:
        pip install -e ".[ollama,sentence-transformers]"
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Literal

from ennoia import BaseSemantic, BaseStructure, Field, Pipeline, Store
from ennoia.adapters.embedding.sentence_transformers import SentenceTransformerEmbedding
from ennoia.adapters.llm.ollama import OllamaAdapter
from ennoia.store import InMemoryStructuredStore, InMemoryVectorStore


class DocMeta(BaseStructure):
    """Extract basic document metadata."""

    category: Annotated[
        Literal["legal", "medical", "financial", "other"],
        Field(
            description=("One of the given categories that relevant the most to the given document")
        ),
    ]
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

    document = (
        "On March 15, 2023, the Court of Appeals held that employers must "
        "provide reasonable accommodation for commuting-related disabilities "
        "under the Washington Law Against Discrimination. The court reversed "
        "the lower court's grant of summary judgment to the employer."
    )

    result = pipeline.index(text=document, source_id="doc_001")
    print("Indexed:", result.summary())

    results = pipeline.search(
        query="court holdings on employer liability",
        filters={"category": "legal"},
        top_k=5,
    )
    print(f"\n{len(results)} hit(s):")
    for hit in results:
        print(
            f"  [{hit.source_id}] score={hit.score:.3f} "
            f"structural={hit.structural} semantic={hit.semantic}"
        )


if __name__ == "__main__":
    main()
