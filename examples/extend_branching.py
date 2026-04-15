"""extend() conditional branching — run a child schema only when parent
confidence clears a threshold.

The pipeline records ``_confidence`` on the parent instance (enabled by
``BaseStructure.model_config = ConfigDict(extra="allow")``), so
``extend()`` can branch on ``self._confidence`` without any extra
plumbing. Run with a local Ollama instance:

    ollama pull qwen3:0.6b
    python examples/extend_branching.py
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from ennoia import BaseSemantic, BaseStructure, Pipeline, Store
from ennoia.adapters.embedding.sentence_transformers import SentenceTransformerEmbedding
from ennoia.adapters.llm.ollama import OllamaAdapter
from ennoia.store import InMemoryStructuredStore, InMemoryVectorStore


class CaseMeta(BaseStructure):
    """Extract the jurisdiction and decision date from a case ruling."""

    jurisdiction: Literal["WA", "NY", "TX", "CA"]
    date_decided: date

    def extend(self) -> list[type[BaseStructure] | type[BaseSemantic]]:
        confidence = float(getattr(self, "_confidence", 0.0))
        if self.jurisdiction == "WA" and confidence >= 0.8:
            return [WashingtonAppellateDetail]
        return []


class WashingtonAppellateDetail(BaseStructure):
    """Extract the division number of the Washington appellate court."""

    division: Literal["I", "II", "III"]


def main() -> None:
    pipeline = Pipeline(
        schemas=[CaseMeta],
        store=Store(vector=InMemoryVectorStore(), structured=InMemoryStructuredStore()),
        llm=OllamaAdapter(model="qwen3:0.6b"),
        embedding=SentenceTransformerEmbedding(model="all-MiniLM-L6-v2"),
    )

    text = (
        "Division II of the Washington Court of Appeals, on March 15, 2023, "
        "reversed the trial court's judgment..."
    )
    result = pipeline.index(text=text, source_id="wa_case_001")

    print("Extracted schemas:", list(result.structural.keys()))
    print("Confidences:", result.confidences)
    if "WashingtonAppellateDetail" in result.structural:
        print("Branch fired — division:", result.structural["WashingtonAppellateDetail"])
    else:
        print("Branch did not fire (jurisdiction non-WA or confidence below threshold).")


if __name__ == "__main__":
    main()
