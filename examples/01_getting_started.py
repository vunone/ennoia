"""Recipe 01 — getting started.

A minimal end-to-end run: declare a structural and a semantic extractor,
index a single Master Services Agreement into in-memory stores, then run a
hybrid (filter + vector) search and print the hit.

Run it::

    python examples/01_getting_started.py

The script uses a local Ollama instance by default. Set ``OPENAI_API_KEY``
to switch to OpenAI transparently — see ``examples/README.md``.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Literal

from _data import AGREEMENT_SOURCE_ID, AGREEMENT_TEXT, make_embedding, make_llm

from ennoia import BaseSemantic, BaseStructure, Field, Pipeline, Store
from ennoia.store import InMemoryStructuredStore, InMemoryVectorStore

# --- Extractors ------------------------------------------------------------
#
# The class docstring is the extractor-level prompt the LLM receives. Each
# field carries its own ``description=`` which rides on the JSON schema, so
# the LLM sees per-field instructions alongside the type contract.


class ContractMeta(BaseStructure):
    """Extract the header of a master services agreement."""

    governing_law: Annotated[
        Literal["Delaware", "New York", "California"],
        Field(description="US state whose law governs the agreement."),
    ]
    effective_date: Annotated[
        date,
        Field(description="Date on which the agreement takes effect."),
    ]


class ContractSummary(BaseSemantic):
    """What services does the provider deliver under this agreement?"""


# --- Pipeline --------------------------------------------------------------


def main() -> None:
    pipeline = Pipeline(
        schemas=[ContractMeta, ContractSummary],
        store=Store(
            vector=InMemoryVectorStore(),
            structured=InMemoryStructuredStore(),
        ),
        llm=make_llm(),
        embedding=make_embedding(),
    )

    index_result = pipeline.index(text=AGREEMENT_TEXT, source_id=AGREEMENT_SOURCE_ID)
    print("Indexed:", index_result.summary())

    hits = pipeline.search(
        query="scope of services the provider delivers",
        filters={"governing_law": "Delaware"},
        top_k=3,
    )

    print(f"\n{len(hits)} hit(s):")
    for hit in hits:
        print(f"  [{hit.source_id}] score={hit.score:.3f}")
        for name, answer in hit.semantic.items():
            print(f"    {name}: {answer}")


if __name__ == "__main__":
    main()
