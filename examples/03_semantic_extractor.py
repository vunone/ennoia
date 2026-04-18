"""Recipe 03 — semantic extractors.

``BaseSemantic`` is Ennoia's unit of question-shaped extraction. Each
subclass declares one question via its docstring; the LLM answers the
question against the document, the answer is embedded, and each answer
becomes its own row in the vector store.

You can attach as many semantic extractors as you want to a pipeline. Each
answers its own question on the same document, independently, and shows
up as a separate entry in a search hit's ``semantic`` payload.

Run it::

    python examples/03_semantic_extractor.py
"""

from __future__ import annotations

from _data import AGREEMENT_SOURCE_ID, AGREEMENT_TEXT, make_embedding, make_llm

from ennoia import BaseSemantic, Pipeline, Store
from ennoia.store import InMemoryStructuredStore, InMemoryVectorStore


class ScopeOfServices(BaseSemantic):
    """What services does the provider deliver under this agreement?"""


class PaymentObligations(BaseSemantic):
    """Describe the payment schedule, invoice cadence, and any late-fee terms."""


class GoverningLawSummary(BaseSemantic):
    """Which jurisdiction governs the agreement and where must disputes be heard?"""


def main() -> None:
    pipeline = Pipeline(
        schemas=[ScopeOfServices, PaymentObligations, GoverningLawSummary],
        store=Store(
            vector=InMemoryVectorStore(),
            structured=InMemoryStructuredStore(),
        ),
        llm=make_llm(),
        embedding=make_embedding(),
    )

    result = pipeline.index(text=AGREEMENT_TEXT, source_id=AGREEMENT_SOURCE_ID)

    print("Answers extracted from the agreement:")
    for name, answer in result.semantic.items():
        confidence = result.confidences.get(name, 1.0)
        print(f"\n  [{name}] (confidence {confidence:.2f})")
        print(f"    {answer}")

    # ``index=`` narrows the vector search to one semantic extractor at a
    # time. Useful when multiple extractors answer questions on the same
    # document and you want hits from only one of them.
    print("\nRanking by similarity against the payment-obligations index:")
    hits = pipeline.search(
        query="invoice cadence and late fees",
        top_k=1,
        index="PaymentObligations",
    )
    for hit in hits:
        print(f"  [{hit.source_id}] score={hit.score:.3f}")
        print(f"    {hit.semantic.get('PaymentObligations', '')}")


if __name__ == "__main__":
    main()
