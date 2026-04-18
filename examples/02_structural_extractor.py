"""Recipe 02 — structural extractor deep dive.

``BaseStructure`` is Ennoia's unit of structured extraction. The LLM is
asked to fill a typed schema once per document, and the result is the
source of truth for the structured store and any filter that runs against
it.

This recipe shows, on the same contract used across the examples:

- the ``Annotated[T, Field(description="...")]`` pattern — the LLM sees
  the ``description`` on every field;
- ``Field(operators=[...])`` to narrow which filter operators are exposed
  for a field;
- ``Field(filterable=False)`` to keep a field out of the filter contract
  entirely;
- how the per-extraction confidence surfaces on the index result.

Run it::

    python examples/02_structural_extractor.py
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Literal

from _data import AGREEMENT_SOURCE_ID, AGREEMENT_TEXT, make_embedding, make_llm

from ennoia import BaseStructure, Field, Pipeline, Store
from ennoia.store import InMemoryStructuredStore, InMemoryVectorStore


class ContractMeta(BaseStructure):
    """Extract the header of a master services agreement.

    The class docstring is the extractor-level prompt. The LLM receives it
    verbatim as the "Task" section of the extraction prompt, alongside the
    JSON schema generated from the fields below.
    """

    # ``Literal`` gives the LLM a closed set to choose from and lets Ennoia
    # infer ``eq`` / ``in`` as the available filter operators automatically.
    governing_law: Annotated[
        Literal["Delaware", "New York", "California"],
        Field(description="US state whose law governs the agreement."),
    ]

    # ``date`` auto-exposes range operators (``eq``, ``gt``, ``gte``, ``lt``,
    # ``lte``). No override needed.
    effective_date: Annotated[
        date,
        Field(description="Date on which the agreement takes effect."),
    ]

    # ``int`` would default to the full range-comparison set. Here we narrow
    # it to equality-only via ``operators=[...]`` — filter users will get a
    # ``FilterValidationError`` if they try ``contract_value__gte``.
    contract_value: Annotated[
        int,
        Field(
            description=(
                "Total fees across the initial term, in whole US dollars. "
                "Use 0 when the document does not state a value."
            ),
            operators=["eq"],
        ),
    ]

    # ``filterable=False`` excludes the field from the filter contract. It
    # still gets extracted and persisted — it just isn't queryable.
    counterparty_signatory: Annotated[
        str,
        Field(
            description="Full name of the person signing on behalf of the client.",
            filterable=False,
        ),
    ]


def main() -> None:
    pipeline = Pipeline(
        schemas=[ContractMeta],
        store=Store(
            vector=InMemoryVectorStore(),
            structured=InMemoryStructuredStore(),
        ),
        llm=make_llm(),
        embedding=make_embedding(),
    )

    result = pipeline.index(text=AGREEMENT_TEXT, source_id=AGREEMENT_SOURCE_ID)

    # ``result.structural`` is a ``dict[str, BaseStructure]``. Narrow the
    # base type with ``isinstance`` to regain the concrete field types.
    meta = result.structural["ContractMeta"]
    assert isinstance(meta, ContractMeta)

    print("Extracted ContractMeta:")
    print(f"  governing_law         = {meta.governing_law}")
    print(f"  effective_date        = {meta.effective_date}")
    print(f"  contract_value        = ${meta.contract_value:,}")
    print(f"  counterparty_signatory = {meta.counterparty_signatory}")
    print(f"  confidence            = {result.confidences['ContractMeta']:.2f}")

    # Filtering demonstrates the three operator policies declared above.
    in_delaware = pipeline.filter({"governing_law": "Delaware"})
    print(f"\nMatches on governing_law=Delaware: {in_delaware}")

    exact_value = pipeline.filter({"contract_value": meta.contract_value})
    print(f"Matches on exact contract_value: {exact_value}")


if __name__ == "__main__":
    main()
