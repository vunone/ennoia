"""Recipe 04 — all three extractors with extend() branching.

This recipe is the full shape of an Ennoia pipeline in one file. It pairs:

- a structural extractor (:class:`ContractMeta`) that conditionally
  **branches** the extraction DAG via :meth:`extend` — ``PaymentTerms`` is
  only extracted for contracts above a dollar threshold;
- a semantic extractor (:class:`ContractSummary`) that answers a question
  against the document;
- a collection extractor (:class:`Party`) that pulls every party named in
  the agreement, with deduplication and an embed-template.

``Schema.extensions`` on :class:`ContractMeta` declares which classes
:meth:`extend` is allowed to return; returning an undeclared class raises
a ``SchemaError`` at extraction time.

Run it::

    python examples/04_extend_branching.py
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, ClassVar, Literal

from _data import AGREEMENT_SOURCE_ID, AGREEMENT_TEXT, make_embedding, make_llm

from ennoia import (
    BaseCollection,
    BaseSemantic,
    BaseStructure,
    Field,
    Pipeline,
    Store,
)
from ennoia.store import InMemoryStructuredStore, InMemoryVectorStore


class PaymentTerms(BaseStructure):
    """Extract the payment cadence and late-fee terms of a high-value contract."""

    net_days: Annotated[
        int,
        Field(description="Net payment window in days (e.g. 30 for net-30 terms)."),
    ]
    late_fee_pct: Annotated[
        float,
        Field(description="Late fee as a decimal per month (0.015 means 1.5%/month)."),
    ]


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
    contract_value: Annotated[
        int,
        Field(description="Total fees in USD across the initial term; 0 if not stated."),
    ]

    def extend(
        self,
    ) -> list[type[BaseStructure] | type[BaseSemantic] | type[BaseCollection]]:
        # The DAG branches here: PaymentTerms only runs for contracts whose
        # total value is at least $100k. For smaller agreements the extra
        # LLM call is skipped entirely.
        return [PaymentTerms] if self.contract_value >= 100_000 else []

    class Schema(BaseStructure.Schema):
        # Every class ``extend()`` might return MUST be listed here, even
        # conditionally. The Pipeline walks this list at init to build the
        # superschema and rejects undeclared emissions at runtime.
        extensions: ClassVar[list[type]] = [PaymentTerms]


class ContractSummary(BaseSemantic):
    """What services does the provider deliver under this agreement?"""


class Party(BaseCollection):
    """Extract every party named in the agreement."""

    legal_name: Annotated[
        str,
        Field(description="Full legal name of the party as written in the agreement."),
    ]
    role: Annotated[
        Literal["provider", "client"],
        Field(description="Whether this party provides or receives the services."),
    ]

    def get_unique(self) -> str:
        # Dedup on a case-folded name so "ACME Corporation" and "acme
        # corporation" collapse to a single entity across iterations.
        return self.legal_name.casefold()

    def template(self) -> str:
        # This string is what gets embedded into the vector store per party.
        return f"{self.legal_name} ({self.role})"


def main() -> None:
    pipeline = Pipeline(
        schemas=[ContractMeta, ContractSummary, Party],
        store=Store(
            vector=InMemoryVectorStore(),
            structured=InMemoryStructuredStore(),
        ),
        llm=make_llm(),
        embedding=make_embedding(),
    )

    result = pipeline.index(text=AGREEMENT_TEXT, source_id=AGREEMENT_SOURCE_ID)

    meta = result.structural["ContractMeta"]
    assert isinstance(meta, ContractMeta)
    print("Structural — ContractMeta:")
    print(f"  governing_law  = {meta.governing_law}")
    print(f"  effective_date = {meta.effective_date}")
    print(f"  contract_value = ${meta.contract_value:,}")

    # PaymentTerms is present iff the extend() branch fired.
    if "PaymentTerms" in result.structural:
        terms = result.structural["PaymentTerms"]
        assert isinstance(terms, PaymentTerms)
        print("\nextend() branch fired — PaymentTerms:")
        print(f"  net_days      = {terms.net_days}")
        print(f"  late_fee_pct  = {terms.late_fee_pct}")
    else:
        print("\nextend() branch skipped (contract_value below threshold).")

    print("\nSemantic — ContractSummary:")
    print(f"  {result.semantic['ContractSummary']}")

    print("\nCollection — Party:")
    for party in result.collections.get("Party", []):
        assert isinstance(party, Party)
        print(f"  - {party.legal_name} ({party.role})")

    # Filter on the parent extractor's fields, then rank by semantic similarity.
    hits = pipeline.search(
        query="late payment penalties and invoice terms",
        filters={"governing_law": "Delaware", "contract_value__gte": 100_000},
        top_k=3,
    )
    print(f"\n{len(hits)} hit(s) on governing_law=Delaware AND contract_value>=$100k:")
    for hit in hits:
        print(f"  [{hit.source_id}] score={hit.score:.3f}")


if __name__ == "__main__":
    main()
