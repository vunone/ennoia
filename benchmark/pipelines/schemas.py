"""DDI schemas tailored to CUAD legal contracts.

Four schemas, designed so each CUAD question shape has an ideal matching row:

- :class:`ContractMeta` (structural) — contract identity (type, parties, dates,
  governing law). Identity questions (Document Name, Parties, Agreement Date,
  Effective Date, Governing Law) answer from here via the structured record;
  retrieval still reaches them via the overview semantic.
- :class:`ClauseInventory` (structural) — the 8-bucket clause catalogue used as
  the ``contains``-filter for category-scoped questions.
- :class:`ContractOverview` (semantic) — a per-contract gist covering
  identity + the materially-present clauses with their key parameters. One
  embedded row per contract; tailored for identity-style queries and as a
  fallback for category queries when no per-clause row matches tightly.
- :class:`ClauseMention` (collection) — **one embedded row per distinct clause**
  the contract actually addresses. Each row carries the clause's CUAD bucket,
  a plain-English gist, and a verbatim excerpt long enough for the generator
  to quote directly. This is the high-leverage index for CUAD: queries of
  the form "Highlight the parts related to <Category>" match a specific
  clause row directly, instead of ranking against a whole-contract summary.

Categories below are the canonical 41-clause taxonomy from the CUAD release,
mapped down into 8 broad buckets so the inventory filter stays tractable.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, ClassVar, Literal

from ennoia import BaseCollection, BaseSemantic, BaseStructure, Field

ContractType = Literal[
    "SaaS",
    "License",
    "Distribution",
    "Employment",
    "NDA",
    "MSA",
    "Lease",
    "JointVenture",
    "Supply",
    "Reseller",
    "Maintenance",
    "Other",
]

# Eight broad buckets in lower_snake_case — small enum, no capitalisation or
# hyphenation variants for nano-class models to hallucinate. Each bucket covers
# a related sub-family of CUAD's 41 clause categories.
CuadClauseBucket = Literal[
    "termination_renewal",
    "competition_exclusivity",
    "assignment_control",
    "ip_licensing",
    "liability_indemnity",
    "warranty_support",
    "payment_commitment",
    "audit_third_party",
]


class ContractMeta(BaseStructure):
    """Extract the contract's high-level identity: type, parties, dates, governing law.

    Read the contract carefully and identify:
    - The closest matching contract type from the allowed list.
    - All party legal names (companies, persons, or both) as they appear.
    - The effective date the contract takes force, if explicitly stated.
    - The expiration / termination date, if explicitly stated.
    - The governing-law jurisdiction (e.g. "Delaware", "England and Wales").

    Use null for any field that is not explicitly stated in the contract.
    """

    contract_type: ContractType
    parties: Annotated[
        list[str],
        Field(description="All party legal names exactly as written in the contract."),
    ]
    effective_date: Annotated[
        date | None,
        Field(description="Effective / commencement date in ISO 8601 format (YYYY-MM-DD)."),
    ] = None
    expiration_date: Annotated[
        date | None,
        Field(description="Expiration / termination date in ISO 8601 format (YYYY-MM-DD)."),
    ] = None
    governing_law: str | None = None


class ClauseInventory(BaseStructure):
    """Catalogue which broad clause buckets appear in this contract.

    Pick from this fixed list. Output the values EXACTLY as written
    (lowercase, underscores) — no capitalisation, no hyphens, no synonyms.

    - termination_renewal: termination, renewal, expiration, notice periods
    - competition_exclusivity: non-compete, non-solicit, exclusivity, MFN
    - assignment_control: anti-assignment, change of control, ROFR/ROFO/ROFN
    - ip_licensing: IP ownership, license grants, source code escrow
    - liability_indemnity: liability caps, indemnity, insurance, damages
    - warranty_support: warranties, post-termination services
    - payment_commitment: revenue share, pricing, minimum commitments
    - audit_third_party: audit rights, third-party beneficiary

    Include a bucket if ANY clause in that family is materially addressed.
    """

    clauses_present: Annotated[
        list[CuadClauseBucket],
        Field(description="Every clause bucket present in the contract."),
    ]


class ContractOverview(BaseSemantic):
    """Summarise this contract covering, in order:

    1. Contract type (e.g. distribution agreement, SaaS order form).
    2. Parties (legal names) and the relationship between them.
    3. Effective date and term length if stated.
    4. Governing law if stated.
    5. The commercially material clauses present with their key parameters
       (e.g. "non-compete 5 years in North America"; "IP assignment to
       licensor"; "liability cap equal to 12 months' fees").

    Produce enough detail to support downstream quote-level answers about
    any of these items. Preserve the contract's own legal language where
    it is material; do not paraphrase specific dollar amounts, durations,
    jurisdictions, or party names.
    """


class ClauseMention(BaseCollection):
    """Extract every distinct clause this contract materially addresses.

    For EACH clause (non-compete, license grant, liability cap, indemnity,
    audit rights, etc.) emit one entity. Skip boilerplate clauses
    (notices, severability, counterparts, headings, entire-agreement) —
    they are not useful for retrieval.

    Each entity must pick the single best-fitting `category` bucket from
    the 8-bucket taxonomy:

    - termination_renewal: termination, renewal, expiration, notice periods
    - competition_exclusivity: non-compete, non-solicit, exclusivity, MFN
    - assignment_control: anti-assignment, change of control, ROFR/ROFO/ROFN
    - ip_licensing: IP ownership, license grants, source code escrow
    - liability_indemnity: liability caps, indemnity, insurance, damages
    - warranty_support: warranties, post-termination services
    - payment_commitment: revenue share, pricing, minimum commitments
    - audit_third_party: audit rights, third-party beneficiary

    Field requirements:
    - `category`: one of the 8 buckets, lowercase, exact.
    - `subtype`: the fine-grained CUAD-style label (e.g. "non-compete",
      "license grant", "cap on liability", "audit rights"). Use the
      standard legal label when obvious.
    - `gist`: plain-English summary describing what the clause actually
      obligates: who, what, scope/duration/amount.
    - `verbatim`: a verbatim excerpt from the contract that anchors the
      clause — enough text to support an extractive answer a lawyer could
      quote directly, including the full obligation and any qualifying
      conditions. Preserve capitalisation and punctuation exactly.

    Aim for one entity per distinct clause. Two clauses that address the
    same subject under different conditions count as two entities.
    Set `is_done: true` once every materially-present clause is covered.
    """

    category: CuadClauseBucket
    subtype: Annotated[
        str,
        Field(description="Short fine-grained label (e.g. 'non-compete', 'cap on liability')."),
    ]
    gist: Annotated[
        str,
        Field(
            description=(
                "Plain-English summary of what the clause obligates: who, what, "
                "scope/duration/amount."
            )
        ),
    ]
    verbatim: Annotated[
        str,
        Field(
            description=(
                "Verbatim excerpt from the contract that anchors the clause — "
                "enough text for the generator to quote an extractive answer "
                "directly, including the full obligation and any qualifying "
                "conditions. Preserve capitalisation and punctuation exactly."
            )
        ),
    ]

    class Schema:
        # Safety cap — the loop otherwise relies on ``is_done`` + no-new-uniques
        # to terminate, which is fine for well-behaved models but leaves a
        # pathological prompt unbounded. 8 iterations × ~6 entities/iter is a
        # plenty-high ceiling for CUAD-size contracts.
        max_iterations: ClassVar[int | None] = 8

    def get_unique(self) -> str:
        # Dedup deterministically on (category, subtype, first 40 chars of
        # verbatim) — two iterations that re-emit the same clause with minor
        # phrasing differences still collapse. Keeps the random-token default
        # from allowing duplicates across the iterative loop.
        anchor = self.verbatim.strip().lower()[:40]
        return f"{self.category}|{self.subtype.lower().strip()}|{anchor}"

    def template(self) -> str:
        # Embedding-friendly one-liner: category + subtype tags up front so
        # category-scoped queries key straight onto them, gist carries the
        # semantic content, verbatim anchors the exact phrase a generator
        # can quote.
        return f'[{self.category}] [{self.subtype}] {self.gist} — "{self.verbatim}"'
