"""Shared fixtures for the Ennoia example recipes.

Every recipe that needs a document imports ``AGREEMENT_TEXT`` from this
module so the parties, dates, and clauses stay consistent across scripts.

The helpers ``make_llm`` / ``make_embedding`` let recipes run against
Ollama locally (the default) or hosted OpenAI when ``OPENAI_API_KEY`` is
set, without every recipe repeating the adapter-selection dance.
"""

from __future__ import annotations

import os

from ennoia.adapters.embedding.base import EmbeddingAdapter
from ennoia.adapters.llm.base import LLMAdapter

__all__ = [
    "AGREEMENT_SOURCE_ID",
    "AGREEMENT_TEXT",
    "make_embedding",
    "make_llm",
]


AGREEMENT_SOURCE_ID = "msa-acme-globex"

AGREEMENT_TEXT = """\
MASTER SERVICES AGREEMENT

This Master Services Agreement ("Agreement") is entered into as of June 1, 2025
(the "Effective Date"), by and between ACME Corporation, a Delaware corporation
with its principal office in Wilmington, Delaware ("Provider"), and Globex Ltd,
a Delaware limited liability company with its principal office in Dover,
Delaware ("Client"). Provider and Client are each referred to herein as a
"Party" and collectively as the "Parties."

1. Scope of Services.
   Provider shall deliver managed cloud-infrastructure services, 24x7
   monitoring, and quarterly capacity reviews for the Client's production
   environment. The Parties may extend the scope through written statements of
   work signed by both Parties.

2. Term.
   The initial term of this Agreement is three (3) years from the Effective
   Date. The Agreement renews for successive one-year terms unless either
   Party provides written notice of non-renewal at least sixty (60) days prior
   to the end of the then-current term.

3. Fees and Payment Terms.
   The total fees for the initial term are one million two hundred thousand
   U.S. dollars ($1,200,000), payable in equal monthly installments of
   thirty-three thousand three hundred thirty-three U.S. dollars ($33,333).
   Invoices are due net thirty (30) days from the invoice date. Amounts unpaid
   after thirty (30) days accrue a late fee of 1.5% per month (0.015) on the
   outstanding balance.

4. Indemnification.
   Each Party shall defend, indemnify, and hold harmless the other Party from
   third-party claims arising out of the indemnifying Party's (a) breach of
   this Agreement, (b) gross negligence or willful misconduct, or (c)
   infringement of third-party intellectual-property rights.

5. Governing Law.
   This Agreement shall be governed by and construed in accordance with the
   laws of the State of Delaware, without regard to its conflict-of-laws
   principles. Exclusive jurisdiction and venue lie in the state and federal
   courts located in New Castle County, Delaware.

6. Entire Agreement.
   This Agreement, together with any executed statements of work, constitutes
   the entire agreement between the Parties and supersedes all prior
   understandings, whether written or oral, relating to its subject matter.

IN WITNESS WHEREOF, the Parties have executed this Agreement as of the
Effective Date.

ACME Corporation                    Globex Ltd
By: /s/ Jane Morgan                 By: /s/ Samir Patel
Name: Jane Morgan                   Name: Samir Patel
Title: Chief Executive Officer      Title: Chief Operating Officer
"""


def make_llm() -> LLMAdapter:
    """Return an LLM adapter chosen from the environment.

    ``OPENAI_API_KEY`` wins if it is set; otherwise the recipe runs against a
    local Ollama instance. Switching backends across recipes is therefore a
    matter of exporting (or unsetting) one environment variable.
    """
    if os.environ.get("OPENAI_API_KEY"):
        from ennoia.adapters.llm.openai import OpenAIAdapter

        return OpenAIAdapter(
            model=os.environ.get("ENNOIA_OPENAI_MODEL", "gpt-4o-mini"),
        )

    from ennoia.adapters.llm.ollama import OllamaAdapter

    return OllamaAdapter(model=os.environ.get("ENNOIA_OLLAMA_MODEL", "qwen3:0.6b"))


def make_embedding() -> EmbeddingAdapter:
    """Return an embedding adapter chosen from the environment.

    Matches :func:`make_llm`: OpenAI when ``OPENAI_API_KEY`` is set,
    sentence-transformers otherwise.
    """
    if os.environ.get("OPENAI_API_KEY"):
        from ennoia.adapters.embedding.openai import OpenAIEmbedding

        return OpenAIEmbedding(
            model=os.environ.get("ENNOIA_OPENAI_EMBEDDING", "text-embedding-3-small"),
        )

    from ennoia.adapters.embedding.sentence_transformers import SentenceTransformerEmbedding

    return SentenceTransformerEmbedding(
        model=os.environ.get("ENNOIA_ST_MODEL", "all-MiniLM-L6-v2"),
    )
