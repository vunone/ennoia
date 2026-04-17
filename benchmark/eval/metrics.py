"""Retrieval recall@k computation.

CUAD QA is single-document: every question has exactly one gold contract.
Recall@k therefore reduces to "is the gold ``source_id`` in the top-k
retrieved unique sources?".

Both pipelines flatten their retrieval into an ordered, deduplicated list
of ``source_id``s (langchain: chunks grouped by document; ennoia: union of
every ``search`` tool call during the agent loop). This helper takes that
list directly so the metric is unaffected by whichever retrieval flow
produced it.
"""

from __future__ import annotations


def gold_contract_in_topk(
    gold_contract_id: str,
    retrieved_source_ids: list[str],
    k: int,
) -> bool:
    """Return True if ``gold_contract_id`` is among the first ``k`` retrieved sources."""
    return gold_contract_id in retrieved_source_ids[:k]
