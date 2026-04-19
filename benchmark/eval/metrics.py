"""Retrieval precision@k and hit-rate helpers for the product benchmark.

Each query has exactly one gold product docid, so precision@k reduces to
``1/k`` if the gold docid is among the first ``k`` retrieved unique
docids, else ``0.0``. A companion ``hit_at_k`` returns the boolean for
plotting / hit-rate summaries.
"""

from __future__ import annotations


def hit_at_k(gold_docid: str, retrieved: list[str], k: int) -> bool:
    if k <= 0:
        return False
    return gold_docid in retrieved[:k]


def precision_at_k(gold_docid: str, retrieved: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return 1.0 / k if hit_at_k(gold_docid, retrieved, k) else 0.0
