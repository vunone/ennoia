"""Unit tests for cost accounting on the product benchmark."""

from __future__ import annotations

import pytest

from benchmark.eval.cost import (
    BudgetExceeded,
    RunningCost,
    cost_for,
    count_tokens,
    estimate_run_cost,
)


def test_count_tokens_returns_positive_int() -> None:
    assert count_tokens("hello world") > 0


def test_cost_for_unknown_model_is_zero() -> None:
    assert cost_for("definitely-not-a-model", 1000, 1000) == 0.0


def test_cost_for_known_model_scales_linearly() -> None:
    # text-embedding-3-small is priced at 0.02/M input.
    assert cost_for("text-embedding-3-small", 1_000_000, 0) == pytest.approx(0.02)


def test_running_cost_charges_and_tracks_by_model() -> None:
    cost = RunningCost(max_usd=10.0)
    delta = cost.charge("text-embedding-3-small", 1_000_000, 0)
    assert delta == pytest.approx(0.02)
    assert cost.spent_usd == pytest.approx(0.02)
    summary = cost.summary()
    assert summary["tokens_by_model"] == {
        "text-embedding-3-small": {"input": 1_000_000, "output": 0}
    }


def test_running_cost_raises_when_cap_breached() -> None:
    cost = RunningCost(max_usd=0.0)
    with pytest.raises(BudgetExceeded):
        cost.charge("text-embedding-3-small", 1_000_000, 0)


def test_estimate_run_cost_returns_bucket_totals() -> None:
    estimate = estimate_run_cost(n_products=10, n_queries=30)
    assert set(estimate.keys()) == {
        "extraction_usd",
        "embedding_usd",
        "qa_usd",
        "judge_usd",
        "total_usd",
    }
    # Only embedding is paid under the default free-tier LLM handle.
    assert estimate["extraction_usd"] == 0.0
    assert estimate["qa_usd"] == 0.0
    assert estimate["judge_usd"] == 0.0
    assert estimate["total_usd"] == pytest.approx(estimate["embedding_usd"])
