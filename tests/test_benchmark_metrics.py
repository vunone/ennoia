"""Unit tests for precision@k / hit@k on the ESCI product benchmark."""

from __future__ import annotations

from benchmark.eval.metrics import hit_at_k, precision_at_k


def test_hit_at_k_true_when_gold_within_top_k() -> None:
    retrieved = ["a", "b", "c", "d"]
    assert hit_at_k("c", retrieved, 5) is True
    assert hit_at_k("c", retrieved, 3) is True


def test_hit_at_k_false_when_gold_beyond_top_k() -> None:
    retrieved = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "gold"]
    assert hit_at_k("gold", retrieved, 10) is False
    assert hit_at_k("gold", retrieved, 5) is False


def test_hit_at_k_false_when_gold_missing() -> None:
    assert hit_at_k("x", ["a", "b"], 5) is False


def test_hit_at_k_false_for_non_positive_k() -> None:
    assert hit_at_k("a", ["a"], 0) is False
    assert hit_at_k("a", ["a"], -1) is False


def test_precision_at_k_matches_spec_hit() -> None:
    assert precision_at_k("a", ["a", "b"], 5) == 1 / 5
    assert precision_at_k("a", ["a", "b"], 10) == 1 / 10


def test_precision_at_k_zero_on_miss() -> None:
    assert precision_at_k("z", ["a", "b"], 5) == 0.0


def test_precision_at_k_zero_for_non_positive_k() -> None:
    assert precision_at_k("a", ["a"], 0) == 0.0
