"""Unit tests for the comparison-chart renderer."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from benchmark import plot


def _write_summary(path: Path) -> None:
    rows = [
        {
            "pipeline": "ennoia",
            "difficulty": "broad",
            "n": 5,
            "precision@5": 0.2,
            "precision@10": 0.1,
            "hit@5": 1.0,
            "hit@10": 1.0,
            "correct": 0.8,
            "partial": 0.1,
            "hallucinated": 0.0,
            "abstained": 0.1,
        },
        {
            "pipeline": "langchain",
            "difficulty": "broad",
            "n": 5,
            "precision@5": 0.1,
            "precision@10": 0.05,
            "hit@5": 0.5,
            "hit@10": 0.5,
            "correct": 0.4,
            "partial": 0.2,
            "hallucinated": 0.3,
            "abstained": 0.1,
        },
        {
            "pipeline": "ennoia",
            "difficulty": "medium",
            "n": 5,
            "precision@5": 0.2,
            "precision@10": 0.1,
            "hit@5": 1.0,
            "hit@10": 1.0,
            "correct": 0.9,
            "partial": 0.0,
            "hallucinated": 0.0,
            "abstained": 0.1,
        },
        {
            "pipeline": "langchain",
            "difficulty": "medium",
            "n": 5,
            "precision@5": 0.1,
            "precision@10": 0.05,
            "hit@5": 0.5,
            "hit@10": 0.5,
            "correct": 0.5,
            "partial": 0.2,
            "hallucinated": 0.2,
            "abstained": 0.1,
        },
        {
            "pipeline": "ennoia",
            "difficulty": "high",
            "n": 5,
            "precision@5": 0.2,
            "precision@10": 0.1,
            "hit@5": 1.0,
            "hit@10": 1.0,
            "correct": 1.0,
            "partial": 0.0,
            "hallucinated": 0.0,
            "abstained": 0.0,
        },
        {
            "pipeline": "langchain",
            "difficulty": "high",
            "n": 5,
            "precision@5": 0.1,
            "precision@10": 0.05,
            "hit@5": 0.5,
            "hit@10": 0.5,
            "correct": 0.6,
            "partial": 0.2,
            "hallucinated": 0.2,
            "abstained": 0.0,
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_render_writes_png(tmp_path: Path) -> None:
    summary = tmp_path / "summary.csv"
    _write_summary(summary)
    out = tmp_path / "chart.png"
    path = plot.render(summary, out)
    assert path == out
    assert out.exists() and out.stat().st_size > 0


def test_render_raises_on_empty_summary(tmp_path: Path) -> None:
    summary = tmp_path / "empty.csv"
    summary.write_text(
        "pipeline,difficulty,n,precision@5,precision@10,hit@5,hit@10,correct,partial,hallucinated,abstained\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="empty summary"):
        plot.render(summary, tmp_path / "out.png")


def test_main_missing_summary_returns_error_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(plot, "RESULTS_DIR", tmp_path)
    rc = plot.main([])
    assert rc == 1


def test_main_writes_chart(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(plot, "RESULTS_DIR", tmp_path)
    summary = tmp_path / "summary_latest.csv"
    _write_summary(summary)
    rc = plot.main([])
    assert rc == 0
    assert (tmp_path / "chart_latest.png").exists()


def test_main_with_explicit_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(plot, "RESULTS_DIR", tmp_path)
    summary = tmp_path / "custom.csv"
    _write_summary(summary)
    rc = plot.main([str(summary)])
    assert rc == 0


def test_render_tolerates_missing_metric_columns(tmp_path: Path) -> None:
    # Legacy / partial summary CSV without all metric columns still renders.
    summary = tmp_path / "partial.csv"
    summary.write_text(
        "pipeline,difficulty,n,correct\nennoia,broad,5,0.8\nlangchain,broad,5,0.4\n",
        encoding="utf-8",
    )
    out = tmp_path / "partial.png"
    plot.render(summary, out)
    assert out.exists()
