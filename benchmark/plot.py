"""Render the comparison chart from a summary CSV.

Reads ``results/summary_latest.csv`` (or a path given as the first arg) and
writes a grouped bar chart to ``results/chart_latest.png`` plus a timestamped
copy. The chart is the deliverable screenshot for the README.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from benchmark.config import K_VALUES, MODEL_GEN, MODEL_JUDGE, RESULTS_DIR

METRIC_ORDER = [
    *(f"recall@{k}" for k in K_VALUES),
    "correct",
    "partial",
    "hallucinated",
    "abstained",
]
METRIC_LABEL = {
    "recall@5": "Recall@5",
    "recall@10": "Recall@10",
    "correct": "Judge: correct",
    "partial": "Judge: partial",
    "hallucinated": "Judge: hallucinated",
    "abstained": "Judge: abstained",
}
PIPELINE_COLORS = {
    "ennoia": "#2f7dc1",
    "langchain": "#d9822b",
}
PIPELINE_HATCH = {"ennoia": "", "langchain": "//"}


def _read_summary(path: Path) -> dict[str, dict[str, float]]:
    by_pipeline: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            metrics = {k: float(row[k]) for k in METRIC_ORDER if k in row}
            metrics["n"] = float(row.get("n", 0))
            by_pipeline[row["pipeline"]] = metrics
    return by_pipeline


def render(summary_path: Path, out_path: Path) -> Path:
    by_pipeline = _read_summary(summary_path)
    if not by_pipeline:
        raise RuntimeError(f"empty summary: {summary_path}")

    pipelines = sorted(by_pipeline.keys())  # alphabetical: ennoia first
    x = np.arange(len(METRIC_ORDER))
    width = 0.8 / len(pipelines)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for idx, pipe in enumerate(pipelines):
        offsets = x + (idx - (len(pipelines) - 1) / 2) * width
        values = [by_pipeline[pipe].get(m, 0.0) for m in METRIC_ORDER]
        bars = ax.bar(
            offsets,
            values,
            width,
            label=pipe,
            color=PIPELINE_COLORS.get(pipe, "#888"),
            hatch=PIPELINE_HATCH.get(pipe, ""),
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.0%}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABEL[m] for m in METRIC_ORDER], rotation=15, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Share of questions")
    n_total = int(next(iter(by_pipeline.values())).get("n", 0))
    ax.set_title(
        f"CUAD QA — n={n_total}, k={max(K_VALUES)}, generator={MODEL_GEN}, judge={MODEL_JUDGE}"
    )
    ax.legend(title="Pipeline", loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.text(
        0.99,
        0.01,
        f"Generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        ha="right",
        va="bottom",
        fontsize=7,
        color="#666",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    summary_path = Path(args[0]) if args else RESULTS_DIR / "summary_latest.csv"
    if not summary_path.exists():
        print(f"[plot] no summary at {summary_path}; run benchmark.runner first", file=sys.stderr)
        return 1
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_stamped = RESULTS_DIR / f"chart_{stamp}.png"
    render(summary_path, out_stamped)
    out_latest = RESULTS_DIR / "chart_latest.png"
    out_latest.write_bytes(out_stamped.read_bytes())
    print(f"[plot] wrote {out_stamped} and {out_latest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
