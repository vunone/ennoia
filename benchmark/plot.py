"""Render the comparison chart from a summary CSV.

Reads ``results/summary_latest.csv`` (or a path given as the first arg)
and writes a grouped bar chart to ``results/chart_latest.png`` plus a
timestamped copy. Each metric is grouped by difficulty band
(broad/medium/high) with one bar per pipeline; ennoia and langchain are
shown side by side within each band.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from benchmark.config import MODEL_EMBED, MODEL_LLM, RESULTS_DIR

METRIC_ORDER = [
    "precision@5",
    "precision@10",
    "hit@5",
    "hit@10",
    "correct",
    "partial",
    "hallucinated",
    "abstained",
]
METRIC_LABEL = {
    "precision@5": "Precision@5",
    "precision@10": "Precision@10",
    "hit@5": "Hit@5",
    "hit@10": "Hit@10",
    "correct": "Judge: correct",
    "partial": "Judge: partial",
    "hallucinated": "Judge: hallucinated",
    "abstained": "Judge: abstained",
}
_DIFFICULTIES = ("broad", "medium", "high")
PIPELINE_COLORS = {"ennoia": "#2f7dc1", "langchain": "#d9822b"}
PIPELINE_HATCH = {"ennoia": "", "langchain": "//"}


def _read_summary(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    out: dict[tuple[str, str], dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            key = (row["pipeline"], row["difficulty"])
            metrics: dict[str, float] = {}
            for metric in METRIC_ORDER:
                if metric in row:
                    metrics[metric] = float(row[metric])
            metrics["n"] = float(row.get("n", 0))
            out[key] = metrics
    return out


def render(summary_path: Path, out_path: Path) -> Path:
    rows = _read_summary(summary_path)
    if not rows:
        raise RuntimeError(f"empty summary: {summary_path}")
    pipelines = sorted({p for p, _ in rows})
    n_metrics = len(METRIC_ORDER)

    fig, axes = plt.subplots(nrows=1, ncols=len(_DIFFICULTIES), figsize=(16, 5.5), sharey=True)
    for col, band in enumerate(_DIFFICULTIES):
        ax = axes[col]
        x = np.arange(n_metrics)
        width = 0.8 / max(len(pipelines), 1)
        for idx, pipe in enumerate(pipelines):
            offsets = x + (idx - (len(pipelines) - 1) / 2) * width
            values = [rows.get((pipe, band), {}).get(m, 0.0) for m in METRIC_ORDER]
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
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
        ax.set_xticks(x)
        ax.set_xticklabels([METRIC_LABEL[m] for m in METRIC_ORDER], rotation=25, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_title(f"difficulty: {band}")
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        if col == 0:
            ax.set_ylabel("Share / precision")
        if col == len(_DIFFICULTIES) - 1:
            ax.legend(title="Pipeline", loc="upper right")

    fig.suptitle(
        f"ESCI product discovery — gen/judge={MODEL_LLM} embed={MODEL_EMBED}",
        fontsize=11,
    )
    fig.text(
        0.99,
        0.01,
        f"Generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        ha="right",
        va="bottom",
        fontsize=7,
        color="#666",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
