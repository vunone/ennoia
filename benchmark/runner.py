"""Benchmark orchestration CLI (ESCI product discovery).

Loads the prepared dataset, runs both pipelines end-to-end (retrieve +
answer, each pipeline owns its own generator) through the LLM-as-judge,
persists per-(query, pipeline) records as they complete (crash-safe),
and writes an aggregated summary CSV grouped by ``(pipeline,
difficulty)``. Pass ``--dry-run`` to exercise the wiring without
spending a cent.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

from tqdm import tqdm

from benchmark.config import (
    DATASET_PATH,
    DEFAULT_NUM_SAMPLES,
    MODEL_EMBED,
    MODEL_LLM,
    RESULTS_DIR,
    RETRIEVAL_TOP_K,
    SEED,
    THREADS,
)
from benchmark.data.prep import Dataset, Product, QueryCase, load_dataset
from benchmark.eval.cost import BudgetExceeded, RunningCost, count_tokens, estimate_run_cost
from benchmark.eval.judge import JudgeResult, judge_answer, make_judge_llm
from benchmark.eval.metrics import hit_at_k, precision_at_k
from benchmark.pipelines.base import Pipeline

Difficulty = Literal["broad", "medium", "high"]
_DIFFICULTIES: tuple[Difficulty, ...] = ("broad", "medium", "high")


class RunRecord(TypedDict):
    docid: str
    difficulty: Difficulty
    query: str
    pipeline: str
    retrieved_docids: list[str]
    answer: str
    verdict: str
    rationale: str
    precision_at_5: float
    precision_at_10: float
    hit_at_5: bool
    hit_at_10: bool
    trace: list[dict[str, Any]]


def _utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _synthetic_dataset(n: int) -> Dataset:
    """Tiny in-memory dataset for ``--dry-run`` without touching HuggingFace."""
    products: list[Product] = [
        {
            "docid": f"prod-{i}",
            "title": f"Synthetic product {i}",
            "text": f"A stub product for dry-run testing #{i}.",
            "bullet_points": [f"feature-{i}-a", f"feature-{i}-b"],
            "brand": f"BrandX{i % 3}",
            "color": "black",
            "price_usd": 20 + i * 5,
        }
        for i in range(max(n, 5))
    ]
    bands: tuple[str, ...] = ("broad", "medium", "high")
    queries: list[QueryCase] = [
        {
            "docid": p["docid"],
            "broad": f"a stub item for use case {i}",
            "medium": f"synthetic product with feature-{i}-a",
            "high": f"{p['brand']} synthetic product feature-{i}-a",
            "price_band": cast(Any, bands[i % 3]),
        }
        for i, p in enumerate(products)
    ]
    return {
        "products": products,
        "queries": queries,
        "seed": SEED,
        "num_samples": len(products),
        "start_offset": 0,
        "model": "synthetic",
    }


def _build_pipelines(only: str | None) -> list[Pipeline]:
    pipelines: list[Pipeline] = []
    if only != "langchain":
        from benchmark.pipelines.ennoia_pipeline import EnnoiaPipeline

        pipelines.append(EnnoiaPipeline())
    if only != "ennoia":
        from benchmark.pipelines.langchain_pipeline import LangchainPipeline

        pipelines.append(LangchainPipeline())
    return pipelines


def _iter_cases(dataset: Dataset) -> list[tuple[QueryCase, Product, Difficulty, str]]:
    """Expand each query case into one (case, gold, difficulty, query_text) per band."""
    by_id = {p["docid"]: p for p in dataset["products"]}
    cases: list[tuple[QueryCase, Product, Difficulty, str]] = []
    for case in dataset["queries"]:
        gold = by_id.get(case["docid"])
        if gold is None:
            continue
        for band in _DIFFICULTIES:
            cases.append((case, gold, band, cast(str, case[band])))
    return cases


async def _process_case(
    case: QueryCase,
    gold: Product,
    difficulty: Difficulty,
    query_text: str,
    pipelines: list[Pipeline],
    judge_llm: Any,
    cost: RunningCost,
    sem: asyncio.Semaphore,
    titles_by_docid: dict[str, str],
) -> list[RunRecord]:
    async def run_one(pipe: Pipeline) -> RunRecord | None:
        try:
            async with sem:
                run = await pipe.answer(query_text)
                cost.charge(MODEL_LLM, run.prompt_tokens, run.completion_tokens)
                retrieved_top = run.retrieved_source_ids[:RETRIEVAL_TOP_K]
                retrieval_rows = [
                    {"docid": docid, "title": titles_by_docid.get(docid, "")}
                    for docid in retrieved_top
                ]
                verdict: JudgeResult = await judge_answer(
                    question=query_text,
                    gold_product=cast(dict[str, Any], gold),
                    retrieved_top_k=retrieval_rows,
                    candidate_answer=run.answer,
                    llm=judge_llm,
                )
                cost.charge(
                    MODEL_LLM,
                    count_tokens(query_text) + count_tokens(run.answer) + 200,
                    count_tokens(verdict["rationale"]) + 20,
                )
                return {
                    "docid": case["docid"],
                    "difficulty": difficulty,
                    "query": query_text,
                    "pipeline": pipe.name,
                    "retrieved_docids": list(run.retrieved_source_ids),
                    "answer": run.answer,
                    "verdict": verdict["verdict"],
                    "rationale": verdict["rationale"],
                    "precision_at_5": precision_at_k(case["docid"], run.retrieved_source_ids, 5),
                    "precision_at_10": precision_at_k(case["docid"], run.retrieved_source_ids, 10),
                    "hit_at_5": hit_at_k(case["docid"], run.retrieved_source_ids, 5),
                    "hit_at_10": hit_at_k(case["docid"], run.retrieved_source_ids, 10),
                    "trace": list(run.trace),
                }
        except BudgetExceeded:
            raise
        except Exception as exc:
            print(
                f"\n[runner] {pipe.name} failed docid={case['docid']} "
                f"band={difficulty}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return None

    raw = await asyncio.gather(*(run_one(p) for p in pipelines))
    return [r for r in raw if r is not None]


def _summarise(raw_path: Path) -> list[dict[str, Any]]:
    """Aggregate per-(pipeline, difficulty) metrics from the JSONL raw log."""
    buckets: dict[tuple[str, str], dict[str, float]] = {}
    counts: Counter[tuple[str, str]] = Counter()
    with raw_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            key = (row["pipeline"], row["difficulty"])
            bucket = buckets.setdefault(
                key,
                {
                    "n": 0.0,
                    "correct": 0.0,
                    "partial": 0.0,
                    "hallucinated": 0.0,
                    "abstained": 0.0,
                    "precision_at_5_sum": 0.0,
                    "precision_at_10_sum": 0.0,
                    "hit_at_5": 0.0,
                    "hit_at_10": 0.0,
                },
            )
            bucket["n"] += 1
            bucket[row["verdict"]] = bucket.get(row["verdict"], 0.0) + 1
            bucket["precision_at_5_sum"] += float(row.get("precision_at_5", 0.0))
            bucket["precision_at_10_sum"] += float(row.get("precision_at_10", 0.0))
            if row.get("hit_at_5"):
                bucket["hit_at_5"] += 1
            if row.get("hit_at_10"):
                bucket["hit_at_10"] += 1
            counts[key] += 1

    rows: list[dict[str, Any]] = []
    for (pipe, difficulty), bucket in sorted(buckets.items()):
        n = bucket["n"] or 1
        rows.append(
            {
                "pipeline": pipe,
                "difficulty": difficulty,
                "n": int(bucket["n"]),
                "precision@5": bucket["precision_at_5_sum"] / n,
                "precision@10": bucket["precision_at_10_sum"] / n,
                "hit@5": bucket["hit_at_5"] / n,
                "hit@10": bucket["hit_at_10"] / n,
                "correct": bucket["correct"] / n,
                "partial": bucket["partial"] / n,
                "hallucinated": bucket["hallucinated"] / n,
                "abstained": bucket["abstained"] / n,
            }
        )
    return rows


def _write_summary(summary_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with summary_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_dryrun_records(raw_path: Path, dataset: Dataset, limit: int) -> None:
    rng = random.Random(SEED)
    by_id = {p["docid"]: p for p in dataset["products"]}
    cases = dataset["queries"][:limit]
    bias = {"ennoia": 0.78, "langchain": 0.55}
    with raw_path.open("w", encoding="utf-8") as fh:
        for case in cases:
            gold = by_id.get(case["docid"])
            if gold is None:
                continue
            for band in _DIFFICULTIES:
                for pipe in ("ennoia", "langchain"):
                    u = rng.random()
                    p = bias[pipe]
                    if u < p:
                        verdict = "correct"
                    elif u < p + 0.10:
                        verdict = "partial"
                    elif u < p + 0.18:
                        verdict = "abstained"
                    else:
                        verdict = "hallucinated"
                    hit5 = rng.random() < (p + 0.05)
                    hit10 = hit5 or rng.random() < 0.15
                    record: RunRecord = {
                        "docid": case["docid"],
                        "difficulty": band,
                        "query": case[band],
                        "pipeline": pipe,
                        "retrieved_docids": [case["docid"]] if hit5 else ["noise-1"],
                        "answer": "(dry-run synthetic)",
                        "verdict": verdict,
                        "rationale": "(dry-run synthetic)",
                        "precision_at_5": (1 / 5) if hit5 else 0.0,
                        "precision_at_10": (1 / 10) if hit10 else 0.0,
                        "hit_at_5": hit5,
                        "hit_at_10": hit10,
                        "trace": [],
                    }
                    fh.write(json.dumps(record) + "\n")


async def _run_live(
    dataset: Dataset,
    only: str | None,
    skip_index: bool,
    cost: RunningCost,
    raw_path: Path,
    threads: int,
) -> None:
    pipelines = _build_pipelines(only)
    judge_llm = make_judge_llm()

    if not skip_index:
        print(
            f"[runner] indexing {len(dataset['products'])} products in "
            f"{len(pipelines)} pipeline(s)..."
        )
        for pipe in pipelines:
            print(f"[runner] >>> indexing with {pipe.name}")
            await pipe.index_corpus(dataset["products"])
        print(f"[runner] indexing complete; cost so far: ${cost.spent_usd:.4f}")

    sem = asyncio.Semaphore(threads)
    cases = _iter_cases(dataset)
    titles_by_docid = {p["docid"]: p["title"] for p in dataset["products"]}

    print(
        f"[runner] running {len(cases)} (query, band) pairs x {len(pipelines)} pipeline(s) "
        f"with threads={threads}..."
    )
    tasks = [
        asyncio.create_task(
            _process_case(case, gold, band, query, pipelines, judge_llm, cost, sem, titles_by_docid)
        )
        for case, gold, band, query in cases
    ]
    bar = tqdm(total=len(tasks), desc="QA+judge", unit="q")
    # Whatever completes before an interrupt is already flushed to
    # ``raw_path``; main() runs ``_summarise`` on it regardless, so a
    # Ctrl+C on a hung final task still produces a chart from the
    # partial results.
    with raw_path.open("w", encoding="utf-8") as fh:
        try:
            for fut in asyncio.as_completed(tasks):
                records = await fut
                for record in records:
                    fh.write(json.dumps(record) + "\n")
                fh.flush()
                bar.update(1)
                bar.set_postfix(spent=f"${cost.spent_usd:.2f}")
        except (BudgetExceeded, KeyboardInterrupt) as exc:
            label = "BUDGET STOP" if isinstance(exc, BudgetExceeded) else "INTERRUPTED"
            print(f"\n[runner] {label}: {type(exc).__name__}: {exc}", file=sys.stderr)
            for t in tasks:
                t.cancel()
            # Drain cancellations so pending clients close cleanly before
            # the outer event loop tears down. ``asyncio.wait`` just waits
            # for completion — it does not re-raise exceptions (including
            # KeyboardInterrupt from tasks that already finished with one).
            # ``tasks`` is non-empty by construction: we only reach this
            # except after at least one ``await fut`` iteration started.
            await asyncio.wait(tasks)
        finally:
            bar.close()


def _write_chart(summary_path: Path, stamp: str) -> None:
    from benchmark.plot import render

    chart_stamped = RESULTS_DIR / f"chart_{stamp}.png"
    render(summary_path, chart_stamped)
    latest_chart = RESULTS_DIR / "chart_latest.png"
    latest_chart.write_bytes(chart_stamped.read_bytes())
    print(f"[runner] chart: {chart_stamped} (alias: {latest_chart})")


def _finalize_outputs(raw_path: Path, summary_path: Path, chart: bool, stamp: str) -> None:
    """Summarise ``raw_path``, write the CSV + latest aliases, optionally chart."""
    rows = _summarise(raw_path)
    _write_summary(summary_path, rows)
    latest_summary = RESULTS_DIR / "summary_latest.csv"
    latest_raw = RESULTS_DIR / "raw_latest.jsonl"
    latest_summary.write_text(summary_path.read_text(encoding="utf-8"), encoding="utf-8")
    # The summarize-only path may already have raw_path == latest_raw;
    # copying onto itself would truncate the source. Compare resolved paths.
    if raw_path.resolve() != latest_raw.resolve():
        latest_raw.write_text(raw_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[runner] wrote {raw_path} and {summary_path}")
    print(f"[runner] aliases: {latest_raw} {latest_summary}")
    if chart:
        _write_chart(summary_path, stamp)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="benchmark.runner")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help="Max query cases (products) to evaluate.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--threads", type=int, default=THREADS)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument(
        "--chart", action="store_true", help="Render the comparison chart after summarising."
    )
    parser.add_argument("--max-cost-usd", type=float, default=30.0)
    parser.add_argument("--only", choices=("ennoia", "langchain"), default=None)
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true", help="No API calls; emit synthetic records."
    )
    parser.add_argument(
        "--confirm-cost", action="store_true", help="Skip the pre-flight estimate gate."
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help=(
            "Skip running the pipelines; re-summarise an existing raw JSONL and "
            "(with --chart) render the chart. Use this to recover from a hung or "
            "interrupted run — whatever was flushed to disk is still usable."
        ),
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=None,
        help=(
            "Raw JSONL to summarise with --summarize-only. Defaults to results/raw_latest.jsonl."
        ),
    )
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = _utc_stamp()
    summary_path = RESULTS_DIR / f"summary_{stamp}.csv"

    if args.summarize_only:
        raw_path = args.raw or (RESULTS_DIR / "raw_latest.jsonl")
        if not raw_path.exists():
            print(f"[runner] no raw file at {raw_path}", file=sys.stderr)
            return 1
        line_count = sum(1 for _ in raw_path.open(encoding="utf-8"))
        print(f"[runner] summarize-only: {raw_path} ({line_count} records)")
        _finalize_outputs(raw_path, summary_path, args.chart, stamp)
        return 0

    raw_path = RESULTS_DIR / f"raw_{stamp}.jsonl"
    dataset = _synthetic_dataset(args.num_samples) if args.dry_run else load_dataset(args.dataset)
    # Truncate to num_samples without rebuilding; the runner iterates the
    # slice so downstream cases are capped.
    dataset["products"] = dataset["products"][: args.num_samples]
    dataset["queries"] = dataset["queries"][: args.num_samples]

    if args.dry_run:
        print(f"[runner] DRY-RUN: emitting synthetic records for {len(dataset['queries'])} cases")
        _write_dryrun_records(raw_path, dataset, args.num_samples)
    else:
        estimate = estimate_run_cost(len(dataset["products"]), len(dataset["queries"]) * 3)
        print(f"[runner] models: gen/judge={MODEL_LLM} embed={MODEL_EMBED}")
        print(f"[runner] pre-flight estimate: {estimate}")
        if estimate["total_usd"] > args.max_cost_usd and not args.confirm_cost:
            print(
                f"[runner] estimated ${estimate['total_usd']:.2f} > cap ${args.max_cost_usd:.2f}; "
                "pass --confirm-cost to proceed."
            )
            return 2
        cost = RunningCost(args.max_cost_usd)
        try:
            asyncio.run(
                _run_live(
                    dataset=dataset,
                    only=args.only,
                    skip_index=args.skip_index,
                    cost=cost,
                    raw_path=raw_path,
                    threads=args.threads,
                )
            )
        except KeyboardInterrupt:
            # Ctrl+C that escaped the inner handler (e.g. during indexing);
            # still finalise whatever has been flushed so far.
            print(
                "\n[runner] INTERRUPTED during indexing; summarising partial results.",
                file=sys.stderr,
            )
        print(f"[runner] final cost summary: {cost.summary()}")

    _finalize_outputs(raw_path, summary_path, args.chart, stamp)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
