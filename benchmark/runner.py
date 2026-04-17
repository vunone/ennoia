"""Benchmark orchestration CLI.

Loads the CUAD sample, runs both pipelines end-to-end (retrieve + answer,
each pipeline owns its own generator) through the LLM-as-judge, persists
per-question records as they complete (crash-safe), and writes an
aggregated summary CSV at the end. Pass ``--dry-run`` to exercise the
wiring without spending a cent.
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
from typing import Any, TypedDict

from tqdm import tqdm

from benchmark.config import (
    DEFAULT_CONTRACT_COUNT,
    DEFAULT_QA_COUNT,
    K_VALUES,
    MODEL_GEN,
    MODEL_JUDGE,
    QA_CONCURRENCY,
    RESULTS_DIR,
    SEED,
)
from benchmark.data.loader import Question, Sample, load_or_build_sample
from benchmark.eval.cost import BudgetExceeded, RunningCost, count_tokens, estimate_run_cost
from benchmark.eval.judge import JudgeResult, judge_answer, make_judge_llm
from benchmark.eval.metrics import gold_contract_in_topk
from benchmark.pipelines.base import Pipeline

PipelineName = str  # "ennoia" | "langchain"


class QARecord(TypedDict):
    question_id: str
    contract_id: str
    category: str
    question: str
    pipeline: PipelineName
    retrieved_source_ids: list[str]
    answer: str
    verdict: str
    rationale: str
    recall: dict[str, bool]
    has_gold: bool
    trace: list[dict[str, Any]]


def _utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _synthetic_sample(n_questions: int) -> Sample:
    """Tiny in-memory sample for ``--dry-run`` without touching HuggingFace."""
    questions: list[Question] = [
        {
            "question_id": f"dry-{i}",
            "contract_id": f"contract-{i % 5}",
            "question": f'Highlight the parts (if any) of this contract related to "Non-Compete" #{i}',
            "category": "Non-Compete",
            "gold_answers": [f"Synthetic gold span {i}"],
            "has_answer": True,
        }
        for i in range(max(n_questions, 5))
    ]
    contracts = [
        {"source_id": f"contract-{i}", "title": f"contract-{i}", "text": f"Stub contract {i}."}
        for i in range(5)
    ]
    return {
        "contracts": contracts,
        "questions": questions,
        "seed": SEED,
        "contract_count": len(contracts),
        "qa_count": len(questions),
    }


def _build_pipelines(only: str | None) -> list[Pipeline]:
    # Imports inside to avoid pulling langchain when --only ennoia (and vice versa).
    pipelines: list[Pipeline] = []
    if only != "langchain":
        from benchmark.pipelines.ennoia_pipeline import EnnoiaPipeline

        pipelines.append(EnnoiaPipeline())
    if only != "ennoia":
        from benchmark.pipelines.langchain_pipeline import LangchainPipeline

        pipelines.append(LangchainPipeline())
    return pipelines


async def _process_question(
    question: Question,
    pipelines: list[Pipeline],
    judge_llm: Any,
    cost: RunningCost,
    sem: asyncio.Semaphore,
) -> list[QARecord]:
    async def run_one(pipe: Pipeline) -> QARecord | None:
        try:
            async with sem:
                run = await pipe.answer(question)
                cost.charge(MODEL_GEN, run.prompt_tokens, run.completion_tokens)
                verdict: JudgeResult = await judge_answer(
                    question["question"], question["gold_answers"], run.answer, judge_llm
                )
                cost.charge(
                    MODEL_JUDGE,
                    count_tokens(question["question"]) + count_tokens(run.answer) + 200,
                    count_tokens(verdict["rationale"]) + 20,
                )
                return {
                    "question_id": question["question_id"],
                    "contract_id": question["contract_id"],
                    "category": question["category"],
                    "question": question["question"],
                    "pipeline": pipe.name,
                    "retrieved_source_ids": run.retrieved_source_ids,
                    "answer": run.answer,
                    "verdict": verdict["verdict"],
                    "rationale": verdict["rationale"],
                    "recall": {
                        f"recall@{k}": gold_contract_in_topk(
                            question["contract_id"], run.retrieved_source_ids, k
                        )
                        for k in K_VALUES
                    },
                    "has_gold": question["has_answer"],
                    "trace": list(run.trace),
                }
        except BudgetExceeded:
            # Budget cap is intentional — propagate so the runner can stop.
            raise
        except Exception as exc:
            # Single API timeout / 5xx must not abort the whole 500-question
            # run. Log and drop this (question, pipeline) pair from the
            # results; the per-pipeline ``n`` count will reflect the skip.
            print(
                f"\n[runner] {pipe.name} failed q={question['question_id']}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return None

    raw = await asyncio.gather(*(run_one(p) for p in pipelines))
    return [r for r in raw if r is not None]


def _summarise(raw_path: Path) -> list[dict[str, Any]]:
    """Aggregate per-pipeline metrics from a JSONL file.

    Also emits ``answered_given_retrieved`` — the share of (positive,
    recall@max-k hit) rows that the generator actually answered with
    ``correct`` or ``partial``. This is the retrieval-independent measure of
    generator-context quality; a headline recall@k can be high while this
    number is low, which is exactly the diagnostic signal the ennoia vs
    shred-embed comparison hinges on.
    """
    top_k = max(K_VALUES)
    counts: dict[str, dict[str, int]] = {}
    totals: Counter[str] = Counter()
    with raw_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            pipe = row["pipeline"]
            bucket = counts.setdefault(
                pipe,
                {
                    "n": 0,
                    "correct": 0,
                    "partial": 0,
                    "hallucinated": 0,
                    "abstained": 0,
                    "retrieved_hit": 0,
                    "retrieved_hit_answered": 0,
                    **{f"recall@{k}": 0 for k in K_VALUES},
                },
            )
            bucket["n"] += 1
            bucket[row["verdict"]] = bucket.get(row["verdict"], 0) + 1
            for k in K_VALUES:
                if row["recall"][f"recall@{k}"]:
                    bucket[f"recall@{k}"] += 1
            if row["has_gold"] and row["recall"].get(f"recall@{top_k}"):
                bucket["retrieved_hit"] += 1
                if row["verdict"] in ("correct", "partial"):
                    bucket["retrieved_hit_answered"] += 1
            totals[pipe] += 1

    rows: list[dict[str, Any]] = []
    for pipe, bucket in counts.items():
        n = bucket["n"] or 1
        retrieved_hit = bucket["retrieved_hit"]
        rows.append(
            {
                "pipeline": pipe,
                "n": bucket["n"],
                **{f"recall@{k}": bucket[f"recall@{k}"] / n for k in K_VALUES},
                "correct": bucket["correct"] / n,
                "partial": bucket["partial"] / n,
                "hallucinated": bucket["hallucinated"] / n,
                "abstained": bucket["abstained"] / n,
                "answered_given_retrieved": (
                    bucket["retrieved_hit_answered"] / retrieved_hit if retrieved_hit else 0.0
                ),
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


def _write_dryrun_records(
    raw_path: Path,
    sample: Sample,
    limit: int,
) -> None:
    """Emit deterministic synthetic records so the plot path is exercised without spend."""
    rng = random.Random(SEED)
    questions = sample["questions"][:limit]
    pipelines = ["ennoia", "langchain"]
    # Bias the synthetic numbers so plots look sensible: ennoia better.
    bias = {"ennoia": 0.78, "langchain": 0.55}
    with raw_path.open("w", encoding="utf-8") as fh:
        for q in questions:
            for pipe in pipelines:
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
                record: QARecord = {
                    "question_id": q["question_id"],
                    "contract_id": q["contract_id"],
                    "category": q["category"],
                    "question": q["question"],
                    "pipeline": pipe,
                    "retrieved_source_ids": [q["contract_id"]],
                    "answer": "(dry-run synthetic)",
                    "verdict": verdict,
                    "rationale": "(dry-run synthetic)",
                    "recall": {
                        f"recall@{k}": rng.random() < (p + 0.10 if k == 5 else p + 0.18)
                        for k in K_VALUES
                    },
                    "has_gold": q["has_answer"],
                    "trace": [],
                }
                fh.write(json.dumps(record) + "\n")


async def _run_live(
    sample: Sample,
    only: str | None,
    skip_index: bool,
    cost: RunningCost,
    raw_path: Path,
    questions_to_run: list[Question],
) -> None:
    pipelines = _build_pipelines(only)
    judge_llm = make_judge_llm()

    if not skip_index:
        print(
            f"[runner] indexing {len(sample['contracts'])} contracts in "
            f"{len(pipelines)} pipeline(s) (full text)..."
        )
        # Run pipelines sequentially during indexing so their tqdm bars don't
        # interleave on the same TTY. Each pipeline still parallelises
        # internally up to its own concurrency cap.
        for pipe in pipelines:
            print(f"[runner] >>> indexing with {pipe.name}")
            await pipe.index_corpus(sample["contracts"])
        print(f"[runner] indexing complete; cost so far: ${cost.spent_usd:.4f}")

    sem = asyncio.Semaphore(QA_CONCURRENCY)

    print(
        f"[runner] running {len(questions_to_run)} questions x {len(pipelines)} pipeline(s) "
        f"with concurrency={QA_CONCURRENCY}..."
    )
    tasks = [
        asyncio.create_task(_process_question(q, pipelines, judge_llm, cost, sem))
        for q in questions_to_run
    ]
    bar = tqdm(total=len(tasks), desc="QA+judge", unit="q")
    with raw_path.open("w", encoding="utf-8") as fh:
        try:
            for fut in asyncio.as_completed(tasks):
                question_records = await fut
                for record in question_records:
                    fh.write(json.dumps(record) + "\n")
                fh.flush()
                bar.update(1)
                bar.set_postfix(spent=f"${cost.spent_usd:.2f}")
        except BudgetExceeded as exc:
            print(f"\n[runner] BUDGET STOP: {exc}", file=sys.stderr)
            for t in tasks:
                t.cancel()
        finally:
            bar.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="benchmark.runner")
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_QA_COUNT, help="Max questions to evaluate."
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--contract-count", type=int, default=DEFAULT_CONTRACT_COUNT)
    parser.add_argument("--max-cost-usd", type=float, default=30.0)
    parser.add_argument("--only", choices=("ennoia", "langchain"), default=None)
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true", help="No API calls; emit synthetic records."
    )
    parser.add_argument("--rebuild-sample", action="store_true")
    parser.add_argument(
        "--confirm-cost", action="store_true", help="Skip the pre-flight estimate prompt."
    )
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = _utc_stamp()
    raw_path = RESULTS_DIR / f"raw_{stamp}.jsonl"
    summary_path = RESULTS_DIR / f"summary_{stamp}.csv"

    if args.dry_run:
        sample = _synthetic_sample(args.limit)
    else:
        sample = load_or_build_sample(
            contract_count=args.contract_count,
            qa_count=max(args.limit, DEFAULT_QA_COUNT),
            seed=args.seed,
            rebuild=args.rebuild_sample,
        )
    questions_to_run = sample["questions"][: args.limit]

    if args.dry_run:
        print(f"[runner] DRY-RUN: emitting synthetic records for {len(questions_to_run)} questions")
        _write_dryrun_records(raw_path, sample, args.limit)
    else:
        estimate = estimate_run_cost(len(sample["contracts"]), len(questions_to_run))
        print(f"[runner] pre-flight estimate: {estimate}")
        if estimate["total_usd"] > args.max_cost_usd and not args.confirm_cost:
            print(
                f"[runner] estimated ${estimate['total_usd']:.2f} > cap ${args.max_cost_usd:.2f}; "
                "pass --confirm-cost to proceed."
            )
            return 2
        cost = RunningCost(args.max_cost_usd)
        asyncio.run(
            _run_live(
                sample=sample,
                only=args.only,
                skip_index=args.skip_index,
                cost=cost,
                raw_path=raw_path,
                questions_to_run=questions_to_run,
            )
        )
        print(f"[runner] final cost summary: {cost.summary()}")

    rows = _summarise(raw_path)
    _write_summary(summary_path, rows)
    latest_summary = RESULTS_DIR / "summary_latest.csv"
    latest_raw = RESULTS_DIR / "raw_latest.jsonl"
    latest_summary.write_text(summary_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_raw.write_text(raw_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[runner] wrote {raw_path} and {summary_path}")
    print(f"[runner] aliases: {latest_raw} {latest_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
