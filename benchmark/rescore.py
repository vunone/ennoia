"""Pooled LLM-judge rescoring of ESCI benchmark runs (TEMPORARY TOOL).

The live runner (``benchmark/runner.py``) scores each query against a
single gold docid. That is methodologically correct only for ``high``
difficulty queries — broad and medium queries frequently have multiple
valid products in the 1k-product corpus, so a retriever that surfaces
those valid-but-non-gold matches is penalised.

This script re-scores any existing ``benchmark/results/raw_*.jsonl``
without re-running the pipelines:

1. Build a pool per query: union of every pipeline's top-10 plus gold.
2. Pass 1 — ask an LLM judge, per ``(query, docid)``, for a relevance
   label (highly_relevant / relevant / not_relevant) plus a
   constraints-satisfied flag.
3. Pass 2 — ask the same judge, per ``(query, pipeline)``, to grade the
   recommendation the pipeline returned, given the labelled pool.
4. Emit precision / strict-precision / recall / hit / MRR / nDCG
   alongside the old single-gold metrics in a CSV + chart.

Both passes use a disk-backed cache so a rerun after a crash (or after
re-running against a longer input) only pays for new pairs. Delete the
cache file to force a full re-judge.

This file is intentionally self-contained and will be removed once the
equivalent judging path lives inside the live runner.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import re
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

import matplotlib.pyplot as plt
import numpy as np

from benchmark.config import (
    DATASET_PATH,
    JUDGE_TIMEOUT_SEC,
    K_VALUES,
    MODEL_LLM,
    RESULTS_DIR,
    THREADS,
)
from benchmark.pipelines._retry import async_with_retry
from ennoia.adapters.llm.openrouter import OpenRouterAdapter

Relevance = Literal["highly_relevant", "relevant", "not_relevant"]
AnswerStatus = Literal["correct", "partial", "incorrect", "hallucinated", "abstained"]
_ANSWER_STATUSES: tuple[AnswerStatus, ...] = (
    "correct",
    "partial",
    "incorrect",
    "hallucinated",
    "abstained",
)
_RELEVANCE_LABELS: tuple[Relevance, ...] = ("highly_relevant", "relevant", "not_relevant")
_POSITIVE_LABELS: frozenset[Relevance] = frozenset({"highly_relevant", "relevant"})
_GAIN_BY_LABEL: dict[Relevance, int] = {"highly_relevant": 2, "relevant": 1, "not_relevant": 0}


class Record(TypedDict, total=False):
    """Shape of one entry in ``raw_*.jsonl`` emitted by the runner."""

    docid: str
    difficulty: str
    query: str
    pipeline: str
    retrieved_docids: list[str]
    answer: str
    verdict: str
    precision_at_5: float
    precision_at_10: float
    hit_at_5: bool
    hit_at_10: bool


class Qrel(TypedDict):
    relevance: Relevance
    constraints_satisfied: bool | Literal["n/a"]
    reasoning: str


class AnswerQrel(TypedDict):
    answer_status: AnswerStatus
    reasoning: str


# ---------------------------------------------------------------------------
# Product catalogue — single point of access.
# ---------------------------------------------------------------------------

_PRODUCT_CACHE: dict[str, dict[str, Any]] | None = None


def get_product(docid: str) -> dict[str, Any]:
    """Return the product record for ``docid`` from the benchmark corpus.

    Wired to ``benchmark.config.DATASET_PATH`` (the JSON dump produced by
    ``benchmark/data/prep.py``). Raises :class:`KeyError` if the docid is
    absent — that is a loud signal that the run file and the catalogue
    disagree, and we prefer surfacing it over silent fallbacks.
    """
    global _PRODUCT_CACHE
    if _PRODUCT_CACHE is None:
        with DATASET_PATH.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        _PRODUCT_CACHE = {
            cast(str, p["docid"]): cast("dict[str, Any]", p) for p in payload["products"]
        }
    return _PRODUCT_CACHE[docid]


# ---------------------------------------------------------------------------
# IO.
# ---------------------------------------------------------------------------


def load_records(path: Path) -> list[Record]:
    """Read a jsonl file into a list of :class:`Record`."""
    out: list[Record] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            out.append(cast(Record, json.loads(stripped)))
    return out


def build_pool(records: list[Record], top_k: int = max(K_VALUES)) -> dict[str, set[str]]:
    """Per-query union of every pipeline's top-``top_k`` plus the gold docid."""
    pool: dict[str, set[str]] = defaultdict(set)
    for r in records:
        pool[r["query"]].update(r.get("retrieved_docids", [])[:top_k])
        pool[r["query"]].add(r["docid"])
    return dict(pool)


def difficulty_index(records: list[Record]) -> dict[str, str]:
    """Map ``query → difficulty``. Assumes every row for a query agrees."""
    out: dict[str, str] = {}
    for r in records:
        out[r["query"]] = r["difficulty"]
    return out


# ---------------------------------------------------------------------------
# Cache.
# ---------------------------------------------------------------------------


class JudgeCache:
    """Append-only JSONL cache, ``(key) → response``.

    Loaded eagerly; appends are guarded by a single :class:`asyncio.Lock`
    so concurrent pass-1 judges cannot interleave lines.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._data: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    def load(self) -> None:
        if not self._path.exists():
            return
        with self._path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    entry = json.loads(stripped)
                    self._data[entry["key"]] = entry["response"]
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    print(
                        f"[rescore] skip corrupt cache line {lineno}: {exc}",
                        file=sys.stderr,
                    )

    def get(self, key: str) -> dict[str, Any] | None:
        return self._data.get(key)

    async def put(self, key: str, kind: str, response: dict[str, Any]) -> None:
        self._data[key] = response
        async with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, "kind": kind, "response": response}) + "\n")

    def __len__(self) -> int:
        return len(self._data)


def _cache_key(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _relevance_key(query: str, docid: str, difficulty: str, model: str) -> str:
    return _cache_key("relevance", "v1", query, docid, difficulty, model)


def _answer_key(record: Record, model: str) -> str:
    return _cache_key(
        "answer",
        "v1",
        record["query"],
        record["pipeline"],
        record.get("answer", "") or "",
        model,
    )


# ---------------------------------------------------------------------------
# Parsing.
# ---------------------------------------------------------------------------


class ParseFailure(Exception):
    """Raised when the judge's text cannot be coerced into a JSON object."""


_JSON_FENCE_RE = re.compile(r"```(?:\w+)?\s*", re.IGNORECASE)
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_json(raw: str) -> dict[str, Any]:
    """Extract a JSON object from free-form LLM output.

    Handles raw JSON, plain triple-backtick fences, ``json``-tagged fences,
    and trailing prose. Raises :class:`ParseFailure` otherwise.
    """
    stripped = _JSON_FENCE_RE.sub("", raw).replace("```", "").strip()
    m = _JSON_OBJ_RE.search(stripped)
    if m is None:
        raise ParseFailure(f"no JSON object in: {raw[:160]!r}")
    try:
        # The object regex matches ``{...}`` so a successful ``json.loads``
        # here always returns a ``dict``.
        return cast("dict[str, Any]", json.loads(m.group(0)))
    except json.JSONDecodeError as exc:
        raise ParseFailure(f"{exc}: {raw[:160]!r}") from exc


def extract_chosen_docid(answer: str) -> str | None:
    """Best-effort parse of the pipeline's ``answer`` JSON to its docid."""
    try:
        data = parse_json(answer)
    except ParseFailure:
        return None
    docid = data.get("docid")
    if isinstance(docid, str) and docid.strip():
        return docid.strip()
    return None


def _to_qrel(data: dict[str, Any]) -> Qrel:
    raw_rel = data.get("relevance")
    relevance: Relevance = raw_rel if raw_rel in _RELEVANCE_LABELS else "not_relevant"
    raw_con = data.get("constraints_satisfied")
    constraints: bool | Literal["n/a"] = raw_con if raw_con in (True, False, "n/a") else "n/a"
    reasoning = str(data.get("reasoning", ""))[:400]
    return {
        "relevance": relevance,
        "constraints_satisfied": constraints,
        "reasoning": reasoning,
    }


def _to_answer_qrel(data: dict[str, Any]) -> AnswerQrel:
    raw_status = data.get("answer_status")
    status: AnswerStatus = raw_status if raw_status in _ANSWER_STATUSES else "abstained"
    reasoning = str(data.get("reasoning", ""))[:400]
    return {"answer_status": status, "reasoning": reasoning}


# ---------------------------------------------------------------------------
# Judge.
# ---------------------------------------------------------------------------


@dataclass
class JudgeDeps:
    cache: JudgeCache
    client: OpenRouterAdapter
    model: str
    semaphore: asyncio.Semaphore


_DIFFICULTY_HINT: dict[str, str] = {
    "broad": "generic-category query; constraints usually n/a",
    "medium": "mixes topic with a soft constraint (e.g. price / attribute)",
    "high": "narrow intent; strong constraints expected",
}


_RELEVANCE_PROMPT = """You are a relevance assessor grading products against a shopper query. You do not know which retrieval system surfaced this product.

# Query
{query}

# Difficulty
{difficulty} — {hint}

# Product under review
Title: {title}
Brand: {brand}
Price (USD): {price}
Description: {text}
Features:
{bullets}

# Rubric
- Topical relevance: is the product about what the query asks for? Near-synonym category matches count (bird ↔ parrot ↔ penguin = relevant; bird ↔ dog toy = not_relevant).
- Constraints = explicit query mentions of brand / price / size / specific attribute. For `broad` queries, constraints are usually `n/a`. For `medium` / `high`, evaluate whether they are satisfied.
- `highly_relevant`: topical match AND constraints satisfied (or n/a).
- `relevant`: topical match, constraints partially met or not applicable.
- `not_relevant`: off-topic.

Return ONLY a JSON object — no prose, no code fences:
{{"relevance": "highly_relevant" | "relevant" | "not_relevant", "constraints_satisfied": true | false | "n/a", "reasoning": "<one short sentence>"}}
"""

_ANSWER_PROMPT = """You grade a product-recommendation system. A shopper issued a query; the system returned one recommendation (or abstained). Decide the verdict.

# Query
{query}

# Difficulty
{difficulty}

# Candidate recommendation (raw pipeline output)
{answer}

# Retrieval pool with pre-labelled relevance (gold hidden)
{pool_block}

# Verdict rubric
- `correct`: the chosen docid is `highly_relevant`, or it is `relevant` on a `broad` query.
- `partial`: the chosen docid is `relevant` but constraints are violated (medium / high only).
- `incorrect`: the chosen docid is `not_relevant`, or is not in the retrieval pool at all.
- `hallucinated`: the recommendation asserts facts that contradict the product's actual description (e.g. wrong brand / color / price class). Flag this even if topically relevant.
- `abstained`: the system returned NOT_FOUND, empty output, or no parseable docid.

Return ONLY a JSON object — no prose, no code fences:
{{"answer_status": "correct" | "partial" | "incorrect" | "hallucinated" | "abstained", "reasoning": "<one short sentence>"}}
"""


def _format_bullets(bullets: list[str]) -> str:
    if not bullets:
        return "- (none)"
    return "\n".join(f"- {b}" for b in bullets[:6])


def _format_pool_block(
    pool_labels: dict[str, Qrel],
    get_product_fn: Callable[[str], dict[str, Any]],
) -> str:
    if not pool_labels:
        return "(empty pool)"
    lines: list[str] = []
    for docid in sorted(pool_labels):
        qrel = pool_labels[docid]
        try:
            prod = get_product_fn(docid)
            title = str(prod.get("title", ""))[:140]
            brand = str(prod.get("brand", "?"))
        except KeyError:
            title, brand = "<unknown>", "?"
        lines.append(
            f"- docid={docid} | relevance={qrel['relevance']} | "
            f"constraints={qrel['constraints_satisfied']} | brand={brand} | title={title}"
        )
    return "\n".join(lines)


async def _llm_json(prompt: str, deps: JudgeDeps) -> dict[str, Any] | None:
    """Call the judge once, parse, on failure retry once with a nudge.

    Returns ``None`` if even the second attempt fails to parse; callers
    apply a safe fallback and move on so a single flaky reply cannot halt
    the run.
    """
    prompts = (
        prompt,
        prompt + "\n\nReturn ONLY a valid JSON object — no prose, no code fences.",
    )
    for attempt, p in enumerate(prompts):
        async with deps.semaphore:
            raw = await async_with_retry(lambda p=p: deps.client.complete_text(p))
        try:
            return parse_json(raw)
        except ParseFailure:
            if attempt == len(prompts) - 1:
                return None
    return None  # pragma: no cover - loop exhausts via the attempt check above.


async def judge_relevance(
    query: str,
    docid: str,
    difficulty: str,
    deps: JudgeDeps,
    get_product_fn: Callable[[str], dict[str, Any]],
) -> Qrel:
    key = _relevance_key(query, docid, difficulty, deps.model)
    cached = deps.cache.get(key)
    if cached is not None:
        return _to_qrel(cached)

    try:
        prod = get_product_fn(docid)
    except KeyError:
        fallback: Qrel = {
            "relevance": "not_relevant",
            "constraints_satisfied": "n/a",
            "reasoning": f"docid {docid!r} not in catalogue",
        }
        await deps.cache.put(key, "relevance", cast("dict[str, Any]", fallback))
        return fallback

    prompt = _RELEVANCE_PROMPT.format(
        query=query,
        difficulty=difficulty,
        hint=_DIFFICULTY_HINT.get(difficulty, ""),
        title=str(prod.get("title", "")),
        brand=str(prod.get("brand", "?")),
        price=str(prod.get("price_usd", "?")),
        text=str(prod.get("text", ""))[:1200],
        bullets=_format_bullets(list(prod.get("bullet_points") or [])),
    )
    data = await _llm_json(prompt, deps)
    if data is None:
        qrel: Qrel = {
            "relevance": "not_relevant",
            "constraints_satisfied": "n/a",
            "reasoning": "parse-failure after one retry",
        }
    else:
        qrel = _to_qrel(data)
    await deps.cache.put(key, "relevance", cast("dict[str, Any]", qrel))
    return qrel


async def judge_answer(
    record: Record,
    pool_labels: dict[str, Qrel],
    deps: JudgeDeps,
    get_product_fn: Callable[[str], dict[str, Any]],
) -> AnswerQrel:
    key = _answer_key(record, deps.model)
    cached = deps.cache.get(key)
    if cached is not None:
        return _to_answer_qrel(cached)

    answer = record.get("answer", "") or ""
    if not answer.strip() or answer.strip() == "NOT_FOUND":
        verdict: AnswerQrel = {"answer_status": "abstained", "reasoning": "empty or NOT_FOUND"}
        await deps.cache.put(key, "answer", cast("dict[str, Any]", verdict))
        return verdict

    prompt = _ANSWER_PROMPT.format(
        query=record["query"],
        difficulty=record["difficulty"],
        answer=answer.strip()[:2000],
        pool_block=_format_pool_block(pool_labels, get_product_fn),
    )
    data = await _llm_json(prompt, deps)
    if data is None:
        verdict = {"answer_status": "abstained", "reasoning": "parse-failure after one retry"}
    else:
        verdict = _to_answer_qrel(data)
    await deps.cache.put(key, "answer", cast("dict[str, Any]", verdict))
    return verdict


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------


def _should_report(done: int, total: int) -> bool:
    return done % 50 == 0 or done == total


async def run_pass_1(
    pool: dict[str, set[str]],
    difficulty_by_query: dict[str, str],
    deps: JudgeDeps,
    get_product_fn: Callable[[str], dict[str, Any]],
) -> dict[str, dict[str, Qrel]]:
    tasks: list[asyncio.Task[tuple[str, str, Qrel]]] = []
    for query, docids in pool.items():
        difficulty = difficulty_by_query[query]
        for docid in docids:

            async def _one(
                q: str = query, d: str = docid, diff: str = difficulty
            ) -> tuple[str, str, Qrel]:
                return q, d, await judge_relevance(q, d, diff, deps, get_product_fn)

            tasks.append(asyncio.create_task(_one()))

    labels: dict[str, dict[str, Qrel]] = defaultdict(dict)
    total = len(tasks)
    for done, coro in enumerate(asyncio.as_completed(tasks), 1):
        query, docid, qrel = await coro
        labels[query][docid] = qrel
        if _should_report(done, total):
            print(f"[rescore] pass1 {done}/{total}", file=sys.stderr)
    return dict(labels)


async def run_pass_2(
    records: list[Record],
    pool_labels: dict[str, dict[str, Qrel]],
    deps: JudgeDeps,
    get_product_fn: Callable[[str], dict[str, Any]],
) -> dict[tuple[str, str], AnswerQrel]:
    tasks: list[asyncio.Task[tuple[tuple[str, str], AnswerQrel]]] = []
    for record in records:
        key = (record["query"], record["pipeline"])

        async def _one(
            r: Record = record,
            k: tuple[str, str] = key,
        ) -> tuple[tuple[str, str], AnswerQrel]:
            return k, await judge_answer(r, pool_labels.get(r["query"], {}), deps, get_product_fn)

        tasks.append(asyncio.create_task(_one()))

    verdicts: dict[tuple[str, str], AnswerQrel] = {}
    total = len(tasks)
    for done, coro in enumerate(asyncio.as_completed(tasks), 1):
        k, verdict = await coro
        verdicts[k] = verdict
        if _should_report(done, total):
            print(f"[rescore] pass2 {done}/{total}", file=sys.stderr)
    return verdicts


# ---------------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------------


def _relevance_of(docid: str, qrels: dict[str, Qrel]) -> Relevance:
    return qrels[docid]["relevance"] if docid in qrels else "not_relevant"


def precision_at_k(retrieved: list[str], qrels: dict[str, Qrel], k: int, *, strict: bool) -> float:
    if k <= 0:
        return 0.0
    accept: frozenset[Relevance] = frozenset({"highly_relevant"}) if strict else _POSITIVE_LABELS
    hits = sum(1 for d in retrieved[:k] if _relevance_of(d, qrels) in accept)
    return hits / k


def recall_at_k(retrieved: list[str], qrels: dict[str, Qrel], k: int) -> float:
    total = sum(1 for q in qrels.values() if q["relevance"] in _POSITIVE_LABELS)
    if total == 0 or k <= 0:
        return 0.0
    hits = sum(1 for d in retrieved[:k] if _relevance_of(d, qrels) in _POSITIVE_LABELS)
    return hits / total


def hit_at_k(gold: str, retrieved: list[str], k: int) -> bool:
    if k <= 0:
        return False
    return gold in retrieved[:k]


def mrr_at_k(retrieved: list[str], qrels: dict[str, Qrel], k: int) -> float:
    if k <= 0:
        return 0.0
    for idx, d in enumerate(retrieved[:k], 1):
        if _relevance_of(d, qrels) in _POSITIVE_LABELS:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(retrieved: list[str], qrels: dict[str, Qrel], k: int) -> float:
    if k <= 0:
        return 0.0
    dcg = 0.0
    for idx, d in enumerate(retrieved[:k]):
        gain = _GAIN_BY_LABEL[_relevance_of(d, qrels)]
        dcg += (2**gain - 1) / math.log2(idx + 2)
    ideal_gains = sorted((_GAIN_BY_LABEL[q["relevance"]] for q in qrels.values()), reverse=True)
    idcg = sum((2**gain - 1) / math.log2(idx + 2) for idx, gain in enumerate(ideal_gains[:k]))
    return dcg / idcg if idcg > 0 else 0.0


def compute_metrics(
    records: list[Record],
    pool_labels: dict[str, dict[str, Qrel]],
    answer_verdicts: dict[tuple[str, str], AnswerQrel],
) -> list[dict[str, Any]]:
    """Aggregate per-record metrics into one row per ``(pipeline, difficulty)``.

    New pool-based metrics sit alongside the old single-gold metrics that
    were stored on each record; the CSV surfaces both for A/B readability.
    """
    groups: dict[tuple[str, str], list[Record]] = defaultdict(list)
    for r in records:
        groups[(r["pipeline"], r["difficulty"])].append(r)

    rows: list[dict[str, Any]] = []
    for (pipeline, difficulty), group in sorted(groups.items()):
        n = len(group)
        totals: dict[str, float] = defaultdict(float)
        status_counts: dict[AnswerStatus, int] = defaultdict(int)

        for r in group:
            qrels = pool_labels.get(r["query"], {})
            retrieved = r.get("retrieved_docids", [])
            gold = r["docid"]
            for k in K_VALUES:
                totals[f"precision@{k}"] += precision_at_k(retrieved, qrels, k, strict=False)
                totals[f"strict_precision@{k}"] += precision_at_k(retrieved, qrels, k, strict=True)
                totals[f"recall@{k}"] += recall_at_k(retrieved, qrels, k)
                totals[f"hit@{k}"] += 1.0 if hit_at_k(gold, retrieved, k) else 0.0
            totals["mrr@10"] += mrr_at_k(retrieved, qrels, 10)
            totals["ndcg@10"] += ndcg_at_k(retrieved, qrels, 10)
            totals["old_precision@5"] += float(r.get("precision_at_5", 0.0))
            totals["old_precision@10"] += float(r.get("precision_at_10", 0.0))
            totals["old_hit@5"] += 1.0 if r.get("hit_at_5") else 0.0
            totals["old_hit@10"] += 1.0 if r.get("hit_at_10") else 0.0

            verdict = answer_verdicts.get((r["query"], pipeline))
            if verdict is not None:
                status_counts[verdict["answer_status"]] += 1

        row: dict[str, Any] = {"pipeline": pipeline, "difficulty": difficulty, "n": n}
        for key, total in totals.items():
            row[key] = total / n if n > 0 else 0.0
        for status in _ANSWER_STATUSES:
            row[status] = status_counts.get(status, 0) / n if n > 0 else 0.0
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Outputs.
# ---------------------------------------------------------------------------


_CSV_COLUMNS: tuple[str, ...] = (
    "pipeline",
    "difficulty",
    "n",
    "precision@5",
    "precision@10",
    "strict_precision@5",
    "strict_precision@10",
    "recall@5",
    "recall@10",
    "hit@5",
    "hit@10",
    "mrr@10",
    "ndcg@10",
    "correct",
    "partial",
    "incorrect",
    "hallucinated",
    "abstained",
    "old_precision@5",
    "old_precision@10",
    "old_hit@5",
    "old_hit@10",
)


def write_summary_csv(out_dir: Path, timestamp: str, rows: list[dict[str, Any]]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamped = out_dir / f"rescored_summary_{timestamp}.csv"
    with stamped.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(_CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in _CSV_COLUMNS})
    latest = out_dir / "rescored_summary_latest.csv"
    latest.write_bytes(stamped.read_bytes())
    return stamped


def write_qrels_jsonl(
    out_dir: Path, timestamp: str, pool_labels: dict[str, dict[str, Qrel]]
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"rescored_qrels_{timestamp}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for query in sorted(pool_labels):
            for docid in sorted(pool_labels[query]):
                fh.write(
                    json.dumps({"query": query, "docid": docid, **pool_labels[query][docid]}) + "\n"
                )
    return path


def write_answers_jsonl(
    out_dir: Path,
    timestamp: str,
    records: list[Record],
    answer_verdicts: dict[tuple[str, str], AnswerQrel],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"rescored_answers_{timestamp}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            verdict = answer_verdicts.get((r["query"], r["pipeline"]))
            payload: dict[str, Any] = {
                "query": r["query"],
                "pipeline": r["pipeline"],
                "difficulty": r["difficulty"],
                "gold_docid": r["docid"],
                "chosen_docid": extract_chosen_docid(r.get("answer", "") or ""),
                "old_verdict": r.get("verdict"),
                "new_verdict": verdict["answer_status"] if verdict else None,
                "new_reasoning": verdict["reasoning"] if verdict else None,
            }
            fh.write(json.dumps(payload) + "\n")
    return path


_CHART_METRICS: tuple[str, ...] = (
    "precision@10",
    "recall@10",
    "mrr@10",
    "ndcg@10",
    "correct",
    "hallucinated",
)
_DIFFICULTIES: tuple[str, ...] = ("broad", "medium")
_PIPELINE_COLORS: dict[str, str] = {"ennoia": "#2f7dc1", "langchain": "#d9822b"}


def render_chart(out_dir: Path, timestamp: str, rows: list[dict[str, Any]]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"rescored_chart_{timestamp}.png"
    indexed = {(row["pipeline"], row["difficulty"]): row for row in rows}
    pipelines = sorted({p for p, _ in indexed})

    fig, axes = plt.subplots(nrows=1, ncols=len(_DIFFICULTIES), figsize=(16, 5.5), sharey=True)
    for col, band in enumerate(_DIFFICULTIES):
        ax = axes[col]
        x = np.arange(len(_CHART_METRICS))
        width = 0.8 / max(len(pipelines), 1)
        for idx, pipe in enumerate(pipelines):
            offsets = x + (idx - (len(pipelines) - 1) / 2) * width
            values = [float(indexed.get((pipe, band), {}).get(m, 0.0)) for m in _CHART_METRICS]
            bars = ax.bar(
                offsets,
                values,
                width,
                label=pipe,
                color=_PIPELINE_COLORS.get(pipe, "#888"),
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
        ax.set_xticklabels(list(_CHART_METRICS), rotation=25, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_title(f"query details: {'rich' if band == 'medium' else 'poor'}")
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        if col == 0:
            ax.set_ylabel("score")
        if col == len(_DIFFICULTIES) - 1:
            ax.legend(title="Pipeline", loc="upper right")

    fig.suptitle("ESCI — Ennoia vs LangChain naive RAG", fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def print_summary_table(rows: list[dict[str, Any]]) -> None:
    header = (
        "| pipeline | band | n | old p@5 | new p@5 | strict p@5 | recall@5 "
        "| old hit@5 | mrr@10 | ndcg@10 | correct | halluc |"
    )
    sep = "|" + "|".join(["---"] * 12) + "|"
    print(header)
    print(sep)
    for row in rows:
        print(
            f"| {row['pipeline']} | {row['difficulty']} | {int(row['n'])} "
            f"| {row.get('old_precision@5', 0.0):.3f} "
            f"| {row.get('precision@5', 0.0):.3f} "
            f"| {row.get('strict_precision@5', 0.0):.3f} "
            f"| {row.get('recall@5', 0.0):.3f} "
            f"| {row.get('old_hit@5', 0.0):.3f} "
            f"| {row.get('mrr@10', 0.0):.3f} "
            f"| {row.get('ndcg@10', 0.0):.3f} "
            f"| {row.get('correct', 0.0):.3f} "
            f"| {row.get('hallucinated', 0.0):.3f} |"
        )


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmark.rescore",
        description=(
            "Temporary: rescore an existing raw_*.jsonl with a pooled LLM judge. "
            "Recall is pool-recall (only what either pipeline retrieved) — "
            "document-level recall is not recoverable without re-scanning the corpus."
        ),
    )
    parser.add_argument("--input", type=Path, default=RESULTS_DIR / "raw_latest.jsonl")
    parser.add_argument("--cache", type=Path, default=RESULTS_DIR / ".rescore_cache.jsonl")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--judge-model", default=MODEL_LLM)
    parser.add_argument("--concurrency", type=int, default=THREADS)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _default_client_factory(model: str) -> OpenRouterAdapter:
    return OpenRouterAdapter(model=model, timeout=JUDGE_TIMEOUT_SEC)


async def main(
    args: argparse.Namespace,
    *,
    client_factory: Callable[[str], OpenRouterAdapter] = _default_client_factory,
    get_product_fn: Callable[[str], dict[str, Any]] = get_product,
) -> int:
    records = load_records(args.input)
    if not records:
        print(f"[rescore] no records in {args.input}", file=sys.stderr)
        return 1

    pool = build_pool(records)
    diff_index = difficulty_index(records)
    cache = JudgeCache(args.cache)
    cache.load()

    total_p1 = sum(len(docids) for docids in pool.values())
    pending_p1 = sum(
        1
        for q, ds in pool.items()
        for d in ds
        if cache.get(_relevance_key(q, d, diff_index[q], args.judge_model)) is None
    )
    pending_p2 = sum(1 for r in records if cache.get(_answer_key(r, args.judge_model)) is None)

    print(
        f"[rescore] records={len(records)} queries={len(pool)} "
        f"pool_size={total_p1} cached={len(cache)} "
        f"pass1_pending={pending_p1} pass2_pending={pending_p2}",
        file=sys.stderr,
    )

    if args.dry_run:
        return 0

    deps = JudgeDeps(
        cache=cache,
        client=client_factory(args.judge_model),
        model=args.judge_model,
        semaphore=asyncio.Semaphore(max(1, args.concurrency)),
    )

    pool_labels = await run_pass_1(pool, diff_index, deps, get_product_fn)
    answer_verdicts = await run_pass_2(records, pool_labels, deps, get_product_fn)

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    summary = compute_metrics(records, pool_labels, answer_verdicts)
    write_summary_csv(args.output_dir, timestamp, summary)
    write_qrels_jsonl(args.output_dir, timestamp, pool_labels)
    write_answers_jsonl(args.output_dir, timestamp, records, answer_verdicts)
    render_chart(args.output_dir, timestamp, summary)
    print_summary_table(summary)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(asyncio.run(main(build_parser().parse_args())))
