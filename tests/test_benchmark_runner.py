"""Unit tests for the benchmark runner aggregation + record construction.

Covers the two pieces wired up for the diagnostic re-run:

- ``_process_question`` threads the pipeline's agent ``trace`` into the
  persisted ``QARecord`` so an aborted/completed run can be audited for
  tool-call patterns (e.g. how often did the ennoia agent call get_full?).
- ``_summarise`` emits ``answered_given_retrieved`` — the share of
  positives whose gold contract was retrieved at top-k AND ultimately
  answered. This is the retrieval-independent measure of
  generator-context quality.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from benchmark import runner
from benchmark.data.loader import Question
from benchmark.eval.cost import RunningCost
from benchmark.eval.judge import make_judge_llm
from benchmark.pipelines.base import PipelineRun
from benchmark.pipelines.generator import make_generator_llm
from ennoia.adapters.llm.ollama import OllamaAdapter
from ennoia.adapters.llm.openai import OpenAIAdapter


class _FakePipe:
    name = "fake"

    def __init__(self, run: PipelineRun) -> None:
        self._run = run

    async def index_corpus(self, contracts: list[Any]) -> None:
        return None

    async def answer(self, question: Question) -> PipelineRun:
        return self._run


QUESTION: Question = {
    "question_id": "q-1",
    "contract_id": "contract-1",
    "question": "Highlight non-compete clauses.",
    "category": "Non-Compete",
    "gold_answers": ["Non-compete 5y."],
    "has_answer": True,
}


async def test_process_question_threads_trace_into_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = [
        {"iteration": 0, "tool": "get_search_schema", "args": {}},
        {"iteration": 1, "tool": "search", "args": {"query": "non-compete"}},
        {"iteration": 2, "tool": "get_full", "args": {"document_id": "contract-1"}},
    ]
    run = PipelineRun(
        retrieved_source_ids=["contract-1"],
        answer="5-year non-compete in NA.",
        prompt_tokens=100,
        completion_tokens=20,
        trace=trace,
    )
    pipe = _FakePipe(run)

    async def fake_judge(
        question: str,
        gold_answers: list[str],
        candidate_answer: str,
        llm: Any,
    ) -> dict[str, str]:
        return {"verdict": "correct", "rationale": "matches gold."}

    monkeypatch.setattr(runner, "judge_answer", fake_judge)

    cost = RunningCost(max_usd=5.0)
    sem = asyncio.Semaphore(1)
    records = await runner._process_question(QUESTION, [pipe], judge_llm=None, cost=cost, sem=sem)

    assert len(records) == 1
    record = records[0]
    assert record["pipeline"] == "fake"
    assert record["verdict"] == "correct"
    # The trace survives end-to-end — a deep copy so future pipelines can
    # mutate their own list safely, but with equal contents.
    assert record["trace"] == trace
    assert record["trace"] is not trace
    # Retrieval recall is still computed.
    assert record["recall"]["recall@5"] is True
    assert record["recall"]["recall@10"] is True


def test_summarise_emits_answered_given_retrieved(tmp_path: Path) -> None:
    # Two pipelines, each with 4 rows. Design the fixture so both have the
    # same recall@10 but different generator-context quality.
    #   pipeA positives: 3 with gold, 2 retrieved gold, 1 answered -> 1/2
    #   pipeB positives: 3 with gold, 2 retrieved gold, 2 answered -> 2/2
    # 1 negative per pipeline is untouched by the metric.
    rows: list[dict[str, Any]] = [
        # pipeA — retrieved + correct
        {
            "pipeline": "pipeA",
            "verdict": "correct",
            "has_gold": True,
            "recall": {"recall@5": True, "recall@10": True},
        },
        # pipeA — retrieved but abstained (context-starved signal)
        {
            "pipeline": "pipeA",
            "verdict": "abstained",
            "has_gold": True,
            "recall": {"recall@5": True, "recall@10": True},
        },
        # pipeA — gold missed in top-10 (excluded from denominator)
        {
            "pipeline": "pipeA",
            "verdict": "abstained",
            "has_gold": True,
            "recall": {"recall@5": False, "recall@10": False},
        },
        # pipeA — negative (excluded from denominator)
        {
            "pipeline": "pipeA",
            "verdict": "correct",
            "has_gold": False,
            "recall": {"recall@5": False, "recall@10": False},
        },
        # pipeB — retrieved + correct x2
        {
            "pipeline": "pipeB",
            "verdict": "correct",
            "has_gold": True,
            "recall": {"recall@5": True, "recall@10": True},
        },
        {
            "pipeline": "pipeB",
            "verdict": "partial",
            "has_gold": True,
            "recall": {"recall@5": False, "recall@10": True},
        },
        # pipeB — gold missed (excluded)
        {
            "pipeline": "pipeB",
            "verdict": "hallucinated",
            "has_gold": True,
            "recall": {"recall@5": False, "recall@10": False},
        },
        # pipeB — negative (excluded)
        {
            "pipeline": "pipeB",
            "verdict": "correct",
            "has_gold": False,
            "recall": {"recall@5": False, "recall@10": False},
        },
    ]
    raw = tmp_path / "raw.jsonl"
    with raw.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    summary = {r["pipeline"]: r for r in runner._summarise(raw)}
    assert summary["pipeA"]["answered_given_retrieved"] == pytest.approx(0.5)
    assert summary["pipeB"]["answered_given_retrieved"] == pytest.approx(1.0)
    # Sanity: existing columns still populated.
    assert summary["pipeA"]["n"] == 4
    assert summary["pipeA"]["recall@10"] == pytest.approx(0.5)


def test_generator_factory_returns_local_ollama_adapter() -> None:
    # Regression guard for the local-generator switch: the langchain
    # baseline's answer step must route through Ollama, not OpenAI, so
    # the benchmark can run without an OPENAI_API_KEY for the generator.
    llm = make_generator_llm()
    assert isinstance(llm, OllamaAdapter)


def test_judge_factory_stays_on_openai() -> None:
    # Judge is the only paid path; regressing it to a local model would
    # silently collapse the quality buffer between generator and judge.
    llm = make_judge_llm()
    assert isinstance(llm, OpenAIAdapter)


def test_summarise_handles_pipeline_with_no_retrieved_hits(tmp_path: Path) -> None:
    # Degenerate case: no positives retrieved at top-k — denominator is zero
    # and we must report 0.0, not raise ZeroDivisionError.
    rows: list[dict[str, Any]] = [
        {
            "pipeline": "weak",
            "verdict": "abstained",
            "has_gold": True,
            "recall": {"recall@5": False, "recall@10": False},
        },
        {
            "pipeline": "weak",
            "verdict": "correct",
            "has_gold": False,
            "recall": {"recall@5": False, "recall@10": False},
        },
    ]
    raw = tmp_path / "raw.jsonl"
    with raw.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    summary = {r["pipeline"]: r for r in runner._summarise(raw)}
    assert summary["weak"]["answered_given_retrieved"] == 0.0
