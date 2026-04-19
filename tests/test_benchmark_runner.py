"""Unit tests for the benchmark runner orchestration + summary."""

from __future__ import annotations

import asyncio
import csv
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from benchmark import runner as runner_mod
from benchmark.data.prep import Dataset, Product, QueryCase
from benchmark.eval.cost import BudgetExceeded, RunningCost
from benchmark.pipelines.base import PipelineRun


class _RecordingPipeline:
    def __init__(self, name: str, retrieved: list[str], answer: str) -> None:
        self.name = name
        self._retrieved = retrieved
        self._answer = answer
        self.indexed: list[list[Product]] = []
        self.queries: list[str] = []

    async def index_corpus(self, products: list[Product]) -> None:
        self.indexed.append(products)

    async def answer(self, query: str) -> PipelineRun:
        self.queries.append(query)
        return PipelineRun(
            retrieved_source_ids=list(self._retrieved),
            answer=self._answer,
            prompt_tokens=10,
            completion_tokens=5,
        )


class _FakeJudge:
    def __init__(self, verdict: str) -> None:
        self.verdict = verdict
        self.calls = 0

    async def complete_text(self, prompt: str) -> str:
        self.calls += 1
        return json.dumps({"verdict": self.verdict, "rationale": "ok"})


def _dataset() -> Dataset:
    products: list[Product] = [
        {
            "docid": "p0",
            "title": "Stand A",
            "text": "",
            "bullet_points": ["aluminium"],
            "brand": "A",
            "color": "silver",
            "price_usd": 40,
        },
        {
            "docid": "p1",
            "title": "Stand B",
            "text": "",
            "bullet_points": [],
            "brand": "B",
            "color": "black",
            "price_usd": 90,
        },
    ]
    queries: list[QueryCase] = [
        {
            "docid": "p0",
            "broad": "stand under $50",
            "medium": "laptop stand",
            "high": "A aluminium stand",
            "price_band": "broad",
        },
        {
            "docid": "p1",
            "broad": "stand",
            "medium": "laptop stand under $100",
            "high": "B black stand",
            "price_band": "medium",
        },
    ]
    return {
        "products": products,
        "queries": queries,
        "seed": 42,
        "num_samples": 2,
        "start_offset": 0,
        "model": "test",
    }


# -- iter_cases --------------------------------------------------------------


def test_iter_cases_expands_three_bands_per_query() -> None:
    cases = runner_mod._iter_cases(_dataset())
    assert len(cases) == 6  # 2 products x 3 bands
    bands = [b for _, _, b, _ in cases]
    assert bands.count("broad") == 2
    assert bands.count("medium") == 2
    assert bands.count("high") == 2


def test_iter_cases_skips_queries_without_matching_product() -> None:
    ds = _dataset()
    ds["queries"].append(
        {
            "docid": "missing",
            "broad": "q",
            "medium": "q",
            "high": "q",
            "price_band": "broad",
        }
    )
    cases = runner_mod._iter_cases(ds)
    assert len(cases) == 6  # orphan query dropped


# -- process_case ------------------------------------------------------------


async def test_process_case_scores_and_emits_one_record_per_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    judge = _FakeJudge("correct")
    pipelines = [
        _RecordingPipeline("ennoia", retrieved=["p0", "p1"], answer='{"docid":"p0"}'),
        _RecordingPipeline("langchain", retrieved=["p1"], answer='{"docid":"p1"}'),
    ]
    ds = _dataset()
    gold = ds["products"][0]
    case = ds["queries"][0]
    cost = RunningCost(max_usd=10.0)
    sem = asyncio.Semaphore(2)
    titles = {p["docid"]: p["title"] for p in ds["products"]}

    records = await runner_mod._process_case(
        case, gold, "high", case["high"], pipelines, judge, cost, sem, titles
    )
    assert len(records) == 2
    ennoia = next(r for r in records if r["pipeline"] == "ennoia")
    assert ennoia["precision_at_5"] == 1 / 5
    assert ennoia["hit_at_5"] is True
    assert ennoia["verdict"] == "correct"
    lc = next(r for r in records if r["pipeline"] == "langchain")
    assert lc["precision_at_5"] == 0.0  # p0 not in retrieved
    assert lc["hit_at_5"] is False


async def test_process_case_drops_failing_pipeline_without_abort() -> None:
    class _Boom:
        name = "boom"

        async def index_corpus(self, products: list[Product]) -> None: ...

        async def answer(self, query: str) -> PipelineRun:
            raise RuntimeError("down")

    ok = _RecordingPipeline("ok", retrieved=["p0"], answer='{"docid":"p0"}')
    judge = _FakeJudge("correct")
    ds = _dataset()
    cost = RunningCost(max_usd=10.0)
    sem = asyncio.Semaphore(2)
    titles = {p["docid"]: p["title"] for p in ds["products"]}
    records = await runner_mod._process_case(
        ds["queries"][0],
        ds["products"][0],
        "broad",
        "stand",
        [_Boom(), ok],  # type: ignore[list-item]
        judge,
        cost,
        sem,
        titles,
    )
    assert len(records) == 1
    assert records[0]["pipeline"] == "ok"


async def test_process_case_budget_exceeded_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from benchmark import config as config_mod

    monkeypatch.setitem(config_mod.PRICING, config_mod.MODEL_LLM, {"input": 1.0, "output": 0.0})
    pipe = _RecordingPipeline("ennoia", retrieved=["p0"], answer="NOT_FOUND")
    # Force a non-trivial token count so the charge is non-zero.
    pipe = _RecordingPipeline("ennoia", retrieved=["p0"], answer="NOT_FOUND")
    judge = _FakeJudge("abstained")
    cost = RunningCost(max_usd=0.0)
    sem = asyncio.Semaphore(1)
    ds = _dataset()
    titles = {p["docid"]: p["title"] for p in ds["products"]}
    with pytest.raises(BudgetExceeded):
        await runner_mod._process_case(
            ds["queries"][0],
            ds["products"][0],
            "broad",
            "stand",
            [pipe],
            judge,
            cost,
            sem,
            titles,
        )


# -- summarise + write_summary ----------------------------------------------


def test_summarise_groups_by_pipeline_and_difficulty(tmp_path: Path) -> None:
    raw = tmp_path / "raw.jsonl"
    rows = [
        {
            "docid": "p0",
            "difficulty": "broad",
            "query": "q",
            "pipeline": "ennoia",
            "retrieved_docids": ["p0"],
            "answer": "a",
            "verdict": "correct",
            "rationale": "r",
            "precision_at_5": 0.2,
            "precision_at_10": 0.1,
            "hit_at_5": True,
            "hit_at_10": True,
            "trace": [],
        },
        {
            "docid": "p0",
            "difficulty": "broad",
            "query": "q",
            "pipeline": "ennoia",
            "retrieved_docids": [],
            "answer": "a",
            "verdict": "partial",
            "rationale": "r",
            "precision_at_5": 0.0,
            "precision_at_10": 0.0,
            "hit_at_5": False,
            "hit_at_10": False,
            "trace": [],
        },
        {
            "docid": "p0",
            "difficulty": "high",
            "query": "q",
            "pipeline": "langchain",
            "retrieved_docids": ["p0"],
            "answer": "a",
            "verdict": "hallucinated",
            "rationale": "r",
            "precision_at_5": 0.2,
            "precision_at_10": 0.1,
            "hit_at_5": True,
            "hit_at_10": True,
            "trace": [],
        },
    ]
    raw.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out = runner_mod._summarise(raw)
    by_key = {(r["pipeline"], r["difficulty"]): r for r in out}
    assert by_key[("ennoia", "broad")]["correct"] == 0.5
    assert by_key[("ennoia", "broad")]["partial"] == 0.5
    assert by_key[("ennoia", "broad")]["precision@5"] == 0.1
    assert by_key[("ennoia", "broad")]["hit@5"] == 0.5
    assert by_key[("langchain", "high")]["hallucinated"] == 1.0


def test_write_summary_writes_csv(tmp_path: Path) -> None:
    out = tmp_path / "s.csv"
    runner_mod._write_summary(out, [{"pipeline": "ennoia", "difficulty": "broad", "correct": 1.0}])
    with out.open() as fh:
        reader = list(csv.DictReader(fh))
    assert reader == [{"pipeline": "ennoia", "difficulty": "broad", "correct": "1.0"}]


def test_write_summary_noop_on_empty(tmp_path: Path) -> None:
    out = tmp_path / "s.csv"
    runner_mod._write_summary(out, [])
    assert not out.exists()


# -- dry-run path + main -----------------------------------------------------


def test_synthetic_dataset_shape() -> None:
    ds = runner_mod._synthetic_dataset(3)
    assert len(ds["products"]) >= 3
    assert len(ds["queries"]) == len(ds["products"])
    assert all({"broad", "medium", "high"} <= set(q.keys()) for q in ds["queries"])


def test_write_dryrun_records_produces_six_records_per_query(tmp_path: Path) -> None:
    ds = _dataset()
    out = tmp_path / "raw.jsonl"
    runner_mod._write_dryrun_records(out, ds, limit=2)
    lines = out.read_text().strip().splitlines()
    # 2 queries x 3 bands x 2 pipelines = 12
    assert len(lines) == 12
    first = json.loads(lines[0])
    assert first["pipeline"] in ("ennoia", "langchain")
    assert first["difficulty"] in ("broad", "medium", "high")


def test_write_dryrun_records_skips_missing_gold(tmp_path: Path) -> None:
    ds = _dataset()
    ds["queries"].append(
        {
            "docid": "missing",
            "broad": "a",
            "medium": "b",
            "high": "c",
            "price_band": "broad",
        }
    )
    out = tmp_path / "raw.jsonl"
    runner_mod._write_dryrun_records(out, ds, limit=5)
    for line in out.read_text().splitlines():
        row = json.loads(line)
        assert row["docid"] in ("p0", "p1")


def test_write_dryrun_records_emits_all_four_verdicts(tmp_path: Path) -> None:
    """Synthesise enough cases that every verdict branch is exercised."""
    # Many products → large enough sample space to cover all four verdicts.
    products: list[Product] = [
        {
            "docid": f"p{i}",
            "title": f"T{i}",
            "text": "",
            "bullet_points": [],
            "brand": "b",
            "color": "c",
            "price_usd": 20 + i,
        }
        for i in range(50)
    ]
    queries: list[QueryCase] = [
        {
            "docid": p["docid"],
            "broad": "a",
            "medium": "b",
            "high": "c",
            "price_band": "broad",
        }
        for p in products
    ]
    ds: Dataset = {
        "products": products,
        "queries": queries,
        "seed": 42,
        "num_samples": 50,
        "start_offset": 0,
        "model": "m",
    }
    out = tmp_path / "raw.jsonl"
    runner_mod._write_dryrun_records(out, ds, limit=50)
    verdicts = {json.loads(line)["verdict"] for line in out.read_text().splitlines()}
    assert {"correct", "partial", "abstained", "hallucinated"} <= verdicts


def test_main_dry_run_writes_summary_and_chart(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runner_mod, "RESULTS_DIR", tmp_path)
    rc = runner_mod.main(["--dry-run", "--num-samples=2", "--chart"])
    assert rc == 0
    assert (tmp_path / "summary_latest.csv").exists()
    assert (tmp_path / "raw_latest.jsonl").exists()
    assert (tmp_path / "chart_latest.png").exists()


def test_main_live_blocks_on_estimate_over_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runner_mod, "RESULTS_DIR", tmp_path)
    ds = _dataset()
    dataset_path = tmp_path / "ds.json"
    dataset_path.write_text(json.dumps(ds), encoding="utf-8")

    def fake_estimate(_products: int, _queries: int, **_: Any) -> dict[str, float]:
        return {
            "extraction_usd": 0.0,
            "embedding_usd": 0.0,
            "qa_usd": 0.0,
            "judge_usd": 0.0,
            "total_usd": 999.0,
        }

    monkeypatch.setattr(runner_mod, "estimate_run_cost", fake_estimate)
    rc = runner_mod.main(
        [
            f"--dataset={dataset_path}",
            "--num-samples=1",
            "--max-cost-usd=1",
        ]
    )
    assert rc == 2


def test_make_judge_uses_openrouter() -> None:
    adapter = runner_mod.make_judge_llm()
    assert adapter.__class__.__name__ == "OpenRouterAdapter"


# -- live orchestration ------------------------------------------------------


async def test_run_live_drives_both_semaphores(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ds = _dataset()

    class _FakeEnnoia:
        def __init__(self) -> None:
            self.name = "ennoia"

        async def index_corpus(self, products: list[Product]) -> None:
            return None

        async def answer(self, query: str) -> PipelineRun:
            return PipelineRun(retrieved_source_ids=["p0", "p1"], answer="NOT_FOUND")

    class _FakeLC:
        def __init__(self) -> None:
            self.name = "langchain"

        async def index_corpus(self, products: list[Product]) -> None:
            return None

        async def answer(self, query: str) -> PipelineRun:
            return PipelineRun(retrieved_source_ids=["p1"], answer="NOT_FOUND")

    monkeypatch.setattr(runner_mod, "_build_pipelines", lambda only: [_FakeEnnoia(), _FakeLC()])
    monkeypatch.setattr(runner_mod, "make_judge_llm", lambda: _FakeJudge("abstained"))

    raw = tmp_path / "raw.jsonl"
    cost = RunningCost(max_usd=10.0)
    await runner_mod._run_live(
        dataset=ds,
        only=None,
        skip_index=False,
        cost=cost,
        raw_path=raw,
        threads=2,
    )
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    # 2 products * 3 bands * 2 pipelines = 12 rows
    assert len(rows) == 12


async def test_run_live_stops_on_budget_exceeded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ds = _dataset()

    class _ExpensivePipe:
        name = "greedy"

        async def index_corpus(self, products: list[Product]) -> None:
            return None

        async def answer(self, query: str) -> PipelineRun:
            return PipelineRun(
                retrieved_source_ids=["p0"],
                answer="",
                prompt_tokens=10_000_000,
                completion_tokens=0,
            )

    # Charge the MODEL_LLM model with a non-zero price so the budget trips
    # on the first call.
    from benchmark import config as config_mod

    monkeypatch.setitem(config_mod.PRICING, config_mod.MODEL_LLM, {"input": 1.0, "output": 0.0})
    monkeypatch.setattr(runner_mod, "_build_pipelines", lambda only: [_ExpensivePipe()])
    monkeypatch.setattr(runner_mod, "make_judge_llm", lambda: _FakeJudge("abstained"))

    cost = RunningCost(max_usd=0.01)
    raw = tmp_path / "raw.jsonl"
    await runner_mod._run_live(
        dataset=ds,
        only="ennoia",
        skip_index=True,
        cost=cost,
        raw_path=raw,
        threads=1,
    )
    captured = capsys.readouterr()
    assert "BUDGET STOP" in captured.err or "BUDGET STOP" in captured.out


def test_module_entrypoint_uses_main(monkeypatch: pytest.MonkeyPatch) -> None:
    # Cover the ``python -m benchmark.runner`` path via raise SystemExit.
    monkeypatch.setattr(sys, "argv", ["runner", "--dry-run", "--num-samples=1"])
    # Exercise via the public ``main`` rather than reimporting at __main__.
    rc = runner_mod.main(["--dry-run", "--num-samples=1"])
    assert rc == 0


def test_build_pipelines_selects_by_only(monkeypatch: pytest.MonkeyPatch) -> None:
    # Stub both pipeline constructors so the import doesn't touch
    # network-facing SDK bootstrapping.
    from benchmark.pipelines import ennoia_pipeline, langchain_pipeline

    monkeypatch.setattr(
        ennoia_pipeline, "EnnoiaPipeline", lambda: type("E", (), {"name": "ennoia"})()
    )
    monkeypatch.setattr(
        langchain_pipeline,
        "LangchainPipeline",
        lambda: type("L", (), {"name": "langchain"})(),
    )
    both = runner_mod._build_pipelines(None)
    ennoia_only = runner_mod._build_pipelines("ennoia")
    lc_only = runner_mod._build_pipelines("langchain")
    assert [p.name for p in both] == ["ennoia", "langchain"]
    assert [p.name for p in ennoia_only] == ["ennoia"]
    assert [p.name for p in lc_only] == ["langchain"]


def test_main_live_runs_pipelines_and_writes_chart(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Exercise the live (non-dry-run) branch with stubbed pipelines + chart."""
    monkeypatch.setattr(runner_mod, "RESULTS_DIR", tmp_path)
    ds = _dataset()
    dataset_path = tmp_path / "ds.json"
    dataset_path.write_text(json.dumps(ds), encoding="utf-8")

    class _FakeEnnoia:
        name = "ennoia"

        async def index_corpus(self, products: list[Product]) -> None:
            return None

        async def answer(self, query: str) -> PipelineRun:
            return PipelineRun(retrieved_source_ids=["p0", "p1"], answer="NOT_FOUND")

    class _FakeLC:
        name = "langchain"

        async def index_corpus(self, products: list[Product]) -> None:
            return None

        async def answer(self, query: str) -> PipelineRun:
            return PipelineRun(retrieved_source_ids=["p1"], answer="NOT_FOUND")

    monkeypatch.setattr(runner_mod, "_build_pipelines", lambda only: [_FakeEnnoia(), _FakeLC()])
    monkeypatch.setattr(runner_mod, "make_judge_llm", lambda: _FakeJudge("abstained"))

    rc = runner_mod.main(
        [
            f"--dataset={dataset_path}",
            "--num-samples=2",
            "--threads=2",
            "--chart",
            "--max-cost-usd=10",
        ]
    )
    assert rc == 0
    assert (tmp_path / "chart_latest.png").exists()
    assert (tmp_path / "summary_latest.csv").exists()


# -- --summarize-only + Ctrl+C recovery -------------------------------------


def _write_partial_raw(path: Path, n: int = 3) -> None:
    """Emit ``n`` valid RunRecord lines — mimics an interrupted run."""
    rows = [
        {
            "docid": f"p{i}",
            "difficulty": "broad",
            "query": "q",
            "pipeline": "ennoia",
            "retrieved_docids": [f"p{i}"],
            "answer": "{}",
            "verdict": "correct",
            "rationale": "ok",
            "precision_at_5": 0.2,
            "precision_at_10": 0.1,
            "hit_at_5": True,
            "hit_at_10": True,
            "trace": [],
        }
        for i in range(n)
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_main_summarize_only_uses_latest_raw(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runner_mod, "RESULTS_DIR", tmp_path)
    _write_partial_raw(tmp_path / "raw_latest.jsonl", n=5)
    rc = runner_mod.main(["--summarize-only", "--chart"])
    assert rc == 0
    assert (tmp_path / "summary_latest.csv").exists()
    assert (tmp_path / "chart_latest.png").exists()
    # summarize-only must NOT truncate raw_latest by overwriting onto itself.
    assert (tmp_path / "raw_latest.jsonl").stat().st_size > 0


def test_main_summarize_only_accepts_explicit_raw_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runner_mod, "RESULTS_DIR", tmp_path)
    custom = tmp_path / "partial_run.jsonl"
    _write_partial_raw(custom, n=2)
    rc = runner_mod.main(["--summarize-only", f"--raw={custom}"])
    assert rc == 0
    assert (tmp_path / "summary_latest.csv").exists()


def test_main_summarize_only_fails_when_raw_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runner_mod, "RESULTS_DIR", tmp_path)
    rc = runner_mod.main(["--summarize-only"])
    assert rc == 1


async def test_run_live_recovers_from_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A KeyboardInterrupt mid-run must cancel pending tasks and let
    ``main()`` finalise whatever was flushed before the interrupt.

    We inject the interrupt at the ``await fut`` boundary by wrapping
    ``asyncio.as_completed`` — yielding one real future (to flush a record)
    then a sentinel that raises on await. Raising KeyboardInterrupt from
    inside a task coroutine would trip pytest's signal handler before our
    except clause runs, so this simulates the user-initiated Ctrl+C more
    faithfully.
    """
    ds = _dataset()
    first_record_flushed = asyncio.Event()

    class _NormalPipe:
        name = "ennoia"

        async def index_corpus(self, products: list[Product]) -> None:
            return None

        async def answer(self, query: str) -> PipelineRun:
            first_record_flushed.set()
            return PipelineRun(retrieved_source_ids=["p0"], answer="NOT_FOUND")

    monkeypatch.setattr(runner_mod, "_build_pipelines", lambda only: [_NormalPipe()])
    monkeypatch.setattr(runner_mod, "make_judge_llm", lambda: _FakeJudge("abstained"))

    original_as_completed = runner_mod.asyncio.as_completed

    def wrapped_as_completed(aws: Any, *args: Any, **kwargs: Any) -> Any:
        inner = iter(original_as_completed(aws, *args, **kwargs))

        def gen() -> Any:
            # Yield the first future normally (records flush to disk).
            try:
                yield next(inner)
            except StopIteration:
                return

            async def _raise_interrupt() -> Any:
                raise KeyboardInterrupt("simulated ctrl+c")

            yield _raise_interrupt()

        return gen()

    monkeypatch.setattr(runner_mod.asyncio, "as_completed", wrapped_as_completed)

    cost = RunningCost(max_usd=10.0)
    raw = tmp_path / "raw.jsonl"
    await runner_mod._run_live(
        dataset=ds, only="ennoia", skip_index=True, cost=cost, raw_path=raw, threads=1
    )
    # First record flushed before the interrupt; file closed cleanly.
    lines = [line for line in raw.read_text().splitlines() if line]
    assert len(lines) >= 1
    assert first_record_flushed.is_set()


def test_main_live_handles_keyboard_interrupt_during_indexing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """KeyboardInterrupt escaping ``asyncio.run`` (e.g. during indexing)
    is caught by ``main()`` so the partial raw JSONL still gets summarised."""
    monkeypatch.setattr(runner_mod, "RESULTS_DIR", tmp_path)
    ds = _dataset()
    dataset_path = tmp_path / "ds.json"
    dataset_path.write_text(json.dumps(ds), encoding="utf-8")

    # Stage the raw_<stamp>.jsonl that _run_live would have produced by
    # having _run_live raise immediately and pre-seeding the expected file.
    def fake_run_live(**kwargs: Any) -> Any:
        kwargs["raw_path"].write_text(
            json.dumps(
                {
                    "docid": "p0",
                    "difficulty": "broad",
                    "query": "q",
                    "pipeline": "ennoia",
                    "retrieved_docids": ["p0"],
                    "answer": "a",
                    "verdict": "correct",
                    "rationale": "r",
                    "precision_at_5": 0.2,
                    "precision_at_10": 0.1,
                    "hit_at_5": True,
                    "hit_at_10": True,
                    "trace": [],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        raise KeyboardInterrupt()

    async def _async_wrapper(**kwargs: Any) -> None:
        fake_run_live(**kwargs)

    monkeypatch.setattr(runner_mod, "_run_live", _async_wrapper)

    rc = runner_mod.main(
        [
            f"--dataset={dataset_path}",
            "--num-samples=1",
            "--max-cost-usd=10",
            "--chart",
        ]
    )
    assert rc == 0
    assert (tmp_path / "summary_latest.csv").exists()
    assert (tmp_path / "chart_latest.png").exists()
