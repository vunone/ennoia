"""Unit tests for the temporary pooled-judge rescoring script."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path
from typing import Any

import pytest

from benchmark import rescore
from benchmark.rescore import (
    AnswerQrel,
    JudgeCache,
    JudgeDeps,
    ParseFailure,
    Qrel,
    Record,
    _answer_key,
    _format_bullets,
    _format_pool_block,
    _relevance_key,
    _should_report,
    _to_answer_qrel,
    _to_qrel,
    build_parser,
    build_pool,
    compute_metrics,
    difficulty_index,
    extract_chosen_docid,
    hit_at_k,
    load_records,
    main,
    mrr_at_k,
    ndcg_at_k,
    parse_json,
    precision_at_k,
    print_summary_table,
    recall_at_k,
    render_chart,
    run_pass_1,
    run_pass_2,
    write_answers_jsonl,
    write_qrels_jsonl,
    write_summary_csv,
)

# ---------------------------------------------------------------------------
# Fakes.
# ---------------------------------------------------------------------------


class _FakeClient:
    """Stand-in for :class:`OpenRouterAdapter` — returns a scripted queue."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []

    async def complete_text(self, prompt: str) -> str:
        self.calls.append(prompt)
        if not self._responses:
            raise AssertionError("FakeClient ran out of scripted responses")
        return self._responses.pop(0)


def _fake_catalog() -> dict[str, dict[str, Any]]:
    return {
        "P1": {
            "docid": "P1",
            "title": "Acme Parrot Plush",
            "brand": "Acme",
            "price_usd": 20,
            "text": "A soft parrot plushie.",
            "bullet_points": ["Soft", "Colorful"],
        },
        "P2": {
            "docid": "P2",
            "title": "Acme Penguin Plush",
            "brand": "Acme",
            "price_usd": 25,
            "text": "",
            "bullet_points": [],
        },
        "P3": {
            "docid": "P3",
            "title": "Dog chew bone",
            "brand": "RuffCo",
            "price_usd": 10,
            "text": "Durable chew toy",
            "bullet_points": [],
        },
    }


def _make_deps(cache: JudgeCache, responses: list[str]) -> tuple[JudgeDeps, _FakeClient]:
    client = _FakeClient(responses)
    return (
        JudgeDeps(
            cache=cache,
            client=client,  # type: ignore[arg-type]
            model="test-model",
            semaphore=asyncio.Semaphore(4),
        ),
        client,
    )


# ---------------------------------------------------------------------------
# get_product — lazy dataset load + KeyError.
# ---------------------------------------------------------------------------


def test_get_product_lazy_loads_from_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = {"products": [{"docid": "A1", "title": "thing"}]}
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")
    monkeypatch.setattr(rescore, "DATASET_PATH", path)
    monkeypatch.setattr(rescore, "_PRODUCT_CACHE", None)

    assert rescore.get_product("A1")["title"] == "thing"
    # Second call uses in-memory cache (no second read).
    path.unlink()
    assert rescore.get_product("A1")["title"] == "thing"
    with pytest.raises(KeyError):
        rescore.get_product("missing")


# ---------------------------------------------------------------------------
# Loaders / pool.
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
        fh.write("\n")  # trailing blank line to exercise the skip branch
    return path


def test_load_records_skips_blank_lines(tmp_path: Path) -> None:
    path = _write_jsonl(
        tmp_path / "r.jsonl",
        [{"docid": "A", "query": "q", "pipeline": "x", "difficulty": "broad"}],
    )
    out = load_records(path)
    assert len(out) == 1
    assert out[0]["docid"] == "A"


def test_build_pool_unions_and_injects_gold() -> None:
    records: list[Record] = [
        {
            "docid": "G",
            "query": "q1",
            "pipeline": "ennoia",
            "difficulty": "broad",
            "retrieved_docids": ["A", "B"],
        },
        {
            "docid": "G",
            "query": "q1",
            "pipeline": "langchain",
            "difficulty": "broad",
            "retrieved_docids": ["B", "C"],
        },
        {
            "docid": "H",
            "query": "q2",
            "pipeline": "ennoia",
            "difficulty": "high",
            "retrieved_docids": [],
        },
    ]
    pool = build_pool(records)
    assert pool["q1"] == {"A", "B", "C", "G"}
    assert pool["q2"] == {"H"}


def test_difficulty_index_maps_every_query() -> None:
    records: list[Record] = [
        {"docid": "G", "query": "q1", "pipeline": "a", "difficulty": "broad"},
        {"docid": "G", "query": "q1", "pipeline": "b", "difficulty": "broad"},
        {"docid": "H", "query": "q2", "pipeline": "a", "difficulty": "high"},
    ]
    assert difficulty_index(records) == {"q1": "broad", "q2": "high"}


# ---------------------------------------------------------------------------
# JudgeCache.
# ---------------------------------------------------------------------------


def test_judge_cache_load_nonexistent_is_noop(tmp_path: Path) -> None:
    cache = JudgeCache(tmp_path / "cache.jsonl")
    cache.load()
    assert len(cache) == 0


def test_judge_cache_round_trip_and_corrupt_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "cache.jsonl"
    path.write_text(
        '{"key": "k1", "kind": "relevance", "response": {"relevance": "relevant"}}\n'
        "\n"
        "NOT_JSON_AT_ALL\n"
        '{"no_key": true}\n'
        '{"key": "k2", "kind": "answer", "response": {"answer_status": "correct"}}\n',
        encoding="utf-8",
    )
    cache = JudgeCache(path)
    cache.load()
    err = capsys.readouterr().err
    assert "skip corrupt cache line" in err
    assert len(cache) == 2
    assert cache.get("k1") == {"relevance": "relevant"}
    assert cache.get("missing") is None

    async def _put() -> None:
        await cache.put("k3", "relevance", {"relevance": "not_relevant"})

    asyncio.run(_put())
    assert cache.get("k3") == {"relevance": "not_relevant"}
    # The new entry was appended, preserving earlier lines.
    saved = path.read_text(encoding="utf-8").splitlines()
    assert any('"k3"' in ln for ln in saved)


# ---------------------------------------------------------------------------
# Keys are deterministic.
# ---------------------------------------------------------------------------


def test_cache_keys_stable() -> None:
    assert _relevance_key("q", "d", "broad", "m") == _relevance_key("q", "d", "broad", "m")
    assert _relevance_key("q", "d", "broad", "m") != _relevance_key("q", "d", "high", "m")

    record: Record = {
        "docid": "G",
        "query": "q",
        "pipeline": "ennoia",
        "difficulty": "broad",
        "answer": "",
    }
    assert _answer_key(record, "m") == _answer_key(record, "m")


# ---------------------------------------------------------------------------
# Parsing.
# ---------------------------------------------------------------------------


def test_parse_json_plain() -> None:
    assert parse_json('{"relevance": "relevant"}') == {"relevance": "relevant"}


def test_parse_json_fenced() -> None:
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_parse_json_plain_fence() -> None:
    assert parse_json('```\n{"a": 2}\n```') == {"a": 2}


def test_parse_json_trailing_prose() -> None:
    assert parse_json('thoughts...\n{"a": 3}\nmore prose') == {"a": 3}


def test_parse_json_no_object_raises() -> None:
    with pytest.raises(ParseFailure):
        parse_json("no braces here")


def test_parse_json_invalid_raises() -> None:
    with pytest.raises(ParseFailure):
        parse_json('{"bad": unquoted}')


def test_parse_json_rejects_bare_array() -> None:
    # Array syntax has no ``{...}`` so the object regex finds nothing.
    with pytest.raises(ParseFailure):
        parse_json("[1, 2, 3]")


def test_extract_chosen_docid_variants() -> None:
    assert extract_chosen_docid('{"docid": "X"}') == "X"
    assert extract_chosen_docid("NOT_FOUND") is None
    assert extract_chosen_docid('{"docid": ""}') is None
    assert extract_chosen_docid('{"docid": 42}') is None


def test_to_qrel_coerces_invalid_fields() -> None:
    q = _to_qrel({"relevance": "weird", "constraints_satisfied": "maybe"})
    assert q["relevance"] == "not_relevant"
    assert q["constraints_satisfied"] == "n/a"

    q2 = _to_qrel({"relevance": "relevant", "constraints_satisfied": True, "reasoning": "ok"})
    assert q2["constraints_satisfied"] is True
    assert q2["reasoning"] == "ok"


def test_to_answer_qrel_coerces_invalid_status() -> None:
    assert _to_answer_qrel({"answer_status": "invalid"})["answer_status"] == "abstained"
    ok = _to_answer_qrel({"answer_status": "correct", "reasoning": "r"})
    assert ok["answer_status"] == "correct"


# ---------------------------------------------------------------------------
# Formatters.
# ---------------------------------------------------------------------------


def test_format_bullets_empty_and_populated() -> None:
    assert _format_bullets([]) == "- (none)"
    rendered = _format_bullets(["one", "two"])
    assert "- one" in rendered and "- two" in rendered


def test_format_pool_block_handles_missing_catalog() -> None:
    catalog = _fake_catalog()

    def gp(docid: str) -> dict[str, Any]:
        return catalog[docid]

    labels: dict[str, Qrel] = {
        "P1": {"relevance": "highly_relevant", "constraints_satisfied": True, "reasoning": ""},
        "MISSING": {"relevance": "not_relevant", "constraints_satisfied": "n/a", "reasoning": ""},
    }
    block = _format_pool_block(labels, gp)
    assert "docid=P1" in block and "brand=Acme" in block
    assert "docid=MISSING" in block and "brand=?" in block and "<unknown>" in block

    assert _format_pool_block({}, gp) == "(empty pool)"


# ---------------------------------------------------------------------------
# Judge calls.
# ---------------------------------------------------------------------------


def test_judge_relevance_cache_hit_skips_client(tmp_path: Path) -> None:
    cache = JudgeCache(tmp_path / "c.jsonl")

    async def _prime() -> None:
        await cache.put(
            _relevance_key("q", "P1", "broad", "test-model"),
            "relevance",
            {
                "relevance": "highly_relevant",
                "constraints_satisfied": True,
                "reasoning": "cached",
            },
        )

    asyncio.run(_prime())

    deps, client = _make_deps(cache, [])

    qrel = asyncio.run(
        rescore.judge_relevance("q", "P1", "broad", deps, _fake_catalog().__getitem__)
    )
    assert qrel["relevance"] == "highly_relevant"
    assert client.calls == []


def test_judge_relevance_missing_catalog_marks_not_relevant(tmp_path: Path) -> None:
    cache = JudgeCache(tmp_path / "c.jsonl")
    deps, client = _make_deps(cache, [])

    def gp(_: str) -> dict[str, Any]:
        raise KeyError("absent")

    qrel = asyncio.run(rescore.judge_relevance("q", "GONE", "medium", deps, gp))
    assert qrel["relevance"] == "not_relevant"
    assert "not in catalogue" in qrel["reasoning"]
    assert client.calls == []  # short-circuits before the LLM


def test_judge_relevance_llm_success(tmp_path: Path) -> None:
    cache = JudgeCache(tmp_path / "c.jsonl")
    reply = json.dumps(
        {"relevance": "highly_relevant", "constraints_satisfied": True, "reasoning": "ok"}
    )
    deps, client = _make_deps(cache, [reply])
    catalog = _fake_catalog()

    qrel = asyncio.run(rescore.judge_relevance("q", "P1", "medium", deps, catalog.__getitem__))
    assert qrel["relevance"] == "highly_relevant"
    assert client.calls  # prompt was issued
    # Cached on disk for next run.
    fresh = JudgeCache(cache._path)  # type: ignore[attr-defined]
    fresh.load()
    assert fresh.get(_relevance_key("q", "P1", "medium", "test-model")) is not None


def test_judge_relevance_parse_failure_fallback(tmp_path: Path) -> None:
    cache = JudgeCache(tmp_path / "c.jsonl")
    deps, _client = _make_deps(cache, ["not json", "still not json"])
    qrel = asyncio.run(
        rescore.judge_relevance("q", "P1", "high", deps, _fake_catalog().__getitem__)
    )
    assert qrel["relevance"] == "not_relevant"
    assert "parse-failure" in qrel["reasoning"]


def test_llm_json_retry_then_success(tmp_path: Path) -> None:
    cache = JudgeCache(tmp_path / "c.jsonl")
    reply_ok = json.dumps({"answer_status": "correct", "reasoning": "ok"})
    deps, client = _make_deps(cache, ["garbage", reply_ok])
    out = asyncio.run(rescore._llm_json("prompt", deps))
    assert out == {"answer_status": "correct", "reasoning": "ok"}
    # Two calls: original prompt + nudge variant.
    assert len(client.calls) == 2
    assert "Return ONLY a valid JSON object" in client.calls[1]


def test_judge_answer_empty_answer_abstains(tmp_path: Path) -> None:
    cache = JudgeCache(tmp_path / "c.jsonl")
    deps, client = _make_deps(cache, [])
    record: Record = {
        "docid": "G",
        "query": "q",
        "pipeline": "ennoia",
        "difficulty": "broad",
        "answer": "",
    }
    verdict = asyncio.run(rescore.judge_answer(record, {}, deps, _fake_catalog().__getitem__))
    assert verdict["answer_status"] == "abstained"
    assert client.calls == []


def test_judge_answer_not_found_abstains(tmp_path: Path) -> None:
    cache = JudgeCache(tmp_path / "c.jsonl")
    deps, client = _make_deps(cache, [])
    record: Record = {
        "docid": "G",
        "query": "q",
        "pipeline": "ennoia",
        "difficulty": "broad",
        "answer": "NOT_FOUND",
    }
    verdict = asyncio.run(rescore.judge_answer(record, {}, deps, _fake_catalog().__getitem__))
    assert verdict["answer_status"] == "abstained"
    assert client.calls == []


def test_judge_answer_cache_hit(tmp_path: Path) -> None:
    cache = JudgeCache(tmp_path / "c.jsonl")
    record: Record = {
        "docid": "G",
        "query": "q",
        "pipeline": "ennoia",
        "difficulty": "broad",
        "answer": '{"docid": "P1", "reason": "ok"}',
    }

    async def _prime() -> None:
        await cache.put(
            _answer_key(record, "test-model"),
            "answer",
            {"answer_status": "correct", "reasoning": "cached"},
        )

    asyncio.run(_prime())
    deps, client = _make_deps(cache, [])
    verdict = asyncio.run(rescore.judge_answer(record, {}, deps, _fake_catalog().__getitem__))
    assert verdict["answer_status"] == "correct"
    assert client.calls == []


def test_judge_answer_llm_success(tmp_path: Path) -> None:
    cache = JudgeCache(tmp_path / "c.jsonl")
    reply = json.dumps({"answer_status": "partial", "reasoning": "close"})
    deps, client = _make_deps(cache, [reply])
    record: Record = {
        "docid": "G",
        "query": "q",
        "pipeline": "ennoia",
        "difficulty": "medium",
        "answer": '{"docid": "P1", "reason": "ok"}',
    }
    pool_labels: dict[str, Qrel] = {
        "P1": {"relevance": "relevant", "constraints_satisfied": False, "reasoning": ""},
    }
    gp = _fake_catalog().__getitem__
    verdict = asyncio.run(rescore.judge_answer(record, pool_labels, deps, gp))
    assert verdict["answer_status"] == "partial"
    assert client.calls


def test_judge_answer_parse_failure_abstains(tmp_path: Path) -> None:
    cache = JudgeCache(tmp_path / "c.jsonl")
    deps, _ = _make_deps(cache, ["nope", "still nope"])
    record: Record = {
        "docid": "G",
        "query": "q",
        "pipeline": "ennoia",
        "difficulty": "medium",
        "answer": '{"docid": "P1"}',
    }
    verdict = asyncio.run(rescore.judge_answer(record, {}, deps, _fake_catalog().__getitem__))
    assert verdict["answer_status"] == "abstained"
    assert "parse-failure" in verdict["reasoning"]


# ---------------------------------------------------------------------------
# Pass orchestration.
# ---------------------------------------------------------------------------


def test_should_report_both_branches() -> None:
    assert _should_report(50, 1000) is True  # modulo hit
    assert _should_report(7, 7) is True  # final tick
    assert _should_report(7, 99) is False  # neither


def _rel(relevance: str, constraints: Any) -> str:
    return json.dumps(
        {"relevance": relevance, "constraints_satisfied": constraints, "reasoning": ""}
    )


def test_run_pass_1_covers_pool(tmp_path: Path) -> None:
    cache = JudgeCache(tmp_path / "c.jsonl")
    # Two queries, two docids each → 4 LLM calls.
    replies = [
        _rel("highly_relevant", True),
        _rel("relevant", "n/a"),
        _rel("not_relevant", False),
        _rel("relevant", True),
    ]
    deps, _ = _make_deps(cache, replies)
    pool = {"q1": {"P1", "P2"}, "q2": {"P1", "P3"}}
    diff = {"q1": "broad", "q2": "medium"}
    labels = asyncio.run(run_pass_1(pool, diff, deps, _fake_catalog().__getitem__))
    assert set(labels) == {"q1", "q2"}
    assert set(labels["q1"]) == {"P1", "P2"}


def test_run_pass_2_covers_records(tmp_path: Path) -> None:
    cache = JudgeCache(tmp_path / "c.jsonl")
    deps, _ = _make_deps(
        cache,
        [
            json.dumps({"answer_status": "correct", "reasoning": ""}),
            json.dumps({"answer_status": "incorrect", "reasoning": ""}),
        ],
    )
    records: list[Record] = [
        {
            "docid": "P1",
            "query": "q1",
            "pipeline": "ennoia",
            "difficulty": "broad",
            "answer": '{"docid": "P1"}',
            "retrieved_docids": ["P1"],
        },
        {
            "docid": "P1",
            "query": "q1",
            "pipeline": "langchain",
            "difficulty": "broad",
            "answer": '{"docid": "P3"}',
            "retrieved_docids": ["P3"],
        },
    ]
    pool_labels: dict[str, dict[str, Qrel]] = {
        "q1": {
            "P1": {"relevance": "highly_relevant", "constraints_satisfied": True, "reasoning": ""},
            "P3": {"relevance": "not_relevant", "constraints_satisfied": "n/a", "reasoning": ""},
        }
    }
    verdicts = asyncio.run(run_pass_2(records, pool_labels, deps, _fake_catalog().__getitem__))
    assert verdicts[("q1", "ennoia")]["answer_status"] == "correct"
    assert verdicts[("q1", "langchain")]["answer_status"] == "incorrect"


# ---------------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------------


def test_metric_helpers_k_zero_and_empty_qrels() -> None:
    qrels: dict[str, Qrel] = {}
    assert precision_at_k(["A"], qrels, 0, strict=False) == 0.0
    assert recall_at_k(["A"], qrels, 5) == 0.0  # no positives
    assert recall_at_k(["A"], qrels, 0) == 0.0
    assert hit_at_k("A", ["A"], 0) is False
    assert mrr_at_k(["A"], qrels, 0) == 0.0
    assert mrr_at_k(["A"], qrels, 5) == 0.0  # no positives
    assert ndcg_at_k(["A"], qrels, 0) == 0.0


def test_precision_and_strict() -> None:
    qrels: dict[str, Qrel] = {
        "A": {"relevance": "highly_relevant", "constraints_satisfied": True, "reasoning": ""},
        "B": {"relevance": "relevant", "constraints_satisfied": "n/a", "reasoning": ""},
        "C": {"relevance": "not_relevant", "constraints_satisfied": "n/a", "reasoning": ""},
    }
    retrieved = ["A", "B", "C"]
    assert precision_at_k(retrieved, qrels, 3, strict=False) == pytest.approx(2 / 3)
    assert precision_at_k(retrieved, qrels, 3, strict=True) == pytest.approx(1 / 3)
    assert recall_at_k(retrieved, qrels, 2) == pytest.approx(1.0)
    assert hit_at_k("B", retrieved, 2) is True
    assert mrr_at_k(retrieved, qrels, 3) == pytest.approx(1.0)


def test_ndcg_matches_manual_computation() -> None:
    qrels: dict[str, Qrel] = {
        "A": {"relevance": "highly_relevant", "constraints_satisfied": True, "reasoning": ""},
        "B": {"relevance": "relevant", "constraints_satisfied": "n/a", "reasoning": ""},
    }
    retrieved = ["B", "A"]
    gains = [1, 2]
    dcg = sum((2**g - 1) / math.log2(idx + 2) for idx, g in enumerate(gains))
    ideal = sorted(gains, reverse=True)
    idcg = sum((2**g - 1) / math.log2(idx + 2) for idx, g in enumerate(ideal))
    assert ndcg_at_k(retrieved, qrels, 2) == pytest.approx(dcg / idcg)


def test_ndcg_zero_when_no_relevant() -> None:
    qrels: dict[str, Qrel] = {
        "X": {"relevance": "not_relevant", "constraints_satisfied": "n/a", "reasoning": ""},
    }
    assert ndcg_at_k(["X"], qrels, 5) == 0.0


def test_compute_metrics_tolerates_missing_verdict() -> None:
    records: list[Record] = [
        {
            "docid": "A",
            "query": "q1",
            "pipeline": "ennoia",
            "difficulty": "broad",
            "retrieved_docids": ["A"],
        }
    ]
    pool_labels: dict[str, dict[str, Qrel]] = {
        "q1": {
            "A": {"relevance": "relevant", "constraints_satisfied": "n/a", "reasoning": ""},
        }
    }
    rows = compute_metrics(records, pool_labels, {})  # no verdicts at all
    assert rows[0]["correct"] == 0.0


def test_compute_metrics_includes_old_and_new(tmp_path: Path) -> None:
    records: list[Record] = [
        {
            "docid": "A",
            "query": "q1",
            "pipeline": "ennoia",
            "difficulty": "broad",
            "retrieved_docids": ["A", "B", "C"],
            "precision_at_5": 0.2,
            "precision_at_10": 0.1,
            "hit_at_5": True,
            "hit_at_10": True,
            "verdict": "correct",
            "answer": '{"docid": "A"}',
        },
        {
            "docid": "A",
            "query": "q1",
            "pipeline": "langchain",
            "difficulty": "broad",
            "retrieved_docids": ["C", "B", "A"],
            "precision_at_5": 0.2,
            "precision_at_10": 0.1,
            "hit_at_5": True,
            "hit_at_10": True,
            "verdict": "correct",
            "answer": '{"docid": "C"}',
        },
    ]
    pool_labels: dict[str, dict[str, Qrel]] = {
        "q1": {
            "A": {"relevance": "highly_relevant", "constraints_satisfied": True, "reasoning": ""},
            "B": {"relevance": "relevant", "constraints_satisfied": "n/a", "reasoning": ""},
            "C": {"relevance": "not_relevant", "constraints_satisfied": "n/a", "reasoning": ""},
        }
    }
    verdicts: dict[tuple[str, str], AnswerQrel] = {
        ("q1", "ennoia"): {"answer_status": "correct", "reasoning": ""},
        ("q1", "langchain"): {"answer_status": "incorrect", "reasoning": ""},
    }
    rows = compute_metrics(records, pool_labels, verdicts)
    ennoia = next(r for r in rows if r["pipeline"] == "ennoia")
    assert ennoia["n"] == 1
    assert ennoia["hit@5"] == 1.0
    assert ennoia["correct"] == 1.0
    assert ennoia["hallucinated"] == 0.0
    assert ennoia["old_precision@5"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Outputs.
# ---------------------------------------------------------------------------


def test_write_outputs_and_chart(tmp_path: Path) -> None:
    records: list[Record] = [
        {
            "docid": "A",
            "query": "q1",
            "pipeline": "ennoia",
            "difficulty": "broad",
            "retrieved_docids": ["A", "B"],
            "precision_at_5": 0.2,
            "precision_at_10": 0.1,
            "hit_at_5": True,
            "hit_at_10": True,
            "verdict": "correct",
            "answer": '{"docid": "A"}',
        },
    ]
    pool_labels: dict[str, dict[str, Qrel]] = {
        "q1": {
            "A": {"relevance": "highly_relevant", "constraints_satisfied": True, "reasoning": ""},
            "B": {"relevance": "relevant", "constraints_satisfied": "n/a", "reasoning": ""},
        }
    }
    verdicts: dict[tuple[str, str], AnswerQrel] = {
        ("q1", "ennoia"): {"answer_status": "correct", "reasoning": ""},
    }
    rows = compute_metrics(records, pool_labels, verdicts)
    ts = "20260419T000000Z"
    summary_path = write_summary_csv(tmp_path, ts, rows)
    assert summary_path.exists()
    latest = tmp_path / "rescored_summary_latest.csv"
    assert latest.exists()
    qrel_path = write_qrels_jsonl(tmp_path, ts, pool_labels)
    assert qrel_path.exists()
    ans_path = write_answers_jsonl(tmp_path, ts, records, verdicts)
    assert ans_path.exists()
    chart_path = render_chart(tmp_path, ts, rows)
    assert chart_path.exists()
    print_summary_table(rows)  # smoke


# ---------------------------------------------------------------------------
# CLI / main.
# ---------------------------------------------------------------------------


def test_default_client_factory_type() -> None:
    client = rescore._default_client_factory("some-model")
    assert client.__class__.__name__ == "OpenRouterAdapter"


def test_build_parser_has_expected_defaults(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args([])
    assert args.dry_run is False
    assert args.judge_model  # non-empty default


def test_main_empty_input_returns_error(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    args = argparse.Namespace(
        input=empty,
        cache=tmp_path / "c.jsonl",
        output_dir=tmp_path / "out",
        judge_model="test-model",
        concurrency=2,
        dry_run=True,
    )
    rc = asyncio.run(main(args))
    assert rc == 1


def test_main_dry_run_does_not_invoke_client(tmp_path: Path) -> None:
    records = [
        {
            "docid": "A",
            "query": "q1",
            "pipeline": "ennoia",
            "difficulty": "broad",
            "retrieved_docids": ["A"],
            "answer": '{"docid": "A"}',
            "precision_at_5": 0.2,
            "precision_at_10": 0.1,
            "hit_at_5": True,
            "hit_at_10": True,
            "verdict": "correct",
        }
    ]
    inp = tmp_path / "in.jsonl"
    with inp.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    args = argparse.Namespace(
        input=inp,
        cache=tmp_path / "c.jsonl",
        output_dir=tmp_path / "out",
        judge_model="test-model",
        concurrency=1,
        dry_run=True,
    )

    def factory(_: str) -> Any:
        raise AssertionError("client factory must not be called in dry-run")

    rc = asyncio.run(main(args, client_factory=factory, get_product_fn=_fake_catalog().__getitem__))
    assert rc == 0


def test_main_full_run_produces_outputs(tmp_path: Path) -> None:
    records = [
        {
            "docid": "P1",
            "query": "q1",
            "pipeline": "ennoia",
            "difficulty": "broad",
            "retrieved_docids": ["P1", "P2"],
            "answer": '{"docid": "P1", "reason": "ok"}',
            "precision_at_5": 0.2,
            "precision_at_10": 0.1,
            "hit_at_5": True,
            "hit_at_10": True,
            "verdict": "correct",
        },
        {
            "docid": "P1",
            "query": "q1",
            "pipeline": "langchain",
            "difficulty": "broad",
            "retrieved_docids": ["P2", "P1"],
            "answer": '{"docid": "P2", "reason": "alt"}',
            "precision_at_5": 0.0,
            "precision_at_10": 0.1,
            "hit_at_5": False,
            "hit_at_10": True,
            "verdict": "partial",
        },
    ]
    inp = tmp_path / "in.jsonl"
    with inp.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    replies = [
        _rel("highly_relevant", True),
        _rel("relevant", "n/a"),
        json.dumps({"answer_status": "correct", "reasoning": ""}),
        json.dumps({"answer_status": "partial", "reasoning": ""}),
    ]
    client = _FakeClient(replies)

    def factory(_: str) -> Any:
        return client

    args = argparse.Namespace(
        input=inp,
        cache=tmp_path / "c.jsonl",
        output_dir=tmp_path / "out",
        judge_model="test-model",
        concurrency=2,
        dry_run=False,
    )
    rc = asyncio.run(main(args, client_factory=factory, get_product_fn=_fake_catalog().__getitem__))
    assert rc == 0
    out_dir = tmp_path / "out"
    assert (out_dir / "rescored_summary_latest.csv").exists()
    assert any(p.name.startswith("rescored_chart_") for p in out_dir.iterdir())
