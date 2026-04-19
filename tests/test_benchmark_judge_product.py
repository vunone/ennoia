"""Unit tests for the product-recommendation judge."""

from __future__ import annotations

import json
from typing import Any

from benchmark.eval import judge as judge_mod
from benchmark.eval.judge import _parse_verdict, judge_answer, make_judge_llm


class _FakeLLM:
    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.prompts: list[str] = []

    async def complete_text(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._reply


def test_parse_verdict_plain_json() -> None:
    raw = json.dumps({"verdict": "correct", "rationale": "spot on"})
    assert _parse_verdict(raw) == {"verdict": "correct", "rationale": "spot on"}


def test_parse_verdict_fenced_json_block() -> None:
    raw = "```json\n" + json.dumps({"verdict": "partial", "rationale": "close"}) + "\n```"
    assert _parse_verdict(raw) == {"verdict": "partial", "rationale": "close"}


def test_parse_verdict_unknown_verdict_maps_to_hallucinated() -> None:
    raw = json.dumps({"verdict": "brilliant", "rationale": "n/a"})
    assert _parse_verdict(raw)["verdict"] == "hallucinated"


def test_parse_verdict_unparseable_reports_hallucinated() -> None:
    result = _parse_verdict("totally not json")
    assert result["verdict"] == "hallucinated"
    assert result["rationale"].startswith("unparseable:")


def test_parse_verdict_invalid_json_reports_hallucinated() -> None:
    result = _parse_verdict('{"verdict": "correct", "rationale": "oops')
    assert result["verdict"] == "hallucinated"


def test_parse_verdict_truncates_rationale() -> None:
    long = "r" * 400
    raw = json.dumps({"verdict": "correct", "rationale": long})
    result = _parse_verdict(raw)
    assert len(result["rationale"]) == 300


async def test_judge_answer_includes_gold_title_brand_bullets() -> None:
    llm = _FakeLLM(json.dumps({"verdict": "correct", "rationale": "match"}))
    gold: dict[str, Any] = {
        "docid": "SECRET_DOC_ID",
        "title": "Laptop Stand",
        "brand": "ACME",
        "bullet_points": ["aluminium", "foldable"],
    }
    retrieved = [{"docid": "d1", "title": "Laptop Stand"}]
    await judge_answer(
        question="aluminium laptop stand",
        gold_product=gold,
        retrieved_top_k=retrieved,
        candidate_answer='{"docid":"d1","reason":"best match"}',
        llm=llm,  # type: ignore[arg-type]
    )
    prompt = llm.prompts[0]
    assert "Laptop Stand" in prompt
    assert "ACME" in prompt
    assert "aluminium" in prompt
    # Gold docid must not leak into the judge prompt — it would let the
    # judge cheat against the candidate's suggested docid string-match.
    assert "SECRET_DOC_ID" not in prompt


async def test_judge_answer_handles_abstention() -> None:
    llm = _FakeLLM(json.dumps({"verdict": "abstained", "rationale": "said NOT_FOUND"}))
    result = await judge_answer(
        question="q",
        gold_product={"title": "T", "brand": "B", "bullet_points": []},
        retrieved_top_k=[],
        candidate_answer="NOT_FOUND",
        llm=llm,  # type: ignore[arg-type]
    )
    assert result["verdict"] == "abstained"


async def test_judge_answer_empty_retrieval_rendered() -> None:
    llm = _FakeLLM(json.dumps({"verdict": "hallucinated", "rationale": "no hits"}))
    await judge_answer(
        question="q",
        gold_product={"title": "T", "brand": "B", "bullet_points": []},
        retrieved_top_k=[],
        candidate_answer="made up",
        llm=llm,  # type: ignore[arg-type]
    )
    assert "(no results retrieved)" in llm.prompts[0]


async def test_judge_answer_empty_bullets_rendered() -> None:
    llm = _FakeLLM(json.dumps({"verdict": "correct", "rationale": "ok"}))
    await judge_answer(
        question="q",
        gold_product={"title": "T", "brand": "B", "bullet_points": []},
        retrieved_top_k=[{"docid": "d1", "title": "X"}],
        candidate_answer="ok",
        llm=llm,  # type: ignore[arg-type]
    )
    assert "- (none)" in llm.prompts[0]


def test_make_judge_llm_returns_openrouter_adapter() -> None:
    adapter = make_judge_llm()
    assert adapter.__class__.__name__ == "OpenRouterAdapter"
    assert adapter.model  # has a non-empty handle


async def test_judge_answer_empty_candidate_shown_as_placeholder() -> None:
    llm = _FakeLLM(json.dumps({"verdict": "abstained", "rationale": "empty"}))
    await judge_answer(
        question="q",
        gold_product={"title": "T", "brand": "B", "bullet_points": ["a"]},
        retrieved_top_k=[{"docid": "d1", "title": "X"}],
        candidate_answer="",
        llm=llm,  # type: ignore[arg-type]
    )
    assert "(empty)" in llm.prompts[0]


def test_strip_fences_noop_on_plain_text() -> None:
    assert judge_mod._strip_fences("plain text") == "plain text"


def test_strip_fences_trims_trailing_backticks_mid_text() -> None:
    # Raw form some free models emit: "```{...}```more noise"
    # After strip("`") the leading backticks go, the trailing backticks stop
    # at "more", so a ``` survives inside and gets split off.
    raw = '```{"verdict":"correct","rationale":"r"}```more'
    cleaned = judge_mod._strip_fences(raw)
    assert cleaned.endswith("}")
    assert "more" not in cleaned


def test_parse_verdict_invalid_json_inside_braces() -> None:
    result = _parse_verdict("{broken: no quotes}")
    assert result["verdict"] == "hallucinated"
    assert "unparseable:" in result["rationale"]
