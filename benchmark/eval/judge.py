"""LLM-as-judge: classify a generated answer against the gold span.

Uses ``gpt-5.4`` (more capable than the generator) so the judge has a
quality buffer over both pipelines under test. Judge prompt explicitly
explains that CUAD gold answers are extractive spans, so paraphrased-but-
correct answers should be ``correct`` rather than ``partial``.
"""

from __future__ import annotations

import re
from typing import Literal, TypedDict

from benchmark.config import JUDGE_TIMEOUT_SEC, MODEL_JUDGE
from ennoia.adapters.llm.openai import OpenAIAdapter

Verdict = Literal["correct", "partial", "hallucinated", "abstained"]


class JudgeResult(TypedDict):
    verdict: Verdict
    rationale: str


_JUDGE_PROMPT = """You are an evaluator grading a legal-QA system. Compare a candidate answer against gold answer spans extracted from the contract by human annotators.

# Question
{question}

# Gold answer spans (extracted by human annotators; may be a list, may be empty)
{gold}

# Candidate answer
{answer}

# Instructions
- The gold spans are *extractive* — short verbatim quotes from the contract. The candidate answer may paraphrase them; paraphrased-but-faithful answers count as "correct".
- If the gold span list is empty, the correct response is the literal string NOT_FOUND. Any other answer is a hallucination.
- Verdicts:
  - "correct": candidate captures the substance of the gold answer (or correctly says NOT_FOUND when gold is empty).
  - "partial": candidate captures part of the gold answer but misses material content, OR conflates the gold answer with extra unsupported content.
  - "hallucinated": candidate asserts facts not supported by the gold spans (and gold is non-empty).
  - "abstained": candidate said NOT_FOUND but the gold spans were non-empty (a miss but not a hallucination).

Return ONLY JSON in this exact shape:
{{"verdict": "correct" | "partial" | "hallucinated" | "abstained", "rationale": "<one sentence>"}}
"""


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_verdict(raw: str) -> JudgeResult:
    import json

    match = _JSON_RE.search(raw)
    if match is None:
        return {"verdict": "hallucinated", "rationale": f"unparseable judge output: {raw[:120]}"}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"verdict": "hallucinated", "rationale": f"invalid JSON from judge: {raw[:120]}"}
    verdict = data.get("verdict", "hallucinated")
    if verdict not in ("correct", "partial", "hallucinated", "abstained"):
        verdict = "hallucinated"
    return {"verdict": verdict, "rationale": str(data.get("rationale", ""))[:300]}


def _format_gold(gold: list[str]) -> str:
    if not gold:
        return "[] (empty — the contract does not address this question)"
    bullets = "\n".join(f"- {span}" for span in gold)
    return bullets


async def judge_answer(
    question: str,
    gold_answers: list[str],
    candidate_answer: str,
    llm: OpenAIAdapter,
) -> JudgeResult:
    prompt = _JUDGE_PROMPT.format(
        question=question,
        gold=_format_gold(gold_answers),
        answer=candidate_answer or "(empty)",
    )
    raw = await llm.complete_text(prompt)
    return _parse_verdict(raw)


def make_judge_llm(model: str = MODEL_JUDGE) -> OpenAIAdapter:
    return OpenAIAdapter(model=model, timeout=JUDGE_TIMEOUT_SEC)
