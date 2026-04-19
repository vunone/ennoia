"""LLM-as-judge: classify a product recommendation against the gold docid.

The judge sees the shopper query, the gold product (title + brand +
bullets — no docid) and the top-K retrieved ``(docid, title)`` pairs
alongside the candidate recommendation. Same verdict enum as the legacy
CUAD judge; the prompt is rewritten for product discovery and tolerates
fenced JSON blocks that free OpenRouter models sometimes emit.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal, TypedDict

from benchmark.config import JUDGE_TIMEOUT_SEC, MODEL_LLM
from benchmark.pipelines._retry import async_with_retry
from ennoia.adapters.llm.openrouter import OpenRouterAdapter

Verdict = Literal["correct", "partial", "hallucinated", "abstained"]


class JudgeResult(TypedDict):
    verdict: Verdict
    rationale: str


_JUDGE_PROMPT = """You are grading a product-recommendation system. A shopper issued a search query; the system recommended one product (or said NOT_FOUND). Decide whether the recommendation matches the gold product.

# Shopper query
{question}

# Gold product (the correct answer)
Title: {gold_title}
Brand: {gold_brand}
Features:
{gold_bullets}

# Top retrieval results (what retrieval surfaced)
{retrieved_block}

# Candidate recommendation
{answer}

# Verdict rubric
- "correct": the recommendation matches the gold product (same docid, or the candidate JSON names a distractor whose title+brand clearly denote the same listing).
- "partial": the recommendation is a related-but-different product (same brand-and-category, or clearly in the same narrow product_type family as the gold, but not the exact item the shopper is looking for).
- "hallucinated": the recommendation is a product substantively unrelated to the gold, OR the candidate asserts facts the retrieval did not surface.
- "abstained": the candidate said NOT_FOUND.

Return ONLY a JSON object in this exact shape — no prose, no fences:
{{"verdict": "correct" | "partial" | "hallucinated" | "abstained", "rationale": "<one sentence>"}}
"""


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_VERDICTS: tuple[Verdict, ...] = ("correct", "partial", "hallucinated", "abstained")


def _strip_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json\n"):
            text = text[5:]
        if "```" in text:
            text = text.split("```", 1)[0]
    return text


def _parse_verdict(raw: str) -> JudgeResult:
    text = _strip_fences(raw)
    match = _JSON_RE.search(text)
    if match is None:
        return {"verdict": "hallucinated", "rationale": f"unparseable: {raw[:120]}"}
    try:
        data = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return {"verdict": "hallucinated", "rationale": f"unparseable: {raw[:120]}"}
    raw_verdict = data.get("verdict", "hallucinated")
    verdict: Verdict = raw_verdict if raw_verdict in _VERDICTS else "hallucinated"
    return {"verdict": verdict, "rationale": str(data.get("rationale", ""))[:300]}


def _format_bullets(bullets: list[str]) -> str:
    if not bullets:
        return "- (none)"
    return "\n".join(f"- {b}" for b in bullets[:6])


def _format_retrieved(retrieved: list[dict[str, Any]]) -> str:
    if not retrieved:
        return "(no results retrieved)"
    lines = []
    for idx, row in enumerate(retrieved, 1):
        docid = str(row.get("docid", "?"))
        title = str(row.get("title", ""))[:140]
        lines.append(f"{idx}. docid={docid} — {title}")
    return "\n".join(lines)


async def judge_answer(
    question: str,
    gold_product: dict[str, Any],
    retrieved_top_k: list[dict[str, Any]],
    candidate_answer: str,
    llm: OpenRouterAdapter,
) -> JudgeResult:
    prompt = _JUDGE_PROMPT.format(
        question=question,
        gold_title=gold_product.get("title", ""),
        gold_brand=gold_product.get("brand", ""),
        gold_bullets=_format_bullets(list(gold_product.get("bullet_points", []))),
        retrieved_block=_format_retrieved(retrieved_top_k),
        answer=candidate_answer or "(empty)",
    )
    raw = await async_with_retry(lambda: llm.complete_text(prompt))
    return _parse_verdict(raw)


def make_judge_llm(model: str = MODEL_LLM) -> OpenRouterAdapter:
    return OpenRouterAdapter(model=model, timeout=JUDGE_TIMEOUT_SEC)
