"""Token / cost accounting and pre-flight estimation.

The OpenAI adapters in ennoia don't surface ``response.usage`` to callers,
so the live counter here uses ``tiktoken`` to estimate after each call by
hashing the prompt + answer rather than reading the API response. That's
inexact (within ~5% for cl100k-family encoders) but good enough to enforce
a hard ``--max-cost-usd`` cap without changing core ennoia interfaces.
"""

from __future__ import annotations

import threading

import tiktoken

from benchmark.config import MODEL_EMBED, MODEL_GEN, MODEL_JUDGE, PRICING


def _encoder() -> tiktoken.Encoding:
    # gpt-5.4 family is post-cl100k but tiktoken may not ship a dedicated
    # encoder yet; cl100k_base estimates within a few percent and never
    # double-counts.
    try:
        return tiktoken.encoding_for_model("gpt-4o")
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


_ENC = _encoder()


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


def cost_for(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = PRICING.get(model)
    if pricing is None:
        return 0.0
    return (
        input_tokens / 1_000_000 * pricing["input"] + output_tokens / 1_000_000 * pricing["output"]
    )


class BudgetExceeded(RuntimeError):
    """Raised by :class:`RunningCost.charge` when the spend cap is hit."""


class RunningCost:
    """Thread-safe live spend tracker with a hard cap."""

    def __init__(self, max_usd: float) -> None:
        self.max_usd = max_usd
        self.spent_usd = 0.0
        self.tokens_by_model: dict[str, dict[str, int]] = {}
        self._lock = threading.Lock()

    def charge(self, model: str, input_tokens: int, output_tokens: int) -> float:
        delta = cost_for(model, input_tokens, output_tokens)
        with self._lock:
            self.spent_usd += delta
            bucket = self.tokens_by_model.setdefault(model, {"input": 0, "output": 0})
            bucket["input"] += input_tokens
            bucket["output"] += output_tokens
            spent = self.spent_usd
        if spent > self.max_usd:
            raise BudgetExceeded(f"Spend ${spent:.2f} exceeded cap ${self.max_usd:.2f}; aborting.")
        return delta

    def summary(self) -> dict[str, object]:
        with self._lock:
            return {
                "spent_usd": round(self.spent_usd, 4),
                "max_usd": self.max_usd,
                "tokens_by_model": {k: dict(v) for k, v in self.tokens_by_model.items()},
            }


def estimate_run_cost(
    n_contracts: int,
    n_questions: int,
    avg_contract_tokens: int = 25_000,
    avg_question_context_tokens: int = 3_000,
    avg_answer_tokens: int = 200,
    avg_judge_input_tokens: int = 1_000,
    avg_judge_output_tokens: int = 80,
    schemas_per_contract: int = 3,
    chunks_per_contract: int = 30,
    agent_turn_multiplier: float = 3.0,
) -> dict[str, float]:
    """Pre-flight estimate with conservative assumptions.

    Returns per-bucket USD plus a total. Use to decide whether to require
    ``--confirm-cost`` before letting a long run start.

    ``agent_turn_multiplier`` scales the ennoia-side QA bucket to account
    for the tool-calling loop: each question goes through ~3 assistant
    turns on average (discover -> search -> final) with schema + hit
    payloads echoed back each time.
    """
    extraction_input = n_contracts * avg_contract_tokens * schemas_per_contract
    extraction_output = n_contracts * 500 * schemas_per_contract  # JSON outputs are short.
    embedding_tokens_ennoia = n_contracts * 600  # one summary per contract
    embedding_tokens_lc = n_contracts * chunks_per_contract * 300
    lc_qa_input = n_questions * avg_question_context_tokens
    lc_qa_output = n_questions * avg_answer_tokens
    ennoia_qa_input = int(n_questions * avg_question_context_tokens * agent_turn_multiplier)
    ennoia_qa_output = int(n_questions * avg_answer_tokens * agent_turn_multiplier)
    judge_input = n_questions * avg_judge_input_tokens
    judge_output = n_questions * avg_judge_output_tokens

    # Generator + embedder are typically local (zero-priced) so these two
    # buckets round to $0; the cost cap exists to protect the judge spend.
    extraction_usd = cost_for(MODEL_GEN, extraction_input, extraction_output)
    embedding_usd = cost_for(MODEL_EMBED, embedding_tokens_ennoia + embedding_tokens_lc, 0)
    qa_usd = cost_for(MODEL_GEN, lc_qa_input + ennoia_qa_input, lc_qa_output + ennoia_qa_output)
    judge_usd = cost_for(MODEL_JUDGE, judge_input, judge_output)
    total = extraction_usd + embedding_usd + qa_usd + judge_usd

    return {
        "extraction_usd": round(extraction_usd, 4),
        "embedding_usd": round(embedding_usd, 4),
        "qa_usd": round(qa_usd, 4),
        "judge_usd": round(judge_usd, 4),
        "total_usd": round(total, 4),
    }
