"""Token / cost accounting and pre-flight estimation.

The ennoia adapters don't surface ``response.usage`` to callers, so the
live counter here uses ``tiktoken`` to estimate after each call by
encoding the prompt + answer rather than reading the API response. That's
inexact (within ~5% for cl100k-family encoders) but good enough to
enforce a hard ``--max-cost-usd`` cap without changing core ennoia
interfaces. For the product benchmark only embedding is paid; the LLM
model is free-tier OpenRouter so its tokens are still counted (for audit)
but zero-priced.
"""

from __future__ import annotations

import threading

import tiktoken

from benchmark.config import MODEL_EMBED, MODEL_LLM, PRICING


def _encoder() -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model("o200k_base")
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
    n_products: int,
    n_queries: int,
    avg_product_tokens: int = 600,
    avg_question_context_tokens: int = 800,
    avg_answer_tokens: int = 80,
    avg_judge_input_tokens: int = 800,
    avg_judge_output_tokens: int = 60,
    schemas_per_product: int = 3,
    agent_turn_multiplier: float = 3.0,
) -> dict[str, float]:
    """Pre-flight estimate with conservative assumptions.

    The free OpenRouter LLM tokens round to $0 so the total is dominated
    by embedding tokens. The cost cap still exists to catch mis-typed
    model names that route to a paid OpenAI model.
    """
    extraction_input = n_products * avg_product_tokens * schemas_per_product
    extraction_output = n_products * 300 * schemas_per_product
    embedding_tokens = n_products * 600  # summary + features rows
    lc_embed_tokens = n_products * avg_product_tokens
    qa_input = int(n_queries * avg_question_context_tokens * agent_turn_multiplier)
    qa_output = int(n_queries * avg_answer_tokens * agent_turn_multiplier)
    judge_input = n_queries * avg_judge_input_tokens
    judge_output = n_queries * avg_judge_output_tokens

    extraction_usd = cost_for(MODEL_LLM, extraction_input, extraction_output)
    embedding_usd = cost_for(MODEL_EMBED, embedding_tokens + lc_embed_tokens, 0)
    qa_usd = cost_for(MODEL_LLM, qa_input, qa_output)
    judge_usd = cost_for(MODEL_LLM, judge_input, judge_output)
    total = extraction_usd + embedding_usd + qa_usd + judge_usd

    return {
        "extraction_usd": round(extraction_usd, 4),
        "embedding_usd": round(embedding_usd, 4),
        "qa_usd": round(qa_usd, 4),
        "judge_usd": round(judge_usd, 4),
        "total_usd": round(total, 4),
    }
