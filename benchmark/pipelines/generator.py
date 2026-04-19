"""Shared answer-generation step used by the langchain baseline.

The ennoia pipeline generates its own answer inside its agent loop; the
langchain baseline has no agent, so it hands retrieved product cards to
this prompt and lets the LLM pick a docid. The prompt mirrors the
ennoia-side response contract (JSON ``{"docid","reason"}`` or
``NOT_FOUND``) so the judge grades both pipelines the same way.
"""

from __future__ import annotations

from benchmark.config import GEN_TIMEOUT_SEC, MODEL_LLM
from ennoia.adapters.llm.base import LLMAdapter
from ennoia.adapters.llm.openrouter import OpenRouterAdapter

_PROMPT_TEMPLATE = """You are a shopping assistant. The user issued the query below. Pick the single best product from the retrieved candidates.

# Query
{question}

# Retrieved candidates (docid + listing excerpt, in rank order)
{context}

# Rules
- Recommend exactly ONE product.
- Reply with ONLY a JSON object: {{"docid": "<docid>", "reason": "<one sentence>"}}.
- If no candidate plausibly matches the query, reply exactly: NOT_FOUND
- Do not mention price — it is not in the data.

# Answer
"""


def format_context(question: str, blocks: list[tuple[str, str, float]]) -> str:
    if not blocks:
        body = "(no candidates retrieved)"
    else:
        lines = [
            f"[{idx}] docid={source_id} score={score:.3f}\n{text}"
            for idx, (source_id, text, score) in enumerate(blocks, 1)
        ]
        body = "\n\n".join(lines)
    return _PROMPT_TEMPLATE.format(question=question, context=body)


async def generate_answer(prompt: str, llm: LLMAdapter) -> str:
    return (await llm.complete_text(prompt)).strip()


def make_generator_llm(model: str = MODEL_LLM) -> LLMAdapter:
    """Return the LLM backing the langchain baseline's answer step."""
    return OpenRouterAdapter(model=model, timeout=GEN_TIMEOUT_SEC)
