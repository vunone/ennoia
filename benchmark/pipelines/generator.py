"""Shared answer-generation step used by the langchain baseline.

The ennoia pipeline generates its own answer inside the agent loop
(:mod:`benchmark.pipelines.ennoia_pipeline`); langchain has no agent and
instead hands retrieved chunks to this same prompt. The prompt forces
``NOT_FOUND`` when the context lacks the answer — measured by the judge as
a non-hallucinated abstention.
"""

from __future__ import annotations

from benchmark.config import GEN_TIMEOUT_SEC, MODEL_GEN, OLLAMA_HOST
from ennoia.adapters.llm.base import LLMAdapter
from ennoia.adapters.llm.ollama import OllamaAdapter

_PROMPT_TEMPLATE = """You are a legal contract analyst. Answer the question below using ONLY the provided contract excerpts. If the excerpts do not contain enough information to answer, reply exactly: NOT_FOUND

# Question
{question}

# Contract excerpts
{context}

# Instructions
- Quote or paraphrase only what appears in the excerpts.
- Do not draw on outside legal knowledge.
- If multiple excerpts are relevant, synthesise them.
- If the answer is not present, reply exactly: NOT_FOUND

# Answer
"""


def format_context(question: str, blocks: list[tuple[str, str, float]]) -> str:
    """Render ``question`` + excerpt blocks into the shared prompt template.

    ``blocks`` is a list of ``(source_id, text, score)`` tuples in rank order.
    """
    if not blocks:
        body = "(no excerpts retrieved)"
    else:
        lines = [
            f"[{idx}] (source={source_id}, score={score:.3f})\n{text}"
            for idx, (source_id, text, score) in enumerate(blocks, 1)
        ]
        body = "\n\n".join(lines)
    return _PROMPT_TEMPLATE.format(question=question, context=body)


async def generate_answer(prompt: str, llm: LLMAdapter) -> str:
    return (await llm.complete_text(prompt)).strip()


def make_generator_llm(model: str = MODEL_GEN) -> LLMAdapter:
    """Return the LLM backing the langchain baseline's answer step.

    Defaults to a local Ollama adapter; swap ``BENCHMARK_MODEL_GEN`` +
    ``OLLAMA_HOST`` at runtime to point at a different local (or
    OpenAI-compatible) backend. The return type is ``LLMAdapter`` so the
    caller isn't coupled to the concrete backend.
    """
    return OllamaAdapter(model=model, host=OLLAMA_HOST, timeout=GEN_TIMEOUT_SEC)
