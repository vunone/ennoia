"""Unit tests for the shared generator prompt used by the LangChain baseline."""

from __future__ import annotations

from benchmark.pipelines.generator import format_context, generate_answer, make_generator_llm


def test_format_context_with_blocks_renders_docids() -> None:
    prompt = format_context(
        "looking for a stand",
        [("p0", "Aluminium stand", 0.9), ("p1", "USB stand", 0.7)],
    )
    assert "docid=p0" in prompt
    assert "docid=p1" in prompt
    assert "looking for a stand" in prompt


def test_format_context_empty_blocks_rendered_placeholder() -> None:
    prompt = format_context("q", [])
    assert "(no candidates retrieved)" in prompt


async def test_generate_answer_strips_whitespace() -> None:
    class _FakeLLM:
        async def complete_text(self, prompt: str) -> str:
            return '  {"docid":"p0"}\n  '

    result = await generate_answer("prompt", _FakeLLM())  # type: ignore[arg-type]
    assert result == '{"docid":"p0"}'


def test_make_generator_llm_returns_openrouter() -> None:
    adapter = make_generator_llm()
    assert adapter.__class__.__name__ == "OpenRouterAdapter"
