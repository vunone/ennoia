"""Adapter URI parsing — happy path + malformed inputs."""

from __future__ import annotations

import pytest
import typer

from ennoia.adapters.llm.anthropic import AnthropicAdapter
from ennoia.adapters.llm.ollama import OllamaAdapter
from ennoia.adapters.llm.openai import OpenAIAdapter
from ennoia.cli.factories import parse_embedding_spec, parse_llm_spec


def test_parses_ollama() -> None:
    adapter = parse_llm_spec("ollama:qwen3:0.6b")
    assert isinstance(adapter, OllamaAdapter)
    assert adapter.model == "qwen3:0.6b"


def test_parses_openai() -> None:
    adapter = parse_llm_spec("openai:gpt-4o-mini")
    assert isinstance(adapter, OpenAIAdapter)
    assert adapter.model == "gpt-4o-mini"


def test_parses_anthropic() -> None:
    adapter = parse_llm_spec("anthropic:claude-sonnet-4-20250514")
    assert isinstance(adapter, AnthropicAdapter)
    assert adapter.model == "claude-sonnet-4-20250514"


def test_unknown_llm_prefix() -> None:
    with pytest.raises(typer.BadParameter):
        parse_llm_spec("vllm:qwen")


def test_missing_colon_rejected() -> None:
    with pytest.raises(typer.BadParameter):
        parse_llm_spec("ollama")


def test_embedding_sentence_transformers() -> None:
    adapter = parse_embedding_spec("sentence-transformers:all-MiniLM-L6-v2")
    assert hasattr(adapter, "embed_query")


def test_embedding_openai() -> None:
    adapter = parse_embedding_spec("openai-embedding:text-embedding-3-small")
    assert hasattr(adapter, "embed_query")


def test_unknown_embedding_prefix() -> None:
    with pytest.raises(typer.BadParameter):
        parse_embedding_spec("bad:model")
