"""Adapter URI parsing for the CLI.

Each adapter is addressed by a URI-style string:

- ``ollama:qwen3:0.6b``
- ``openai:gpt-4o-mini``
- ``anthropic:claude-sonnet-4-20250514``
- ``sentence-transformers:all-MiniLM-L6-v2``
- ``openai-embedding:text-embedding-3-small``

The prefix before the first ``:`` picks the adapter class; the remainder is
the model identifier passed verbatim. Unknown prefixes raise
``typer.BadParameter`` so the CLI surfaces a clean error.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:
    from ennoia.adapters.embedding.base import EmbeddingAdapter
    from ennoia.adapters.llm.base import LLMAdapter

__all__ = ["parse_embedding_spec", "parse_llm_spec"]


def _split_spec(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        raise typer.BadParameter(
            f"Adapter spec must be 'prefix:model', got {spec!r}.",
        )
    prefix, _, model = spec.partition(":")
    if not prefix or not model:
        raise typer.BadParameter(f"Malformed adapter spec: {spec!r}.")
    return prefix, model


def parse_llm_spec(spec: str) -> LLMAdapter:
    prefix, model = _split_spec(spec)
    if prefix == "ollama":
        from ennoia.adapters.llm.ollama import OllamaAdapter

        return OllamaAdapter(model=model)
    if prefix == "openai":
        from ennoia.adapters.llm.openai import OpenAIAdapter

        return OpenAIAdapter(model=model)
    if prefix == "anthropic":
        from ennoia.adapters.llm.anthropic import AnthropicAdapter

        return AnthropicAdapter(model=model)
    raise typer.BadParameter(
        f"Unknown LLM adapter prefix {prefix!r}. Expected one of: ollama, openai, anthropic."
    )


def parse_embedding_spec(spec: str) -> EmbeddingAdapter:
    prefix, model = _split_spec(spec)
    if prefix == "sentence-transformers":
        from ennoia.adapters.embedding.sentence_transformers import SentenceTransformerEmbedding

        return SentenceTransformerEmbedding(model=model)
    if prefix == "openai-embedding":
        from ennoia.adapters.embedding.openai import OpenAIEmbedding

        return OpenAIEmbedding(model=model)
    raise typer.BadParameter(
        f"Unknown embedding adapter prefix {prefix!r}. "
        "Expected one of: sentence-transformers, openai-embedding."
    )
