"""Adapter + store URI parsing for the CLI.

Each adapter is addressed by a URI-style string:

- ``ollama:qwen3:0.6b``
- ``openai:gpt-4o-mini``
- ``anthropic:claude-sonnet-4-20250514``
- ``sentence-transformers:all-MiniLM-L6-v2``
- ``openai-embedding:text-embedding-3-small``

Stores use the same prefix scheme on the ``--store`` flag:

- ``./my_index`` (no prefix) or ``file:./my_index`` — filesystem store
  via :meth:`ennoia.store.composite.Store.from_path`.
- ``qdrant:<collection>`` — :class:`~ennoia.store.hybrid.qdrant.QdrantHybridStore`;
  ``--qdrant-url`` / ``--qdrant-api-key`` supply the endpoint.
- ``pgvector:<collection>`` —
  :class:`~ennoia.store.hybrid.pgvector.PgVectorHybridStore`; ``--pg-dsn``
  supplies the PostgreSQL DSN.

The prefix before the first ``:`` picks the adapter / store class; the
remainder is the model identifier (or collection name) passed verbatim.
Unknown prefixes raise ``typer.BadParameter`` so the CLI surfaces a clean
error.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:
    from ennoia.adapters.embedding.base import EmbeddingAdapter
    from ennoia.adapters.llm.base import LLMAdapter
    from ennoia.store.base import HybridStore
    from ennoia.store.composite import Store

__all__ = ["parse_embedding_spec", "parse_llm_spec", "parse_store_spec"]


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


_STORE_PREFIXES = frozenset({"qdrant", "pgvector", "file"})


def parse_store_spec(
    spec: str,
    *,
    collection: str = "documents",
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
    pg_dsn: str | None = None,
) -> Store | HybridStore:
    """Resolve a ``--store`` CLI value into a concrete store instance.

    Three forms:

    - ``<path>`` (no prefix) or ``file:<path>`` — filesystem store;
      ``collection`` (from ``--collection``) becomes the subdirectory
      under ``path``.
    - ``qdrant:<collection>`` — :class:`QdrantHybridStore`; requires
      ``qdrant_url`` (``--qdrant-url`` or ``ENNOIA_QDRANT_URL``).
      ``qdrant_api_key`` is optional.
    - ``pgvector:<collection>`` — :class:`PgVectorHybridStore`; requires
      ``pg_dsn`` (``--pg-dsn`` or ``ENNOIA_PG_DSN``).
    """
    prefix, _, remainder = spec.partition(":")
    # Only treat the value as a URI when the prefix is one we own; otherwise
    # pass the whole string through to ``Store.from_path`` so paths with
    # colons on disk still work.
    if prefix not in _STORE_PREFIXES:
        from ennoia.store.composite import Store

        return Store.from_path(spec, collection=collection)

    if not remainder:
        raise typer.BadParameter(
            f"Store spec {spec!r} is missing the value after ':'. "
            f"Expected '{prefix}:<{'collection' if prefix != 'file' else 'path'}>'."
        )

    if prefix == "file":
        from ennoia.store.composite import Store

        return Store.from_path(remainder, collection=collection)

    if prefix == "qdrant":
        if not qdrant_url:
            raise typer.BadParameter("qdrant: stores require --qdrant-url (or ENNOIA_QDRANT_URL).")
        from ennoia.store.hybrid.qdrant import QdrantHybridStore

        return QdrantHybridStore(collection=remainder, url=qdrant_url, api_key=qdrant_api_key)

    # prefix == "pgvector" (exhausted _STORE_PREFIXES)
    if not pg_dsn:
        raise typer.BadParameter("pgvector: stores require --pg-dsn (or ENNOIA_PG_DSN).")
    from ennoia.store.hybrid.pgvector import PgVectorHybridStore

    return PgVectorHybridStore(dsn=pg_dsn, collection=remainder)
