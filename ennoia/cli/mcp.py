"""``ennoia mcp`` — boot the FastMCP read-only agent server.

Supported transports per ``.ref/USAGE.md §5``:

- ``sse``  — Server-Sent Events. Default for remote agents.
- ``http`` — streamable-HTTP transport.
- ``stdio`` — for local agents that pipe JSON over stdin/stdout.
"""

from __future__ import annotations

from pathlib import Path

import typer

from ennoia.cli.config import require_option
from ennoia.cli.factories import parse_embedding_spec, parse_llm_spec, parse_store_spec
from ennoia.index.pipeline import Pipeline
from ennoia.server import ServerContext, no_auth, static_bearer_auth
from ennoia.server.mcp import create_mcp

__all__ = ["mcp_command"]


_SUPPORTED_TRANSPORTS = {"sse", "stdio", "http"}


def mcp_command(
    store: str | None = typer.Option(
        None,
        "--store",
        envvar="ENNOIA_STORE",
        help=(
            "Store URI. Plain path / 'file:<path>' → filesystem. "
            "'qdrant:<collection>' → Qdrant (needs --qdrant-url). "
            "'pgvector:<collection>' → pgvector (needs --pg-dsn)."
        ),
    ),
    collection: str = typer.Option(
        "documents",
        "--collection",
        envvar="ENNOIA_COLLECTION",
        help="Collection name for filesystem stores. Ignored for qdrant/pgvector.",
    ),
    qdrant_url: str | None = typer.Option(
        None, "--qdrant-url", envvar="ENNOIA_QDRANT_URL", help="Qdrant endpoint URL."
    ),
    qdrant_api_key: str | None = typer.Option(
        None, "--qdrant-api-key", envvar="ENNOIA_QDRANT_API_KEY", help="Qdrant API key."
    ),
    pg_dsn: str | None = typer.Option(
        None, "--pg-dsn", envvar="ENNOIA_PG_DSN", help="pgvector PostgreSQL DSN."
    ),
    schema: Path | None = typer.Option(
        None, "--schema", envvar="ENNOIA_SCHEMA", help="Python module declaring schemas."
    ),
    transport: str = typer.Option(
        "sse", "--transport", envvar="ENNOIA_TRANSPORT", help="sse | stdio | http"
    ),
    host: str = typer.Option("127.0.0.1", "--host", envvar="ENNOIA_HOST"),
    port: int = typer.Option(8090, "--port", envvar="ENNOIA_PORT"),
    llm: str = typer.Option("ollama:qwen3:0.6b", "--llm", envvar="ENNOIA_LLM"),
    embedding: str = typer.Option(
        "sentence-transformers:all-MiniLM-L6-v2",
        "--embedding",
        envvar="ENNOIA_EMBEDDING",
    ),
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        envvar="ENNOIA_API_KEY",
        help="Bearer token. Defaults to ENNOIA_API_KEY env.",
    ),
    disable_auth: bool = typer.Option(
        False,
        "--no-auth",
        help="Allow unauthenticated tool calls. Overrides --api-key. For local dev only.",
    ),
) -> None:
    """Serve Ennoia's read-only MCP tools — ``discover_schema``, ``search``, ``retrieve``."""
    from ennoia.cli.main import load_schemas

    if transport not in _SUPPORTED_TRANSPORTS:
        raise typer.BadParameter(
            f"Unknown transport {transport!r}. Choose one of: "
            + ", ".join(sorted(_SUPPORTED_TRANSPORTS))
        )
    if not disable_auth and not api_key:
        raise typer.BadParameter(
            "No auth configured. Set --api-key, export ENNOIA_API_KEY, or pass --no-auth."
        )
    store = require_option(store, "--store", "store")
    schema = require_option(schema, "--schema", "schema")
    auth = no_auth() if disable_auth else static_bearer_auth(api_key or "")

    classes = load_schemas(schema)
    pipeline = Pipeline(
        schemas=classes,
        store=parse_store_spec(
            store,
            collection=collection,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            pg_dsn=pg_dsn,
        ),
        llm=parse_llm_spec(llm),
        embedding=parse_embedding_spec(embedding),
    )
    ctx = ServerContext(pipeline=pipeline, auth=auth)
    mcp_server = create_mcp(ctx)

    # FastMCP's ``run`` dispatches to the right transport internally; it
    # blocks until the server exits.
    if transport == "stdio":
        mcp_server.run(transport="stdio")
    elif transport == "http":
        mcp_server.run(transport="http", host=host, port=port)
    else:
        # transport in {"sse"} by the early validation above.
        mcp_server.run(transport="sse", host=host, port=port)
