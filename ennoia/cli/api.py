"""``ennoia api`` — boot the FastAPI REST server against any supported store."""

from __future__ import annotations

from pathlib import Path

import typer

from ennoia.cli.factories import parse_embedding_spec, parse_llm_spec, parse_store_spec
from ennoia.index.pipeline import Pipeline
from ennoia.server import ServerContext, no_auth, static_bearer_auth
from ennoia.server.api import create_app
from ennoia.utils.imports import require_module

__all__ = ["api_command"]


def api_command(
    store: str = typer.Option(
        ...,
        "--store",
        help=(
            "Store URI. Plain path / 'file:<path>' → filesystem. "
            "'qdrant:<collection>' → Qdrant (needs --qdrant-url). "
            "'pgvector:<collection>' → pgvector (needs --pg-dsn)."
        ),
    ),
    collection: str = typer.Option(
        "documents",
        "--collection",
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
    schema: Path = typer.Option(..., "--schema", help="Python module declaring schemas."),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8080, "--port"),
    llm: str = typer.Option("ollama:qwen3:0.6b", "--llm"),
    embedding: str = typer.Option("sentence-transformers:all-MiniLM-L6-v2", "--embedding"),
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        envvar="ENNOIA_API_KEY",
        help="Bearer token enforced on every request. Defaults to ENNOIA_API_KEY env.",
    ),
    disable_auth: bool = typer.Option(
        False,
        "--no-auth",
        help="Allow unauthenticated requests. Overrides --api-key. Intended for local dev.",
    ),
) -> None:
    """Serve Ennoia's REST API — ``/discover``, ``/filter``, ``/search``, ``/retrieve``, ``/index``, ``/delete``."""
    from ennoia.cli.main import load_schemas

    if not disable_auth and not api_key:
        raise typer.BadParameter(
            "No auth configured. Set --api-key, export ENNOIA_API_KEY, or pass --no-auth."
        )
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
    app = create_app(ctx)

    uvicorn = require_module("uvicorn", "server")
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    uvicorn.Server(config).run()
