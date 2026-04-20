"""Typer CLI entrypoint — ``ennoia try | index | search``.

The commands mirror ``docs/cli.md``. Adapter and store selection both use
the URI prefix syntax from :mod:`ennoia.cli.factories`. ``--store`` takes
a filesystem path (default), ``file:<path>``, ``qdrant:<collection>`` (with
``--qdrant-url`` / ``--qdrant-api-key``), or ``pgvector:<collection>``
(with ``--pg-dsn``).
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from ennoia.cli.config import INI_FILENAME, load_ini, require_option, write_template
from ennoia.cli.factories import parse_embedding_spec, parse_llm_spec, parse_store_spec
from ennoia.craft import CraftError, run_craft_loop
from ennoia.index.exceptions import FilterValidationError
from ennoia.index.extractor import (
    CONFIDENCE_KEY,
    extract_collection,
    extract_semantic,
    extract_structural,
)
from ennoia.index.pipeline import Pipeline
from ennoia.schema.base import BaseCollection, BaseSemantic, BaseStructure
from ennoia.schema.roots import identify_roots
from ennoia.utils.filters import parse_bool, split_filter_key

__all__ = ["app"]


app = typer.Typer(
    add_completion=False,
    help="Ennoia — LLM-powered document pre-indexing and hybrid retrieval.",
    no_args_is_help=True,
)


@app.callback()
def root_callback(
    config: Path = typer.Option(
        Path(INI_FILENAME),
        "--config",
        help=(f"Path to the INI config (default: ./{INI_FILENAME}). Missing file is not an error."),
    ),
    no_config: bool = typer.Option(
        False,
        "--no-config",
        help="Skip loading the INI config even if it exists.",
    ),
) -> None:
    """Ennoia — LLM-powered document pre-indexing and hybrid retrieval."""
    if not no_config:
        load_ini(config)


def _register_server_commands() -> None:
    """Register the ``api`` and ``mcp`` subcommands.

    Kept lazy so the ``ennoia try|index|search`` commands still work without
    the ``[server]`` extra installed — importing those modules pulls in
    FastAPI / FastMCP.
    """
    from ennoia.cli.api import api_command
    from ennoia.cli.mcp import mcp_command

    app.command("api")(api_command)
    app.command("mcp")(mcp_command)


_register_server_commands()


def load_schemas(
    path: Path,
) -> list[type[BaseStructure] | type[BaseSemantic] | type[BaseCollection]]:
    """Load the top-level extractor schemas declared in ``path``.

    The loader prefers an explicit module-level ``ennoia_schema`` list
    (written by ``ennoia craft`` and documented for hand-authored files).
    When the variable is absent, every module-level
    :class:`BaseStructure` / :class:`BaseSemantic` / :class:`BaseCollection`
    subclass is discovered and reduced to the DAG roots via
    :func:`ennoia.schema.roots.identify_roots`: if any class declares
    ``Schema.extensions``, only those classes are returned; otherwise all
    discovered classes are returned.
    """
    if not path.exists():
        raise typer.BadParameter(f"Schema file not found: {path}")

    spec = importlib.util.spec_from_file_location("_ennoia_user_schemas", path)
    if spec is None or spec.loader is None:
        raise typer.BadParameter(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    if hasattr(module, "ennoia_schema"):
        return _coerce_ennoia_schema(module.ennoia_schema, path)

    classes: list[type[BaseStructure] | type[BaseSemantic] | type[BaseCollection]] = []
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ != module.__name__:
            continue
        if (
            (issubclass(obj, BaseStructure) and obj is not BaseStructure)
            or (issubclass(obj, BaseSemantic) and obj is not BaseSemantic)
            or (issubclass(obj, BaseCollection) and obj is not BaseCollection)
        ):
            classes.append(obj)

    if not classes:
        raise typer.BadParameter(
            f"No BaseStructure/BaseSemantic/BaseCollection subclasses found in {path}."
        )
    return identify_roots(classes)


def _coerce_ennoia_schema(
    value: object, path: Path
) -> list[type[BaseStructure] | type[BaseSemantic] | type[BaseCollection]]:
    """Validate the module-level ``ennoia_schema`` variable from ``path``."""
    if not isinstance(value, list):
        raise typer.BadParameter(
            f"ennoia_schema in {path} must be a list of schema classes, got {type(value).__name__}."
        )
    entries: list[object] = value  # pyright: ignore[reportUnknownVariableType]
    if not entries:
        raise typer.BadParameter(
            f"ennoia_schema in {path} is empty — declare at least one top-level schema."
        )
    roots: list[type[BaseStructure] | type[BaseSemantic] | type[BaseCollection]] = []
    for entry in entries:
        if inspect.isclass(entry) and issubclass(
            entry, BaseStructure | BaseSemantic | BaseCollection
        ):
            roots.append(entry)
            continue
        name = entry.__name__ if inspect.isclass(entry) else repr(entry)
        raise typer.BadParameter(
            f"ennoia_schema in {path} contains {name!r}, which is not a "
            "BaseStructure, BaseSemantic, or BaseCollection subclass."
        )
    return roots


def _coerce_filter_value(raw_value: str, operator: str) -> Any:
    """Coerce a CLI string value into the operator-appropriate Python value."""
    if operator == "is_null":
        return parse_bool(raw_value)
    if operator in {"in", "contains_all", "contains_any"}:
        return [v.strip() for v in raw_value.split(",")]
    return raw_value


def _parse_filters(raw: list[str]) -> dict[str, Any]:
    """Parse ``--filter k=v`` / ``--filter k__op=v`` pairs from the CLI."""
    parsed: dict[str, Any] = {}
    for item in raw:
        if "=" not in item:
            raise typer.BadParameter(f"Filter must be 'key=value', got {item!r}.")
        key, _, value = item.partition("=")
        field, operator = split_filter_key(key.strip())
        final_key = f"{field}__{operator}" if operator != "eq" else field
        parsed[final_key] = _coerce_filter_value(value, operator)
    return parsed


def _read_document(path: Path) -> str:
    if not path.exists():
        raise typer.BadParameter(f"Document not found: {path}")
    return path.read_text(encoding="utf-8", errors="replace")


def _render_field(console: Console, key: str, value: Any, indent: str = "  ") -> None:
    """Render a single ``key: value`` row with cyan key, default value."""
    rendered_value = "" if value is None else repr(value)
    console.print(f"{indent}[cyan]{key}[/cyan]: {rendered_value}", highlight=False)


def _render_header(
    console: Console,
    kind: str,
    schema_name: str,
    confidence: float | None = None,
) -> None:
    """Render ``Extractor[Kind]: Name  (confidence: 0.90)`` header line."""
    suffix = f"  [yellow](confidence: {confidence:.2f})[/yellow]" if confidence is not None else ""
    console.print(
        f"[bold]Extractor[[magenta]{kind}[/magenta]]:[/bold] [bold cyan]{schema_name}[/bold cyan]"
        + suffix,
        highlight=False,
    )


@app.command("try")
def try_command(
    document: Path = typer.Argument(..., help="Path to a document file."),
    schema: Path | None = typer.Option(
        None,
        "--schema",
        envvar="ENNOIA_SCHEMA",
        help="Python module declaring schemas.",
    ),
    llm: str = typer.Option(
        "ollama:qwen3:0.6b",
        "--llm",
        envvar="ENNOIA_LLM",
        help="LLM adapter URI.",
    ),
) -> None:
    """Run a single extraction pass and print fields + confidences.

    ``try`` is a schema / extraction debug tool — no embeddings, no store.
    Use ``ennoia index`` / ``ennoia search`` when you want persistence and
    semantic search.
    """
    schema = require_option(schema, "--schema", "schema")
    classes = load_schemas(schema)
    text = _read_document(document)
    llm_adapter = parse_llm_spec(llm)
    console = Console()

    async def run() -> None:
        async def _run_one(
            cls: type[BaseStructure] | type[BaseSemantic] | type[BaseCollection],
        ) -> tuple[str, Any]:
            if issubclass(cls, BaseCollection):
                entities, _ = await extract_collection(
                    schema=cls, text=text, context_additions=[], llm=llm_adapter
                )
                return "collection", entities
            if issubclass(cls, BaseStructure):
                instance, _ = await extract_structural(
                    schema=cls, text=text, context_additions=[], llm=llm_adapter
                )
                return "structural", instance
            answer, confidence = await extract_semantic(schema=cls, text=text, llm=llm_adapter)
            return "semantic", (answer, confidence)

        results = await asyncio.gather(*(_run_one(cls) for cls in classes))
        for index, (cls, (kind, payload)) in enumerate(zip(classes, results, strict=True)):
            if index > 0:
                console.print()
            if kind == "structural":
                _render_header(console, "BaseStructure", cls.__name__, payload.confidence)
                dumped = {
                    k: v for k, v in payload.model_dump(mode="json").items() if k != CONFIDENCE_KEY
                }
                for key, value in dumped.items():
                    _render_field(console, key, value)
                extended = payload.extend()
                if extended:
                    names = ", ".join(c.__name__ for c in extended)
                    console.print(f"  [dim]→ extend():[/dim] {names}", highlight=False)
            elif kind == "collection":
                _render_header(console, "BaseCollection", cls.__name__)
                if not payload:
                    console.print("  [dim](no entities extracted)[/dim]", highlight=False)
                for entity in payload:
                    console.print(
                        f"  [dim]-[/dim] [yellow](confidence: {entity.confidence:.2f})[/yellow]",
                        highlight=False,
                    )
                    dumped = {
                        k: v
                        for k, v in entity.model_dump(mode="json").items()
                        if k != CONFIDENCE_KEY
                    }
                    for key, value in dumped.items():
                        _render_field(console, key, value, indent="      ")
            else:
                answer, confidence = payload
                _render_header(console, "BaseSemantic", cls.__name__, confidence)
                console.print(f"  {answer!r}", highlight=False)

    asyncio.run(run())


@app.command("index")
def index_command(
    directory: Path = typer.Argument(..., help="Directory whose files should be indexed."),
    schema: Path | None = typer.Option(None, "--schema", envvar="ENNOIA_SCHEMA"),
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
        help=(
            "Collection name for filesystem stores. "
            "Ignored for qdrant/pgvector (collection is in --store)."
        ),
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
    llm: str = typer.Option("ollama:qwen3:0.6b", "--llm", envvar="ENNOIA_LLM"),
    embedding: str = typer.Option(
        "sentence-transformers:all-MiniLM-L6-v2",
        "--embedding",
        envvar="ENNOIA_EMBEDDING",
    ),
    no_threads: bool = typer.Option(
        False,
        "--no-threads",
        help=(
            "Serialise LLM and embedding calls (concurrency=1). "
            "Use with local backends (Ollama) on resource-constrained machines."
        ),
    ),
) -> None:
    """Index every file in ``directory`` against the declared schemas."""
    schema = require_option(schema, "--schema", "schema")
    store = require_option(store, "--store", "store")
    if not directory.is_dir():
        raise typer.BadParameter(f"Not a directory: {directory}")

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
        concurrency=1 if no_threads else None,
    )

    indexed = 0
    for entry in sorted(directory.iterdir()):
        if not entry.is_file():
            continue
        text = entry.read_text(encoding="utf-8", errors="replace")
        result = pipeline.index(text=text, source_id=entry.name)
        status = "REJECTED" if result.rejected else "ok"
        typer.echo(f"[{status}] {entry.name}")
        indexed += 1 if not result.rejected else 0
    typer.echo(f"Indexed {indexed} document(s) into {store}.")


@app.command("search")
def search_command(
    query: str = typer.Argument(..., help="Natural-language query."),
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
        None,
        "--schema",
        envvar="ENNOIA_SCHEMA",
        help=(
            "Schemas used for filter validation. Optional — when omitted, filters are "
            "forwarded without validation (the structured store may still reject them)."
        ),
    ),
    filters_raw: list[str] = typer.Option(
        [],
        "--filter",
        help="Filter in 'key=value' or 'key__op=value' form. Repeatable.",
    ),
    top_k: int = typer.Option(5, "--top-k"),
    llm: str = typer.Option("ollama:qwen3:0.6b", "--llm", envvar="ENNOIA_LLM"),
    embedding: str = typer.Option(
        "sentence-transformers:all-MiniLM-L6-v2",
        "--embedding",
        envvar="ENNOIA_EMBEDDING",
    ),
    no_threads: bool = typer.Option(
        False,
        "--no-threads",
        help=(
            "Serialise LLM and embedding calls (concurrency=1). "
            "Use with local backends (Ollama) on resource-constrained machines."
        ),
    ),
) -> None:
    """Search the filesystem store with optional structured filters."""
    store = require_option(store, "--store", "store")
    filters = _parse_filters(filters_raw)

    classes = load_schemas(schema) if schema is not None else []
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
        concurrency=1 if no_threads else None,
    )

    # CLI values arrive as strings; coerce against store records at eval time.
    # For date/number comparisons the stores coerce on their own path.
    try:
        result = pipeline.search(query=query, filters=filters, top_k=top_k)
    except FilterValidationError as err:
        typer.echo(json.dumps(err.to_dict(), indent=2))
        raise typer.Exit(code=2) from err

    if not result:
        typer.echo("No hits.")
        return

    typer.echo(f"Results ({len(result)} found):\n")
    for idx, hit in enumerate(result, start=1):
        semantic_preview = ""
        if hit.semantic:
            key, text = next(iter(hit.semantic.items()))
            snippet = text if len(text) <= 160 else text[:157] + "..."
            semantic_preview = f"\n   {key}: {snippet}"
        typer.echo(f"{idx}. [{hit.source_id}]  (score: {hit.score:.3f})")
        if hit.structural:
            typer.echo(f"   {json.dumps(hit.structural, default=str)}")
        if semantic_preview:
            typer.echo(semantic_preview)


@app.command("craft")
def craft_command(
    document: Path = typer.Argument(..., help="Sample document the schema should fit."),
    output: Path = typer.Option(
        ...,
        "--output",
        help=(
            "Path to the schema .py file to write. "
            "If it already exists, the current contents are passed to the LLM "
            "and improved instead of rewritten from scratch."
        ),
    ),
    llm: str | None = typer.Option(
        None,
        "--llm",
        envvar="ENNOIA_LLM",
        help="LLM adapter URI, e.g. openai:gpt-4o-mini.",
    ),
    task: str = typer.Option(
        ...,
        "--task",
        help="Natural-language description of the retrieval task the schema must serve.",
    ),
    max_retries: int = typer.Option(
        2,
        "--max-retries",
        min=0,
        help="Budget for validation retries after the first attempt (default: 2).",
    ),
) -> None:
    """Draft (or improve) a schema file with an LLM from a sample document.

    Prototype-only. The generated schema imports cleanly, but field
    selection, docstrings, types, and ``extend()`` logic all need human
    review before the schema is handed to ``ennoia index``.
    """
    typer.echo(
        "[craft] WARNING: Prototype only. The generated schema is a starting point — "
        "review field choices, docstrings, types, and extend() logic before "
        "using it with `ennoia index` on real data."
    )
    llm = require_option(llm, "--llm", "llm")
    text = _read_document(document)
    llm_adapter = parse_llm_spec(llm)
    existing_schema = (
        output.read_text(encoding="utf-8") if output.exists() and output.is_file() else None
    )

    def _announce(attempt: int, stage: str) -> None:
        typer.echo(f"[craft] attempt {attempt + 1}/{max_retries + 1}: {stage}")

    try:
        asyncio.run(
            run_craft_loop(
                llm=llm_adapter,
                task=task,
                document=text,
                output_path=output,
                existing_schema=existing_schema,
                max_retries=max_retries,
                on_attempt=_announce,
            )
        )
    except CraftError as err:
        typer.echo(f"[craft] failed: {err}")
        if output.exists():
            typer.echo(f"[craft] partial output left at {output} for inspection.")
        raise typer.Exit(code=1) from err

    typer.echo(f"[craft] wrote draft schema to {output}")
    typer.echo(
        "[craft] WARNING: Review the draft before indexing: confirm the extractor "
        "kinds, docstrings (they are the LLM prompts), field types / Literals, "
        "and any extend() branches match your retrieval task."
    )


@app.command("init")
def init_command(
    path: Path = typer.Option(
        Path(INI_FILENAME),
        "--path",
        help=f"Destination for the template (default: ./{INI_FILENAME}).",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite the destination if it already exists.",
    ),
) -> None:
    """Write a template ``ennoia.ini`` for the current project."""
    written = write_template(path, force=force)
    typer.echo(f"Wrote {written}. Edit it and re-run any ennoia command without flags.")


def main() -> None:  # pragma: no cover — thin console-script wrapper.
    app()
