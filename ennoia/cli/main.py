"""Typer CLI entrypoint — ``ennoia try | index | search``.

The commands mirror ``docs/cli.md``. Adapter selection uses the URI syntax
implemented in :mod:`ennoia.cli.factories`; the store lives under
``--store <path>`` and is always a :meth:`Store.from_path` filesystem store.
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

from ennoia.cli.factories import parse_embedding_spec, parse_llm_spec
from ennoia.index.exceptions import FilterValidationError
from ennoia.index.extractor import extract_semantic, extract_structural
from ennoia.index.pipeline import Pipeline
from ennoia.schema.base import BaseSemantic, BaseStructure
from ennoia.store.composite import Store
from ennoia.utils.filters import KNOWN_OPERATORS, parse_bool, split_filter_key

__all__ = ["app"]


app = typer.Typer(
    add_completion=False,
    help="Ennoia — LLM-powered document pre-indexing and hybrid retrieval.",
    no_args_is_help=True,
)


def _load_schemas(path: Path) -> list[type[BaseStructure] | type[BaseSemantic]]:
    """Load schema classes from a Python module file."""
    if not path.exists():
        raise typer.BadParameter(f"Schema file not found: {path}")

    spec = importlib.util.spec_from_file_location("_ennoia_user_schemas", path)
    if spec is None or spec.loader is None:
        raise typer.BadParameter(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    classes: list[type[BaseStructure] | type[BaseSemantic]] = []
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ != module.__name__:
            continue
        if (issubclass(obj, BaseStructure) and obj is not BaseStructure) or (
            issubclass(obj, BaseSemantic) and obj is not BaseSemantic
        ):
            classes.append(obj)

    if not classes:
        raise typer.BadParameter(f"No BaseStructure/BaseSemantic subclasses found in {path}.")
    return classes


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
        if operator not in KNOWN_OPERATORS:
            raise typer.BadParameter(f"Unknown operator {operator!r} in filter {item!r}.")
        final_key = f"{field}__{operator}" if operator != "eq" else field
        parsed[final_key] = _coerce_filter_value(value, operator)
    return parsed


def _read_document(path: Path) -> str:
    if not path.exists():
        raise typer.BadParameter(f"Document not found: {path}")
    return path.read_text(encoding="utf-8", errors="replace")


@app.command("try")
def try_command(
    document: Path = typer.Argument(..., help="Path to a document file."),
    schema: Path = typer.Option(..., "--schema", help="Python module declaring schemas."),
    llm: str = typer.Option("ollama:qwen3:0.6b", "--llm", help="LLM adapter URI."),
    embedding: str = typer.Option(
        "sentence-transformers:all-MiniLM-L6-v2",
        "--embedding",
        help="Embedding adapter URI (loaded but unused; kept for parity with index/search).",
    ),
) -> None:
    """Run a single extraction pass and print fields + confidences."""
    _ = embedding  # Reserved for later — keeps the flag stable.
    classes = _load_schemas(schema)
    text = _read_document(document)
    llm_adapter = parse_llm_spec(llm)

    async def run() -> None:
        for cls in classes:
            if issubclass(cls, BaseStructure):
                instance, confidence = await extract_structural(
                    schema=cls, text=text, context_additions=[], llm=llm_adapter
                )
                dumped = {
                    k: v for k, v in instance.model_dump(mode="json").items() if k != "_confidence"
                }
                typer.echo(f"Schema: {cls.__name__}")
                for key, value in dumped.items():
                    typer.echo(f"  {key}: {value!r}  (confidence: {confidence:.2f})")
                extended = instance.extend()
                if extended:
                    typer.echo("  -> extend(): " + ", ".join(c.__name__ for c in extended))
            else:
                answer, confidence = await extract_semantic(schema=cls, text=text, llm=llm_adapter)
                typer.echo(f"Schema: {cls.__name__}")
                typer.echo(f"  {answer!r}  (confidence: {confidence:.2f})")

    asyncio.run(run())


@app.command("index")
def index_command(
    directory: Path = typer.Argument(..., help="Directory whose files should be indexed."),
    schema: Path = typer.Option(..., "--schema"),
    store: Path = typer.Option(..., "--store", help="Filesystem store directory."),
    llm: str = typer.Option("ollama:qwen3:0.6b", "--llm"),
    embedding: str = typer.Option("sentence-transformers:all-MiniLM-L6-v2", "--embedding"),
) -> None:
    """Index every file in ``directory`` against the declared schemas."""
    if not directory.is_dir():
        raise typer.BadParameter(f"Not a directory: {directory}")

    classes = _load_schemas(schema)
    pipeline = Pipeline(
        schemas=classes,
        store=Store.from_path(store),
        llm=parse_llm_spec(llm),
        embedding=parse_embedding_spec(embedding),
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
    store: Path = typer.Option(..., "--store"),
    schema: Path | None = typer.Option(
        None,
        "--schema",
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
    llm: str = typer.Option("ollama:qwen3:0.6b", "--llm"),
    embedding: str = typer.Option("sentence-transformers:all-MiniLM-L6-v2", "--embedding"),
) -> None:
    """Search the filesystem store with optional structured filters."""
    filters = _parse_filters(filters_raw)

    classes = _load_schemas(schema) if schema is not None else []
    pipeline = Pipeline(
        schemas=classes,
        store=Store.from_path(store),
        llm=parse_llm_spec(llm),
        embedding=parse_embedding_spec(embedding),
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


def main() -> None:  # pragma: no cover — thin console-script wrapper.
    app()
