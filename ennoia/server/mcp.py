"""FastMCP read-only server exposing ``discover_schema``, ``filter``, ``search``, ``retrieve``.

Tool surface is intentionally narrower than the REST surface — agents can
read and reason about the index but cannot index new documents or delete
existing ones. This mirrors ``.ref/USAGE.md §5``: the agent flow is
discover → filter → search(filter_ids=...) → retrieve.

Auth wiring uses FastMCP's middleware hook: every tool invocation runs
through a bearer-token check before the tool body executes.
"""

# pyright: reportUnusedFunction=false
# Tool functions are registered via ``@mcp.tool``; pyright's strict mode
# doesn't detect decorator-based registration.
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ennoia.schema import describe
from ennoia.utils.imports import require_module

if TYPE_CHECKING:  # pragma: no cover
    from fastmcp import FastMCP

    from ennoia.server.context import ServerContext

__all__ = ["create_mcp"]


def create_mcp(ctx: ServerContext) -> FastMCP:
    """Build the FastMCP server bound to ``ctx``."""
    fastmcp_mod = require_module("fastmcp", "server")
    FastMCP = fastmcp_mod.FastMCP

    mcp = FastMCP("ennoia")

    @mcp.tool
    async def discover_schema() -> dict[str, Any]:
        """Return Ennoia's unified discovery payload for this store.

        Shape matches ``docs/filters.md §Schema Discovery`` — a list of
        structural fields (with types, operators, and enum options) plus a
        list of semantic indices (with descriptions).
        """
        return describe(ctx.pipeline.schemas())

    @mcp.tool
    async def filter(filters: dict[str, Any] | None = None) -> list[str]:
        """Return the ``source_id``s matching a structured filter.

        Pass the returned ids back to ``search`` via ``filter_ids=`` to complete
        the two-phase MCP flow.
        """
        return await ctx.pipeline.afilter(filters)

    @mcp.tool
    async def search(
        query: str,
        top_k: int = 5,
        filter_ids: list[str] | None = None,
        index: str | None = None,
    ) -> list[dict[str, Any]]:
        """Vector search. Pass ``filter_ids`` to restrict to a pre-filtered set."""
        result = await ctx.pipeline.asearch(
            query=query,
            filter_ids=filter_ids,
            index=index,
            top_k=top_k,
        )
        return [
            {
                "source_id": hit.source_id,
                "score": hit.score,
                "structural": hit.structural,
                "semantic": hit.semantic,
            }
            for hit in result.hits
        ]

    @mcp.tool
    async def retrieve(id: str) -> dict[str, Any] | None:  # noqa: A002 — spec uses ``id``
        """Return the full structured record for a ``source_id``, or None if absent."""
        return await ctx.pipeline.aretrieve(id)

    return mcp
