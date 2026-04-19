"""FastMCP read-only server exposing ``discover_schema``, ``search``, ``retrieve``.

Tool surface is intentionally narrower than the REST surface — agents can
read and reason about the index but cannot index new documents or delete
existing ones. The canonical agent flow is **discover → search → retrieve**:
``search`` internally performs the two-phase plan (structural filter,
then vector search over survivors) so agents need only one tool call per
retrieval.

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


_MCP_INSTRUCTIONS = """\
Ennoia is a retrieval system over a corpus of documents that have been \
pre-processed by LLM extractors. Every document has:
  - structural fields (typed: str / enum / int / float / bool / date) used \
for exact filtering,
  - semantic indices (named summaries extracted per document) used for \
vector search.

Recommended call order for any question:

  1. discover_schema()  — ALWAYS call this first in a session. It lists the \
structural fields available for `filter` (with their operators and, for \
enums, allowed values) and the semantic indices available for `index`. \
Never guess field names — they come from discovery.

  2. search(query, filter=..., index=..., limit=...) — one call that runs \
the structured filter and vector search together. Use `filter` to narrow \
by metadata (jurisdiction, date, category...) and `query` for the \
semantic intent. Prefer `index=` when the question maps to one specific \
extracted summary.

  3. retrieve(id) — pull the full structured record for any source_id \
worth citing. Call once per result you intend to reference.

Filter grammar: keys are either `field` (equality) or `field__op`, where \
`op` is one of eq, in, gt, gte, lt, lte, contains, contains_any, \
contains_all, startswith, is_null. discover_schema() reports which \
operators each field supports — respect it; unsupported operators raise \
a validation error.

Filter-validation errors include `field`, `operator`, and `message` — \
read them, fix the call, try again. Do not retry blindly.\
"""


def create_mcp(ctx: ServerContext) -> FastMCP:
    """Build the FastMCP server bound to ``ctx``."""
    fastmcp_mod = require_module("fastmcp", "server")
    FastMCP = fastmcp_mod.FastMCP

    mcp = FastMCP("ennoia", instructions=_MCP_INSTRUCTIONS)

    @mcp.tool
    async def discover_schema() -> dict[str, Any]:
        """Introspect the indexed corpus.

        Returns the list of structural fields (with types, supported filter
        operators, and enum options) plus the list of semantic indices (with
        their descriptions) that ``search`` understands.

        Call this FIRST in a session, before any ``search`` call. Use the
        returned field names as keys in the ``filter`` argument to ``search``,
        and the returned index names as the ``index`` argument. Never guess
        field or index names — they are corpus-specific.

        Returns: {"structural_fields": [{"name", "type", "operators",
        "options"?, "sources"}], "semantic_indices": [{"name", "description",
        "kind"}]}.
        """
        return describe(ctx.pipeline.schemas())

    @mcp.tool
    async def search(
        query: str,
        filter: dict[str, Any] | None = None,  # noqa: A002 — spec uses ``filter``
        limit: int = 10,
        index: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve documents by combining a structured filter and a semantic query.

        One call, one round-trip — structural filtering and vector search run
        together under the hood.

        WHEN TO USE:
          - For any question that can be answered from the indexed corpus.
          - Pass ``filter`` whenever discover_schema() has revealed structural
            fields relevant to the question (date ranges, jurisdictions,
            categories, booleans). Filtering first is cheap and dramatically
            improves ranking quality.
          - Pass ``index`` when the question maps to one specific extracted
            summary (e.g. ``index="Holding"`` for a legal-holdings question).
            Omit it to search across all semantic indices.
          - Start with ``limit=10``; lower it only if the downstream step
            cannot handle that many results.

        FILTER SHAPE: a dict whose keys are either ``field`` (equality) or
        ``field__op``. Valid operators per field are reported by
        discover_schema(). Example::

            {"jurisdiction": "WA", "date_decided__gte": "2020-01-01",
             "is_overruled": False}

        Internally runs a two-phase plan: structural filter first, then vector
        search restricted to the survivors. You do not need to chain two
        calls — this one call does both.

        Returns a list of hits ordered by descending score. Each hit has
        ``source_id`` (pass to ``retrieve`` for the full record), ``score``,
        ``structural`` (the structured fields for that document), and
        ``semantic`` (the extracted summary text for the matching index).

        On invalid filters, raises a tool error with a payload of shape
        ``{"error": "invalid_filter", "field", "operator", "message"}`` — read
        the message and retry with a corrected filter.
        """
        result = await ctx.pipeline.asearch(
            query=query,
            filters=filter,
            top_k=limit,
            index=index,
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
        """Fetch the full structured record for one ``source_id``, or None if absent.

        Call this after ``search`` for any hit you intend to cite, so you can
        access every extracted field (not just the subset shown inline in
        search results). ``id`` must be a ``source_id`` returned by ``search``.
        """
        return await ctx.pipeline.aretrieve(id)

    return mcp
