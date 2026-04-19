# Cookbook: MCP agent loop

A worked example of the canonical Ennoia agent flow from
`.ref/USAGE.md §5`:

**discover → search → retrieve**

The agent starts by asking the server what's indexed, then issues one
`search` call that combines a structured filter with a semantic query,
and finally pulls the full structured record for whichever documents
are worth citing. Each step is a separate MCP tool call, so an
LLM-driven agent can plan the flow autonomously — no bespoke retrieval
code required on the agent side.

## Prerequisites

- Ennoia indexed against a schema set (`ennoia index ./docs …`).
- The `server` extra: `pip install "ennoia[server,filesystem,ollama,sentence-transformers]"`.
- A running MCP server:

  ```bash
  export ENNOIA_API_KEY="your-secret"
  ennoia mcp \
    --store ./my_index \
    --schema my_schemas.py \
    --transport sse \
    --host 127.0.0.1 \
    --port 8090 \
    --api-key "$ENNOIA_API_KEY"
  ```

## Client script

The script below uses `fastmcp.Client` directly so each step is
inspectable — substitute your own agent framework (OpenAI tools,
Anthropic tool use, LangGraph, etc.) in production.

```python
import asyncio
import os

from fastmcp import Client


MCP_URL = "http://127.0.0.1:8090/sse"
QUESTION = (
    "Is there precedent in Washington state after 2020 for "
    "employer duty to accommodate commuting-related disabilities?"
)


async def main() -> None:
    headers = {"Authorization": f"Bearer {os.environ['ENNOIA_API_KEY']}"}
    async with Client(MCP_URL, transport="sse", headers=headers) as client:
        # 1. discover — what schemas, fields, operators, and semantic
        #    indices does this store expose?
        discovery = await client.call_tool("discover_schema", {})
        print_discovery(discovery)

        # 2. search — one call combining a structured filter and a
        #    semantic query. The filter keys come straight from the
        #    discovery payload. `index="Holding"` pins the ranking to a
        #    single semantic schema; omit it to search all of them.
        hits = await client.call_tool(
            "search",
            {
                "query": QUESTION,
                "filter": {
                    "jurisdiction": "WA",
                    "date_decided__gte": "2020-01-01",
                    "is_overruled": False,
                },
                "index": "Holding",
                "limit": 3,
            },
        )
        for hit in hits:
            print(f"  {hit['source_id']:<20} score={hit['score']:.3f}")

        # 3. retrieve — pull the full structured record for each hit
        #    so the agent can cite jurisdiction, date, parties, etc.
        for hit in hits:
            record = await client.call_tool("retrieve", {"id": hit["source_id"]})
            print(hit["source_id"], record)


def print_discovery(payload: dict) -> None:
    fields = payload.get("structural_fields", [])
    indices = payload.get("semantic_indices", [])
    print(f"discover_schema() -> {len(fields)} fields, {len(indices)} indices")
    for field in fields:
        ops = ",".join(field["operators"])
        print(f"  {field['name']:<20} {field['type']:<10} [{ops}]")
    for idx in indices:
        print(f"  [semantic] {idx['name']}: {idx['description']}")


asyncio.run(main())
```

## What the agent "sees"

The discovery payload is the ground truth every downstream step builds on:

```json
{
  "structural_fields": [
    {"name": "jurisdiction", "type": "enum",
     "options": ["WA", "NY", "TX"],
     "operators": ["eq", "in"], "sources": ["CaseDocument"]},
    {"name": "date_decided", "type": "date",
     "operators": ["eq", "gt", "gte", "lt", "lte"],
     "sources": ["CaseDocument"]},
    {"name": "is_overruled", "type": "bool",
     "operators": ["eq"], "sources": ["CaseDocument"]}
  ],
  "semantic_indices": [
    {"name": "Holding", "description": "What is the core legal holding of this case?"},
    {"name": "Facts",   "description": "Summarise the operative facts."}
  ]
}
```

Because the structural fields come from the **superschema** (see
[Concepts — Superschema](../concepts.md#superschema)), the agent never
has to reason about which `extend()` branch fired for a given
document — every possible field is known up front and namespaced
consistently.

## Two-phase retrieval, one tool call

Passing `filter=` to `search` is not a hint — it's the whole plan. The
server runs the structured filter first, then vector-ranks only the
survivors (or pushes both phases into a single native query on hybrid
backends like Qdrant / pgvector). This is the same two-phase plan the
SDK's `Pipeline.search(filters=…)` uses internally; the MCP surface just
exposes it as one tool so the agent can plan with a single decision.
See [Concepts — Two-phase retrieval](../concepts.md#two-phase-retrieval)
for the rationale.

## Error handling

`search` raises an MCP tool error on invalid filters, with a message
matching the [filter validation payload](../filters.md#validation-errors):

```json
{"error": "invalid_filter",
 "field": "jurisdiction", "operator": "gt",
 "message": "Field 'jurisdiction' (type: enum) does not support operator 'gt'. Supported operators: eq, in."}
```

Agents should catch the tool error, inspect `error` / `field` /
`operator`, and re-plan using the operators advertised in discovery.
