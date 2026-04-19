"""Ennoia DDI+agentic pipeline for product recommendation.

The LLM is given three tools that mirror Ennoia's public surface:

- ``get_search_schema()`` — wraps :func:`ennoia.schema.describe` so the
  agent can inspect which structural fields and semantic indices exist.
- ``search(query, filter)`` — single-call filter+vector search, mapped
  onto :meth:`ennoia.Pipeline.asearch`. Always returns up to
  ``RETRIEVAL_TOP_K`` hits (benchmark-fixed, not agent-configurable) so
  every pipeline is measured at the same cutoff.
- ``get_full(document_id)`` — fetches the full original product record
  for a given ``source_id`` returned by ``search``. Lets the agent
  confirm a candidate against the unredacted document before answering.

Precision@k is measured over the ordered, deduplicated union of
``source_id``s returned by every ``search`` call the agent made.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from tqdm.asyncio import tqdm_asyncio

from benchmark.config import (
    AGENT_SYSTEM_PROMPT,
    GEN_TIMEOUT_SEC,
    MAX_AGENT_ITERATIONS,
    MODEL_EMBED,
    MODEL_LLM,
    RETRIEVAL_TOP_K,
    THREADS,
)
from benchmark.data.prep import Product
from benchmark.pipelines._retry import async_with_retry
from benchmark.pipelines.base import PipelineRun
from benchmark.pipelines.schemas import ProductMeta, ProductSummary
from ennoia import Pipeline as EnnoiaCorePipeline
from ennoia.adapters.embedding.openai import OpenAIEmbedding
from ennoia.adapters.llm.openrouter import OpenRouterAdapter
from ennoia.schema import describe
from ennoia.store import InMemoryStructuredStore, InMemoryVectorStore, Store

_TOOLS_SPEC: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_search_schema",
            "description": (
                "Return the searchable schema: structural fields with their "
                "types and allowed operators, plus semantic indices with "
                "descriptions. Call this FIRST before any search."
            ),
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Run a filtered vector search over the product catalogue. "
                "Provide a semantic query and (optionally) a structural "
                "filter built from fields returned by get_search_schema "
                "(e.g. brand, category, product_type). Returns up to 10 "
                "hits with source_id (= docid), score, structural record, "
                "and best-matching semantic snippet."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language query for the vector index.",
                    },
                    "filter": {
                        "type": "object",
                        "description": (
                            "Structural filter dict using the FLAT "
                            "'<field>__<operator>' key convention, e.g. "
                            '{"brand__eq": "ASUS", "price_usd__lt": 80}. '
                            "The nested form {field: {op: value}} is "
                            "REJECTED with FilterValidationError. "
                            "Supported operators per field come from "
                            "get_search_schema. Use {} or omit for no filter."
                        ),
                        "additionalProperties": True,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_full",
            "description": (
                "Fetch the full original product record by source_id "
                "(returned from a prior search hit). Use to confirm a "
                "candidate against the unredacted document before "
                "answering."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "source_id from a prior search hit.",
                    },
                },
                "required": ["document_id"],
                "additionalProperties": False,
            },
        },
    },
]


def _product_text(product: Product) -> str:
    """Render a ``Product`` into the flat text blob the extractor sees."""
    parts: list[str] = [product["title"]]
    if product["brand"] and product["brand"] != "unknown":
        parts.append(f"Brand: {product['brand']}")
    if product["color"] and product["color"] != "unknown":
        parts.append(f"Color: {product['color']}")
    # ``Price: $X`` is the anchor ProductMeta.price_usd reads from; keep the
    # literal format stable so the extraction prompt continues to match.
    parts.append(f"Price: ${product['price_usd']}")
    if product["text"]:
        parts.append(product["text"])
    if product["bullet_points"]:
        parts.append("\n".join(f"- {b}" for b in product["bullet_points"]))
    return "\n\n".join(parts)


class EnnoiaPipeline:
    name = "ennoia"

    def __init__(
        self,
        gen_model: str = MODEL_LLM,
        embed_model: str = MODEL_EMBED,
        threads: int = THREADS,
        max_iterations: int = MAX_AGENT_ITERATIONS,
    ) -> None:
        self._pipeline = EnnoiaCorePipeline(
            schemas=[ProductMeta, ProductSummary],
            store=Store(
                vector=InMemoryVectorStore(),
                structured=InMemoryStructuredStore(),
            ),
            llm=OpenRouterAdapter(model=gen_model, timeout=GEN_TIMEOUT_SEC),
            embedding=OpenAIEmbedding(model=embed_model),
            concurrency=threads,
        )
        self._gen_model = gen_model
        self._threads = threads
        self._max_iterations = max_iterations
        # Lazy-imported inside ``answer`` so tests can monkeypatch the
        # factory. The agent loop drives an OpenAI-shaped client pointed at
        # OpenRouter's OpenAI-compatible endpoint.
        self._client_factory = _default_openrouter_client

    async def index_corpus(self, products: list[Product]) -> None:
        outer = asyncio.Semaphore(self._threads)
        failures: list[tuple[str, str]] = []

        async def index_one(product: Product) -> None:
            async with outer:
                try:
                    await self._aindex_with_retry(product)
                except Exception as exc:
                    failures.append((product["docid"], f"{type(exc).__name__}: {exc}"))

        await tqdm_asyncio.gather(
            *(index_one(p) for p in products),
            desc="ennoia index",
            unit="product",
        )
        if failures:
            print(f"[ennoia] {len(failures)}/{len(products)} products failed to index:")
            for docid, reason in failures[:10]:
                print(f"[ennoia]   - {docid}: {reason}")
            if len(failures) > 10:
                print(f"[ennoia]   ... and {len(failures) - 10} more")

    async def _aindex_with_retry(self, product: Product) -> None:
        text = _product_text(product)
        docid = product["docid"]

        def _on_retry(attempt: int, exc: BaseException, delay: float) -> None:
            print(
                f"[ennoia] {type(exc).__name__} on {docid} "
                f"(attempt {attempt + 1}); waiting {delay:.1f}s"
            )

        await async_with_retry(
            lambda: self._pipeline.aindex(text=text, source_id=docid),
            on_retry=_on_retry,
        )

    async def answer(self, query: str) -> PipelineRun:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        retrieved_ids: list[str] = []
        trace: list[dict[str, Any]] = []
        prompt_tokens = 0
        completion_tokens = 0
        final_answer = "NOT_FOUND"

        # ``async with`` closes the underlying httpx transport when ``answer``
        # returns — otherwise the GC finalizer schedules ``aclose()`` on the
        # outer event loop after it's already closed, producing noisy
        # ``RuntimeError: Event loop is closed`` tracebacks at shutdown.
        async with self._client_factory() as client:
            for iteration in range(self._max_iterations):
                is_last = iteration == self._max_iterations - 1
                response = await client.chat.completions.create(
                    model=self._gen_model,
                    messages=messages,
                    tools=_TOOLS_SPEC,
                    tool_choice="none" if is_last else "auto",
                )
                usage = getattr(response, "usage", None)
                if usage is not None:
                    prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
                    completion_tokens += getattr(usage, "completion_tokens", 0) or 0

                message = response.choices[0].message
                tool_calls = getattr(message, "tool_calls", None) or []
                if not tool_calls:
                    final_answer = (message.content or "").strip() or "NOT_FOUND"
                    break

                messages.append(_assistant_message_dict(message))
                for call in tool_calls:
                    name = call.function.name
                    raw_args = call.function.arguments or "{}"
                    try:
                        args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        args = {}
                    result = await self._dispatch_tool(name, args, retrieved_ids)
                    trace.append({"iteration": iteration, "tool": name, "args": args})
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": json.dumps(result, default=str),
                        }
                    )
            else:
                # Loop exhausted without a no-tool-call message; force a final
                # answer turn with tool_choice="none".
                forced = await client.chat.completions.create(
                    model=self._gen_model,
                    messages=messages,
                    tool_choice="none",
                )
                usage = getattr(forced, "usage", None)
                if usage is not None:
                    prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
                    completion_tokens += getattr(usage, "completion_tokens", 0) or 0
                final_answer = (forced.choices[0].message.content or "").strip() or "NOT_FOUND"

        return PipelineRun(
            retrieved_source_ids=retrieved_ids,
            answer=final_answer,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            trace=trace,
        )

    async def _dispatch_tool(
        self,
        name: str,
        args: dict[str, Any],
        retrieved_ids: list[str],
    ) -> Any:
        if name == "get_search_schema":
            return describe(self._pipeline.schemas())
        if name == "search":
            query = str(args.get("query", "")).strip()
            raw_filter = args.get("filter")
            filters: dict[str, Any] | None = (
                raw_filter if isinstance(raw_filter, dict) and raw_filter else None
            )
            try:
                result = await self._pipeline.asearch(
                    query=query, filters=filters, top_k=RETRIEVAL_TOP_K
                )
            except Exception as exc:
                return {"error": f"{type(exc).__name__}: {exc}"}
            hits_payload: list[dict[str, Any]] = []
            for hit in result.hits:
                if hit.source_id and hit.source_id not in retrieved_ids:
                    retrieved_ids.append(hit.source_id)
                hits_payload.append(
                    {
                        "source_id": hit.source_id,
                        "score": float(hit.score),
                        "structural": hit.structural,
                        "semantic": hit.semantic,
                    }
                )
            return {"hits": hits_payload}
        if name == "get_full":
            document_id = str(args.get("document_id", "")).strip()
            if not document_id:
                return {"error": "document_id is required"}
            record = await self._pipeline.aretrieve(document_id)
            return {"document_id": document_id, "record": record}
        return {"error": f"unknown tool: {name}"}


def _assistant_message_dict(message: Any) -> dict[str, Any]:
    tool_calls = []
    for call in message.tool_calls or []:
        tool_calls.append(
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
        )
    return {
        "role": "assistant",
        "content": message.content,
        "tool_calls": tool_calls,
    }


def _default_openrouter_client() -> Any:
    from openai import AsyncOpenAI

    return AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY", "missing"),
        timeout=GEN_TIMEOUT_SEC,
    )
