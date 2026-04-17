"""Ennoia DDI+RAG pipeline driven by an agentic tool loop.

The LLM is given three tools that mirror Ennoia's public surface:

- ``get_search_schema()`` — wraps :func:`ennoia.schema.describe` so the agent
  can inspect which structural fields and semantic indices exist.
- ``search(query, filter, limit)`` — single-call filter+vector search, mapped
  onto :meth:`ennoia.Pipeline.asearch` with ``filters=`` (the one-shot flow).
- ``get_full(document_id)`` — escape hatch mapped onto
  :meth:`ennoia.Pipeline.aretrieve`; the system prompt instructs the agent to
  only call it when the search hits lack enough detail for a precise answer.

Recall@k is measured over the ordered, deduplicated union of ``source_id``s
returned by every ``search`` call the agent made during the loop.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from openai import APIConnectionError, APITimeoutError
from tqdm.asyncio import tqdm_asyncio

from benchmark.config import (
    AGENT_SYSTEM_PROMPT,
    GEN_TIMEOUT_SEC,
    INDEX_CONCURRENCY,
    MAX_AGENT_ITERATIONS,
    MODEL_EMBED,
    MODEL_GEN,
    OLLAMA_HOST,
    RETRIEVAL_TOP_K,
)
from benchmark.data.loader import Contract, Question
from benchmark.pipelines.base import PipelineRun
from benchmark.pipelines.schemas import (
    ClauseInventory,
    ClauseMention,
    ContractMeta,
    ContractOverview,
)
from ennoia import Pipeline as EnnoiaCorePipeline
from ennoia.adapters.embedding.sentence_transformers import SentenceTransformerEmbedding
from ennoia.adapters.llm.ollama import OllamaAdapter
from ennoia.schema import describe
from ennoia.store import InMemoryStructuredStore, InMemoryVectorStore, Store

# Per-contract indexing hits the LLM for every extraction schema, so a
# flaky connection can drop a double-digit slice of the corpus before the
# final chart is drawn. Retry transient network errors (connection resets,
# read timeouts) with linear backoff — persistent failures still surface
# to the caller unchanged. 6 total attempts with linear backoff
# ``initial * attempt`` (so 0s, 1s, 2s, 3s, 4s between attempts — 10s
# worst-case added delay for a contract that ultimately still fails).
_INDEX_RETRY_ATTEMPTS = 6
_INDEX_RETRY_INITIAL_BACKOFF_SEC = 1.0
_INDEX_RETRYABLE_EXC: tuple[type[BaseException], ...] = (APIConnectionError, APITimeoutError)

_TOOLS_SPEC: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_search_schema",
            "description": (
                "Return the searchable schema: structural fields with their types "
                "and allowed operators, plus semantic indices with descriptions. "
                "Call this FIRST before any search to learn which filters are valid."
            ),
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Run a filtered vector search. Provide a semantic query string and "
                "(optionally) a structural filter built from the fields returned by "
                "get_search_schema. Returns up to `limit` hits with source_id, score, "
                "the structural record, and the best-matching semantic snippet."
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
                            "Structural filter dict (e.g. "
                            '{"clauses_present__contains": "ip_licensing"}). Use {} '
                            "or omit for no structural filter."
                        ),
                        "additionalProperties": True,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of hits to return.",
                        "minimum": 1,
                        "maximum": 25,
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
                "Fetch the FULL structured record for a single source_id. Call "
                "this on the top search hit once it plausibly matches the "
                "question's target contract — the full record is what supports "
                "a precise extractive answer. Skip only if no hit plausibly "
                "matches (then reply NOT_FOUND)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "The source_id returned by a prior search hit.",
                    },
                },
                "required": ["document_id"],
                "additionalProperties": False,
            },
        },
    },
]


class EnnoiaPipeline:
    name = "ennoia"

    def __init__(
        self,
        gen_model: str = MODEL_GEN,
        embed_model: str = MODEL_EMBED,
        concurrency: int = INDEX_CONCURRENCY,
        max_iterations: int = MAX_AGENT_ITERATIONS,
    ) -> None:
        self._pipeline = EnnoiaCorePipeline(
            schemas=[ContractMeta, ClauseInventory, ContractOverview, ClauseMention],
            store=Store(
                vector=InMemoryVectorStore(),
                structured=InMemoryStructuredStore(),
            ),
            llm=OllamaAdapter(model=gen_model, host=OLLAMA_HOST, timeout=GEN_TIMEOUT_SEC),
            embedding=SentenceTransformerEmbedding(model=embed_model),
            concurrency=concurrency,
        )
        self._gen_model = gen_model
        self._max_iterations = max_iterations
        # Lazy-imported inside ``answer`` so tests can monkeypatch the factory
        # before the first call; kept as an attribute for that hook. The
        # agent loop still drives an OpenAI-shaped client — we point it at
        # Ollama's OpenAI-compatible endpoint so the tool-calling surface
        # stays identical to the prior run.
        self._client_factory = _default_openai_client

    async def index_corpus(self, contracts: list[Contract]) -> None:
        outer = asyncio.Semaphore(INDEX_CONCURRENCY)
        failures: list[tuple[str, str]] = []

        async def index_one(contract: Contract) -> None:
            async with outer:
                try:
                    await self._aindex_with_retry(contract)
                except Exception as exc:
                    failures.append((contract["source_id"], f"{type(exc).__name__}: {exc}"))

        await tqdm_asyncio.gather(
            *(index_one(c) for c in contracts),
            desc="ennoia index",
            unit="doc",
        )
        if failures:
            print(f"[ennoia] {len(failures)}/{len(contracts)} contracts failed to index:")
            for source_id, reason in failures[:10]:
                print(f"[ennoia]   - {source_id}: {reason}")
            if len(failures) > 10:
                print(f"[ennoia]   ... and {len(failures) - 10} more")

    async def _aindex_with_retry(self, contract: Contract) -> None:
        """Call the underlying pipeline's aindex with retry-on-transient.

        Retries only transient network errors from the OpenAI SDK
        (``APIConnectionError``, ``APITimeoutError``). Any other exception
        propagates on the first attempt — this is not a general-purpose
        retry; it is a poor-connection shim around the extraction calls.
        """
        for attempt in range(_INDEX_RETRY_ATTEMPTS):
            try:
                await self._pipeline.aindex(
                    text=contract["text"],
                    source_id=contract["source_id"],
                )
                return
            except _INDEX_RETRYABLE_EXC as exc:
                if attempt == _INDEX_RETRY_ATTEMPTS - 1:
                    raise
                backoff = _INDEX_RETRY_INITIAL_BACKOFF_SEC * (attempt)
                print(
                    f"[ennoia] transient {type(exc).__name__} on "
                    f"{contract['source_id']} (attempt {attempt + 1}/"
                    f"{_INDEX_RETRY_ATTEMPTS}); retrying in {backoff:.1f}s"
                )
                await asyncio.sleep(backoff)

    async def answer(self, question: Question) -> PipelineRun:
        client = self._client_factory()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": question["question"]},
        ]
        retrieved_ids: list[str] = []
        trace: list[dict[str, Any]] = []
        prompt_tokens = 0
        completion_tokens = 0
        final_answer = "NOT_FOUND"

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
            # answer turn with tool_choice="none" so we always get something
            # to judge.
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
            limit = int(args.get("limit", RETRIEVAL_TOP_K))
            limit = max(1, min(limit, 25))
            try:
                result = await self._pipeline.asearch(query=query, filters=filters, top_k=limit)
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
    """Convert an OpenAI ChatCompletionMessage to the dict form the SDK
    accepts back as a prior assistant message."""
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


def _default_openai_client() -> Any:
    from openai import AsyncOpenAI

    # Ollama ships an OpenAI-compatible endpoint at `/v1` that accepts the
    # standard `tools` / `tool_choice` parameters. Re-using AsyncOpenAI here
    # keeps the agent loop's tool-call parsing identical to the prior run
    # against the real OpenAI API — no separate code path to maintain.
    return AsyncOpenAI(
        base_url=f"{OLLAMA_HOST}/v1",
        api_key=os.environ.get("OLLAMA_API_KEY", "ollama"),
        timeout=GEN_TIMEOUT_SEC,
    )
