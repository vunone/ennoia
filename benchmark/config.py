"""Centralised constants for the ESCI product-discovery benchmark.

Model IDs, paths, sampling parameters, and pricing live here so every module
points at one source of truth.

Both pipelines share **one** OpenRouter model for answer generation and
judging (free Gemma by default) plus OpenAI ``text-embedding-3-small`` for
the vector index. The only paid bucket is embedding; everything else is
zero-priced so long runs are essentially free.

Override at runtime without editing code via env vars:
``BENCHMARK_MODEL_LLM``, ``BENCHMARK_MODEL_EMBED``.
"""

from __future__ import annotations

import os
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parent
DATA_DIR = BENCHMARK_ROOT / "data"
HF_CACHE_DIR = DATA_DIR / "cache"
DATASET_PATH = DATA_DIR / "dataset.json"
RESULTS_DIR = BENCHMARK_ROOT / "results"

# One model runs both generation and judging. Gemma free on OpenRouter;
# override via BENCHMARK_MODEL_LLM for a different OpenRouter handle.
MODEL_LLM = os.environ.get("BENCHMARK_MODEL_LLM", "google/gemma-4-26b-a4b-it")
MODEL_EMBED = os.environ.get("BENCHMARK_MODEL_EMBED", "text-embedding-3-small")

# Dataset slug on HuggingFace. US-only corpus (~1.21M products).
HF_ESCI_DATASET = "spacemanidol/ESCI-product-dataset-corpus-us"

DEFAULT_NUM_SAMPLES = 1000
SEED = 42

# Cap streaming skip so a pathologically high offset cannot scan the whole
# dataset. Halved on each retry when the slice comes back short.
MAX_START_OFFSET = 1_000_000

# Precision@k and the retrieval-side top-k the runner feeds to the judge.
K_VALUES = (5, 10)
RETRIEVAL_TOP_K = max(K_VALUES)

# Single threads knob. Drives both per-pipeline indexing concurrency and
# runner QA concurrency; also used by prep.py for parallel query generation.
THREADS = 8

# Agent tool-calling loop cap. One iteration is one LLM call that may emit
# zero or more tool calls; when the assistant answers without invoking a
# tool the loop exits. Six iterations covers discover -> search -> refine
# -> get_full -> answer with margin. On exhaustion the runner forces one
# more turn with tool_choice="none" so every query produces a judgable
# answer.
MAX_AGENT_ITERATIONS = 6

AGENT_SYSTEM_PROMPT = """You are a shopping assistant recommending a single product from an Ennoia catalogue index. You have three tools:

1. get_search_schema() — returns the structural fields (with types and operators) and semantic indices available on products. Call this FIRST, before any `search()` call, so you know which `filter` fields exist.
2. search(query, filter, limit) — runs a filtered vector search. Pass a natural-language query plus (when helpful) a structural filter built from the discovered fields — brand, category, product_type, color. Returns top hits with docid (= source_id), score, structural record, and best-matching semantic snippet.
3. get_full(document_id) — fetches the full original product record by source_id (returned from a prior search hit). Use to confirm a candidate against the unredacted document before recommending it.

Standard flow:
- Call `get_search_schema` once, up front.
- Call `search` with a semantic query plus (if useful) a structural filter.
- Optionally call `get_full` on the most promising hit to confirm before recommending.
- Reply with ONLY a JSON object: {"docid": "<product_id>", "reason": "<one short sentence>"} — where ``docid`` is the product the user should buy.
- If no hit plausibly matches, reply exactly: NOT_FOUND

Rules:
- Use ONLY information returned by the tools.
- Recommend exactly one product. Do not list alternatives.
- Price is not indexed and must not appear in your reasoning.
"""

# Per-call timeouts.
GEN_TIMEOUT_SEC = 120.0
JUDGE_TIMEOUT_SEC = 120.0

# USD per 1M tokens. Free OpenRouter models are zero-priced so the cost
# tracker still records token counts for audit while the spend cap only
# applies to the embedder.
PRICING: dict[str, dict[str, float]] = {
    MODEL_LLM: {"input": 0.0, "output": 0.0},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
}
