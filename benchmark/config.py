"""Centralised constants for the CUAD benchmark.

Model IDs, paths, sampling parameters, and pricing live here so every module
points at one source of truth.

By default the benchmark runs the **generator and embedder locally** (Ollama
+ sentence-transformers) and keeps only the **judge on the OpenAI API**.
Running the generator locally makes the full benchmark essentially free —
only judge tokens cost money — at the price of slower per-call latency and
weaker tool-calling reliability on a ≤1B-parameter model.

Override model IDs at runtime via env vars without editing code:
``BENCHMARK_MODEL_GEN``, ``BENCHMARK_MODEL_EMBED``, ``BENCHMARK_MODEL_JUDGE``,
``OLLAMA_HOST``.
"""

from __future__ import annotations

import os
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parent
DATA_DIR = BENCHMARK_ROOT / "data"
HF_CACHE_DIR = DATA_DIR / "cache"
SAMPLE_PATH = DATA_DIR / "sample.json"
DRYRUN_FIXTURES_PATH = DATA_DIR / "dryrun_fixtures.json"
RESULTS_DIR = BENCHMARK_ROOT / "results"

# Generator and embedder run locally by default. Judge stays on OpenAI
# because grading quality is the most load-bearing part of the chart.
MODEL_GEN = os.environ.get("BENCHMARK_MODEL_GEN", "qwen3:0.6b")
MODEL_EMBED = os.environ.get("BENCHMARK_MODEL_EMBED", "all-MiniLM-L6-v2")
MODEL_JUDGE = os.environ.get("BENCHMARK_MODEL_JUDGE", "gpt-5.4")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Per-contract curation: we sample a stratified corpus, then for each
# contract keep a fixed mix of positive (clause present, gold span non-empty)
# and negative (clause absent, gold empty) questions. Positive-heavy to mirror
# realistic usage; negatives guard against confident-wrong-document
# hallucination. 50 contracts x 10 questions = 500 total.
DEFAULT_CONTRACT_COUNT = 50
DEFAULT_QA_COUNT = 500
POS_PER_CONTRACT = 7
NEG_PER_CONTRACT = 3
SEED = 42

K_VALUES = (5, 10)
RETRIEVAL_TOP_K = max(K_VALUES)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

INDEX_CONCURRENCY = 1
QA_CONCURRENCY = 1

# Hard cap on the ennoia agent tool-calling loop. One iteration is a single
# LLM call that may emit zero or more tool calls; when the assistant answers
# without invoking a tool the loop exits. If the cap is hit the runner forces
# one more turn with tool_choice="none" so the question always produces an
# answer for the judge. 6 iterations covers the expected trajectory
# (discover -> search -> refine -> get_full -> answer) with margin.
MAX_AGENT_ITERATIONS = 6

# System prompt for the ennoia agent loop. Mirrors the production DDI flow:
# discover schema -> filtered search -> fetch full record on the best
# candidate -> answer from the full record. The prior "get_full is an escape
# hatch" wording starved the generator of context even when retrieval had
# already surfaced the right contract; `search` returns retrieval-optimised
# snippets (short verbatim anchors + gists), while `get_full` returns the
# full structured record that actually supports a precise extractive answer.
AGENT_SYSTEM_PROMPT = """You are a legal contract analyst answering a question over an Ennoia document index. You have three tools:

1. get_search_schema() — returns the structural fields (with types and operators) and semantic indices available. Call this FIRST, before any `search()` call, to know which `filter` fields are acceptable.
2. search(query, filter, limit) — runs a filtered vector search. Pass a natural-language semantic query and (optionally) a structural filter built from the discovered fields, to improve the search accuracy. Returns top hits with source_id, score, structural record, and best-matching semantic snippet.
3. get_full(document_id) — fetch the FULL structured record for a source_id. `search` returns retrieval-optimised snippets (short gists + anchor quotes); the full record is what supports a precise, quoteable answer.

Standard flow:
- Call `get_search_schema` once, before any search.
- Call `search` with a semantic query plus (if useful) a structural filter built from the schema.
- If at least one hit plausibly matches the contract the question is about, call `get_full(document_id)` on that top candidate and answer from the full record. Snippet-only answers miss material clause text.
- If no hit plausibly matches (low scores, wrong contract identity, wrong clause family), do NOT call `get_full` speculatively — reply exactly `NOT_FOUND`.
- Only call `search` a second time if the first search returned nothing usable (do not keep refining endlessly).

Rules:
- Use ONLY information returned by the tools. Do not rely on outside legal knowledge.
- Quote or paraphrase only what the tools returned; preserve the verbatim wording where the gold answer is likely extractive.
- If the tools do not contain enough information to answer, reply exactly: NOT_FOUND
"""

# Per-call timeouts. The generator's is long because a small local model
# can stall on a long retrieval context; the judge's matches that to
# swallow OpenAI's own tail-latency spikes.
GEN_TIMEOUT_SEC = 300.0
JUDGE_TIMEOUT_SEC = 240.0

# USD per 1M tokens. Local models are zero-priced so the cost tracker still
# records token counts for audit while the spend cap only applies to the
# judge. Kept as a mapping (rather than a single "judge price") so running
# the benchmark against an OpenAI generator stays a config-only switch.
PRICING: dict[str, dict[str, float]] = {
    MODEL_GEN: {"input": 0.0, "output": 0.0},
    MODEL_EMBED: {"input": 0.0, "output": 0.0},
    "gpt-5.4": {"input": 1.25, "output": 10.00},
    "gpt-5.4-nano": {"input": 0.05, "output": 0.40},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
}

HF_DATASET = "theatticusproject/cuad-qa"
