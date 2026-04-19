# Product-discovery benchmark

The `benchmark/` harness in the repository compares **Ennoia's DDI
agent loop** against a textbook **LangChain one-embedding-per-product**
baseline on a sampled slice of the
[ESCI-US product corpus](https://huggingface.co/datasets/spacemanidol/ESCI-product-dataset-corpus-us)
(1.21M US products; fields `docid / title / text / bullet_points /
brand / color`).

## Methodology

| Component             | Setting                                                                 |
| --------------------- | ----------------------------------------------------------------------- |
| Dataset               | `spacemanidol/ESCI-product-dataset-corpus-us`, seeded random slice      |
| Sample                | `--num-samples 1000` products (seed `42`)                               |
| Query set             | 3 LLM-generated shopper variants per product: `broad` / `medium` / `high` |
| Generator + Judge     | `google/gemma-4-26b-a4b-it:free` via OpenRouter (zero-priced)           |
| Embedding             | `text-embedding-3-small` via OpenAI (the only paid bucket)              |
| Retrieval k           | top-10 (precision@5 also reported)                                      |
| Ennoia schemas        | `ProductMeta` (brand / color / category / product_type) + `ProductSummary` + `ProductFeatures` (one row per bullet) |
| Ennoia retrieval      | Agent loop with `get_search_schema` / `search` / `get_full`. Max 6 iterations. |
| LangChain baseline    | One `Document` per product (title + description + bullets concatenated), no chunking |

Both pipelines share the same LLM and the same embedder, so the only
variable under test is retrieval shape — naive single-embedding vector
search vs DDI structural-plus-semantic agentic search.

## Query bands

Each product gets three shopper-style queries, modelling the realistic
search-box spectrum:

| Band     | Brand? | Name / attrs?        | Example                                          |
| -------- | :----: | -------------------- | ------------------------------------------------ |
| `broad`  | no     | generic use-case     | "a thing to cool down my laptop"                 |
| `medium` | no     | product name or attr | "external laptop cooler with usb power"          |
| `high`   | yes    | brand + meta         | "ASUS ROG laptop stand with RGB lights"          |

Price mentions are forbidden by the prompt and post-checked in prep;
ESCI has no price field, so any leakage is fabrication.

## Reproducing

```bash
uv pip install -e ".[benchmark]"
export OPENROUTER_API_KEY=sk-or-...
export OPENAI_API_KEY=sk-...

# Prep (writes benchmark/data/dataset.json).
uv run python -m benchmark.data.prep \
    --num-samples 1000 --seed 42 --threads 16

# Dry-run sanity check — no API calls.
uv run python -m benchmark.runner --dry-run --num-samples 20 --chart

# Full run (cost dominated by the OpenAI embedder; <$0.10 at n=1000).
uv run python -m benchmark.runner --threads 8 --num-samples 1000 --chart
```

The full option list and environment overrides live in
[`benchmark/README.md`](https://github.com/vunone/ennoia/tree/main/benchmark/README.md).

## What the chart shows

`benchmark/plot.py` renders three side-by-side panels — one per
difficulty band — each with bars for precision@5, precision@10, hit@5,
hit@10, and the four judge verdicts (correct / partial / hallucinated
/ abstained), for Ennoia and LangChain.

LangChain's single-embedding baseline is strong on **high**-precision
queries (brand + feature appear verbatim in the embedding input),
decent on **medium**, and collapses on **broad** queries where there's
no lexical anchor. Ennoia's agent can still pivot on structural
filters (`product_type__eq`, `category__eq`) even when the query is
vague. The three-panel layout makes that asymmetry explicit instead of
hiding it in a single aggregated number.
