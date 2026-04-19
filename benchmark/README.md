# ESCI product-discovery benchmark

Compares the **ennoia DDI agent loop** against a textbook **LangChain
one-embedding-per-product** baseline on a sampled slice of the
[ESCI-US product corpus](https://huggingface.co/datasets/spacemanidol/ESCI-product-dataset-corpus-us).

![Benchmark chart](results/chart_latest.png)

## Methodology

| Component             | Setting                                                                                              |
| --------------------- | ---------------------------------------------------------------------------------------------------- |
| Dataset               | `spacemanidol/ESCI-product-dataset-corpus-us` (1.21M US products)                                    |
| Sample                | `--num-samples 1000` products from a seeded random start offset (seed `42`)                          |
| Query set             | 3 LLM-generated variants per product: **broad** (no brand/name), **medium** (product name or attr), **high** (brand + meta). Price is forbidden and post-checked. |
| Generator + Judge     | `google/gemma-4-26b-a4b-it:free` via OpenRouter (free tier, zero-priced)                             |
| Embedding             | `text-embedding-3-small` via OpenAI (paid — the only spend bucket)                                   |
| Retrieval k           | top-10 (precision@5 also reported)                                                                   |
| Ennoia schemas        | `ProductMeta` (brand / color / category / product_type, structural) + `ProductSummary` (semantic) + `ProductFeatures` (collection, one row per feature) |
| Ennoia retrieval      | Agent loop — LLM calls `get_search_schema`, `search(query, filter, limit)`, `get_full(document_id)` tools. Max 6 iterations. Source IDs aggregated across every `search` call. |
| LangChain pipeline    | One `Document` per product (title + description + bullets concatenated) → `text-embedding-3-small` → `InMemoryVectorStore` + shared generator prompt. **No chunking.** |

Both pipelines run the **same** OpenRouter model for answer generation
and judging, and the **same** OpenAI embedder, so the only variable under
test is retrieval shape — naive single-embedding vector search vs DDI
structural-plus-semantic agentic search.

## Query-band design

Each product gets three shopper-style queries, modelling a realistic
search-box spectrum:

| Band     | Mentions brand | Mentions name / attrs | Example                                       |
| -------- | :------------: | :-------------------: | --------------------------------------------- |
| `broad`  | no             | generic use-case only | "a thing to cool down my laptop"              |
| `medium` | no             | product name or attr  | "external laptop cooler with usb power"       |
| `high`   | yes            | brand + meta          | "ASUS ROG laptop stand with RGB lights"       |

The prep script post-checks the LLM output for price leakage
(`$`, `USD`, `cheap`, `under \d+`, …). Leaks trigger one regeneration; on
a second failure the product is dropped from the dataset.

## Reproducing

```bash
# 1. Install the benchmark extra (pulls langchain, datasets, matplotlib,
#    tiktoken, tqdm, openai).
uv pip install -e ".[benchmark]"

# 2. Export provider keys.
export OPENROUTER_API_KEY=sk-or-...
export OPENAI_API_KEY=sk-...

# 3. Prepare the dataset (writes benchmark/data/dataset.json).
uv run python -m benchmark.data.prep \
    --num-samples 1000 --seed 42 --threads 16 \
    --llm google/gemma-4-26b-a4b-it:free

# 4. Dry-run smoke test — no API calls.
uv run python -m benchmark.runner --dry-run --num-samples 20 --chart

# 5. Full run.
uv run python -m benchmark.runner --threads 8 --num-samples 1000 --chart
uv run python -m benchmark.plot
```

The only paid bucket is OpenAI embedding (`text-embedding-3-small`,
$0.02 / 1M input tokens). At n=1000 the embedder spend is under $0.05.
The runner still enforces a `--max-cost-usd` cap to catch mis-typed
model names that would route to a paid OpenAI chat endpoint by mistake.

### CLI flags — prep

| Flag            | Default                                 | Purpose                                          |
| --------------- | --------------------------------------- | ------------------------------------------------ |
| `--num-samples` | 1000                                    | Products to slice from ESCI.                     |
| `--seed`        | 42                                      | Seeds the random start offset.                   |
| `--threads`     | 8                                       | Parallel OpenRouter calls for query generation.  |
| `--output`      | `benchmark/data/dataset.json`           | Where to write the prepared dataset.             |
| `--llm`         | `google/gemma-4-26b-a4b-it:free`        | OpenRouter model handle.                         |

### CLI flags — runner

| Flag                       | Default                        | Purpose                                                |
| -------------------------- | ------------------------------ | ------------------------------------------------------ |
| `--num-samples N`          | 1000                           | Max product cases to evaluate.                         |
| `--seed N`                 | 42                             | Sampling seed.                                         |
| `--threads N`              | 8                              | Drives both indexing + QA asyncio semaphores.          |
| `--dataset PATH`           | `benchmark/data/dataset.json`  | Prepared dataset JSON.                                 |
| `--chart`                  | False                          | Render the comparison chart after summarising.         |
| `--max-cost-usd X`         | 30                             | Hard spend cap; aborts the run when exceeded.          |
| `--only ennoia\|langchain` | both                           | Run a single pipeline (debug).                         |
| `--skip-index`             | False                          | Reuse existing in-memory index (no-op for fresh runs). |
| `--dry-run`                | False                          | No API calls; emit synthetic records to validate plot. |
| `--confirm-cost`           | False                          | Skip the pre-flight estimate gate.                     |

### Environment overrides

| Env var                 | Default                            | Purpose                                            |
| ----------------------- | ---------------------------------- | -------------------------------------------------- |
| `BENCHMARK_MODEL_LLM`   | `google/gemma-4-26b-a4b-it:free`   | OpenRouter handle (gen + judge share one model).   |
| `BENCHMARK_MODEL_EMBED` | `text-embedding-3-small`           | OpenAI embedding model.                            |
| `OPENROUTER_API_KEY`    | —                                  | OpenRouter auth for gen/judge.                     |
| `OPENAI_API_KEY`        | —                                  | OpenAI auth for embedding.                         |

## Outputs

After a run, `benchmark/results/` contains:

- `raw_<timestamp>.jsonl` — one record per `(product, difficulty, pipeline)`
  triple: query text, retrieved docids, generated answer, judge verdict,
  rationale, precision@k + hit@k flags, and (for ennoia) the agent trace.
- `summary_<timestamp>.csv` — aggregated metrics grouped by
  `(pipeline, difficulty)`.
- `chart_<timestamp>.png` — grouped bar chart (three panels, one per
  difficulty band, also copied to `chart_latest.png`).

## Metrics

- **Precision@k** (`k ∈ {5, 10}`): `1/k` when the gold product's `docid`
  is in the top-k retrieved unique docids, else `0`. Equivalent to the
  standard single-gold precision definition.
- **Hit@k**: boolean — gold `docid` present in top-k. Useful for plotting
  retrieval quality as a share of queries.
- **Judge verdicts** (`correct` / `partial` / `hallucinated` / `abstained`):
  - `correct` — recommendation matches the gold product.
  - `partial` — related-but-different product (same brand-and-category or
    narrow product_type family).
  - `hallucinated` — recommendation substantively unrelated, or asserts
    facts retrieval did not surface.
  - `abstained` — candidate said `NOT_FOUND`.

## Expected shape of the result

The benchmark is designed to isolate the regime where DDI's structural
filters pay off. LangChain's single-embedding baseline is strong on
**high**-precision queries (brand + feature are verbatim in the
embedding input), decent on **medium**, and collapses on **broad**
queries — a 3-word "thing to cool my laptop" has no lexical anchor in
the product listings. Ennoia's agent-loop can still pivot on structural
filters (`product_type__eq`, `category__eq`) even when the query is
vague. The chart renders all three difficulty bands side-by-side so the
asymmetry shows up explicitly.

## Limitations

- ESCI has no price field. Queries do not mention price; the prep script
  refuses and post-checks any price leakage from the query-generation
  LLM. A benchmark exercising numeric `price__lt` filters is out of
  scope here.
- LLM-as-judge using the same Gemma free tier as the generator carries a
  mild self-grading bias. The judge prompt forces JSON output and is
  fed the gold product (title + brand + bullets, never the docid) plus
  the retrieval top-K, so it cannot trivially favour the candidate's
  string-matching output.
- Single-locale (`us`) sample.
- LangChain baseline is intentionally textbook — no reranking, no
  metadata filters, no HyDE. A tuned LangChain pipeline could close some
  of the gap from the other direction.

## Cost guardrails

The runner prints a pre-flight token/USD estimate before the first call.
Live spend is tracked per call (estimated via `tiktoken`) and the run
aborts when `--max-cost-usd` is hit. Partial results are flushed as they
complete, so an aborted run is still analysable.
