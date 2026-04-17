# CUAD precision benchmark

Compares the **ennoia DDI agent loop** against a textbook **langchain
shred-embed RAG** baseline on a stratified sample of the
[CUAD legal-contracts QA dataset](https://huggingface.co/datasets/theatticusproject/cuad-qa).

![Benchmark chart](results/chart_latest.png)

## Methodology

| Component             | Setting                                                        |
| --------------------- | -------------------------------------------------------------- |
| Dataset               | `theatticusproject/cuad-qa` (SQuAD-format)                     |
| Sample                | 50 contracts × 10 questions = 500 Q&A pairs (7 positive + 3 negative per contract), seed `42` |
| Query reformulation   | Each CUAD question prefixed with identity context (parties / type / date / governing law) parsed from the filename + CUAD gold spans. See "Query reformulation" below. |
| Generator (both runs) | `qwen3:0.6b` via local Ollama (`OLLAMA_HOST`, default `http://localhost:11434`) |
| Embedding (both runs) | `all-MiniLM-L6-v2` via `sentence-transformers` (local CPU)     |
| Judge                 | `gpt-5.4` via OpenAI (more capable than the generator — quality buffer; the only paid model) |
| Retrieval k           | top-10 (recall@5 also reported)                                |
| Ennoia schemas        | `ContractMeta`, `ClauseInventory` (structural) + `ContractOverview` (semantic) + `ClauseMention` (collection, one row per clause) |
| Ennoia retrieval      | Agent loop — LLM calls `get_search_schema`, `search(query, filter, limit)`, `get_full(document_id)` tools. Max 6 iterations. Source IDs aggregated across every `search` call for recall@k. `get_full` is now part of the standard flow (fetch full record on the top candidate before answering), not an escape hatch — see "What the current numbers actually show" below. |
| Langchain pipeline    | `RecursiveCharacterTextSplitter(1000, 150)` → local `all-MiniLM-L6-v2` embeddings → `InMemoryVectorStore` + shared generator prompt |

Both pipelines run the same local `qwen3:0.6b` generator under the same
NOT_FOUND escape, so the variable under test is **retrieval + agent
reasoning quality**, not generator skill. The agent loop still drives an
OpenAI-shaped client — it just points at Ollama's `/v1/chat/completions`
OpenAI-compatible endpoint, which accepts the same `tools` / `tool_choice`
parameters. The agent is never told the CUAD clause label for a question —
it discovers the filterable fields itself.

## Query reformulation

CUAD's original questions say "parts (if any) of **this contract**
related to …" — they assume a single-document review and never identify
which contract the user means. In a multi-document index that makes
retrieval impossible: every contract gets every category asked, and ~52%
of questions legitimately expect `NOT_FOUND`, collapsing the benchmark
into noise.

We restore the disambiguating context a real user would naturally
provide. Each question is prefixed with identity facets parsed
deterministically from the CUAD filename + identity-gold spans
(`Parties`, `Agreement Date`, `Governing Law`, `Document Name`), e.g.

> **Original:** *"Highlight the parts (if any) of this contract related to 'Non-Compete'…"*
>
> **Reformulated:** *"In the 2019 Development Agreement between Fuelcell
> Energy Inc and Exxonmobil Research and Engineering Company, highlight
> the parts (if any) of this contract related to 'Non-Compete'…"*

Three templates are used (party-focused / title-focused / law-focused),
chosen deterministically by hashing the CUAD question id. Gold spans are
preserved verbatim — we only change the wrapper. Identity categories
themselves (Parties, Document Name, Agreement Date, Effective Date,
Governing Law) are excluded from the sampled questions since the
reformulation embeds their answer.

Per contract we keep **7 positive + 3 negative** questions (non-empty
gold / empty gold respectively). Positive-heavy to mirror realistic
usage; negatives guard against confident-wrong-document hallucination.

## Reproducing

```bash
# 1. Install the benchmark extra (pulls langchain, datasets, matplotlib,
#    tiktoken, tqdm, sentence-transformers, ollama-python).
uv pip install -e ".[benchmark]"

# 2. Start Ollama locally and pull the generator model (~400MB on disk).
#    Skip if you already run ollama + have the tag cached.
ollama serve &
ollama pull qwen3:0.6b

# 3. Smoke-test the wiring without spending any money.
uv run python -m benchmark.runner --dry-run --limit 20
uv run python -m benchmark.plot

# 4. Tiny live smoke (~$0.05 — only judge tokens cost money now).
export OPENAI_API_KEY=sk-...
uv run python -m benchmark.runner --limit 5 --max-cost-usd 1

# 5. Full run. With the local generator the only spend is the judge
#    (~$2–$4 at n=500). The local generator is slow on CPU, so budget
#    wall-clock hours rather than dollars. Cap defaults to $30.
uv run python -m benchmark.runner --limit 500
uv run python -m benchmark.plot
```

The first non-dry run downloads CUAD into `benchmark/data/cache/`
(gitignored, ~50 MB) and writes the stratified slice to
`benchmark/data/sample.json`. It also downloads the `all-MiniLM-L6-v2`
sentence-transformers model into the Hugging Face cache (~80 MB) the
first time either pipeline indexes. Subsequent runs reuse the cached
sample and model for full reproducibility under the same seed.

### CLI flags

| Flag                 | Default | Purpose                                                |
| -------------------- | ------- | ------------------------------------------------------ |
| `--limit N`          | 500     | Max questions to evaluate.                             |
| `--seed N`           | 42      | Sampling seed.                                         |
| `--contract-count N` | 50      | How many contracts to keep in the corpus.              |
| `--max-cost-usd X`   | 30      | Hard spend cap; aborts the run when exceeded.          |
| `--only ennoia\|langchain` | both | Run a single pipeline (debug).                       |
| `--skip-index`       | False   | Reuse existing in-memory index (no-op for fresh runs). |
| `--dry-run`          | False   | No API calls; emit synthetic records to validate plot. |
| `--rebuild-sample`   | False   | Re-download CUAD and regenerate `sample.json`.         |
| `--confirm-cost`     | False   | Skip the pre-flight estimate gate.                     |

### Model overrides

Swap the generator tag, the sentence-transformers embedder, the judge
model, or the Ollama endpoint without code changes:

```bash
# Defaults — local generator + embedder, OpenAI-only judge.
export BENCHMARK_MODEL_GEN=qwen3:0.6b
export BENCHMARK_MODEL_EMBED=all-MiniLM-L6-v2
export BENCHMARK_MODEL_JUDGE=gpt-5.4
export OLLAMA_HOST=http://localhost:11434

# Example: use a stronger local model (pull it first with `ollama pull`).
export BENCHMARK_MODEL_GEN=qwen3:1.7b
```

Going back to a hosted generator (e.g. OpenAI / Azure OpenAI) requires
swapping the adapter types in `benchmark/pipelines/ennoia_pipeline.py`
and `benchmark/pipelines/generator.py` — the default path is wired for
a local generator because that's the setup that lets the benchmark run
for the price of the judge alone.

## Outputs

After a run, `benchmark/results/` contains:

- `raw_<timestamp>.jsonl` — one record per `(question, pipeline)` pair: question
  text, retrieved `source_id`s, generated answer, judge verdict, judge rationale,
  and recall flags.
- `summary_<timestamp>.csv` — aggregated per-pipeline metrics.
- `chart_<timestamp>.png` — the grouped bar chart (also copied to `chart_latest.png`).

## Metrics

- **Recall@k**: did the gold contract's `source_id` appear in the top-k unique
  retrieved sources? For langchain this is the dedup of the top-10 chunks;
  for ennoia it is the dedup of every `source_id` the agent saw across all
  `search` tool calls. CUAD QA is single-document, so this directly measures
  retrieval quality.
- **Judge verdicts** (`correct` / `partial` / `hallucinated` / `abstained`):
  - `correct` — answer captures the gold span (paraphrased OK).
  - `partial` — captures part of the gold answer or adds unsupported content.
  - `hallucinated` — asserts unsupported facts when the gold span is non-empty.
  - `abstained` — said `NOT_FOUND` but the contract did contain the answer
    (a miss, but not a hallucination — the prompt explicitly invites this).
- **`answered_given_retrieved`** — share of positive questions whose gold
  contract landed in the top-`max(K_VALUES)` retrieval AND ultimately scored
  `correct` or `partial` from the judge. This is the retrieval-independent
  measure of generator-context quality: a pipeline can have strong recall
  and still bottom out here if the context it hands the generator is too
  thin to answer from.

## What the current numbers actually show

The chart above reports a single full run at n=500. Three things are worth
flagging before reading it as "shred-embed beats DDI":

1. **The retrieval gap is real but bounded by the benchmark shape.** At
   n=500, langchain leads recall@10 by ~5pp (z≈2.6, p<0.01) and recall@5
   by ~7pp (z≈3.2). CUAD QA is **single-document** — every question has
   exactly one gold contract — so recall@k here measures one narrow axis
   of retrieval, not the multi-document filter+semantic shape DDI is
   designed around. A future benchmark with set-valued gold (listed as
   future work below) is the honest test of DDI's structural advantage.
2. **The dominant effect is answer-step context, not retrieval.**
   Conditioning on positive questions where the gold contract *did* land
   in the top 10, ennoia still abstained **70%** of the time vs
   langchain's **57%** — a 13pp gap on the retrieval-independent slice.
   In the first full run, ennoia's schemas were retrieval-optimised
   (≤60-word overview, ≤30-word verbatim anchors) while langchain handed
   the generator ~10 000 characters of raw contract text; on extractive
   CUAD questions the generator can only quote what's in its window, so
   ennoia's window was structurally too small — 81% of positive questions
   had a gold span longer than 30 words. This is now fixed at the schema
   level: `ContractOverview` and `ClauseMention` no longer carry length
   caps on their summary/verbatim fields (the LLM picks the natural
   clause boundary), and the full contract text is passed to the
   extraction step (the former 50 000-char truncation is removed). The
   `get_full` tool was also previously restricted by the system prompt to
   an escape hatch and its call trace was not persisted; both are now
   fixed — `get_full` is part of the standard flow (see "Methodology"
   above), and every ennoia record carries its agent `trace` for audit.
3. **Secondary claims are noisier than the chart suggests.** The
   `correct` delta (29% vs 34%) is borderline at n=500 (z≈1.7, p≈0.09);
   the `hallucinated` delta (1.4% vs 2.8%) is **not** statistically
   significant (z≈1.5). Treat the "ennoia hallucinates less" story as
   suggestive, not settled, until n is larger or the answer-step fix
   lands.

The re-run that measures whether the fix closes the gap is intentionally
not included in this PR — it is the next scheduled benchmark spend. The
success criterion for that run is: `answered_given_retrieved` closes the
gap to langchain while recall@10 moves only marginally.

## Limitations

- CUAD QA is **single-document** — every question has exactly one gold
  contract. Multi-document recall is not exercised, so the axis where DDI
  is expected to win by construction (filter-scoped retrieval across a
  heterogeneous corpus) is not measured here. A multi-document benchmark
  with set-valued gold is the honest next step and is listed as future
  work.
- LLM-as-judge using `gpt-5.4` to grade `gpt-5.4-nano` answers carries a
  known leniency bias. The judge prompt mitigates this by including the
  gold spans verbatim and forcing the model to grade against them.
- Sample of 50 contracts / ~500 questions — enough for the headline
  retrieval deltas to be statistically significant, but marginal for the
  `correct` and `hallucinated` deltas; see "What the current numbers
  actually show" above for the specific z-scores.
- The langchain baseline is intentionally textbook (no reranking, no
  metadata filters). A heavily tuned langchain pipeline could close some
  of the gap from the other direction.

## Cost guardrails

The runner prints a pre-flight token/USD estimate before the first call. Live
spend is tracked per call (estimated via `tiktoken` cl100k for the gpt-5.4
family) and the run aborts when `--max-cost-usd` is hit. Partial results are
flushed as they complete, so an aborted run is still analysable.
