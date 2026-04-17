# CUAD precision benchmark

A reproducible head-to-head between **ennoia DDI agent retrieval** and a
textbook **langchain shred-embed RAG** baseline on the
[CUAD legal contracts QA dataset](https://huggingface.co/datasets/theatticusproject/cuad-qa).
The harness lives in [`benchmark/`](https://github.com/vunone/ennoia/tree/main/benchmark)
in the repo and is *not* shipped in the published wheel.

The point isn't just to publish a chart — it's to show *how* you instrument a
DDI pipeline against a baseline so you can run the same comparison on your own
domain (medical, finance, internal compliance) before committing to one or the
other.

## Why CUAD is a good fit for DDI

CUAD's questions are **categorical** — every question of the form
*"Highlight the parts (if any) of this contract related to '<clause name>'…"*
asks whether a specific clause appears in the contract. Naive chunk-and-embed
RAG has to discover the right clause through embedding similarity alone, which
breaks down when contracts share boilerplate language across unrelated
clauses.

DDI's pre-extraction step solves this directly: a `ClauseInventory` schema
extracts which clause buckets are present in each contract *before* indexing.
At query time, the ennoia agent discovers that schema via a tool call,
narrows candidates with a structural filter, and only then does vector search
rank within that subset — mirroring exactly what a production LLM app built
on top of Ennoia's MCP tools would do.

## The ennoia agent loop

The ennoia pipeline exposes three tools to the generator LLM (the same
local `qwen3:0.6b` model the langchain baseline uses for answer
generation — only the judge calls OpenAI):

| Tool | Purpose |
| --- | --- |
| `get_search_schema()` | Returns the discoverable structural fields (types + operators) and semantic indices. The system prompt forces the agent to call this first. |
| `search(query, filter, limit)` | Single-call filter+vector search. The agent builds `filter` itself from what `get_search_schema` returned. |
| `get_full(document_id)` | Fetch the full structured record for one document. Called on the top candidate once a `search` hit plausibly matches the target contract — the full record is what supports a precise extractive answer. Not called speculatively: if no hit plausibly matches, the agent replies `NOT_FOUND` directly. |

The loop runs up to `MAX_AGENT_ITERATIONS = 6` turns. If the agent hasn't
produced a textual answer by then, the runner forces one last turn with
`tool_choice="none"` so every question produces a verdict (or the literal
`NOT_FOUND`) for the judge.

**Recall@k** is measured over the ordered, deduplicated union of
`source_id`s returned by every `search` tool call during the loop — that
is, every document the agent *saw* during its reasoning. This matches how
Ennoia is used in production: the quality metric is "did the gold
document surface while the agent was searching?", not "did it appear in
some single pre-filtered call".

## Pipeline shapes side by side

```python
# Langchain baseline — chunks every contract uniformly, no metadata.
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
# Local sentence-transformers embeddings so the benchmark can run without
# spending on a hosted embedding model.
embeddings = SentenceTransformer("all-MiniLM-L6-v2")  # wrapped as LC Embeddings
store = await InMemoryVectorStore.afrom_documents(chunks, embeddings)
hits = await store.asimilarity_search_with_score(question, k=10)
# … feed hits to the shared generator prompt → NOT_FOUND or an answer.
```

```python
# Ennoia DDI agent — pre-extracts structured + semantic indices, then
# lets an LLM drive the retrieval via tool calls at query time.
pipeline = Pipeline(
    schemas=[ContractMeta, ClauseInventory, ContractOverview, ClauseMention],
    store=Store(vector=InMemoryVectorStore(), structured=InMemoryStructuredStore()),
    llm=OllamaAdapter(model="qwen3:0.6b", host="http://localhost:11434"),
    embedding=SentenceTransformerEmbedding(model="all-MiniLM-L6-v2"),
)
# ... index every contract via pipeline.aindex(text, source_id) ...

# Query time: the LLM sees question + three tools, decides how to use them.
# The agent loop reuses the OpenAI SDK pointed at Ollama's OpenAI-compatible
# /v1/chat/completions endpoint so tool calling stays identical to the
# hosted-API version.
# Typical trace: get_search_schema → search(filter=…) → get_full → answer
```

The full schemas live in
[benchmark/pipelines/schemas.py](https://github.com/vunone/ennoia/blob/main/benchmark/pipelines/schemas.py).

## Query reformulation (honest disclosure)

CUAD's original questions say "parts (if any) of *this contract*
related to …" and never identify which contract. Over a multi-document
index that makes retrieval meaningless: every contract in the corpus
gets every category asked, and ~52% of questions legitimately expect
`NOT_FOUND` because the clause is absent from that specific contract.
Running an agent against such queries measures nothing interesting.

We restore the disambiguating context a real user would naturally
provide. Each question is prefixed with identity facets parsed
deterministically from the CUAD filename + the dataset's own
identity-category gold spans (`Parties`, `Agreement Date`,
`Governing Law`, `Document Name`):

> **Original:** *"Highlight the parts (if any) of this contract related
> to 'Non-Compete'…"*
>
> **Reformulated (party template):** *"In the 2019 Development Agreement
> between Fuelcell Energy Inc and Exxonmobil Research and Engineering
> Company, highlight the parts (if any) of this contract related to
> 'Non-Compete'…"*

Three templates (party-focused, title-focused, law-focused) are picked
deterministically by hashing the CUAD question id. Gold spans are
preserved verbatim — only the wrapper changes. Identity categories are
excluded from the sampled question set since the reformulation already
embeds their answer.

Per contract we retain **7 positive + 3 negative** questions so the
benchmark still measures honest-`NOT_FOUND` behaviour without being
dominated by empty-gold noise.

## Honest disclosure — how to read the chart

CUAD is a *narrow* test for DDI. It is single-document extractive QA:
every question has exactly one gold contract, and the expected answer is
a short verbatim passage from that contract. Three caveats belong in
front of any reader of the chart:

1. **The retrieval delta is bounded by the benchmark shape.** A
   shred-embed pipeline ranks 1000-char raw chunks against the reformulated
   query; an identity-disambiguated query ("In the 2019 Development
   Agreement between X and Y, highlight...") has dense surface overlap
   with those chunks, so naive RAG competes well on *single-doc* recall.
   DDI's structural advantage — "find every contract with clause X under
   law Y" — never gets exercised here because CUAD has no such queries.
2. **The answer-step gap was larger than the retrieval gap, and was
   structural.** In the first full run (n=500), conditioning on
   positives where the gold contract *did* land in the top-10, ennoia
   abstained 70% of the time while langchain abstained 57% — a 13pp
   gap on the retrieval-independent slice. 81% of positive questions
   have a gold span longer than 30 words; the previous ≤30-word
   `ClauseMention.verbatim` cap and ≤60-word `ContractOverview` cap put
   the extractive answer out of reach by schema design. Those caps have
   since been removed and the extraction step now receives the full
   contract (the former 50 000-char truncation is gone), so the
   structured record carries clause-length quotes. The `get_full` tool
   existed but the prior system prompt discouraged its use; that prompt
   is now rewritten so `get_full` is part of the standard flow, and
   every ennoia record persists its agent `trace` so the next run can
   be audited for how often `get_full` actually fires. The
   `answered_given_retrieved` column in the summary CSV is the clean
   per-pipeline read of this effect.
3. **Secondary claims are not yet settled at n=500.** The `correct`
   delta (29% vs 34%) is borderline (z≈1.7, p≈0.09); the `hallucinated`
   delta (1.4% vs 2.8%) is not statistically significant (z≈1.5).
   "Ennoia hallucinates less" is suggestive in this sample, not proven.

The right way to read the current chart is: *shred-embed competes well
on single-document extractive QA, and an incautious reading of the
headline will miss that the dominant effect is generator-context size,
not retrieval quality*. A multi-document filter+semantic benchmark —
set-valued gold over the full CUAD corpus, synthesised from CUAD's
categorical + identity metadata — is the honest next step and is listed
as future work.

## How the harness ensures fairness

- **Same generator**: both pipelines use a local `qwen3:0.6b` via Ollama.
  Langchain hands its chunks to a fixed prompt; ennoia runs the same model
  in a tool loop. Neither pipeline gets access to a stronger model at
  generation time — the only paid model is the judge.
- **Same embedder**: both pipelines use a local `all-MiniLM-L6-v2`
  sentence-transformers model, wrapped with a minimal langchain
  `Embeddings` shim on the baseline side. Retrieval quality is a
  pipeline comparison, not an embedding-model comparison.
- **No category leakage**: the agent is never told the CUAD clause label
  for a question. It must infer which filter to apply from the discovered
  schema alone — same information a production user's LLM would have.
- **Reformulation is uniform**: both pipelines see the same reformulated
  query. The identity context is available as much to naive RAG as to the
  DDI agent — the agent just has structural fields to filter by, while
  the baseline must rely on similarity ranking alone.
- **Same NOT_FOUND escape**: both prompts instruct the model to reply
  `NOT_FOUND` rather than guess. The judge counts that as `abstained` — a
  non-hallucination — so neither pipeline can mask poor recall as
  confident wrong answers.
- **Stronger judge**: an LLM-as-judge using `gpt-5.4` (not nano) grades
  every answer against the human-extracted gold spans and emits one of
  `correct | partial | hallucinated | abstained` plus a one-sentence
  rationale.
- **Fixed seed**: stratified sample is deterministic per `--seed`; the
  selected QA slice is committed to `benchmark/data/sample.json` so a
  re-run reproduces the corpus exactly without re-downloading CUAD.

## Reproducing locally

```bash
pip install -e ".[benchmark]"

# Generator runs locally via Ollama — pull once, reuse across runs.
ollama serve &
ollama pull qwen3:0.6b

# Judge is the only paid model.
export OPENAI_API_KEY=sk-...

# Wiring smoke (no spend).
python -m benchmark.runner --dry-run --limit 20
python -m benchmark.plot

# Full run. Only the judge costs money now (~$2–$4 at n=500); the local
# generator trades dollars for wall-clock hours.
python -m benchmark.runner --limit 500 --max-cost-usd 30
python -m benchmark.plot
```

The runner persists per-question records to
`benchmark/results/raw_<timestamp>.jsonl` *as they complete*, so an
aborted-by-budget run is still analysable. Each ennoia row carries a
`trace` listing which tools the agent called and with what arguments.

## Adapting to your domain

To run the same comparison on your own corpus:

1. Pick a QA dataset over your documents (or build one — gold answers can be
   short paraphrases, not just extractive spans; tweak the judge prompt).
2. Replace [`benchmark/pipelines/schemas.py`](https://github.com/vunone/ennoia/blob/main/benchmark/pipelines/schemas.py)
   with your own DDI schemas. The high-leverage move is a structural enum
   that pre-classifies what each document contains — the agent will
   discover it automatically via `get_search_schema`.
3. Swap out [`benchmark/data/loader.py`](https://github.com/vunone/ennoia/blob/main/benchmark/data/loader.py)
   to load your dataset into the same `Contract` / `Question` typed dicts.
4. Everything else (agent loop, judge, metrics, plot) stays the same.

The same harness shape generalises beyond legal — medical protocol QA,
financial 10-K analysis, internal policy compliance — anywhere retrieval
quality depends on document *type* or *category* rather than just text
similarity.
