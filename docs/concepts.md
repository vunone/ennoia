# Concepts

Ennoia introduces **Declarative Document Indexing Schemas (DDI Schemas)**
for RAG — a pre-indexing approach where LLM-powered extraction is defined
through schemas and executed *before* documents enter any store,
replacing naive chunk-and-embed with structured, queryable indices.

> Traditional RAG is like feeding your documents through a shredder and
> then trying to answer questions by pulling out strips of paper one by
> one. Ennoia is like reading each document first, taking structured
> notes on what matters, and then searching your notes — while keeping
> the originals on the shelf.

## The indexing model

An Ennoia pipeline runs an LLM over each document to produce two kinds of
output:

- **Structural extractions.** A `BaseStructure` is a Pydantic model whose
  fields describe the document's metadata — enums, dates, booleans, lists.
  The extractor prompts the LLM with the class docstring as the task and
  the JSON Schema as the contract, then validates the response with
  Pydantic. Structural extractions drive filtering at search time.
- **Semantic extractions.** A `BaseSemantic` is a question the LLM answers
  in natural language. The answer is embedded and stored for vector
  search. Each semantic class becomes a named "index" the agent can query
  specifically.

Both are declared by the user, in user code. Ennoia ships the machinery
(prompting, retries, parallel execution, storage) but has no opinion about
what you extract.

## `extend()` — the runtime DAG

Inter-schema dependencies are expressed at runtime: a `BaseStructure`'s
`extend()` method returns further schemas to run after the parent is
extracted. The parent instance is available via `self`, including its
self-reported confidence via `self.confidence`, so conditional branching
— "only attempt the expensive schema if the parent is above 0.85
confidence" — is a one-liner.

```python
class CaseDocument(BaseStructure):
    """Extract case metadata."""
    jurisdiction: Literal["WA", "NY", "TX"]

    def extend(self):
        if self.jurisdiction == "WA":
            return [WashingtonSpecificSchema]
        return []
```

The pipeline builds layers dynamically and runs each layer with
`asyncio.gather`, so sibling branches within a layer extract concurrently.

## Superschema

Every class `extend()` can return must be listed in `Schema.extensions`
on the emitting schema. At pipeline initialization, Ennoia walks this
declaration transitively and collapses the fields from every reachable
structural schema into a single unified field space called the
**superschema**. Filters, discovery output, and the structured store all
read from this one source of truth, so an agent never has to reason
about which `extend()` branch fired — every possible field is known
ahead of time, merged, and namespaced consistently. See
[schemas.md](schemas.md) for the merging rules and
[filters.md](filters.md) for the discovery payload shape.

## Confidence

Confidence is not a declared field on `BaseStructure`. If it were, it would
appear at the top of the JSON Schema and bias the model to guess a score
before filling in the real fields. Instead, the extractor dynamically
appends `extraction_confidence` as the final property of the prompted
schema and instructs the model to emit it last — so the score is
evaluated *after* the extraction has been produced. `BaseStructure` and
`BaseCollection` are configured with `ConfigDict(extra="allow")`, which
lets the model's `extraction_confidence` ride on the validated instance.
Access it via the `self.confidence` property; when the LLM omits the
field the property falls back to `Schema.default_confidence` (default
`1.0`). The pipeline strips the extra before persistence.

## Two-phase retrieval

Queries flow through the same two-phase path:

1. **Structured filter.** The query's `filters=` dict is validated against
   the schema's filter contract (unknown fields / unsupported operators
   raise `FilterValidationError`) and evaluated against the structured
   store to narrow the candidate set.
2. **Vector search.** The query is embedded, and cosine similarity runs
   *within* the candidate set.

This is why filter correctness matters: the vector search never sees
documents the filter eliminated. See [filters.md](filters.md) for the full
operator language.

## Storage

A composite `Store` pairs a structured backend (in-memory, SQLite,
Parquet) with a vector backend (in-memory, filesystem NumPy). The two
phases operate on their respective halves. Custom backends implement the
`StructuredStore` / `VectorStore` ABCs — see [stores.md](stores.md).

## Observability

Every extraction, indexing, and search operation emits a typed dataclass
event through an optional `Emitter`. Handlers are synchronous and isolated
(exceptions are logged and swallowed), so you can wire cost tracking,
quality monitoring, or ad-hoc debugging without touching the pipeline
itself.
