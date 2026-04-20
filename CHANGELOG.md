# Changelog

All notable changes to Ennoia are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
from v0.1.0 onward.

## [0.4.1] — 2026-04-20

### Fixed

- **CI green across the Python 3.11 / 3.12 / 3.13 matrix.** Pyright
  strict flagged `files(__package__)` in `ennoia/prompts/__init__.py`
  because `__package__` is typed as `str | None`; switched to
  `files(__name__)`, which is always a non-optional module name. The
  `ennoia try` `--schema`-missing test (`tests/test_cli_config.py`) also
  broke on CI because Rich-rendered Typer `BadParameter` output wraps
  `--schema` in ANSI color runs that defeat literal substring asserts;
  the test now strips ANSI escapes before matching.

## [0.4.0] — 2026-04-19

### Fixed

- **Nested filter values are no longer silently swallowed.**
  `Pipeline.search()` now raises `FilterValidationError` when a filter
  value is a mapping (e.g. `{"price_usd": {"lt": 80}}`), with a message
  that shows the flat-form rewrite (`{"price_usd__lt": 80}`). Previously
  the nested form was treated as an equality comparison against a dict,
  which always matched zero rows — so agents that picked the
  Elasticsearch/MongoDB-style syntax appeared to search successfully but
  retrieved nothing. Surfaced by the ESCI benchmark, where Gemma's
  nested price filters collapsed the medium-band ennoia run.
- **Benchmark `search` tool no longer lets the agent pick `limit`.**
  The `search` tool in `benchmark/pipelines/ennoia_pipeline.py` now
  always returns up to `RETRIEVAL_TOP_K` (10) hits to match the
  LangChain baseline; weak models were routinely picking `limit: 5`,
  which capped precision@10 at 0.1 even when the filter worked.

### Changed

- **Benchmark reworked: CUAD → ESCI product discovery.** The
  `benchmark/` harness now compares the Ennoia DDI agent loop against a
  naive one-embedding-per-product LangChain baseline on a sampled slice
  of `spacemanidol/ESCI-product-dataset-corpus-us`. LLM (gen + judge)
  runs on OpenRouter's free `google/gemma-4-26b-a4b-it:free`; embedder
  is OpenAI `text-embedding-3-small`. A prep script (`python -m
  benchmark.data.prep`) downloads a seeded slice of the corpus and
  LLM-generates three shopper-style query variants per product
  (`broad` / `medium` / `high`), with price mentions forbidden and
  post-checked. The runner (`python -m benchmark.runner --threads N
  --chart`) renders a three-panel chart of precision@k + judge verdicts
  per difficulty band. DDI schemas: `ProductMeta` (brand / color /
  category / product_type), `ProductSummary`, `ProductFeatures`
  (one row per bullet). The new bundled prompt
  `ennoia/prompts/benchmark_queries.md` drives query generation. All
  CUAD identity-reformulation / clause-taxonomy / Ollama scaffolding is
  removed.

### Added

- **`ennoia.ini` config file + `ennoia init` command.** A new
  `[ennoia]` section supplies defaults for `--llm`, `--embedding`,
  `--store`, `--schema`, `--collection`, `--qdrant-url`,
  `--qdrant-api-key`, `--pg-dsn`, `--host`, `--port`, `--transport`,
  and `--api-key`; a `[env]` section exports provider keys
  (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, …) via
  `os.environ.setdefault` before adapter clients initialize. Explicit
  CLI flags and shell-exported env vars always override the file.
  `ennoia init` writes a commented template; `--config <path>` /
  `--no-config` override the auto-load. See
  [docs/cli.md#configuration-via-ennoiaini](docs/cli.md).
- **`self.confidence` property + `Schema.default_confidence`.**
  `BaseStructure` and `BaseCollection` expose a read-only `confidence`
  property that returns the LLM's self-reported value when present, or
  `self.Schema.default_confidence` (defaulting to `1.0`) when the model
  omitted the field or emitted something invalid. Subclasses override
  the per-schema default via the inner `class Schema`. `extend()`
  consumers should read `self.confidence` instead of the (now removed)
  `self._confidence` attribute. New helper
  `get_schema_default_confidence(cls)` parallels the existing
  `get_schema_namespace` / `get_schema_extensions` /
  `get_schema_max_iterations` helpers.
- **MCP server-level instructions.** `create_mcp` now registers a
  `FastMCP(instructions=…)` block that briefs the LLM on the canonical
  `discover_schema → search → retrieve` call order and the filter
  grammar. MCP clients surface it to the model alongside the tool list,
  so an agent has the retrieval playbook before its first tool call.
  Per-tool docstrings were also rewritten as LLM-facing prompts
  (when-to-use, example, error shape) rather than terse API summaries.
- **`ennoia craft` CLI subcommand** — LLM-driven schema scaffolding from
  a sample document. Given `--task` ("filter products by price / colour")
  and a document path, an LLM emits a fixed three-class skeleton —
  `Metadata(BaseStructure)` with 5–10 plain filterable fields (no
  `Literal`, no `Optional`, no `list` — a human tightens those later
  once the value space is known), `QuestionAnswer(BaseCollection)` with
  `max_iterations = 1` generating ten Q&A pairs per document for
  semantic search, and `Summary(BaseSemantic)` for a one-sentence
  overview. If the output file exists, its current contents are passed
  through so the LLM *improves* the draft instead of rewriting it. Each
  attempt is validated by importing the file; on failure the traceback
  is fed back to the LLM with a `--max-retries` budget (default 2).
  Ships a bundled authoring guide at `ennoia/prompts/craft.md` — loaded
  via `importlib.resources`, not a public doc. Provider context-length
  errors surface with an actionable hint; Ennoia does not chunk the
  document — pick a model whose window fits it.
  **Prototype only:** the command emits a banner on start and after
  completion explicitly marking the output as a first draft. Tighten
  types, rename fields, and refine docstrings before running
  `ennoia index` on real data.
- **`ennoia.prompts` package** — tiny loader (`load_prompt("name")`) for
  prompt assets shipped with the wheel. Hatch `force-include` keeps
  non-`.py` assets out of accidental exclusion.
- **`load_schemas()` recognises `BaseCollection`** — the CLI loader used
  by `try`, `index`, and `search` now accepts all three extractor kinds,
  so schemas produced by `ennoia craft` that lean on collections work
  end-to-end through the rest of the CLI.
- **`OpenRouterAdapter` + `OpenRouterEmbedding`** — LLM and embedding
  adapters against OpenRouter's OpenAI-compatible API
  (`https://openrouter.ai/api/v1`). `OPENROUTER_API_KEY` environment
  fallback. Exposed on the CLI as `openrouter:<model>` (for `--llm`) and
  `openrouter-embedding:<model>` (for `--embedding`). Ships under a new
  `openrouter` extra that aliases `openai` — the official `openai` SDK is
  the canonical client, no new third-party dependency.
- **`BaseCollection`** — third extraction kind alongside `BaseStructure`
  and `BaseSemantic`. Extracts a *list* of structured entities from a
  document by iteratively prompting the LLM with a `<PreviouslyExtracted>`
  block until `is_done`, an empty result, no-new-uniques, or
  `Schema.max_iterations` stops the loop. Each entity renders to text via
  `template()` and lands in the vector store as its own row, so a
  collection with N entities behaves like N `BaseSemantic` answers under
  the same index name. New hooks: `is_valid()` (raise `SkipItem` to drop
  just the entity; `RejectException` still drops the whole document),
  `get_unique()` (dedup key, random by default), `template()`,
  `extend()` per entity.
- **`SkipItem` exception** — new per-entity skip signal for
  `BaseCollection.is_valid()`.
- **`benchmark/` directory** — CUAD-based comparison harness pitting
  ennoia DDI+RAG against a textbook langchain shred-embed RAG baseline.
  Produces a grouped bar chart (recall@k + LLM-as-judge verdict shares)
  saved to `benchmark/results/chart_latest.png`. Not shipped in the
  published wheel; excluded from pyright `include` and coverage `source`.
- **`benchmark` extra** — `pip install -e ".[benchmark]"` pulls in
  `datasets`, `langchain`, `langchain-openai`, `langchain-community`,
  `langchain-text-splitters`, `matplotlib`, `tiktoken`, `tqdm`. Folded
  into `[all]`.

### Changed

- **BREAKING — self-reported confidence renamed to `extraction_confidence`.**
  The JSON property the extractor injects for `BaseStructure` and each
  `BaseCollection` entity is now `extraction_confidence` (was
  `_confidence`); the semantic trailing tag is now
  `<extraction_confidence>0.xx</extraction_confidence>` (was
  `<confidence>`). Leading-underscore names read as "internal" to many
  models, which encouraged dropping the field precisely when the value
  would be low. `extend()` callers that read `self._confidence` must
  migrate to `self.confidence` — no compatibility shim is provided.
- **`ennoia try` output no longer repeats the scalar per field.**
  Structural and semantic extractions now print confidence once on the
  schema header line (`Schema: Doc  (confidence: 0.90)`); per-entity
  confidence for collections continues to appear on each row. The
  previous output implied per-field confidence that Ennoia does not
  produce.
- **BREAKING — MCP toolset simplified.** The `filter` MCP tool is
  removed; the `search` tool now takes `filter=` (was: separate `filter`
  tool + `search(filter_ids=…)`) and runs the two-phase plan internally.
  New signature: `search(query, filter=None, limit=10, index=None)` —
  note `top_k` is renamed to `limit`. Rationale: LLM agents consistently
  struggled to chain the two near-identical tools. The SDK
  `Pipeline.afilter` method and the REST `POST /filter` endpoint are
  unaffected.
- **BREAKING — `HybridStore.upsert` signature.** Replaces
  `vectors: dict[str, list[float]]` with `entries: list[VectorEntry]`,
  where `VectorEntry` captures `(index_name, vector, text, unique)`. The
  storage model flips from one-row-per-document-with-many-named-vectors
  to **one row per vector entry, with the structural fields denormalized
  across the document's rows**. This unifies how `BaseSemantic` and
  `BaseCollection` land in a hybrid backend: each row carries a single
  vector + its index name + its text, and a single native query can
  filter on structural fields and rank by vector similarity at once. The
  pipeline collapses multi-row hits to one per `source_id` at search
  time. Affected backends: `PgVectorHybridStore` (table schema change:
  composite rows keyed by `vector_id`), `QdrantHybridStore` (one point
  per entry with a single unnamed vector), `MockStore` (row-per-entry
  in-memory model). Third-party `HybridStore` subclasses must be updated.
- **`semantic_indices` discovery payload** now includes a `kind` field
  (`"semantic"` or `"collection"`) per entry so agents can distinguish
  one-answer-per-doc indices from many-entries-per-doc indices.
- **`parse_semantic_vector_id`** now returns a 3-tuple
  `(source_id, index_name, unique | None)` to parse the new 3-part form
  emitted for collection entries.

## [0.3.0] — 2026-04-16

Production interfaces (REST, MCP), full hybrid-store adapter set
(Qdrant, pgvector), and a public testing utility package. Docs ship as
a mkdocs-material site. Governance scaffolding lands for the first time
(`SECURITY.md`, `CODE_OF_CONDUCT.md`, `CODEOWNERS`, issue + PR templates,
dependabot).

Integration tests against live services (Qdrant, Ollama, pgvector,
OpenAI, Anthropic) and performance benchmarks are **deferred** to a
future release — the Stage 3 PR is already large. Stage 3 unit coverage
stays at 100% line + branch via in-memory fakes and scripted mocks.

### Added

- **`ennoia api` CLI subcommand** — boots the FastAPI REST server
  against a filesystem store. Endpoints: `GET /discover`, `POST /filter`,
  `POST /search`, `GET /retrieve/{source_id}`, `POST /index`,
  `DELETE /delete/{source_id}`. Filter-validation failures surface as
  HTTP 422 with the canonical `docs/filters.md` error shape.
- **`ennoia mcp` CLI subcommand** — boots the FastMCP read-only server.
  Transports: `sse | stdio | http` per `.ref/USAGE.md §5`. Exposes
  `discover_schema`, `filter`, `search`, `retrieve` tools.
- **`ennoia.server` package** — `create_app(ctx)` (FastAPI) and
  `create_mcp(ctx)` (FastMCP). Shared `ServerContext(pipeline, auth)`.
  Pluggable `AuthHook` protocol with `static_bearer_auth`,
  `env_bearer_auth` (reads `ENNOIA_API_KEY`), and `no_auth` builders.
- **`ennoia.testing` package** — `MockLLMAdapter` (scripted/callable/list
  responses), `MockEmbeddingAdapter` (deterministic hash-seeded unit
  vectors), `MockStore` (in-memory `HybridStore` with cosine similarity).
  Exposed to downstream pytest suites via a `pytest11` entry point:
  `mock_store`, `mock_llm`, `mock_embedding` fixtures are auto-discovered.
- **`ennoia.store.vector.qdrant.QdrantVectorStore`** — pure vector
  backend with async `qdrant-client`. Stable UUIDv5 point ids;
  `restrict_to` and `index` filters honour the Stage 3 Pipeline
  contract.
- **`ennoia.store.hybrid.qdrant.QdrantHybridStore`** — single-collection
  hybrid backend using Qdrant's named vectors (one slot per semantic
  index). Filter translator covers `eq`, `in`, range, `is_null`,
  `contains_any`, and list-contains natively; `startswith`,
  string-`contains`, and `contains_all` post-filter candidate payloads
  via the canonical `apply_filters`.
- **`ennoia.store.hybrid.pgvector.PgVectorHybridStore`** — PostgreSQL +
  pgvector hybrid backend. Single jsonb column + nullable per-index
  `vector(N)` columns materialised on first use. Full 11-operator SQL
  translator (`_sql_filter.build_where`) — no Python residual.
- **Multi-collection support end-to-end.** Every store accepts a
  `collection: str = "documents"` kwarg (SQLite table name, Parquet /
  NumPy file basename, Qdrant collection, pgvector table). `Store.from_path`
  gains `collection=` and places the filesystem layout under
  `<path>/<collection>/`, so multiple pipelines can share one project
  root without colliding. Collection names validate against
  `^[A-Za-z_][A-Za-z0-9_]*$` via the new
  `ennoia.store.base.validate_collection_name` helper.
- **CLI reaches the hybrid stores.** `ennoia index`, `ennoia search`,
  `ennoia api`, and `ennoia mcp` gain a prefix-qualified `--store`:
  plain path / `file:<path>` → filesystem; `qdrant:<collection>` (with
  `--qdrant-url` / `--qdrant-api-key`, env-var backed) →
  `QdrantHybridStore`; `pgvector:<collection>` (with `--pg-dsn`) →
  `PgVectorHybridStore`. A new `parse_store_spec` in
  `ennoia.cli.factories` mirrors the existing `parse_llm_spec` /
  `parse_embedding_spec` pattern. A `--collection` flag configures the
  filesystem backend (defaults to `documents`).
- **Pipeline public API** gains `afilter`, `aretrieve`, `adelete`, plus
  sync mirrors (`filter`, `retrieve`, `delete`). `asearch` accepts a
  new `filter_ids=` kw-only parameter for the MCP two-phase flow
  (`filter → search(filter_ids=...)`), and a new `index=` parameter to
  restrict vector search to a single semantic index. `filters=` and
  `filter_ids=` are mutually exclusive.
- **Store ABC additions:** `StructuredStore.delete`,
  `VectorStore.delete_by_source`, `HybridStore.delete`, and
  `HybridStore.filter`. Declared as concrete methods raising
  `NotImplementedError` (not `@abstractmethod`) so Stage-2 subclasses
  that predate the contract remain instantiable.
- **`HybridStore` persistence in `Pipeline._apersist`** — the
  `NotImplementedError` from earlier stages is replaced with a full
  single-roundtrip upsert path (vectors dict keyed by semantic index
  name).
- **Pyproject extras:** `qdrant`, `pgvector`, `server`, `docs`. The
  `all` extra expands to include them. New pytest11 entry point:
  `ennoia = "ennoia.testing.fixtures"`.
- **Governance:** `SECURITY.md`, `CODE_OF_CONDUCT.md`, `CODEOWNERS`,
  `.github/ISSUE_TEMPLATE/` (bug, feature, config), PR template,
  dependabot.yml.
- **Docs site:** `mkdocs.yml` + `mkdocstrings-python` API reference +
  `mike` versioning. New pages: `serve.md` (REST + MCP), `testing.md`
  (ennoia.testing), `api-reference.md`, cookbook entries (`mcp-agent`,
  `custom-adapter`).

### Changed

- **Breaking:** `PgVectorHybridStore(table_prefix=...)` → `collection=...`.
  The parameter rename unifies naming with every other store; the
  generated table name is now the collection itself (no `_docs` suffix).
  Existing deployments must rename their table or recreate the store
  under the new name.
- **Breaking:** `ennoia try --embedding` is removed. `try` is a
  single-document schema/LLM debug tool and never touched embeddings;
  the flag was documented "loaded but unused" and is gone.
- `VectorStore.search` and `HybridStore.hybrid_search` gain a kw-only
  `index: str | None = None` parameter for semantic-index scoping.
  Existing built-in backends honour it; third-party subclasses will
  need to accept and propagate it.
- `_ensure_collection` on `QdrantHybridStore` derives vector dimensions
  from the first upsert and remembers the named-vector spec for the
  lifetime of the instance; subsequent calls skip the roundtrip.
- Spec docs (`.ref/USAGE.md §5`, `.ref/IMPLEMENTATION.md §Stage 3`)
  updated to reflect the `ennoia api` / `ennoia mcp` command split
  (the original spec called for a single `ennoia serve` — revised
  during Stage 3 planning for clearer separation of REST vs MCP).

### Migration

- Third-party `VectorStore` / `HybridStore` subclasses must add the
  kw-only `index: str | None = None` parameter to `search` /
  `hybrid_search`. The parameter is optional and safe to ignore on
  backends that don't model per-index scoping.
- Third-party stores that want delete support should override the new
  `delete` / `delete_by_source` methods; the defaults raise
  `NotImplementedError("override me")`.
- `PgVectorHybridStore(table_prefix="foo")` callers: pass `collection="foo"`
  instead. The generated table name changes from `foo_docs` to `foo`.
- `Store.from_path(path)` now builds under `<path>/documents/` by
  default (the `documents` segment is the new collection subdirectory).
  Existing on-disk indices built with v0.2.x will not be found at the
  old layout — either move the files into a collection subdirectory or
  re-run `ennoia index` pointed at the new collection.
- `ennoia try` no longer accepts `--embedding`. Drop the flag from any
  existing shell scripts.

## [0.2.1] — 2026-04-16

All I/O is async: embedding adapters, store backends, and the query
planner. The pipeline parallelises embedding work per document and
exposes a single concurrency knob for resource-constrained runs.

### Added

- **Async embedding contract:** `EmbeddingAdapter.embed`,
  `embed_document`, and `embed_query` are now `async`. New
  `embed_batch(texts)` on the ABC — default fans out to `embed` via
  `asyncio.gather`; `OpenAIEmbedding` overrides it to issue one
  `embeddings.create(input=[...])` round-trip, and
  `SentenceTransformerEmbedding` forwards the full list to a single
  `encode` call. The pipeline calls `embed_batch` whenever a document
  has multiple semantic fields.
- **Async store contract:** every method on `StructuredStore`,
  `VectorStore`, and `HybridStore` is `async`.
  `SQLiteStructuredStore` switched from stdlib `sqlite3` to
  `aiosqlite`, with a per-loop connection (rebinds when the running
  loop differs from the cached one — same loop-lifecycle invariant as
  the LLM adapters). `ParquetStructuredStore` and
  `FilesystemVectorStore` dispatch their blocking pandas / numpy
  calls via `asyncio.to_thread`.
- **`Pipeline(concurrency: int | None = None)`** — caps simultaneous
  LLM extractions and embedding calls. `None` means unbounded
  `asyncio.gather`; `1` serialises every call. The cap is shared
  across the executor's structural and semantic phases and the
  per-document embedding gather in `_apersist`.
- **`ennoia index --no-threads` / `ennoia search --no-threads`** —
  passes `concurrency=1` for local-Ollama users on
  resource-constrained machines.
- **Regression tests:** `tests/test_parallel_embedding.py` proves
  multi-semantic documents go through one batched embed call;
  `tests/test_pipeline_concurrency.py` pins the `--no-threads` semantics
  end-to-end; `tests/test_store_sqlite.py::test_survives_multiple_event_loops`
  guards the aiosqlite per-loop rebind.

### Changed

- `aiosqlite>=0.20` is now a base dependency (SQLite is the default
  structured backend).
- `Pipeline._persist` was renamed to `async _apersist` and parallelises
  per-document embedding via `embed_batch`; vector upserts stay
  sequential because concrete vector stores write the same backing
  file/connection per call.
- `plan_search` is now `async`; callers must `await` it.

### Migration

- Custom `EmbeddingAdapter` subclasses must change `def embed` →
  `async def embed`. Override `embed_batch` only if the backend has a
  native list-input API.
- Custom `StructuredStore` / `VectorStore` / `HybridStore` subclasses
  must change every method to `async def`. For backends without an
  async client, wrap blocking calls in `await asyncio.to_thread(...)`.
- Code that called `store.upsert(...)` / `store.get(...)` /
  `store.filter(...)` / `store.search(...)` outside the pipeline must
  now `await` those calls.

## [0.2.0] — 2026-04-15

Emission manifest & superschema: the framework now knows every possible
field at pipeline init.

### Added

- **`Schema` inner class** on `BaseStructure` with two optional
  attributes: `namespace: str | None` (default `None` — flat merge) and
  `extensions: list[type]` (default `[]` — the complete set of classes
  `extend()` may return). Bare class, no base to extend. `BaseSemantic`
  has no `Schema` — semantic schemas are terminal text emitters.
- **Emission manifest** (`ennoia.schema.manifest`) — `build_manifest`
  resolves the transitive closure of `Schema.extensions` with cycle
  detection, BFS ordering, and namespace/field-name validation against
  reserved filter operators.
- **Superschema** (`ennoia.schema.merging`) — `build_superschema`
  collapses the manifest into a unified `{name: SuperField}` map. Flat
  merging by default; `Schema.namespace` prefixes with `{ns}__`.
  Multi-source fields merge per explicit type-compatibility rules
  (identity, `Literal` union, `Optional` absorption, `list[T]` recursion).
- **Runtime validation** in the executor: `extend()` returning a class
  not declared in `Schema.extensions` raises `SchemaError` before any
  child LLM call.
- **`SchemaError`** (hard failures — cycles, incompatible types,
  reserved names, undeclared emissions) and **`SchemaWarning`** (soft
  — divergent descriptions) added to `ennoia.index.exceptions`.

### Changed

- `Pipeline.__init__` now computes the manifest + superschema once and
  caches both on the instance. Filter validation reads the superschema
  as its single source of truth.
- `Pipeline._persist` null-fills the structured-store record with every
  superschema field so inactive branches still produce a uniform column
  set. Fields from namespaced schemas are written under `{ns}__{field}`.
- `ennoia.describe(schemas)` now returns the superschema discovery
  payload. New keys per structural field: `sources` (list of
  contributing class names) and `has_divergent_descriptions` (present
  only when multi-source). Matches `docs/filters.md §Schema Discovery`.
- `ennoia.schema.operators.type_label` and `.unwrap_optional` are now
  public (previously `_`-prefixed) — reused by the merging module.
- `ennoia.index.validation.validate_filters` takes a `Superschema`
  instead of a list of schemas; `build_filter_contract` removed (the
  superschema is the contract).

### Migration

Any schema that returns child classes from `extend()` must now declare
them in `Schema.extensions`; undeclared returns raise `SchemaError` at
index time.

```python
class Parent(BaseStructure):
    """..."""
    class Schema:
        extensions = [Child]

    def extend(self):
        return [Child]
```

## [0.1.0] — 2026-04-15

First public release. Stage 2 (Working Concept) in
`.ref/IMPLEMENTATION.md` — pip-installable, CLI, strict CI.

### Added

- **Schema layer:** operator inference per field type, `ennoia.Field`
  overrides (`operators=[...]`, `filterable=False`), `describe_schema()`
  class method, and top-level `ennoia.describe(schemas)` producing the
  canonical filter contract JSON.
- **Index layer:** layer-wise parallel executor
  (`ennoia.index.executor`), `extend()` parent-context injection into
  child prompts, self-reported `_confidence` dynamically appended to the
  JSON Schema last and surfaced on `IndexResult.confidences`,
  `FilterValidationError` with the `docs/filters.md` error shape.
- **Stores:** `SQLiteStructuredStore`, `ParquetStructuredStore`,
  `FilesystemVectorStore`, and `Store.from_path` for a filesystem-backed
  composite store matching the CLI default.
- **Adapters:** `OpenAIAdapter`, `AnthropicAdapter`, `OpenAIEmbedding`,
  with environment-variable API-key fallback.
- **Events:** typed event dataclasses (`ExtractionEvent`, `IndexEvent`,
  `SearchEvent`) + synchronous `Emitter` pub/sub.
- **CLI:** `ennoia try | index | search`, adapter URI syntax
  (`ollama:qwen3:0.6b` etc.), filter validation wired to the shared
  error shape.
- **Tooling:** strict CI (ruff check + ruff format + pyright strict +
  pytest) across Python 3.11 / 3.12 / 3.13; PyPI trusted-publisher
  release workflow; pre-commit config.
- **Docs:** `docs/` directory with concepts, quickstart, schema and
  filter guides, CLI reference, adapter + store notes.

### Changed

- `BaseStructure.model_config = ConfigDict(extra="allow")` so the
  extractor can carry `_confidence` on the instance without declaring
  it as a field.
- Pipeline orchestration moved from the serial Stage 1 loop in
  `pipeline.py` to a layer-wise parallel executor
  (`index/executor.py`).
- `KNOWN_OPERATORS` extended to the full set: `contains`, `startswith`,
  `contains_all`, `contains_any`, `is_null`.
