# Changelog

All notable changes to Ennoia are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
from v0.1.0 onward.

## [0.2.1] — Unreleased

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

## [0.2.0] — Unreleased

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

## [0.1.0] — Unreleased

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
