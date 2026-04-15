# Changelog

All notable changes to Ennoia are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
from v0.1.0 onward.

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
