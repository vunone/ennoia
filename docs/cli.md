# CLI

Install the `cli` extra to get the `ennoia` console script:

```bash
pip install "ennoia[cli,ollama,sentence-transformers]"
```

## Configuration via `ennoia.ini`

Every CLI command can read its defaults from an `ennoia.ini` file in the
current directory, so you don't have to repeat `--llm`, `--embedding`,
`--store`, API keys, and so on on every invocation. The preferred flow is
to run `ennoia init` once, edit the generated file, and then call
subcommands with just their positional arguments.

```bash
ennoia init                 # writes ./ennoia.ini
# edit ennoia.ini
ennoia index ./docs          # reads store, schema, llm, embedding from the INI
ennoia search "query text"   # same
```

The INI has two sections:

- `[ennoia]` — flag defaults. Each key maps 1:1 to a CLI option.
- `[env]` — raw environment variables (typically provider API keys).
  Values are exported before adapter clients initialize, so SDKs like
  `openai` / `anthropic` pick them up automatically.

All writes go through `os.environ.setdefault`, which gives this
precedence order:

    explicit --flag  >  shell env  >  INI [ennoia] / [env]  >  hardcoded default

An explicit CLI flag always wins, and a shell-exported variable (e.g.
`export OPENAI_API_KEY=…` in your profile) always wins over the file.

### Template

```ini
[ennoia]
llm = ollama:qwen3:0.6b
embedding = sentence-transformers:all-MiniLM-L6-v2

store = ./ennoia_index
collection = documents

# schema = ./schemas.py

# qdrant_url = http://localhost:6333
# qdrant_api_key =
# pg_dsn = postgresql://user:pass@localhost:5432/ennoia

# host = 127.0.0.1
# port = 8080
# transport = sse
# api_key =

[env]
# OPENAI_API_KEY =
# ANTHROPIC_API_KEY =
# OPENROUTER_API_KEY =
```

### Key map

| `[ennoia]` key | Env var | CLI flag(s) |
|---|---|---|
| `llm` | `ENNOIA_LLM` | `--llm` on `try`, `index`, `search`, `craft`, `api`, `mcp` |
| `embedding` | `ENNOIA_EMBEDDING` | `--embedding` on `index`, `search`, `api`, `mcp` |
| `store` | `ENNOIA_STORE` | `--store` on `index`, `search`, `api`, `mcp` |
| `schema` | `ENNOIA_SCHEMA` | `--schema` on `try`, `index`, `search`, `api`, `mcp` |
| `collection` | `ENNOIA_COLLECTION` | `--collection` on `index`, `search`, `api`, `mcp` |
| `qdrant_url` | `ENNOIA_QDRANT_URL` | `--qdrant-url` |
| `qdrant_api_key` | `ENNOIA_QDRANT_API_KEY` | `--qdrant-api-key` |
| `pg_dsn` | `ENNOIA_PG_DSN` | `--pg-dsn` |
| `host` | `ENNOIA_HOST` | `--host` on `api`, `mcp` |
| `port` | `ENNOIA_PORT` | `--port` on `api`, `mcp` |
| `transport` | `ENNOIA_TRANSPORT` | `--transport` on `mcp` |
| `api_key` | `ENNOIA_API_KEY` | `--api-key` on `api`, `mcp` |

Unknown keys in `[ennoia]` are rejected so typos surface immediately.
Paths (e.g. `schema`, `store`) are resolved relative to the process's
working directory, not the INI file's directory.

### Overriding the config path

```bash
ennoia --config ~/configs/prod.ini index ./docs
ennoia --no-config index ./docs --schema … --store … --llm …
```

`--config` points at an alternative INI; `--no-config` skips loading
anything and forces all values to come from flags or shell env.

## `ennoia init` — bootstrap a config file

```bash
ennoia init                         # writes ./ennoia.ini
ennoia init --path ~/ennoia.ini     # pick a different destination
ennoia init --force                 # overwrite an existing file
```

The command refuses to overwrite an existing file unless `--force` is
passed, so an accidental re-run never clobbers your edits.

## `ennoia craft` — scaffold a schema from a document

!!! warning "Prototype only"
    `craft` produces a **first draft**, not production-ready schemas.
    It gives you a minimal three-class skeleton — `Metadata`,
    `QuestionAnswer`, `Summary` — with the `Metadata` field set picked
    from one document sample. The LLM deliberately avoids `Literal`
    enums, `Optional`, `list`, and conditional `extend()` branches
    because it cannot enumerate the full value space from one example.
    **Read every field and docstring before running `ennoia index` on
    real data**: tighten types to `Literal` once you know the enum set,
    add `list[T]` where fields are plural, and drop or rename fields
    that turned out irrelevant to your retrieval task.

Point an LLM at a sample document and a retrieval task; Ennoia writes a
ready-to-edit schema module to ``--output``. If the output file already
exists, its current contents are passed to the LLM so it *improves* the
schema instead of rewriting it from scratch.

```bash
ennoia craft ./product_docs/product_1.txt \
  --output schema.py \
  --llm openai:gpt-4o-mini \
  --task "filter products by price, category, and colour"
```

| Flag | Required | Default | Meaning |
|---|---|---|---|
| `--output` | ✅ | — | Path to the `.py` file to write (or improve in place) |
| `--llm` | ✅ | — | LLM adapter URI, e.g. `openai:gpt-4o-mini` |
| `--task` | ✅ | — | Natural-language retrieval goal the schema must serve |
| `--max-retries` | | `2` | Budget for validation retries after the first attempt |

After each LLM call the output is validated by importing it. If the
import fails, the traceback is fed back to the LLM on the next attempt.
The command exits non-zero only after exhausting the retry budget;
the partially-written file is left on disk for inspection.

The command echoes the prototype warning both on start and after writing
the file, so manual review is impossible to miss.

Documents are sent verbatim — Ennoia does not chunk them — so the chosen
model needs a context window large enough to hold the sample; if the
provider returns a context-length error, it surfaces with an actionable
hint.

### What the draft looks like

`craft` always produces the same three-class shape. Only the `Metadata`
field set and the three docstrings are tailored to your sample:

```python
from datetime import date
from typing import Annotated
from ennoia import BaseCollection, BaseSemantic, BaseStructure, Field


class Metadata(BaseStructure):
    """<one-line prompt tailored to your doc type and retrieval task>"""

    # 5–10 plain filterable fields: str, int, float, bool, date only.
    ...


class QuestionAnswer(BaseCollection):
    """Generate ten question-and-answer pairs covering the key facts..."""

    question: Annotated[str, Field(description="...")]
    answer: Annotated[str, Field(description="...")]

    class Schema:
        max_iterations = 1

    def get_unique(self) -> str:
        return self.question.casefold()

    def template(self) -> str:
        return f"{self.question}\n{self.answer}"


class Summary(BaseSemantic):
    """Summarize in one or two sentences what this document is about."""
```

### Review checklist

Before handing the draft to `ennoia index`, walk through the file and
confirm each item:

- **Every `Metadata` field must be corpus-generic.** Ask yourself, for
  each field: *"would ten other documents of the same kind carry this
  field too, with different values?"* Drop fields that only apply to
  the sample — `default_port` on an API page, `mentions_arbitration`
  on a contract. Document-specific facts belong in `QuestionAnswer` /
  `Summary`, not `Metadata`. Fewer generic fields beats more
  document-specific ones.
- **Boolean flags are yes/no questions every document can answer.**
  Good: `in_stock`, `auto_renews`, `is_opinion`. Bad: `has_payment_terms`,
  `covers_ukraine` — topic predicates collapse to `false` across the
  corpus and don't filter anything useful.
- **Tighten types where you can.** `craft` emits plain `str` everywhere
  a categorical filter might belong; replace those with `Literal[...]`
  once you know the value set, and widen single scalars to `list[T]`
  where the document actually carries multiples.
- **Docstrings are corpus-generic, not sample-specific.** Every
  docstring runs as a prompt against *every* document you index. Strip
  out the sample's subject matter — *"about the REST and MCP
  interfaces"*, *"this arbitration clause"*, *"this RGB keyboard"*.
  Name the corpus type (*"documentation page"*, *"contract"*, *"product
  page"*) and what to extract, never the sample's specific topic.
  Read each docstring with a *different* document in the same corpus in
  mind — if it no longer makes sense, tighten it.
- **Metadata docstring** reflects the retrieval task at the corpus
  level — it is the LLM's extraction prompt at index time.
- **QuestionAnswer docstring** targets the corpus, not the sample. The
  `ten pairs` count is a sensible default; adjust up or down by editing
  the docstring, not by changing `max_iterations`.
- **Summary docstring** asks the right question for your corpus
  (e.g. *"What contract is this and who are the parties?"* beats the
  generic default on legal documents — but keep it corpus-wide, not
  tied to the sample's topic).

## `ennoia try` — iterate on schemas

Run a single extraction pass against one document and print the fields,
confidences, and `extend()` chain. Nothing is written to a store.

Confidence is a **per-extraction** scalar for `BaseStructure` /
`BaseSemantic` (shown once on the schema header), and **per-entity** for
`BaseCollection` (shown next to each row). Ennoia does not self-report
per-field confidence.

```bash
ennoia try ./sample.txt --schema my_schemas.py
```

Output is colorised when stdout is a TTY (plain when piped). Each
extractor prints as a ``Extractor[Kind]: Name`` header, followed by
indented key/value rows. Collections show one entity per ``-`` block.

```
Extractor[BaseStructure]: CaseDocument  (confidence: 0.90)
  jurisdiction: 'WA'
  court_level: 'appellate'
  date_decided: '2023-03-15'
  is_overruled: False
  → extend(): WashingtonAppellateSchema

Extractor[BaseCollection]: Citation
  - (confidence: 0.88)
      case: 'Smith v. Jones'
      year: 1998
  - (confidence: 0.76)
      case: 'Doe v. Roe'
      year: 2004

Extractor[BaseSemantic]: Holding  (confidence: 0.91)
  'The court reversed the lower court's finding, holding that...'
```

## `ennoia index` — index a folder

Walks a directory and indexes each file through the pipeline. The
`--store` flag accepts a prefix-qualified spec — filesystem path by
default, `qdrant:<collection>` or `pgvector:<collection>` when pointing
at a production vector backend. See [Store backends](#store-backends) below.

```bash
ennoia index ./docs \
  --schema my_schemas.py \
  --store ./my_index \
  --collection invoices \
  --llm ollama:qwen3:0.6b \
  --embedding sentence-transformers:all-MiniLM-L6-v2
```

Multiple pipelines can sit under one project root by passing different
`--collection` values — each gets its own `<root>/<collection>/` subtree.

By default, independent schemas in a layer extract in parallel via
`asyncio.gather`. On a resource-constrained machine running local
Ollama, pass `--no-threads` to serialise LLM and embedding calls
(equivalent to `Pipeline(..., concurrency=1)`):

```bash
ennoia index ./docs --schema my_schemas.py --store ./my_index --no-threads
```

## `ennoia search` — hybrid search

```bash
ennoia search "employer duty to accommodate disability" \
  --schema my_schemas.py \
  --store ./my_index \
  --filter "jurisdiction=WA" \
  --filter "date_decided__gt=2020-01-01" \
  --top-k 5
```

Filters go through the same validator the SDK uses. An unsupported
operator exits with code `2` and an error JSON:

```json
{
  "error": "invalid_filter",
  "field": "jurisdiction",
  "operator": "gt",
  "message": "Field 'jurisdiction' (type: enum) does not support operator 'gt'. Supported operators: eq, in."
}
```

## `ennoia api` — REST server

Boot a FastAPI server against any supported store. Full CRUD surface —
`/discover`, `/filter`, `/search`, `/retrieve`, `/index`, `/delete`.

```bash
ennoia api --store qdrant:cases \
  --qdrant-url http://localhost:6333 \
  --schema my_schemas.py \
  --api-key "$ENNOIA_API_KEY"
```

Requires the `server` extra. See [Servers — REST + MCP](serve.md) for
every flag, the endpoint table, and auth configuration.

## `ennoia mcp` — MCP tool server

Read-only agent surface — `discover_schema`, `filter`, `search`,
`retrieve`. Supports `sse`, `stdio`, and `http` transports.

```bash
ennoia mcp --store qdrant:cases \
  --qdrant-url http://localhost:6333 \
  --schema my_schemas.py \
  --transport sse \
  --api-key "$ENNOIA_API_KEY"
```

Requires the `server` extra. See [Servers — REST + MCP](serve.md) and
the [MCP agent cookbook](cookbook/mcp-agent.md).

## Store backends

`--store` accepts three forms across `index`, `search`, `api`, and `mcp`:

| Form | Backend | Required extra flags |
|---|---|---|
| `<path>` or `file:<path>` | Filesystem (Parquet + NumPy) | `--collection` (default `documents`) |
| `qdrant:<collection>` | `QdrantHybridStore` | `--qdrant-url` (or `ENNOIA_QDRANT_URL`); `--qdrant-api-key` / `ENNOIA_QDRANT_API_KEY` optional |
| `pgvector:<collection>` | `PgVectorHybridStore` | `--pg-dsn` (or `ENNOIA_PG_DSN`) |

Collection names must match `^[A-Za-z_][A-Za-z0-9_]*$` so the same name
is valid as a subdirectory, SQLite table, Qdrant collection, and
PostgreSQL table.

## Adapter URIs

Every `--llm` / `--embedding` argument is a URI of the form
`prefix:model`. Current prefixes:

| Flag | Prefix | Example |
|---|---|---|
| `--llm` | `ollama` | `ollama:qwen3:0.6b` |
| `--llm` | `openai` | `openai:gpt-4o-mini` |
| `--llm` | `anthropic` | `anthropic:claude-sonnet-4-20250514` |
| `--llm` | `openrouter` | `openrouter:meta-llama/llama-3.1-8b-instruct` |
| `--embedding` | `sentence-transformers` | `sentence-transformers:all-MiniLM-L6-v2` |
| `--embedding` | `openai-embedding` | `openai-embedding:text-embedding-3-small` |
| `--embedding` | `openrouter-embedding` | `openrouter-embedding:openai/text-embedding-3-small` |

API keys for OpenAI / Anthropic are read from `OPENAI_API_KEY` /
`ANTHROPIC_API_KEY` if not passed explicitly. OpenRouter (LLM and
embeddings) reads `OPENROUTER_API_KEY`.
