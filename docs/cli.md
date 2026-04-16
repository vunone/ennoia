# CLI

Install the `cli` extra to get the `ennoia` console script:

```bash
pip install "ennoia[cli,ollama,sentence-transformers]"
```

## `ennoia try` — iterate on schemas

Run a single extraction pass against one document and print the fields,
confidences, and `extend()` chain. Nothing is written to a store.

```bash
ennoia try ./sample.txt --schema my_schemas.py
```

```
Schema: CaseDocument
  jurisdiction: 'WA'        (confidence: 0.95)
  court_level:  'appellate' (confidence: 0.87)
  date_decided: '2023-03-15'(confidence: 0.92)
  is_overruled: False       (confidence: 0.78)
  -> extend(): WashingtonAppellateSchema

Schema: Holding
  'The court reversed the lower court's finding, holding that...'  (confidence: 0.91)
```

## `ennoia index` — index a folder

Walks a directory and indexes each file through the pipeline. The
`--store` flag builds a filesystem-backed store under the given path
(Parquet for structured, NumPy for vectors).

```bash
ennoia index ./docs \
  --schema my_schemas.py \
  --store ./my_index \
  --llm ollama:qwen3:0.6b \
  --embedding sentence-transformers:all-MiniLM-L6-v2
```

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

## Adapter URIs

Every `--llm` / `--embedding` argument is a URI of the form
`prefix:model`. Current prefixes:

| Flag | Prefix | Example |
|---|---|---|
| `--llm` | `ollama` | `ollama:qwen3:0.6b` |
| `--llm` | `openai` | `openai:gpt-4o-mini` |
| `--llm` | `anthropic` | `anthropic:claude-sonnet-4-20250514` |
| `--embedding` | `sentence-transformers` | `sentence-transformers:all-MiniLM-L6-v2` |
| `--embedding` | `openai-embedding` | `openai-embedding:text-embedding-3-small` |

API keys for OpenAI / Anthropic are read from `OPENAI_API_KEY` /
`ANTHROPIC_API_KEY` if not passed explicitly.
