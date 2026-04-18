# Ennoia — runnable examples

Every concept in the docs has a matching standalone script here. Each script imports only the public API, runs end-to-end, and prints something you can verify.

## Prerequisites

- **Python 3.11+**
- An LLM backend — pick one:
  - **Local (default):** [Ollama](https://ollama.com/) running locally with the `qwen3:0.6b` model pulled.
    ```bash
    ollama pull qwen3:0.6b
    ```
  - **Hosted:** set `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY`) and the scripts will switch automatically.
- Install Ennoia with the extras the script needs:
  ```bash
  pip install "ennoia[ollama,sentence-transformers]"
  ```

## Running

Every script is invoked the same way:

```bash
python examples/01_getting_started.py
```

All scripts share the same fabricated Master Services Agreement (ACME Corp ↔ Globex Ltd, effective 2025-06-01, Delaware law), defined once in [`_data.py`](_data.py) and imported where needed. If you open two recipes in parallel, the parties, dates, and clauses match.

## Environment variables

Scripts read configuration from the environment so you can run the same file against Ollama locally or OpenAI/Anthropic in CI.

| Variable | What it controls | Default |
|---|---|---|
| `OPENAI_API_KEY` | If set, scripts use `OpenAIAdapter` + `OpenAIEmbedding` instead of Ollama + sentence-transformers. | unset |
| `ENNOIA_OPENAI_MODEL` | OpenAI model name when `OPENAI_API_KEY` is set. | `gpt-4o-mini` |
| `ENNOIA_OPENAI_EMBEDDING` | OpenAI embedding model when `OPENAI_API_KEY` is set. | `text-embedding-3-small` |
| `ENNOIA_OLLAMA_MODEL` | Ollama model name for local runs. | `qwen3:0.6b` |
| `ENNOIA_ST_MODEL` | sentence-transformers model name for local runs. | `all-MiniLM-L6-v2` |
| `QDRANT_URL` | Required by `08_qdrant_hybrid.py`; the script skips cleanly without it. | unset |
| `PGVECTOR_DSN` | Required by `09_pgvector_hybrid.py`; the script skips cleanly without it. | unset |

## Recipes

| # | Script | What it shows |
|---|---|---|
| 01 | [`01_getting_started.py`](01_getting_started.py) | Minimal end-to-end: one structural and one semantic extractor, in-memory stores, index + search. |
| 02 | [`02_structural_extractor.py`](02_structural_extractor.py) | `BaseStructure` deep dive: `Annotated[..., Field(description=...)]`, operator overrides, `filterable=False`, confidence. |
| 03 | [`03_semantic_extractor.py`](03_semantic_extractor.py) | Multiple `BaseSemantic` classes on the same document; docstring-as-question. |
| 04 | [`04_extend_branching.py`](04_extend_branching.py) | All three extractor kinds on one pipeline plus `extend()` branching the DAG on a structural field; `Schema.extensions` declaration. |

*Recipes 05 – 14 land in the next pass of the docs rewrite; run `python examples/04_extend_branching.py` to see every extractor kind in one run.*

## Troubleshooting

- `ConnectionError` hitting `http://localhost:11434`: Ollama isn't running. Start it with `ollama serve` (or set `OPENAI_API_KEY`).
- `ExtractionError: Failed to extract ... after retry`: the LLM returned JSON that didn't match the schema twice. Lower the temperature or switch to a stronger model via the env vars above.
- First run is slow: sentence-transformers downloads the model (~90 MB) the first time. Subsequent runs are fast.
