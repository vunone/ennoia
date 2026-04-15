# Examples

Three runnable scripts that double as Stage 1 smoke tests.

## Prerequisites

1. Install Ennoia with local extras:
   ```bash
   pip install -e ".[ollama,sentence-transformers]"
   ```
2. Install and start [Ollama](https://ollama.com), then pull a small model:
   ```bash
   ollama pull qwen3:0.6b
   ```
   Any Ollama-hosted model with JSON-format support works; `qwen3:0.6b`
   is small enough to run on a laptop CPU.

## Scripts

- `hello_world.py` — the canonical Stage 1 demo from `.ref/USAGE.md §1`.
  Indexes one document with a structural + semantic schema and runs a
  filtered search.
- `multi_schema.py` — two structural and two semantic schemas. Proves
  the DAG walks multiple nodes in one indexing pass.
- `filter_miss.py` — shows that a mismatched structured filter
  eliminates candidates before vector search even runs.

## Run

```bash
python examples/hello_world.py
python examples/multi_schema.py
python examples/filter_miss.py
```
