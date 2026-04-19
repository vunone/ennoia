# Adapters

Adapters are stateless transformers that sit between the pipeline and an
external service (LLM inference, embeddings). Every adapter inherits from
one of two abstract base classes:

- `LLMAdapter` — `async complete_json(prompt)` and `async complete_text(prompt)`.
- `EmbeddingAdapter` — `async embed(text)`; the ABC provides concrete
  `embed_document(text)` / `embed_query(text)` (both delegate to `embed`)
  and `embed_batch(texts)` (parallel-gather fallback that backends with a
  native list-input API can override for a single round-trip).

ABCs live in `ennoia/adapters/{llm,embedding}/base.py`. Inheritance (not
structural typing) is used so that partial implementations fail loudly with
`TypeError` at instantiation, matching the contract convention in
[stores.md](stores.md).

All adapter I/O is async. Sync libraries (sentence-transformers,
transformers' `encode`) are dispatched via `asyncio.to_thread` so the event
loop stays responsive while CPU-bound model work runs.

## Built-in LLM adapters

| Adapter | Extra | Notes |
|---|---|---|
| `OllamaAdapter` | `ollama` | Local inference via `format="json"`. |
| `OpenAIAdapter` | `openai` | `response_format={"type": "json_object"}`; `OPENAI_API_KEY` fallback. |
| `AnthropicAdapter` | `anthropic` | Forced tool-use for structured output; `ANTHROPIC_API_KEY` fallback. |
| `OpenRouterAdapter` | `openrouter` | OpenAI-compatible; reuses the `openai` SDK against `https://openrouter.ai/api/v1`; `OPENROUTER_API_KEY` fallback. |

Every adapter creates a **fresh client per call**. httpx transports are
event-loop bound, and the pipeline re-enters `asyncio.run()` from each
sync `index()` / `search()`, so caching an `AsyncClient` across calls
surfaces "Event loop is closed" in some Python runtimes.

`OpenAIAdapter` and `OllamaAdapter` share a `parse_json_object(content, source)`
helper colocated with the `LLMAdapter` ABC. Anthropic's tool-use parse is
backend-specific and stays in the adapter.

## Built-in embedding adapters

| Adapter | Extra | Notes |
|---|---|---|
| `SentenceTransformerEmbedding` | `sentence-transformers` | Lazy-loads the model on first call; `encode` runs in `asyncio.to_thread`. `embed_batch` forwards the full list to a single `encode` call. |
| `OpenAIEmbedding` | `openai` | `AsyncOpenAI` client, fresh per call. `embed_batch` issues one `embeddings.create(input=[...])` round-trip for all texts. |
| `OpenRouterEmbedding` | `openrouter` | `AsyncOpenAI` against OpenRouter's `/api/v1`; single-round-trip `embed_batch`; `OPENROUTER_API_KEY` fallback. |

The pipeline uses `embed_batch` whenever a document has multiple semantic
fields, so multi-semantic schemas only pay one network round-trip per
document with `OpenAIEmbedding`.

## Custom adapters

Inherit the ABC, implement the abstract method(s), pass the instance into
`Pipeline(...)`:

```python
from ennoia.adapters.llm import LLMAdapter

class VllmAdapter(LLMAdapter):
    async def complete_json(self, prompt: str) -> dict[str, object]:
        # call your vLLM server, parse JSON
        ...

    async def complete_text(self, prompt: str) -> str:
        ...
```

For embeddings, override `embed` only — `embed_document`, `embed_query`,
and `embed_batch` are inherited and route through `embed`:

```python
from ennoia.adapters.embedding import EmbeddingAdapter

class MyEmbedding(EmbeddingAdapter):
    async def embed(self, text: str) -> list[float]:
        ...
```

If a backend needs asymmetric doc/query flows (e.g. Voyage's `input_type`),
override `embed_document` and `embed_query` directly — the ABC methods are
not `@final`. Backends with a native list-input API should also override
`embed_batch` for the single-round-trip win.

For sync libraries, dispatch the blocking call via `asyncio.to_thread`
inside `embed` (and `embed_batch` if the library accepts a list) — the
sentence-transformers adapter is the canonical example.

## Optional-dependency loading

Adapters that need third-party packages load them through
`ennoia.utils.imports.require_module`, which raises a uniform
`ImportError: {pkg} is required. Install with 'pip install ennoia[{extra}]'`
on miss. Follow the same pattern in custom adapters so users get a
consistent error surface.

## Further reading

- [Cookbook — Custom adapters and stores](cookbook/custom-adapter.md)
  walks through full worked examples for a vLLM LLM adapter, a Voyage
  embedding adapter with asymmetric doc/query flows, and a custom
  `HybridStore` — including the event-loop and optional-dependency
  conventions every built-in adapter follows.
