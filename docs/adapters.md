# Adapters

Adapters are stateless transformers that sit between the pipeline and an
external service (LLM inference, embeddings). They conform to two
protocols:

- `LLMAdapter` — `async complete_json(prompt)` and `async complete_text(prompt)`.
- `EmbeddingAdapter` — synchronous `embed_document(text)` / `embed_query(text)`.

Protocols live in `ennoia/adapters/{llm,embedding}/protocols.py` and are
`@runtime_checkable`, so custom adapters can duck-type in without
inheritance.

## Built-in LLM adapters

| Adapter | Extra | Notes |
|---|---|---|
| `OllamaAdapter` | `ollama` | Local inference via `format="json"`. |
| `OpenAIAdapter` | `openai` | `response_format={"type": "json_object"}`; `OPENAI_API_KEY` fallback. |
| `AnthropicAdapter` | `anthropic` | Forced tool-use for structured output; `ANTHROPIC_API_KEY` fallback. |

Every adapter creates a **fresh client per call**. httpx transports are
event-loop bound, and the pipeline re-enters `asyncio.run()` from each
sync `index()` / `search()`, so caching an `AsyncClient` across calls
surfaces "Event loop is closed" in some Python runtimes.

## Built-in embedding adapters

| Adapter | Extra | Notes |
|---|---|---|
| `SentenceTransformerEmbedding` | `sentence-transformers` | Lazy-loads the model on first call. |
| `OpenAIEmbedding` | `openai` | Synchronous SDK client; `OPENAI_API_KEY` fallback. |

## Custom adapters

Implement the protocol and pass the instance into `Pipeline(...)`. No
inheritance required.

```python
class VllmAdapter:
    async def complete_json(self, prompt: str) -> dict[str, object]:
        # call your vLLM server, parse JSON
        ...

    async def complete_text(self, prompt: str) -> str:
        ...
```

The same pattern applies to custom embedding adapters (e.g. a fine-tuned
local encoder) and to custom stores (see [stores.md](stores.md)).

## Optional-dependency loading

Adapters that need third-party packages load them through
`ennoia.utils.imports.require_module`, which raises a uniform
`ImportError: {pkg} is required. Install with 'pip install ennoia[{extra}]'`
on miss. Follow the same pattern in custom adapters so users get a
consistent error surface.
