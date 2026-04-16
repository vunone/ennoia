"""Scripted mocks for LLM, embedding, and store backends.

Design goals:

- **Deterministic.** No randomness, no network. Tests depending on these
  objects should be reproducible byte-for-byte across runs.
- **No Ollama / OpenAI install needed.** The mocks only depend on the ennoia
  core (pydantic). Users who want to unit-test their schemas can
  ``pip install ennoia[dev]`` without pulling any heavy ML deps.
- **Round-trippable.** :class:`MockStore` speaks the full :class:`HybridStore`
  contract so the pipeline can index + filter + search + retrieve end-to-end
  against it in a single test.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from typing import Any, cast

from ennoia.adapters.embedding.base import EmbeddingAdapter
from ennoia.adapters.llm.base import LLMAdapter
from ennoia.store.base import HybridStore
from ennoia.utils.filters import apply_filters

__all__ = ["MockEmbeddingAdapter", "MockLLMAdapter", "MockStore"]


class MockLLMAdapter(LLMAdapter):
    """Scripted LLM adapter.

    ``json_responses`` and ``text_responses`` can each be either:

    - A plain mapping from a substring of the prompt to the response. The
      first matching key (by iteration order) wins.
    - A callable that takes the full prompt and returns the response.
    - A plain list, consumed in order (one response per call).

    Unmatched prompts raise :class:`AssertionError` to surface miswired tests.
    """

    def __init__(
        self,
        json_responses: (
            dict[str, dict[str, Any]]
            | Callable[[str], dict[str, Any]]
            | list[dict[str, Any]]
            | None
        ) = None,
        text_responses: (dict[str, str] | Callable[[str], str] | list[str] | None) = None,
    ) -> None:
        self._json = json_responses or {}
        self._text = text_responses or {}
        self.json_calls: list[str] = []
        self.text_calls: list[str] = []

    async def complete_json(self, prompt: str) -> dict[str, Any]:
        self.json_calls.append(prompt)
        return _pick_response(self._json, prompt, "JSON")  # type: ignore[return-value]

    async def complete_text(self, prompt: str) -> str:
        self.text_calls.append(prompt)
        return _pick_response(self._text, prompt, "text")  # type: ignore[return-value]


def _pick_response(source: Any, prompt: str, label: str) -> Any:
    if callable(source):
        return source(prompt)
    if isinstance(source, dict):
        # Longest key first so more specific matches win.
        source_dict = cast(dict[str, Any], source)
        for key in sorted(source_dict.keys(), key=len, reverse=True):
            if key in prompt:
                return source_dict[key]
    if isinstance(source, list):
        source_list = cast(list[Any], source)
        if not source_list:
            raise AssertionError(f"MockLLMAdapter ran out of scripted {label} responses.")
        return source_list.pop(0)
    raise AssertionError(f"MockLLMAdapter has no scripted {label} response for prompt: {prompt!r}")


class MockEmbeddingAdapter(EmbeddingAdapter):
    """Deterministic hash-seeded embeddings.

    Each call returns a unit-norm vector of length ``dim`` derived from SHA-256
    of the input text. Identical input → identical output across processes and
    platforms; small text changes → different vectors, so similarity search in
    tests produces meaningful (if not semantically faithful) ordering.
    """

    def __init__(self, dim: int = 8) -> None:
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        self.dim = dim
        self.calls: list[str] = []

    async def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # Expand the 32-byte digest into ``dim`` floats by re-hashing as needed.
        raw: list[int] = []
        seed = digest
        while len(raw) < self.dim:
            raw.extend(seed[:32])
            seed = hashlib.sha256(seed).digest()
        # Map bytes [0, 255] -> [-1.0, 1.0], then L2-normalise.
        values = [((b / 255.0) * 2.0) - 1.0 for b in raw[: self.dim]]
        norm = math.sqrt(sum(v * v for v in values))
        if norm == 0.0:  # pragma: no cover — SHA-256 can't collapse to all-128 bytes
            return values
        return [v / norm for v in values]


class MockStore(HybridStore):
    """In-memory :class:`HybridStore` implementation for user test suites.

    Stores the flattened structural record + a dict of named vectors per
    ``source_id``, answers filters via the canonical
    :func:`ennoia.utils.filters.apply_filters`, and scores search hits with
    cosine similarity.
    """

    def __init__(self) -> None:
        self._records: dict[str, dict[str, Any]] = {}
        self._vectors: dict[str, dict[str, list[float]]] = {}

    async def upsert(
        self,
        source_id: str,
        data: dict[str, Any],
        vectors: dict[str, list[float]],
    ) -> None:
        self._records[source_id] = dict(data)
        self._vectors[source_id] = {k: list(v) for k, v in vectors.items()}

    async def hybrid_search(
        self,
        filters: dict[str, Any],
        query_vector: list[float],
        top_k: int,
        *,
        index: str | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        candidate_ids = apply_filters(self._records.items(), filters or {})
        scored: list[tuple[str, float, dict[str, Any]]] = []
        for source_id in candidate_ids:
            named = self._vectors.get(source_id, {})
            pool = {index: named[index]} if index and index in named else named
            for name, vec in pool.items():
                score = _cosine(query_vector, vec)
                if score is None:
                    continue
                meta = {
                    **self._records.get(source_id, {}),
                    "source_id": source_id,
                    "index": name,
                }
                scored.append((f"{source_id}:{name}", score, meta))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:top_k]

    async def get(self, source_id: str) -> dict[str, Any] | None:
        record = self._records.get(source_id)
        return dict(record) if record is not None else None

    async def filter(self, filters: dict[str, Any]) -> list[str]:
        return apply_filters(self._records.items(), filters or {})

    async def delete(self, source_id: str) -> bool:
        removed = self._records.pop(source_id, None) is not None
        self._vectors.pop(source_id, None)
        return removed


def _cosine(a: list[float], b: list[float]) -> float | None:
    if not a or not b:
        return None
    if len(a) != len(b):
        return None
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return None
    return dot / (na * nb)
