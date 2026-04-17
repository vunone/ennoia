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
from dataclasses import dataclass
from typing import Any, cast

from ennoia.adapters.embedding.base import EmbeddingAdapter
from ennoia.adapters.llm.base import LLMAdapter
from ennoia.store.base import HybridStore, VectorEntry
from ennoia.utils.filters import apply_filters
from ennoia.utils.ids import make_semantic_vector_id

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


@dataclass
class _Row:
    """One denormalized row in :class:`MockStore`.

    The ``data`` payload is identical across all rows sharing a ``source_id``;
    the store writes it once per row to mirror what real hybrid backends do on
    disk.
    """

    vector_id: str
    source_id: str
    index_name: str
    unique: str | None
    text: str
    data: dict[str, Any]
    vector: list[float]


class MockStore(HybridStore):
    """In-memory :class:`HybridStore` with row-per-entry semantics.

    A document with one ``BaseSemantic`` yields one row; a ``BaseCollection``
    with N entities yields N rows sharing the same ``data``. The filter evaluator
    matches rows individually and the pipeline dedups by ``source_id`` at the
    search boundary.
    """

    def __init__(self) -> None:
        self._rows: dict[str, _Row] = {}

    async def upsert(
        self,
        source_id: str,
        data: dict[str, Any],
        entries: list[VectorEntry],
    ) -> None:
        # Replace-by-source semantics: any existing rows for this document are
        # dropped before the new ones land. A doc previously indexed with 5
        # collection entries and now re-indexed with 3 correctly shrinks to 3.
        self._rows = {vid: row for vid, row in self._rows.items() if row.source_id != source_id}
        for entry in entries:
            vector_id = make_semantic_vector_id(source_id, entry.index_name, entry.unique)
            self._rows[vector_id] = _Row(
                vector_id=vector_id,
                source_id=source_id,
                index_name=entry.index_name,
                unique=entry.unique,
                text=entry.text,
                data=dict(data),
                vector=list(entry.vector),
            )

    async def hybrid_search(
        self,
        filters: dict[str, Any],
        query_vector: list[float],
        top_k: int,
        *,
        index: str | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        allowed_ids = set(
            apply_filters(
                ((row.vector_id, row.data) for row in self._rows.values()),
                filters or {},
            )
        )
        scored: list[tuple[str, float, dict[str, Any]]] = []
        for row in self._rows.values():
            if row.vector_id not in allowed_ids:
                continue
            if index is not None and row.index_name != index:
                continue
            score = _cosine(query_vector, row.vector)
            if score is None:
                continue
            meta = {
                **row.data,
                "source_id": row.source_id,
                "index": row.index_name,
                "text": row.text,
            }
            if row.unique is not None:
                meta["unique"] = row.unique
            scored.append((row.vector_id, score, meta))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:top_k]

    async def get(self, source_id: str) -> dict[str, Any] | None:
        for row in self._rows.values():
            if row.source_id == source_id:
                return dict(row.data)
        return None

    async def filter(self, filters: dict[str, Any]) -> list[str]:
        allowed_vector_ids = set(
            apply_filters(
                ((row.vector_id, row.data) for row in self._rows.values()),
                filters or {},
            )
        )
        # Preserve first-seen-row order per source_id so results are
        # deterministic across runs while still being distinct.
        seen: list[str] = []
        seen_set: set[str] = set()
        for row in self._rows.values():
            if row.vector_id not in allowed_vector_ids:
                continue
            if row.source_id in seen_set:
                continue
            seen.append(row.source_id)
            seen_set.add(row.source_id)
        return seen

    async def delete(self, source_id: str) -> bool:
        before = len(self._rows)
        self._rows = {vid: row for vid, row in self._rows.items() if row.source_id != source_id}
        return len(self._rows) < before


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
