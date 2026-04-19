"""Unit tests for the LangChain product-baseline pipeline.

Stubs LangChain's ``InMemoryVectorStore`` and the embedder shim so the
tests exercise the pipeline's control flow without HTTP traffic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from benchmark.data.prep import Product
from benchmark.pipelines import langchain_pipeline as lcp


@dataclass
class _FakeDoc:
    page_content: str
    metadata: dict[str, Any]


class _FakeStore:
    def __init__(self, ranked_docids: list[str], corpus_texts: dict[str, str]) -> None:
        self._ranked = ranked_docids
        self._texts = corpus_texts
        self.queries: list[tuple[str, int]] = []

    async def asimilarity_search_with_score(
        self, query: str, k: int
    ) -> list[tuple[_FakeDoc, float]]:
        self.queries.append((query, k))
        return [
            (
                _FakeDoc(page_content=self._texts.get(d, ""), metadata={"source_id": d}),
                0.9 - i * 0.1,
            )
            for i, d in enumerate(self._ranked[:k])
        ]


async def _from_docs(documents: list[Any], embeddings: Any) -> _FakeStore:
    texts = {doc.metadata["source_id"]: doc.page_content for doc in documents}
    return _FakeStore(ranked_docids=list(texts.keys()), corpus_texts=texts)


class _FakeGenerator:
    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.prompts: list[str] = []

    async def complete_text(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._reply


def _make_pipeline(monkeypatch: pytest.MonkeyPatch, reply: str) -> lcp.LangchainPipeline:
    monkeypatch.setattr(
        lcp.InMemoryVectorStore, "afrom_documents", staticmethod(_from_docs), raising=False
    )
    monkeypatch.setattr(lcp, "_OpenAIEmbeddingsShim", lambda model: object())
    pipe = lcp.LangchainPipeline(embed_model="test-embed")
    pipe._generator = _FakeGenerator(reply)  # type: ignore[attr-defined]
    return pipe


_PRODUCTS: list[Product] = [
    {
        "docid": "p0",
        "title": "Aluminium Laptop Stand",
        "text": "",
        "bullet_points": ["foldable", "aluminium"],
        "brand": "ACME",
        "color": "silver",
        "price_usd": 45,
    },
    {
        "docid": "p1",
        "title": "USB-C Laptop Stand",
        "text": "stand with USB hub",
        "bullet_points": [],
        "brand": "BetaCo",
        "color": "unknown",
        "price_usd": 72,
    },
    {
        "docid": "p2",
        "title": "Third Product",
        "text": "",
        "bullet_points": [],
        "brand": "unknown",
        "color": "unknown",
        "price_usd": 15,
    },
]


async def test_index_then_answer_returns_ordered_docids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipe = _make_pipeline(monkeypatch, '{"docid":"p0","reason":"best match"}')
    await pipe.index_corpus(_PRODUCTS)
    run = await pipe.answer("aluminium laptop stand")
    assert run.retrieved_source_ids[:3] == ["p0", "p1", "p2"]
    assert run.answer == '{"docid":"p0","reason":"best match"}'
    # Generator prompt must include the shared format_context anchor.
    prompts = pipe._generator.prompts  # type: ignore[attr-defined]
    assert "docid=p0" in prompts[0]


async def test_answer_before_index_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    pipe = _make_pipeline(monkeypatch, "")
    with pytest.raises(RuntimeError, match="index_corpus"):
        await pipe.answer("q")


async def test_product_with_empty_bullets_still_indexed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipe = _make_pipeline(monkeypatch, "NOT_FOUND")
    products: list[Product] = [
        {
            "docid": "only",
            "title": "Empty-bullets Product",
            "text": "desc",
            "bullet_points": [],
            "brand": "unknown",
            "color": "unknown",
            "price_usd": 25,
        }
    ]
    await pipe.index_corpus(products)
    run = await pipe.answer("something")
    assert run.retrieved_source_ids == ["only"]


async def test_empty_answer_falls_back_to_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    pipe = _make_pipeline(monkeypatch, "")
    await pipe.index_corpus(_PRODUCTS[:1])
    run = await pipe.answer("q")
    assert run.answer == "NOT_FOUND"


def test_product_to_text_includes_all_fields() -> None:
    text = lcp._product_to_text(_PRODUCTS[0])
    assert "Aluminium Laptop Stand" in text
    assert "ACME" in text
    assert "silver" in text
    assert "Price: $45" in text
    assert "foldable" in text


def test_product_to_text_skips_unknown_and_empty_fields() -> None:
    text = lcp._product_to_text(_PRODUCTS[2])
    assert "Third Product" in text
    assert "Brand:" not in text
    assert "Color:" not in text
    # Price is always included — no "unknown" sentinel.
    assert "Price: $15" in text


class _FakeOpenAIEmbedding:
    def __init__(self, model: str) -> None:
        self.model = model

    async def embed(self, text: str) -> list[float]:
        return [0.1, 0.2]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]


def test_openai_embeddings_shim_embed_batch_and_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lcp, "OpenAIEmbedding", _FakeOpenAIEmbedding)
    shim = lcp._OpenAIEmbeddingsShim("test")
    assert shim.embed_documents(["a", "b"]) == [[0.1, 0.2], [0.1, 0.2]]
    assert shim.embed_documents([]) == []
    assert shim.embed_query("q") == [0.1, 0.2]


async def test_duplicate_and_empty_source_ids_dedup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty source_id and duplicate ids must be skipped / deduplicated."""
    pipe = _make_pipeline(monkeypatch, "NOT_FOUND")

    # Stub store directly with a fixed, messy retrieval order.
    class _MessyStore:
        async def asimilarity_search_with_score(
            self, query: str, k: int
        ) -> list[tuple[_FakeDoc, float]]:
            return [
                (_FakeDoc(page_content="x", metadata={"source_id": "p0"}), 0.9),
                (_FakeDoc(page_content="y", metadata={"source_id": ""}), 0.8),
                (_FakeDoc(page_content="z", metadata={"source_id": "p0"}), 0.7),
                (_FakeDoc(page_content="w", metadata={"source_id": "p1"}), 0.6),
            ]

    pipe._store = _MessyStore()  # type: ignore[attr-defined]
    run = await pipe.answer("q")
    assert run.retrieved_source_ids == ["p0", "p1"]


def test_make_pipeline_constructs_shim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lcp, "OpenAIEmbedding", _FakeOpenAIEmbedding)
    # Avoid constructing a real OpenRouter client.
    monkeypatch.setattr(
        lcp,
        "make_generator_llm",
        lambda: type("L", (), {"complete_text": lambda self, p: ""})(),
    )
    pipe = lcp.LangchainPipeline(embed_model="text-embedding-3-small")
    assert isinstance(pipe._embeddings, lcp._OpenAIEmbeddingsShim)
