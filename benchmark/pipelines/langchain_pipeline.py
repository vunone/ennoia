"""Naive LangChain baseline: one embedding per product.

Concatenates ``title + text + bullets`` into a single ``Document`` per
product, embeds once with OpenAI ``text-embedding-3-small`` (same model
the Ennoia pipeline uses for its vector index), stores in
``InMemoryVectorStore``, and serves queries via ``similarity_search`` +
the shared generator prompt.

No chunking. No metadata filters. No reranking. The canonical naive-RAG
comparator every public tutorial ships.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_community.vectorstores import InMemoryVectorStore  # type: ignore[import-untyped]
from langchain_core.documents import Document  # type: ignore[import-untyped]
from langchain_core.embeddings import Embeddings  # type: ignore[import-untyped]
from tqdm import tqdm

from benchmark.config import MODEL_EMBED, RETRIEVAL_TOP_K
from benchmark.data.prep import Product
from benchmark.eval.cost import count_tokens
from benchmark.pipelines._retry import async_with_retry
from benchmark.pipelines.base import PipelineRun
from benchmark.pipelines.generator import format_context, generate_answer, make_generator_llm
from ennoia.adapters.embedding.openai import OpenAIEmbedding


@dataclass(slots=True)
class _RetrievedDoc:
    source_id: str
    text: str
    score: float


EMBED_BATCH_SIZE = 64


class _OpenAIEmbeddingsShim(Embeddings):
    """Minimal LangChain ``Embeddings`` wrapper over Ennoia's OpenAI adapter.

    Using Ennoia's adapter keeps the baseline honest: the same embedding
    model runs against the same SDK on both sides of the comparison.
    LangChain wants sync methods, so we bridge to the async adapter via
    a fresh event loop per call.

    ``embed_documents`` chunks the corpus into ``EMBED_BATCH_SIZE`` slices
    and drives a tqdm bar over them — so the 1000-product corpus indexes
    with visible progress, each batch retries rate-limit errors
    independently, and we stay well below OpenAI's per-request token cap.
    """

    def __init__(self, model: str) -> None:
        self._adapter = OpenAIEmbedding(model=model)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        import asyncio

        if not texts:
            return []
        vectors: list[list[float]] = []
        bar = tqdm(total=len(texts), desc="langchain embed", unit="product")
        try:
            for start in range(0, len(texts), EMBED_BATCH_SIZE):
                chunk = texts[start : start + EMBED_BATCH_SIZE]
                batch = asyncio.run(async_with_retry(lambda c=chunk: self._adapter.embed_batch(c)))
                vectors.extend(batch)
                bar.update(len(chunk))
        finally:
            bar.close()
        return vectors

    def embed_query(self, text: str) -> list[float]:
        import asyncio

        return asyncio.run(async_with_retry(lambda: self._adapter.embed(text)))


def _product_to_text(product: Product) -> str:
    parts: list[str] = [product["title"]]
    if product["brand"] and product["brand"] != "unknown":
        parts.append(f"Brand: {product['brand']}")
    if product["color"] and product["color"] != "unknown":
        parts.append(f"Color: {product['color']}")
    # Price lives in the text blob for the LangChain baseline too — the
    # embedder sees it alongside title + description. Fair comparison
    # against Ennoia, which uses the same blob for its extraction step.
    parts.append(f"Price: ${product['price_usd']}")
    if product["text"]:
        parts.append(product["text"])
    if product["bullet_points"]:
        parts.append("\n".join(product["bullet_points"]))
    return "\n\n".join(parts)


class LangchainPipeline:
    name = "langchain"

    def __init__(self, embed_model: str = MODEL_EMBED) -> None:
        self._embeddings: Embeddings = _OpenAIEmbeddingsShim(embed_model)
        self._store: InMemoryVectorStore | None = None
        self._generator = make_generator_llm()

    async def index_corpus(self, products: list[Product]) -> None:
        documents: list[Document] = []
        for product in tqdm(products, desc="langchain docs", unit="product"):
            documents.append(
                Document(
                    page_content=_product_to_text(product),
                    metadata={"source_id": product["docid"]},
                )
            )
        self._store = await InMemoryVectorStore.afrom_documents(documents, self._embeddings)
        print(f"[langchain] indexed {len(documents)} products")

    async def answer(self, query: str) -> PipelineRun:
        if self._store is None:
            raise RuntimeError("LangchainPipeline.index_corpus must be called before answer.")
        results = await self._store.asimilarity_search_with_score(query, k=RETRIEVAL_TOP_K)
        docs = [
            _RetrievedDoc(
                source_id=str(doc.metadata.get("source_id", "")),
                text=doc.page_content,
                score=float(score),
            )
            for doc, score in results
        ]
        retrieved_ids: list[str] = []
        for doc in docs:
            if doc.source_id and doc.source_id not in retrieved_ids:
                retrieved_ids.append(doc.source_id)

        context_blocks = [(d.source_id, d.text, d.score) for d in docs]
        prompt = format_context(query, context_blocks)
        answer_text = (await generate_answer(prompt, self._generator)) or "NOT_FOUND"

        prompt_tokens = count_tokens(prompt)
        completion_tokens = count_tokens(answer_text)

        return PipelineRun(
            retrieved_source_ids=retrieved_ids,
            answer=answer_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
