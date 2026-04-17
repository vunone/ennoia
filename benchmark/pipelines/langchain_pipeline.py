"""Textbook langchain shred-embed RAG baseline.

``RecursiveCharacterTextSplitter`` -> local sentence-transformers embedder ->
``InMemoryVectorStore`` (langchain-community) -> ``similarity_search``. No
metadata filters, no reranking — the canonical naive-RAG comparator that
every public RAG tutorial recommends.

The pipeline produces its answer by handing the retrieved chunks to the
shared generator prompt from :mod:`benchmark.pipelines.generator`, so the
only variable that changes vs. the ennoia pipeline is retrieval quality +
agent reasoning, not the generator template.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_community.vectorstores import InMemoryVectorStore  # type: ignore[import-untyped]
from langchain_core.documents import Document  # type: ignore[import-untyped]
from langchain_core.embeddings import Embeddings  # type: ignore[import-untyped]
from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore[import-untyped]
from tqdm import tqdm

from benchmark.config import CHUNK_OVERLAP, CHUNK_SIZE, MODEL_EMBED, RETRIEVAL_TOP_K
from benchmark.data.loader import Contract, Question
from benchmark.eval.cost import count_tokens
from benchmark.pipelines.base import PipelineRun
from benchmark.pipelines.generator import format_context, generate_answer, make_generator_llm


@dataclass(slots=True)
class _RetrievedChunk:
    source_id: str
    text: str
    score: float


class _LocalSentenceTransformerEmbeddings(Embeddings):
    """Minimal langchain ``Embeddings`` shim over ``sentence_transformers``.

    Saves us the extra dependency on ``langchain-huggingface`` — the
    langchain baseline only needs ``embed_documents`` + ``embed_query`` to
    drive ``InMemoryVectorStore``. Matching the model on both sides of the
    comparison (ennoia uses ennoia's own ``SentenceTransformerEmbedding``)
    keeps retrieval a fair apples-to-apples test.
    """

    def __init__(self, model: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model: Any = SentenceTransformer(model)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        matrix = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
        return [row.tolist() for row in matrix]

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode(text, convert_to_numpy=True, normalize_embeddings=False)
        return vector.tolist()


class LangchainPipeline:
    name = "langchain"

    def __init__(self, embed_model: str = MODEL_EMBED) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        self._embeddings = _LocalSentenceTransformerEmbeddings(embed_model)
        self._store: InMemoryVectorStore | None = None
        self._generator = make_generator_llm()

    async def index_corpus(self, contracts: list[Contract]) -> None:
        documents: list[Document] = []
        for contract in tqdm(contracts, desc="langchain chunk", unit="doc"):
            chunks = self._splitter.split_text(contract["text"])
            for idx, chunk in enumerate(chunks):
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={"source_id": contract["source_id"], "chunk": idx},
                    )
                )
        print(f"[langchain] embedding {len(documents)} chunks (one batched call)...")
        self._store = await InMemoryVectorStore.afrom_documents(documents, self._embeddings)
        print(f"[langchain] indexed {len(documents)} chunks across {len(contracts)} contracts")

    async def answer(self, question: Question) -> PipelineRun:
        if self._store is None:
            raise RuntimeError("LangchainPipeline.index_corpus must be called before answer.")
        results = await self._store.asimilarity_search_with_score(
            question["question"], k=RETRIEVAL_TOP_K
        )
        chunks = [
            _RetrievedChunk(
                source_id=str(doc.metadata.get("source_id", "")),
                text=doc.page_content,
                score=float(score),
            )
            for doc, score in results
        ]
        retrieved_ids: list[str] = []
        for chunk in chunks:
            if chunk.source_id and chunk.source_id not in retrieved_ids:
                retrieved_ids.append(chunk.source_id)

        context_blocks = [(chunk.source_id, chunk.text, chunk.score) for chunk in chunks]
        prompt = format_context(question["question"], context_blocks)
        answer_text = (await generate_answer(prompt, self._generator)) or "NOT_FOUND"

        # tiktoken estimate since neither the Ollama nor the OpenAI
        # ennoia adapter surfaces ``response.usage`` to callers. Keeps
        # token accounting uniform across pipelines.
        prompt_tokens = count_tokens(prompt)
        completion_tokens = count_tokens(answer_text)

        return PipelineRun(
            retrieved_source_ids=retrieved_ids,
            answer=answer_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
