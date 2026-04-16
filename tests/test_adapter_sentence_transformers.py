"""SentenceTransformerEmbedding unit tests — fake the SDK + env-var side effects."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import ennoia.adapters.embedding.sentence_transformers as st_module
from ennoia.adapters.embedding.sentence_transformers import (
    SentenceTransformerEmbedding,
    _silence_model_loading,
)


class _FakeSentenceTransformer:
    instances: list[_FakeSentenceTransformer] = []

    def __init__(self, model: str, device: str | None = None) -> None:
        self.model = model
        self.device = device
        self.encode_calls: list[Any] = []
        type(self).instances.append(self)

    def encode(self, text: str | list[str], **_kwargs: Any) -> Any:
        self.encode_calls.append(text)
        if isinstance(text, list):
            # Match real SentenceTransformer.encode: a list returns a 2-D array.
            return np.asarray([[0.1, 0.2, 0.3] for _ in text], dtype=float)
        return np.asarray([0.1, 0.2, 0.3], dtype=float)


@pytest.fixture()
def patched_st(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeSentenceTransformer.instances = []
    fake_module = SimpleNamespace(SentenceTransformer=_FakeSentenceTransformer)
    monkeypatch.setattr(st_module, "require_module", lambda *_args: fake_module)


def test_silence_model_loading_sets_env_and_logger_levels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var in (
        "TRANSFORMERS_VERBOSITY",
        "TRANSFORMERS_NO_ADVISORY_WARNINGS",
        "HF_HUB_DISABLE_PROGRESS_BARS",
        "TOKENIZERS_PARALLELISM",
    ):
        monkeypatch.delenv(var, raising=False)
    for name in ("transformers", "huggingface_hub", "sentence_transformers"):
        logging.getLogger(name).setLevel(logging.NOTSET)

    _silence_model_loading()

    import os

    assert os.environ["TRANSFORMERS_VERBOSITY"] == "error"
    assert os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] == "1"
    assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
    for name in ("transformers", "huggingface_hub", "sentence_transformers"):
        assert logging.getLogger(name).level == logging.ERROR


def test_silence_respects_upstream_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRANSFORMERS_VERBOSITY", "info")
    _silence_model_loading()

    import os

    assert os.environ["TRANSFORMERS_VERBOSITY"] == "info"


def test_get_model_lazily_loads_once(patched_st: None) -> None:
    adapter = SentenceTransformerEmbedding(model="test-model", device="cpu")
    first = adapter._get_model()
    second = adapter._get_model()
    assert first is second
    # SentenceTransformer constructor called exactly once.
    assert len(_FakeSentenceTransformer.instances) == 1
    assert first.model == "test-model"
    assert first.device == "cpu"


async def test_embed_returns_list_of_floats(patched_st: None) -> None:
    adapter = SentenceTransformerEmbedding(model="test-model")
    vec = await adapter.embed("hello")
    assert vec == [pytest.approx(0.1), pytest.approx(0.2), pytest.approx(0.3)]
    assert _FakeSentenceTransformer.instances[0].encode_calls == ["hello"]


async def test_embed_document_and_query_share_model(patched_st: None) -> None:
    adapter = SentenceTransformerEmbedding(model="test-model")
    await adapter.embed_document("doc")
    await adapter.embed_query("query")
    # Single model instance serves both paths — no reload per call.
    assert len(_FakeSentenceTransformer.instances) == 1
    assert _FakeSentenceTransformer.instances[0].encode_calls == ["doc", "query"]


async def test_embed_batch_makes_one_encode_call(patched_st: None) -> None:
    """`encode` accepts a list — one threadpool hop and one model invocation."""
    adapter = SentenceTransformerEmbedding(model="test-model")
    vectors = await adapter.embed_batch(["a", "b", "c"])
    assert (
        vectors
        == [
            [pytest.approx(0.1), pytest.approx(0.2), pytest.approx(0.3)],
        ]
        * 3
    )
    # Single ``encode`` call for the whole batch — saves N-1 model invocations.
    assert _FakeSentenceTransformer.instances[0].encode_calls == [["a", "b", "c"]]


async def test_embed_batch_empty_short_circuits(patched_st: None) -> None:
    adapter = SentenceTransformerEmbedding(model="test-model")
    assert await adapter.embed_batch([]) == []
    # No model load and no encode call for an empty batch.
    assert _FakeSentenceTransformer.instances == []
