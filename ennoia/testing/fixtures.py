"""pytest fixtures exposed to downstream suites via the ``pytest11`` entry point.

Users who ``pip install ennoia[dev]`` (or ``ennoia`` alone) get three fixtures
for free — no conftest imports required:

- ``mock_store``  — a fresh :class:`ennoia.testing.MockStore`.
- ``mock_llm``    — a :class:`ennoia.testing.MockLLMAdapter` with no scripted
  responses (test supplies them via ``fixture.json_responses = {...}`` if
  needed — or builds its own adapter).
- ``mock_embedding`` — an 8-dim :class:`ennoia.testing.MockEmbeddingAdapter`.
- ``mock_pipeline`` — a :class:`~ennoia.index.pipeline.Pipeline` pre-wired with
  the three fixtures above. Callers pass ``schemas=`` at the test boundary by
  creating their own ``Pipeline`` if they need non-trivial schema shapes.
"""

from __future__ import annotations

import pytest

from ennoia.testing.mocks import MockEmbeddingAdapter, MockLLMAdapter, MockStore


@pytest.fixture
def mock_store() -> MockStore:
    return MockStore()


@pytest.fixture
def mock_llm() -> MockLLMAdapter:
    return MockLLMAdapter()


@pytest.fixture
def mock_embedding() -> MockEmbeddingAdapter:
    return MockEmbeddingAdapter(dim=8)
