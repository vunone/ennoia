"""Testing utilities for downstream users — mock adapters, fixtures, helpers.

The :mod:`ennoia.testing.fixtures` module is wired into pytest via the
``pytest11`` entry point in ``pyproject.toml`` so user test suites that
``pip install ennoia`` get ``mock_pipeline`` / ``mock_store`` / ``sample_schemas``
for free without a conftest import dance.
"""

from __future__ import annotations

from ennoia.testing.mocks import MockEmbeddingAdapter, MockLLMAdapter, MockStore

__all__ = ["MockEmbeddingAdapter", "MockLLMAdapter", "MockStore"]
