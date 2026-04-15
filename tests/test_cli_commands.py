"""End-to-end CLI tests using typer.testing.CliRunner and a fake LLM plugin."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("pandas")

from typer.testing import CliRunner

import ennoia.cli.main as cli_main

_SCHEMA_MODULE = '''
from typing import Literal
from ennoia import BaseStructure, BaseSemantic

class Doc(BaseStructure):
    """Extract doc metadata."""
    category: Literal["legal", "medical"]

class Summary(BaseSemantic):
    """What is the document about?"""
'''


class _FakeLLM:
    async def complete_json(self, prompt: str) -> dict[str, object]:
        return {"category": "legal", "_confidence": 0.9}

    async def complete_text(self, prompt: str) -> str:
        return "A legal document. <confidence>0.9</confidence>"


class _FakeEmbedding:
    def embed_document(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


@pytest.fixture()
def patched_adapters(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap adapter factories so the CLI runs without real network / downloads."""

    def fake_llm(spec: str):
        return _FakeLLM()

    def fake_embedding(spec: str):
        return _FakeEmbedding()

    monkeypatch.setattr(cli_main, "parse_llm_spec", fake_llm)
    monkeypatch.setattr(cli_main, "parse_embedding_spec", fake_embedding)


@pytest.fixture()
def schema_file(tmp_path: Path) -> Path:
    schema = tmp_path / "schemas.py"
    schema.write_text(_SCHEMA_MODULE)
    return schema


def test_try_command_prints_fields_and_confidence(
    patched_adapters: None, schema_file: Path, tmp_path: Path
) -> None:
    doc = tmp_path / "doc.txt"
    doc.write_text("body")
    runner = CliRunner()
    result = runner.invoke(
        cli_main.app,
        [
            "try",
            str(doc),
            "--schema",
            str(schema_file),
            "--llm",
            "fake:model",
            "--embedding",
            "fake:model",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Schema: Doc" in result.output
    assert "category" in result.output
    assert "confidence: 0.90" in result.output
    assert "Schema: Summary" in result.output


def test_index_then_search_roundtrip(
    patched_adapters: None, schema_file: Path, tmp_path: Path
) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.txt").write_text("body 1")
    (docs_dir / "b.txt").write_text("body 2")
    store_dir = tmp_path / "idx"

    runner = CliRunner()
    idx = runner.invoke(
        cli_main.app,
        [
            "index",
            str(docs_dir),
            "--schema",
            str(schema_file),
            "--store",
            str(store_dir),
            "--llm",
            "fake:m",
            "--embedding",
            "fake:m",
        ],
    )
    assert idx.exit_code == 0, idx.output
    assert "Indexed 2 document(s)" in idx.output

    srch = runner.invoke(
        cli_main.app,
        [
            "search",
            "legal documents",
            "--schema",
            str(schema_file),
            "--store",
            str(store_dir),
            "--filter",
            "category=legal",
            "--top-k",
            "5",
            "--llm",
            "fake:m",
            "--embedding",
            "fake:m",
        ],
    )
    assert srch.exit_code == 0, srch.output
    assert "Results" in srch.output


def test_search_rejects_invalid_filter(
    patched_adapters: None, schema_file: Path, tmp_path: Path
) -> None:
    store_dir = tmp_path / "idx"
    Path(store_dir).mkdir()
    runner = CliRunner()
    result = runner.invoke(
        cli_main.app,
        [
            "search",
            "x",
            "--schema",
            str(schema_file),
            "--store",
            str(store_dir),
            "--filter",
            "category__gt=legal",
            "--llm",
            "fake:m",
            "--embedding",
            "fake:m",
        ],
    )
    assert result.exit_code == 2
    assert '"error": "invalid_filter"' in result.output
