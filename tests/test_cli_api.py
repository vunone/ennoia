"""CLI wiring for ``ennoia api`` — verify option parsing + schema loading."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from ennoia.cli.main import app


def _schema_file(tmp_path: Path) -> Path:
    src = tmp_path / "schemas.py"
    src.write_text(
        "from ennoia import BaseStructure, BaseSemantic\n"
        "class Doc(BaseStructure):\n"
        "    '''Extract doc metadata.'''\n"
        "    cat: str\n"
        "class Summary(BaseSemantic):\n"
        "    '''Summarise.'''\n"
    )
    return src


def test_api_requires_auth_or_no_auth_flag(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "api",
            "--store",
            str(tmp_path / "idx"),
            "--schema",
            str(_schema_file(tmp_path)),
            "--llm",
            "ollama:qwen3:0.6b",
            "--embedding",
            "sentence-transformers:all-MiniLM-L6-v2",
        ],
    )
    assert result.exit_code != 0
    assert "No auth configured" in result.output


def test_api_rejects_when_no_schema_classes(tmp_path: Path) -> None:
    empty_schema = tmp_path / "empty.py"
    empty_schema.write_text("# no schemas\n")
    result = CliRunner().invoke(
        app,
        [
            "api",
            "--store",
            str(tmp_path / "idx"),
            "--schema",
            str(empty_schema),
            "--no-auth",
        ],
    )
    assert result.exit_code != 0


def test_api_runs_uvicorn_with_app_and_port(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    class _FakeConfig:
        def __init__(self, app_: object, host: str, port: int, log_level: str) -> None:
            captured["app"] = app_
            captured["host"] = host
            captured["port"] = port
            captured["log_level"] = log_level

    class _FakeServer:
        def __init__(self, config: _FakeConfig) -> None:
            captured["config"] = config

        def run(self) -> None:
            captured["ran"] = True

    # Point the uvicorn module the adapter resolves to via require_module at our fakes.
    import uvicorn  # pyright: ignore[reportMissingImports]

    monkeypatch.setattr(uvicorn, "Config", _FakeConfig)
    monkeypatch.setattr(uvicorn, "Server", _FakeServer)

    # Also stub out the adapter resolvers so we don't instantiate real ollama/sentence-transformers.
    from ennoia.cli import api as api_module

    monkeypatch.setattr(api_module, "parse_llm_spec", lambda _uri: _StubLLM())
    monkeypatch.setattr(api_module, "parse_embedding_spec", lambda _uri: _StubEmb())

    result = CliRunner().invoke(
        app,
        [
            "api",
            "--store",
            str(tmp_path / "idx"),
            "--schema",
            str(_schema_file(tmp_path)),
            "--no-auth",
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
        ],
    )
    if result.exit_code != 0:
        print(result.output)
    assert result.exit_code == 0
    assert captured["ran"] is True
    assert captured["host"] == "0.0.0.0"
    assert captured["port"] == 9000


class _StubLLM:
    async def complete_json(self, prompt: str) -> dict[str, object]:
        return {}

    async def complete_text(self, prompt: str) -> str:
        return ""


class _StubEmb:
    async def embed(self, text: str) -> list[float]:
        return [0.0]

    async def embed_document(self, text: str) -> list[float]:
        return [0.0]

    async def embed_query(self, text: str) -> list[float]:
        return [0.0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]


def test_api_qdrant_store_wiring(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Prove that ``--store qdrant:<collection>`` reaches the hybrid adapter
    # through parse_store_spec without requiring the real Qdrant package.
    captured: dict[str, object] = {}

    from ennoia.cli import api as api_module

    def _fake_parse_store_spec(
        spec: str,
        *,
        collection: str,
        qdrant_url: str | None,
        qdrant_api_key: str | None,
        pg_dsn: str | None,
    ) -> object:
        captured["spec"] = spec
        captured["collection"] = collection
        captured["qdrant_url"] = qdrant_url
        captured["qdrant_api_key"] = qdrant_api_key
        captured["pg_dsn"] = pg_dsn

        class _FakeStore:
            pass

        return _FakeStore()

    class _FakeConfig:
        def __init__(self, *a: object, **kw: object) -> None:
            pass

    class _FakeServer:
        def __init__(self, *a: object, **kw: object) -> None:
            pass

        def run(self) -> None:
            captured["ran"] = True

    import uvicorn  # pyright: ignore[reportMissingImports]

    monkeypatch.setattr(uvicorn, "Config", _FakeConfig)
    monkeypatch.setattr(uvicorn, "Server", _FakeServer)
    monkeypatch.setattr(api_module, "parse_store_spec", _fake_parse_store_spec)
    monkeypatch.setattr(api_module, "parse_llm_spec", lambda _uri: _StubLLM())
    monkeypatch.setattr(api_module, "parse_embedding_spec", lambda _uri: _StubEmb())

    result = CliRunner().invoke(
        app,
        [
            "api",
            "--store",
            "qdrant:my_coll",
            "--qdrant-url",
            "http://qdrant.example:6333",
            "--qdrant-api-key",
            "sekret",
            "--schema",
            str(_schema_file(tmp_path)),
            "--no-auth",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["spec"] == "qdrant:my_coll"
    assert captured["qdrant_url"] == "http://qdrant.example:6333"
    assert captured["qdrant_api_key"] == "sekret"
    assert captured["ran"] is True
