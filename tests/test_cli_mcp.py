"""CLI wiring for ``ennoia mcp`` — verify transport selection + auth gate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from ennoia.cli.main import app


def _schema_file(tmp_path: Path) -> Path:
    src = tmp_path / "schemas.py"
    src.write_text(
        "from ennoia import BaseStructure\n"
        "class Doc(BaseStructure):\n"
        "    '''Extract doc metadata.'''\n"
        "    cat: str\n"
    )
    return src


def _invoke(args: list[str], monkeypatch: pytest.MonkeyPatch) -> tuple[int, list[Any]]:
    ran_calls: list[Any] = []

    class _FakeMCP:
        def run(self, **kwargs: Any) -> None:
            ran_calls.append(kwargs)

    from ennoia.cli import mcp as mcp_module

    monkeypatch.setattr(mcp_module, "create_mcp", lambda _ctx: _FakeMCP())
    monkeypatch.setattr(
        mcp_module,
        "parse_llm_spec",
        lambda _uri: type("L", (), {"complete_json": None, "complete_text": None})(),
    )

    class _Emb:
        async def embed(self, t: str) -> list[float]:
            return [0.0]

        async def embed_document(self, t: str) -> list[float]:
            return [0.0]

        async def embed_query(self, t: str) -> list[float]:
            return [0.0]

        async def embed_batch(self, ts: list[str]) -> list[list[float]]:
            return [[0.0] for _ in ts]

    monkeypatch.setattr(mcp_module, "parse_embedding_spec", lambda _uri: _Emb())

    result = CliRunner().invoke(app, args)
    return result.exit_code, ran_calls


def test_mcp_sse_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    code, calls = _invoke(
        [
            "mcp",
            "--store",
            str(tmp_path / "idx"),
            "--schema",
            str(_schema_file(tmp_path)),
            "--no-auth",
        ],
        monkeypatch,
    )
    assert code == 0
    assert calls == [{"transport": "sse", "host": "127.0.0.1", "port": 8090}]


def test_mcp_http_passes_host_port(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    code, calls = _invoke(
        [
            "mcp",
            "--store",
            str(tmp_path / "idx"),
            "--schema",
            str(_schema_file(tmp_path)),
            "--transport",
            "http",
            "--host",
            "0.0.0.0",
            "--port",
            "9100",
            "--no-auth",
        ],
        monkeypatch,
    )
    assert code == 0
    assert calls == [{"transport": "http", "host": "0.0.0.0", "port": 9100}]


def test_mcp_stdio_does_not_pass_host_port(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    code, calls = _invoke(
        [
            "mcp",
            "--store",
            str(tmp_path / "idx"),
            "--schema",
            str(_schema_file(tmp_path)),
            "--transport",
            "stdio",
            "--no-auth",
        ],
        monkeypatch,
    )
    assert code == 0
    assert calls == [{"transport": "stdio"}]


def test_mcp_unknown_transport_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    code, _ = _invoke(
        [
            "mcp",
            "--store",
            str(tmp_path / "idx"),
            "--schema",
            str(_schema_file(tmp_path)),
            "--transport",
            "websocket",
            "--no-auth",
        ],
        monkeypatch,
    )
    assert code != 0


def test_mcp_requires_auth(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    code, _ = _invoke(
        [
            "mcp",
            "--store",
            str(tmp_path / "idx"),
            "--schema",
            str(_schema_file(tmp_path)),
        ],
        monkeypatch,
    )
    assert code != 0


def test_mcp_accepts_api_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    code, calls = _invoke(
        [
            "mcp",
            "--store",
            str(tmp_path / "idx"),
            "--schema",
            str(_schema_file(tmp_path)),
            "--api-key",
            "abc",
        ],
        monkeypatch,
    )
    assert code == 0
    assert len(calls) == 1


def test_mcp_pgvector_store_wiring(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # ``--store pgvector:<collection>`` must reach parse_store_spec with the
    # correct kwargs. We stub parse_store_spec and confirm the routing
    # without requiring asyncpg / pgvector at test time.
    captured: dict[str, object] = {}

    from ennoia.cli import mcp as mcp_module

    def _fake_parse(
        spec: str,
        *,
        collection: str,
        qdrant_url: str | None,
        qdrant_api_key: str | None,
        pg_dsn: str | None,
    ) -> object:
        captured["spec"] = spec
        captured["collection"] = collection
        captured["pg_dsn"] = pg_dsn

        class _S:
            pass

        return _S()

    class _FakeMCP:
        def run(self, **kwargs: Any) -> None:
            captured["transport"] = kwargs.get("transport")

    monkeypatch.setattr(mcp_module, "create_mcp", lambda _ctx: _FakeMCP())
    monkeypatch.setattr(mcp_module, "parse_store_spec", _fake_parse)
    monkeypatch.setattr(
        mcp_module,
        "parse_llm_spec",
        lambda _uri: type("L", (), {"complete_json": None, "complete_text": None})(),
    )

    class _Emb:
        async def embed(self, t: str) -> list[float]:
            return [0.0]

        async def embed_document(self, t: str) -> list[float]:
            return [0.0]

        async def embed_query(self, t: str) -> list[float]:
            return [0.0]

        async def embed_batch(self, ts: list[str]) -> list[list[float]]:
            return [[0.0] for _ in ts]

    monkeypatch.setattr(mcp_module, "parse_embedding_spec", lambda _uri: _Emb())

    result = CliRunner().invoke(
        app,
        [
            "mcp",
            "--store",
            "pgvector:emails",
            "--pg-dsn",
            "postgresql://user:pass@pg/db",
            "--schema",
            str(_schema_file(tmp_path)),
            "--no-auth",
            "--transport",
            "stdio",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["spec"] == "pgvector:emails"
    assert captured["pg_dsn"] == "postgresql://user:pass@pg/db"
    assert captured["transport"] == "stdio"
