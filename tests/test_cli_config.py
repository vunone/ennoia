"""Tests for ``ennoia.cli.config`` and the ``ennoia init`` command.

These cover the INI loader's setdefault semantics, the unknown-key
guard, the ``ennoia init`` template writer, and the app-level callback
that wires ``ennoia.ini`` into every subcommand.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("pandas")

import typer
from typer.testing import CliRunner

import ennoia.cli.main as cli_main
from ennoia.adapters.embedding import EmbeddingAdapter
from ennoia.adapters.llm import LLMAdapter
from ennoia.cli.config import (
    INI_FILENAME,
    INI_TEMPLATE,
    KEY_TO_ENVVAR,
    load_ini,
    require_option,
    write_template,
)

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences so substring asserts survive Rich styling."""
    return _ANSI_RE.sub("", text)


_ALL_ENV_VARS: tuple[str, ...] = (
    *KEY_TO_ENVVAR.values(),
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENROUTER_API_KEY",
    "EXTRA_CUSTOM",
)


@pytest.fixture(autouse=True)
def _reset_env() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    """Snapshot + restore every Ennoia-related env var.

    ``load_ini`` uses ``os.environ.setdefault`` directly, which isn't
    tracked by ``monkeypatch``. Without an explicit restore, values set
    by one test leak into other test files run later in the same
    session.
    """
    snapshot = {name: os.environ.get(name) for name in _ALL_ENV_VARS}
    for name in _ALL_ENV_VARS:
        os.environ.pop(name, None)
    try:
        yield
    finally:
        for name in _ALL_ENV_VARS:
            original = snapshot[name]
            if original is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = original


# ---------------------------------------------------------------------------
# load_ini
# ---------------------------------------------------------------------------


def test_load_ini_missing_file_returns_false(tmp_path: Path) -> None:
    assert load_ini(tmp_path / "nope.ini") is False


def test_load_ini_populates_ennoia_env_vars(tmp_path: Path) -> None:
    ini = tmp_path / "ennoia.ini"
    ini.write_text(
        "[ennoia]\n"
        "llm = openai:gpt-4o-mini\n"
        "embedding = openai-embedding:text-embedding-3-small\n"
        "store = qdrant:docs\n"
        "collection = invoices\n"
        "schema = ./schemas.py\n"
        "qdrant_url = http://qdrant:6333\n"
        "qdrant_api_key = qkey\n"
        "pg_dsn = postgresql://u:p@h/db\n"
        "host = 0.0.0.0\n"
        "port = 9000\n"
        "transport = http\n"
        "api_key = bearer-token\n"
    )
    import os

    assert load_ini(ini) is True
    assert os.environ["ENNOIA_LLM"] == "openai:gpt-4o-mini"
    assert os.environ["ENNOIA_EMBEDDING"] == "openai-embedding:text-embedding-3-small"
    assert os.environ["ENNOIA_STORE"] == "qdrant:docs"
    assert os.environ["ENNOIA_COLLECTION"] == "invoices"
    assert os.environ["ENNOIA_SCHEMA"] == "./schemas.py"
    assert os.environ["ENNOIA_QDRANT_URL"] == "http://qdrant:6333"
    assert os.environ["ENNOIA_QDRANT_API_KEY"] == "qkey"
    assert os.environ["ENNOIA_PG_DSN"] == "postgresql://u:p@h/db"
    assert os.environ["ENNOIA_HOST"] == "0.0.0.0"
    assert os.environ["ENNOIA_PORT"] == "9000"
    assert os.environ["ENNOIA_TRANSPORT"] == "http"
    assert os.environ["ENNOIA_API_KEY"] == "bearer-token"


def test_load_ini_passes_through_env_section_case_preserved(tmp_path: Path) -> None:
    ini = tmp_path / "ennoia.ini"
    ini.write_text("[env]\nOPENAI_API_KEY = sk-openai\nANTHROPIC_API_KEY = sk-anthropic\n")
    import os

    assert load_ini(ini) is True
    assert os.environ["OPENAI_API_KEY"] == "sk-openai"
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-anthropic"


def test_load_ini_rejects_unknown_ennoia_key(tmp_path: Path) -> None:
    ini = tmp_path / "ennoia.ini"
    ini.write_text("[ennoia]\nunknown_field = whatever\n")
    with pytest.raises(typer.BadParameter, match="Unknown key 'unknown_field'"):
        load_ini(ini)


def test_load_ini_setdefault_semantics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENNOIA_LLM", "preset:model")
    ini = tmp_path / "ennoia.ini"
    ini.write_text("[ennoia]\nllm = from-file:model\n")

    import os

    load_ini(ini)
    assert os.environ["ENNOIA_LLM"] == "preset:model"


def test_load_ini_skips_empty_values(tmp_path: Path) -> None:
    ini = tmp_path / "ennoia.ini"
    ini.write_text("[ennoia]\nllm =\nschema =\n[env]\nOPENAI_API_KEY =\n")

    import os

    load_ini(ini)
    assert "ENNOIA_LLM" not in os.environ
    assert "ENNOIA_SCHEMA" not in os.environ
    assert "OPENAI_API_KEY" not in os.environ


def test_load_ini_ignores_missing_sections(tmp_path: Path) -> None:
    only_env = tmp_path / "a.ini"
    only_env.write_text("[env]\nEXTRA_CUSTOM = value\n")
    only_ennoia = tmp_path / "b.ini"
    only_ennoia.write_text("[ennoia]\nllm = x:y\n")
    empty = tmp_path / "c.ini"
    empty.write_text("")

    import os

    assert load_ini(only_env) is True
    assert os.environ["EXTRA_CUSTOM"] == "value"

    assert load_ini(only_ennoia) is True
    assert os.environ["ENNOIA_LLM"] == "x:y"

    assert load_ini(empty) is True


# ---------------------------------------------------------------------------
# require_option
# ---------------------------------------------------------------------------


def test_require_option_returns_value_when_set() -> None:
    assert require_option("abc", "--store", "store") == "abc"


def test_require_option_raises_when_none() -> None:
    with pytest.raises(typer.BadParameter) as excinfo:
        require_option(None, "--store", "store")
    message = str(excinfo.value)
    assert "--store" in message
    assert "ENNOIA_STORE" in message
    assert "ennoia.ini" in message


def test_try_command_body_error_when_schema_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--schema is no longer a Typer-required option; confirm the body
    still surfaces a clean ``typer.BadParameter`` when neither flag nor
    env nor INI provides it."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "doc.txt").write_text("body")
    result = CliRunner().invoke(cli_main.app, ["--no-config", "try", "doc.txt"])
    assert result.exit_code != 0
    output = _strip_ansi(result.output)
    assert "--schema" in output
    assert "ENNOIA_SCHEMA" in output


# ---------------------------------------------------------------------------
# write_template / `ennoia init`
# ---------------------------------------------------------------------------


def test_write_template_creates_file(tmp_path: Path) -> None:
    dest = tmp_path / "ennoia.ini"
    out = write_template(dest, force=False)
    assert out == dest
    assert dest.read_text(encoding="utf-8") == INI_TEMPLATE


def test_write_template_refuses_overwrite_without_force(tmp_path: Path) -> None:
    dest = tmp_path / "ennoia.ini"
    dest.write_text("existing")
    with pytest.raises(typer.BadParameter, match="already exists"):
        write_template(dest, force=False)
    assert dest.read_text() == "existing"


def test_write_template_overwrites_with_force(tmp_path: Path) -> None:
    dest = tmp_path / "ennoia.ini"
    dest.write_text("existing")
    write_template(dest, force=True)
    assert dest.read_text(encoding="utf-8") == INI_TEMPLATE


def test_init_command_default_path_is_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli_main.app, ["init"])
    assert result.exit_code == 0, result.output
    written = tmp_path / INI_FILENAME
    assert written.read_text(encoding="utf-8") == INI_TEMPLATE
    assert "Wrote" in result.output


def test_init_command_custom_path_and_force(tmp_path: Path) -> None:
    dest = tmp_path / "custom.ini"
    dest.write_text("stale")
    result = CliRunner().invoke(cli_main.app, ["init", "--path", str(dest), "--force"])
    assert result.exit_code == 0, result.output
    assert dest.read_text(encoding="utf-8") == INI_TEMPLATE


def test_init_command_refuses_existing_file(tmp_path: Path) -> None:
    dest = tmp_path / "ennoia.ini"
    dest.write_text("keep me")
    result = CliRunner().invoke(cli_main.app, ["init", "--path", str(dest)])
    assert result.exit_code != 0
    assert "already exists" in result.output
    assert dest.read_text() == "keep me"


# ---------------------------------------------------------------------------
# Root callback + envvar= hookups
# ---------------------------------------------------------------------------


class _FakeLLM(LLMAdapter):
    async def complete_json(self, prompt: str) -> dict[str, object]:
        return {"category": "legal", "extraction_confidence": 0.9}

    async def complete_text(self, prompt: str) -> str:
        return "body <extraction_confidence>0.9</extraction_confidence>"


class _FakeEmbedding(EmbeddingAdapter):
    async def embed(self, text: str) -> list[float]:
        return [1.0, 0.0]


_SCHEMA_MODULE = '''
from typing import Literal
from ennoia import BaseStructure, BaseSemantic

class Doc(BaseStructure):
    """Extract doc metadata."""
    category: Literal["legal", "medical"]

class Summary(BaseSemantic):
    """What is the document about?"""
'''


def _write_schema(tmp_path: Path) -> Path:
    schema = tmp_path / "schemas.py"
    schema.write_text(_SCHEMA_MODULE)
    return schema


def _patch_adapters(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    captured: dict[str, object] = {}

    def fake_llm(spec: str) -> _FakeLLM:
        captured["llm_spec"] = spec
        return _FakeLLM()

    def fake_embedding(spec: str) -> _FakeEmbedding:
        captured["embedding_spec"] = spec
        return _FakeEmbedding()

    monkeypatch.setattr(cli_main, "parse_llm_spec", fake_llm)
    monkeypatch.setattr(cli_main, "parse_embedding_spec", fake_embedding)
    return captured


def test_root_callback_loads_ini_from_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    schema = _write_schema(tmp_path)
    (tmp_path / "doc.txt").write_text("body")
    (tmp_path / INI_FILENAME).write_text(f"[ennoia]\nllm = from-ini:model\nschema = {schema}\n")
    captured = _patch_adapters(monkeypatch)

    result = CliRunner().invoke(cli_main.app, ["try", "doc.txt"])
    assert result.exit_code == 0, result.output
    assert captured["llm_spec"] == "from-ini:model"


def test_root_callback_no_config_skips_ini(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    schema = _write_schema(tmp_path)
    (tmp_path / "doc.txt").write_text("body")
    (tmp_path / INI_FILENAME).write_text(f"[ennoia]\nllm = from-ini:model\nschema = {schema}\n")
    _patch_adapters(monkeypatch)

    # Without the INI, --schema is required and nothing sets it → must fail.
    result = CliRunner().invoke(cli_main.app, ["--no-config", "try", "doc.txt"])
    assert result.exit_code != 0
    assert "schema" in result.output.lower()


def test_root_callback_custom_config_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    schema = _write_schema(tmp_path)
    (tmp_path / "doc.txt").write_text("body")
    alt = tmp_path / "alt.ini"
    alt.write_text(f"[ennoia]\nllm = alt:model\nschema = {schema}\n")
    captured = _patch_adapters(monkeypatch)

    result = CliRunner().invoke(cli_main.app, ["--config", str(alt), "try", "doc.txt"])
    assert result.exit_code == 0, result.output
    assert captured["llm_spec"] == "alt:model"


def test_cli_flag_overrides_ini_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    schema = _write_schema(tmp_path)
    (tmp_path / "doc.txt").write_text("body")
    (tmp_path / INI_FILENAME).write_text(f"[ennoia]\nllm = from-ini:model\nschema = {schema}\n")
    captured = _patch_adapters(monkeypatch)

    result = CliRunner().invoke(
        cli_main.app,
        ["try", "doc.txt", "--llm", "explicit:model"],
    )
    assert result.exit_code == 0, result.output
    assert captured["llm_spec"] == "explicit:model"


def test_env_section_preloads_before_adapter_init(tmp_path: Path) -> None:
    """The ``[env]`` section populates provider env vars before an
    adapter is constructed, so ``OpenAIAdapter.__init__`` picks the key
    up from ``os.environ.get`` — the exit criterion from
    ``.ref/TASK.md``."""
    from ennoia.adapters.llm.openai import OpenAIAdapter

    ini = tmp_path / "ennoia.ini"
    ini.write_text("[env]\nOPENAI_API_KEY = sk-from-ini\n")

    load_ini(ini)
    adapter = OpenAIAdapter(model="gpt-4o-mini")
    assert adapter.api_key == "sk-from-ini"
