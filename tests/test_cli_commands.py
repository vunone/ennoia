"""End-to-end CLI tests using typer.testing.CliRunner and a fake LLM plugin."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("pandas")

from typer.testing import CliRunner

import ennoia.cli.main as cli_main
from ennoia.adapters.embedding import EmbeddingAdapter
from ennoia.adapters.llm import LLMAdapter

_SCHEMA_MODULE = '''
from typing import Literal
from ennoia import BaseStructure, BaseSemantic

class Doc(BaseStructure):
    """Extract doc metadata."""
    category: Literal["legal", "medical"]

class Summary(BaseSemantic):
    """What is the document about?"""
'''


class _FakeLLM(LLMAdapter):
    async def complete_json(self, prompt: str) -> dict[str, object]:
        return {"category": "legal", "extraction_confidence": 0.9}

    async def complete_text(self, prompt: str) -> str:
        return "A legal document. <extraction_confidence>0.9</extraction_confidence>"


class _FakeEmbedding(EmbeddingAdapter):
    async def embed(self, text: str) -> list[float]:
        return [1.0, 0.0]


@pytest.fixture(autouse=True)
def _isolate_cli_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Hermetic cwd + clean env so the repo-root ``ennoia.ini`` never leaks in.

    The CLI auto-loads ``./ennoia.ini`` from the working directory; running
    pytest from the repo root would otherwise pick up the project's own dev
    config (which points at a non-existent ``./schema.py``) and fail flag
    validation in any test that doesn't pass ``--schema`` explicitly.
    """
    isolated = tmp_path / "_cli_cwd"
    isolated.mkdir()
    monkeypatch.chdir(isolated)
    for name in (
        "ENNOIA_LLM",
        "ENNOIA_EMBEDDING",
        "ENNOIA_STORE",
        "ENNOIA_SCHEMA",
        "ENNOIA_COLLECTION",
        "ENNOIA_QDRANT_URL",
        "ENNOIA_QDRANT_API_KEY",
        "ENNOIA_PG_DSN",
        "ENNOIA_HOST",
        "ENNOIA_PORT",
        "ENNOIA_TRANSPORT",
        "ENNOIA_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


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
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Extractor[BaseStructure]: Doc" in result.output
    assert "category" in result.output
    assert "confidence: 0.90" in result.output
    assert "Extractor[BaseSemantic]: Summary" in result.output


def test_try_command_rejects_unknown_embedding_flag(
    patched_adapters: None, schema_file: Path, tmp_path: Path
) -> None:
    # ``--embedding`` was removed in v0.3.0 — confirm typer rejects it.
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
    assert result.exit_code != 0
    assert "--embedding" in result.output or "No such option" in result.output


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


def test_index_then_search_respects_collection_flag(
    patched_adapters: None, schema_file: Path, tmp_path: Path
) -> None:
    # Two pipelines under one store root with different --collection values
    # must not see each other's documents.
    docs_a = tmp_path / "invoices"
    docs_a.mkdir()
    (docs_a / "inv_1.txt").write_text("invoice body")
    docs_b = tmp_path / "emails"
    docs_b.mkdir()
    (docs_b / "email_1.txt").write_text("email body")
    store_dir = tmp_path / "shared"

    runner = CliRunner()
    assert (
        runner.invoke(
            cli_main.app,
            [
                "index",
                str(docs_a),
                "--schema",
                str(schema_file),
                "--store",
                str(store_dir),
                "--collection",
                "invoices",
                "--llm",
                "fake:m",
                "--embedding",
                "fake:m",
            ],
        ).exit_code
        == 0
    )
    assert (
        runner.invoke(
            cli_main.app,
            [
                "index",
                str(docs_b),
                "--schema",
                str(schema_file),
                "--store",
                str(store_dir),
                "--collection",
                "emails",
                "--llm",
                "fake:m",
                "--embedding",
                "fake:m",
            ],
        ).exit_code
        == 0
    )

    # Search the invoices collection — result must not mention email_1.
    srch = runner.invoke(
        cli_main.app,
        [
            "search",
            "anything",
            "--store",
            str(store_dir),
            "--collection",
            "invoices",
            "--llm",
            "fake:m",
            "--embedding",
            "fake:m",
        ],
    )
    assert srch.exit_code == 0, srch.output
    assert "email_1.txt" not in srch.output
    assert (store_dir / "invoices").is_dir()
    assert (store_dir / "emails").is_dir()


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


# ---------------------------------------------------------------------------
# CLI error paths — the happy-path tests above cover success; these cover the
# typer.BadParameter branches so the user surfaces clean errors instead of
# Python tracebacks.
# ---------------------------------------------------------------------------


def testload_schemas_rejects_missing_file(tmp_path: Path) -> None:
    import typer

    with pytest.raises(typer.BadParameter, match="Schema file not found"):
        cli_main.load_schemas(tmp_path / "nope.py")


def testload_schemas_rejects_unimportable_path(tmp_path: Path) -> None:
    # A file without a recognised loader suffix: ``spec_from_file_location``
    # returns ``None`` because it can't pick a loader for ``.txt``.
    import typer

    bogus = tmp_path / "not_a_module.txt"
    bogus.write_text("# not python")
    with pytest.raises(typer.BadParameter, match="Cannot import"):
        cli_main.load_schemas(bogus)


def testload_schemas_rejects_module_with_no_schemas(tmp_path: Path) -> None:
    import typer

    empty = tmp_path / "empty.py"
    empty.write_text("x = 1\n")
    with pytest.raises(typer.BadParameter, match="No BaseStructure/BaseSemantic/BaseCollection"):
        cli_main.load_schemas(empty)


def testload_schemas_ignores_unrelated_classes(tmp_path: Path) -> None:
    # A module with helper classes alongside schemas must return only the
    # schemas — covers the ``isinstance`` filter branch.
    schema_file = tmp_path / "mixed.py"
    schema_file.write_text(
        """
from ennoia import BaseStructure

class Helper:
    pass

class Doc(BaseStructure):
    \"\"\"Doc.\"\"\"
    value: str
"""
    )
    classes = cli_main.load_schemas(schema_file)
    names = [c.__name__ for c in classes]
    assert names == ["Doc"]


def test_parse_filters_rejects_missing_equals() -> None:
    import typer

    with pytest.raises(typer.BadParameter, match="Filter must be"):
        cli_main._parse_filters(["no_equals_sign"])


def test_parse_filters_unknown_suffix_keeps_field_intact() -> None:
    # ``split_filter_key`` leaves unknown suffixes as part of the field name,
    # so ``foo__notreal`` parses as field ``foo__notreal`` + op ``eq``.
    assert cli_main._parse_filters(["field__notreal=x"]) == {"field__notreal": "x"}


def test_parse_filters_eq_drops_operator_suffix() -> None:
    assert cli_main._parse_filters(["category=legal"]) == {"category": "legal"}


def test_parse_filters_preserves_explicit_suffix() -> None:
    assert cli_main._parse_filters(["n__gt=5"]) == {"n__gt": "5"}


def test_coerce_filter_value_is_null_parses_bool() -> None:
    assert cli_main._coerce_filter_value("true", "is_null") is True
    assert cli_main._coerce_filter_value("false", "is_null") is False


def test_coerce_filter_value_list_operators_split_on_comma() -> None:
    assert cli_main._coerce_filter_value("a, b, c", "in") == ["a", "b", "c"]
    assert cli_main._coerce_filter_value("x,y", "contains_all") == ["x", "y"]
    assert cli_main._coerce_filter_value("x,y", "contains_any") == ["x", "y"]


def test_coerce_filter_value_scalar_operator_passes_through() -> None:
    assert cli_main._coerce_filter_value("legal", "eq") == "legal"


def test_index_rejects_non_directory(
    patched_adapters: None, schema_file: Path, tmp_path: Path
) -> None:
    not_a_dir = tmp_path / "regular-file.txt"
    not_a_dir.write_text("hello")

    runner = CliRunner()
    result = runner.invoke(
        cli_main.app,
        [
            "index",
            str(not_a_dir),
            "--schema",
            str(schema_file),
            "--store",
            str(tmp_path / "store"),
            "--llm",
            "fake:m",
            "--embedding",
            "fake:m",
        ],
    )
    assert result.exit_code != 0
    assert "Not a directory" in result.output


def test_index_skips_subdirectories(
    patched_adapters: None, schema_file: Path, tmp_path: Path
) -> None:
    # ``index`` iterates the directory and must ignore subdirectories so
    # nesting doesn't crash the run.
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.txt").write_text("body")
    (docs_dir / "subdir").mkdir()

    runner = CliRunner()
    result = runner.invoke(
        cli_main.app,
        [
            "index",
            str(docs_dir),
            "--schema",
            str(schema_file),
            "--store",
            str(tmp_path / "store"),
            "--llm",
            "fake:m",
            "--embedding",
            "fake:m",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Indexed 1 document(s)" in result.output


def test_try_command_reports_missing_document(
    patched_adapters: None, schema_file: Path, tmp_path: Path
) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli_main.app,
        [
            "try",
            str(tmp_path / "missing.txt"),
            "--schema",
            str(schema_file),
            "--llm",
            "fake:m",
        ],
    )
    assert result.exit_code != 0
    assert "Document not found" in result.output


def test_search_reports_no_hits_when_store_empty(
    patched_adapters: None, schema_file: Path, tmp_path: Path
) -> None:
    # Build an empty store directory — search against it returns zero hits.
    store_dir = tmp_path / "idx"
    store_dir.mkdir()

    runner = CliRunner()
    result = runner.invoke(
        cli_main.app,
        [
            "search",
            "any query",
            "--store",
            str(store_dir),
            "--llm",
            "fake:m",
            "--embedding",
            "fake:m",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "No hits." in result.output


def test_index_command_without_explicit_flags_when_ini_present(
    patched_adapters: None,
    schema_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Full INI happy-path: all required flags (schema, store, llm, embedding)
    # come from ennoia.ini; only the positional directory is passed.
    for name in (
        "ENNOIA_LLM",
        "ENNOIA_EMBEDDING",
        "ENNOIA_STORE",
        "ENNOIA_SCHEMA",
        "ENNOIA_COLLECTION",
    ):
        monkeypatch.delenv(name, raising=False)

    monkeypatch.chdir(tmp_path)
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.txt").write_text("body 1")
    store_dir = tmp_path / "idx"

    (tmp_path / "ennoia.ini").write_text(
        f"[ennoia]\nschema = {schema_file}\nstore = {store_dir}\nllm = fake:m\nembedding = fake:m\n"
    )

    result = CliRunner().invoke(cli_main.app, ["index", str(docs_dir)])
    assert result.exit_code == 0, result.output
    assert "Indexed 1 document(s)" in result.output


def test_search_renders_hits(patched_adapters: None, schema_file: Path, tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.txt").write_text("body-a")
    store_dir = tmp_path / "idx"

    runner = CliRunner()
    runner.invoke(
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
    result = runner.invoke(
        cli_main.app,
        [
            "search",
            "legal",
            "--store",
            str(store_dir),
            "--llm",
            "fake:m",
            "--embedding",
            "fake:m",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Results" in result.output
    assert "a.txt" in result.output
