"""Tests for the ``ennoia craft`` CLI subcommand and its helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import ennoia.cli.main as cli_main
from ennoia.adapters.llm.base import LLMAdapter
from ennoia.craft import (
    CraftError,
    CraftLLMError,
    CraftParseError,
    CraftValidationError,
    extract_python_block,
    run_craft_loop,
    validate_schema_file,
)
from ennoia.craft.prompts import (
    build_retry_prompt,
    build_system_prompt,
    build_user_prompt,
)
from ennoia.prompts import load_prompt

_VALID_SCHEMA = '''\
from typing import Literal
from ennoia import BaseStructure


class Product(BaseStructure):
    """Extract product catalogue metadata."""

    category: Literal["electronics", "apparel"]
    price_usd: float
'''

_VALID_SCHEMA_V2 = '''\
from typing import Literal
from ennoia import BaseStructure


class Product(BaseStructure):
    """Extract product catalogue metadata (improved)."""

    category: Literal["electronics", "apparel", "home"]
    price_usd: float
    color: str
'''

_SYNTAX_ERROR_SCHEMA = '''\
from ennoia import BaseStructure


class Product(BaseStructure)   # missing colon
    """Extract catalogue metadata."""

    price_usd: float
'''

_IMPORT_ERROR_SCHEMA = """\
from ennoia import DefinitelyNotAThing


class Whatever:
    pass
"""

_NO_SCHEMAS_MODULE = """\
VALUE = 42
"""


class _FakeCraftLLM(LLMAdapter):
    """Scripted LLM — each ``complete_text`` call pops the next reply.

    Replies may be strings (returned verbatim) or exceptions (raised).
    All incoming prompts are captured on :attr:`prompts` so tests can
    assert substring inclusion.
    """

    def __init__(self, replies: list[str | Exception]) -> None:
        self._replies = list(replies)
        self.prompts: list[str] = []

    async def complete_json(self, prompt: str) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError("craft never calls complete_json")

    async def complete_text(self, prompt: str) -> str:
        self.prompts.append(prompt)
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _wrap(source: str, lang: str = "python") -> str:
    fence = f"```{lang}" if lang else "```"
    return f"{fence}\n{source}```\n"


# ---------------------------------------------------------------------------
# extract_python_block
# ---------------------------------------------------------------------------


def test_extract_python_block_single_tagged() -> None:
    out = extract_python_block(_wrap("print('hi')\n"))
    assert out == "print('hi')"


def test_extract_python_block_single_bare() -> None:
    out = extract_python_block(_wrap("x = 1\n", lang=""))
    assert out == "x = 1"


def test_extract_python_block_py_alias() -> None:
    out = extract_python_block(_wrap("y = 2\n", lang="py"))
    assert out == "y = 2"


def test_extract_python_block_python3_alias() -> None:
    out = extract_python_block(_wrap("z = 3\n", lang="python3"))
    assert out == "z = 3"


def test_extract_python_block_ignores_non_python_lang() -> None:
    mixed = "```json\n{}\n```\n" + _wrap("ok = True\n")
    assert extract_python_block(mixed) == "ok = True"


def test_extract_python_block_multi_block_concatenates_tagged_only() -> None:
    # Two tagged blocks + a bare block — the bare one is dropped when any
    # tagged block is present.
    reply = (
        _wrap("import os\n")
        + "some prose\n"
        + _wrap("x = os.getcwd()\n")
        + _wrap("y = 2\n", lang="")
    )
    out = extract_python_block(reply)
    assert out == "import os\n\nx = os.getcwd()"


def test_extract_python_block_multi_bare_concatenates() -> None:
    reply = _wrap("a = 1\n", lang="") + _wrap("b = 2\n", lang="")
    out = extract_python_block(reply)
    assert out == "a = 1\n\nb = 2"


def test_extract_python_block_indented_fence_strips_indent() -> None:
    reply = "    ```python\n    x = 1\n    y = 2\n    ```\n"
    assert extract_python_block(reply) == "x = 1\ny = 2"


def test_extract_python_block_crlf_normalised() -> None:
    reply = "```python\r\nvalue = 1\r\n```\r\n"
    assert extract_python_block(reply) == "value = 1"


def test_extract_python_block_raises_when_missing() -> None:
    with pytest.raises(CraftParseError, match="no python code block"):
        extract_python_block("just prose, no fences here")


def test_extract_python_block_raises_when_only_non_python() -> None:
    with pytest.raises(CraftParseError):
        extract_python_block("```json\n{}\n```\n")


# ---------------------------------------------------------------------------
# validate_schema_file
# ---------------------------------------------------------------------------


def test_validate_schema_file_accepts_valid_module(tmp_path: Path) -> None:
    path = tmp_path / "schema.py"
    path.write_text(_VALID_SCHEMA)
    assert validate_schema_file(path) is None


def test_validate_schema_file_reports_syntax_error(tmp_path: Path) -> None:
    path = tmp_path / "schema.py"
    path.write_text(_SYNTAX_ERROR_SCHEMA)
    result = validate_schema_file(path)
    assert result is not None
    assert "SyntaxError" in result


def test_validate_schema_file_reports_import_error(tmp_path: Path) -> None:
    path = tmp_path / "schema.py"
    path.write_text(_IMPORT_ERROR_SCHEMA)
    result = validate_schema_file(path)
    assert result is not None
    assert "ImportError" in result or "cannot import name" in result


def test_validate_schema_file_reports_missing_schemas(tmp_path: Path) -> None:
    path = tmp_path / "schema.py"
    path.write_text(_NO_SCHEMAS_MODULE)
    result = validate_schema_file(path)
    assert result is not None
    assert "No schema subclasses" in result


def test_validate_schema_file_rejects_unimportable_suffix(tmp_path: Path) -> None:
    # ``spec_from_file_location`` returns None for unrecognised suffixes,
    # which exercises the defensive early-return branch.
    path = tmp_path / "schema.txt"
    path.write_text(_VALID_SCHEMA)
    result = validate_schema_file(path)
    assert result is not None
    assert "Cannot build an import spec" in result


def test_validate_schema_file_cleans_sys_modules(tmp_path: Path) -> None:
    before = {k for k in sys.modules if k.startswith("_ennoia_craft_probe_")}
    for idx in range(3):
        path = tmp_path / f"schema_{idx}.py"
        path.write_text(_VALID_SCHEMA)
        assert validate_schema_file(path) is None
    after = {k for k in sys.modules if k.startswith("_ennoia_craft_probe_")}
    assert before == after


def test_validate_schema_file_accepts_semantic_and_collection(tmp_path: Path) -> None:
    path = tmp_path / "schema.py"
    path.write_text(
        '''\
from ennoia import BaseSemantic, BaseCollection


class Summary(BaseSemantic):
    """What is this document about?"""


class Mention(BaseCollection):
    """Every named entity mentioned."""

    name: str
'''
    )
    assert validate_schema_file(path) is None


# ---------------------------------------------------------------------------
# prompt builders
# ---------------------------------------------------------------------------


def test_load_prompt_returns_craft_guide() -> None:
    text = load_prompt("craft")
    assert len(text) > 500
    for marker in ("BaseStructure", "BaseSemantic", "BaseCollection"):
        assert marker in text


def test_build_system_prompt_contains_guide_and_reminder() -> None:
    prompt = build_system_prompt()
    assert "BaseStructure" in prompt
    assert "Formatting reminder" in prompt


def test_build_user_prompt_without_existing_schema() -> None:
    prompt = build_user_prompt(
        task="filter by color", document="A product page.", existing_schema=None
    )
    assert "# Task" in prompt
    assert "filter by color" in prompt
    assert "# Document sample" in prompt
    assert "A product page." in prompt
    assert "# Current schema" not in prompt


def test_build_user_prompt_with_existing_schema() -> None:
    prompt = build_user_prompt(
        task="filter by color",
        document="A product page.",
        existing_schema=_VALID_SCHEMA,
    )
    assert "# Current schema" in prompt
    assert "class Product" in prompt


def test_build_user_prompt_ignores_blank_existing_schema() -> None:
    prompt = build_user_prompt(task="t", document="d", existing_schema="   \n  ")
    assert "# Current schema" not in prompt


def test_build_retry_prompt_includes_all_sections() -> None:
    user = build_user_prompt(task="t", document="d", existing_schema=None)
    retry = build_retry_prompt(previous_user_prompt=user, previous_reply="bad reply", error="boom")
    assert user in retry
    assert "# Your previous reply" in retry
    assert "bad reply" in retry
    assert "# It failed with" in retry
    assert "boom" in retry


# ---------------------------------------------------------------------------
# run_craft_loop
# ---------------------------------------------------------------------------


async def test_run_craft_loop_happy_path(tmp_path: Path) -> None:
    llm = _FakeCraftLLM([_wrap(_VALID_SCHEMA)])
    output = tmp_path / "schema.py"
    await run_craft_loop(
        llm=llm,
        task="filter products by category",
        document="sample product doc",
        output_path=output,
        existing_schema=None,
    )
    assert output.exists()
    assert "class Product" in output.read_text()
    assert len(llm.prompts) == 1


async def test_run_craft_loop_improves_existing_schema(tmp_path: Path) -> None:
    output = tmp_path / "schema.py"
    output.write_text(_VALID_SCHEMA)
    llm = _FakeCraftLLM([_wrap(_VALID_SCHEMA_V2)])
    await run_craft_loop(
        llm=llm,
        task="add a color filter",
        document="product page body",
        output_path=output,
        existing_schema=output.read_text(),
    )
    assert "color: str" in output.read_text()
    assert _VALID_SCHEMA in llm.prompts[0]


async def test_run_craft_loop_retries_on_missing_code_block(tmp_path: Path) -> None:
    llm = _FakeCraftLLM(["no fences just prose", _wrap(_VALID_SCHEMA)])
    output = tmp_path / "schema.py"
    await run_craft_loop(
        llm=llm,
        task="t",
        document="d",
        output_path=output,
        existing_schema=None,
    )
    assert output.exists()
    assert len(llm.prompts) == 2
    assert "did not contain a python code block" in llm.prompts[1]


async def test_run_craft_loop_retries_on_syntax_error(tmp_path: Path) -> None:
    llm = _FakeCraftLLM([_wrap(_SYNTAX_ERROR_SCHEMA), _wrap(_VALID_SCHEMA)])
    output = tmp_path / "schema.py"
    await run_craft_loop(
        llm=llm,
        task="t",
        document="d",
        output_path=output,
        existing_schema=None,
    )
    assert output.exists()
    assert "SyntaxError" in llm.prompts[1]


async def test_run_craft_loop_retries_on_import_error(tmp_path: Path) -> None:
    llm = _FakeCraftLLM([_wrap(_IMPORT_ERROR_SCHEMA), _wrap(_VALID_SCHEMA)])
    output = tmp_path / "schema.py"
    await run_craft_loop(
        llm=llm,
        task="t",
        document="d",
        output_path=output,
        existing_schema=None,
    )
    assert output.exists()
    assert "ImportError" in llm.prompts[1] or "cannot import name" in llm.prompts[1]


async def test_run_craft_loop_exhausts_retries(tmp_path: Path) -> None:
    llm = _FakeCraftLLM(
        [_wrap(_SYNTAX_ERROR_SCHEMA), _wrap(_SYNTAX_ERROR_SCHEMA), _wrap(_SYNTAX_ERROR_SCHEMA)]
    )
    output = tmp_path / "schema.py"
    with pytest.raises(CraftValidationError, match="SyntaxError"):
        await run_craft_loop(
            llm=llm,
            task="t",
            document="d",
            output_path=output,
            existing_schema=None,
        )
    # Partial output is left for inspection.
    assert output.exists()
    assert len(llm.prompts) == 3


async def test_run_craft_loop_exhausts_retries_on_missing_block(tmp_path: Path) -> None:
    llm = _FakeCraftLLM(["prose", "more prose", "still no code"])
    output = tmp_path / "schema.py"
    with pytest.raises(CraftValidationError):
        await run_craft_loop(
            llm=llm,
            task="t",
            document="d",
            output_path=output,
            existing_schema=None,
            max_retries=2,
        )
    # No code block ever parsed → file never written.
    assert not output.exists()


async def test_run_craft_loop_zero_retries_single_attempt(tmp_path: Path) -> None:
    llm = _FakeCraftLLM([_wrap(_SYNTAX_ERROR_SCHEMA)])
    output = tmp_path / "schema.py"
    with pytest.raises(CraftValidationError):
        await run_craft_loop(
            llm=llm,
            task="t",
            document="d",
            output_path=output,
            existing_schema=None,
            max_retries=0,
        )
    assert len(llm.prompts) == 1


async def test_run_craft_loop_wraps_provider_exception(tmp_path: Path) -> None:
    llm = _FakeCraftLLM([RuntimeError("context_length_exceeded: 1M > 128k")])
    output = tmp_path / "schema.py"
    with pytest.raises(CraftLLMError, match="context"):
        await run_craft_loop(
            llm=llm,
            task="t",
            document="d" * 100,
            output_path=output,
            existing_schema=None,
        )


async def test_run_craft_loop_reports_progress(tmp_path: Path) -> None:
    events: list[tuple[int, str]] = []
    llm = _FakeCraftLLM([_wrap(_VALID_SCHEMA)])
    await run_craft_loop(
        llm=llm,
        task="t",
        document="d",
        output_path=tmp_path / "schema.py",
        existing_schema=None,
        on_attempt=lambda i, s: events.append((i, s)),
    )
    assert (0, "calling LLM") in events
    assert (0, "validating schema") in events


async def test_run_craft_loop_raises_validation_error_when_last_error_empty(
    tmp_path: Path,
) -> None:
    # Zero retries + an empty reply list triggers the final fallback string path.
    # We drive it via ``max_retries = -1`` so the loop body never runs.
    llm = _FakeCraftLLM([])
    output = tmp_path / "schema.py"
    with pytest.raises(CraftValidationError, match="exhausted retries"):
        await run_craft_loop(
            llm=llm,
            task="t",
            document="d",
            output_path=output,
            existing_schema=None,
            max_retries=-1,
        )


# ---------------------------------------------------------------------------
# Typer command
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_llm() -> _FakeCraftLLM:
    return _FakeCraftLLM([])


@pytest.fixture()
def patched_craft_llm(monkeypatch: pytest.MonkeyPatch, fake_llm: _FakeCraftLLM) -> _FakeCraftLLM:
    monkeypatch.setattr(cli_main, "parse_llm_spec", lambda spec: fake_llm)
    return fake_llm


def test_craft_cli_happy_path(patched_craft_llm: _FakeCraftLLM, tmp_path: Path) -> None:
    doc = tmp_path / "product.txt"
    doc.write_text("A product page.")
    output = tmp_path / "schema.py"
    patched_craft_llm._replies = [_wrap(_VALID_SCHEMA)]

    runner = CliRunner()
    result = runner.invoke(
        cli_main.app,
        [
            "craft",
            str(doc),
            "--output",
            str(output),
            "--llm",
            "fake:model",
            "--task",
            "filter by category",
        ],
    )
    assert result.exit_code == 0, result.output
    assert output.exists()
    assert "wrote draft schema" in result.output
    # The prototype-only warning must fire both before and after the run
    # so the user can't miss that the output needs manual review.
    assert "Prototype only" in result.output
    assert "Review the draft" in result.output
    # The crafted file must also be loadable via the downstream CLI entrypoint.
    assert [c.__name__ for c in cli_main.load_schemas(output)] == ["Product"]


def test_craft_cli_improves_existing(patched_craft_llm: _FakeCraftLLM, tmp_path: Path) -> None:
    doc = tmp_path / "product.txt"
    doc.write_text("A product page.")
    output = tmp_path / "schema.py"
    output.write_text(_VALID_SCHEMA)
    patched_craft_llm._replies = [_wrap(_VALID_SCHEMA_V2)]

    runner = CliRunner()
    result = runner.invoke(
        cli_main.app,
        [
            "craft",
            str(doc),
            "--output",
            str(output),
            "--llm",
            "fake:model",
            "--task",
            "extend with color",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "color: str" in output.read_text()
    assert _VALID_SCHEMA in patched_craft_llm.prompts[0]


def test_craft_cli_reports_missing_document(
    patched_craft_llm: _FakeCraftLLM, tmp_path: Path
) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli_main.app,
        [
            "craft",
            str(tmp_path / "missing.txt"),
            "--output",
            str(tmp_path / "schema.py"),
            "--llm",
            "fake:m",
            "--task",
            "x",
        ],
    )
    assert result.exit_code != 0
    assert "Document not found" in result.output


def test_craft_cli_exhausts_retries(patched_craft_llm: _FakeCraftLLM, tmp_path: Path) -> None:
    doc = tmp_path / "product.txt"
    doc.write_text("doc")
    output = tmp_path / "schema.py"
    patched_craft_llm._replies = [
        _wrap(_SYNTAX_ERROR_SCHEMA),
        _wrap(_SYNTAX_ERROR_SCHEMA),
        _wrap(_SYNTAX_ERROR_SCHEMA),
    ]

    runner = CliRunner()
    result = runner.invoke(
        cli_main.app,
        [
            "craft",
            str(doc),
            "--output",
            str(output),
            "--llm",
            "fake:m",
            "--task",
            "x",
        ],
    )
    assert result.exit_code == 1
    assert "failed" in result.output
    assert "partial output" in result.output
    assert output.exists()


def test_craft_cli_wraps_llm_error(patched_craft_llm: _FakeCraftLLM, tmp_path: Path) -> None:
    doc = tmp_path / "product.txt"
    doc.write_text("doc")
    output = tmp_path / "schema.py"
    patched_craft_llm._replies = [RuntimeError("context_length_exceeded")]

    runner = CliRunner()
    result = runner.invoke(
        cli_main.app,
        [
            "craft",
            str(doc),
            "--output",
            str(output),
            "--llm",
            "fake:m",
            "--task",
            "x",
        ],
    )
    assert result.exit_code == 1
    assert "context" in result.output


def test_craft_cli_respects_max_retries_flag(
    patched_craft_llm: _FakeCraftLLM, tmp_path: Path
) -> None:
    doc = tmp_path / "product.txt"
    doc.write_text("doc")
    output = tmp_path / "schema.py"
    # One bad, one good — with max_retries=1 we get 2 attempts, so it succeeds.
    patched_craft_llm._replies = [_wrap(_SYNTAX_ERROR_SCHEMA), _wrap(_VALID_SCHEMA)]

    runner = CliRunner()
    result = runner.invoke(
        cli_main.app,
        [
            "craft",
            str(doc),
            "--output",
            str(output),
            "--llm",
            "fake:m",
            "--task",
            "x",
            "--max-retries",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "attempt 1/2" in result.output
    assert "attempt 2/2" in result.output


def test_craft_cli_existing_output_directory_is_ignored(
    patched_craft_llm: _FakeCraftLLM, tmp_path: Path
) -> None:
    # If --output happens to point at a directory path, do not try to read it;
    # fall through to the "no existing schema" branch.
    doc = tmp_path / "product.txt"
    doc.write_text("doc")
    output = tmp_path / "schema_dir"
    output.mkdir()
    patched_craft_llm._replies = [_wrap(_VALID_SCHEMA)]

    runner = CliRunner()
    result = runner.invoke(
        cli_main.app,
        [
            "craft",
            str(doc),
            "--output",
            str(output),
            "--llm",
            "fake:m",
            "--task",
            "x",
        ],
    )
    # output path is a directory → write_text will raise; loop wraps it.
    # This exercises the CraftError branch when output is not a writable file.
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# load_schemas widening — BaseCollection is now recognised.
# ---------------------------------------------------------------------------


def test_load_schemas_accepts_basecollection(tmp_path: Path) -> None:
    path = tmp_path / "schemas.py"
    path.write_text(
        '''\
from ennoia import BaseCollection, BaseStructure


class Meta(BaseStructure):
    """Doc header."""

    title: str


class Mention(BaseCollection):
    """Every mention in the doc."""

    name: str
'''
    )
    names = sorted(c.__name__ for c in cli_main.load_schemas(path))
    assert names == ["Mention", "Meta"]


# ---------------------------------------------------------------------------
# try_command with a BaseCollection — covers the new branch in cli/main.py.
# ---------------------------------------------------------------------------


class _CollectionLLM(LLMAdapter):
    """LLM that returns one batch of entities and then signals ``is_done``."""

    def __init__(self) -> None:
        self._calls = 0

    async def complete_json(self, prompt: str) -> dict[str, Any]:
        self._calls += 1
        if self._calls == 1:
            return {
                "entities_list": [
                    {"name": "Alpha", "extraction_confidence": 0.9},
                    {"name": "Beta", "extraction_confidence": 0.8},
                ],
                "is_done": True,
            }
        return {"entities_list": [], "is_done": True}  # pragma: no cover

    async def complete_text(self, prompt: str) -> str:  # pragma: no cover
        raise NotImplementedError


def test_try_command_handles_basecollection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cli_main, "parse_llm_spec", lambda spec: _CollectionLLM())
    schema = tmp_path / "schemas.py"
    schema.write_text(
        '''\
from ennoia import BaseCollection


class Mention(BaseCollection):
    """Every mention in the doc."""

    name: str

    def get_unique(self) -> str:
        return self.name.casefold()
'''
    )
    doc = tmp_path / "doc.txt"
    doc.write_text("body")

    runner = CliRunner()
    result = runner.invoke(
        cli_main.app,
        [
            "try",
            str(doc),
            "--schema",
            str(schema),
            "--llm",
            "fake:m",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Mention" in result.output
    assert "Alpha" in result.output
    assert "Beta" in result.output


def test_try_command_handles_empty_collection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _EmptyLLM(LLMAdapter):
        async def complete_json(self, prompt: str) -> dict[str, Any]:
            return {"entities_list": [], "is_done": True}

        async def complete_text(self, prompt: str) -> str:  # pragma: no cover
            raise NotImplementedError

    monkeypatch.setattr(cli_main, "parse_llm_spec", lambda spec: _EmptyLLM())
    schema = tmp_path / "schemas.py"
    schema.write_text(
        '''\
from ennoia import BaseCollection


class Mention(BaseCollection):
    """Every mention."""

    name: str
'''
    )
    doc = tmp_path / "doc.txt"
    doc.write_text("body")

    runner = CliRunner()
    result = runner.invoke(
        cli_main.app,
        [
            "try",
            str(doc),
            "--schema",
            str(schema),
            "--llm",
            "fake:m",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "no entities extracted" in result.output


# ---------------------------------------------------------------------------
# Exception hierarchy sanity
# ---------------------------------------------------------------------------


def test_craft_exception_hierarchy() -> None:
    assert issubclass(CraftLLMError, CraftError)
    assert issubclass(CraftValidationError, CraftError)
    assert issubclass(CraftParseError, ValueError)
