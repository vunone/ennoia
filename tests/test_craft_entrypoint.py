"""Tests for the ``ennoia_schema`` append step after craft validation."""

from __future__ import annotations

from pathlib import Path

from ennoia.craft.entrypoint import append_entrypoint


def _write(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")


def test_appends_entrypoint_for_flat_schema(tmp_path: Path) -> None:
    source = '''\
from ennoia import BaseSemantic, BaseStructure


class Metadata(BaseStructure):
    """Metadata."""

    title: str


class Summary(BaseSemantic):
    """Summary?"""
'''
    path = tmp_path / "schema.py"
    _write(path, source)
    append_entrypoint(path)

    out = path.read_text(encoding="utf-8")
    assert out.endswith("ennoia_schema = [Metadata, Summary]\n")
    assert "# `ennoia_schema` lists only the top-level DAG schemas" in out


def test_appends_parent_when_extensions_present(tmp_path: Path) -> None:
    source = '''\
from typing import ClassVar

from ennoia import BaseSemantic, BaseStructure


class Product(BaseStructure):
    """Product."""

    name: str


class Summary(BaseSemantic):
    """Summary?"""


class Page(BaseStructure):
    """Page classifier."""

    kind: str

    class Schema:
        extensions: ClassVar[list[type]] = [Product, Summary]
'''
    path = tmp_path / "schema.py"
    _write(path, source)
    append_entrypoint(path)

    out = path.read_text(encoding="utf-8")
    assert out.endswith("ennoia_schema = [Page]\n")


def test_idempotent_when_entrypoint_already_present(tmp_path: Path) -> None:
    source = '''\
from ennoia import BaseStructure


class Doc(BaseStructure):
    """Doc."""

    title: str


ennoia_schema = [Doc]
'''
    path = tmp_path / "schema.py"
    _write(path, source)
    before = path.read_text(encoding="utf-8")
    append_entrypoint(path)
    assert path.read_text(encoding="utf-8") == before


def test_idempotent_for_annotated_assignment(tmp_path: Path) -> None:
    # Users may write an annotated assignment; the AST check handles both.
    source = '''\
from ennoia import BaseStructure


class Doc(BaseStructure):
    """Doc."""

    title: str


ennoia_schema: list[type] = [Doc]
'''
    path = tmp_path / "schema.py"
    _write(path, source)
    before = path.read_text(encoding="utf-8")
    append_entrypoint(path)
    assert path.read_text(encoding="utf-8") == before


def test_no_op_when_no_schema_classes(tmp_path: Path) -> None:
    # Defensive — the craft loop only calls us after validation, so this
    # branch is normally unreachable, but it keeps the helper total.
    source = "x = 1\n"
    path = tmp_path / "schema.py"
    _write(path, source)
    append_entrypoint(path)
    assert path.read_text(encoding="utf-8") == source


def test_ignores_unrelated_annotated_assignment(tmp_path: Path) -> None:
    # A module-level AnnAssign whose target is NOT ``ennoia_schema`` must
    # not short-circuit the append.
    source = '''\
from ennoia import BaseStructure


CONFIG: dict[str, int] = {"x": 1}


class Doc(BaseStructure):
    """Doc."""

    title: str
'''
    path = tmp_path / "schema.py"
    _write(path, source)
    append_entrypoint(path)
    assert path.read_text(encoding="utf-8").endswith("ennoia_schema = [Doc]\n")


def test_ignores_non_schema_classes(tmp_path: Path) -> None:
    # A module with a helper class alongside schemas — only the schemas
    # land in the emitted ``ennoia_schema`` list.
    source = '''\
from ennoia import BaseStructure


class Helper:
    pass


class Doc(BaseStructure):
    """Doc."""

    title: str
'''
    path = tmp_path / "schema.py"
    _write(path, source)
    append_entrypoint(path)
    assert path.read_text(encoding="utf-8").endswith("ennoia_schema = [Doc]\n")


def test_handles_source_without_trailing_newline(tmp_path: Path) -> None:
    source = 'from ennoia import BaseSemantic\n\n\nclass S(BaseSemantic):\n    """S?"""'
    path = tmp_path / "schema.py"
    _write(path, source)
    append_entrypoint(path)
    out = path.read_text(encoding="utf-8")
    # The appended block starts on a new line regardless of trailing newline.
    assert out.endswith("ennoia_schema = [S]\n")
    assert 'class S(BaseSemantic):\n    """S?"""\n\n' in out
