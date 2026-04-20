"""Append the ``ennoia_schema`` entrypoint variable to a crafted schema file.

The craft loop produces a Python module with one or more schema classes
but no hint about which of them are DAG roots. Downstream tools
(``ennoia try``, ``ennoia index``, the ``Pipeline`` ``schemas=`` kwarg)
want that information explicit, so this module appends a deterministic
block at the end of the file:

    # `ennoia_schema` lists only the top-level DAG schemas ...
    ennoia_schema = [Metadata, QuestionAnswer, Summary]

The append is idempotent — if the module already declares
``ennoia_schema``, the file is left alone so hand-edited root lists
survive a re-run of ``ennoia craft``.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import sys
import uuid
from pathlib import Path

from ennoia.schema.base import BaseCollection, BaseSemantic, BaseStructure
from ennoia.schema.roots import SchemaClass, identify_roots

__all__ = ["append_entrypoint"]


_ENTRYPOINT_HEADER = (
    "# `ennoia_schema` lists only the top-level DAG schemas — classes that other\n"
    "# schemas don't reference via `Schema.extensions`. Children are reached\n"
    "# transitively at index time through `extend()`, so they don't belong here."
)


def _has_entrypoint(source: str) -> bool:
    """Return True if ``source`` already assigns a module-level ``ennoia_schema``."""
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "ennoia_schema":
                    return True
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "ennoia_schema"
        ):
            return True
    return False


def _discover_schema_classes(module: object) -> list[SchemaClass]:
    """Return module-level schema classes in declaration order."""
    module_name = getattr(module, "__name__", None)
    classes: list[SchemaClass] = []
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ != module_name:
            continue
        if issubclass(obj, BaseStructure | BaseSemantic | BaseCollection):
            classes.append(obj)
    # getmembers sorts alphabetically — re-sort to source line order so the
    # emitted list matches how the user wrote the file.
    classes.sort(key=lambda c: inspect.getsourcelines(c)[1])
    return classes


def append_entrypoint(schema_path: Path) -> None:
    """Append ``ennoia_schema = [...]`` to ``schema_path`` if not already there.

    The schema file is imported under a UUID-suffixed synthetic module
    name so repeated calls in the same process do not collide and the
    import cache stays clean for later CLI commands. The caller is
    responsible for ensuring ``schema_path`` imports cleanly — the craft
    loop runs :func:`validate_schema_file` first.
    """
    source = schema_path.read_text(encoding="utf-8")
    if _has_entrypoint(source):
        return

    module_name = f"_ennoia_entrypoint_probe_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, schema_path)
    assert spec is not None and spec.loader is not None  # validated upstream
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        classes = _discover_schema_classes(module)
        if not classes:
            return
        roots = identify_roots(classes)
    finally:
        sys.modules.pop(module_name, None)

    names = ", ".join(cls.__name__ for cls in roots)
    separator = "" if source.endswith("\n") else "\n"
    block = f"{separator}\n\n{_ENTRYPOINT_HEADER}\nennoia_schema = [{names}]\n"
    schema_path.write_text(source + block, encoding="utf-8")
