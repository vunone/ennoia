"""Validate a freshly-crafted schema module by importing it.

The craft retry loop needs a validator that *returns* a formatted error
string on failure (so the error can be fed to the LLM) instead of
raising. This module owns that contract — it is deliberately separate
from :func:`ennoia.cli.main.load_schemas`, which is shaped for CLI
ingress and raises ``typer.BadParameter``.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import sys
import traceback
import uuid
from pathlib import Path

from ennoia.schema.base import BaseCollection, BaseSemantic, BaseStructure

__all__ = ["validate_schema_file"]


_SCHEMA_BASES: tuple[type, ...] = (BaseStructure, BaseSemantic, BaseCollection)


def validate_schema_file(path: Path) -> str | None:
    """Validate a Python schema file by parsing + importing it.

    Returns ``None`` if the file parses, imports, and defines at least
    one subclass of ``BaseStructure`` / ``BaseSemantic`` / ``BaseCollection``.
    Otherwise returns a formatted error string suitable to feed back to
    an LLM in a retry prompt.

    The module is imported under a UUID-suffixed synthetic name and
    popped from :data:`sys.modules` in a ``finally`` — so consecutive
    retries in the same process do not collide, and later CLI commands
    (``ennoia try`` etc.) get a clean import slate.
    """
    source = path.read_text(encoding="utf-8")

    try:
        ast.parse(source, filename=str(path))
    except SyntaxError as err:
        line = err.lineno if err.lineno is not None else "?"
        return f"SyntaxError at line {line}: {err.msg}"

    module_name = f"_ennoia_craft_probe_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        return f"Cannot build an import spec for {path}."

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        try:
            spec.loader.exec_module(module)
        except Exception:
            return traceback.format_exc()

        found = [
            obj
            for _, obj in inspect.getmembers(module, inspect.isclass)
            if obj.__module__ == module.__name__
            and issubclass(obj, _SCHEMA_BASES)
            and obj not in _SCHEMA_BASES
        ]
        if not found:
            return (
                "No schema subclasses defined — the module must define at least one "
                "BaseStructure, BaseSemantic, or BaseCollection subclass."
            )
        return None
    finally:
        sys.modules.pop(module_name, None)
