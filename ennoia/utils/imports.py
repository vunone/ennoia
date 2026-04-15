"""Uniform lazy-import helper for optional dependencies."""

from __future__ import annotations

import importlib
from types import ModuleType

__all__ = ["require_module"]


def require_module(name: str, extra: str) -> ModuleType:
    """Import ``name`` or raise a uniform ``ImportError`` naming the install extra.

    Used by adapters and stores whose dependencies live behind
    ``pip install ennoia[<extra>]`` so the error shape is identical everywhere.
    """
    try:
        return importlib.import_module(name)
    except ImportError as err:
        raise ImportError(
            f"{name} is required. Install with `pip install ennoia[{extra}]`."
        ) from err
