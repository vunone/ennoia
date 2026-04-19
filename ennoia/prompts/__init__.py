"""Bundled prompt assets shipped with the wheel.

Prompts live as sibling ``.md`` files and are read through
:func:`load_prompt`. The loader uses :mod:`importlib.resources` so the
lookup works identically from a source checkout, an editable install, a
zipapp, or an installed wheel.
"""

from __future__ import annotations

from functools import cache
from importlib.resources import files

__all__ = ["load_prompt"]


@cache
def load_prompt(name: str) -> str:
    """Return the contents of ``ennoia/prompts/{name}.md`` as text."""
    return files(__name__).joinpath(f"{name}.md").read_text(encoding="utf-8")
