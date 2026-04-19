"""Extract a Python source block from free-form LLM output.

LLMs tend to wrap generated code in a fenced block — sometimes tagged
``python``, sometimes bare, occasionally indented, and occasionally split
across multiple blocks (imports separate from class defs). This module
normalises all those shapes into a single Python source string.
"""

from __future__ import annotations

import re

__all__ = ["CraftParseError", "extract_python_block"]


_PYTHON_LANGS = frozenset({"", "python", "py", "python3"})

# Matches a fenced code block. Captures the opening indent so we can
# strip it uniformly from every body line, the optional language tag,
# and the body itself (non-greedy so we pick the shortest valid close).
_FENCE_RE = re.compile(
    r"^(?P<indent>[ \t]*)```(?P<lang>[A-Za-z0-9_+\-]*)[ \t]*\n"
    r"(?P<body>.*?)\n"
    r"(?P=indent)```[ \t]*$",
    re.DOTALL | re.MULTILINE,
)


class CraftParseError(ValueError):
    """Raised when no Python code block can be found in an LLM reply."""


def _strip_indent(body: str, indent: str) -> str:
    """Remove ``indent`` from the start of each line in ``body``.

    The fence regex anchors the closing ``` to the same indent as the
    opener, so every body line is guaranteed to start with that indent —
    we still guard with an explicit ``startswith`` check to stay correct
    on exotic inputs.
    """
    if not indent:
        return body
    stripped: list[str] = []
    for line in body.split("\n"):
        stripped.append(line[len(indent) :] if line.startswith(indent) else line)
    return "\n".join(stripped)


def extract_python_block(text: str) -> str:
    """Return the Python source contained in the first (or concatenated)
    fenced code block of ``text``.

    Rules:

    1. Line endings are normalised to ``\\n`` before matching.
    2. Blocks tagged ``python`` / ``py`` / ``python3`` or left bare are
       candidates; other language tags are ignored.
    3. Zero candidates raises :class:`CraftParseError`.
    4. One candidate is returned verbatim (with the opening indent
       stripped from every body line).
    5. Multiple candidates are joined in order with a blank line, with
       tagged blocks preferred over bare when both are present.

    The function never falls back to the raw text — silent fallbacks hide
    prompt-format bugs, and the craft retry loop needs the failure
    surfaced so it can feed the parse error back to the LLM.
    """
    normalised = text.replace("\r\n", "\n").replace("\r", "\n")

    candidates: list[tuple[str, str]] = []  # (lang_lower, body)
    for match in _FENCE_RE.finditer(normalised):
        lang = match.group("lang").lower()
        if lang not in _PYTHON_LANGS:
            continue
        body = _strip_indent(match.group("body"), match.group("indent"))
        candidates.append((lang, body))

    if not candidates:
        raise CraftParseError(
            "no python code block found — wrap your reply in a ```python fenced block."
        )
    if len(candidates) == 1:
        return candidates[0][1]

    tagged = [body for lang, body in candidates if lang]
    chosen = tagged or [body for _, body in candidates]
    return "\n\n".join(chosen)
