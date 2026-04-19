"""``ennoia.ini`` support — load flag defaults + preload provider env vars.

The CLI entrypoint calls :func:`load_ini` from its top-level Typer callback,
which runs *before* any subcommand parses its options. Values from the
``[ennoia]`` section populate ``ENNOIA_*`` env vars (e.g. ``llm`` →
``ENNOIA_LLM``); Typer's ``envvar=`` hookup on each subcommand then surfaces
them as flag defaults. Values from the ``[env]`` section are exported
verbatim (case preserved) so adapter SDKs such as ``openai`` /
``anthropic`` pick up provider keys when their clients initialize.

All writes use :func:`os.environ.setdefault`, so:

    explicit --flag  >  shell env  >  INI [ennoia] / [env]  >  hardcoded default
"""

from __future__ import annotations

import configparser
import os
from pathlib import Path
from typing import TypeVar

import typer

T = TypeVar("T")

__all__ = [
    "INI_FILENAME",
    "INI_TEMPLATE",
    "KEY_TO_ENVVAR",
    "load_ini",
    "require_option",
    "write_template",
]


INI_FILENAME = "ennoia.ini"


KEY_TO_ENVVAR: dict[str, str] = {
    "llm": "ENNOIA_LLM",
    "embedding": "ENNOIA_EMBEDDING",
    "store": "ENNOIA_STORE",
    "schema": "ENNOIA_SCHEMA",
    "collection": "ENNOIA_COLLECTION",
    "qdrant_url": "ENNOIA_QDRANT_URL",
    "qdrant_api_key": "ENNOIA_QDRANT_API_KEY",
    "pg_dsn": "ENNOIA_PG_DSN",
    "host": "ENNOIA_HOST",
    "port": "ENNOIA_PORT",
    "transport": "ENNOIA_TRANSPORT",
    "api_key": "ENNOIA_API_KEY",
}


INI_TEMPLATE = """\
# Ennoia CLI configuration.
#
# [ennoia] keys populate ENNOIA_* env vars via os.environ.setdefault()
# before command dispatch, so Typer's envvar= hookups pick them up as
# flag defaults. Explicit --flags always override.
#
# [env] keys are exported verbatim (also via setdefault) before adapter
# clients initialize -- use it for provider API keys.
#
# Shell-exported env vars take precedence over this file.

[ennoia]
llm = ollama:qwen3:0.6b
embedding = sentence-transformers:all-MiniLM-L6-v2

store = ./ennoia_index
collection = documents

# schema = ./schemas.py

# qdrant_url = http://localhost:6333
# qdrant_api_key =
# pg_dsn = postgresql://user:pass@localhost:5432/ennoia

# host = 127.0.0.1
# port = 8080
# transport = sse
# api_key =

[env]
# OPENAI_API_KEY =
# ANTHROPIC_API_KEY =
# OPENROUTER_API_KEY =
"""


def load_ini(path: Path) -> bool:
    """Apply ``ennoia.ini`` values to ``os.environ`` via ``setdefault``.

    Returns ``True`` when the file was read, ``False`` when it did not
    exist (missing file is not an error — callers run the CLI without
    any config). Unknown keys in ``[ennoia]`` raise
    :class:`typer.BadParameter` so typos surface immediately.
    """
    if not path.exists():
        return False

    parser = configparser.ConfigParser()
    # Preserve case for ``[env]`` so OPENAI_API_KEY etc. round-trip.
    parser.optionxform = str  # type: ignore[assignment,method-assign]
    parser.read(path, encoding="utf-8")

    if parser.has_section("ennoia"):
        for raw_key, value in parser.items("ennoia"):
            key = raw_key.lower()
            if key not in KEY_TO_ENVVAR:
                allowed = ", ".join(sorted(KEY_TO_ENVVAR))
                raise typer.BadParameter(
                    f"Unknown key {raw_key!r} in [ennoia] section of {path}. Allowed: {allowed}."
                )
            if not value:
                continue
            os.environ.setdefault(KEY_TO_ENVVAR[key], value)

    if parser.has_section("env"):
        for key, value in parser.items("env"):
            if not value:
                continue
            os.environ.setdefault(key, value)

    return True


def require_option(value: T | None, flag: str, ini_key: str) -> T:
    """Guard INI-backed options that have no CLI default.

    Options like ``--store`` / ``--schema`` accept ``None`` at the Typer
    layer so ``--help`` does not display ``[required]``; callers invoke
    this helper at the top of the command body to surface a uniform
    error when neither the CLI flag, the shell env, nor ``ennoia.ini``
    provided a value.
    """
    if value is None:
        raise typer.BadParameter(
            f"Missing {flag}. Pass {flag} on the command line, export "
            f"{KEY_TO_ENVVAR[ini_key]}, or set {ini_key!r} in [ennoia] "
            f"of ennoia.ini."
        )
    return value


def write_template(path: Path, *, force: bool) -> Path:
    """Write :data:`INI_TEMPLATE` to ``path``.

    Refuses to overwrite an existing file unless ``force=True``. Returns
    the final path so the caller can echo it.
    """
    if path.exists() and not force:
        raise typer.BadParameter(f"{path} already exists. Pass --force to overwrite.")
    path.write_text(INI_TEMPLATE, encoding="utf-8")
    return path
