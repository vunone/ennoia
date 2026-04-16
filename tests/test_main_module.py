"""``python -m ennoia`` entrypoint — exercise the module so coverage sees it."""

from __future__ import annotations

import runpy
import sys

import pytest


def test_python_dash_m_ennoia_invokes_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    # ``ennoia --help`` exits with code 0; ``no_args_is_help=True`` is code 2
    # because typer treats "show help because no command was given" as an error.
    monkeypatch.setattr(sys, "argv", ["ennoia", "--help"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("ennoia", run_name="__main__")
    assert exc.value.code == 0


def test_importing_main_module_as_library_is_a_noop() -> None:
    # Covers the ``if __name__ == "__main__":`` False-branch: importing the
    # module normally must not invoke the CLI.
    import importlib

    mod = importlib.import_module("ennoia.__main__")
    # Import succeeded without side effects — ``app`` is available but not invoked.
    assert callable(mod.app)
