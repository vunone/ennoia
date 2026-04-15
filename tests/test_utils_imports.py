"""Unit tests for ennoia.utils.imports — lazy optional-dependency loader."""

from __future__ import annotations

import json as _json

import pytest

from ennoia.utils.imports import require_module


def test_require_module_returns_real_module_on_success():
    # Using stdlib `json` as a stand-in for any installed optional dep.
    assert require_module("json", "dev") is _json


def test_require_module_raises_with_uniform_install_hint():
    with pytest.raises(ImportError) as exc_info:
        require_module("definitely_not_a_module_xyz", "foo")

    message = str(exc_info.value)
    assert "definitely_not_a_module_xyz" in message
    assert "pip install ennoia[foo]" in message


def test_require_module_preserves_original_cause():
    with pytest.raises(ImportError) as exc_info:
        require_module("definitely_not_a_module_xyz", "foo")

    assert isinstance(exc_info.value.__cause__, ImportError)
