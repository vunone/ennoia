"""ennoia.server.auth — AuthHook and its built-in implementations."""

from __future__ import annotations

import pytest

from ennoia.server.auth import env_bearer_auth, no_auth, static_bearer_auth


async def test_static_bearer_auth_accepts_matching_token() -> None:
    hook = static_bearer_auth("sekret")
    assert await hook("sekret") is True


async def test_static_bearer_auth_rejects_wrong_token() -> None:
    hook = static_bearer_auth("sekret")
    assert await hook("wrong") is False
    assert await hook(None) is False


async def test_no_auth_accepts_anything() -> None:
    hook = no_auth()
    assert await hook(None) is True
    assert await hook("anything") is True


def test_env_bearer_auth_returns_none_when_env_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ENNOIA_API_KEY", raising=False)
    assert env_bearer_auth() is None


async def test_env_bearer_auth_reads_configured_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENNOIA_API_KEY", "env-key")
    hook = env_bearer_auth()
    assert hook is not None
    assert await hook("env-key") is True
    assert await hook("other") is False


async def test_env_bearer_auth_accepts_custom_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_KEY", "xyz")
    hook = env_bearer_auth("MY_KEY")
    assert hook is not None
    assert await hook("xyz") is True
