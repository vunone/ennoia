"""Pluggable auth for the REST and MCP servers.

:class:`AuthHook` is a minimal Protocol: ``await hook(token_or_none) -> bool``.
Returning ``True`` admits the request; ``False`` triggers a 401 on the REST
side and a tool-call rejection on the MCP side.

Built-in hooks:

- :func:`static_bearer_auth` — compare the bearer token against a fixed string.
- :func:`env_bearer_auth` — read the expected token from ``ENNOIA_API_KEY``,
  or return ``None`` when the env var is unset so the caller can refuse to
  start the server without explicit ``no_auth``.
- :func:`no_auth` — accept every request. Meant for local development.
"""

from __future__ import annotations

import os
from typing import Protocol

__all__ = [
    "AuthHook",
    "env_bearer_auth",
    "no_auth",
    "static_bearer_auth",
]


class AuthHook(Protocol):
    async def __call__(self, token: str | None) -> bool: ...


def static_bearer_auth(api_key: str) -> AuthHook:
    """Return a hook that accepts exactly ``api_key`` as the bearer token."""

    async def _hook(token: str | None) -> bool:
        return token == api_key

    return _hook


def env_bearer_auth(var_name: str = "ENNOIA_API_KEY") -> AuthHook | None:
    """Bearer-auth hook reading the expected key from an env var.

    Returns ``None`` when the env var is unset so the caller can decide
    whether to proceed with :func:`no_auth` or fail-closed.
    """
    key = os.environ.get(var_name)
    if not key:
        return None
    return static_bearer_auth(key)


def no_auth() -> AuthHook:
    """Hook that accepts every request; for local dev only."""

    async def _hook(token: str | None) -> bool:
        return True

    return _hook
