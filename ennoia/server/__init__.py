"""Remote interface servers — FastAPI REST + FastMCP.

Both servers read from a shared :class:`~ennoia.server.context.ServerContext`
built once at startup and delegate every request back to the underlying
:class:`~ennoia.index.pipeline.Pipeline`. The REST surface is full CRUD;
the MCP surface is read-only so agents cannot mutate the store.
"""

from __future__ import annotations

from ennoia.server.auth import AuthHook, env_bearer_auth, no_auth, static_bearer_auth
from ennoia.server.context import ServerContext

__all__ = [
    "AuthHook",
    "ServerContext",
    "env_bearer_auth",
    "no_auth",
    "static_bearer_auth",
]
