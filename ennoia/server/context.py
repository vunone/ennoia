"""Shared state both servers (:mod:`.api`, :mod:`.mcp`) read from.

Built once at startup and passed into ``create_app`` / ``create_mcp``. Wraps
a configured :class:`~ennoia.index.pipeline.Pipeline` plus an
:class:`~ennoia.server.auth.AuthHook`. Events emitted by the pipeline during
remote calls flow through :attr:`ennoia.index.pipeline.Pipeline.events`, so
observability is inherited for free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ennoia.index.pipeline import Pipeline
    from ennoia.server.auth import AuthHook

__all__ = ["ServerContext"]


@dataclass
class ServerContext:
    pipeline: Pipeline
    auth: AuthHook
