"""Synchronous typed event bus.

Design constraints:

- Observability must never break the pipeline. Handler exceptions are logged
  and swallowed.
- Type-dispatched: a handler subscribed for ``ExtractionEvent`` only receives
  events exactly of that type (no subclass fan-in — keep dispatch obvious).
- Default :class:`NullEmitter` is a no-op, so ``Pipeline`` can hold an emitter
  unconditionally without conditional ``if self.events:`` guards.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from typing import Any, TypeVar, cast

__all__ = ["Emitter", "NullEmitter"]

_log = logging.getLogger(__name__)

E = TypeVar("E")


class Emitter:
    def __init__(self) -> None:
        self._handlers: dict[type[object], list[Callable[[Any], None]]] = defaultdict(list)

    def subscribe(self, event_type: type[E], handler: Callable[[E], None]) -> None:
        self._handlers[event_type].append(cast(Callable[[Any], None], handler))

    def emit(self, event: object) -> None:
        for handler in self._handlers.get(type(event), []):
            try:
                handler(event)
            except Exception:
                _log.exception("Event handler raised on %s", type(event).__name__)


class NullEmitter(Emitter):
    """Emitter that discards every event. Default for pipelines."""

    def emit(self, event: object) -> None:
        return None
