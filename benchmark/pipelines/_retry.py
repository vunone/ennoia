"""Shared async-with-retry helper for OpenRouter / OpenAI transient errors.

Indexing, query-generation, agent turns, and judge calls all hit the same
``openai.AsyncOpenAI`` surface (OpenRouter exposes an OpenAI-compatible
endpoint; so does the Ollama shim we kept for regression coverage). A
single retry helper here keeps the policy identical across every caller.

Backoff policy:

- **Network errors** (``APIConnectionError`` / ``APITimeoutError``): linear
  backoff ``initial * attempt`` — 0s, 1s, 2s, 3s, ... — cheap retries
  because these are usually transient TCP hiccups.
- **Rate-limit errors** (``RateLimitError`` — very common on free-tier
  OpenRouter models): parse the ``retry-after`` / ``x-ratelimit-reset``
  header from the HTTP 429 response; if absent, wait
  ``RATE_LIMIT_DEFAULT_WAIT_SEC`` (30s). Capped at
  ``RATE_LIMIT_MAX_WAIT_SEC`` (120s) so a runaway server directive can't
  hang a run.

Validation or schema failures surface on the first attempt — retrying
them wastes tokens and masks real bugs.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

from openai import APIConnectionError, APITimeoutError, RateLimitError

T = TypeVar("T")

RETRY_ATTEMPTS = 8
RETRY_INITIAL_BACKOFF_SEC = 1.0
RATE_LIMIT_DEFAULT_WAIT_SEC = 30.0
RATE_LIMIT_MAX_WAIT_SEC = 120.0
RETRYABLE_EXC: tuple[type[BaseException], ...] = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
)


def _rate_limit_backoff(exc: RateLimitError) -> float:
    """Return the server-suggested retry delay, or the default fallback.

    Checks the standard ``retry-after`` HTTP header first (as RFC 6585
    mandates for 429s); if absent, tries OpenAI-family
    ``x-ratelimit-reset-requests``. Values are clamped to
    ``RATE_LIMIT_MAX_WAIT_SEC`` so a misbehaving server cannot wedge the
    run indefinitely.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) if response is not None else None
    if headers:
        for key in ("retry-after", "x-ratelimit-reset-requests", "x-ratelimit-reset"):
            raw = headers.get(key)
            if raw is None:
                continue
            try:
                delay = float(raw)
            except (TypeError, ValueError):
                continue
            return max(0.0, min(delay, RATE_LIMIT_MAX_WAIT_SEC))
    return RATE_LIMIT_DEFAULT_WAIT_SEC


def _delay_for(exc: BaseException, attempt: int, initial_backoff: float) -> float:
    if isinstance(exc, RateLimitError):
        return _rate_limit_backoff(exc)
    return initial_backoff * attempt


async def async_with_retry(
    factory: Callable[[], Awaitable[T]],
    *,
    attempts: int = RETRY_ATTEMPTS,
    initial_backoff: float = RETRY_INITIAL_BACKOFF_SEC,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
) -> T:
    """Run ``factory()`` with retry on transient network + rate-limit errors.

    ``factory`` must be a zero-arg callable that returns a fresh awaitable
    on each attempt — not a single awaitable to be retried, which cannot be
    awaited twice. Only ``APIConnectionError``, ``APITimeoutError``, and
    ``RateLimitError`` are retried; anything else raises on the first
    attempt so real bugs are not masked by retry noise.

    ``on_retry`` receives ``(attempt_index, exception, delay_seconds)`` on
    each retry — used by callers to emit progress-bar postfix updates.
    """
    for attempt in range(attempts):
        try:
            return await factory()
        except RETRYABLE_EXC as exc:
            if attempt == attempts - 1:
                raise
            delay = _delay_for(exc, attempt, initial_backoff)
            if on_retry is not None:
                on_retry(attempt, exc, delay)
            await asyncio.sleep(delay)
    raise RuntimeError(  # pragma: no cover - unreachable; the loop raises on exhaustion.
        "async_with_retry exited the loop without returning"
    )
