"""Unit tests for the shared async_with_retry helper."""

from __future__ import annotations

import httpx
import pytest
from openai import APIConnectionError, APITimeoutError, RateLimitError

from benchmark.pipelines import _retry


def _make_rate_limit(headers: dict[str, str] | None = None) -> RateLimitError:
    request = httpx.Request("GET", "https://example.com")
    response = httpx.Response(429, request=request, headers=headers or {})
    return RateLimitError(message="slow down", response=response, body=None)


async def test_returns_value_on_first_success() -> None:
    async def ok() -> str:
        return "ok"

    assert await _retry.async_with_retry(ok) == "ok"


async def test_retries_and_recovers_on_transient(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(_retry.asyncio, "sleep", fake_sleep)

    attempts = {"n": 0}

    async def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise APIConnectionError(request=None)  # type: ignore[arg-type]
        if attempts["n"] == 2:
            raise APITimeoutError(request=None)  # type: ignore[arg-type]
        return "done"

    observed: list[tuple[int, type[BaseException], float]] = []
    result = await _retry.async_with_retry(
        flaky, on_retry=lambda a, e, d: observed.append((a, type(e), d))
    )
    assert result == "done"
    assert attempts["n"] == 3
    # Linear backoff starting from attempt=0 (no sleep), then 1x on attempt=1.
    assert sleeps == [0.0, _retry.RETRY_INITIAL_BACKOFF_SEC]
    assert [(a, e) for a, e, _ in observed] == [(0, APIConnectionError), (1, APITimeoutError)]


async def test_retries_rate_limit_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(_retry.asyncio, "sleep", fake_sleep)
    attempts = {"n": 0}

    async def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise _make_rate_limit()
        return "ok"

    assert await _retry.async_with_retry(flaky) == "ok"
    assert attempts["n"] == 3
    # Without a retry-after header we fall back to the default constant
    # wait, not the short linear backoff used for network errors.
    assert sleeps == [_retry.RATE_LIMIT_DEFAULT_WAIT_SEC, _retry.RATE_LIMIT_DEFAULT_WAIT_SEC]


async def test_rate_limit_parses_retry_after_header(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(_retry.asyncio, "sleep", fake_sleep)

    async def flaky() -> str:
        if not sleeps:
            raise _make_rate_limit(headers={"retry-after": "7"})
        return "ok"

    assert await _retry.async_with_retry(flaky) == "ok"
    assert sleeps == [7.0]


async def test_rate_limit_parses_alternate_reset_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(_retry.asyncio, "sleep", fake_sleep)

    async def flaky() -> str:
        if not sleeps:
            raise _make_rate_limit(headers={"x-ratelimit-reset-requests": "3"})
        return "ok"

    assert await _retry.async_with_retry(flaky) == "ok"
    assert sleeps == [3.0]


async def test_rate_limit_caps_excessive_server_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(_retry.asyncio, "sleep", fake_sleep)

    async def flaky() -> str:
        if not sleeps:
            raise _make_rate_limit(headers={"retry-after": "9999"})
        return "ok"

    await _retry.async_with_retry(flaky)
    assert sleeps == [_retry.RATE_LIMIT_MAX_WAIT_SEC]


async def test_rate_limit_ignores_unparseable_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(_retry.asyncio, "sleep", fake_sleep)

    async def flaky() -> str:
        if not sleeps:
            raise _make_rate_limit(headers={"retry-after": "banana"})
        return "ok"

    await _retry.async_with_retry(flaky)
    assert sleeps == [_retry.RATE_LIMIT_DEFAULT_WAIT_SEC]


def test_rate_limit_backoff_without_response_uses_default() -> None:
    exc = _make_rate_limit()
    exc.response = None  # type: ignore[assignment]
    assert _retry._rate_limit_backoff(exc) == _retry.RATE_LIMIT_DEFAULT_WAIT_SEC


async def test_raises_after_exhausting_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr(_retry.asyncio, "sleep", fake_sleep)

    async def always_fails() -> str:
        raise APITimeoutError(request=None)  # type: ignore[arg-type]

    with pytest.raises(APITimeoutError):
        await _retry.async_with_retry(always_fails, attempts=3)


async def test_does_not_retry_non_transient_errors() -> None:
    calls = {"n": 0}

    async def boom() -> str:
        calls["n"] += 1
        raise ValueError("bad input")

    with pytest.raises(ValueError, match="bad input"):
        await _retry.async_with_retry(boom)
    assert calls["n"] == 1
