"""``TracedHttpClient.probe``: a health check that asks once and logs nothing above DEBUG.

A machine that is often off is probed on a schedule; :meth:`TracedHttpClient.request` would retry
and log every exhausted call at ERROR. These tests pin the probe's contract over
``httpx.MockTransport``: one attempt, ``True`` only for a status below 400, ``False`` (never an
exception) for a network failure, the circuit breaker untouched, and no log line above DEBUG.
"""

from __future__ import annotations

import logging

import httpx
import pytest
from threetears.core.http_client import TracedHttpClient

_BASE_URL = "https://gpu.example.test"


class _CountingBreaker:
    """counts every call of the circuit-breaker protocol, so a test can assert there were none."""

    def __init__(self) -> None:
        self.calls = 0

    def check(self) -> None:
        """counts a gate check."""
        self.calls += 1

    def record_success(self) -> None:
        """counts a success."""
        self.calls += 1

    def record_failure(self) -> None:
        """counts a failure."""
        self.calls += 1


def _client(handler: httpx.MockTransport, **kwargs: object) -> TracedHttpClient:
    params: dict[str, object] = {"upstream_base_url": _BASE_URL, "transport": handler, "max_attempts": 3}
    params.update(kwargs)
    return TracedHttpClient(**params)  # type: ignore[arg-type]


def _answering(status: int, seen: list[httpx.Request]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, text="ok")

    return httpx.MockTransport(handler)


def _failing(exc: Exception, seen: list[httpx.Request]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        raise exc

    return httpx.MockTransport(handler)


@pytest.mark.parametrize(("status", "expected"), [(200, True), (204, True), (302, True), (404, False), (503, False)])
async def test_an_answer_below_400_is_true_and_any_other_is_false(status: int, expected: bool) -> None:
    seen: list[httpx.Request] = []
    async with _client(_answering(status, seen)) as client:
        assert await client.probe("/health") is expected
    assert len(seen) == 1, "a probe asks once, whatever the answer"


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ConnectError("refused"),
        httpx.ConnectTimeout("connect timed out"),
        httpx.ReadTimeout("read timed out"),
        httpx.RemoteProtocolError("dropped"),
    ],
)
async def test_a_network_failure_is_false_after_one_try_and_logs_nothing_above_debug(
    exc: Exception, caplog: pytest.LogCaptureFixture
) -> None:
    seen: list[httpx.Request] = []
    caplog.set_level(logging.DEBUG, logger="threetears.core.http_client")  # another test may hold threetears at INFO
    async with _client(_failing(exc, seen)) as client:
        assert await client.probe("/health") is False
    assert len(seen) == 1, "a probe never retries"
    assert not [r for r in caplog.records if r.levelno > logging.DEBUG], (
        "a probe of a machine that is off is not an error"
    )
    assert any(r.levelno == logging.DEBUG and r.name == "threetears.core.http_client" for r in caplog.records)


async def test_the_circuit_breaker_is_neither_checked_nor_fed() -> None:
    breaker = _CountingBreaker()
    seen: list[httpx.Request] = []
    async with _client(_failing(httpx.ConnectError("refused"), seen), circuit_breaker=breaker) as client:
        assert await client.probe("/health") is False
    async with _client(_answering(200, seen), circuit_breaker=breaker) as client:
        assert await client.probe("/health") is True
    assert breaker.calls == 0


async def test_method_headers_and_timeout_reach_the_wire() -> None:
    seen: list[httpx.Request] = []
    async with _client(_answering(200, seen)) as client:
        assert await client.probe("/health", method="HEAD", headers={"x-probe": "1"}, timeout=1.5)
    (request,) = seen
    assert request.method == "HEAD"
    assert request.headers["x-probe"] == "1"
    assert request.url == httpx.URL(f"{_BASE_URL}/health")
    assert request.extensions["timeout"]["read"] == 1.5


async def test_a_client_with_no_fixed_upstream_probes_an_absolute_url() -> None:
    seen: list[httpx.Request] = []
    async with TracedHttpClient(upstream_base_url=None, transport=_answering(200, seen)) as client:
        assert await client.probe("http://chonk.lan:8188/system_stats")
    assert seen[0].url.host == "chonk.lan"
