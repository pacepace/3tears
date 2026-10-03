"""a standalone tool pod's FIRST NATS connect: retried while the platform starts, then failing loud.

Driven through the front door -- ``ToolServer.serve()`` with ``nats_connect`` patched -- because the
retry is what decides whether a pod that boots before its platform ever serves. Its knobs are read
from the environment (``THREETEARS_TOOL_POD_CONNECT_RETRY_SECONDS`` and
``THREETEARS_TOOL_POD_CONNECT_RETRY_BACKOFF_CAP``), and a backoff cap of zero or less once stopped a
pod whose NATS was UP from starting at all: the retry refused its own schedule before the first
attempt, with nothing logged.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from threetears.agent.tools.config import get_connect_retry_backoff_cap
from threetears.agent.tools.server import ToolServer
from threetears.nats.errors import NatsClientError

_LOGGER = "threetears.agent.tools.server"
_BUDGET = "THREETEARS_TOOL_POD_CONNECT_RETRY_SECONDS"
_CAP = "THREETEARS_TOOL_POD_CONNECT_RETRY_BACKOFF_CAP"


def _connected_client() -> AsyncMock:
    """a NATS client wired enough for serve()'s JWKS warm-up and its publishes.

    :return: the client
    :rtype: AsyncMock
    """
    nc = AsyncMock()
    nc.renew_credential = MagicMock()  # synchronous on the real client
    nc.is_connected = True
    nc.is_closed = False
    nc.is_healthy = True
    nc.request_raw = AsyncMock(return_value=json.dumps({"keys": []}).encode("utf-8"))
    return nc


async def _serve_briefly(server: ToolServer) -> None:
    """run serve() until it has connected and published, then shut it down.

    :param server: the server under test
    :ptype server: ToolServer
    :return: nothing
    :rtype: None
    """
    serving = asyncio.create_task(server.serve())
    for _ in range(100):
        if server.is_connected or serving.done():
            break
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.05)
    await server.shutdown()
    serving.cancel()
    await asyncio.gather(serving, return_exceptions=True)


def _retry_logs(caplog: pytest.LogCaptureFixture) -> list[Any]:
    return [r for r in caplog.records if r.name == _LOGGER and "connect not ready" in r.getMessage()]


@pytest.mark.asyncio
async def test_a_pod_whose_nats_is_up_connects_on_the_first_try_without_a_pause(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv(_CAP, "0")  # a refused setting must never stop a pod whose NATS is up
    server = ToolServer(nats_url="nats://localhost:9999", pod_id="connect-first-try")
    nc = _connected_client()

    with (
        caplog.at_level(logging.INFO, logger=_LOGGER),
        patch("threetears.agent.tools.server.nats_connect", AsyncMock(return_value=nc)) as connect,
    ):
        await _serve_briefly(server)

    assert connect.await_count == 1
    assert _retry_logs(caplog) == []


@pytest.mark.asyncio
async def test_a_platform_still_starting_is_waited_for(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv(_BUDGET, "5")
    monkeypatch.setenv(_CAP, "0.02")
    server = ToolServer(nats_url="nats://localhost:9999", pod_id="connect-waits")
    nc = _connected_client()
    connect = AsyncMock(side_effect=[NatsClientError("nats: no servers"), OSError("connection refused"), nc])

    with caplog.at_level(logging.WARNING, logger=_LOGGER), patch("threetears.agent.tools.server.nats_connect", connect):
        await _serve_briefly(server)

    assert connect.await_count == 3
    attempts = [getattr(r, "extra_data", {})["attempt"] for r in _retry_logs(caplog)]
    assert attempts == [1, 2]


@pytest.mark.asyncio
async def test_a_spent_budget_fails_loud_naming_how_many_attempts_ran(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv(_BUDGET, "0.15")
    monkeypatch.setenv(_CAP, "0.02")
    server = ToolServer(nats_url="nats://localhost:9999", pod_id="connect-gives-up")
    connect = AsyncMock(side_effect=NatsClientError("nats: no servers"))

    with (
        caplog.at_level(logging.ERROR, logger=_LOGGER),
        patch("threetears.agent.tools.server.nats_connect", connect),
        pytest.raises(NatsClientError, match="no servers"),
    ):
        await server.serve()

    failed = [r for r in caplog.records if r.name == _LOGGER and "within the retry budget" in r.getMessage()]
    assert len(failed) == 1
    extra = getattr(failed[0], "extra_data", {})
    assert extra["attempts"] == connect.await_count
    assert connect.await_count > 1


@pytest.mark.asyncio
async def test_a_failure_that_is_not_a_connect_failure_is_raised_at_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_BUDGET, "5")
    monkeypatch.setenv(_CAP, "0.02")
    server = ToolServer(nats_url="nats://localhost:9999", pod_id="connect-bug")
    connect = AsyncMock(side_effect=RuntimeError("a bug, not a platform still starting"))

    with patch("threetears.agent.tools.server.nats_connect", connect), pytest.raises(RuntimeError, match="a bug"):
        await server.serve()

    assert connect.await_count == 1


@pytest.mark.parametrize("raw", ["0", "-1"])
def test_a_backoff_cap_that_is_not_positive_is_refused_naming_the_variable(
    raw: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv(_CAP, raw)

    with caplog.at_level(logging.WARNING):
        cap = get_connect_retry_backoff_cap()

    assert cap > 0, "the default stands in for a refused value"
    assert any(_CAP in r.getMessage() for r in caplog.records), "the refusal names the variable"
