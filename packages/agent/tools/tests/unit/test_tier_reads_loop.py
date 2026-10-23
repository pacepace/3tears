"""The tool server's timer logs how its pod's collection reads were answered, by table and tier.

The counters are the process's, so the timer is too: every tool server in a process holds the one timer, and the
last to stop logs the reads since its last line.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from threetears.agent.tools.server import ToolServer
from threetears.core.collections.tier_reads import ReadTier, log_tier_reads, log_tier_reads_every, record_tier_read


@pytest.mark.asyncio
async def test_the_timer_logs_the_reads_each_period(caplog: pytest.LogCaptureFixture) -> None:
    log_tier_reads()
    record_tier_read("namespaces", ReadTier.L3)
    with caplog.at_level(logging.INFO, logger="threetears.core.collections.tier_reads"):
        task = asyncio.create_task(log_tier_reads_every(every=0.01))
        await asyncio.sleep(0.05)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    assert "collection reads by tier" in [r.getMessage() for r in caplog.records]
    assert log_tier_reads() == {}


def _mock_nc() -> AsyncMock:
    """a NATS client double sufficient to drive ``serve`` and ``shutdown``.

    :return: the double
    :rtype: AsyncMock
    """
    mock_nc = AsyncMock()
    mock_nc.is_connected = True
    mock_nc.subscribe = AsyncMock()
    mock_nc.publish = AsyncMock()
    mock_nc.renew_credential = MagicMock()
    mock_nc.request_raw = AsyncMock(return_value=json.dumps({"keys": []}).encode("utf-8"))
    return mock_nc


async def _serving(pod_id: str) -> tuple[ToolServer, asyncio.Task[None]]:
    """a tool server serving on a NATS double, once it has started.

    :param pod_id: the server's pod id
    :ptype pod_id: str
    :return: the server and its serve task
    :rtype: tuple[ToolServer, asyncio.Task[None]]
    """
    server = ToolServer(nats_url="nats://localhost:9999", namespace="testns", pod_id=pod_id)
    serve_task = asyncio.create_task(server.serve())
    await server.wait_ready(timeout=1.0)
    await asyncio.sleep(0)
    return server, serve_task


def _lines(caplog: pytest.LogCaptureFixture) -> list[object]:
    """the reads each "collection reads by tier" line logged.

    :param caplog: the captured records
    :ptype caplog: pytest.LogCaptureFixture
    :return: each line's reads, in order
    :rtype: list[object]
    """
    return [
        getattr(r, "extra_data", {}).get("reads")
        for r in caplog.records
        if r.getMessage() == "collection reads by tier"
    ]


@pytest.mark.asyncio
async def test_the_tool_server_runs_the_timer_and_its_stop_logs_the_last_reads(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log_tier_reads()
    with (
        patch("threetears.agent.tools.server.nats_connect", return_value=_mock_nc()),
        patch("threetears.core.collections.tier_reads.TIER_READS_LOG_SECONDS", 0.01),
        caplog.at_level(logging.INFO, logger="threetears.core.collections.tier_reads"),
    ):
        server, serve_task = await _serving("tier-pod")
        record_tier_read("namespaces", ReadTier.L3)
        for _ in range(100):
            if _lines(caplog):
                break
            await asyncio.sleep(0.01)
        assert _lines(caplog) == [{"namespaces": {"l3": 1}}], "the running server's timer logs each period"

        record_tier_read("namespaces", ReadTier.L1)
        await server.shutdown()
        await asyncio.wait_for(serve_task, timeout=1.0)
    assert _lines(caplog)[-1] == {"namespaces": {"l1": 1}}, "stopping logs the reads since the last line"
    assert log_tier_reads() == {}


@pytest.mark.asyncio
async def test_two_tool_servers_in_one_process_share_one_timer(caplog: pytest.LogCaptureFixture) -> None:
    log_tier_reads()
    with (
        patch("threetears.agent.tools.server.nats_connect", side_effect=[_mock_nc(), _mock_nc()]),
        caplog.at_level(logging.INFO, logger="threetears.core.collections.tier_reads"),
    ):
        first, first_task = await _serving("tier-pod-a")
        second, second_task = await _serving("tier-pod-b")
        record_tier_read("namespaces", ReadTier.L2)

        await first.shutdown()
        await asyncio.wait_for(first_task, timeout=1.0)
        assert _lines(caplog) == [], "the process's timer runs on while another server holds it"

        await second.shutdown()
        await asyncio.wait_for(second_task, timeout=1.0)
    assert _lines(caplog) == [{"namespaces": {"l2": 1}}], "the last server to stop logs the process's reads once"
