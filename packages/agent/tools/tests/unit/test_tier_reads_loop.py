"""The tool server's timer logs how its pod's collection reads were answered, by table and tier.

The counters are the process's, so the timer is too: every tool server in a process holds the one timer, and the
last to stop logs the reads since its last line.
"""

from __future__ import annotations

import asyncio
import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from threetears.agent.tools.server import ToolServer
from threetears.core.collections.tier_reads import (
    ReadTier,
    hold_tier_reads_timer,
    record_tier_read,
    release_tier_reads_timer,
)


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


async def _drained() -> dict[str, dict[str, int]]:
    """the reads since the last line, as the last holder of the process's timer logs them."""
    hold_tier_reads_timer()
    return await release_tier_reads_timer()


@pytest.mark.asyncio
async def test_the_tool_server_holds_the_timer_while_it_runs_and_its_stop_logs_the_last_reads(
    caplog: pytest.LogCaptureFixture,
) -> None:
    await _drained()
    with (
        patch("threetears.agent.tools.server.nats_connect", return_value=_mock_nc()),
        caplog.at_level(logging.INFO, logger="threetears.core.collections.tier_reads"),
    ):
        server, serve_task = await _serving("tier-pod")
        record_tier_read("namespaces", ReadTier.L3)
        # the server holds the timer: a holder that lets go beside it logs nothing
        assert await _drained() == {}, "the running server holds the process's timer"
        assert _lines(caplog) == []

        await server.shutdown()
        await asyncio.wait_for(serve_task, timeout=1.0)
    assert _lines(caplog) == [{"namespaces": {"l3": 1}}], "stopping logs the reads since the last line"
    assert await _drained() == {}


@pytest.mark.asyncio
async def test_two_tool_servers_in_one_process_share_one_timer(caplog: pytest.LogCaptureFixture) -> None:
    await _drained()
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
