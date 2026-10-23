"""The tool server's timer logs how its pod's collection reads were answered, by table and tier."""

from __future__ import annotations

import asyncio
import contextlib
import logging

import pytest

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


def test_the_tool_server_starts_the_timer() -> None:
    from threetears.agent.tools import server

    assert server.log_tier_reads_every is log_tier_reads_every
