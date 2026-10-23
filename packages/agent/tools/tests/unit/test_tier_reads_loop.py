"""The tool server logs how its pod's collection reads were answered, by table and tier, on a timer."""

from __future__ import annotations

import asyncio
import contextlib
import logging

import pytest

from threetears.agent.tools.server import ToolServer
from threetears.core.collections.tier_reads import ReadTier, record_tier_read, take_tier_reads


@pytest.mark.asyncio
async def test_the_loop_logs_the_reads_each_period(caplog: pytest.LogCaptureFixture) -> None:
    take_tier_reads()
    record_tier_read("namespaces", ReadTier.L3)
    server = object.__new__(ToolServer)  # the loop reads none of the server's state
    with caplog.at_level(logging.INFO, logger="threetears.core.collections.tier_reads"):
        task = asyncio.create_task(server._tier_reads_loop(every=0.01))  # noqa: SLF001 - the loop serve() starts
        await asyncio.sleep(0.05)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    assert "collection reads by tier" in [r.getMessage() for r in caplog.records]
    assert take_tier_reads() == {}
