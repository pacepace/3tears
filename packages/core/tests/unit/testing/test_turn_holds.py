"""turn_holds: each turn of the event loop timed on the loop thread's CPU clock (through the asyncio confinement module,
``threetears.core.testing._asyncio_internals``), and asyncio's own run put back after the block."""

from __future__ import annotations

import asyncio
import time

import pytest

from threetears.core.testing.loop_lateness import turn_holds


def _spin(seconds: float) -> float:
    """hold the thread on CPU for about ``seconds``; the CPU seconds it took."""
    began = time.thread_time()
    while time.thread_time() - began < seconds:
        sum(range(1000))
    return time.thread_time() - began


# bounds its own time on the host (turn_holds): run apart, never beside other work (tests/timing_suite.py)
@pytest.mark.loop_timing
async def test_a_turn_that_holds_the_loop_is_timed_as_one_hold() -> None:
    """Flips if a callback's CPU is not counted as one turn (split, lost, or not timed at all), or a quiet turn is
    counted as long: one callback spinning 20 ms is the longest turn, at least its own spin, and nothing else is."""
    spun: list[float] = []
    with turn_holds() as holds:
        asyncio.get_running_loop().call_soon(lambda: spun.append(_spin(0.02)))
        await asyncio.sleep(0.01)
    longest, what = max(holds)
    assert longest >= spun[0] * 0.9, (longest, spun)
    assert "lambda" in what
    assert sorted(held for held, _ in holds)[-2] < 0.005, sorted(holds)[-3:]


# times the loop's turns on the host (turn_holds), which only a test run apart may (tests/timing_suite.py)
@pytest.mark.loop_timing
async def test_asyncio_runs_its_callbacks_as_before_after_the_block() -> None:
    """Flips if the block leaves asyncio's run wrapped: no turn is recorded after it ends (the turn that left the block
    is the last, recorded as it ends)."""
    with turn_holds() as holds:
        await asyncio.sleep(0)
    await asyncio.sleep(0)
    counted = len(holds)
    await asyncio.sleep(0)
    asyncio.get_running_loop().call_soon(lambda: None)
    await asyncio.sleep(0)
    assert len(holds) == counted > 0
