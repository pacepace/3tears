"""The tier-read counters: the process's one timer, its line each period, and the last holder's line.

Which tier a collection's key read is counted against is tested with the pods of
``test_table_drop_and_derived_caches`` (``TestReadsAreCountedByTier``).
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from threetears.core.collections.tier_reads import (
    ReadTier,
    hold_tier_reads_timer,
    record_tier_read,
    release_tier_reads_timer,
)

_LINE = "collection reads by tier"


async def _drained() -> dict[str, dict[str, int]]:
    """the reads since the last line, as the last holder of the timer logs them."""
    hold_tier_reads_timer()
    return await release_tier_reads_timer()


def _lines(caplog: pytest.LogCaptureFixture) -> list[object]:
    return [getattr(r, "extra_data", {}).get("reads") for r in caplog.records if r.getMessage() == _LINE]


@pytest.mark.asyncio
async def test_the_last_holder_logs_the_reads_since_the_last_line_once_and_starts_them_again(
    caplog: pytest.LogCaptureFixture,
) -> None:
    await _drained()
    caplog.clear()  # the reads earlier tests in this process left, logged by the drain
    with caplog.at_level(logging.INFO, logger="threetears.core.collections.tier_reads"):
        hold_tier_reads_timer()
        hold_tier_reads_timer()
        record_tier_read("t", ReadTier.L1)
        record_tier_read("t", ReadTier.L1)
        record_tier_read("t", ReadTier.L3)
        assert await release_tier_reads_timer() == {}, "another holder keeps the timer running"
        assert _lines(caplog) == []
        assert await release_tier_reads_timer() == {"t": {"l1": 2, "l3": 1}}
        assert await _drained() == {}, "a line starts the counts again"
    assert _lines(caplog) == [{"t": {"l1": 2, "l3": 1}}], "nothing is logged when no read happened"


@pytest.mark.asyncio
async def test_the_timer_logs_the_reads_each_period(caplog: pytest.LogCaptureFixture) -> None:
    await _drained()
    caplog.clear()  # the reads earlier tests in this process left, logged by the drain
    with caplog.at_level(logging.INFO, logger="threetears.core.collections.tier_reads"):
        hold_tier_reads_timer(every=0.01)
        record_tier_read("namespaces", ReadTier.L3)
        for _ in range(100):
            if _lines(caplog):
                break
            await asyncio.sleep(0.01)
        assert _lines(caplog) == [{"namespaces": {"l3": 1}}]
        assert await release_tier_reads_timer() == {}
    assert _lines(caplog) == [{"namespaces": {"l3": 1}}]


@pytest.mark.asyncio
async def test_releasing_a_timer_nothing_holds_is_refused() -> None:
    with pytest.raises(RuntimeError, match="released more times than it was held"):
        await release_tier_reads_timer()


@pytest.mark.asyncio
async def test_the_total_since_the_process_started_is_read_without_starting_the_counts_again() -> None:
    from threetears.core.collections import tier_reads_total

    before = tier_reads_total().get("totals-table", {})
    record_tier_read("totals-table", ReadTier.L1)
    record_tier_read("totals-table", ReadTier.L1)
    record_tier_read("totals-table", ReadTier.L2)
    after = tier_reads_total()["totals-table"]
    assert after == {
        "l1": before.get("l1", 0) + 2,
        "l2": before.get("l2", 0) + 1,
        **({"l3": before["l3"]} if "l3" in before else {}),
    }
    assert tier_reads_total()["totals-table"] == after, "a read of the total starts nothing again"

    drained = await _drained()
    assert drained["totals-table"] == {"l1": 2, "l2": 1}, "the line keeps its own counts since its last line"
    assert tier_reads_total()["totals-table"] == after, "and the line starts none of the total again"
    record_tier_read("totals-table", ReadTier.L3)
    assert tier_reads_total()["totals-table"] == {**after, "l3": before.get("l3", 0) + 1}


def test_a_hold_on_a_second_open_loop_is_served_by_the_first_loops_timer_and_neither_fails_its_release() -> None:
    import threading

    first = asyncio.new_event_loop()
    ready = threading.Event()
    runner = threading.Thread(target=lambda: (first.call_soon(ready.set), first.run_forever()), daemon=True)
    runner.start()
    ready.wait()
    try:
        asyncio.run_coroutine_threadsafe(_hold(), first).result(timeout=5)

        async def second_loop() -> dict[str, dict[str, int]]:
            hold_tier_reads_timer()
            return await release_tier_reads_timer()

        assert asyncio.run(second_loop()) == {}, "the second loop's hold started nothing, and its release logs nothing"
        record_tier_read("two-loops", ReadTier.L2)

        # a second loop that holds while the first loop's timer runs and lets go after it stopped
        async def held_past_the_first() -> dict[str, dict[str, int]]:
            hold_tier_reads_timer()
            released = asyncio.run_coroutine_threadsafe(release_tier_reads_timer(), first).result(timeout=5)
            assert released["two-loops"] == {"l2": 1}, "the first loop's timer was never taken from it"
            return await release_tier_reads_timer()

        assert asyncio.run(held_past_the_first()) == {}, "its release neither raises nor logs"
    finally:
        first.call_soon_threadsafe(first.stop)
        runner.join(timeout=5)
        first.close()


async def _hold() -> None:
    hold_tier_reads_timer()
