"""``PeriodicTask``: the one start/stop/interval loop every background sweeper shares.

3tears carried the same ``start`` / ``stop`` / ``while: sleep; tick`` shell eight times (presence
sweeper, registry health, mcp catch-up, the write-buffer flusher, …) and consumers carry more, each
differing in whether it sleeps first, how ``stop`` waits, and what one bad tick does to the loop. These
tests pin the single contract they all move onto.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta

import pytest

from threetears.observe import PeriodicTask

_LOG = logging.getLogger("test.periodic")


def _tasks_named(name: str) -> list[asyncio.Task[object]]:
    return [t for t in asyncio.all_tasks() if t.get_name() == name and not t.done()]


async def _wait_for(event: asyncio.Event, timeout: float = 1.0) -> None:
    await asyncio.wait_for(event.wait(), timeout=timeout)


async def test_start_twice_runs_one_loop() -> None:
    periodic = PeriodicTask(lambda: asyncio.sleep(0), interval=10, name="once", logger=_LOG)
    periodic.start()
    periodic.start()
    try:
        assert periodic.running
        assert len(_tasks_named("once")) == 1
    finally:
        await periodic.stop()


async def test_stop_is_idempotent_and_safe_before_start() -> None:
    periodic = PeriodicTask(lambda: asyncio.sleep(0), interval=10, name="idle", logger=_LOG)
    await periodic.stop()
    periodic.start()
    await periodic.stop()
    await periodic.stop()
    assert not periodic.running
    assert _tasks_named("idle") == []


async def test_it_can_be_started_again_after_a_stop() -> None:
    ticked = asyncio.Event()

    async def tick() -> None:
        ticked.set()

    periodic = PeriodicTask(tick, interval=10, name="restart", logger=_LOG, first_delay=0)
    periodic.start()
    await periodic.stop()
    ticked.clear()
    periodic.start()
    try:
        await _wait_for(ticked)
    finally:
        await periodic.stop()


async def test_the_default_sleeps_before_the_first_tick() -> None:
    ticked = asyncio.Event()

    async def tick() -> None:
        ticked.set()

    periodic = PeriodicTask(tick, interval=10, name="sleep-first", logger=_LOG)
    periodic.start()
    try:
        await asyncio.sleep(0.05)
        assert not ticked.is_set()
    finally:
        await periodic.stop()


async def test_a_zero_first_delay_ticks_before_the_first_sleep() -> None:
    ticked = asyncio.Event()

    async def tick() -> None:
        ticked.set()

    periodic = PeriodicTask(tick, interval=10, name="tick-first", logger=_LOG, first_delay=0)
    periodic.start()
    try:
        await _wait_for(ticked)
    finally:
        await periodic.stop()


async def test_first_delay_sets_only_the_first_sleep() -> None:
    stamps: list[float] = []
    loop = asyncio.get_running_loop()
    first = asyncio.Event()

    async def tick() -> None:
        stamps.append(loop.time())
        first.set()

    started = loop.time()
    periodic = PeriodicTask(tick, interval=10, name="first-delay", logger=_LOG, first_delay=0.05)
    periodic.start()
    try:
        await _wait_for(first)
        assert stamps[0] - started >= 0.04, "the first tick waits first_delay, not the interval"
        await asyncio.sleep(0.1)
        assert len(stamps) == 1, "after the first tick the configured interval applies"
    finally:
        await periodic.stop()


def test_a_negative_first_delay_is_refused() -> None:
    with pytest.raises(ValueError, match="first_delay"):
        PeriodicTask(lambda: asyncio.sleep(0), interval=1, name="bad-first", logger=_LOG, first_delay=-1)


async def test_a_failing_tick_is_logged_and_the_loop_keeps_going(caplog: pytest.LogCaptureFixture) -> None:
    calls = 0
    second = asyncio.Event()

    async def tick() -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("tick blew up")
        second.set()

    periodic = PeriodicTask(tick, interval=0.01, name="flaky", logger=_LOG, first_delay=0)
    with caplog.at_level(logging.WARNING, logger="test.periodic"):
        periodic.start()
        try:
            await _wait_for(second)
        finally:
            await periodic.stop()
    failures = [r for r in caplog.records if r.levelno == logging.WARNING and "flaky" in r.getMessage()]
    assert failures, "a failing tick is logged"
    assert failures[0].exc_info is not None and "tick blew up" in str(failures[0].exc_info[1])


async def test_stop_during_a_tick_cancels_it_and_returns() -> None:
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def tick() -> None:
        entered.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    periodic = PeriodicTask(tick, interval=10, name="stuck", logger=_LOG, first_delay=0)
    periodic.start()
    await _wait_for(entered)
    await asyncio.wait_for(periodic.stop(), timeout=1.0)
    assert cancelled.is_set()
    assert not periodic.running


async def test_a_returned_delay_overrides_only_the_next_sleep() -> None:
    stamps: list[float] = []
    loop = asyncio.get_running_loop()
    second = asyncio.Event()

    async def tick() -> float | None:
        stamps.append(loop.time())
        if len(stamps) == 2:
            second.set()
        return 0.01 if len(stamps) == 1 else None

    periodic = PeriodicTask(tick, interval=10, name="fast-once", logger=_LOG, first_delay=0)
    periodic.start()
    try:
        await _wait_for(second)
        await asyncio.sleep(0.1)
        assert len(stamps) == 2, "after the override, the configured interval applies again"
    finally:
        await periodic.stop()


async def test_run_once_runs_one_isolated_tick(caplog: pytest.LogCaptureFixture) -> None:
    calls = 0

    async def tick() -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("boom")

    periodic = PeriodicTask(tick, interval=10, name="manual", logger=_LOG)
    with caplog.at_level(logging.WARNING, logger="test.periodic"):
        await periodic.run_once()
    assert calls == 1
    assert any("manual" in r.getMessage() for r in caplog.records)
    assert not periodic.running, "run_once does not start the loop"


@pytest.mark.parametrize("interval", [0, -1, timedelta(0)])
def test_the_interval_must_be_positive(interval: float | timedelta) -> None:
    with pytest.raises(ValueError, match="interval"):
        PeriodicTask(lambda: asyncio.sleep(0), interval=interval, name="bad", logger=_LOG)


def test_a_timedelta_interval_is_accepted() -> None:
    periodic = PeriodicTask(lambda: asyncio.sleep(0), interval=timedelta(seconds=30), name="td", logger=_LOG)
    assert periodic.interval == 30.0


async def test_a_tick_can_stop_its_own_loop() -> None:
    """A "stop when the work is done" tick must not recurse into its own cancellation."""
    ticks = 0
    periodic: PeriodicTask

    async def tick() -> None:
        nonlocal ticks
        ticks += 1
        await periodic.stop()

    periodic = PeriodicTask(tick, interval=0.01, name="self-stop", logger=_LOG, first_delay=0)
    periodic.start()
    for _ in range(50):
        await asyncio.sleep(0.01)
        if not periodic.running:
            break
    assert not periodic.running
    assert ticks == 1
    assert _tasks_named("self-stop") == []


async def test_stop_returns_even_when_a_tick_swallows_its_cancellation() -> None:
    entered = asyncio.Event()

    async def tick() -> None:
        entered.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            pass  # a badly behaved tick

    periodic = PeriodicTask(tick, interval=10, name="swallower", logger=_LOG, first_delay=0)
    periodic.start()
    await _wait_for(entered)
    await asyncio.wait_for(periodic.stop(), timeout=1.0)
    assert not periodic.running


async def test_concurrent_stops_both_wait_for_the_loop_to_end() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def tick() -> None:
        entered.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            await release.wait()  # slow, well-behaved teardown
            raise

    periodic = PeriodicTask(tick, interval=10, name="slow-stop", logger=_LOG, first_delay=0)
    periodic.start()
    await _wait_for(entered)
    first = asyncio.create_task(periodic.stop())
    second = asyncio.create_task(periodic.stop())
    await asyncio.sleep(0.05)
    assert not first.done() and not second.done(), "neither stop returns while the loop is still ending"
    release.set()
    await asyncio.wait_for(asyncio.gather(first, second), timeout=1.0)
    assert _tasks_named("slow-stop") == []


@pytest.mark.parametrize("returned", [True, -1.0, float("nan"), "soon"])
async def test_a_returned_value_that_is_not_a_delay_keeps_the_interval(returned: object) -> None:
    async def tick() -> object:
        return returned

    periodic = PeriodicTask(tick, interval=10, name="odd-return", logger=_LOG)
    assert await periodic.run_once() is None


async def test_run_once_returns_the_requested_delay() -> None:
    async def tick() -> float:
        return 2.5

    assert await PeriodicTask(tick, interval=10, name="delay", logger=_LOG).run_once() == 2.5


async def test_a_failed_tick_is_logged_with_the_callers_message(caplog: pytest.LogCaptureFixture) -> None:
    async def tick() -> None:
        raise RuntimeError("boom")

    periodic = PeriodicTask(tick, interval=10, name="named", logger=_LOG, failure_message="presence sweep failed")
    with caplog.at_level(logging.WARNING, logger="test.periodic"):
        await periodic.run_once()
    assert [r.getMessage() for r in caplog.records] == ["presence sweep failed"]
