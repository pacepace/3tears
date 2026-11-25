"""each stall's line names its cause: the collector, with its generation and length, or every thread's stack taken
while the loop was held -- even when whatever held it kept the interpreter lock throughout, so no Python thread could
look.

these run on the real clock: the held-loop dump is written by ``faulthandler``'s own timer, which no test clock moves.
each hold is several times the bound, and the loop that is not held is held to a bound far above any scheduling
hiccup, so a slow machine cannot make or break a stall.
"""

from __future__ import annotations

import asyncio
import gc
import os
import signal
import sys
import tempfile
import time
from collections.abc import Callable
from typing import Any

import pytest

from threetears.observe.loop_watchdog import LoopWatchdog

from packages.observe.tests.loop_watchdog_test_support import HANG_SECONDS, Log

#: pytest-timeout's default method is SIGALRM, the signal the held-loop dump owns: a test holding it would leave the
#: watchdog no dump to take (it leaves an alarm someone else handles alone), so these tests time out on a thread
pytestmark = pytest.mark.timeout(600, method="thread")

#: how long each deliberate hold lasts: several times the bound the tests set
HOLD_SECONDS = 0.3

#: the HOLD_SECONDS holds and the forced collection are these tests' subject, made on the loop on purpose: the
#: loop-stall guard (threetears.core.testing.loop_stall, 0.25 s of loop CPU a turn) would fail them for the very stall
#: the watchdog must log. Ten times HOLD_SECONDS leaves a loaded pool host room (0.30-0.42 s measured on monad-1)
_DELIBERATE_HOLD = pytest.mark.loop_stall_limit(
    seconds=10 * HOLD_SECONDS, why="the test holds the loop on purpose: the stall is what the watchdog must log"
)

#: the held-loop dump is Linux's alone: off Linux no SIGALRM is armed (``loop_watchdog._HeldStacks.claim``: on macOS the
#: signal ended the loop thread's wait for the interpreter lock with an error CPython takes as fatal)
_LINUX_DUMP = pytest.mark.skipif(sys.platform != "linux", reason="the held-loop dump is armed on Linux alone")


def _calibrated(work: Callable[[int], object], size: int, seconds: float) -> int:
    """the size of ``work`` that takes about ``seconds``, from its fastest of three runs at ``size``, scaled up (linear
    work); the fastest, so a cold first run does not make the hold short."""
    took = []
    for _ in range(3):
        began = time.perf_counter()
        work(size)
        took.append(time.perf_counter() - began)
    return max(size, int(size * seconds / max(min(took), 1e-6)))


def _sum_in_c(n: int) -> int:
    """C code that keeps the interpreter lock for its whole run: ``sum`` over a range never returns to the
    interpreter's loop, so no other thread runs Python meanwhile."""
    return sum(range(n))


def _holds_the_lock_in_c(n: int) -> int:
    """the frame the held-loop dump must name."""
    return _sum_in_c(n)


async def _until(logged: Log, matches: Callable[[str, dict[str, Any]], bool]) -> list[dict[str, Any]]:
    """wait until the watchdog has logged a warning ``matches`` accepts; what those warnings carried."""
    async with asyncio.timeout(HANG_SECONDS):
        while not (found := [data for message, data in logged.warnings if matches(message, data)]):
            await asyncio.sleep(0.01)
    return found


def _held_in_c(message: str, data: dict[str, Any]) -> bool:
    """the line of the stall the test made: its dump names the code that held the loop."""
    return "_holds_the_lock_in_c" in data.get("held_stacks", "")


def _with_the_collector(message: str, data: dict[str, Any]) -> bool:
    """the line of a stall a collection overlapped."""
    return "gc" in data and "cause" in data


@_LINUX_DUMP
@_DELIBERATE_HOLD
async def test_a_loop_held_with_the_lock_in_c_is_logged_with_the_loops_stack_taken_while_held() -> None:
    logged = Log()
    size = _calibrated(_sum_in_c, 1_000_000, HOLD_SECONDS)
    watchdog = LoopWatchdog(stall_seconds=0.05, logger=logged)
    watchdog.start()
    try:
        assert watchdog.health()["stack_dumps"], "the first watchdog owns the process's dump"
        await asyncio.sleep(0.05)
        _holds_the_lock_in_c(size)
        stalls = await _until(logged, _held_in_c)
        health = watchdog.health()
    finally:
        watchdog.stop()
    # no Python thread ran while the lock was held: a line the watchdog thread logs is from after it let go, its
    # stacks whatever ran next. the dump taken during the hold names the code that held it
    assert len(stalls) == 1, logged.warnings
    stall = stalls[0]
    assert stall["stalled_seconds"] >= 0.1, "held past twice the bound"
    assert stall["cause"] == "not the collector: see held_stacks"
    held = stall["held_stacks"]
    assert "_holds_the_lock_in_c" in held and "(event loop)" in held, held
    assert held.index("(event loop)") < held.index("_holds_the_lock_in_c"), "the loop's frames follow its header"
    assert "Thread 0x" not in held, "the loop thread's stack alone: the all-threads walk crashes on exiting threads"
    assert health["stalls"] >= 1 and health["longest_ms"] >= 100


def _cyclic_garbage(n: int) -> None:
    """``n`` small reference cycles, unreachable at once: only the collector frees them."""
    for _ in range(n):
        a: list[Any] = []
        b: list[Any] = [a]
        a.append(b)


def _timed_collection(n: int) -> float:
    """how long a full collection takes with ``n`` more containers alive for it to walk."""
    kept: list[list[Any]] = [[] for _ in range(n)]
    began = time.perf_counter()
    gc.collect()
    took = time.perf_counter() - began
    del kept
    return took


@_DELIBERATE_HOLD
async def test_a_forced_collection_is_timed_and_names_the_stall_it_caused() -> None:
    logged = Log()
    # enough containers for a full collection of about 80 ms, four times the bound below; a cap keeps it small
    size = min(4_000_000, int(500_000 * 0.08 / max(_timed_collection(500_000), 1e-4)))
    kept: list[list[Any]] = [[] for _ in range(size)]
    watchdog = LoopWatchdog(stall_seconds=0.02, gc_slow_seconds=0.01, logger=logged)
    watchdog.start()
    try:
        await asyncio.sleep(0.05)
        # no automatic collection between making the garbage and collecting it, to free some of it first
        gc.disable()
        try:
            _cyclic_garbage(1000)
            gc.collect()
        finally:
            gc.enable()
        slow = await _until(logged, lambda message, data: "garbage collection" in message and data["collected"] >= 2000)
        stalls = await _until(logged, _with_the_collector)
        health = watchdog.health()
    finally:
        watchdog.stop()
        del kept
    collection = slow[0]
    assert collection["generation"] == 2 and collection["ms"] >= 10 and collection["collected"] >= 2000, collection
    stall = stalls[0]
    assert stall["cause"].startswith("gc gen 2, "), stall
    assert stall["gc"]["longest"][0]["generation"] == 2 and stall["gc"]["ms"] >= stall["stalled_seconds"] * 500
    assert health["slow_collections"] >= 1 and health["longest_collection_ms"] >= 10


async def test_a_loop_that_runs_and_collects_quickly_logs_nothing() -> None:
    logged = Log()
    watchdog = LoopWatchdog(stall_seconds=0.25, gc_slow_seconds=0.1, logger=logged)
    watchdog.start()
    try:
        for _ in range(30):
            _cyclic_garbage(100)
            gc.collect(0)
            await asyncio.sleep(0.01)
        # a look after the last beat, so nothing is still on its way to the log
        await asyncio.sleep(0.05)
        health = watchdog.health()
    finally:
        watchdog.stop()
    assert logged.warnings == [] and logged.records == []
    assert health["stalls"] == 0 and health["slow_collections"] == 0


@_LINUX_DUMP
async def test_the_process_has_one_dump_owner_and_it_is_given_back() -> None:
    callbacks = len(gc.callbacks)
    first, second = LoopWatchdog(logger=Log()), LoopWatchdog(logger=Log())
    first.start()
    second.start()
    try:
        assert first.health()["stack_dumps"] and not second.health()["stack_dumps"]
    finally:
        second.stop()
        first.stop()
    third = LoopWatchdog(logger=Log())
    third.start()
    try:
        assert third.health()["stack_dumps"], "a stopped owner gives the dump back"
    finally:
        third.stop()
    assert len(gc.callbacks) == callbacks, "the collector's clock is removed on stop"


def test_a_held_loop_dump_is_bounded_and_marks_the_loop() -> None:
    watchdog = LoopWatchdog(logger=Log(), frames=3, stack_chars=600)
    frames = "".join(f'  File "/app/main.py", line {n} in f{n}\n' for n in range(10))
    held = watchdog.format_held_stacks(f"Stack (most recent call first):\n{frames}")
    assert held.startswith("Stack (most recent call first): (event loop)\n"), held
    assert held.count('File "') == 3 and "7 outer frames not shown" in held
    long = "".join(f'  File "/app/{"x" * 200}.py", line {n} in f{n}\n' for n in range(10))
    bounded = LoopWatchdog(logger=Log(), stack_chars=600).format_held_stacks(f"Stack (most recent call first):\n{long}")
    assert len(bounded) <= 600 + 100 and "bounded at 600 chars" in bounded


@_LINUX_DUMP
async def test_a_dump_that_cannot_be_set_up_leaves_the_watchdog_running(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*args: object, **kwargs: object) -> None:
        raise OSError("no in-memory or temporary file here")

    monkeypatch.setattr(os, "memfd_create", broken, raising=False)
    monkeypatch.setattr(tempfile, "TemporaryFile", broken)
    logged = Log()
    # a bound no scheduling hiccup reaches: on a loaded machine (a parallel suite) a 50 ms one logged a stall here
    watchdog = LoopWatchdog(stall_seconds=HANG_SECONDS, logger=logged)
    watchdog.start()
    try:
        assert watchdog.health()["stack_dumps"] is False
    finally:
        watchdog.stop()
    assert [message for message, _ in logged.warnings] == [
        "the loop watchdog runs without held-loop dumps: the dump could not be set up"
    ]


def _theirs(signum: int, frame: Any) -> None:
    """someone else's ``SIGALRM`` handler."""


async def test_a_watchdog_leaves_an_alarm_someone_else_handles_alone() -> None:
    signal.signal(signal.SIGALRM, _theirs)
    try:
        watchdog = LoopWatchdog(logger=Log())
        watchdog.start()
        try:
            assert not watchdog.health()["stack_dumps"]
        finally:
            watchdog.stop()
        assert signal.getsignal(signal.SIGALRM) is _theirs
    finally:
        signal.signal(signal.SIGALRM, signal.SIG_DFL)


@_LINUX_DUMP
@_DELIBERATE_HOLD
async def test_an_alarm_taken_over_and_given_back_is_handled_again_by_the_next_watchdog() -> None:
    first = LoopWatchdog(logger=Log())
    first.start()
    try:
        signal.signal(signal.SIGALRM, _theirs)
        async with asyncio.timeout(HANG_SECONDS):
            while first.health()["stack_dumps"]:
                await asyncio.sleep(0.01)
        await asyncio.sleep(0.15)
    finally:
        first.stop()
        signal.signal(signal.SIGALRM, signal.SIG_DFL)
    # the next watchdog's alarm must reach a dump: were SIGALRM still at its default, it would end this process
    logged = Log()
    size = _calibrated(_sum_in_c, 1_000_000, HOLD_SECONDS)
    second = LoopWatchdog(stall_seconds=0.05, logger=logged)
    second.start()
    try:
        assert second.health()["stack_dumps"]
        await asyncio.sleep(0.05)
        _holds_the_lock_in_c(size)
        assert await _until(logged, _held_in_c)
    finally:
        second.stop()


@_LINUX_DUMP
async def test_a_watchdog_whose_alarm_is_taken_over_stops_arming_it_and_says_so() -> None:
    logged = Log()
    watchdog = LoopWatchdog(logger=logged)
    watchdog.start()
    try:
        assert watchdog.health()["stack_dumps"]
        signal.signal(signal.SIGALRM, _theirs)
        taken = await _until(logged, lambda message, data: "SIGALRM" in message)
        # past the last alarm a beat set before the look saw the takeover
        await asyncio.sleep(0.15)
        assert not watchdog.health()["stack_dumps"] and taken[0]["handler"].startswith("<function _theirs")
        assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), "the beats no longer set the alarm"
    finally:
        watchdog.stop()
        assert signal.getsignal(signal.SIGALRM) is _theirs, "their handler is left as it is"
        signal.signal(signal.SIGALRM, signal.SIG_DFL)


@pytest.mark.skipif(sys.platform == "linux", reason="off Linux alone")
async def test_off_linux_no_alarm_is_claimed_so_no_signal_reaches_the_loop_thread() -> None:
    """Flips if a SIGALRM dump is claimed off Linux: on macOS the alarm, fired in a stall, ended the loop thread's wait
    for the interpreter lock with an error CPython takes as fatal (2026-11-22, three pod crashes in one evening)."""
    import signal

    watchdog = LoopWatchdog(stall_seconds=1.0, logger=Log())
    watchdog.start()
    try:
        assert not watchdog.health()["stack_dumps"], "no dump is claimed off Linux"
        assert signal.getsignal(signal.SIGALRM) == signal.SIG_DFL
    finally:
        watchdog.stop()
