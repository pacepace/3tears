"""On a stall the watchdog names every live thread's stack, bounded; a failure in the watchdog itself is logged
loudly and it keeps watching; its health says so, and says it stopped once it has.

the watchdog runs on a clock the test moves (:mod:`.loop_watchdog_test_support`), so the only stalls are the
holds the test makes, however busy the machine is.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from threetears.observe import loop_watchdog as module

from packages.observe.tests.loop_watchdog_test_support import Log, Watched

#: pytest-timeout's default method is SIGALRM, the signal the held-loop dump owns: a test holding it would leave the
#: watchdog no dump to take (it leaves an alarm someone else handles alone), so these tests time out on a thread
pytestmark = pytest.mark.timeout(600, method="thread")


def _recognisable_worker_busy_in_python(stop: threading.Event) -> None:
    """a worker thread's work: the frame the stall's log must show."""
    while not stop.is_set():
        sum(range(1000))


def _deliberately_held_loop(watched: Watched, *steps: float) -> None:
    watched.hold(*steps)


async def test_a_stall_names_every_live_threads_stack_with_the_loop_marked() -> None:
    logged = Log()
    stop = threading.Event()
    worker = threading.Thread(target=_recognisable_worker_busy_in_python, args=(stop,), name="report-compute-7")
    watched = Watched(stall_seconds=0.1, beat_seconds=0.02, repeat_seconds=60.0, logger=logged)
    worker.start()
    watched.watchdog.start()
    try:
        await watched.ran_again()
        _deliberately_held_loop(watched, 0.4)
        await watched.ran_again()
    finally:
        watched.watchdog.stop()
        stop.set()
        worker.join()
    stalls = logged.stalls()
    assert len(stalls) == 1, logged.warnings
    stacks = stalls[0]["thread_stacks"]
    assert "'report-compute-7'" in stacks and "_recognisable_worker_busy_in_python" in stacks
    assert "(event loop)" in stacks and "_deliberately_held_loop" in stacks
    assert stacks.index("(event loop)") < stacks.index("report-compute-7"), "the loop's stack comes first"
    assert "'loop-watchdog'" not in stacks, "the watchdog's own stack says nothing"


def _deep(depth: int, ready: threading.Event, stop: threading.Event) -> None:
    if depth:
        _deep(depth - 1, ready, stop)
        return
    ready.set()
    stop.wait()


async def test_the_thread_stacks_are_bounded() -> None:
    logged = Log()
    stop = threading.Event()
    workers = []
    for n in range(12):
        ready = threading.Event()
        workers.append(threading.Thread(target=_deep, args=(80, ready, stop), name=f"deep-{n}"))
        workers[-1].start()
        ready.wait()
    watched = Watched(stall_seconds=0.1, beat_seconds=0.02, repeat_seconds=60.0, logger=logged, stack_chars=3000)
    watched.watchdog.start()
    try:
        await watched.ran_again()
        _deliberately_held_loop(watched, 0.3)
        await watched.ran_again()
    finally:
        watched.watchdog.stop()
        stop.set()
        for worker in workers:
            worker.join()
    stacks = logged.stalls()[0]["thread_stacks"]
    assert len(stacks) <= 3000 + 200, len(stacks)
    assert "more threads not shown" in stacks
    first = stacks.split("--- thread ")[1]
    assert first.count('File "') <= module.FRAMES, "each thread's frames are capped"


class _FailsOnce(Log):
    """a log whose first warning fails, as a frame that cannot be formatted would."""

    def __init__(self) -> None:
        super().__init__()
        self.failed = False

    def warning(self, message: str, *, extra: dict[str, Any]) -> None:
        if not self.failed:
            self.failed = True
            raise RuntimeError("a frame could not be formatted")
        super().warning(message, extra=extra)


async def test_a_failure_in_the_watchdog_is_logged_loudly_and_a_later_stall_is_still_reported() -> None:
    logged = _FailsOnce()
    watched = Watched(stall_seconds=0.1, beat_seconds=0.02, repeat_seconds=60.0, logger=logged)
    watchdog = watched.watchdog
    watchdog.start()
    try:
        await watched.ran_again()
        _deliberately_held_loop(watched, 0.3)
        await watched.ran_again()
        _deliberately_held_loop(watched, 0.3)
        await watched.ran_again()
        health = watchdog.health()
    finally:
        watchdog.stop()
    errors = [(level, message, data) for level, message, data in logged.records if level >= 40]
    assert len(errors) == 1, logged.records
    assert "loop watchdog failed" in errors[0][1] and errors[0][2]["exc_type"] == "RuntimeError"
    stalls = logged.stalls()
    assert len(stalls) == 1, "the stall after the failure is still reported"
    assert any("_deliberately_held_loop" in data["thread_stacks"] for data in stalls)
    assert health["alive"] is True and health["failures"] == 1 and health["stall_ms"] == 100
    assert health["stalls"] == 2, health
    assert health["longest_ms"] >= 300, "the longest silence seen is kept, to the ms"
    assert watchdog.health()["alive"] is False, "a stopped watchdog says it is no longer watching"
