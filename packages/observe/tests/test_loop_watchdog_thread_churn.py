"""the held-loop dump survives threads that come and go while it is written.

the reports tool pod died of SIGSEGV in a local burst replay (2026-11-21, CPython 3.14.3): ``faulthandler``'s
all-threads dump, answering the watchdog's alarm, walked the interpreter's thread list while a thread exiting freed its
state (``tstate_is_freed`` under ``_Py_DumpTracebackThreads``; CPython gh-116008). the test runs that pod's shape in a
child process -- the loop held past the bound hundreds of times, so the alarm fires each time, while threads start and
exit in a tight loop -- so a crash fails the test instead of killing the runner. on the all-threads dump the child died
of SIGSEGV within a second, every run (5 of 5 on macOS and 3 of 3 on Linux, CPython 3.14.3); it must now run to the
end, with every dump the loop thread's own -- on Linux a process-wide alarm interrupted a churning thread 1 time in 7.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import threading
import time
from typing import Any

import pytest

from threetears.observe.loop_watchdog import LoopWatchdog

from packages.observe.tests.loop_watchdog_test_support import HANG_SECONDS, Log

#: pytest-timeout's default method is SIGALRM, the signal the held-loop dump owns: a test holding it would leave the
#: watchdog no dump to take (it leaves an alarm someone else handles alone), so these tests time out on a thread
pytestmark = pytest.mark.timeout(600, method="thread")

#: how long the child churns threads under the alarm; the all-threads dump crashed within the first second
#: the held-loop dump is Linux's alone: off Linux no SIGALRM is armed (``loop_watchdog._HeldStacks.claim``: on macOS the
#: signal ended the loop thread's wait for the interpreter lock with an error CPython takes as fatal)
_LINUX_DUMP = pytest.mark.skipif(sys.platform != "linux", reason="the held-loop dump is armed on Linux alone")

CHURN_SECONDS = 2.0

#: the fewest dumps the child must take for the run to count: each one is a chance for the old walk to crash, and a
#: run takes several hundred (a hold every ~6 ms)
MIN_DUMPS = 50

_CHILD = """
import asyncio, json, signal, sys, threading, time

from threetears.observe import loop_watchdog as module

SECONDS = float(sys.argv[1])
dumps = []
taken = module._HeldStacks.take


def take(self):
    dump = taken(self)
    if dump:
        dumps.append(dump)
    return dump


module._HeldStacks.take = take


class Quiet:
    def warning(self, *args, **kwargs):
        pass

    def log(self, *args, **kwargs):
        pass


def churn(stop):
    # threads that start and exit as fast as they can: each exit frees a thread state
    while not stop.is_set():
        threads = [threading.Thread(target=lambda: None) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()


def _hold_the_loop():
    # past the bound, so the alarm fires; a sleep lets go of the lock, so the churn runs while the dump is written
    time.sleep(0.004)


async def main():
    watchdog = module.LoopWatchdog(stall_seconds=0.002, beat_seconds=0.001, logger=Quiet())
    watchdog.start()
    dumping = watchdog.health()["stack_dumps"]
    stop = threading.Event()
    spawners = [threading.Thread(target=churn, args=(stop,)) for _ in range(4)]
    for spawner in spawners:
        spawner.start()
    end = time.monotonic() + SECONDS
    while time.monotonic() < end:
        _hold_the_loop()
        await asyncio.sleep(0.0015)
    stop.set()
    for spawner in spawners:
        spawner.join()
    watchdog.stop()
    print(json.dumps({"stack_dumps": dumping, "dumps": dumps}))


asyncio.run(main())
"""


@_LINUX_DUMP
@pytest.mark.uncacheable("a fresh interpreter churning threads under the alarm is the point")
def test_the_held_loop_dump_survives_threads_exiting_while_it_is_written() -> None:
    child = subprocess.run(  # noqa: S603 - this interpreter, a fixed script
        [sys.executable, "-c", _CHILD, str(CHURN_SECONDS)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert child.returncode == 0, f"the child died (code {child.returncode}): {child.stderr[-2000:]}"
    result = json.loads(child.stdout.strip().splitlines()[-1])
    assert result["stack_dumps"], "the loop runs on the child's main thread: the watchdog takes dumps"
    dumps: list[str] = result["dumps"]
    assert len(dumps) >= MIN_DUMPS, f"only {len(dumps)} dumps: the alarm did not race the churn enough to count"
    for dump in dumps:
        assert dump.count("Stack (most recent call first):") == 1, dump
        assert "Thread 0x" not in dump, f"a dump walked other threads: {dump}"
    # the loop thread's every time: never a churning thread's (its frames run from ``threading``'s bootstrap). a
    # loop dump may stop short at ``<invalid frame>`` when the alarm caught a frame being built, so it is not
    # required to reach the loop's own frames
    stray = [dump for dump in dumps if "in churn" in dump or "in _bootstrap" in dump]
    assert not stray, f"{len(stray)} of {len(dumps)} dumps were another thread's: {stray[0]}"
    assert any("_hold_the_loop" in dump for dump in dumps), "the dumps are taken while the loop is held"


#: how long the loop off the main thread is held: several times the bound the test sets
HOLD_SECONDS = 0.3


def _held_off_the_main_thread() -> None:
    """the frame the dump of a loop off the main thread must name."""
    time.sleep(HOLD_SECONDS)


def test_a_loop_off_the_main_thread_is_dumped_on_linux_and_not_at_all_elsewhere() -> None:
    # Linux aims the alarm at the loop thread itself; elsewhere a process-wide alarm interrupts the main thread, whose
    # stack is not the loop's, so no dump is taken
    logged = Log()
    seen: dict[str, Any] = {}

    async def watch() -> None:
        watchdog = LoopWatchdog(stall_seconds=0.05, logger=logged)
        watchdog.start()
        try:
            seen["health"] = watchdog.health()
            await asyncio.sleep(0.05)
            _held_off_the_main_thread()
            async with asyncio.timeout(HANG_SECONDS):
                while not logged.ends():
                    await asyncio.sleep(0.01)
        finally:
            watchdog.stop()

    thread = threading.Thread(target=lambda: asyncio.run(watch()), name="off-main-loop")
    thread.start()
    # no SIGALRM timeout: the module times out on a thread, or the watchdog would leave the alarm alone and take no dump
    thread.join(timeout=3 * HANG_SECONDS)
    assert not thread.is_alive(), "the loop off the main thread did not finish"
    held = [end.get("held_stacks", "") for end in logged.ends()]
    assert seen["health"]["alive"], seen
    if sys.platform == "linux":
        assert seen["health"]["stack_dumps"], seen
        assert any("_held_off_the_main_thread" in stacks for stacks in held), held
        assert not any("test_a_loop_off_the_main_thread" in stacks for stacks in held), "the main thread's stack"
    else:
        assert not seen["health"]["stack_dumps"], seen
        assert held == [""], held
