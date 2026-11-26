"""the loop-stall guard (``threetears.core.testing.loop_stall``): a test whose event loop ran one turn past the limit
fails, naming the function that held it; a quiet test passes; a raised limit needs a reason; it works under xdist.

Each case runs its own pytest session in a subprocess with the plugin loaded: an in-process session would share this
process's wrapped ``Handle._run``, and its deliberate stalls would fail this test.
"""

from __future__ import annotations

import re

import pytest

#: the guard's limit (``loop_stall._STALL_LIMIT_SECONDS``); each failure says it, which the first test pins
STALL_LIMIT_SECONDS = 0.25

_PLUGIN = ("-p", "threetears.core.testing.loop_stall", "-p", "no:cacheprovider", "-o", "asyncio_mode=auto")

_SPINNER = f"""
import asyncio
import time

import pytest


def hold_the_loop_with_cpu(seconds):
    began = time.thread_time()
    while time.thread_time() - began < seconds:
        sum(range(1000))


async def test_spins_on_the_loop():
    await asyncio.sleep(0)
    hold_the_loop_with_cpu({STALL_LIMIT_SECONDS * 2})


async def test_spins_off_the_loop():
    await asyncio.to_thread(hold_the_loop_with_cpu, {STALL_LIMIT_SECONDS * 2})


async def test_sleeps():
    await asyncio.sleep({STALL_LIMIT_SECONDS * 2})


@pytest.mark.loop_stall_limit(seconds={STALL_LIMIT_SECONDS * 8}, why="the test's subject is a deliberate hold, issue #1")
async def test_spins_under_a_raised_limit():
    hold_the_loop_with_cpu({STALL_LIMIT_SECONDS * 2})


def test_sync_runs_its_own_loop():
    async def body():
        hold_the_loop_with_cpu({STALL_LIMIT_SECONDS * 2})

    asyncio.run(body())
"""


def _run(pytester: pytest.Pytester, *args: str) -> pytest.RunResult:
    pytester.makepyfile(test_subject=_SPINNER)
    return pytester.runpytest_subprocess(*_PLUGIN, *args)


def test_a_turn_held_past_the_limit_fails_the_test_naming_the_function_that_held_it(pytester: pytest.Pytester) -> None:
    """Flips if a CPU hold on the loop passes (in an async test or a sync test's own loop), if the failure does not
    name the function that held it, or if work off the loop, a sleep or a raised limit fails a test."""
    result = _run(pytester, "-q")
    result.assert_outcomes(passed=3, failed=2)
    out = result.stdout.str()
    assert "FAILED test_subject.py::test_spins_on_the_loop" in out
    assert "FAILED test_subject.py::test_sync_runs_its_own_loop" in out
    assert "held by: test_subject.hold_the_loop_with_cpu" in out
    assert "the event loop was held" in out
    assert f"the limit is {STALL_LIMIT_SECONDS:.3f} s" in out


def test_the_guard_fails_a_stall_on_an_xdist_worker(pytester: pytest.Pytester) -> None:
    """Flips if a worker does not load the guard, or its failure does not reach the controller's report."""
    result = _run(pytester, "-q", "-n", "2", "-p", "xdist")
    result.assert_outcomes(passed=3, failed=2)
    assert "held by: test_subject.hold_the_loop_with_cpu" in result.stdout.str()


@pytest.mark.parametrize(
    "marker",
    [
        f"seconds={STALL_LIMIT_SECONDS * 4}",
        f'seconds={STALL_LIMIT_SECONDS * 4}, why="slow"',
        f'seconds={STALL_LIMIT_SECONDS / 2}, why="a reason long enough to be read as one"',
        f'{STALL_LIMIT_SECONDS * 4}, why="a reason long enough to be read as one"',
    ],
    ids=["no-why", "short-why", "lowers-the-limit", "positional-seconds"],
)
def test_a_raised_limit_that_does_not_raise_it_or_say_why_refuses_the_run(
    pytester: pytest.Pytester, marker: str
) -> None:
    """Flips if a limit marker with no reason, a one-word reason, a lower limit or a positional seconds is accepted."""
    pytester.makepyfile(
        test_subject=f"""
import pytest

@pytest.mark.loop_stall_limit({marker})
async def test_x():
    pass
"""
    )
    result = pytester.runpytest_subprocess(*_PLUGIN, "-q")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*loop-stall limits that cannot stand*", "*test_subject.py::test_x*loop_stall_limit*"])


#: the heap an earlier test leaves: a million tracked objects (lists). The next test runs :data:`_COLLECTIONS` full
#: collections in one turn, standing in for one automatic full collection on a slower host and a bigger heap: unfrozen,
#: that turn held the loop 2.97 s of its CPU (~0.12 s a walk of the subject session's heap; CPython 3.14, M-series
#: laptop, 2026-11-25), 12x the limit; frozen before the test, 0.000 s, the walk being of the test's own few objects
_LEFT_OBJECTS = 1_000_000
_COLLECTIONS = 25

_LEFTOVERS = f"""
import asyncio
import gc
import time

#: what the first test leaves for the session, as an earlier test's objects outlive it
LEFT = []


def hold_the_loop_with_cpu(seconds):
    began = time.thread_time()
    while time.thread_time() - began < seconds:
        sum(range(1000))


def test_leaves_a_big_heap():
    LEFT.extend([index] for index in range({_LEFT_OBJECTS}))


async def test_collects_in_one_turn():
    await asyncio.sleep(0)
    began = time.thread_time()
    for _ in range({_COLLECTIONS}):
        gc.collect()
    print(f"collected in {{time.thread_time() - began:.3f}} s of the loop's CPU")
    assert len(LEFT) == {_LEFT_OBJECTS}


async def test_spins_on_the_loop_after_the_heap():
    await asyncio.sleep(0)
    hold_the_loop_with_cpu({STALL_LIMIT_SECONDS * 2})
"""


def test_a_collection_walking_an_earlier_tests_heap_is_not_charged_to_the_test(pytester: pytest.Pytester) -> None:
    """Flips if the guard charges a test for the garbage collector walking what an earlier test left (the heap is not
    frozen before the test), or if freezing it hides a test's own CPU hold on the loop."""
    pytester.makepyfile(test_subject=_LEFTOVERS)
    result = pytester.runpytest_subprocess(*_PLUGIN, "-q", "-s")
    out = result.stdout.str()
    result.assert_outcomes(passed=2, failed=1)
    assert "FAILED test_subject.py::test_spins_on_the_loop_after_the_heap" in out
    assert "held by: test_subject.hold_the_loop_with_cpu" in out
    held = [float(seconds) for seconds in re.findall(r"collected in ([0-9.]+) s", out)]
    assert len(held) == 1 and held[0] < STALL_LIMIT_SECONDS, out
