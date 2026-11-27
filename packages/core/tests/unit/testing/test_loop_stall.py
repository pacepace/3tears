"""the loop-stall guard (``threetears.core.testing.loop_stall``): a test whose event loop ran one turn past the limit
fails, naming the function that held it; a quiet test passes; a raised limit needs a reason; it works under xdist; in a
shared suite the limit scales by the host's factor (up to its cap, past which a stall is not judged), in an exclusive
one it does not.

Each case runs its own pytest session in a subprocess with the plugin loaded: an in-process session would share this
process's wrapped ``Handle._run``, and its deliberate stalls would fail this test. Each subject session scripts the
host's probe from its conftest (``HostFactor.scripted``), so its verdicts rest on the factor the case names, never on
how busy the runner is.
"""

from __future__ import annotations

import re

import pytest

from threetears.core.testing.host_factor import HostFactor

#: the host factor's reference probe and cap, as the guard's own HostFactor holds them
REFERENCE_PROBE_SECONDS = HostFactor().reference
FACTOR_CAP = HostFactor().cap

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


def _host_at(pytester: pytest.Pytester, factor: float) -> None:
    """the subject session's host scripted at ``factor`` x the reference, and ``loop_timing`` declared."""
    pytester.makeconftest(
        f"""
from threetears.core.testing.host_factor import HostFactor

HostFactor.scripted = lambda: {factor!r}


def pytest_configure(config):
    config.addinivalue_line("markers", "loop_timing: runs in an exclusive suite")
"""
    )


def _run(pytester: pytest.Pytester, *args: str) -> pytest.RunResult:
    _host_at(pytester, 1.0)
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
    assert f"the limit is {STALL_LIMIT_SECONDS:.3f} s on this host: {STALL_LIMIT_SECONDS:.3f} s x 1.00" in out
    assert "host factor 1.00" in out


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
    _host_at(pytester, 1.0)
    pytester.makepyfile(test_subject=_LEFTOVERS)
    result = pytester.runpytest_subprocess(*_PLUGIN, "-q", "-s")
    out = result.stdout.str()
    result.assert_outcomes(passed=2, failed=1)
    assert "FAILED test_subject.py::test_spins_on_the_loop_after_the_heap" in out
    assert "held by: test_subject.hold_the_loop_with_cpu" in out
    held = [float(seconds) for seconds in re.findall(r"collected in ([0-9.]+) s", out)]
    assert len(held) == 1 and held[0] < STALL_LIMIT_SECONDS, out


#: a session that loads DuckDB and passes: its process must exit 0 too
_LOADS_DUCKDB = """
import duckdb


def test_quiet():
    assert duckdb.execute("SELECT 1").fetchall() == [(1,)]
"""


def test_a_session_that_loaded_duckdb_exits_0(pytester: pytest.Pytester) -> None:
    """Flips if the guard leaves the heap frozen at the session's end: the interpreter's shutdown then never frees
    DuckDB's module, and its import cache, destroyed by ``exit()`` after the interpreter, aborts the process
    (``Fatal Python error: gilstate_tss_set``, exit 134) though every test passed."""
    _host_at(pytester, 1.0)
    pytester.makepyfile(test_subject=_LOADS_DUCKDB)
    result = pytester.runpytest_subprocess(*_PLUGIN, "-q")
    result.assert_outcomes(passed=1)
    assert "Fatal Python error" not in result.stderr.str(), result.stderr.str()
    assert result.ret == 0, f"the session exited {result.ret}: {result.stderr.str()}"


_HOLDS = """
import time

import pytest


def hold_the_loop_with_cpu(seconds):
    began = time.thread_time()
    while time.thread_time() - began < seconds:
        sum(range(1000))


async def test_holds_under_the_scaled_limit():
    hold_the_loop_with_cpu({under})


async def test_holds_past_the_scaled_limit():
    hold_the_loop_with_cpu({past})


@pytest.mark.loop_timing
async def test_in_an_exclusive_suite():
    hold_the_loop_with_cpu({under})


@pytest.mark.loop_stall_limit(seconds={raised}, why="the test's subject is a deliberate hold, issue #1")
async def test_under_a_raised_limit_scaled():
    hold_the_loop_with_cpu({raised_under})
"""


def test_a_shared_suite_scales_the_limit_by_the_host_factor_and_an_exclusive_one_does_not(
    pytester: pytest.Pytester,
) -> None:
    """Flips if a host twice as slow as the reference is held to the reference's 0.25 s (a 0.3 s hold fails), if the
    scaled limit lets a turn past 0.25 s x 2 through, if a ``loop_timing`` test (an exclusive suite) takes the factor,
    if a raised limit is not scaled with it, or if the failure does not name the factor and the probes."""
    _host_at(pytester, 2.0)
    limit = STALL_LIMIT_SECONDS * 2
    pytester.makepyfile(
        test_subject=_HOLDS.format(
            under=STALL_LIMIT_SECONDS * 1.2,
            past=STALL_LIMIT_SECONDS * 2.4,
            raised=STALL_LIMIT_SECONDS * 1.2,
            raised_under=STALL_LIMIT_SECONDS * 2,
        )
    )
    result = pytester.runpytest_subprocess(*_PLUGIN, "-q")
    result.assert_outcomes(passed=2, failed=2)
    out = result.stdout.str()
    assert "FAILED test_subject.py::test_holds_past_the_scaled_limit" in out
    assert "FAILED test_subject.py::test_in_an_exclusive_suite" in out
    probe_ms = REFERENCE_PROBE_SECONDS * 2 * 1000
    assert (
        f"the limit is {limit:.3f} s on this host: {STALL_LIMIT_SECONDS:.3f} s x 2.00 (host factor 2.00: the probe "
        f"took {probe_ms:.2f} ms of loop-thread CPU"
    ) in out
    assert f"against its {REFERENCE_PROBE_SECONDS * 1000:.2f} ms reference" in out
    assert f"the limit is {STALL_LIMIT_SECONDS:.3f} s (an exclusive suite, loop_timing: no host factor)" in out


def test_a_stall_on_a_host_past_the_cap_could_not_be_measured(pytester: pytest.Pytester) -> None:
    """Flips if a host busier than the cap judges a turn past the capped limit as a stall red or lets it pass silently
    (it must fail as could-not-measure, naming the host's factor), or if a turn under the capped limit fails there."""
    factor = FACTOR_CAP + 1
    _host_at(pytester, factor)
    capped = STALL_LIMIT_SECONDS * FACTOR_CAP
    pytester.makepyfile(
        test_subject=_HOLDS.format(
            under=STALL_LIMIT_SECONDS * 2,
            past=capped + STALL_LIMIT_SECONDS / 2,
            raised=STALL_LIMIT_SECONDS * 1.2,
            raised_under=STALL_LIMIT_SECONDS,
        )
    )
    result = pytester.runpytest_subprocess(*_PLUGIN, "-q", "-k", "scaled_limit")
    result.assert_outcomes(passed=1, failed=1)
    out = result.stdout.str()
    assert "FAILED test_subject.py::test_holds_past_the_scaled_limit" in out
    assert "event loop stall could not be judged in call" in out
    assert f"could not measure: this host spends {factor:.2f}x the reference host's CPU" in out
    assert f"past the {FACTOR_CAP:.1f}x cap" in out
    assert f"the limit is {capped:.3f} s on this host: {STALL_LIMIT_SECONDS:.3f} s x {FACTOR_CAP:.2f}, capped" in out
    assert "event loop stalled in" not in out


_WAITS_THEN_BURNS = """
import asyncio
import concurrent.futures
import threading
import time


def burn(seconds):
    began = time.thread_time()
    while time.thread_time() - began < seconds:
        sum(range(1000))


def wait_for_the_worker(seconds):
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        pool.submit(burn, seconds).result()


def burn_on_the_loop(seconds):
    began = time.thread_time()
    while time.thread_time() - began < seconds:
        sum(range(1000))


async def test_waits_then_burns():
    await asyncio.sleep(0)
    wait_for_the_worker({waits})
    burn_on_the_loop({burns})
"""


def test_the_culprit_is_the_frame_that_spent_the_loop_threads_cpu_not_one_that_waited(
    pytester: pytest.Pytester,
) -> None:
    """Flips if the culprit is weighted by samples (wall time) rather than the loop thread's CPU: the turn waits on a
    worker's future twice as long as it burns, so most samples hold the wait, which spends no CPU of its own."""
    _host_at(pytester, 1.0)
    pytester.makepyfile(
        test_subject=_WAITS_THEN_BURNS.format(waits=STALL_LIMIT_SECONDS * 4, burns=STALL_LIMIT_SECONDS * 1.6)
    )
    result = pytester.runpytest_subprocess(*_PLUGIN, "-q")
    result.assert_outcomes(failed=1)
    out = result.stdout.str()
    assert "held by: test_subject.burn_on_the_loop" in out, out
    assert re.search(r"held by: test_subject\.burn_on_the_loop .*\(([0-9.]+) s of the ([0-9.]+) s CPU", out), out


_LOOPS_IN_ENDED_THREADS = """
import asyncio
import threading
import time


def test_loops_run_in_threads_that_end():
    for _ in range(50):
        worker = threading.Thread(target=asyncio.run, args=(asyncio.sleep(0),))
        worker.start()
        worker.join()
    time.sleep(0.3)  # the watcher looks at every ended loop thread, many times over
"""


def test_the_watcher_survives_loop_threads_that_have_ended(pytester: pytest.Pytester) -> None:
    """Flips if the watcher reads an ended loop thread's CPU clock through its ident (a freed ``pthread_t`` on Linux):
    that killed test processes with SIGSEGV on the pool (2026-11-26)."""
    _host_at(pytester, 1.0)
    pytester.makepyfile(test_subject=_LOOPS_IN_ENDED_THREADS)
    result = pytester.runpytest_subprocess(*_PLUGIN, "-q")
    result.assert_outcomes(passed=1)
    assert result.ret == 0, result.stderr.str()
