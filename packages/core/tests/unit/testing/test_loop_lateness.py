"""``threetears.core.testing.loop_lateness``: a test loop's lateness counts only where the host's own idle loop, in a
process of its own, was not late at the same moment.

Pure judgements over recorded ticks (no timing), and one short timed run of the host's ticker itself.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import os
import subprocess
import sys
import threading
import time
import inspect
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path

import pytest

from threetears.core.testing.loop_lateness import (
    CouldNotMeasure,
    HostTicker,
    Tick,
    added_lateness,
    hold_the_loop,
    host_lateness,
    latest,
    ticks_while,
    trusted,
)
from threetears.core.testing import loop_lateness


def test_lateness_the_host_shared_at_the_same_moment_is_not_the_loops() -> None:
    """a tick 20 ms late while the host loop was late over the same span adds nothing."""
    host = [(0.0, 0.001), (0.001, 0.0231), (0.0231, 0.0242)]
    assert added_lateness([(0.002, 0.023)], host) == pytest.approx(0.0, abs=1e-9)


def test_a_load_spike_at_another_moment_credits_none_of_a_stall() -> None:
    """the host loop 30 ms late before a 20 ms stall credits none of it."""
    host = [(0.0, 0.031), *((0.031 + n * 0.001, 0.032 + n * 0.001) for n in range(100))]
    assert added_lateness([(0.050, 0.071)], host) == pytest.approx(0.020, abs=1e-9)


def test_a_spike_overlapping_part_of_a_stall_credits_only_the_overlap() -> None:
    """the host late for 5 ms of a 20 ms stall: 15 ms is the loop's."""
    host = [(n * 0.001, (n + 1) * 0.001) for n in range(4)]
    host += [(0.004, 0.010), *((0.010 + n * 0.001, 0.011 + n * 0.001) for n in range(30))]
    assert added_lateness([(0.004, 0.025)], host) == pytest.approx(0.015, abs=1e-9)


def test_a_tick_asking_longer_is_late_only_past_what_it_asked() -> None:
    """a 2 ms tick that woke 2.5 ms after it asked was 0.5 ms late, with the host idle throughout."""
    host = [(n * 0.001, (n + 1) * 0.001) for n in range(10)]
    assert added_lateness([(0.001, 0.0035)], host, every=0.002) == pytest.approx(0.0005, abs=1e-9)
    assert latest([(0.001, 0.0035)], every=0.002) == pytest.approx(0.0005, abs=1e-9)


def test_a_tick_with_no_host_baseline_at_its_moment_is_refused() -> None:
    """the baseline is taken at the same moments as the loop's ticks, or the judgement refuses to run."""
    with pytest.raises(AssertionError, match="no host baseline"):
        added_lateness([(1.0, 1.002)], [(0.0, 0.001), (0.001, 0.002)])


@pytest.mark.loop_timing
async def test_the_host_ticks_through_the_loops_ticks_and_a_planted_block_is_the_loops() -> None:
    """the host's ticker spans the loop's ticks with no gap past a tick, and 30 ms held on the loop is counted as
    the loop's (the host, in its own process, is not late for it)."""
    with HostTicker() as host:
        ticks = await ticks_while(lambda: asyncio.sleep(0.1))
        held = await ticks_while(_held_for(0.03))
    assert host.ticks[0][0] <= ticks[0][0] and held[-1][1] <= host.ticks[-1][1]
    assert all(later[0] - earlier[1] < 0.001 for earlier, later in itertools.pairwise(host.ticks))
    assert added_lateness(held, host.ticks) >= 0.02


def _held_for(seconds: float) -> Callable[[], Awaitable[None]]:
    """a run on the loop that holds it for ``seconds`` between two turns."""

    async def work() -> None:
        await asyncio.sleep(0.01)
        hold_the_loop(seconds)
        await asyncio.sleep(0.01)

    return work


#: a budget the trust tests judge by
_BUDGET = 0.020


def _quiet_host(began: float, ended: float) -> list[Tick]:
    """a host idle loop ticking on time over ``began``-``ended``."""
    count = round((ended - began) / 0.001)
    return [(began + n * 0.001, began + (n + 1) * 0.001) for n in range(count)]


def _busy_host(began: float, ended: float, late: float) -> list[Tick]:
    """a host idle loop on time over the span but for one tick ``late`` past its 1 ms, mid-way."""
    middle = (began + ended) / 2
    return [*_quiet_host(began, middle), (middle, middle + 0.001 + late), *_quiet_host(middle + 0.001 + late, ended)]


def _window(measured: tuple[list[Tick], list[Tick]]) -> tuple[list[Tick], list[Tick]]:
    return measured


def test_the_host_lateness_counts_only_inside_the_measured_window() -> None:
    """the host's idle loop late before the measured window says nothing about it; late inside it, it does."""
    host = [(0.0, 0.051), *_quiet_host(0.051, 0.2)]
    assert host_lateness(host, [(0.1, 0.102), (0.102, 0.104)]) == pytest.approx(0.0, abs=1e-9)
    assert host_lateness(_busy_host(0.0, 0.2, 0.03), [(0.05, 0.15)]) == pytest.approx(0.03, abs=1e-9)


async def test_a_quiet_host_is_measured_once_and_its_measurement_judged() -> None:
    """a quiet run: the first measurement is trusted and returned, no retake."""
    taken: list[int] = []

    async def measure() -> tuple[list[Tick], list[Tick]]:
        taken.append(1)
        return [(0.05, 0.052)], _quiet_host(0.0, 0.2)

    assert await trusted(measure, _window, host_may_be_late=_BUDGET) == ([(0.05, 0.052)], _quiet_host(0.0, 0.2))
    assert len(taken) == 1


async def test_a_busy_host_is_retaken_and_a_quiet_retake_is_judged() -> None:
    """a passing burst of host load: the busy measurement is dropped, the next quiet one judged."""
    hosts = iter([_busy_host(0.0, 0.2, 0.05), _quiet_host(0.0, 0.2)])

    async def measure() -> tuple[list[Tick], list[Tick]]:
        return [(0.05, 0.15)], next(hosts)

    _, host = await trusted(measure, _window, host_may_be_late=_BUDGET)
    assert host == _quiet_host(0.0, 0.2)


async def test_a_host_busy_in_every_attempt_could_not_measure_never_a_pass_or_a_red() -> None:
    """a host whose idle loop runs 50 ms late in every window: the test fails as unmeasurable, with its own message,
    after a bounded number of measurements; not "the code was late", and never a pass."""
    taken: list[int] = []

    async def measure() -> tuple[list[Tick], list[Tick]]:
        taken.append(1)
        return [(0.05, 0.15)], _busy_host(0.0, 0.2, 0.05)

    with pytest.raises(CouldNotMeasure, match=r"^could not measure: the host's idle loop ran 50\.0 ms late"):
        await trusted(measure, _window, host_may_be_late=_BUDGET)
    assert len(taken) == inspect.signature(trusted).parameters["attempts"].default


class _ScriptedHost:
    """a :class:`HostTicker` whose idle loop is as late as the script says, once over each block it ticks beside."""

    script: Iterator[float] = iter(())

    def __init__(self) -> None:
        self.ticks: list[Tick] = []

    def __enter__(self) -> _ScriptedHost:
        self._began = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        late = next(_ScriptedHost.script)
        ended = time.perf_counter()
        self.ticks = [(self._began, ended + 0.001 + late), (ended + 0.001 + late, ended + 0.002 + late)]


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch) -> Callable[[list[float]], None]:
    """script the host's idle loop's lateness over each CPU probe's first take (a session's reference)."""

    def script(lates: list[float]) -> None:
        _ScriptedHost.script = iter(lates)

    monkeypatch.setattr(loop_lateness, "HostTicker", _ScriptedHost)
    return script


@pytest.mark.loop_timing
async def test_a_planted_block_on_a_quiet_host_is_trusted_and_red() -> None:
    """30 ms held on the loop: the host's own loop, in its own process, is not late for it, so the measurement is
    trusted and the judgement is red."""

    async def measure() -> tuple[list[Tick], list[Tick]]:
        with HostTicker() as host:
            held = await ticks_while(_held_for(0.03))
        return held, host.ticks

    held, host = await trusted(measure, _window, host_may_be_late=_BUDGET)
    assert added_lateness(held, host) >= _BUDGET


@pytest.fixture
def spin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Callable[[list[float]], tuple[Callable[[int], float], list[int]]]:
    """a scripted CPU probe for :func:`trusted`'s ``probe=``: each call makes a new one (a new session's), answering
    the given seconds take by take, with the host record under ``tmp_path``; returns it and the thread counts it was
    asked for."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    def script(seconds: list[float]) -> tuple[Callable[[int], float], list[int]]:
        taken = iter(seconds)
        asked: list[int] = []

        def probe(threads: int) -> float:
            asked.append(threads)
            return next(taken)

        return probe, asked

    return script


def _performance_cores() -> int:
    """the host's performance cores as its OS reports them: Apple silicon's perflevel0, every usable core elsewhere."""
    usable = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    if sys.platform == "darwin":
        answered = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "hw.perflevel0.logicalcpu"], capture_output=True, text=True, check=False
        ).stdout.strip()
        if answered.isdigit():
            return max(1, min(int(answered), usable))
    return usable


def _on_time(windows: list[tuple[list[Tick], list[Tick]]]) -> Callable[[], Awaitable[tuple[list[Tick], list[Tick]]]]:
    """a measurement handing back ``windows`` in turn."""
    handed = iter(windows)

    async def measure() -> tuple[list[Tick], list[Tick]]:
        return next(handed)

    return measure


#: two loop windows the host's idle loop was on time over
_WINDOWS = [([(0.05, 0.15)], _quiet_host(0.0, 0.2)), ([(0.25, 0.35)], _quiet_host(0.2, 0.4))]


async def test_a_window_on_crowded_cores_is_retaken_though_the_idle_loop_was_on_time(
    scripted: Callable[[list[float]], None],
    spin: Callable[[list[float]], tuple[Callable[[int], float], list[int]]],
) -> None:
    """Flips if a multi-thread measurement whose threads waited for cores is kept because a 1 ms idle loop stayed
    punctual (a real-size serving read went 2.0x on a 10-core host at load ~9), or a retake on free cores is not."""
    scripted([0.0])
    probe, asked = spin([0.010, 0.010, 0.024, 0.010, 0.011])

    ticks, _ = await trusted(_on_time(_WINDOWS), _window, host_may_be_late=_BUDGET, threads=2, probe=probe)
    assert ticks == [(0.25, 0.35)]
    assert asked == [min(2, _performance_cores())] * 5, "the session's first take, then before and after each attempt"


async def test_crowded_cores_in_every_attempt_could_not_measure(
    scripted: Callable[[list[float]], None],
    spin: Callable[[list[float]], tuple[Callable[[int], float], list[int]]],
) -> None:
    """Flips if a multi-thread measurement taken on crowded cores every time passes, is judged, or retakes without
    bound."""
    scripted([0.0])
    probe, _ = spin([0.010, *([0.020] * 6)])
    taken: list[int] = []

    async def measure() -> tuple[list[Tick], list[Tick]]:
        taken.append(1)
        return _WINDOWS[0]

    with pytest.raises(CouldNotMeasure, match=r"^could not measure: the path's threads waited for cores"):
        await trusted(measure, _window, host_may_be_late=_BUDGET, threads=4, attempts=3, probe=probe)
    assert len(taken) == 3


async def test_a_measurement_naming_no_threads_spins_nothing_and_one_thread_spins_one(
    scripted: Callable[[list[float]], None],
    spin: Callable[[list[float]], tuple[Callable[[int], float], list[int]]],
) -> None:
    """Flips if a measurement that names no threads pays for the spin, or a one-thread CPU measurement (a thread's
    own CPU time, which a busy sibling core or a throttled one inflates) is not checked on one thread."""
    scripted([0.0])
    probe, asked = spin([0.010] * 3)

    await trusted(_on_time(_WINDOWS), _window, host_may_be_late=_BUDGET, probe=probe)
    assert asked == []
    await trusted(_on_time(_WINDOWS), _window, host_may_be_late=_BUDGET, threads=1, probe=probe)
    assert asked == [1, 1, 1]


async def test_the_reference_is_the_host_s_fastest_spin_and_a_faster_take_lowers_it(
    scripted: Callable[[list[float]], None],
    spin: Callable[[list[float]], tuple[Callable[[int], float], list[int]]],
) -> None:
    """Flips if crowding is judged against anything but the lower of the session's first take and the host's record,
    or a faster take does not lower the record: first take 10 ms; 12 ms reads 1.20x; 8 ms is the fastest yet, so the
    next 12 ms reads 1.50x."""
    scripted([0.0])
    probe, _ = spin([0.010, 0.012, 0.008, 0.012, 0.008])

    with pytest.raises(CouldNotMeasure, match=r"took 1\.20x .*took 1\.50x"):
        await trusted(
            _on_time(_WINDOWS),
            _window,
            host_may_be_late=_BUDGET,
            threads=4,
            attempts=2,
            cpu_may_be_crowded=0.0,
            probe=probe,
        )


async def test_a_session_busy_from_its_start_judges_against_the_host_s_record(
    scripted: Callable[[list[float]], None],
    spin: Callable[[list[float]], tuple[Callable[[int], float], list[int]]],
) -> None:
    """Flips if a session busy from start to finish calibrates itself busy: its first take (slow) loses to the
    host's record, so its slow windows read crowded."""
    scripted([0.0])
    quiet, _ = spin([0.010] * 3)
    await trusted(_on_time(_WINDOWS), _window, host_may_be_late=_BUDGET, threads=4, probe=quiet)
    scripted([0.0])
    busy, _ = spin([0.030] * 3)

    with pytest.raises(CouldNotMeasure, match=r"took 3\.00x"):
        await trusted(_on_time(_WINDOWS), _window, host_may_be_late=_BUDGET, threads=4, attempts=1, probe=busy)


async def test_a_host_with_no_record_and_a_busy_start_trusts_nothing(
    tmp_path: Path,
    scripted: Callable[[list[float]], None],
    spin: Callable[[list[float]], tuple[Callable[[int], float], list[int]]],
) -> None:
    """Flips if a host with no record calibrates on a first take its idle loop was late over, or such a session
    trusts any multi-thread window."""
    scripted([0.05])
    probe, _ = spin([0.030])
    taken: list[int] = []

    async def measure() -> tuple[list[Tick], list[Tick]]:
        taken.append(1)
        return _WINDOWS[0]

    with pytest.raises(CouldNotMeasure, match=r"^could not measure: no quiet reference"):
        await trusted(measure, _window, host_may_be_late=_BUDGET, threads=4, attempts=1, probe=probe)
    assert taken == [1]
    assert not list(tmp_path.rglob("*.json")), "a busy session's take must never become the host's record"


async def test_a_spin_runs_on_the_performance_cores_at_most(
    scripted: Callable[[list[float]], None],
    spin: Callable[[list[float]], tuple[Callable[[int], float], list[int]]],
) -> None:
    """Flips if a path's threads past the host's performance cores spin on its efficiency cores too (an idle Apple
    host of 4 + 6 cores read 2x across all ten)."""
    cores = _performance_cores()
    for path in (1, 2, cores, cores + 1, 64):
        scripted([0.0])
        probe, asked = spin([0.010] * 3)
        await trusted(_on_time(_WINDOWS), _window, host_may_be_late=_BUDGET, threads=path, probe=probe)
        assert asked == [min(path, cores)] * 3, f"a path on {path} threads"


def _hogged(threads: int) -> Callable[[], None]:
    """start ``threads`` threads hashing off the interpreter lock; returns what stops them."""
    stop = threading.Event()

    def hog() -> None:
        while not stop.is_set():
            hashlib.sha256(bytes(1 << 20)).digest()

    hogs = [threading.Thread(target=hog) for _ in range(threads)]
    for thread in hogs:
        thread.start()

    def stopped() -> None:
        stop.set()
        for thread in hogs:
            thread.join()

    return stopped


async def _nothing() -> tuple[list[Tick], list[Tick]]:
    """a measurement of nothing: no loop ticks, so the host's idle loop is judged over no window."""
    return [], []


@pytest.mark.loop_timing
async def test_a_spin_planted_on_the_path_s_threads_makes_the_window_untrusted_and_a_quiet_one_is_trusted() -> None:
    """Red first: a spin on the path's threads beside the measurement makes it untrusted ("could not measure"); the
    same measurement on a quiet host is trusted -- on a host of fast and slow cores too, asked for more threads than
    it has fast cores (the efficiency cores alone must not trip it). The planted spin takes every core the path's
    threads can run on, not only the fast ones its probe spins on: hogs on the fast cores alone leave a host of fast
    and slow cores' slow ones free, and the probe's threads only move there (about 1.5x on an Apple 4 + 6, the limit
    itself), which says nothing about a planted crowd."""
    path = 64

    assert await trusted(_nothing, _window, host_may_be_late=0.020, threads=path) == ([], [])
    stop = _hogged(min(path, getattr(os, "process_cpu_count", os.cpu_count)() or 1))
    try:
        with pytest.raises(CouldNotMeasure, match="waited for cores"):
            await trusted(_nothing, _window, host_may_be_late=1.0, threads=path)
    finally:
        stop()
