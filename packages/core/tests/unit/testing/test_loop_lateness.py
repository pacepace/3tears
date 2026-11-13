"""``threetears.core.testing.loop_lateness``: a test loop's lateness counts only where the host's own idle loop, in a
process of its own, was not late at the same moment.

Pure judgements over recorded ticks (no timing), and one short timed run of the host's ticker itself.
"""

from __future__ import annotations

import asyncio
import itertools
from collections.abc import Awaitable, Callable

import pytest

from threetears.core.testing.loop_lateness import (
    MEASURE_ATTEMPTS,
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
    assert len(taken) == MEASURE_ATTEMPTS


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
