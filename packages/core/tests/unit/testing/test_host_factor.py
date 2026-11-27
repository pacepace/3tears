"""the host's factor (``threetears.core.testing.host_factor``) on a scripted probe and clock: the limit a loop turn is
held to scales with it in a shared suite, up to its cap; an exclusive suite takes none; one noisy probe moves nothing."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest

from threetears.core.testing.host_factor import HostFactor, limit_for

#: the guard's limit on the reference host (``loop_stall._STALL_LIMIT_SECONDS``)
LIMIT = 0.25

#: the scripted host's reference probe, cap and probe interval
REFERENCE_PROBE_SECONDS = 0.004
FACTOR_CAP = 4.0
PROBE_EVERY_SECONDS = 30.0


class _Clock:
    """a scripted monotonic clock."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _takes(*factors: float) -> Callable[[], float]:
    """a probe that takes ``factor`` x the reference, one factor a take, in turn."""
    each: Iterator[float] = iter(factors)
    return lambda: REFERENCE_PROBE_SECONDS * next(each)


def _host(factor: float, clock: _Clock | None = None) -> HostFactor:
    """a host whose every take reads ``factor`` x the reference."""
    return _scripted(lambda: REFERENCE_PROBE_SECONDS * factor, clock or _Clock())


def _scripted(take: Callable[[], float], clock: _Clock) -> HostFactor:
    """a host on a scripted probe and clock, against the scripted reference and cap."""
    return HostFactor(
        take=take, clock=clock, reference=REFERENCE_PROBE_SECONDS, cap=FACTOR_CAP, every=PROBE_EVERY_SECONDS
    )


@pytest.mark.parametrize(("factor", "limit"), [(1.0, LIMIT), (2.0, LIMIT * 2), (0.5, LIMIT)])
def test_the_limit_is_the_reference_limit_times_the_host_factor(factor: float, limit: float) -> None:
    """Flips if a host as fast as the reference is not held to 0.25 s, a host twice as slow not to 0.5 s, or a host
    faster than the reference to a limit tighter than the reference's (the factor floors at 1)."""
    reading = _host(factor).reading()
    assert reading.factor == pytest.approx(max(1.0, factor))
    held_to = limit_for(LIMIT, reading)
    assert held_to.seconds == pytest.approx(limit)
    assert held_to.judgeable


def test_a_host_past_the_cap_is_held_to_the_capped_limit_and_not_judged_past_it() -> None:
    """Flips if a host past the cap scales the limit past cap x 0.25 s, or reads as judgeable."""
    held_to = limit_for(LIMIT, _host(FACTOR_CAP + 1).reading())
    assert held_to.seconds == pytest.approx(LIMIT * FACTOR_CAP)
    assert not held_to.judgeable
    assert f"x {FACTOR_CAP:.2f}, capped at {FACTOR_CAP:.1f}x" in held_to.told()


def test_an_exclusive_suite_takes_no_factor() -> None:
    """Flips if a test in an exclusive suite (no reading) is held to anything but exactly 0.25 s, or is ever unjudged."""
    held_to = limit_for(LIMIT, None)
    assert held_to.seconds == LIMIT
    assert held_to.judgeable
    assert "no host factor" in held_to.told()


def test_a_raised_limit_is_reference_seconds_scaled_by_the_factor() -> None:
    """Flips if a test's ``loop_stall_limit(seconds=1.0)`` is not scaled to 2.0 s on a host twice as slow."""
    assert limit_for(1.0, _host(2.0).reading()).seconds == pytest.approx(2.0)


def test_the_failure_names_the_limit_factor_probes_and_reference() -> None:
    """Flips if the limit's text does not say the scaled seconds, the sum that made them and the probe's numbers."""
    told = limit_for(LIMIT, _host(1.8).reading()).told()
    assert told.startswith(f"{LIMIT * 1.8:.3f} s on this host: {LIMIT:.3f} s x 1.80 (host factor 1.80")
    assert f"{REFERENCE_PROBE_SECONDS * 1.8 * 1000:.2f} ms of loop-thread CPU" in told
    assert f"{REFERENCE_PROBE_SECONDS * 1000:.2f} ms reference" in told


def test_one_noisy_probe_does_not_move_the_factor() -> None:
    """Flips if a probe is not the fastest of its takes, or the factor is not the median of the last probes: one
    stretched take, and then one stretched probe, must leave a host at factor 1."""
    clock = _Clock()
    # three probes at the first sample, three takes each: one take of the second is stretched 9x
    host = _scripted(_takes(1, 1, 1, 9, 1, 1, 1, 1, 1, 6, 6, 6), clock)
    host.sample_if_due()
    assert host.reading().factor == pytest.approx(1.0)
    clock.now = PROBE_EVERY_SECONDS
    host.sample_if_due()  # a whole probe at 6x: one of three
    assert host.reading().factor == pytest.approx(1.0)


def test_the_factor_is_reprobed_every_interval_and_not_between() -> None:
    """Flips if the host is probed again before the interval (CPU between every test) or not after it (a host whose
    load changed keeps its old factor)."""
    clock = _Clock()
    probes: list[float] = []

    def take() -> float:
        probes.append(clock.now)
        return REFERENCE_PROBE_SECONDS * (1.0 if clock.now < PROBE_EVERY_SECONDS else 3.0)

    host = _scripted(take, clock)
    host.sample_if_due()
    first = len(probes)
    clock.now = PROBE_EVERY_SECONDS - 1
    host.sample_if_due()
    assert len(probes) == first
    for step in (1, 2):
        clock.now = PROBE_EVERY_SECONDS * step
        host.sample_if_due()
    assert len(probes) > first
    assert host.reading().factor == pytest.approx(3.0)
