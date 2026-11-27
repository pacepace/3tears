"""a test that starts the host's idle loop (a real HostTicker) without ``loop_timing`` fails at once, saying to mark it,
an integration test as much as a unit one; a marked test and a test that scripts the host are left alone
(tests/timing_suite.py). So does one that bounds its own time on the host's clock: ``turn_holds()``, ``best_seconds()``,
``best_cpu_seconds()`` on a real clock (``TimesApart``)."""

from __future__ import annotations

import pytest

from threetears.core.testing import loop_lateness
from threetears.core.testing.loop_lateness import HostTicker, TimesApart

_TESTS = """
import pytest

from threetears.core.testing import loop_lateness
from threetears.core.testing.loop_lateness import HostTicker


def test_an_unmarked_timing_test():
    with HostTicker():
        pass


@pytest.mark.loop_timing
def test_a_marked_timing_test():
    with HostTicker():
        pass


@pytest.mark.loop_timing
class TestAMarkedClass:
    def test_inside_it(self):
        with HostTicker():
            pass


@pytest.mark.integration
def test_an_unmarked_integration_speed_test():
    with HostTicker():
        pass


@pytest.mark.integration
@pytest.mark.loop_timing
def test_a_marked_integration_speed_test():
    with HostTicker():
        pass


class _Scripted:
    ticks = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


def test_a_scripted_host(monkeypatch):
    monkeypatch.setattr(loop_lateness, "HostTicker", _Scripted)
    with loop_lateness.HostTicker():
        pass


@pytest.fixture
def ticking():
    with HostTicker():
        yield


def test_an_unmarked_fixture_that_times(ticking):
    pass
"""


def test_only_a_test_that_times_apart_may_start_the_host_s_idle_loop(pytester: pytest.Pytester) -> None:
    """Flips if an unmarked test may start a real HostTicker (in its body or a fixture), an integration test as much as
    a unit one, or a marked or scripted one may not."""
    pytester.makeconftest('pytest_plugins = ["tests.timing_suite"]')
    pytester.makeini("[pytest]\nmarkers =\n    integration: needs a service\n    loop_timing: times apart\n")
    pytester.makepyfile(test_timing=_TESTS)
    result = pytester.runpytest_inprocess("-p", "no:cacheprovider", "-p", "no:asyncio", "-rA")
    result.assert_outcomes(passed=4, failed=2, errors=1)
    result.stdout.fnmatch_lines(
        [
            "*ERROR at setup of test_an_unmarked_fixture_that_times*",
            "*runs apart from other work: mark it @pytest.mark.loop_timing*",
            "*test_an_unmarked_timing_test*",
            "*runs apart from other work: mark it @pytest.mark.loop_timing*",
            "*test_an_unmarked_integration_speed_test*",
            "*runs apart from other work: mark it @pytest.mark.loop_timing*",
        ]
    )


def test_the_real_host_ticker_is_back_after_the_session(pytester: pytest.Pytester) -> None:
    """Flips if a test's refusal outlives it: the class keeps what it had before the inner session."""
    before = HostTicker.__dict__["__enter__"]
    pytester.makeconftest('pytest_plugins = ["tests.timing_suite"]')
    pytester.makeini("[pytest]\nmarkers =\n    loop_timing: times apart\n")
    pytester.makepyfile(test_timing=_TESTS)
    pytester.runpytest_inprocess("-p", "no:cacheprovider", "-p", "no:asyncio")
    assert HostTicker.__dict__["__enter__"] is before
    assert loop_lateness.HostTicker is HostTicker


_TIMER_TESTS = """
import asyncio

import pytest

from threetears.core.testing.calibrate import best_cpu_seconds, best_seconds
from threetears.core.testing.loop_lateness import turn_holds


async def _nothing():
    return None


def test_an_unmarked_turn_bound():
    with turn_holds():
        pass


def test_an_unmarked_best_of_n():
    asyncio.run(best_seconds(_nothing))


def test_an_unmarked_cpu_bound():
    asyncio.run(best_cpu_seconds(_nothing))


@pytest.mark.integration
def test_an_unmarked_integration_cpu_bound():
    asyncio.run(best_cpu_seconds(_nothing))


@pytest.fixture
def turns():
    with turn_holds() as holds:
        yield holds


def test_an_unmarked_fixture_that_times_turns(turns):
    pass


def test_a_scripted_cpu_clock():
    ticks = iter(range(100))
    assert asyncio.run(best_cpu_seconds(_nothing, runs=2, clock=lambda: float(next(ticks)))) == 1.0


@pytest.mark.loop_timing
def test_a_marked_turn_bound():
    with turn_holds():
        pass


@pytest.mark.loop_timing
class TestMarked:
    def test_a_best_of_n(self):
        asyncio.run(best_seconds(_nothing))

    def test_a_cpu_bound(self):
        asyncio.run(best_cpu_seconds(_nothing))
"""


def test_only_a_test_that_times_apart_may_bound_its_own_time_on_the_host(pytester: pytest.Pytester) -> None:
    """Flips if an unmarked test may time a loop turn (turn_holds, in its body or a fixture), a best-of-N
    (best_seconds) or a CPU bound on a real clock (best_cpu_seconds), an integration test as much as a unit one, or if
    a marked test, or best_cpu_seconds on a scripted clock, may not."""
    pytester.makeconftest('pytest_plugins = ["tests.timing_suite"]')
    pytester.makeini("[pytest]\nmarkers =\n    integration: needs a service\n    loop_timing: times apart\n")
    pytester.makepyfile(test_timers=_TIMER_TESTS)
    recorded = pytester.inline_run("-p", "no:cacheprovider", "-p", "no:asyncio")
    refused = {
        report.nodeid.rsplit("::", 1)[-1]: str(report.longrepr)
        for report in recorded.getreports("pytest_runtest_logreport")
        if report.failed
    }
    expected = {
        "test_an_unmarked_turn_bound": "turn_holds()",
        "test_an_unmarked_fixture_that_times_turns": "turn_holds()",
        "test_an_unmarked_best_of_n": "best_seconds()",
        "test_an_unmarked_cpu_bound": "best_cpu_seconds()",
        "test_an_unmarked_integration_cpu_bound": "best_cpu_seconds()",
    }
    assert set(refused) == set(expected), refused
    for test, timer in expected.items():
        assert f"on this host ({timer})" in refused[test], (test, refused[test])
        assert "mark it @pytest.mark.loop_timing" in refused[test], (test, refused[test])
    passed, _, _ = recorded.listoutcomes()
    assert sorted(report.nodeid.rsplit("::", 1)[-1] for report in passed) == [
        "test_a_best_of_n",
        "test_a_cpu_bound",
        "test_a_marked_turn_bound",
        "test_a_scripted_cpu_clock",
    ]


def test_the_timers_gate_is_back_after_the_session(pytester: pytest.Pytester) -> None:
    """Flips if a test's refusal of the timers outlives it: the gate keeps what it had before the inner session."""
    before = TimesApart.refusal
    pytester.makeconftest('pytest_plugins = ["tests.timing_suite"]')
    pytester.makeini("[pytest]\nmarkers =\n    integration: needs a service\n    loop_timing: times apart\n")
    pytester.makepyfile(test_timers=_TIMER_TESTS)
    pytester.runpytest_inprocess("-p", "no:cacheprovider", "-p", "no:asyncio")
    assert TimesApart.refusal is before
