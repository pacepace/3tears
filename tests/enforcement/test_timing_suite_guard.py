"""a test that starts the host's idle loop (a real HostTicker) without ``loop_timing`` fails at once, saying to mark it;
a marked test, an integration test and a test that scripts the host are left alone (tests/timing_suite.py)."""

from __future__ import annotations

import pytest

from threetears.core.testing import loop_lateness
from threetears.core.testing.loop_lateness import HostTicker

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
def test_an_integration_speed_test():
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
    """Flips if an unmarked test may start a real HostTicker (in its body or a fixture), or a marked, integration or
    scripted one may not."""
    pytester.makeconftest('pytest_plugins = ["tests.timing_suite"]')
    pytester.makeini("[pytest]\nmarkers =\n    integration: needs a service\n    loop_timing: times apart\n")
    pytester.makepyfile(test_timing=_TESTS)
    result = pytester.runpytest_inprocess("-p", "no:cacheprovider", "-p", "no:asyncio", "-rA")
    result.assert_outcomes(passed=4, failed=1, errors=1)
    result.stdout.fnmatch_lines(
        [
            "*ERROR at setup of test_an_unmarked_fixture_that_times*",
            "*runs apart from other work: mark it @pytest.mark.loop_timing*",
            "*test_an_unmarked_timing_test*",
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
