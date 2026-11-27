"""pytest plugin: a test that times against the host's quiet carries ``loop_timing``, so it runs only apart.

A test whose verdict rests on the host being quiet takes its measurement beside the host's own idle loop
(:class:`threetears.core.testing.loop_lateness.HostTicker`, which ``quietly``, ``trusted`` and
``best_seconds_interleaved`` all start) and fails as ``CouldNotMeasure`` when that loop ran late. In the unit suite it
runs beside other batches (the pool's other projects and suites, the GitHub shards' ``-n 4``), where the host is
seldom quiet: DuckDB's "within twice" ratios failed that way on brooks-nas, brooks-14th-mbp, brooks-5950x and
markpace-mbp (2026-11-13 to 11-15), a red the code did not earn. The ``loop_timing`` marker runs such a test apart:
out of the unit suite, in the pool's ``loop_timing`` suite and CI's serial pass (``scripts/ci-unit-tests.sh
--loop-timing``).

This makes the marker structural. A test not marked ``loop_timing`` that starts a real ``HostTicker`` fails at once,
on every host, naming the marker: a red every run, never a flake. A test that scripts the host (a fake in place of
``HostTicker``) starts none, and is unaffected.

The same holds for a bound on a real-size test's own time that takes no ``HostTicker``: one turn of the loop timed by
``turn_holds()`` (a 10 ms turn while a pass applies 200,000 rows), a best-of-N by ``best_seconds()``, or the CPU of
one by ``best_cpu_seconds()`` on a real clock. A CPU clock skips the time a process waits for a core, yet the pool's
oversubscribed runners still stretched the CPU the same work spent 1.4-3x (brooks-5950x, brooks-nas, 2026-11-26), so
such a bound passed or failed by runner. Each passes ``TimesApart.check`` as it starts
(:class:`threetears.core.testing.loop_lateness.TimesApart`), which this plugin shuts for a test not marked
``loop_timing``. ``best_cpu_seconds`` on a scripted clock measures no host and is unaffected.

An ``integration`` test is held to it too: the integration suite's speed tests run apart from the rest of it, in the
pool's ``integration_timing`` suite (``.testament.toml``; ``scripts/test-integration.sh`` runs it after the suite).
Beside the suite's other batches they failed on numbers they hold alone (2026-11-16: the visible-customer set 3.0x
its statement on brooks-nas, 1.41-1.99x alone), the same red the code did not earn.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import pytest

from threetears.core.testing.loop_lateness import HostTicker, TimesApart

#: the real ``HostTicker.__enter__``, put back after every test
_REAL_ENTER = HostTicker.__enter__

#: what a test outside the timing suite is told
REFUSED = (
    "this test times against the host's quiet (a HostTicker beside it), so it runs apart from other work: "
    "mark it @pytest.mark.loop_timing (tests/timing_suite.py)"
)


#: what a test outside the timing suite that times its own work on this host is told, given the timer
REFUSED_TIMER = (
    "this test bounds its own time on this host ({what}), which a busy host stretches, so it runs apart from other "
    "work: mark it @pytest.mark.loop_timing (tests/timing_suite.py)"
)


def _refused(self: HostTicker) -> HostTicker:
    """``HostTicker.__enter__`` in a test outside the timing suite: fail it, saying why."""
    pytest.fail(REFUSED, pytrace=False)


def _refused_timer(what: str) -> None:
    """``TimesApart``'s refusal in a test outside the timing suite: fail it, naming the timer and the marker."""
    pytest.fail(REFUSED_TIMER.format(what=what), pytrace=False)


def _times_apart(item: pytest.Item) -> bool:
    """whether ``item`` may start the host's idle loop: marked ``loop_timing``, a unit and an integration test alike."""
    return item.get_closest_marker("loop_timing") is not None


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None) -> Generator[None, Any, Any]:
    """the real ``HostTicker`` and an open ``TimesApart`` for a test that times apart, the refusing ones for any other,
    over its setup, call and teardown; whatever was in place before, after.

    :param item: the test
    :ptype item: pytest.Item
    :param nextitem: the test after it
    :ptype nextitem: pytest.Item | None
    :return: the protocol's result
    :rtype: Generator[None, Any, Any]
    """
    before, refusal_before = HostTicker.__dict__["__enter__"], TimesApart.refusal
    apart = _times_apart(item)
    HostTicker.__enter__ = _REAL_ENTER if apart else _refused  # type: ignore[method-assign]
    TimesApart.refusal = None if apart else _refused_timer
    try:
        return (yield)
    finally:
        HostTicker.__enter__ = before  # type: ignore[method-assign]
        TimesApart.refusal = refusal_before
