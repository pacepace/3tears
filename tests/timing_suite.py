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

An ``integration`` test is left alone: the integration suite has no timing suite of its own yet, so its speed tests
still run beside the rest of it (``.testament.toml``).
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import pytest

from threetears.core.testing.loop_lateness import HostTicker

#: the real ``HostTicker.__enter__``, put back after every test
_REAL_ENTER = HostTicker.__enter__

#: what a test outside the timing suite is told
REFUSED = (
    "this test times against the host's quiet (a HostTicker beside it), so it runs apart from other work: "
    "mark it @pytest.mark.loop_timing (tests/timing_suite.py)"
)


def _refused(self: HostTicker) -> HostTicker:
    """``HostTicker.__enter__`` in a test outside the timing suite: fail it, saying why."""
    pytest.fail(REFUSED, pytrace=False)


def _times_apart(item: pytest.Item) -> bool:
    """whether ``item`` may start the host's idle loop: marked ``loop_timing``, or an ``integration`` test."""
    return item.get_closest_marker("loop_timing") is not None or item.get_closest_marker("integration") is not None


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None) -> Generator[None, Any, Any]:
    """the real ``HostTicker`` for a test that times apart, the refusing one for any other, over its setup, call and
    teardown; whatever was in place before, after.

    :param item: the test
    :ptype item: pytest.Item
    :param nextitem: the test after it
    :ptype nextitem: pytest.Item | None
    :return: the protocol's result
    :rtype: Generator[None, Any, Any]
    """
    before = HostTicker.__dict__["__enter__"]
    HostTicker.__enter__ = _REAL_ENTER if _times_apart(item) else _refused  # type: ignore[method-assign]
    try:
        return (yield)
    finally:
        HostTicker.__enter__ = before  # type: ignore[method-assign]
