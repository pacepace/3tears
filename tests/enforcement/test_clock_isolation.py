"""a freeze imports nothing under the fake clock, and a test that leaves the clock frozen fails, not its neighbours."""

from __future__ import annotations

import pytest

# a lazy package whose name, read, imports a module declaring a pydantic ``date`` field: what ``threetears.evals``
# serving ``Model`` is. It is loaded at import, as a test module of another package loads it at collection.
_LAZY = """
import datetime
import importlib
import sys
import types

from pydantic import BaseModel


class _Built(BaseModel):
    count: int


lazy = types.ModuleType("threetears.clock_isolation_probe")


def _serve(name):
    if name == "Stamped":
        return importlib.import_module("stamped_model").Stamped
    raise AttributeError(name)


lazy.__getattr__ = _serve
lazy.__dir__ = lambda: ["Stamped"]
sys.modules[lazy.__name__] = lazy
"""

_TESTS = """
import datetime

from freezegun import freeze_time

import lazy_probe  # noqa: F401

_LEFT_RUNNING = freeze_time(datetime.datetime(2026, 1, 1))


@freeze_time(datetime.datetime(2026, 1, 1))
def test_a_freeze_after_a_lazy_package_loaded_starts():
    assert datetime.date.today() == datetime.date(2026, 1, 1)


def test_a_freeze_left_running_fails_the_test_that_started_it():
    _LEFT_RUNNING.start()


def test_the_next_test_is_not_blamed():
    _LEFT_RUNNING.stop()
"""


_STAMPED = """
import datetime

from pydantic import BaseModel


class Stamped(BaseModel):
    day: datetime.date
"""


def _session(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(stamped_model=_STAMPED, lazy_probe=_LAZY, test_clock=_TESTS)
    pytester.syspathinsert()


def test_without_the_plugin_the_freeze_raises_and_stays_frozen(pytester: pytest.Pytester) -> None:
    """The control, in its own process since the clock it leaves frozen never thaws: the probe is the incident."""
    _session(pytester)
    result = pytester.runpytest_subprocess("-p", "no:cacheprovider", "-p", "no:asyncio", "-k", "lazy_package")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*PydanticSchemaGenerationError*FakeDate*"], consecutive=False)


def test_a_freeze_imports_nothing_under_the_fake_and_a_leak_fails_only_its_test(pytester: pytest.Pytester) -> None:
    pytester.makeconftest('pytest_plugins = ["tests.clock_isolation"]')
    _session(pytester)
    result = pytester.runpytest_inprocess("-p", "no:cacheprovider", "-p", "no:asyncio")
    # a teardown error is reported on top of the passed call, so the leaking test also counts as passed
    result.assert_outcomes(passed=3, errors=1)
    result.stdout.fnmatch_lines(
        [
            "*ERROR at teardown of test_a_freeze_left_running_fails_the_test_that_started_it*",
            "*the clock is still frozen after this test*",
        ],
        consecutive=False,
    )
