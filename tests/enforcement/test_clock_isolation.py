"""a freeze imports nothing under the fake clock, and a leaked freeze blames the test that leaked it, not the ones after."""

from __future__ import annotations

from pathlib import Path

import pytest

#: The repository's root, where ``tests.clock_isolation`` imports from.
_REPO_ROOT = Path(__file__).resolve().parents[2]

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


# the freeze is never stopped, as when start() raised: nothing in the process thaws the clock


def test_the_next_test_is_not_blamed():
    pass


def test_nor_any_test_after_it():
    pass
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


def test_a_freeze_imports_nothing_under_the_fake_and_a_leak_blames_only_its_test(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In its own process too, since the leak is left unrepaired; the plugin is found from the repo root."""
    monkeypatch.setenv("PYTHONPATH", str(_REPO_ROOT))
    pytester.makeconftest('pytest_plugins = ["tests.clock_isolation"]')
    _session(pytester)
    result = pytester.runpytest_subprocess("-p", "no:cacheprovider", "-p", "no:asyncio")
    # the leaking test's call passed and its teardown errs; the two after it err at setup and never run
    result.assert_outcomes(passed=2, errors=3)
    headers = sorted(line.strip("_ ") for line in result.stdout.lines if line.lstrip("_ ").startswith("ERROR at "))
    assert headers == [
        "ERROR at setup of test_nor_any_test_after_it",
        "ERROR at setup of test_the_next_test_is_not_blamed",
        "ERROR at teardown of test_a_freeze_left_running_fails_the_test_that_started_it",
    ]
    result.stdout.fnmatch_lines(
        [
            "*the clock is still frozen after this test*",
            "*the clock was frozen before this test began*an earlier test in this process leaked it, not this one*",
            "*Leaked by: test_clock.py::test_a_freeze_left_running_fails_the_test_that_started_it*",
        ],
        consecutive=False,
    )
