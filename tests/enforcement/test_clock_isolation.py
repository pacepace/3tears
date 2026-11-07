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
import importlib
import sys
import time
import types

import pytest
from freezegun import freeze_time

import lazy_probe  # noqa: F401

_INSTANT = datetime.datetime(2026, 1, 1)


@pytest.fixture(scope="class")
def _class_freeze():
    with freeze_time(_INSTANT):
        yield


@pytest.mark.usefixtures("_class_freeze")
class TestInsideAClassScopedFreeze:
    def test_one(self):
        assert datetime.date.today() == datetime.date(2026, 1, 1)

    def test_two(self):
        assert datetime.date.today() == datetime.date(2026, 1, 1)


@freeze_time(_INSTANT)
class TestUnderAClassDecorator:
    def test_frozen(self):
        assert datetime.date.today() == datetime.date(2026, 1, 1)


@freeze_time(_INSTANT)
def test_a_freeze_after_a_lazy_package_loaded_starts():
    assert datetime.date.today() == datetime.date(2026, 1, 1)


def test_a_monkeypatched_clock_is_no_freeze(monkeypatch):
    monkeypatch.setattr(time, "monotonic", lambda: 1_000.0)


def _start_and_never_stop():
    freeze_time(_INSTANT).start()


def _start_that_raises():
    # a lazy package loaded inside the test, after setup resolved the names: the freeze's walk imports it
    late = types.ModuleType("threetears.clock_isolation_late")

    def _serve(name):
        if name == "Late":
            return importlib.import_module("late_model").Late
        raise AttributeError(name)

    late.__getattr__ = _serve
    late.__dir__ = lambda: ["Late"]
    sys.modules[late.__name__] = late
    with freeze_time(_INSTANT):
        pass


def test_a_freeze_left_running_fails_the_test_that_started_it():
    LEAK_SHAPE()


# the freeze is never stopped: nothing in the process thaws the clock


def test_the_next_test_is_not_blamed():
    pass


def test_nor_any_test_after_it():
    pass
"""

#: The inner session's conftest: the plugin, beside an autouse fixture taking ``monkeypatch`` as the root conftest's does.
_CONFTEST = """
import pytest

pytest_plugins = ["tests.clock_isolation"]


@pytest.fixture(autouse=True)
def _like_the_root_conftest(monkeypatch):
    yield
"""


_STAMPED = """
import datetime

from pydantic import BaseModel


class Stamped(BaseModel):
    day: datetime.date
"""


def _session(pytester: pytest.Pytester, leak: str = "_start_and_never_stop") -> None:
    pytester.makepyfile(
        stamped_model=_STAMPED,
        late_model=_STAMPED.replace("Stamped", "Late"),
        lazy_probe=_LAZY,
        test_clock=_TESTS.replace("LEAK_SHAPE", leak),
    )
    pytester.syspathinsert()


def test_without_the_plugin_the_freeze_raises_and_stays_frozen(pytester: pytest.Pytester) -> None:
    """The control, in its own process since the clock it leaves frozen never thaws: the probe is the incident."""
    _session(pytester)
    result = pytester.runpytest_subprocess("-p", "no:cacheprovider", "-p", "no:asyncio", "-k", "lazy_package")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*PydanticSchemaGenerationError*FakeDate*"], consecutive=False)


@pytest.mark.parametrize(
    ("leak", "failed"),
    [("_start_and_never_stop", 0), ("_start_that_raises", 1)],
    ids=["a start never stopped", "a start that raised"],
)
def test_a_freeze_imports_nothing_under_the_fake_and_a_leak_blames_only_its_test(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, leak: str, failed: int
) -> None:
    """In its own process too, since the leak is left unrepaired; the plugin is found from the repo root.

    The freezes enclosing a class on purpose, and a clock a test monkeypatches, pass; of the two leak shapes, each
    blames the test that leaked it at teardown, and the tests after it at setup, by the leaker's name.
    """
    monkeypatch.setenv("PYTHONPATH", str(_REPO_ROOT))
    pytester.makeconftest(_CONFTEST)
    _session(pytester, leak)
    result = pytester.runpytest_subprocess("-p", "no:cacheprovider", "-p", "no:asyncio")
    headers = sorted(line.strip("_ ") for line in result.stdout.lines if line.lstrip("_ ").startswith("ERROR at "))
    assert headers == [
        "ERROR at setup of test_nor_any_test_after_it",
        "ERROR at setup of test_the_next_test_is_not_blamed",
        "ERROR at teardown of test_a_freeze_left_running_fails_the_test_that_started_it",
    ]
    # the leaking test's call passes (or raises, for a start that raised) and its teardown errs; the two after it
    # err at setup and never run
    result.assert_outcomes(passed=6 - failed, failed=failed, errors=3)
    result.stdout.fnmatch_lines(
        [
            "*the clock is still frozen after this test, by 1 freeze(s) it started*",
            "*an earlier test in this process leaked, not this one*",
            "*Leaked by: test_clock.py::test_a_freeze_left_running_fails_the_test_that_started_it*",
        ],
        consecutive=False,
    )
