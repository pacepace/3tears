"""pytest plugin: no test starts a freeze that imports under the fake clock, and none leaves the clock frozen.

freezegun's ``start()`` swaps ``datetime.date`` and ``datetime.datetime`` for its fakes and only then reads every
attribute of every loaded module, to repoint the ones bound to the real clock. Reading an attribute a package serves
lazily (a module ``__getattr__``, as ``threetears.evals`` serves ``Model``) imports the module that defines it, under
the fakes. A pydantic model there declaring a ``date`` field cannot build on ``FakeDate``, so ``start()`` raises
partway, after it froze the clock and before ``with`` had entered: nothing ever calls ``stop()``. The clock stays
frozen for every later test in the process, and an import of that model in any of them fails the same way. That needs
only one process holding a freeze and a lazy package nobody had read yet, which a runner interleaving packages
(the testament pool) gives and a per-package shard does not.

So before each test's fixtures are set up, every lazy ``threetears`` name already loaded is resolved on the real
clock, and after the test the freezes running must be the ones that ran before it. A test that leaves a freeze
running fails at teardown; every test after it that would run on that freeze fails at setup, naming the test that
leaked it, so a leak blames only the test that made it. A freeze enclosing a group of tests on purpose (a class
decorator, a class- or module-scoped fixture) is not a leak: each test in it starts and ends inside it.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator

import freezegun.api
import pytest


def resolve_lazy_names() -> None:
    """Resolve every name a loaded ``threetears`` module serves lazily (a module ``__getattr__``), until none is left.

    A freeze started afterwards reads only objects that already exist, so it imports nothing under its fakes. A name
    whose module needs an extra this environment lacks raises ``ImportError`` here as it would to a reader, and a
    freeze's own walk skips it the same way; it is skipped.
    """
    seen = 0
    while seen != len(sys.modules):
        seen = len(sys.modules)
        for name, module in list(sys.modules.items()):
            if name.startswith("threetears.") and module is not None and "__getattr__" in vars(module):
                # every name dir() lists that the module does not hold is served by its __getattr__: the names a
                # freeze's walk reads, dunders included
                for attribute in set(dir(module)) - set(vars(module)):
                    try:
                        getattr(module, attribute)
                    except ImportError:  # NOSILENT: an extra this environment lacks; a freeze's walk skips it too
                        continue


#: Each freeze a test left running, by the test that left it, so a later test's setup can name it while it runs.
_LEAKS: list[tuple[str, object]] = []


def _running() -> list[object]:
    """The freezes running now, outermost first: freezegun's stack of clock factories, one per started freeze."""
    return list(freezegun.api.freeze_factories)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    """Resolve the lazy names before any of the test's fixtures is set up, a class- or module-scoped freeze included.

    Under a running freeze the names would be read on its fakes, the hazard itself; a freeze nested inside it walks no
    module, so nothing is read then either.

    :param item: The test about to be set up.
    :ptype item: pytest.Item
    :return: Nothing.
    :rtype: None
    """
    if not _running():
        resolve_lazy_names()


@pytest.fixture(autouse=True)
def _clock_isolation(request: pytest.FixtureRequest) -> Iterator[None]:
    """Fail a test that leaves a freeze it started running, and each test after it that would run on that freeze.

    A freeze running at setup is not by itself a leak: a class decorator or a class- or module-scoped fixture freezes
    for a group of tests on purpose, and each of them starts and ends inside it. A test leaked a freeze when one is
    running at its teardown that was not at its setup. A test that starts on a freeze an earlier test leaked fails
    at setup, naming that test, rather than being blamed at teardown for a freeze it did not start.
    """
    before = _running()
    live = [test for test, leaked in _LEAKS if any(running is leaked for running in before)]
    if live:
        pytest.fail(
            f"the clock was frozen before this test began, by a freeze an earlier test in this process leaked, not "
            f"this one. Leaked by: {live[-1]}.",
            pytrace=False,
        )
    yield
    started = [running for running in _running() if not any(running is held for held in before)]
    if started:
        _LEAKS.extend((request.node.nodeid, leaked) for leaked in started)
        pytest.fail(
            f"the clock is still frozen after this test, by {len(started)} freeze(s) it started: one was never "
            "stopped, or its start() raised, and every later test in the process reads a frozen clock. Freeze in a "
            "`with` block or a decorator, and look for the error its start() raised.",
            pytrace=False,
        )
