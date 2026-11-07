"""pytest plugin: no test starts a freeze that imports under the fake clock, and none leaves the clock frozen.

freezegun's ``start()`` swaps ``datetime.date`` and ``datetime.datetime`` for its fakes and only then reads every
attribute of every loaded module, to repoint the ones bound to the real clock. Reading an attribute a package serves
lazily (a module ``__getattr__``, as ``threetears.evals`` serves ``Model``) imports the module that defines it, under
the fakes. A pydantic model there declaring a ``date`` field cannot build on ``FakeDate``, so ``start()`` raises
partway, after it froze the clock and before ``with`` had entered: nothing ever calls ``stop()``. The clock stays
frozen for every later test in the process, and an import of that model in any of them fails the same way. That needs
only one process holding a freeze and a lazy package nobody had read yet, which a runner interleaving packages
(the testament pool) gives and a per-package shard does not.

So before each test every lazy ``threetears`` name already loaded is resolved, on the real clock, and after it the
clock must be the real one again. A test that leaves it frozen fails at teardown; every test after it in the process
fails at setup, naming the test that leaked it, so a leak blames only the test that made it.
"""

from __future__ import annotations

import datetime
import sys
import time
from collections.abc import Iterator

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


def frozen_names() -> list[str]:
    """The clock names freezegun has swapped for its fakes, empty when the clock is the real one.

    Read off each object's module, not compared with a saved real object: a freeze repoints every module attribute
    bound to the real clock, a saved copy here among them.
    """
    clock = {
        "datetime.datetime": datetime.datetime,
        "datetime.date": datetime.date,
        "time.time": time.time,
        "time.monotonic": time.monotonic,
    }
    return [name for name, value in clock.items() if value.__module__ not in ("datetime", "time")]


#: Each test after whose teardown the clock was found frozen, so a later test's setup can name the last of them.
_LEAKED_BY: list[str] = []


@pytest.fixture(autouse=True)
def _clock_isolation(request: pytest.FixtureRequest) -> Iterator[None]:
    """Refuse to start on a frozen clock; resolve the lazy names; fail the test if it leaves the clock frozen."""
    if frozen := frozen_names():
        leaker = _LEAKED_BY[-1] if _LEAKED_BY else "a freeze outside any test (at import or collection)"
        pytest.fail(
            f"the clock was frozen before this test began ({', '.join(frozen)} are freezegun's): an earlier test in "
            f"this process leaked it, not this one. Leaked by: {leaker}.",
            pytrace=False,
        )
    resolve_lazy_names()
    yield
    if frozen := frozen_names():
        _LEAKED_BY.append(request.node.nodeid)
        pytest.fail(
            f"the clock is still frozen after this test ({', '.join(frozen)} are freezegun's): a freeze it started "
            "was never stopped, or its start() raised, and every later test in the process reads a frozen clock. "
            "Freeze in a `with` block or a decorator, and look for the error its start() raised.",
            pytrace=False,
        )
