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
clock must be the real one again.
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
                for attribute in dir(module):
                    if attribute.startswith("_"):
                        continue
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


@pytest.fixture(autouse=True)
def _clock_isolation() -> Iterator[None]:
    """Resolve the lazy names before the test; fail it if the clock is still frozen after every other teardown."""
    resolve_lazy_names()
    yield
    if frozen := frozen_names():
        pytest.fail(
            f"the clock is still frozen after this test ({', '.join(frozen)} are freezegun's): a freeze it started "
            "was never stopped, or its start() raised, and every later test in the process reads a frozen clock. "
            "Freeze in a `with` block or a decorator, and look for the error its start() raised.",
            pytrace=False,
        )
