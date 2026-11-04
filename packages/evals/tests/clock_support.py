"""One fixed instant for the evals suite's own tests, and a clock frozen at it.

A stored document built without its timestamp stamps the wall clock (``utc_now_iso``), and a test that read the
clock is a fact about one moment, which the test pool cannot cache. So a helper that builds a document its test never
reads the stamp of states :data:`FIXED_AT`, and a test whose code under test stamps the clock itself, and that asserts
nothing about time, runs under :func:`frozen_clock`. A test whose subject is "now" keeps the real clock.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

from freezegun import freeze_time

#: The instant every fixed stamp and frozen clock in the suite reads. A datetime, not an ISO string: freezegun parses
#: a string with dateutil, and that parse reads the wall clock. Its microsecond is set so it is spelled as a clock
#: reading is, fractional seconds and all.
FIXED_INSTANT = datetime(2026, 7, 20, 0, 0, 0, 1, tzinfo=UTC)

#: :data:`FIXED_INSTANT` as a stored document's stamp.
FIXED_AT = FIXED_INSTANT.isoformat()


@contextmanager
def frozen_clock(*, step_seconds: float = 0.0) -> Iterator[None]:
    """The wall clock held at :data:`FIXED_INSTANT` while the block runs.

    The event loop keeps its own clock, so a timeout or a sleep still ends. Reading the frozen clock reads no clock.

    :param step_seconds: How far each read moves the clock on, for a test that orders documents by their stamps;
        ``0`` holds it still.
    :ptype step_seconds: float
    :return: Nothing; the clock is frozen until the block ends.
    :rtype: Iterator[None]
    """
    _import_every_lazy_name()
    with freeze_time(FIXED_INSTANT, real_asyncio=True, auto_tick_seconds=step_seconds):
        yield


def _import_every_lazy_name() -> None:
    """Resolve every name a loaded ``threetears`` module serves lazily (a module ``__getattr__``), until none is left.

    Freezing reads every attribute of every loaded module after it has swapped ``datetime.date`` for its fake, so a
    lazy name resolved then imports its module under the fake, and a pydantic model there declaring a ``date`` field
    fails to build. Resolved here first, the freeze imports nothing.
    """
    seen = 0
    while seen != len(sys.modules):
        seen = len(sys.modules)
        for name, module in list(sys.modules.items()):
            if name.startswith("threetears.") and module is not None and "__getattr__" in vars(module):
                for attribute in dir(module):
                    getattr(module, attribute)
