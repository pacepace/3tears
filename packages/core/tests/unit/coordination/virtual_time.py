"""a virtual-time event loop for the coordination unit tests: idle waits cost no wall-clock time.

:class:`VirtualTimeLoop` is a stock selector loop whose clock stands still while work is ready and
jumps straight to the next timer when nothing is: the production code runs unchanged, and every
interleaving is a function of the schedule alone. Code timed on the event loop's clock
(``loop.time()``, ``asyncio.sleep``, ``asyncio.timeout``) runs on it; code that reads the wall clock
(``time.monotonic``, ``datetime.now``) does not, and a test of such code cannot use it for that clock.
"""

from __future__ import annotations

import asyncio
import functools
import selectors
from collections.abc import Callable, Coroutine
from typing import Any

#: how long, on the REAL clock, the virtual loop waits for I/O when nothing at all is scheduled
#: before calling the test deadlocked -- rather than hanging the run.
_IDLE_REAL_SECONDS = 5.0


class _VirtualClock:
    """the one clock a :class:`VirtualTimeLoop` and its selector share."""

    def __init__(self) -> None:
        self.now = 0.0


# parity-exempt: a selector that delegates every registration to the platform's default selector and only changes how long select() waits
class _AdvancingSelector(selectors.BaseSelector):
    """polls the real selector without blocking, and advances virtual time instead of waiting.

    The loop asks ``select(timeout)`` with ``timeout`` the time until its next timer (``0`` when
    work is ready). Nothing arriving on a real file descriptor means nothing could happen before that
    timer, so the clock moves to it at once.
    """

    def __init__(self, clock: _VirtualClock) -> None:
        self._inner = selectors.DefaultSelector()
        self._clock = clock

    def register(self, fileobj: Any, events: int, data: Any = None) -> selectors.SelectorKey:
        return self._inner.register(fileobj, events, data)

    def unregister(self, fileobj: Any) -> selectors.SelectorKey:
        return self._inner.unregister(fileobj)

    def modify(self, fileobj: Any, events: int, data: Any = None) -> selectors.SelectorKey:
        return self._inner.modify(fileobj, events, data)

    def select(self, timeout: float | None = None) -> list[tuple[selectors.SelectorKey, int]]:
        ready = self._inner.select(0)
        if not ready and timeout is not None and timeout > 0:
            self._clock.now += timeout
        elif not ready and timeout is None:
            # nothing scheduled and nothing ready: only another thread could wake the loop now.
            ready = self._inner.select(_IDLE_REAL_SECONDS)
            if not ready:
                raise RuntimeError("virtual-time loop is idle with nothing scheduled: the test deadlocked")
        return ready

    def close(self) -> None:
        self._inner.close()

    def get_key(self, fileobj: Any) -> selectors.SelectorKey:
        return self._inner.get_key(fileobj)

    def get_map(self) -> Any:
        return self._inner.get_map()


class VirtualTimeLoop(asyncio.SelectorEventLoop):
    """a selector event loop whose ``time()`` is virtual and skips every idle wait."""

    def __init__(self) -> None:
        self.virtual_clock = _VirtualClock()
        super().__init__(selector=_AdvancingSelector(self.virtual_clock))

    def time(self) -> float:
        return self.virtual_clock.now


def in_virtual_time(test: Callable[..., Coroutine[Any, Any, None]]) -> Callable[..., None]:
    """run an async test body to completion on a fresh :class:`VirtualTimeLoop`.

    :param test: the async test function
    :ptype test: Callable[..., Coroutine[Any, Any, None]]
    :return: a synchronous test pytest collects with the same fixtures
    :rtype: Callable[..., None]
    """

    @functools.wraps(test)
    def run(*args: Any, **kwargs: Any) -> None:
        asyncio.run(test(*args, **kwargs), loop_factory=VirtualTimeLoop)

    return run
