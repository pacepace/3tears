"""run a real loop watchdog on a clock the test moves, so a stall is exactly as long as the test says.

The watchdog's thread and the loop's heartbeat both run for real; only the time they read is the
test's. The clock stands still unless the test moves it, so a loop that keeps running is silent
however slowly the machine runs, and a hold is a stall of exactly the length the test gave it. Each
step of a hold waits until the watchdog has looked at the moved clock, so what the watchdog logs
depends only on the steps, never on how the threads were scheduled.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from threetears.observe.loop_watchdog import LoopWatchdog

__all__ = ["HANG_SECONDS", "Log", "TestClock", "Watched"]

#: how long a wait for the watchdog may take before the test fails instead of hanging; never a measure
HANG_SECONDS = 10.0


class Log:
    """records what the watchdog logs: warnings, and the contained failures ``contain`` logs."""

    def __init__(self) -> None:
        self.warnings: list[tuple[str, dict[str, Any]]] = []
        self.records: list[tuple[int, str, dict[str, Any]]] = []

    def warning(self, message: str, *, extra: dict[str, Any]) -> None:
        self.warnings.append((message, extra["extra_data"]))

    def log(self, level: int, message: str, *, exc_info: Any = None, extra: dict[str, Any]) -> None:
        self.records.append((level, message, extra["extra_data"]))

    def stalls(self) -> list[dict[str, Any]]:
        """what each stall warning carried, in order."""
        return [data for message, data in self.warnings if "is stalled" in message]

    def ends(self) -> list[dict[str, Any]]:
        """what each end-of-stall warning carried, in order."""
        return [data for message, data in self.warnings if "ran again" in message]


class TestClock:
    """a monotonic clock that moves only when the test moves it."""

    __test__ = False

    def __init__(self) -> None:
        self._now = 100.0
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self._now

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._now += seconds


class Watched:
    """a started-on-demand :class:`LoopWatchdog` on a :class:`TestClock`, counting the looks its thread takes.

    :param kwargs: passed to :class:`LoopWatchdog`, all but ``clock``
    :ptype kwargs: Any
    """

    def __init__(self, **kwargs: Any) -> None:
        self.clock = TestClock()
        self._looks = 0
        self._looked = threading.Condition()
        self.watchdog = LoopWatchdog(clock=self.clock, looked=self._counted, **kwargs)

    def _counted(self) -> None:
        """the watchdog's ``looked``: one more whole look taken."""
        with self._looked:
            self._looks += 1
            self._looked.notify_all()

    def looked(self) -> None:
        """block until the watchdog has taken one whole look that began after this call."""
        with self._looked:
            # a look under way now may have read the clock before the caller moved it: wait past it
            target = self._looks + 2
            seen = self._looked.wait_for(lambda: self._looks >= target, timeout=HANG_SECONDS)
        assert seen, "the watchdog thread stopped looking"

    async def ran_again(self) -> None:
        """let the loop beat at the clock's reading now, then wait for the watchdog to look at that beat."""
        async with asyncio.timeout(HANG_SECONDS):
            while self.watchdog.last_beat != self.clock():
                await asyncio.sleep(0.001)
        await asyncio.to_thread(self.looked)

    def hold(self, *steps: float) -> None:
        """hold the loop: called on it, no beat runs until this returns.

        the clock moves by each step in turn, and a look of the watchdog's sees each move before the
        next, while the caller's frame is on the loop thread's stack.

        :param steps: how far the clock moves at each step, in seconds
        :ptype steps: float
        """
        for step in steps:
            self.clock.advance(step)
            self.looked()

    async def runs(self, steps: int, step: float) -> None:
        """the loop keeps running while the clock moves ``steps`` times by ``step``, beating after each.

        :param steps: how many times the clock moves
        :ptype steps: int
        :param step: how far each move is, shorter than the stall bound
        :ptype step: float
        """
        for _ in range(steps):
            self.clock.advance(step)
            await self.ran_again()
