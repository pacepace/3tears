"""how late a test's event loop runs past what the host's own idle loop was late at the same moments.

A loop-timing test bounds how long work beside the event loop (threads, a worker process) holds it. A ticker on the
loop measures that hold plus every delay the host itself put on the loop's thread: the scheduler, other processes,
other test batches on a shared runner (GitHub's 4-vCPU runner, a testament pool host). The second part is no property
of the code, and a fixed bound over both fails on a loaded host.

So the host's idle loop ticks beside the test in a process of its own for the whole measurement
(:class:`HostTicker`): a 1 ms ticker with nothing else to do, on the same clock. Whatever makes it late makes any loop
on the host late at that moment; what makes only the test's loop late is the test's own process.
:func:`added_lateness` counts, of each of the test loop's ticks, only the lateness no host tick shared at the same
moment, so a load spike credits exactly the time it overlapped and never hides a stall before or after it.

A measurement is trusted only when the host's idle loop itself stayed within a declared limit over the window
(:func:`trusted`): a host too busy to run an idle loop on time is too busy to say anything about the code's loop. An
untrusted measurement is never a pass and never a skip: it is retaken, a bounded number of times
(:data:`MEASURE_ATTEMPTS`), and when no window is trusted the test fails with :class:`CouldNotMeasure` ("could not
measure: the host's idle loop ran X ms late"), which says the host, not the code, was late. A trusted measurement
over the budget is a real red.

Usage:

.. code-block:: python

    async def measure():
        with HostTicker() as host:
            ticks = await ticks_while(run_the_work)
        return ticks, host.ticks

    ticks, host = await trusted(measure, lambda measured: measured, host_may_be_late=BUDGET)
    assert added_lateness(ticks, host) < BUDGET
"""

from __future__ import annotations

import array
import asyncio
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable
from typing import Any, Final, TypeVar

__all__ = [
    "MEASURE_ATTEMPTS",
    "CouldNotMeasure",
    "HostTicker",
    "Tick",
    "added_lateness",
    "hold_the_loop",
    "host_lateness",
    "latest",
    "ticks_while",
    "trusted",
]

_Measured = TypeVar("_Measured")

#: how many times a measurement the host was too busy for is taken before the test fails as unmeasurable: enough for a
#: passing burst of load (another batch's import, a GC of a neighbour) to end, few enough that a host busy for the
#: whole run fails in a bounded time
MEASURE_ATTEMPTS: Final[int] = 3

#: one tick of a ticker: when it asked to sleep and when it woke, on ``time.perf_counter`` (the host's monotonic clock,
#: one clock for every process on the host)
Tick = tuple[float, float]

#: what each of the host's idle ticks asks to sleep
_HOST_TICK_SECONDS = 0.001

#: the host's idle loop, run beside the test in a process of its own: a 1 ms ticker on an event loop with nothing else
#: to do, recording every tick on the same clock, until its stdin closes
_HOST_TICKER = """
import array, asyncio, sys, time

async def main():
    ticks = array.array("d")
    stop = asyncio.Event()
    asyncio.get_running_loop().add_reader(sys.stdin.fileno(), stop.set)
    sys.stdout.write("ticking\\n")
    sys.stdout.flush()
    while not stop.is_set():
        asked = time.perf_counter()
        await asyncio.sleep(0.001)
        ticks.append(asked)
        ticks.append(time.perf_counter())
    sys.stdout.buffer.write(ticks.tobytes())
    sys.stdout.flush()

asyncio.run(main())
"""


class HostTicker:
    """the host's idle loop ticking in a process of its own while the block runs; its ticks in :attr:`ticks` after.

    :ivar ticks: every tick of the host's idle loop over the block, in order (empty until the block ends)
    """

    def __init__(self) -> None:
        self.ticks: list[Tick] = []
        self._process: subprocess.Popen[bytes] | None = None

    def __enter__(self) -> HostTicker:
        process = subprocess.Popen([sys.executable, "-c", _HOST_TICKER], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self._process = process
        assert process.stdout is not None
        assert process.stdout.readline() == b"ticking\n"
        return self

    def __exit__(self, *exc: object) -> None:
        process = self._process
        assert process is not None and process.stdin is not None and process.stdout is not None
        process.stdin.close()
        recorded = array.array("d")
        recorded.frombytes(process.stdout.read())
        process.wait(timeout=10)
        self.ticks = list(zip(recorded[0::2], recorded[1::2], strict=True))


async def ticks_while(work: Callable[[], Awaitable[Any]], *, every: float = _HOST_TICK_SECONDS) -> list[Tick]:
    """every tick of a ticker on the running loop while ``work()`` ran.

    :param work: what to run on the loop (awaited once)
    :ptype work: Callable[[], Awaitable[Any]]
    :param every: what each tick asks to sleep
    :ptype every: float
    :return: the ticks, in order
    :rtype: list[Tick]
    """
    ticks: list[Tick] = []
    stop = asyncio.Event()

    async def ticker() -> None:
        while not stop.is_set():
            asked = time.perf_counter()
            await asyncio.sleep(every)
            ticks.append((asked, time.perf_counter()))

    ticking = asyncio.create_task(ticker())
    await asyncio.sleep(0.02)
    ticks.clear()
    try:
        await work()
    finally:
        stop.set()
        await ticking
    return ticks


def latest(ticks: list[Tick], *, every: float = _HOST_TICK_SECONDS) -> float:
    """the latest any of ``ticks`` woke past the ``every`` it asked for, in seconds (0 for none).

    :param ticks: a ticker's ticks
    :ptype ticks: list[Tick]
    :param every: what each tick asked to sleep
    :ptype every: float
    :return: the worst lateness
    :rtype: float
    """
    return max((woke - asked - every for asked, woke in ticks), default=0.0)


def added_lateness(ticks: list[Tick], host: list[Tick], *, every: float = _HOST_TICK_SECONDS) -> float:
    """the most any of ``ticks`` was late while the host's idle loop was not, in seconds.

    A tick is late from the ``every`` it asked for until it woke; the host's ticks are late over the same kind of
    span. Only the part of a tick's lateness that no host tick was late for at the same moment counts: a load spike
    credits exactly the time it overlapped, never a stall before or after it.

    :param ticks: the test loop's ticks while the work ran
    :ptype ticks: list[Tick]
    :param host: the host loop's ticks over the whole run, in order (:attr:`HostTicker.ticks`)
    :ptype host: list[Tick]
    :param every: what each of ``ticks`` asked to sleep
    :ptype every: float
    :return: the worst lateness the host loop does not share
    :rtype: float
    :raises AssertionError: when a tick falls outside the host loop's ticking (no baseline at that moment)
    """
    late_spans = [(asked + _HOST_TICK_SECONDS, woke) for asked, woke in host if woke - asked > _HOST_TICK_SECONDS]
    worst = 0.0
    for asked, woke in ticks:
        assert host and host[0][0] <= asked and woke <= host[-1][1], (
            f"no host baseline at {asked:.4f}-{woke:.4f}: the host loop ticked "
            f"{host[0][0] if host else None}-{host[-1][1] if host else None}"
        )
        start = asked + every
        if woke <= start:
            continue
        shared = sum(max(0.0, min(woke, end) - max(start, begin)) for begin, end in late_spans)
        worst = max(worst, woke - start - shared)
    return worst


def hold_the_loop(seconds: float) -> None:
    """hold the loop (and the interpreter lock) for ``seconds`` without a turn: a red case's planted block.

    :param seconds: how long to hold it
    :ptype seconds: float
    """
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


class CouldNotMeasure(AssertionError):
    """the host's own idle loop ran later than a measurement can be trusted at, in every attempt: the host, not the code,
    was late. Never a pass, never a skip."""


def host_lateness(host: list[Tick], ticks: list[Tick]) -> float:
    """the latest the host's idle loop woke past its 1 ms over the span ``ticks`` cover, in seconds (0 for none).

    :param host: the host loop's ticks (:attr:`HostTicker.ticks`)
    :ptype host: list[Tick]
    :param ticks: the test loop's ticks; their first ask to their last wake is the measured window
    :ptype ticks: list[Tick]
    :return: the host's worst lateness inside the window
    :rtype: float
    """
    if not ticks:
        return 0.0
    began, ended = ticks[0][0], ticks[-1][1]
    return latest([(asked, woke) for asked, woke in host if woke >= began and asked <= ended])


async def trusted(
    measure: Callable[[], Awaitable[_Measured]],
    window: Callable[[_Measured], tuple[list[Tick], list[Tick]]],
    *,
    host_may_be_late: float,
    attempts: int = MEASURE_ATTEMPTS,
) -> _Measured:
    """the first of up to ``attempts`` measurements over which the host's idle loop was late by at most
    ``host_may_be_late``.

    :param measure: takes one measurement (runs the work with a :class:`HostTicker` beside it)
    :ptype measure: Callable[[], Awaitable[_Measured]]
    :param window: the measurement's ``(test loop ticks, host loop ticks)``
    :ptype window: Callable[[_Measured], tuple[list[Tick], list[Tick]]]
    :param host_may_be_late: the most the host's idle loop may be late for the measurement to say anything about the
        code: a test passes its own budget, so a host as late as the code may be leaves nothing to judge
    :ptype host_may_be_late: float
    :param attempts: how many measurements to take at most
    :ptype attempts: int
    :return: the trusted measurement
    :rtype: _Measured
    :raises CouldNotMeasure: when the host was too late in every attempt
    """
    seen: list[float] = []
    for _ in range(attempts):
        measured = await measure()
        ticks, host = window(measured)
        late = host_lateness(host, ticks)
        if late <= host_may_be_late:
            return measured
        seen.append(late)
    raise CouldNotMeasure(
        f"could not measure: the host's idle loop ran {max(seen) * 1000:.1f} ms late "
        f"(each of {attempts} attempts past {host_may_be_late * 1000:.1f} ms: "
        f"{', '.join(f'{late * 1000:.1f}' for late in seen)} ms)"
    )
