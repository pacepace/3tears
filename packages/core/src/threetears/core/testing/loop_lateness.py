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
(:func:`trusted`): a host too busy to run an idle loop on time is too busy to say anything about the code's loop. A
measured path that runs on several threads (DuckDB's) needs more than one free core, and a many-core host can be busy
enough to stretch it while a 1 ms idle loop stays punctual (a real-size serving read went 2.0x on a 10-core host at
load ~9, its idle loop on time). So such a measurement also names its threads, and before and after it a fixed spin
off the interpreter lock (:func:`_cpu_crowding`) runs on that many threads at once (held to the performance cores),
against the fastest this host has ever run it on that many -- a record kept per host under the user's cache dir, and
this session's first take, the lower of the two. With that many cores free it takes about as long; when fewer are
free its threads wait their turn, and past :data:`_CPU_MAY_BE_CROWDED` the window is untrusted. A host with no record
whose idle loop was late over the session's first take has no reference, and nothing in that session is trusted. An
untrusted measurement is never a pass and never a skip: it is retaken, a bounded number of times
(:data:`_MEASURE_ATTEMPTS`), and when no window is trusted the test fails with :class:`CouldNotMeasure` ("could not
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
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Final, TypeVar

from threetears.observe import BuildOnce

__all__ = [
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
_MEASURE_ATTEMPTS: Final[int] = 3

#: the most a spin on a measured path's threads at once may take over the host's fastest spin on as many for the window
#: to be trusted: about as long with that many cores free; half again as long means some of them waited for a core
_CPU_MAY_BE_CROWDED: Final[float] = 1.5

#: the most the host's idle loop may be late over a session's first spin for it to stand as the reference on a host
#: with no record yet
_REFERENCE_HOST_MAY_BE_LATE: Final[float] = 0.020

#: the block the CPU spin hashes: hashing a block this size lets go of the interpreter lock, so the spin's threads run
#: at once wherever cores are free
_SPIN_BLOCK: Final[bytes] = bytes(1 << 20)

#: how many blocks each thread of the spin hashes (a few milliseconds)
_SPIN_ROUNDS: Final[int] = 4

#: how many times each side of the spin is taken; its fastest is kept
_SPIN_TAKES: Final[int] = 3

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


def _spin() -> None:
    for _ in range(_SPIN_ROUNDS):
        hashlib.sha256(_SPIN_BLOCK).digest()


def _spun(threads: int) -> float:
    """the seconds ``threads`` threads take to each run the spin, all at once."""
    running = [threading.Thread(target=_spin) for _ in range(threads)]
    began = time.perf_counter()
    for thread in running:
        thread.start()
    for thread in running:
        thread.join()
    return time.perf_counter() - began


def _cpu_crowding(threads: int, probe: Callable[[int], float]) -> float:
    """how much longer a fixed spin takes on ``threads`` threads at once now than the fastest this host has ever run it
    on that many: about 1 with that many cores free, more as fewer are.

    The spin hashes a block off the interpreter lock (:data:`_SPIN_BLOCK`), so its threads run at once wherever cores
    are free; it is taken :data:`_SPIN_TAKES` times and its fastest kept. ``threads`` is held to the host's performance
    cores (:func:`_spin_threads`), so a host of fast and slow cores is compared on its fast ones alone. The reference is
    the lower of this session's first take and the host's record (:func:`_reference`), so a session busy from start to
    finish cannot calibrate itself busy; a faster take lowers the record. With no record and a first take the host's
    idle loop was late for, there is no reference: the answer is infinite, and nothing in the session is trusted.

    :param threads: the threads the measured path runs on
    :ptype threads: int
    :param probe: takes the spin on as many threads at once, in seconds (:func:`_taken` on a real host)
    :ptype probe: Callable[[int], float]
    :return: the spin's seconds now over the reference's
    :rtype: float
    """
    threads = _spin_threads(threads)
    reference = _reference(threads, probe)
    if math.isinf(reference):
        # no quiet reference: nothing this session takes is recorded (a busy take must never become the host's)
        return math.inf
    now = probe(threads)
    _record(threads, now)
    return now / reference


def _spin_threads(threads: int) -> int:
    """the threads a spin for a path on ``threads`` runs on: no more than the host's performance cores.

    :param threads: the threads the path runs on
    :ptype threads: int
    :return: the spin's threads
    :rtype: int
    """
    return max(1, min(threads, _performance_cores()))


def _taken(threads: int) -> float:
    """the fastest of :data:`_SPIN_TAKES` spins on ``threads`` threads at once."""
    return min(_spun(threads) for _ in range(_SPIN_TAKES))


def _usable_cores() -> int:
    """the cores this process may run on."""
    counted = getattr(os, "process_cpu_count", os.cpu_count)()
    return counted or 1


def _performance_cores() -> int:
    """the host's performance cores: Apple silicon's perflevel0 (its efficiency cores are slower, and a spin across
    both would read crowded on an idle host); every usable core elsewhere."""
    if sys.platform == "darwin":
        answered = _sysctl("hw.perflevel0.logicalcpu")
        if answered.isdigit():
            return max(1, min(int(answered), _usable_cores()))
    return _usable_cores()


def _sysctl(name: str) -> str:
    """a macOS sysctl value, or "" when there is none."""
    try:
        done = subprocess.run(["/usr/sbin/sysctl", "-n", name], capture_output=True, text=True, timeout=5, check=False)
    except OSError:
        return ""
    return done.stdout.strip() if done.returncode == 0 else ""


def _cpu_model() -> str:
    """the host's CPU model, as its OS names it."""
    if sys.platform == "darwin":
        named = _sysctl("machdep.cpu.brand_string")
        if named:
            return named
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:  # NOSILENT: no /proc/cpuinfo (macOS, a sandbox): the platform's own names below are the model
        pass
    return f"{platform.machine()} {platform.processor()}".strip()


def _record_path() -> Path:
    """the per-host record of the fastest spins, in the user's cache dir (never the repo)."""
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "threetears" / "cpu-spin-reference.json"


def _record_key(threads: int) -> str:
    return f"{socket.gethostname()}|{_cpu_model()}|{threads}"


def _stored(threads: int) -> float | None:
    """the host's fastest recorded spin on ``threads`` threads, or None."""
    try:
        held = json.loads(_record_path().read_text(encoding="utf-8"))
    except OSError, ValueError:  # NOSILENT: no record yet (or a torn one) is "no record": the session's take stands
        return None
    value = held.get(_record_key(threads)) if isinstance(held, dict) else None
    return float(value) if isinstance(value, (int, float)) and value > 0 else None


def _record(threads: int, seconds: float) -> None:
    """lower the host's record for ``threads`` to ``seconds`` when it is faster (written whole, by rename)."""
    stored = _stored(threads)
    if stored is not None and stored <= seconds:
        return
    path = _record_path()
    try:
        held = json.loads(path.read_text(encoding="utf-8"))
    except OSError, ValueError:
        held = {}
    if not isinstance(held, dict):
        held = {}
    held[_record_key(threads)] = seconds
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_suffix(f".{os.getpid()}.tmp")
    scratch.write_text(json.dumps(held, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(scratch, path)


#: this session's reference per probe and spin thread count: its first take, or None when no reference could be had;
#: taken once per probe and thread count, however many measurements ask first
_SESSION: Final[BuildOnce[tuple[Callable[[int], float], int], float | None]] = BuildOnce()


def _reference(threads: int, probe: Callable[[int], float]) -> float:
    """the spin on ``threads`` threads a window is compared with: the lower of this session's first take and the
    host's record. A host with no record trusts the first take only when its idle loop was on time over it; else the
    session has no reference (infinite: nothing compares under it)."""
    session = _SESSION.get((probe, threads), lambda: _first_take(threads, probe))
    stored = _stored(threads)
    if session is None:
        return stored if stored is not None else math.inf
    return session if stored is None else min(session, stored)


def _first_take(threads: int, probe: Callable[[int], float]) -> float | None:
    """this session's first spin on ``threads`` threads, recorded as the host's when the idle loop was on time over it
    or the host has a record; None when the host has no record and was busy (a busy take never becomes the host's).

    :param threads: the spin's threads
    :ptype threads: int
    :param probe: takes the spin
    :ptype probe: Callable[[int], float]
    :return: the session's reference, or None
    :rtype: float | None
    """
    stored = _stored(threads)
    with HostTicker() as host:
        began = time.perf_counter()
        first = probe(threads)
        ended = time.perf_counter()
    quiet = host_lateness(host.ticks, [(began, ended)]) <= _REFERENCE_HOST_MAY_BE_LATE
    if stored is None and not quiet:
        return None
    _record(threads, first)
    return first if stored is None else min(first, stored)


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
    attempts: int = _MEASURE_ATTEMPTS,
    threads: int | None = None,
    cpu_may_be_crowded: float = _CPU_MAY_BE_CROWDED,
    probe: Callable[[int], float] = _taken,
) -> _Measured:
    """the first of up to ``attempts`` measurements over which the host's idle loop was late by at most
    ``host_may_be_late`` (and, for a path on ``threads`` threads, its cores were free before and after it).

    :param measure: takes one measurement (runs the work with a :class:`HostTicker` beside it)
    :ptype measure: Callable[[], Awaitable[_Measured]]
    :param window: the measurement's ``(test loop ticks, host loop ticks)``
    :ptype window: Callable[[_Measured], tuple[list[Tick], list[Tick]]]
    :param host_may_be_late: the most the host's idle loop may be late for the measurement to say anything about the
        code: a test passes its own budget, so a host as late as the code may be leaves nothing to judge
    :ptype host_may_be_late: float
    :param attempts: how many measurements to take at most
    :ptype attempts: int
    :param threads: the threads the measured path runs on, when more than the loop's own: the window is then trusted
        only when :func:`_cpu_crowding` on that many threads, before and after it, is at most ``cpu_may_be_crowded``
    :ptype threads: int | None
    :param cpu_may_be_crowded: the most :func:`_cpu_crowding` may be
    :ptype cpu_may_be_crowded: float
    :param probe: takes a fixed spin on as many threads at once and answers its seconds: the host's own spin
        (:func:`_taken`) unless a test scripts one; each probe has its own session reference, and every probe's takes
        lower the host's record
    :ptype probe: Callable[[int], float]
    :return: the trusted measurement
    :rtype: _Measured
    :raises CouldNotMeasure: when the host was too late, or its cores too crowded, in every attempt
    """
    seen: list[str] = []
    for _ in range(attempts):
        before = _crowding(threads, probe)
        measured = await measure()
        after = _crowding(threads, probe)
        ticks, host = window(measured)
        verdict = _untrusted(host_lateness(host, ticks), (before, after), host_may_be_late, cpu_may_be_crowded)
        if verdict is None:
            return measured
        seen.append(verdict)
    raise _could_not_measure(seen, attempts)


def _crowding(threads: int | None, probe: Callable[[int], float]) -> float | None:
    """:func:`_cpu_crowding` on ``threads`` threads; None when the measurement names none (a loop's ticks alone)."""
    return None if threads is None else _cpu_crowding(threads, probe)


def _untrusted(
    late: float, crowding: tuple[float | None, float | None], host_may_be_late: float, cpu_may_be_crowded: float
) -> str | None:
    """why a window is untrusted, or None when it is trusted.

    :param late: the host's idle loop's worst lateness over the window
    :ptype late: float
    :param crowding: :func:`_cpu_crowding` before and after it (None when not asked)
    :ptype crowding: tuple[float | None, float | None]
    :param host_may_be_late: the most ``late`` may be
    :ptype host_may_be_late: float
    :param cpu_may_be_crowded: the most either crowding may be
    :ptype cpu_may_be_crowded: float
    :return: the reason, or None
    :rtype: str | None
    """
    if late > host_may_be_late:
        return f"the host's idle loop ran {late * 1000:.1f} ms late (past {host_may_be_late * 1000:.1f} ms)"
    worst = max((taken for taken in crowding if taken is not None), default=None)
    if worst is not None and math.isinf(worst):
        return "no quiet reference for the path's threads: this host has no record and was busy at the session's start"
    if worst is not None and worst > cpu_may_be_crowded:
        return (
            f"the path's threads waited for cores: a spin on them took {worst:.2f}x the host's fastest "
            f"(past {cpu_may_be_crowded:.2f}x)"
        )
    return None


def _could_not_measure(seen: list[str], attempts: int) -> CouldNotMeasure:
    """the failure of a measurement no attempt of which the host was quiet enough for.

    :param seen: why each attempt was untrusted
    :ptype seen: list[str]
    :param attempts: how many were taken
    :ptype attempts: int
    :return: the failure
    :rtype: CouldNotMeasure
    """
    return CouldNotMeasure(f"could not measure: {'; '.join(seen)} (each of {attempts} attempts untrusted)")
