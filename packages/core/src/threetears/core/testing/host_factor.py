"""how much more CPU this host spends on the same work than the reference host, right now: the loop-stall guard's
per-host calibration.

**Why.** :mod:`threetears.core.testing.loop_stall` bounds one event-loop turn by the loop thread's CPU, never the wall
clock, so a host's queue for a core cannot stretch it. An oversubscribed host still stretches the CPU the same work
spends -- caches, memory bandwidth and SMT siblings shared with other batches, a slower core: on the testament pool
the same tests took 1.4-3x the CPU seconds on its busiest runners (brooks-5950x: 30 slots on 16 cores; brooks-nas: 12
on 6) as on its quietest (monad-1; 2026-11-26, per-test ``cpu_user_s + cpu_sys_s`` from testament's history over 26
tests), so turns of 0.08-0.25 s of reference work passed or failed by runner. Alone in an exclusive suite the runners
measure close to each other. So, as :mod:`threetears.core.testing.calibrate` bounds a path by a multiple of a baseline
measured on the same host, the guard's limit is a multiple of what a fixed probe costs on this host now.

**The probe** is the mix a test's loop turns run: a Python pass that builds and sorts a dict of rows, and two
single-threaded Arrow kernels over 100,000 int64 values (``unique``, a hash, and ``sort_indices``), ~4 ms on the
reference host, timed on the calling thread's own CPU clock. Single-threaded on purpose: that clock sees only its own
thread. It runs on the loop thread (pytest's main thread, where the tests' loops run) between tests, never inside a
turn. Arrow is core's optional extra (``3tears[arrow]``) and this module imports without it: a process without
pyarrow runs no Arrow on its loop, so its probe is the Python half alone, against that half's reference.

**Smoothed** two ways, so one noisy reading moves nothing: a probe is the fastest of :data:`_TAKES` takes (an
interrupt or a collection only adds to a take), and the factor is the median of the last :data:`_WINDOW` probes (one
outlier of three cannot be the median). :data:`_WINDOW` probes are taken at the first sample, one more every
:data:`_PROBE_EVERY_SECONDS` after.

factor = max(1.0, median probe / reference): a host faster than the reference is held to the reference's limit,
never a tighter one.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, Final

try:
    import pyarrow as pa
    import pyarrow.compute as pc

    _PYARROW = True
except ImportError:  # NOSILENT: the arrow extra is optional; without it the probe is its Python half, said so
    _PYARROW = False

__all__ = ["HostFactor", "Limit", "Reading", "limit_for"]

#: what each half of one probe costs on the reference host, in its loop thread's CPU seconds.
#:
#: The reference host is monad-1, the pool runner that ran most of the 0.25 s limit's rollout (2026-11-25) and the
#: quietest in the 2026-11-26 history. Measured there 2026-11-26 in the testament image (CPython 3.14.3, pyarrow as
#: locked), load 0.2-3.8 on 64 CPUs, the median of 20 probes each the best of 3 takes, in 16 test processes over two
#: runs: the whole probe 3.44-3.56 ms in ten, 3.82-3.93 ms in six (the cores they landed on), the Python half alone
#: 1.89-1.92 ms and 2.10 ms. The halves below are the slower processes' (2.1 ms, and 3.9 - 2.1 = 1.8 ms), so the
#: reference runner reads 1.0 whichever core it gets. Beside it, the same day: markpace-mbp's runner (an M4's Linux
#: VM) alone in the exclusive loop_timing suite, 3.37 ms; the M4 itself under macOS at load ~3, 4.0-4.5 ms (factor
#: 1.0-1.15).
_REFERENCE_PYTHON_SECONDS: Final[float] = 0.0021
_REFERENCE_ARROW_SECONDS: Final[float] = 0.0018

#: the largest factor the guard scales by; past it a turn over the scaled limit is not judged (could not measure).
#:
#: Why 4x. Over the 26 tests of the 2026-11-26 history, brooks-nas spent a median 3.5x (2.8-5.0x) and brooks-5950x
#: 2-3.3x the CPU monad-1 spent on the same test. 4x keeps the two busiest runners judged on a typical day, and at
#: 0.25 s x 4 = 1.0 s a judged turn is still a twentieth of the 24 s copy-follower freeze this guard exists for. Past
#: it the same runner's own spread is as wide as the factor (brooks-nas: 64-141 s on one test, 2.2x), so a scaled bound
#: would judge that moment's load, not the code.
_FACTOR_CAP: Final[float] = 4.0

#: how often the factor is re-measured, between tests
_PROBE_EVERY_SECONDS: Final[float] = 30.0

#: takes a probe is the fastest of
_TAKES: Final[int] = 3

#: probes the factor is the median of
_WINDOW: Final[int] = 3

#: the Python half's rows, and the Arrow half's values
_PY_ROWS: Final[int] = 20_000
_ARROW_VALUES: Final[int] = 100_000


def _values() -> Any:
    """the Arrow half's input: 100,000 int64 values, ~20,000 distinct, unsorted; None without pyarrow."""
    if not _PYARROW:
        return None
    return pa.array([(index * 7919) % 20011 for index in range(_ARROW_VALUES)], type=pa.int64())


def _probe_seconds(values: Any) -> float:
    """one take of the probe, in the calling thread's CPU seconds: the Python half, and the Arrow half over
    ``values`` (skipped when None, without pyarrow)."""
    began = time.thread_time()
    rows = {f"k{index % 997}": index for index in range(_PY_ROWS)}
    ordered = sorted(rows.items(), key=lambda row: row[1] % 101)
    if values is not None:
        distinct = pc.unique(values)
        order = pc.sort_indices(values)
        assert len(distinct) > 0 and len(order) == len(values)
    spent = time.thread_time() - began
    assert len(ordered) == 997
    return spent


def _reference_seconds() -> float:
    """the reference probe's seconds for the probe this process runs: both halves, or the Python one without Arrow."""
    return _REFERENCE_PYTHON_SECONDS + (_REFERENCE_ARROW_SECONDS if _PYARROW else 0.0)


@dataclass(frozen=True)
class Reading:
    """the host's factor now, and the probes it is from.

    :ivar factor: max(1.0, median probe / reference), uncapped
    :ivar probes: the last probes, seconds each, oldest first
    :ivar reference: the reference probe's seconds
    :ivar cap: the largest factor a limit is scaled by
    """

    factor: float
    probes: tuple[float, ...]
    reference: float
    cap: float

    @property
    def judgeable(self) -> bool:
        """whether the factor is within the cap: a turn over the scaled limit is then a red.

        :return: whether it is
        :rtype: bool
        """
        return self.factor <= self.cap

    def told(self) -> str:
        """the reading as a failure says it.

        :return: the text
        :rtype: str
        """
        each = "/".join(f"{probe * 1000:.2f}" for probe in self.probes)
        return (
            f"host factor {self.factor:.2f}: the probe took {statistics.median(self.probes) * 1000:.2f} ms of loop-thread "
            f"CPU (median of {each} ms, each the best of {_TAKES} takes) against its "
            f"{self.reference * 1000:.2f} ms reference"
        )


@dataclass(frozen=True)
class Limit:
    """the CPU seconds one turn of a test's loop may hold it on this host.

    :ivar seconds: the bound a turn is held to
    :ivar reference: the reference seconds it scales (0.25 s, or a test's ``loop_stall_limit``)
    :ivar reading: the host's factor; None in an exclusive suite, where no factor applies
    """

    seconds: float
    reference: float
    reading: Reading | None

    @property
    def judgeable(self) -> bool:
        """whether a turn over :attr:`seconds` is a red (else the host was too busy to judge it).

        :return: whether it is
        :rtype: bool
        """
        return self.reading is None or self.reading.judgeable

    def told(self) -> str:
        """the limit as a failure says it: ``0.450 s on this host: 0.250 s x 1.80 (host factor 1.80: ...)``.

        :return: the text
        :rtype: str
        """
        if self.reading is None:
            return f"{self.seconds:.3f} s (an exclusive suite, loop_timing: no host factor)"
        reading = self.reading
        scale = f"{min(reading.factor, reading.cap):.2f}"
        capped = f", capped at {reading.cap:.1f}x" if not reading.judgeable else ""
        return f"{self.seconds:.3f} s on this host: {self.reference:.3f} s x {scale}{capped} ({reading.told()})"


def limit_for(reference_seconds: float, reading: Reading | None) -> Limit:
    """the limit a turn is held to: ``reference_seconds`` x the host's factor (capped), or exactly
    ``reference_seconds`` in an exclusive suite (``reading`` None).

    :param reference_seconds: the limit on the reference host (the guard's 0.25 s, or a test's raised one)
    :ptype reference_seconds: float
    :param reading: the host's factor, or None in an exclusive suite
    :ptype reading: Reading | None
    :return: the limit
    :rtype: Limit
    """
    if reading is None:
        return Limit(seconds=reference_seconds, reference=reference_seconds, reading=None)
    return Limit(
        seconds=reference_seconds * min(reading.factor, reading.cap), reference=reference_seconds, reading=reading
    )


class HostFactor:
    """the host's factor, probed at the first sample and every ``every`` seconds after, on the calling thread (call
    :meth:`sample_if_due` between tests, on the loop thread)."""

    #: the factor a test scripts in place of the probe (each take then reads that many times the reference), read at
    #: each take; None for the real probe. A subject session of the guard's own tests sets it from its conftest, so
    #: its verdicts rest on the factor the case names, never on the runner
    scripted: ClassVar[Callable[[], float] | None] = None

    def __init__(
        self,
        *,
        take: Callable[[], float] | None = None,
        clock: Callable[[], float] = time.monotonic,
        reference: float | None = None,
        cap: float = _FACTOR_CAP,
        every: float = _PROBE_EVERY_SECONDS,
    ) -> None:
        """
        :param take: one take of the probe, its seconds; None for the real probe (or :attr:`scripted`)
        :ptype take: Callable[[], float] | None
        :param clock: when, to space the probes
        :ptype clock: Callable[[], float]
        :param reference: the reference probe's seconds; None for the reference host's, measured (both halves, or the
            Python half where pyarrow is not installed)
        :ptype reference: float | None
        :param cap: the largest factor a limit scales by
        :ptype cap: float
        :param every: seconds between probes after the first sample
        :ptype every: float
        """
        self._take = take
        self._values: Any = None
        self._warm = False
        self._clock = clock
        self._reference = _reference_seconds() if reference is None else reference
        self._cap = cap
        self._every = every
        self._probes: tuple[float, ...] = ()
        self._last = 0.0

    @property
    def reference(self) -> float:
        """the reference probe's seconds the factor divides by.

        :return: seconds
        :rtype: float
        """
        return self._reference

    @property
    def cap(self) -> float:
        """the largest factor a limit scales by.

        :return: the cap
        :rtype: float
        """
        return self._cap

    def _one_take(self) -> float:
        if self._take is not None:
            return self._take()
        scripted = HostFactor.scripted
        if scripted is not None:
            return self._reference * scripted()
        if not self._warm:
            self._values = _values()
            _probe_seconds(self._values)  # warm: the first call pays kernel and allocator set-up
            self._warm = True
        return _probe_seconds(self._values)

    def _probe(self) -> float:
        return min(self._one_take() for _ in range(_TAKES))

    def sample_if_due(self) -> None:
        """take :data:`_WINDOW` probes at the first call, one more once ``every`` seconds have passed.

        :return: nothing
        :rtype: None
        """
        now = self._clock()
        if not self._probes:
            self._probes = tuple(self._probe() for _ in range(_WINDOW))
        elif now - self._last >= self._every:
            self._probes = (*self._probes, self._probe())[-_WINDOW:]
        else:
            return
        self._last = now

    def reading(self) -> Reading:
        """the factor now, sampling first if no probe has been taken.

        :return: the reading
        :rtype: Reading
        """
        if not self._probes:
            self.sample_if_due()
        factor = max(1.0, statistics.median(self._probes) / self._reference)
        return Reading(factor=factor, probes=self._probes, reference=self._reference, cap=self._cap)
