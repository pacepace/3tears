"""pytest plugin: a test fails when one turn of an event loop held the loop past :data:`_STALL_LIMIT_SECONDS` (x this
host's slowdown in a shared suite).

**Why.** A serving process answers every call on one event loop, so CPU work on the loop (a pyarrow or numpy pass over
rows, a Python loop a row) holds every other call for as long as it runs. Static rules cannot see it: the same code is
quick at a test's size and slow at a real one. A reports pod's copy follower froze its loop 12.2 s and 24.3 s in
``copy_follower._stage_keyed`` -> ``arrow_keys.distinct_keys`` (2026-11-25), every test of it green. This makes the
rule "no CPU work on the loop" a test failure: any test, at whatever size it runs, whose loop ran one turn past the
limit fails, naming the function that held it.

**How a turn is timed.** Every callback asyncio's loop runs -- a task's step, a socket's read -- is one turn, run
through ``Handle._run``; :func:`~threetears.core.testing._asyncio_internals.timed_handles` (the detector
:func:`~threetears.core.testing.loop_lateness.turn_holds` already uses) wraps it and times each turn on the loop
thread's own CPU clock. CPU, never the wall clock: a loaded host (xdist's workers, a testament pool runner beside other
batches) stretches a turn's wall time past any fixed bound with no fault of the code, and this guard must never fail a
test the code did not earn. What it does not see, so: a turn that blocks without spending CPU (``time.sleep``, a
synchronous socket read on the loop), and a loop that does not run its callbacks through ``Handle._run`` (uvloop).

**How the culprit is named.** A watcher thread looks at each loop thread every :data:`_SAMPLE_EVERY_SECONDS`, reading
its CPU clock (:mod:`~threetears.core.testing._thread_clock`); a turn running longer than that has its thread's stack
sampled (``sys._current_frames``, through :mod:`~threetears.core.testing._interpreter_internals`) while it runs, each
sample worth the CPU the thread spent since the look before it. When the turn ends past the limit, the failure names
the innermost frame of the project's own code (not the standard library or an installed package) that holds the most
of the samples' CPU -- ``threetears.core.arrow_keys.distinct_keys``, not numpy's ``unique`` under it -- and prints the
stack of the last sample that held it. CPU, never samples counted as wall time: a turn that waited 1 s on
``future.result()`` while a worker burned, then spent 0.3 s of its own, was blamed on the wait, which spends no CPU
(2026-11-26). Where the platform reads no other thread's CPU (neither Linux nor macOS), each sample counts one. A C call that keeps the interpreter lock from start
to end (a pyarrow kernel) lets the watcher run only when it returns, and the sample then shows the Python frame that
called it: still the culprit. A turn no sample saw names the callback asyncio ran.

**The limit** is :data:`_STALL_LIMIT_SECONDS` of the reference host's CPU, one setting for every test of every repo
that loads the plugin. A test whose turn must run longer is a bug in the code under test (move the work off the loop:
``asyncio.to_thread`` for IO and work that lets the lock go, a worker process for CPU, data through
:mod:`threetears.core.shared_arrow`) or in the test (synchronous setup belongs outside the loop, in a sync fixture).
There is no exemption list. A test may raise its own limit only with ``@pytest.mark.loop_stall_limit(seconds=...,
why="...")``: ``seconds`` above the limit and a ``why`` that says why (an issue, a reason), checked at collection -- a
marker that does not is a usage error. Its ``seconds`` are reference seconds too, scaled by the host's factor as the
limit is.

**Calibrated per host in a shared suite.** A CPU clock skips a process's wait for a core, but an oversubscribed host
still stretches the CPU the same work spends: on the testament pool the same tests took 1.4-3x the CPU seconds on its
busiest runners (brooks-5950x, 30 slots on 16 cores; brooks-nas, 12 on 6) as on its quietest, monad-1 (2026-11-26),
so turns of 0.08-0.25 s of reference work passed or failed by runner. So in a shared suite (the default unit suite,
the integration suite: tests beside other batches) the limit is 0.25 s x this host's factor
(:mod:`threetears.core.testing.host_factor`): a fixed ~4 ms probe, a Python pass and two Arrow kernels, timed on the
loop thread's CPU clock before the first test and every 30 s after, between tests, never inside a turn; the factor is
max(1, median of the last three probes, each the best of three takes / monad-1's 3.9 ms; 2.1 ms for the Python half
alone where pyarrow is not installed). It is capped at 4x (``host_factor._FACTOR_CAP``, with its why): on a host past
the cap a turn over 0.25 s x 4 is not judged, and the
phase fails as the timing suite's too-busy host does, "could not measure: this host spends N x the reference host's
CPU...", naming the host and the stall -- never a pass, never a skip, never a stall red. In an exclusive suite (a test
marked ``loop_timing``: every ``exclusive = true`` suite of ``.testament.toml`` selects those and only those, and CI
runs them serially) the limit is exactly 0.25 s, no factor and no probe: alone on its runner the host is the
reference's peer, and a bound loosened there would hide work. Every failure names the limit it was held to, the
factor and the probes: "the limit is 0.450 s on this host: 0.250 s x 1.80 (host factor 1.80: the probe took 7.02 ms
...)".

**What a test is charged for.** Its own work, never the collector's walk of what earlier tests left. Python's
automatic full (oldest-generation) collection walks every tracked object the process holds, and in a test process that
is every earlier test's leftovers: on the testament pool trivial turns "stalled" ~1 s, the collector walking an earlier
test's heap (0.10 s at ~300k tracked objects, ~1 s on bigger heaps; 2026-11-25). A pod does not pay that: the SDK's
``settle_before_serving`` freezes its start-up heap once before it serves, so its collections walk only what serving
made. The guard does the same before each test: a young collection (``gc.collect(0)``: the last test's short-lived
cycles, not a walk of the heap) and ``gc.freeze()``, so a collection inside the test walks only what the test made.
It never unfreezes between tests: what an earlier test left that is garbage stays frozen until the session's end,
where :func:`pytest_unconfigure` unfreezes it -- a test process's leftovers, bounded by the session.

**Unfrozen before the interpreter exits.** The interpreter's shutdown frees what reference cycles hold with the
collector, and the collector never walks frozen objects: a module kept by a frozen cycle outlives the shutdown, and
its cleanup never runs. DuckDB's does not: its module's cleanup capsule (``_duckdb._clean_default_connection``) is
never freed, so its import cache (a C++ static) is destroyed by ``exit()`` after the interpreter has gone, takes the
interpreter lock to drop its Python objects and aborts the process (``Fatal Python error: gilstate_tss_set``,
exit 134, Python 3.14.3, DuckDB 1.5.6; ``python -c "import duckdb, gc; gc.freeze()"`` alone does it). Every test
process that loaded DuckDB exited 134 that way, its tests passed. So the session's end unfreezes the heap.

**Cost.** Nothing outside a test session: the plugin is loaded only through ``pytest_plugins`` (or ``-p``), and wraps
asyncio's run from ``pytest_configure`` to ``pytest_unconfigure``. In a session: two clock reads and a thread-local
look a turn, and a watcher thread that wakes every :data:`_SAMPLE_EVERY_SECONDS` and samples only a turn already
running long.

Load it with ``pytest_plugins = ["threetears.core.testing.loop_stall"]`` in the rootdir conftest. It works the same
under xdist (each worker loads it and fails its own tests) and testament (a pytest run like any other).
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import sysconfig
import threading
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from types import FrameType
from typing import Any, Final

import pytest

from threetears.core.testing._asyncio_internals import timed_handles
from threetears.core.testing._interpreter_internals import thread_frames
from threetears.core.testing._thread_clock import own_cpu_clock
from threetears.core.testing.host_factor import HostFactor, Limit, limit_for

#: a pytest plugin: its hooks are its interface, loaded by name (``pytest_plugins``), never imported from
__all__: list[str] = []

#: the longest one turn of an event loop may hold it in a test, in the loop thread's CPU seconds on the reference host
#: (x the host's factor in a shared suite: :mod:`threetears.core.testing.host_factor`).
#:
#: Why 250 ms. The serving pods' own watchdog (the SDK's ``aibots_agents.runtime.loop_watchdog``) logs a held loop at
#: 50 ms, and a call that waits 0.3 s on a held loop is already slow; a test runs on a host up to ~3x slower a core
#: than the laptop it was written on (GitHub's 4-vCPU runner, a pool runner), so 5x the serving bound leaves room for
#: the host without letting any work that grows with rows through. Measured on the rollout (3tears' core unit and
#: loop_timing suites on the testament pool, 2026-11-25): 161 of 7,077 tests held a turn past it, 169 turns from
#: 0.26 s to 62.6 s (median 0.53 s); every one is synchronous work that grows with rows -- test setup building rows on
#: the loop (most), sqlite/DuckDB ``executemany`` on it, the registry applying a large change record, the copy
#: follower's keyed staging -- so the limit separates per-row work from a turn's bookkeeping.
_STALL_LIMIT_SECONDS: Final[float] = 0.25

#: how often the watcher looks at each loop thread, and how long a turn runs before its stack is sampled
_SAMPLE_EVERY_SECONDS: Final[float] = 0.02

#: the marker that raises one test's limit, with why
_MARKER: Final[str] = "loop_stall_limit"

#: the marker of a test that runs in an exclusive suite (alone on its runner): its limit takes no host factor. Every
#: ``exclusive = true`` suite of ``.testament.toml`` selects ``loop_timing`` tests, and only those, and CI runs them
#: serially (``scripts/ci-unit-tests.sh --loop-timing``); tests/enforcement/test_exclusive_suites_are_loop_timing.py
#: holds the manifest to it
_EXCLUSIVE_MARKER: Final[str] = "loop_timing"

#: the shortest ``why`` a raised limit takes: a sentence or an issue link, never a word
_WHY_MIN_CHARS: Final[int] = 20

#: the most frames a failure prints of the culprit's stack (innermost last)
_FRAMES_SHOWN: Final[int] = 30

#: the most samples one turn keeps (a 24 s turn sampled every 20 ms is 1,200); the newest are kept
_SAMPLES_KEPT: Final[int] = 200

#: where the standard library and installed packages live: a frame there is never the culprit
_NOT_OURS: Final[tuple[str, ...]] = tuple(
    {sysconfig.get_paths()["stdlib"], sysconfig.get_paths()["platstdlib"], sysconfig.get_paths()["purelib"]}
) + ("/site-packages/", "<frozen ")


@dataclass(frozen=True)
class _Frame:
    """one frame of a sampled stack."""

    module: str
    qualname: str
    filename: str
    lineno: int

    def ours(self, root: str) -> bool:
        """whether the frame is the project's own code (under the session's ``root``) or 3tears' (installed, in a
        consumer repo): never the standard library's, another package's, a test runner's harness (testament's
        report plugin, outside the root, runs in a collection's callbacks) or this guard's own."""
        if self.filename == __file__ or self.module.startswith("threetears.core.testing._"):
            return False
        if self.module.startswith("threetears."):
            return True
        return self.filename.startswith(root) and not any(part in self.filename for part in _NOT_OURS)

    def named(self) -> str:
        """the frame as a failure names it: ``module.qualname (file:line)``."""
        return f"{self.module}.{self.qualname} ({self.filename}:{self.lineno})"


@dataclass
class _Turn:
    """the turn a loop thread is running: its number, when it started, the stacks sampled while it ran (each with the
    thread's CPU seconds since the watcher's look before it, None where the platform reads no other thread's clock),
    and the thread's CPU at the watcher's last look."""

    thread_id: int
    thread_name: str
    seq: int = 0
    started: float = 0.0
    samples: list[tuple[int, float | None, tuple[_Frame, ...]]] = field(default_factory=list)
    #: the thread's CPU clock, taken on the thread itself (None where the platform has none)
    cpu_clock: Callable[[], float | None] | None = None
    cpu_seen: float | None = None


@dataclass(frozen=True)
class _Stall:
    """one turn past the limit: how long it held the loop, when it ended, and what held it."""

    seconds: float
    ended: float
    thread_name: str
    handle: str
    culprit: str
    sampled: int
    #: the loop thread's CPU seconds the culprit's samples carry (a count of samples where no CPU was read)
    held_by_culprit: float
    #: the CPU seconds all the turn's samples carry; None where the platform reads no other thread's CPU
    sampled_cpu: float | None
    stack: tuple[_Frame, ...]
    #: the other functions the samples held, most first, with their CPU seconds (or counts)
    also: tuple[tuple[str, float], ...] = ()

    def told(self, limit: Limit) -> str:
        """the stall as a failure says it."""
        lines = [
            f"the event loop was held {self.seconds:.3f} s (loop thread CPU) in one turn on thread "
            f"{self.thread_name!r}; the limit is {limit.told()}",
            f"  held by: {self.culprit}  ({self._share()})",
        ]
        if self.also:
            lines.append("  also held by: " + ", ".join(f"{name} ({self._amount(held)})" for name, held in self.also))
        if self.stack:
            lines.append("  stack (innermost last):")
            lines.extend(f"    {frame.named()}" for frame in self.stack[-_FRAMES_SHOWN:])
        return "\n".join(lines)

    def _amount(self, held: float) -> str:
        return f"{held:.3f} s CPU" if self.sampled_cpu is not None else f"{held:.0f} samples"

    def _share(self) -> str:
        if not self.sampled:
            return "no sample: the callback"
        if self.sampled_cpu is None:
            return f"{self.held_by_culprit:.0f} of {self.sampled} samples"
        return f"{self.held_by_culprit:.3f} s of the {self.sampled_cpu:.3f} s CPU its {self.sampled} samples saw"


def _stack_of(frame: FrameType | None) -> tuple[_Frame, ...]:
    """``frame``'s stack, outermost first, from the turn's own first frame: the frames that ran the loop (pytest,
    the runner, ``_run_once``, ``Handle._run``) are cut."""
    frames: list[_Frame] = []
    while frame is not None:
        code = frame.f_code
        if code.co_qualname == "Handle._run" and frame.f_globals.get("__name__") == "asyncio.events":
            break
        frames.append(
            _Frame(
                module=str(frame.f_globals.get("__name__", "?")),
                qualname=code.co_qualname,
                filename=code.co_filename,
                lineno=frame.f_lineno or code.co_firstlineno,
            )
        )
        frame = frame.f_back
    frames.reverse()
    return tuple(frames)


def _culprit(stack: tuple[_Frame, ...], root: str) -> _Frame | None:
    """the innermost frame of the project's own code (under ``root``) in ``stack``."""
    for frame in reversed(stack):
        if frame.ours(root):
            return frame
    return None


class _Guard:
    """the session's guard: asyncio's run wrapped, the loop threads' turns, the watcher, and the stalls seen."""

    def __init__(self, root: str) -> None:
        #: the session's root directory: the project's own code is under it
        self._root = root
        self._local = threading.local()
        #: every thread that has run a loop turn this session (appended to, never removed: a handful a session)
        self._turns: list[_Turn] = []
        #: the stalls since the last test phase took them
        self._stalls: list[_Stall] = []
        self._stop = threading.Event()
        self._watcher = threading.Thread(target=self._watch, name="loop-stall-watcher", daemon=True)
        self._wrapped = contextlib.ExitStack()
        #: whether loop threads' CPU can be read from the watcher (else samples are counted)
        self._cpu_readable = own_cpu_clock() is not None
        #: this host's slowdown against the reference host, for the limit in a shared suite
        self.host = HostFactor()

    def start(self) -> None:
        """wrap asyncio's run and start the watcher."""
        self._wrapped.enter_context(timed_handles(self._ended, starting=self._starting))
        self._watcher.start()

    def stop(self) -> None:
        """put asyncio's run back, stop the watcher and unfreeze what the session froze."""
        self._wrapped.close()
        self._stop.set()
        self._watcher.join(timeout=1.0)

    def _turn(self) -> _Turn:
        """the calling loop thread's turn record, made on its first turn."""
        turn: _Turn | None = getattr(self._local, "turn", None)
        if turn is None:
            current = threading.current_thread()
            ident = threading.get_ident()
            # the thread's clock, taken here on the thread (_thread_clock: never from the watcher by ident), and its CPU
            # now, so the watcher's first sample of it has a look before to count from
            clock = own_cpu_clock()
            cpu = clock() if clock is not None else None
            turn = _Turn(thread_id=ident, thread_name=current.name, cpu_clock=clock, cpu_seen=cpu)
            self._local.turn = turn
            self._turns.append(turn)
        return turn

    def _starting(self) -> None:
        """a turn starts on this thread: number it and stamp it, for the watcher."""
        turn = self._turn()
        turn.seq += 1
        turn.started = time.perf_counter()

    def _ended(self, seconds: float, ended: asyncio.events.Handle) -> None:
        """a turn ended on this thread, ``seconds`` of its CPU: a stall when past the limit (only then is ``ended``, its
        handle, made text)."""
        turn = self._turn()
        seq = turn.seq
        turn.started = 0.0
        samples = turn.samples
        if samples:
            turn.samples = []
        if seconds <= _STALL_LIMIT_SECONDS:
            return
        handle = repr(ended)[:120]
        mine = [(weight, stack) for at, weight, stack in samples if at == seq]
        # a sample is worth the loop thread's CPU since the watcher's look before it, never the wall time it stood
        # for: a wait (``future.result()`` on the loop while a worker burns) spends none and is never the culprit.
        # Where no other thread's CPU can be read, each sample counts one, as wall time; a sample whose read failed (None
        # where there is a reader) counts none
        by_cpu = self._cpu_readable and any(weight is not None for weight, _ in mine)
        named = [
            ((weight or 0.0) if by_cpu else 1.0, culprit, stack)
            for weight, stack in mine
            if (culprit := _culprit(stack, self._root)) is not None
        ]
        # function-local: built and read in this call, never shared
        held_by: dict[str, float] = {}
        for weight, culprit, _ in named:
            name = culprit.module + "." + culprit.qualname
            held_by[name] = held_by.get(name, 0.0) + weight
        ranked = sorted(((name, held) for name, held in held_by.items() if held > 0), key=lambda row: -row[1])
        held = 0.0
        if ranked:
            top, held = ranked[0]
            frame, stack = next(
                (culprit, stack)
                for weight, culprit, stack in reversed(named)
                if culprit.module + "." + culprit.qualname == top and weight > 0
            )
            culprit_text = frame.named()
            also = tuple(ranked[1:4])
        else:
            stack, culprit_text, also = mine[-1][1] if mine else (), f"the callback {handle}", ()
        self._stalls.append(
            _Stall(
                seconds=seconds,
                ended=time.time(),
                thread_name=turn.thread_name,
                handle=handle,
                culprit=culprit_text,
                sampled=len(mine),
                held_by_culprit=held,
                sampled_cpu=sum(weight or 0.0 for weight, _ in mine) if by_cpu else None,
                stack=stack,
                also=also,
            )
        )

    def _watch(self) -> None:
        """sample the stack of every loop thread whose turn has run longer than :data:`_SAMPLE_EVERY_SECONDS`, with
        the CPU the thread spent since the watcher's look before."""
        while not self._stop.wait(_SAMPLE_EVERY_SECONDS):
            now = time.perf_counter()
            frames: dict[int, FrameType] | None = None
            for turn in list(self._turns):
                seq, started = turn.seq, turn.started
                clock = turn.cpu_clock
                cpu = clock() if clock is not None else None
                seen, turn.cpu_seen = turn.cpu_seen, cpu
                if not started or now - started < _SAMPLE_EVERY_SECONDS:
                    continue
                spent = cpu - seen if cpu is not None and seen is not None else None
                if frames is None:
                    frames = thread_frames()
                stack = _stack_of(frames.get(turn.thread_id))
                if turn.seq != seq or not turn.started:
                    continue  # the turn ended while the stack was read: the stack may be the next one's
                samples = turn.samples
                samples.append((seq, spent, stack))
                if len(samples) > _SAMPLES_KEPT:
                    del samples[: len(samples) - _SAMPLES_KEPT]
            frames = None

    def taken(self, until: float) -> list[_Stall]:
        """the stalls that ended by ``until``, taken off the guard."""
        taken = [stall for stall in self._stalls if stall.ended <= until]
        if taken:
            self._stalls = [stall for stall in self._stalls if stall.ended > until]
        return taken


_GUARD_KEY = pytest.StashKey[_Guard]()


def _reference_limit_of(item: pytest.Item) -> float:
    """the test's limit on the reference host: :data:`_STALL_LIMIT_SECONDS`, or the higher one its marker states."""
    marker = item.get_closest_marker(_MARKER)
    return _STALL_LIMIT_SECONDS if marker is None else float(marker.kwargs["seconds"])


def _exclusive(item: pytest.Item) -> bool:
    """whether the test runs in an exclusive suite (alone on its runner): :data:`_EXCLUSIVE_MARKER`."""
    return item.get_closest_marker(_EXCLUSIVE_MARKER) is not None


def _limit_of(item: pytest.Item, guard: _Guard) -> Limit:
    """the test's limit on this host: its reference limit, x the host's factor in a shared suite, as is in an
    exclusive one."""
    return limit_for(_reference_limit_of(item), None if _exclusive(item) else guard.host.reading())


def _marker_fault(marker: pytest.Mark) -> str | None:
    """what is wrong with a ``loop_stall_limit`` marker, or None."""
    if marker.args or set(marker.kwargs) != {"seconds", "why"}:
        return "takes exactly seconds=<float> and why=<str>, as keywords"
    seconds, why = marker.kwargs["seconds"], marker.kwargs["why"]
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or seconds <= _STALL_LIMIT_SECONDS:
        return f"seconds must be a number above the limit, {_STALL_LIMIT_SECONDS} s (it only raises the limit)"
    if not isinstance(why, str) or len(why.strip()) < _WHY_MIN_CHARS:
        return f"why must say why the turn must run this long (an issue, a reason): {_WHY_MIN_CHARS}+ characters"
    return None


def pytest_configure(config: pytest.Config) -> None:
    """declare the marker, wrap asyncio's run and start the watcher for the session.

    :param config: the session's config
    :ptype config: pytest.Config
    :return: nothing
    :rtype: None
    """
    config.addinivalue_line(
        "markers",
        f"{_MARKER}(seconds, why): raise this test's event-loop stall limit above {_STALL_LIMIT_SECONDS} s (reference "
        "seconds, scaled by the host's factor in a shared suite), saying why (threetears.core.testing.loop_stall)",
    )
    guard = _Guard(str(config.rootpath))
    guard.start()
    config.stash[_GUARD_KEY] = guard


def pytest_unconfigure(config: pytest.Config) -> None:
    """put asyncio's run back, stop the watcher and unfreeze what the session froze.

    :param config: the session's config
    :ptype config: pytest.Config
    :return: nothing
    :rtype: None
    """
    guard = config.stash.get(_GUARD_KEY, None)
    if guard is not None:
        guard.stop()
        del config.stash[_GUARD_KEY]
        # what pytest_runtest_setup froze goes back to the collector, so the interpreter's shutdown frees it and every
        # module's cleanup runs while the interpreter is still there (the module's "Unfrozen before the interpreter
        # exits": a module kept by a frozen cycle aborted the process at exit, 134)
        gc.unfreeze()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """refuse the run when a ``loop_stall_limit`` marker does not raise the limit or does not say why.

    :param config: the session's config
    :ptype config: pytest.Config
    :param items: the collected tests
    :ptype items: list[pytest.Item]
    :return: nothing
    :rtype: None
    :raises pytest.UsageError: naming each test whose marker is wrong, and what is wrong with it
    """
    faults = [
        f"{item.nodeid}: @pytest.mark.{_MARKER} {fault}"
        for item in items
        for marker in item.iter_markers(_MARKER)
        if (fault := _marker_fault(marker)) is not None
    ]
    if faults:
        raise pytest.UsageError("loop-stall limits that cannot stand:\n  " + "\n  ".join(faults))


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    """freeze what the process holds before the test, so a garbage collection in it walks only what the test makes.

    Why: the collector's automatic full collection walks every tracked object, and in a test process that is every
    earlier test's leftovers -- ~1 s turns charged to trivial frames on the pool (0.10 s at ~300k tracked objects, ~1 s
    on bigger heaps; 2026-11-25). A pod freezes its heap once after the SDK's ``settle_before_serving``; a test is
    held to the pod's terms. ``gc.collect(0)`` first lets the last test's young cycles go rather than freezing them;
    it walks the youngest generation only. Never unfrozen between tests: :func:`pytest_unconfigure` unfreezes it at the
    session's end, so the interpreter's shutdown frees it. A test that calls ``gc.unfreeze()`` itself undoes this for the rest of its own run.

    :param item: the test about to set up
    :ptype item: pytest.Item
    :return: nothing
    :rtype: None
    """
    guard = item.config.stash.get(_GUARD_KEY, None)
    if guard is None:
        return
    if not _exclusive(item):
        # between tests, on the loop thread, before the freeze (the probe's few objects go with the young collection)
        guard.host.sample_if_due()
    gc.collect(0)
    gc.freeze()


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]) -> Generator[None, Any, Any]:
    """fail the phase (setup, call or teardown) in which a loop turn ran past the test's limit, naming what held it.

    :param item: the test
    :ptype item: pytest.Item
    :param call: the phase
    :ptype call: pytest.CallInfo[None]
    :return: the phase's report
    :rtype: Generator[None, Any, Any]
    """
    report: pytest.TestReport = yield
    guard = item.config.stash.get(_GUARD_KEY, None)
    if guard is None:
        return report
    taken = [stall for stall in guard.taken(call.stop) if stall.ended >= call.start]
    if not taken:
        return report
    limit = _limit_of(item, guard)
    stalls = [stall for stall in taken if stall.seconds > limit.seconds]
    if not stalls:
        return report
    if not limit.judgeable:
        return _could_not_judge(report, call.when, stalls, limit)
    told = "\n\n".join(stall.told(limit) for stall in stalls)
    told += (
        "\n\nMove the work off the loop (asyncio.to_thread for IO and work that lets the interpreter lock go, a worker "
        "process for CPU), or the synchronous setup out of it (a sync fixture). Only a stated reason raises one "
        f"test's limit: @pytest.mark.{_MARKER}(seconds=..., why=...) (threetears.core.testing.loop_stall)."
    )
    if report.passed:
        report.outcome = "failed"
        report.longrepr = f"event loop stalled in {call.when}:\n{told}"
    else:
        report.sections.append((f"event loop stalled in {call.when}", told))
    return report


def _could_not_judge(report: pytest.TestReport, when: str, stalls: list[_Stall], limit: Limit) -> pytest.TestReport:
    """fail the phase as the timing suite fails a measurement on a host too busy for it (``CouldNotMeasure``, "could
    not measure: ..."): the host, not the code, is named. Never a pass, never a skip, never a stall red.

    :param report: the phase's report
    :ptype report: pytest.TestReport
    :param when: the phase
    :ptype when: str
    :param stalls: the turns past the (capped) limit
    :ptype stalls: list[_Stall]
    :param limit: the limit, past its host factor's cap
    :ptype limit: Limit
    :return: the report
    :rtype: pytest.TestReport
    """
    reading = limit.reading
    assert reading is not None  # an exclusive suite's limit is always judgeable
    told = (
        f"could not measure: this host spends {reading.factor:.2f}x the reference host's CPU on the same work, past "
        f"the {reading.cap:.1f}x cap, so a turn's CPU says nothing about the code. Not a pass: rerun it on a quieter "
        "runner.\n\n" + "\n\n".join(stall.told(limit) for stall in stalls)
    )
    if report.passed:
        report.outcome = "failed"
        report.longrepr = f"event loop stall could not be judged in {when}:\n{told}"
    else:
        report.sections.append((f"event loop stall could not be judged in {when}", told))
    return report
