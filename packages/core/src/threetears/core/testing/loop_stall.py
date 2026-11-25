"""pytest plugin: a test fails when one turn of an event loop held the loop past :data:`_STALL_LIMIT_SECONDS`.

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

**How the culprit is named.** A watcher thread looks at each loop thread every :data:`_SAMPLE_EVERY_SECONDS`; a turn
running longer than that has its thread's stack sampled (``sys._current_frames``, through
:mod:`~threetears.core.testing._interpreter_internals`) while it runs. When the turn ends
past the limit, the failure names the innermost frame of the project's own code (not the standard library or an
installed package) that most samples hold -- ``threetears.core.arrow_keys.distinct_keys``, not numpy's ``unique``
under it -- and prints the stack of the last sample that held it. A C call that keeps the interpreter lock from start
to end (a pyarrow kernel) lets the watcher run only when it returns, and the sample then shows the Python frame that
called it: still the culprit. A turn no sample saw names the callback asyncio ran.

**The limit** is :data:`_STALL_LIMIT_SECONDS`, one setting for every test of every repo that loads the plugin. A test
whose turn must run longer is a bug in the code under test (move the work off the loop: ``asyncio.to_thread`` for IO
and work that lets the lock go, a worker process for CPU, data through :mod:`threetears.core.shared_arrow`) or in the
test (synchronous setup belongs outside the loop, in a sync fixture). There is no exemption list. A test may raise its
own limit only with ``@pytest.mark.loop_stall_limit(seconds=..., why="...")``: ``seconds`` above the limit and a
``why`` that says why (an issue, a reason), checked at collection -- a marker that does not is a usage error.

**Cost.** Nothing outside a test session: the plugin is loaded only through ``pytest_plugins`` (or ``-p``), and wraps
asyncio's run from ``pytest_configure`` to ``pytest_unconfigure``. In a session: two clock reads and a thread-local
look a turn, and a watcher thread that wakes every :data:`_SAMPLE_EVERY_SECONDS` and samples only a turn already
running long.

Load it with ``pytest_plugins = ["threetears.core.testing.loop_stall"]`` in the rootdir conftest. It works the same
under xdist (each worker loads it and fails its own tests) and testament (a pytest run like any other).
"""

from __future__ import annotations

import contextlib
import sysconfig
import threading
import time
from collections import Counter
from collections.abc import Generator
from dataclasses import dataclass, field
from types import FrameType
from typing import Any, Final

import pytest

from threetears.core.testing._asyncio_internals import timed_handles
from threetears.core.testing._interpreter_internals import thread_frames

#: a pytest plugin: its hooks are its interface, loaded by name (``pytest_plugins``), never imported from
__all__: list[str] = []

#: the longest one turn of an event loop may hold it in a test, in the loop thread's CPU seconds.
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
    """the turn a loop thread is running: its number, when it started, and the stacks sampled while it ran."""

    thread_id: int
    thread_name: str
    seq: int = 0
    started: float = 0.0
    samples: list[tuple[int, tuple[_Frame, ...]]] = field(default_factory=list)


@dataclass(frozen=True)
class _Stall:
    """one turn past the limit: how long it held the loop, when it ended, and what held it."""

    seconds: float
    ended: float
    thread_name: str
    handle: str
    culprit: str
    sampled: int
    held_by_culprit: int
    stack: tuple[_Frame, ...]
    #: the other functions the samples held, most first, with their counts
    also: tuple[tuple[str, int], ...] = ()

    def told(self, limit: float) -> str:
        """the stall as a failure says it."""
        lines = [
            f"the event loop was held {self.seconds:.3f} s (loop thread CPU) in one turn on thread "
            f"{self.thread_name!r}; the limit is {limit:.3f} s",
            f"  held by: {self.culprit}"
            + (
                f"  ({self.held_by_culprit} of {self.sampled} samples)"
                if self.sampled
                else "  (no sample: the callback)"
            ),
        ]
        if self.also:
            lines.append("  also held by: " + ", ".join(f"{name} ({count})" for name, count in self.also))
        if self.stack:
            lines.append("  stack (innermost last):")
            lines.extend(f"    {frame.named()}" for frame in self.stack[-_FRAMES_SHOWN:])
        return "\n".join(lines)


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

    def start(self) -> None:
        """wrap asyncio's run and start the watcher."""
        self._wrapped.enter_context(timed_handles(self._ended, starting=self._starting))
        self._watcher.start()

    def stop(self) -> None:
        """put asyncio's run back and stop the watcher."""
        self._wrapped.close()
        self._stop.set()
        self._watcher.join(timeout=1.0)

    def _turn(self) -> _Turn:
        """the calling loop thread's turn record, made on its first turn."""
        turn: _Turn | None = getattr(self._local, "turn", None)
        if turn is None:
            current = threading.current_thread()
            turn = _Turn(thread_id=threading.get_ident(), thread_name=current.name)
            self._local.turn = turn
            self._turns.append(turn)
        return turn

    def _starting(self) -> None:
        """a turn starts on this thread: number it and stamp it, for the watcher."""
        turn = self._turn()
        turn.seq += 1
        turn.started = time.perf_counter()

    def _ended(self, seconds: float, handle: str) -> None:
        """a turn ended on this thread, ``seconds`` of its CPU: a stall when past the limit."""
        turn = self._turn()
        seq = turn.seq
        turn.started = 0.0
        samples = turn.samples
        if samples:
            turn.samples = []
        if seconds <= _STALL_LIMIT_SECONDS:
            return
        mine = [stack for at, stack in samples if at == seq]
        named = [(culprit, stack) for stack in mine if (culprit := _culprit(stack, self._root)) is not None]
        if named:
            counts = Counter(culprit.module + "." + culprit.qualname for culprit, _ in named)
            top, held = counts.most_common(1)[0]
            frame, stack = next(
                (culprit, stack) for culprit, stack in reversed(named) if culprit.module + "." + culprit.qualname == top
            )
            culprit_text = frame.named()
            also = tuple(counts.most_common(4)[1:])
        else:
            held, stack, culprit_text, also = 0, mine[-1] if mine else (), f"the callback {handle}", ()
        self._stalls.append(
            _Stall(
                seconds=seconds,
                ended=time.time(),
                thread_name=turn.thread_name,
                handle=handle,
                culprit=culprit_text,
                sampled=len(mine),
                held_by_culprit=held,
                stack=stack,
                also=also,
            )
        )

    def _watch(self) -> None:
        """sample the stack of every loop thread whose turn has run longer than :data:`_SAMPLE_EVERY_SECONDS`."""
        while not self._stop.wait(_SAMPLE_EVERY_SECONDS):
            now = time.perf_counter()
            frames: dict[int, FrameType] | None = None
            for turn in list(self._turns):
                seq, started = turn.seq, turn.started
                if not started or now - started < _SAMPLE_EVERY_SECONDS:
                    continue
                if frames is None:
                    frames = thread_frames()
                stack = _stack_of(frames.get(turn.thread_id))
                if turn.seq != seq or not turn.started:
                    continue  # the turn ended while the stack was read: the stack may be the next one's
                samples = turn.samples
                samples.append((seq, stack))
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


def _limit_of(item: pytest.Item) -> float:
    """the test's limit: :data:`_STALL_LIMIT_SECONDS`, or the higher one its marker states."""
    marker = item.get_closest_marker(_MARKER)
    return _STALL_LIMIT_SECONDS if marker is None else float(marker.kwargs["seconds"])


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
        f"{_MARKER}(seconds, why): raise this test's event-loop stall limit above {_STALL_LIMIT_SECONDS} s, saying why "
        "(threetears.core.testing.loop_stall)",
    )
    guard = _Guard(str(config.rootpath))
    guard.start()
    config.stash[_GUARD_KEY] = guard


def pytest_unconfigure(config: pytest.Config) -> None:
    """put asyncio's run back and stop the watcher.

    :param config: the session's config
    :ptype config: pytest.Config
    :return: nothing
    :rtype: None
    """
    guard = config.stash.get(_GUARD_KEY, None)
    if guard is not None:
        guard.stop()
        del config.stash[_GUARD_KEY]


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
    limit = _limit_of(item)
    stalls = [stall for stall in guard.taken(call.stop) if stall.ended >= call.start and stall.seconds > limit]
    if not stalls:
        return report
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
