"""say when a process's event loop stops running, and what held it.

a pod answers every call on one event loop. anything that holds that loop -- synchronous work on
it, the cycle collector, or C code that keeps the interpreter lock while the loop's thread waits
for it -- leaves every call unanswered: on cobalt dev (2026-10-23) the hub could not hand a reports
tool pod a call within 10 s while one report page computed, and nothing in the pod's log said why.
this makes such a stall visible, with its cause, in every pod.

**three witnesses, one line per stall.**

- *the heartbeat*, a timer on the loop every :data:`BEAT_SECONDS`, stamps the time and measures
  how late it ran. a beat later than the bound is a stall, whoever held the loop: it is recorded
  for the watchdog thread to log, so a stall that held the interpreter lock from start to end (the
  collector, C code) is logged as surely as one that did not.
- *the held-loop dump*: every beat sets a wall-clock alarm aimed at the loop thread one bound
  ahead, and :func:`faulthandler.register` answers ``SIGALRM`` in C, with no interpreter lock: so
  when the loop stays silent past the bound, the loop thread's stack is written *while the stall is
  happening*, whoever holds the lock. the next beat would have pushed the alarm back, so an alarm
  goes off only in a stall. it writes to a file in memory (``memfd`` on Linux; an unlinked
  temporary file elsewhere), read back and emptied by the heartbeat when the late beat lands; the
  read is bounded (:data:`DUMP_BYTES`) and so is what the log line keeps (:data:`FRAMES` frames a
  thread, :data:`STACK_CHARS` in all). re-arming is one system call (~1 us); re-arming
  :func:`faulthandler.dump_traceback_later` instead starts a thread each time, ~50x the cost.
  the alarm is one per process: the first watchdog started owns it, any other runs without the
  dump, and so does a watchdog started where something else already handles ``SIGALRM``; when
  something takes ``SIGALRM`` over later, the watchdog stops arming it at its next look. the
  handler restarts interrupted system calls (``SA_RESTART``), and it only ever fires in a stall.

  **it dumps the loop thread alone, never every thread.** ``faulthandler``'s all-threads dump walks
  the interpreter's thread list from a signal handler, with no lock: a thread that exits meanwhile
  frees its thread state under the walk, and the walk reads freed memory. the reports tool pod died
  of exactly that in a local burst replay (2026-11-21, CPython 3.14.3: SIGSEGV in
  ``tstate_is_freed`` under ``_Py_DumpTracebackThreads``) -- a pod's threads come and go all the
  time (``asyncio.to_thread``, NATS, arrow, DuckDB), and the alarm fires only in a stall, so under
  load. CPython calls that walk unsafe by design (gh-116008, gh-140815; its fix, gh-140921, detects
  only *some* freed memory). the one-thread dump reads only the interrupted thread's own state,
  which cannot be freed while that thread sits in the handler; so the alarm must interrupt the loop
  thread. a process-wide alarm (``setitimer``) goes to whichever thread the kernel picks: on Linux
  1 in 7 went to another thread under thread churn (measured, 2026-11-21), so on Linux the alarm is
  a timer of the loop thread's own (``timer_create`` with ``SIGEV_THREAD_ID``, through ``ctypes``;
  ~1 us a re-arm, as ``setitimer``). macOS has no such timer, and delivers a process-wide signal to
  the first thread that does not block it, the main thread (every one of ~2,000 under the same
  churn): there the alarm is ``setitimer``, and the dump is taken only when the loop runs on the
  main thread, as in every pod. a stall another thread caused by holding the lock shows in the
  watchdog thread's stacks once the lock is let go.
- *the collector's clock*: a :data:`gc.callbacks` entry times every collection (generation,
  duration, objects collected and uncollectable). a collection longer than the collector bound is
  logged on its own, and each stall's line names the collections that overlapped it, longest
  first, and says ``cause: "gc gen N, X ms"`` when they cover half the stall or more.

a watchdog *thread* -- a thread, so it runs while the loop does not -- reads the heartbeat's stamp
every beat. when the loop has been silent past :data:`STALL_SECONDS` *and the thread can run* (the
loop's holder lets go of the interpreter lock: a sleep, a blocking read, Python code that yields
the lock), it logs a warning while the stall goes on, with the loop thread's stack at that moment
(``sys._current_frames``) and every other live thread's, named and bounded. a long stall is logged
again every :data:`REPEAT_SECONDS` with the stacks then, and its end once, with its whole length,
the collections in it and the held-loop dump. a stall the thread never saw going on (the lock held
throughout) is logged once, when it ends, with the same: its length, the collections, the dump.
the thread can only look once the lock is let go, so when it was held, the stacks a line logged
while the stall "goes on" carries are of whatever ran next; the end line's ``cause``, ``gc`` and
``held_stacks`` say what held it.

**the bound is 50 ms**: 0.3 s of a held loop is a very long time for a pod whose every call waits
on it. **its log is bounded:** at most :data:`LOG_BUDGET` stalls are logged in any
:data:`LOG_WINDOW_SECONDS`; past that each is counted (``unlogged``), and the next one logged says
how many went unlogged before it, so a pod that stalls often cannot flood its log. a stall that
lasts past :data:`REPEAT_SECONDS` is logged then, whatever the budget: a long one is never only
counted. slow collections have a budget of their own, the same size. every stall is counted
whether logged or not, and the longest silence seen is kept.

**the cost** (python 3.14, 2026-11-07): the dump's alarm and the collector's clock add about 1 us a
beat and a look, 0.01% of one CPU, to the heartbeat and the thread's own (0.2-0.5% of one CPU idle
on macOS, 1.2-2% in a Linux VM: a 100 Hz timer and thread wake). a stall costs one dump, in C.

the thresholds are the constructor's; a consumer reads them from its own settings (the 14th AI bots SDK's
``LoopWatchdogSettings``, ``FOURTEENAIBOTS_LOOP_WATCHDOG_*``).

the watchdog must not die quietly: a failure in one look (formatting a frame, the logger) is logged
at error through ``contain`` and counted, and the thread looks again. its health
(:meth:`LoopWatchdog.health`) is the process's to serve (the SDK's serving bootstrap exposes its watchdog
as ``loop_watchdog`` while it serves).

the watchdog came from the 14th AI bots SDK (``aibots_agents.runtime.loop_watchdog``), where every pod runs it;
it lives here so a process that does not run on the SDK (the hub) runs the same one. the SDK keeps its settings and
its serving wrappers (``watched_loop``, ``guarded_serving``) and imports the watchdog from here.

this module is imported by name, not re-exported from :mod:`threetears.observe`: importing it registers an exit
hook that turns the process's wall-clock alarm off, and a fork hook, which a process that never watches its loop
should not carry.
"""

from __future__ import annotations

import asyncio
import atexit
import ctypes
import faulthandler
import gc
import itertools
import math
import os
import signal
import sys
import tempfile
import threading
import time
import traceback
from collections.abc import Callable
from typing import Any, Final, NamedTuple

from threetears.observe._cpython_internals import current_frames
from threetears.observe.containment import contain
from threetears.observe.logging import get_logger

__all__ = [
    "BEAT_SECONDS",
    "DUMP_BYTES",
    "FRAMES",
    "GC_KEPT",
    "LATE_KEPT",
    "GC_NAMED",
    "GC_SLOW_SECONDS",
    "LOG_BUDGET",
    "LOG_WINDOW_SECONDS",
    "REPEAT_SECONDS",
    "STACK_CHARS",
    "STALL_SECONDS",
    "LoopWatchdog",
]

log = get_logger(__name__)

#: how often the loop stamps its heartbeat, and the watchdog looks: a fifth of the bound, so a silence is seen
#: within a beat of passing it
BEAT_SECONDS: Final = 0.01

#: the longest the loop may go without a beat: past it the stall is counted and logged
STALL_SECONDS: Final = 0.05

#: a collection longer than this is logged on its own
GC_SLOW_SECONDS: Final = 0.025

#: a stall still going is logged again this often, with the stack then
REPEAT_SECONDS: Final = 10.0

#: the most stalls logged in any :data:`LOG_WINDOW_SECONDS`; the rest are counted, not logged. slow collections
#: have a budget of the same size
LOG_BUDGET: Final = 20

#: the window :data:`LOG_BUDGET` is counted over
LOG_WINDOW_SECONDS: Final = 60.0

#: the most frames of each thread's stack a warning carries, innermost last
FRAMES: Final = 25

#: the most characters of threads' stacks a warning carries, per kind of stack; the loop thread's own is apart
STACK_CHARS: Final = 16_000

#: the most bytes of a held-loop dump read back
DUMP_BYTES: Final = 256 * 1024

#: the collections kept to match against stalls, newest last
GC_KEPT: Final = 512

#: the late beats kept for the watchdog thread, newest last. the heartbeat records one per beat at most and the
#: thread reads them every beat, so it falls this far behind only when it is itself held for this many beats
LATE_KEPT: Final = 64

#: the most collections a stall's line names, longest first
GC_NAMED: Final = 3

#: the header of a ``faulthandler`` dump of the interrupted thread alone
_DUMP_HEADER: Final = "Stack (most recent call first):"

#: whoever owns the process's one wall-clock alarm
_DUMP_OWNER = threading.Lock()


#: the loop thread's own timer while one is armed (Linux), for :func:`_disarm`: a tuple replaced whole by its one
#: owner, the holder of :data:`_DUMP_OWNER`
_THREAD_TIMER: tuple[_ThreadTimer, ...] = ()


def _disarm() -> None:
    """the alarm off: at exit, before ``faulthandler`` gives ``SIGALRM`` back to its default, which ends the
    process."""
    signal.setitimer(signal.ITIMER_REAL, 0)
    for timer in _THREAD_TIMER:
        timer.set(0)


def _alarm_is_free() -> bool:
    """whether nothing in Python handles ``SIGALRM``: ``faulthandler``'s handler is below Python's view of it."""
    return signal.getsignal(signal.SIGALRM) == signal.SIG_DFL and signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


class _Timespec(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]


class _Itimerspec(ctypes.Structure):
    _fields_ = [("it_interval", _Timespec), ("it_value", _Timespec)]


class _Sigevent(ctypes.Structure):
    """Linux's ``struct sigevent`` (64 bytes; glibc and musl alike), for ``SIGEV_THREAD_ID``: the thread id sits
    right after ``sigev_notify``."""

    _fields_ = [
        ("sigev_value", ctypes.c_void_p),
        ("sigev_signo", ctypes.c_int),
        ("sigev_notify", ctypes.c_int),
        ("sigev_notify_thread_id", ctypes.c_int),
        ("_pad", ctypes.c_int * 11),
    ]


#: ``<signal.h>``: the signal goes to the thread ``sigev_notify_thread_id`` names
_SIGEV_THREAD_ID: Final = 4

#: ``<time.h>``
_CLOCK_MONOTONIC: Final = 1


class _ThreadTimer:
    """a one-shot ``SIGALRM`` timer aimed at the calling thread alone (Linux: ``timer_create`` with
    ``SIGEV_THREAD_ID``), so the one-thread dump is that thread's. called through ``PyDLL``, which keeps the
    interpreter lock: a re-arm is one system call, never a wait for the lock.

    :raises OSError: when the system has no such timer or refuses one
    """

    def __init__(self) -> None:
        libc = ctypes.PyDLL(None, use_errno=True)
        try:
            self._settime = libc.timer_settime
            self._delete = libc.timer_delete
            create = libc.timer_create
        except AttributeError as missing:
            raise OSError(f"no per-thread timer here: {missing}") from missing
        self._settime.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        self._delete.argtypes = [ctypes.c_void_p]
        self._timer = ctypes.c_void_p()
        event = _Sigevent(
            sigev_signo=signal.SIGALRM, sigev_notify=_SIGEV_THREAD_ID, sigev_notify_thread_id=threading.get_native_id()
        )
        if create(_CLOCK_MONOTONIC, ctypes.byref(event), ctypes.byref(self._timer)):
            raise OSError(ctypes.get_errno(), "timer_create refused the loop thread's timer")
        self._spec = _Itimerspec()

    def set(self, seconds: float) -> None:
        """go off once ``seconds`` from now; 0 turns it off."""
        whole = int(seconds)
        self._spec.it_value.tv_sec = whole
        self._spec.it_value.tv_nsec = int((seconds - whole) * 1e9)
        if self._settime(self._timer, 0, ctypes.byref(self._spec), None):
            raise OSError(ctypes.get_errno(), "timer_settime refused the loop thread's timer")

    def delete(self) -> None:
        """the timer gone; it never goes off again."""
        self._delete(self._timer)


class _Collection(NamedTuple):
    """one collection the collector ran, on the watchdog's clock."""

    seq: int
    generation: int
    start: float
    end: float
    collected: int
    uncollectable: int

    def said(self) -> dict[str, Any]:
        """the collection as a log line carries it."""
        return {
            "generation": self.generation,
            "ms": round((self.end - self.start) * 1000, 1),
            "collected": self.collected,
            "uncollectable": self.uncollectable,
        }


class _LateBeat(NamedTuple):
    """a beat that ran past the bound: the loop was held from ``start`` to ``end``; ``dump`` is the held-loop
    dump taken meanwhile, or empty."""

    start: float
    end: float
    dump: str


def _dump_file() -> int:
    """a file descriptor for the held-loop dump, in memory where the system allows.

    :return: an open descriptor, the file already unlinked
    :rtype: int
    """
    if sys.platform == "linux":
        return os.memfd_create("loop-watchdog-stacks", os.MFD_CLOEXEC)
    with tempfile.TemporaryFile() as file:
        return os.dup(file.fileno())


class _HeldStacks:
    """the loop thread's stack, written by ``faulthandler`` in C when the loop's alarm goes off, read back on the loop.

    one per process (one ``faulthandler`` handler for ``SIGALRM``): :meth:`claim` returns None when another watchdog
    owns it, or something else handles ``SIGALRM``, or (off Linux) the loop does not run on the main thread, the one
    a process-wide alarm interrupts. built on the loop thread: on Linux the alarm is that thread's own timer.

    :param timeout: the silence after which the dump is written, in seconds
    :ptype timeout: float
    """

    def __init__(self, timeout: float) -> None:
        self._timeout = timeout
        #: something else took ``SIGALRM`` over: the alarm and the handler are theirs, left alone from then on
        self.lost = False
        self._fd = _dump_file()
        self._timer: _ThreadTimer | None = None
        try:
            if sys.platform == "linux":
                global _THREAD_TIMER  # noqa: PLW0603 - the process's one alarm, owned under _DUMP_OWNER
                self._timer = _ThreadTimer()
                _THREAD_TIMER = (*_THREAD_TIMER, self._timer)
            # a registration left from a watchdog whose alarm was taken over: ``faulthandler`` would keep it and not
            # install its handler again, leaving SIGALRM at whatever the other party set back (its default ends the
            # process). the alarm is free here (see :meth:`claim`), so dropping it touches no one's handler
            faulthandler.unregister(signal.SIGALRM)
            # one thread, never all: the all-threads walk reads freed thread states when a thread exits under it
            # (the module's docstring; gh-116008)
            faulthandler.register(signal.SIGALRM, file=self._fd, all_threads=False, chain=False)
        except BaseException:
            self._drop_timer()
            os.close(self._fd)
            raise

    @classmethod
    def claim(cls, timeout: float) -> _HeldStacks | None:
        """the process's held-loop dump, or None when another watchdog has it, ``SIGALRM`` is in use, or (off Linux)
        this is not the main thread: there the alarm interrupts the main thread, so its dump is the loop's only when
        the loop runs there. called on the loop thread.

        :param timeout: the silence after which the dump is written, in seconds
        :ptype timeout: float
        :return: the dump, owned until :meth:`close`
        :rtype: _HeldStacks | None
        """
        if sys.platform != "linux":
            # no alarm off Linux: a process-wide SIGALRM goes to the main thread, the loop's, and fires only in a
            # stall -- when that thread is most often waiting for the interpreter lock another thread holds (a long
            # collection, Arrow's conversions). On macOS the signal ended that wait with an error, which CPython takes
            # as fatal: the reports tool pod died three times so in one evening's burst replays (2026-11-22, CPython
            # 3.14.3, "take_gil: PyCOND_WAIT(gil->cond) failed" then SIGSEGV, the loop-watchdog thread waiting on the
            # lock each time). Linux's alarm is the loop thread's own timer and its condition wait never fails on a
            # signal. The stall is still logged, with the watchdog thread's stacks; only the held-loop dump is Linux's
            return None
        if not _DUMP_OWNER.acquire(blocking=False):
            return None
        if not _alarm_is_free():
            _DUMP_OWNER.release()
            return None
        try:
            return cls(timeout)
        except BaseException:
            _DUMP_OWNER.release()
            raise

    def arm(self) -> None:
        """set the alarm one timeout ahead: the dump is written unless :meth:`arm` or :meth:`take` comes first.

        a dump written since the last arm, by a silence that ended short of a stall on the watchdog's clock, is
        thrown away.
        """
        # the alarm pushed back first: the old one could otherwise go off between the look and the re-arm, and its dump
        # would stay in the file ahead of the next
        self._set(self._timeout)
        if os.lseek(self._fd, 0, os.SEEK_CUR):
            self._empty()

    def take(self) -> str:
        """the dump written since the last arm, emptied; the alarm is off until the next :meth:`arm`.

        :return: the dump's text, at most :data:`DUMP_BYTES`, or empty when none was written
        :rtype: str
        """
        self._set(0)
        size = os.lseek(self._fd, 0, os.SEEK_CUR)
        if not size:
            return ""
        os.lseek(self._fd, 0, os.SEEK_SET)
        data = os.read(self._fd, min(size, DUMP_BYTES))
        self._empty()
        return data.decode("utf-8", errors="replace")

    def _set(self, seconds: float) -> None:
        """the alarm ``seconds`` from now, 0 off: the loop thread's own timer, or the process's."""
        if self._timer is not None:
            self._timer.set(seconds)
        else:
            signal.setitimer(signal.ITIMER_REAL, seconds)

    def _drop_timer(self) -> None:
        if self._timer is not None:
            global _THREAD_TIMER  # noqa: PLW0603 - the process's one alarm, owned under _DUMP_OWNER
            self._timer.delete()
            _THREAD_TIMER = tuple(timer for timer in _THREAD_TIMER if timer is not self._timer)
            self._timer = None

    def _empty(self) -> None:
        os.ftruncate(self._fd, 0)
        os.lseek(self._fd, 0, os.SEEK_SET)

    def close(self) -> None:
        """the alarm off, ``SIGALRM`` given back, the file closed and the process's alarm given up; when something
        else took ``SIGALRM`` over, its alarm and handler are left as they are (and the file open, which
        ``faulthandler`` may still name). the loop thread's own timer is ours alone, deleted either way."""
        if self._timer is not None:
            self._drop_timer()
        elif not self.lost:
            signal.setitimer(signal.ITIMER_REAL, 0)
        if not self.lost:
            faulthandler.unregister(signal.SIGALRM)
            os.close(self._fd)
        _DUMP_OWNER.release()


def _forget_timers() -> None:
    """a forked child has none of its parent's timers (POSIX): nothing of theirs to disarm at its exit."""
    global _THREAD_TIMER  # noqa: PLW0603 - the child's own, before anything in it runs
    _THREAD_TIMER = ()


atexit.register(_disarm)
os.register_at_fork(after_in_child=_forget_timers)


class LoopWatchdog:
    """a heartbeat on one event loop, and a thread that logs when it stops, with what held it.

    :param stall_seconds: the silence logged as a stall
    :ptype stall_seconds: float
    :param beat_seconds: the heartbeat's period, and the watchdog's
    :ptype beat_seconds: float
    :param repeat_seconds: how often a continuing stall is logged again
    :ptype repeat_seconds: float
    :param clock: the monotonic clock (injected by tests)
    :ptype clock: Callable[[], float]
    :param logger: where stalls are logged (injected by tests)
    :ptype logger: Any
    :param frames: the most frames of each thread's stack a warning carries
    :ptype frames: int
    :param stack_chars: the most characters of threads' stacks a warning carries
    :ptype stack_chars: int
    :param gc_slow_seconds: a collection longer than this is logged on its own
    :ptype gc_slow_seconds: float
    :param stack_dumps: whether a held loop's threads are dumped while it is held
    :ptype stack_dumps: bool
    :param looked: called on the watchdog thread after each look, however it went (a test's way to know the
        watchdog has seen what it changed); none by default
    :ptype looked: Callable[[], object] | None
    """

    def __init__(
        self,
        *,
        stall_seconds: float = STALL_SECONDS,
        beat_seconds: float = BEAT_SECONDS,
        repeat_seconds: float = REPEAT_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        logger: Any = log,
        frames: int = FRAMES,
        stack_chars: int = STACK_CHARS,
        gc_slow_seconds: float = GC_SLOW_SECONDS,
        stack_dumps: bool = True,
        looked: Callable[[], object] | None = None,
    ) -> None:
        self._stall = stall_seconds
        self._beat_every = beat_seconds
        self._repeat = repeat_seconds
        self._clock = clock
        self._log = logger
        self._frames = frames
        self._stack_chars = stack_chars
        self._gc_slow = gc_slow_seconds
        self._stack_dumps = stack_dumps
        self._looked = looked
        self._beat = clock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: int | None = None
        self._timer: asyncio.TimerHandle | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._held: _HeldStacks | None = None
        #: beats past the bound, from the heartbeat to the watchdog thread: the heartbeat alone assigns it, a new
        #: tuple each time (never changed in place across threads); the thread reads those ending after ``_late_seen``
        self._late: tuple[_LateBeat, ...] = ()
        self._late_seen = -math.inf
        #: the collections, from the collector's callback to the watchdog thread: the callback alone assigns it, a new
        #: tuple of the newest :data:`GC_KEPT` each time; the thread reads those after ``_gc_seen``
        self._collections: tuple[_Collection, ...] = ()
        self._gc_seq = itertools.count(1)
        self._gc_started = 0.0
        self._gc_seen = 0
        self._failures = 0
        self._stalls = 0
        #: stalls counted but not logged, the log's budget spent
        self._unlogged = 0
        #: stalls not logged since the last one that was, said by the next one logged
        self._unlogged_since_told = 0
        #: the longest silence seen, in seconds
        self._longest = 0.0
        #: when the current log window began, and the stalls logged in it
        self._window_start: float | None = None
        self._logged_in_window = 0
        #: slow collections: counted; those the budget left unlogged since the last logged; the longest; their window
        self._slow_collections = 0
        self._gc_unlogged = 0
        self._longest_collection = 0.0
        self._gc_window_start: float | None = None
        self._gc_logged_in_window = 0

    def health(self) -> dict[str, Any]:
        """the watchdog's own health, for the pod to serve.

        :return: ``alive`` (its thread runs), ``failures`` (looks that failed and were logged), ``stalls`` (silences
            past the bound), ``unlogged`` (those of them the log's budget left unlogged), ``longest_ms`` (the longest
            silence seen, to the ms), ``stall_ms`` (the bound), ``stack_dumps`` (whether held-loop dumps are taken),
            ``slow_collections`` and ``longest_collection_ms`` (collections past the collector's bound)
        :rtype: dict[str, Any]
        """
        thread = self._thread
        return {
            "alive": thread is not None and thread.is_alive(),
            "failures": self._failures,
            "stalls": self._stalls,
            "unlogged": self._unlogged,
            "longest_ms": round(self._longest * 1000),
            "stall_ms": round(self._stall * 1000),
            "stack_dumps": self._held is not None and not self._held.lost,
            "slow_collections": self._slow_collections,
            "longest_collection_ms": round(self._longest_collection * 1000, 1),
        }

    @property
    def last_beat(self) -> float:
        """when the loop last beat, on the watchdog's clock.

        :return: the heartbeat's latest stamp
        :rtype: float
        """
        return self._beat

    @property
    def running(self) -> bool:
        """whether the heartbeat or the watchdog thread is still on: from :meth:`start` until :meth:`stop` returns.

        :return: ``True`` while either runs
        :rtype: bool
        """
        return self._thread is not None or self._timer is not None

    def start(self) -> None:
        """start watching the running loop: called on it.

        :return: nothing
        :rtype: None
        :raises RuntimeError: when no loop runs on this thread, or it is already started
        """
        if self._thread is not None:
            raise RuntimeError("the loop watchdog is already started")
        self._loop = asyncio.get_running_loop()
        self._loop_thread = threading.get_ident()
        self._beat = self._clock()
        if self._stack_dumps:
            try:
                self._held = _HeldStacks.claim(self._stall)
            except (OSError, ValueError, RuntimeError) as error:
                # a diagnostic never stops a pod: without the dump, stalls are still timed and logged
                self._held = None
                self._log.warning(
                    "the loop watchdog runs without held-loop dumps: the dump could not be set up",
                    extra={"extra_data": {"error": repr(error)}},
                )
            if self._held is not None:
                self._held.arm()
        gc.callbacks.append(self._collecting)
        self._timer = self._loop.call_later(self._beat_every, self._beating)
        self._thread = threading.Thread(target=self._watch, name="loop-watchdog", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """stop watching: the heartbeat ends, the thread is joined, the dump and the collector's clock given up.

        :return: nothing
        :rtype: None
        """
        self._stop.set()
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if self._collecting in gc.callbacks:
            gc.callbacks.remove(self._collecting)
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 4 * self._beat_every))
            self._thread = None
        if self._held is not None:
            self._held.close()
            self._held = None

    def _beating(self) -> None:
        """the heartbeat, on the loop: stamp the time, record a beat past the bound with the dump taken meanwhile,
        re-arm the dump when due, come round again."""
        now = self._clock()
        before, self._beat = self._beat, now
        held = self._held if self._held is not None and not self._held.lost else None
        if now - before > self._stall:
            # the stamp moved first: the watchdog thread reads the late beats before the stamp, so it never takes a
            # stall that has ended for one going on
            self._late = (
                *self._late[-(LATE_KEPT - 1) :],
                _LateBeat(before, now, held.take() if held is not None else ""),
            )
        if held is not None:
            held.arm()
        if not self._stop.is_set() and self._loop is not None:
            self._timer = self._loop.call_later(self._beat_every, self._beating)

    def _collecting(self, phase: str, info: dict[str, Any]) -> None:
        """the collector's callback, on whichever thread collects: it only records, the watchdog thread logs."""
        if phase == "start":
            self._gc_started = self._clock()
            return
        self._collections = (
            *self._collections[-(GC_KEPT - 1) :],
            _Collection(
                next(self._gc_seq),
                info["generation"],
                self._gc_started,
                self._clock(),
                info["collected"],
                info["uncollectable"],
            ),
        )

    def _stack(self) -> str:
        """the loop thread's stack now, innermost last; empty when the thread is gone."""
        frame = current_frames().get(self._loop_thread or -1)
        return "" if frame is None else "".join(traceback.format_stack(frame)[-self._frames :])

    def _thread_stacks(self) -> str:
        """every live thread's stack now but the watchdog's own, named, the loop's marked first; bounded.

        A worker thread that holds the interpreter lock (C code that does not release it) stalls the loop
        without the loop's stack showing why; its frame here does.
        """
        frames = current_frames()
        names = {thread.ident: thread.name for thread in threading.enumerate()}
        own = threading.get_ident()
        order = sorted((ident for ident in frames if ident != own), key=lambda ident: ident != self._loop_thread)
        parts: list[str] = []
        used = 0
        for shown, ident in enumerate(order):
            mark = " (event loop)" if ident == self._loop_thread else ""
            header = f"--- thread {names.get(ident, '?')!r} [{ident}]{mark}\n"
            body = "".join(traceback.format_stack(frames[ident])[-self._frames :])
            room = self._stack_chars - used - len(header)
            if len(body) > room:
                # the innermost frames say the most: a thread past the bound keeps its last ones
                if room > 0:
                    parts.append(header + "...\n" + body[-room:])
                    shown += 1
                left = len(order) - shown
                if left:
                    parts.append(f"--- {left} more threads not shown (bounded at {self._stack_chars} chars)\n")
                break
            parts.append(header + body)
            used += len(header) + len(body)
        return "".join(parts)

    def format_held_stacks(self, dump: str) -> str:
        """a held-loop dump as a log line carries it: the loop thread's innermost frames, marked, bounded.

        ``faulthandler`` writes the interrupted thread's stack innermost frame first, under one header.

        :param dump: the dump as ``faulthandler`` wrote it
        :ptype dump: str
        :return: the bounded dump
        :rtype: str
        """
        lines = dump.strip("\n").split("\n")
        first = next((n + 1 for n, line in enumerate(lines) if line.startswith(_DUMP_HEADER)), 0)
        if first:
            lines[first - 1] += " (event loop)"
        kept = lines[: first + self._frames]
        if len(lines) > len(kept):
            kept.append(f"  ... {len(lines) - len(kept)} outer frames not shown")
        text = "\n".join(kept) + "\n"
        if len(text) > self._stack_chars:
            # the header and the innermost frames say the most
            text = text[: self._stack_chars] + f"\n... bounded at {self._stack_chars} chars\n"
        return text

    def _watch(self) -> None:
        """the watchdog thread: look every beat; a look that fails is logged loudly and the next one runs."""
        state: dict[str, Any] = {"stalled_since": None, "last_told": 0.0, "told": False}
        while not self._stop.wait(self._beat_every):
            try:
                self._look(state)
            except Exception as exc:  # noqa: BLE001  # prawduct:allow prawduct/broad-except -- the watchdog must outlive any one look; logged at error
                self._failures += 1
                try:
                    contain(
                        exc,
                        log=self._log,
                        what="the loop watchdog failed one look; it keeps watching",
                        extra={"failures": self._failures},
                        level="error",
                    )
                except Exception as failed:  # noqa: BLE001  # prawduct:allow prawduct/broad-except -- the injected logger itself failed; the module's logs it
                    contain(failed, log=log, what="the loop watchdog's logger failed", level="error")
            if self._looked is not None:
                self._looked()

    def _may_log(self, now: float) -> bool:
        """whether a new stall may be logged: :data:`LOG_BUDGET` in each :data:`LOG_WINDOW_SECONDS`.

        :param now: the clock now
        :ptype now: float
        :return: whether it is logged; when not, it is counted as unlogged
        :rtype: bool
        """
        if self._window_start is None or now - self._window_start >= LOG_WINDOW_SECONDS:
            self._window_start = now
            self._logged_in_window = 0
        allowed = self._logged_in_window < LOG_BUDGET
        if allowed:
            self._logged_in_window += 1
        else:
            self._unlogged += 1
            self._unlogged_since_told += 1
        return allowed

    def _look(self, state: dict[str, Any]) -> None:
        """one look: the collections since the last, the stalls the heartbeat saw end, then the loop now -- a stall
        counted when it starts and logged (within the budget), again every :attr:`_repeat` while it lasts.

        :param state: ``stalled_since`` (the last beat before the stall, or None), ``last_told`` and ``told`` (whether
            this stall is logged), kept between looks
        :ptype state: dict[str, Any]
        """
        now = self._clock()
        held = self._held
        if held is not None and not held.lost and signal.getsignal(signal.SIGALRM) != signal.SIG_DFL:
            held.lost = True
            self._log.warning(
                "the loop watchdog stops taking held-loop dumps: something else now handles SIGALRM",
                extra={"extra_data": {"handler": repr(signal.getsignal(signal.SIGALRM))}},
            )
        self._review_collections(now)
        # the late beats before the stamp: the heartbeat moves the stamp before it records one
        for late in self._late:
            if late.end > self._late_seen:
                self._late_seen = late.end
                self._ended(state, late)
        silent = now - self._beat
        self._longest = max(self._longest, silent)
        stalled_since, last_told = state["stalled_since"], state["last_told"] or 0.0
        if silent <= self._stall:
            return
        if stalled_since is None:
            # the state moves before the log: a failure while logging does not count the same stall every beat
            state["stalled_since"] = self._beat
            state["last_told"] = now
            self._stalls += 1
            state["told"] = self._may_log(now)
            if state["told"]:
                self._tell(silent, unlogged=self._unlogged_since_told)
                self._unlogged_since_told = 0
        elif now - last_told >= self._repeat:
            # a stall still going is logged every repeat, whatever the budget: a long one is never only counted
            state["last_told"] = now
            if not state["told"]:
                state["told"] = True
                self._tell(silent, unlogged=self._unlogged_since_told)
                self._unlogged_since_told = 0
            else:
                self._tell(silent)

    def _ended(self, state: dict[str, Any], late: _LateBeat) -> None:
        """a stall the heartbeat saw end: the end of the one this thread saw going on, or one it never saw (the
        interpreter lock held throughout), counted and logged once, within the budget; either with its causes.

        :param state: the watchdog thread's state (see :meth:`_look`)
        :ptype state: dict[str, Any]
        :param late: the late beat
        :ptype late: _LateBeat
        """
        length = late.end - late.start
        self._longest = max(self._longest, length)
        seen, told = state["stalled_since"] is not None, state["told"]
        state["stalled_since"] = None
        if seen:
            if told:
                self._log.warning(
                    "the event loop ran again after a stall",
                    extra={"extra_data": {"stalled_seconds": round(length, 3), **self._causes(late)}},
                )
            return
        self._stalls += 1
        if not self._may_log(self._clock()):
            return
        data: dict[str, Any] = {"stalled_seconds": round(length, 3), **self._causes(late)}
        if self._unlogged_since_told:
            data["unlogged_before"] = self._unlogged_since_told
            self._unlogged_since_told = 0
        self._log.warning(
            "the event loop stalled with the interpreter lock held: no call was answered", extra={"extra_data": data}
        )

    def _causes(self, late: _LateBeat) -> dict[str, Any]:
        """what held the loop over a stall: the collections that overlapped it and the held-loop dump.

        :param late: the stall
        :ptype late: _LateBeat
        :return: ``cause`` (``"gc gen N, X ms"`` when collections cover half the stall or more, else where to look),
            ``gc`` (how many collections overlapped it, their total ms, the longest) when any did, and
            ``held_stacks`` when a dump was taken
        :rtype: dict[str, Any]
        """
        overlapping = [c for c in self._collections if c.end >= late.start and c.start <= late.end]
        total = sum(c.end - c.start for c in overlapping)
        data: dict[str, Any] = {}
        if overlapping and total * 2 >= late.end - late.start:
            top = max(overlapping, key=lambda c: c.end - c.start)
            more = (
                f" (+{len(overlapping) - 1} more, {round(total * 1000, 1)} ms in all)" if len(overlapping) > 1 else ""
            )
            data["cause"] = f"gc gen {top.generation}, {round((top.end - top.start) * 1000, 1)} ms{more}"
        elif late.dump:
            data["cause"] = "not the collector: see held_stacks"
        else:
            data["cause"] = "not the collector; no dump taken"
        if overlapping:
            longest = sorted(overlapping, key=lambda c: c.end - c.start, reverse=True)[:GC_NAMED]
            data["gc"] = {
                "collections": len(overlapping),
                "ms": round(total * 1000, 1),
                "longest": [c.said() for c in longest],
            }
        if late.dump:
            data["held_stacks"] = self.format_held_stacks(late.dump)
        return data

    def _review_collections(self, now: float) -> None:
        """log the collections since the last look that ran past the collector's bound, within their budget.

        :param now: the clock now
        :ptype now: float
        """
        for collection in self._collections:
            if collection.seq <= self._gc_seen:
                continue
            self._gc_seen = collection.seq
            took = collection.end - collection.start
            if took <= self._gc_slow:
                continue
            self._slow_collections += 1
            self._longest_collection = max(self._longest_collection, took)
            if self._gc_window_start is None or now - self._gc_window_start >= LOG_WINDOW_SECONDS:
                self._gc_window_start = now
                self._gc_logged_in_window = 0
            if self._gc_logged_in_window >= LOG_BUDGET:
                self._gc_unlogged += 1
                continue
            self._gc_logged_in_window += 1
            data = collection.said()
            if self._gc_unlogged:
                data["unlogged_before"] = self._gc_unlogged
                self._gc_unlogged = 0
            self._log.warning("a slow garbage collection held every thread", extra={"extra_data": data})

    def _tell(self, silent: float, *, unlogged: int | None = None) -> None:
        """log a stall going on, with the loop's stack and every other thread's.

        :param silent: how long the loop has been silent
        :ptype silent: float
        :param unlogged: the stalls the budget left unlogged since the last logged; None on a repeat
        :ptype unlogged: int | None
        """
        data: dict[str, Any] = {
            "stalled_seconds": round(silent, 3),
            "loop_stack": self._stack(),
            "thread_stacks": self._thread_stacks(),
        }
        if unlogged:
            data["unlogged_before"] = unlogged
        self._log.warning("the event loop is stalled: no call is answered while it is held", extra={"extra_data": data})
