"""Worker processes started by a fork server of their own, each exiting when the serving process dies.

**Started by a server, never by the serving process.** A pool's worker started with ``spawn`` is a ``fork_exec`` of
the whole serving process, run with the interpreter lock held: on macOS, which has no ``vfork``, it lasts as long as
the process is large, and a pool started or replaced mid-load stalls the event loop (40-77 ms at 1.2-1.9 GB, reports
2026-11-17: faulthandler on the late tick, the loop's thread in ``spawnv_passfds``). :func:`start_worker_server`
starts multiprocessing's fork server, a small single-threaded process, and the resource tracker, once; every worker of
a pool made in its context (the one it returns) is then forked by that server over its socket, and the serving process
forks nothing more. ``preload`` names modules the server imports as it starts, so each worker it forks holds them
already: a worker's start then costs no imports, and the pod holds their pages once, shared until written (reports'
full start 30.0-30.8 s, 22.7-24.0 s with its job modules preloaded, 2026-11-17). Start it as the pod starts, on a
thread: its own start is the one ``fork_exec`` left.

**Exiting with the serving process.** A spawned pool's worker waits on its call queue. When its parent is killed
(SIGKILL, an out-of-memory kill, a test runner's worker that died), the queue never reports an end, so the worker
waits for ever: an orphan holding its memory. :func:`exit_with_parent`, called from the pool's ``initializer``, ends
it instead::

    ProcessPoolExecutor(mp_context=start_worker_server(), initializer=exit_with_parent, initargs=(1.0, os.getpid()))

A daemon thread compares :func:`os.getppid` with the parent the worker started under, once a second, and calls
:func:`os._exit` when it changes (the orphan is adopted by another process). A worker the fork server forked has the
server for its parent, and the server lives while any worker it forked does (each holds the server's liveness pipe):
watching its parent alone, such a worker outlived its killed serving process for ever. Given ``serving``, the thread
watches that process too and exits once it is gone; the server, its last worker gone, exits after it. Not given, it
is the process that started the worker (``multiprocessing.parent_process()``: the serving process for a worker the
server forked, the parent for a spawned one), named by the pid handed over at the start rather than read as the
parent when the watch begins: a parent killed while the worker still imported its modules has left it adopted by
init by then, and a watch of that parent never ends. It runs on every
platform. Linux's ``prctl(PR_SET_PDEATHSIG)`` is not used: it fires when the *thread* that spawned the worker exits,
and a pool spawns its workers from whichever thread submits first (often a short-lived one), so it would kill healthy
workers.
"""

from __future__ import annotations

import ctypes
import multiprocessing
import multiprocessing.context
import os
import sys
import threading
import time
from collections.abc import Sequence
from multiprocessing import forkserver, resource_tracker
from typing import Final

from threetears.observe import get_logger

__all__ = ["exit_with_parent", "name_process", "start_worker_server"]

log = get_logger(__name__)

#: how often the watch compares the parent, in seconds
_INTERVAL: Final[float] = 1.0

#: how every pool's workers start: forked by multiprocessing's fork server, never by the serving process
_WORKER_START: Final = "forkserver"

#: the longest name a process holds (Linux's ``/proc/<pid>/comm``: 16 bytes with its terminating null); a longer one
#: is cut to it
_PROCESS_NAME_LENGTH: Final = 15

#: what ``prctl`` names the calling thread (the process, for its main thread) by: Linux's ``PR_SET_NAME``
_PR_SET_NAME: Final = 15

#: whether this process already watches its parent
_watching = threading.Event()

#: held while the server's preload is set and the server started
_server_lock = threading.Lock()

#: the modules the server was asked to preload by the first start here (one entry); empty before
_server_preload: list[frozenset[str]] = []


def _alive(pid: int) -> bool:
    """whether ``pid`` is a live process.

    :param pid: the process id
    :ptype pid: int
    :return: whether it lives
    :rtype: bool
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # the pid was taken by another user's process: the one watched is gone
        return False
    return True


def _watch(parent: int, serving: int | None, interval: float) -> None:
    """end the process once its parent is no longer ``parent``, or ``serving`` is gone.

    :param parent: the parent's pid at the start
    :ptype parent: int
    :param serving: the serving process's pid, watched too; None for the parent alone
    :ptype serving: int | None
    :param interval: seconds between looks
    :ptype interval: float
    :return: nothing (the process ends)
    :rtype: None
    """
    while os.getppid() == parent and (serving is None or _alive(serving)):
        time.sleep(interval)
    os._exit(0)


def exit_with_parent(interval: float = _INTERVAL, serving: int | None = None) -> None:
    """make this process exit when its parent dies, or ``serving`` does; a pool initializer. Once per process: a
    second call, with or without ``serving``, changes nothing.

    :param interval: seconds between looks at the parent
    :ptype interval: float
    :param serving: the serving process this worker works for, when its parent is another (the fork server); None
        for the process that started this one (``multiprocessing.parent_process()``), or the parent alone for a
        process multiprocessing did not start
    :ptype serving: int | None
    :return: nothing
    :rtype: None
    """
    if _watching.is_set():
        return
    _watching.set()
    parent = os.getppid()
    # the process that started this one, by the pid multiprocessing handed over at the start, never read as the
    # parent here: this runs after the start imported the worker's modules (1.6 s in the hub's read worker), and a
    # spawner killed meanwhile left this process adopted by init, which a watch of the parent begun now waited on
    # for ever (the hub's read worker, every one of four runs alone, 2026-11-18). For a worker the fork server forked it
    # is the serving process
    spawner = multiprocessing.parent_process()
    if serving is None and spawner is not None:
        serving = spawner.pid
    watched = None if serving == parent else serving
    threading.Thread(target=_watch, args=(parent, watched, interval), name="exit-with-parent", daemon=True).start()


def name_process(name: str) -> bool:
    """name this process as ``/proc/<pid>/comm`` (and ``ps``, ``top``) say it: Linux's ``prctl(PR_SET_NAME)`` on the
    calling thread, the process's main thread in a pool worker's initializer. Every worker the fork server forks has
    the server's command line, so its name is what tells one pool's workers from another's (an operator's fault aimed
    at a pool: ``aibots cluster fault kill-process --name``). Nothing elsewhere: macOS has no name another process
    reads so.

    :param name: the name, cut to :data:`_PROCESS_NAME_LENGTH` bytes
    :ptype name: str
    :return: whether the name was set (False off Linux, or when the call failed, said in the log)
    :rtype: bool
    """
    named = False
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        named = libc.prctl(_PR_SET_NAME, name.encode()[:_PROCESS_NAME_LENGTH], 0, 0, 0) == 0
        if not named:
            log.warning(
                "a worker process could not name itself",
                extra={"extra_data": {"name": name, "errno": ctypes.get_errno()}},
            )
    return named


def start_worker_server(preload: Sequence[str] = ()) -> multiprocessing.context.BaseContext:
    """start the fork server, and the resource tracker, now (each a ``fork_exec`` of this process once: its first
    worker or first multiprocessing lock otherwise, mid-load); started already, it starts nothing. Blocking: a
    thread's work, or a pod's start, never a busy event loop's.

    ``preload`` is read by the server as it starts, on top of multiprocessing's own (``__main__``): a server already
    running keeps the modules it started with, and a worker imports what it lacks itself, as a spawned one does
    (said in the log).

    :param preload: modules the server imports as it starts, so every worker it forks holds them
    :ptype preload: Sequence[str]
    :return: the context every pool's workers start in (:data:`_WORKER_START`)
    :rtype: multiprocessing.context.BaseContext
    """
    context = multiprocessing.get_context(_WORKER_START)
    wanted = frozenset(preload)
    with _server_lock:
        if not _server_preload:
            if wanted:
                context.set_forkserver_preload(["__main__", *sorted(wanted)])
            _server_preload.append(wanted)
        elif not wanted <= _server_preload[0]:
            log.info(
                "worker fork server already started: its workers import these modules themselves",
                extra={"extra_data": {"modules": sorted(wanted - _server_preload[0])}},
            )
        resource_tracker.ensure_running()
        forkserver.ensure_running()
    return context
