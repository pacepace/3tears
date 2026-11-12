"""A worker process that exits when the process that started it dies.

A spawned pool's worker waits on its call queue. When its parent is killed (SIGKILL, an out-of-memory kill, a test
runner's worker that died), the queue never reports an end, so the worker waits for ever: an orphan holding its
memory. :func:`exit_with_parent`, called from the pool's ``initializer``, ends it instead::

    ProcessPoolExecutor(mp_context=multiprocessing.get_context("spawn"), initializer=exit_with_parent)

A daemon thread compares :func:`os.getppid` with the parent the worker started under, once a second, and calls
:func:`os._exit` when it changes (the orphan is adopted by another process). It runs on every platform. Linux's
``prctl(PR_SET_PDEATHSIG)`` is not used: it fires when the *thread* that spawned the worker exits, and a pool spawns
its workers from whichever thread submits first (often a short-lived one), so it would kill healthy workers.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Final

__all__ = ["exit_with_parent"]

#: how often the watch compares the parent, in seconds
_INTERVAL: Final[float] = 1.0

#: whether this process already watches its parent
_watching = threading.Event()


def _watch(parent: int, interval: float) -> None:
    """end the process once its parent is no longer ``parent``.

    :param parent: the parent's pid at the start
    :ptype parent: int
    :param interval: seconds between looks
    :ptype interval: float
    :return: nothing (the process ends)
    :rtype: None
    """
    while os.getppid() == parent:
        time.sleep(interval)
    os._exit(0)


def exit_with_parent(interval: float = _INTERVAL) -> None:
    """make this process exit when its parent dies; a pool initializer. Once per process.

    :param interval: seconds between looks at the parent
    :ptype interval: float
    :return: nothing
    :rtype: None
    """
    if _watching.is_set():
        return
    _watching.set()
    threading.Thread(target=_watch, args=(os.getppid(), interval), name="exit-with-parent", daemon=True).start()
