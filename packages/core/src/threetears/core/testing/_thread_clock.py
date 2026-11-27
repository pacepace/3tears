"""the one owner of reading ANOTHER thread's CPU clock, which the loop-stall guard's watcher weights its samples by.

:mod:`threetears.core.testing.loop_stall` names what held a test's event loop from stacks a watcher thread samples
while a long turn runs. A sample is worth the CPU the loop thread spent since the one before it, never the wall time:
a turn that waits on ``future.result()`` while a worker thread burns spends no CPU of its own there, and a wall-time
count named that wait as the culprit. Python reads only the CALLING thread's clock (``time.thread_time``); another's
takes the platform:

- Linux (the testament pool's containers, CI): the thread's CPU clock id, ``time.pthread_getcpuclockid`` taken ON the
  thread itself, read with ``time.clock_gettime`` from the watcher. The id encodes the kernel's thread id, so a read
  after the thread ended fails (EINVAL) and answers None. Never ``pthread_getcpuclockid`` on another thread's ident
  from the watcher: an ident is a ``pthread_t``, a pointer glibc frees when the thread ends, and that call on a
  finished loop thread (``asyncio.run`` in a worker) killed five test processes with SIGSEGV on the pool (2026-11-26).
- macOS (the laptops; no ``pthread_getcpuclockid``): the thread's Mach port, ``pthread_mach_thread_np(pthread_self())``
  taken on the thread itself (no reference taken, nothing to free), read with ``thread_info(THREAD_BASIC_INFO)``, user
  plus system time in microseconds, through ``ctypes``; a port whose thread ended fails and answers None.
- Anywhere else: no clock (None); the guard then counts samples, as it did before.

**Verified on CPython 3.14**, Linux 6.x (the pool's image) and macOS 15 (Apple M4).
"""

from __future__ import annotations

import ctypes
import sys
import threading
import time
from collections.abc import Callable
from typing import Final

__all__ = ["own_cpu_clock"]

#: ``THREAD_BASIC_INFO``'s flavor and its size in ``natural_t`` words (``mach/thread_info.h``: two ``time_value_t``
#: of two ints each, then six ints)
_THREAD_BASIC_INFO: Final[int] = 3
_THREAD_BASIC_INFO_COUNT: Final[int] = 10


def _linux_clock() -> Callable[[], float | None] | None:
    """the calling thread's clock on Linux, or None where ``pthread_getcpuclockid`` is missing."""
    getcpuclockid = getattr(time, "pthread_getcpuclockid", None)
    if getcpuclockid is None:
        return None
    clock = getcpuclockid(threading.get_ident())  # on the thread itself, while it runs

    def read() -> float | None:
        try:
            return time.clock_gettime(clock)
        except OSError:  # NOSILENT: a thread that has ended has no clock; the guard counts its sample as no CPU
            return None

    return read


def _mach_clock() -> Callable[[], float | None] | None:
    """the calling thread's clock on macOS, or None off macOS."""
    if sys.platform != "darwin":
        return None
    libc = ctypes.CDLL(None)
    libc.pthread_self.restype = ctypes.c_void_p
    port_of = libc.pthread_mach_thread_np
    port_of.argtypes = [ctypes.c_void_p]
    port_of.restype = ctypes.c_uint32
    thread_info = libc.thread_info
    thread_info.argtypes = [
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_uint32),
    ]
    thread_info.restype = ctypes.c_int
    port = port_of(libc.pthread_self())  # on the thread itself, while it runs

    def read() -> float | None:
        info = (ctypes.c_int * _THREAD_BASIC_INFO_COUNT)()
        count = ctypes.c_uint32(_THREAD_BASIC_INFO_COUNT)
        if thread_info(port, _THREAD_BASIC_INFO, info, ctypes.byref(count)) != 0:
            return None
        # user_time (seconds, microseconds), then system_time (seconds, microseconds)
        return info[0] + info[1] / 1e6 + info[2] + info[3] / 1e6

    return read


def own_cpu_clock() -> Callable[[], float | None] | None:
    """the CALLING thread's CPU clock, as a function any thread may call: its CPU seconds, or None once it has ended.
    None on a platform with no way to read it from another thread.

    :return: the clock, or None
    :rtype: Callable[[], float | None] | None
    """
    return _linux_clock() or _mach_clock()
