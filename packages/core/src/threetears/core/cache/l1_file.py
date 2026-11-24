"""Where a process's SQLite L1 lives: one WAL file per process and database name, on ``/dev/shm``.

**Why a file, not memdb** (SQ1 audit, 2026-11-15, ``scratchpad/sq1/AUDIT.md`` item 5; Pace said yes 2026-11-15). The
memdb VFS locks the whole database: an event-loop reader that meets a writer on another thread sleeps in SQLite's
busy handler (1, 2, 5, 10 ... 100 ms steps, up to the 5 s ``busy_timeout``). Measured on Linux tmpfs
(``bench_opts_linux.log``), point reads on the loop every 0.5 ms while a thread rewrites keys:

==========================  ===============================  ===========================
writer                      memdb read max                   WAL file read max
==========================  ===============================  ===========================
10k keys, back to back      2,770 ms (starved: 4 reads in 4s) 1.59 ms
10k keys every 2 s          36.3 ms                          2.45 ms
1k keys, 2/s                11.1 ms                          3.45 ms
64 keys, 20/s               1.68 ms                          0.86 ms
==========================  ===============================  ===========================

Write cost is 0.4-1.3x memdb's on tmpfs. WAL lets readers read the last commit while one writer writes, so the
loop never waits on an eviction, a drop or a big pull-through on a thread. Committing memdb in small chunks did not
help (the writer takes the lock back before the sleeping reader wakes: 3,186 ms max at 256-key chunks).

**Why ``/dev/shm``.** It is tmpfs: the file's pages are memory, as memdb's were, counted against the container's
memory limit (a memory-backed ``emptyDir`` in a pod, sized in the deploy config from the measured need: SIZING.md).
``synchronous=OFF``: nothing here must survive a crash (the file never outlives its process, below), so no fsync.
Off Linux (a developer's macOS) there is no ``/dev/shm`` and the file goes to the temp directory: measured on macOS
APFS (``bench_opts.log``) the WAL file's read max was 1.64 ms where memdb starved for 5,202 ms, writes 1.2-1.9x, so
there is one code path everywhere and memdb is gone. ``THREETEARS_L1_DIR`` names another directory.

**Process-local, as memdb was.** Nothing in L1 may outlive its process (monotonic stamps, epoch-task-05; the
pending-invalidation queue, epoch-task-04). So:

- the name carries the pid, a random token made once per process, and the database name:
  ``threetears-l1-<pid>-<token>-<db_name>.sqlite``. The token because a pid alone is not unique on a shared volume:
  a pod's ``emptyDir`` outlives a container restart that can reuse the pid, and two containers of one pod can share
  the volume with a pid each counts from 1;
- the last backend of a name to reset removes the file, its ``-wal`` and ``-shm``; a process that exits cleanly
  removes every file it holds (``atexit``);
- a process's first L1 sweeps the directory of files no live process holds: each file has an ``.owner`` lock file
  its process holds an exclusive ``flock`` on for its whole life, and a file whose lock can be taken is a dead
  process's. A lock, not a pid check: containers of one pod can share the volume with separate pid namespaces,
  where a pid says nothing, while ``flock`` is the kernel's, on the one tmpfs.
"""

from __future__ import annotations

import atexit
import contextlib
import fcntl
import os
import queue
import re
import secrets
import tempfile
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

from threetears.core.frozen_state import with_entry, without_entry
from threetears.observe import BuildOnce, contain, get_logger

__all__ = [
    "acquire",
    "collected",
    "l1_directory",
    "release",
    "sweep",
]

log = get_logger(__name__)

#: names the directory the L1 files go in, in place of /dev/shm (or the temp directory off Linux)
_L1_DIR_ENV: Final = "THREETEARS_L1_DIR"
#: tmpfs on Linux; a memory-backed emptyDir in a pod
_SHM: Final = Path("/dev/shm")
_PREFIX: Final = "threetears-l1-"
#: a database name, as it goes into a file name
_SAFE: Final = re.compile(r"[^A-Za-z0-9_.-]")
#: the files SQLite keeps beside a WAL database
_SIDE_FILES: Final = ("", "-wal", "-shm")


def l1_directory() -> Path:
    """the directory this process's L1 files go in: ``THREETEARS_L1_DIR``, else ``/dev/shm``, else the temp dir.

    :return: the directory
    :rtype: Path
    """
    named = os.environ.get(_L1_DIR_ENV)
    if named:
        return Path(named)
    if _SHM.is_dir() and os.access(_SHM, os.W_OK):
        return _SHM
    return Path(tempfile.gettempdir())


@dataclass
class _L1File:
    """one L1 database file this process holds, and how many backends share it.

    :ivar path: the database file
    :ivar owner_fd: the open ``.owner`` lock file, held under ``flock`` while the process lives
    :ivar holders: backends in this process using the file
    """

    path: Path
    owner_fd: int
    holders: int = 0


_LOCK = threading.Lock()
#: this process's L1 files, by path; replaced whole under :data:`_LOCK`, never changed in place
_FILES: Mapping[str, _L1File] = MappingProxyType({})
#: the files whose holder was collected while :data:`_LOCK` was held (:func:`collected`), given back by the next
#: :func:`acquire` or :func:`release`; a ``SimpleQueue`` because its ``put`` takes no lock a finalizer could wait on
_COLLECTED: queue.SimpleQueue[Path] = queue.SimpleQueue()
#: the directories already swept, each once per process at its first file (the files it removed, by directory)
_SWEPT: BuildOnce[str, int] = BuildOnce()
#: this process's pid and token, made on its first L1 file; keyed by pid, so a forked child makes its own
_ME: BuildOnce[int, str] = BuildOnce()


def _me() -> str:
    """``<pid>-<token>``: this process's part of every file name it makes.

    :return: the pid and the process's random token
    :rtype: str
    """
    pid = os.getpid()
    return _ME.get(pid, lambda: f"{pid}-{secrets.token_hex(4)}")


def _remove(path: Path) -> None:
    """remove a database file and SQLite's files beside it, whichever exist.

    :param path: the database file
    :ptype path: Path
    :return: nothing
    :rtype: None
    """
    for suffix in _SIDE_FILES:
        # NOSILENT: SQLite makes the side files only when it needs them; a missing one is already removed
        with contextlib.suppress(FileNotFoundError):
            os.unlink(f"{path}{suffix}")


def _owner_path(path: Path) -> Path:
    return Path(f"{path}.owner")


def sweep(directory: Path) -> int:
    """remove every L1 file in ``directory`` whose owner lock no live process holds; this process's are kept.

    :param directory: the directory
    :ptype directory: Path
    :return: how many dead processes' databases were removed
    :rtype: int
    """
    removed = 0
    mine = f"{_PREFIX}{_me()}-"
    uid = os.getuid()
    try:
        owners = [
            entry for entry in directory.iterdir() if entry.name.startswith(_PREFIX) and entry.name.endswith(".owner")
        ]
    except OSError as exc:  # NOSILENT: an unreadable directory has nothing this process can sweep; said
        contain(exc, log=log, what="L1 sweep: the directory could not be listed; nothing swept")
        return 0
    for owner in owners:
        if owner.name.startswith(mine):
            continue
        try:
            if owner.stat().st_uid != uid:
                continue
            fd = os.open(owner, os.O_RDWR)
        except OSError:  # NOSILENT: gone already (another process swept it) or not ours to open
            continue
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:  # NOSILENT: a live process holds it; that is the answer
                continue
            # still the file at the path (not one another sweeper removed and its process made again)
            if os.fstat(fd).st_ino != owner.stat().st_ino:
                continue
            _remove(Path(str(owner)[: -len(".owner")]))
            os.unlink(owner)
            removed += 1
        except OSError as exc:  # NOSILENT: a file another sweeper removed first; the next start tries again
            contain(exc, log=log, what="L1 sweep: a dead process's file could not be removed")
        finally:
            os.close(fd)
    if removed:
        log.info(
            "removed dead processes' L1 files", extra={"extra_data": {"directory": str(directory), "removed": removed}}
        )
    return removed


def acquire(db_name: str, directory: Path | None = None) -> Path:
    """the file this process's L1 named ``db_name`` lives in, made fresh for its first holder.

    :param db_name: the database name
    :ptype db_name: str
    :param directory: where; :func:`l1_directory` when ``None``
    :ptype directory: Path | None
    :return: the database file's path
    :rtype: Path
    """
    where = directory if directory is not None else l1_directory()
    path = where / f"{_PREFIX}{_me()}-{_SAFE.sub('_', db_name)}.sqlite"
    key = str(path)
    global _FILES
    with _LOCK:
        _give_back_collected()
        held = _FILES.get(key)
        if held is None:
            # each directory is swept once per process, at its first file
            _SWEPT.get(str(where), lambda: sweep(where))
            owner = _owner_path(path)
            while True:
                # the name is this process's own (pid and token), so the lock is free unless a sweeper holds it
                # for the instant it takes to see the file is not dead: then wait, and check it is still the file
                fd = os.open(owner, os.O_RDWR | os.O_CREAT, 0o600)
                fcntl.flock(fd, fcntl.LOCK_EX)
                try:
                    same = os.fstat(fd).st_ino == owner.stat().st_ino
                except FileNotFoundError:  # NOSILENT: a sweeper removed it between open and lock; take a new one
                    same = False
                if same:
                    break
                os.close(fd)
            _remove(path)
            held = _L1File(path=path, owner_fd=fd)
            _FILES = with_entry(_FILES, key, held)
        held.holders += 1
    return path


def release(path: Path) -> None:
    """one holder lets go of ``path``; the last removes the file and its lock.

    :param path: the database file, as :func:`acquire` answered it
    :ptype path: Path
    :return: nothing
    :rtype: None
    """
    with _LOCK:
        _let_go(path)
        _give_back_collected()


def collected(path: Path) -> None:
    """:func:`release` for a holder the collector freed: a backend's finalizer, never a caller's own release.

    A collection runs on whatever thread allocated when it came due, and runs the finalizers of what it frees
    there: inside :func:`acquire`'s or :func:`release`'s hold of :data:`_LOCK`, a finalizer that waited for the lock
    waited on its own thread for ever (a unit run's ``SQLiteBackend.initialize`` sat 600 s, until pytest-timeout's
    alarm was raised inside the finalizer, which the collector swallowed; 2026-11-16). So the lock is only tried:
    when held, here or on another thread, the file is queued and given back by the next :func:`acquire` or
    :func:`release`, or at exit.

    :param path: the database file, as :func:`acquire` answered it
    :ptype path: Path
    :return: nothing
    :rtype: None
    """
    if not _LOCK.acquire(blocking=False):
        _COLLECTED.put(path)
        return
    try:
        _let_go(path)
        _give_back_collected()
    finally:
        _LOCK.release()


def _let_go(path: Path) -> None:
    """one holder of ``path`` lets go; the last removes the file. :data:`_LOCK` held.

    :param path: the database file
    :ptype path: Path
    :return: nothing
    :rtype: None
    """
    global _FILES
    _FILES, last = _released(_FILES, str(path))
    if last is not None:
        _close(last)


def _released(files: Mapping[str, _L1File], key: str) -> tuple[Mapping[str, _L1File], _L1File | None]:
    """``files`` after one holder of ``key`` lets go, and the file when that was its last holder.

    :param files: the files held, by path
    :ptype files: Mapping[str, _L1File]
    :param key: the file's path
    :ptype key: str
    :return: the files held next (``files`` itself unless the last holder let go), and the file to close or None
    :rtype: tuple[Mapping[str, _L1File], _L1File | None]
    """
    held = files.get(key)
    if held is None:
        return files, None
    held.holders -= 1
    if held.holders > 0:
        return files, None
    return without_entry(files, key), held


def _give_back_collected() -> None:
    """let go of every file queued by :func:`collected`. :data:`_LOCK` held.

    :return: nothing
    :rtype: None
    """
    while True:
        try:
            path = _COLLECTED.get_nowait()
        except queue.Empty:  # NOSILENT: nothing queued is the usual answer
            return
        _let_go(path)


def _close(held: _L1File) -> None:
    _remove(held.path)
    # NOSILENT: a sweep of another process's start may have taken the lock file first; gone is the goal
    with contextlib.suppress(FileNotFoundError):
        os.unlink(_owner_path(held.path))
    os.close(held.owner_fd)


@atexit.register
def _remove_all() -> None:
    """a clean exit leaves no file behind."""
    global _FILES
    with _LOCK:
        _give_back_collected()
        for held in _FILES.values():
            # NOSILENT: at exit nothing can be said; a file left behind is swept by the next process's first L1
            with contextlib.suppress(OSError):
                _close(held)
        _FILES = MappingProxyType({})
