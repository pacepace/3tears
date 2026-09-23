"""Run a regex over text under a wall-clock limit, with stdlib ``re``'s exact semantics.

Stdlib ``re`` cannot be interrupted: a pattern that backtracks catastrophically holds the calling
thread (and the GIL) until it finishes, which can take hours. Live (2026-09-22) an LLM-proposed
``(?P<employer>[^\\n]+)\\n(?:[^\\n]+\\n)*?COUNTY:...`` ran for 20+ minutes on one state's WARN page, on
the caller's event-loop thread, and stopped every other task in the process.

The ``regex`` package has a match timeout, but it is not ``re``: it classifies thousands of
characters differently for ``\\w``/``\\b``/``\\s`` and accepts syntax ``re`` refuses, so patterns that
work today would give different results. So matching stays on stdlib ``re`` and runs in a small
worker process that imports nothing but the standard library.

- **Bounded.** The whole exchange (sending the request, waiting, reading the answer) runs under
  one deadline. A call still unanswered at the deadline kills the worker and raises
  :class:`TimeoutError`; the next call starts a fresh worker.
- **Never out of step.** A worker is kept only after a call that completed its exchange cleanly.
  Any interruption in between -- a timeout, a signal, ``KeyboardInterrupt``, an error -- kills it,
  so a later caller can never read an earlier caller's answer.
- **Thread-safe.** One lock covers each whole exchange, so concurrent callers take turns.
- **Fork-safe.** The pipes are unbuffered and driven with ``os.write``/``os.read``, so no Python
  buffer lock can be held at a fork; a forked child gets a fresh lock and its own worker.
- **No orphan runs on.** The worker caps its own CPU time per request (``RLIMIT_CPU``), so if the
  parent dies mid-match the kernel stops the worker rather than letting it run for hours.

POSIX only (``selectors`` on pipes, ``resource``), like the rest of this package's deployment.
"""

from __future__ import annotations

import json
import os
import selectors
import subprocess
import sys
import threading
import time
from typing import Any, Final, Literal

from threetears.observe import get_logger

__all__ = ["bounded_matches"]

log = get_logger(__name__)

#: CPU seconds a request may use in the worker beyond its caller's own timeout, before the kernel
#: stops the worker. Only reached if the parent died mid-match (it kills a runaway at its deadline).
_ORPHAN_CPU_MARGIN_SECONDS: Final[int] = 10

#: The worker: one JSON request per line on stdin, one JSON response per line on stdout.
#: Standard library only, so it starts in milliseconds and cannot drift from the parent's ``re``.
_WORKER_SOURCE: Final[str] = r"""
import json
import re
import resource
import sys


def cap_cpu(seconds):
    usage = resource.getrusage(resource.RUSAGE_SELF)
    soft = int(usage.ru_utime + usage.ru_stime) + seconds
    _, hard = resource.getrlimit(resource.RLIMIT_CPU)
    if hard != resource.RLIM_INFINITY:
        soft = min(soft, hard)
    resource.setrlimit(resource.RLIMIT_CPU, (soft, hard))


for line in sys.stdin:
    request = json.loads(line)
    cap_cpu(request["cpu_seconds"])
    try:
        compiled = re.compile(request["pattern"], request["flags"])
        if request["mode"] == "search":
            match = compiled.search(request["text"])
            response = {"matches": [] if match is None else [match.groupdict()]}
        else:
            response = {"matches": [match.groupdict() for match in compiled.finditer(request["text"])]}
    except re.error as exc:
        response = {"error": str(exc)}
    sys.stdout.write(json.dumps(response) + "\n")
    sys.stdout.flush()
"""

# One worker per process, used by one caller at a time: the lock covers a whole exchange, so
# concurrent threads (asyncio.to_thread callers included) take turns on the pipe.
_lock = threading.Lock()
_worker: subprocess.Popen[bytes] | None = None


def _reset_after_fork() -> None:
    """In a forked child: a fresh lock and no worker, so the child starts its own.

    The child inherits the parent's lock (possibly held by a thread that does not exist in the
    child) and the parent's worker pipes. It closes its copies of the pipes -- unbuffered raw
    files, so no buffer lock is involved -- and never signals the worker, which is the parent's.
    """
    global _lock, _worker  # noqa: PLW0603 - per-process state, reset for the new process
    _lock = threading.Lock()
    inherited, _worker = _worker, None
    if inherited is not None:
        for stream in (inherited.stdin, inherited.stdout):
            if stream is not None:
                stream.close()


os.register_at_fork(after_in_child=_reset_after_fork)


def bounded_matches(
    pattern: str, flags: int, text: str, *, mode: Literal["search", "finditer"], timeout: float
) -> list[dict[str, str | None]]:
    """Every match's named groups, as ``re.compile(pattern, flags).finditer(text)`` (or the first,
    as ``.search``) would produce them -- computed in the worker, within *timeout* seconds.

    :param pattern: the regex; callers compile it with ``re`` first to report a syntax error
    :ptype pattern: str
    :param flags: ``re`` flags
    :ptype flags: int
    :param text: the text to match against
    :ptype text: str
    :param mode: ``"finditer"`` for every match, ``"search"`` for the first (a list of zero or one)
    :ptype mode: Literal["search", "finditer"]
    :param timeout: seconds the whole exchange may take
    :ptype timeout: float
    :return: each match's ``groupdict()``, in order
    :rtype: list[dict[str, str | None]]
    :raises TimeoutError: when the exchange runs past *timeout*; the worker is killed
    :raises ValueError: when the worker reports the pattern invalid (callers should have caught it)
    :raises RuntimeError: when the worker exits or answers with something unreadable
    """
    # json.dumps escapes every non-ASCII character (and lone surrogates), so both directions of
    # the pipe carry plain ASCII whatever the worker's locale.
    request = (
        json.dumps(
            {
                "pattern": pattern,
                "flags": flags,
                "text": text,
                "mode": mode,
                "cpu_seconds": int(timeout) + _ORPHAN_CPU_MARGIN_SECONDS,
            }
        )
        + "\n"
    ).encode("ascii")
    response: dict[str, Any] = {}
    with _lock:
        deadline = time.monotonic() + timeout
        worker: subprocess.Popen[bytes] | None = None
        clean = False
        try:
            worker = _running_worker()
            _write_by(worker, request, deadline)
            line = _read_line_by(worker, deadline)
            response = json.loads(line) if line else {}
            clean = "matches" in response or "error" in response
        except TimeoutError:
            raise  # a TimeoutError is an OSError too; it must reach the caller as itself
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"regex worker failed: {type(exc).__name__}: {exc}") from exc
        finally:
            # Kept only after a clean exchange. Anything else -- a timeout, a signal,
            # KeyboardInterrupt, a broken or unreadable answer -- leaves the worker out of step
            # with its pipe, so it is killed before the lock is released.
            if not clean and worker is not None:
                _discard_worker(worker)
    if "error" in response:
        raise ValueError(response["error"])
    if "matches" not in response:
        raise RuntimeError("regex worker exited without answering")
    matches: list[dict[str, str | None]] = response["matches"]
    return matches


def _running_worker() -> subprocess.Popen[bytes]:
    """The live worker, started if there is none (or the last one died)."""
    global _worker  # noqa: PLW0603 - one worker per process, guarded by _lock
    if _worker is None or _worker.poll() is not None:
        _worker = subprocess.Popen(  # noqa: S603 - our own interpreter running our own constant source
            [sys.executable, "-I", "-S", "-c", _WORKER_SOURCE],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        # Non-blocking sends: a write only ever takes what the pipe has room for, so the
        # deadline in _write_by holds even if the worker stops reading partway through a request.
        assert _worker.stdin is not None  # noqa: S101 - Popen was given stdin=PIPE
        os.set_blocking(_worker.stdin.fileno(), False)
    return _worker


def _write_by(worker: subprocess.Popen[bytes], payload: bytes, deadline: float) -> None:
    """Send *payload* to the worker, raising :class:`TimeoutError` at *deadline*."""
    assert worker.stdin is not None  # noqa: S101 - Popen was given stdin=PIPE
    fd = worker.stdin.fileno()
    view = memoryview(payload)
    with selectors.DefaultSelector() as selector:
        selector.register(fd, selectors.EVENT_WRITE)
        while view:
            left = deadline - time.monotonic()
            if left <= 0 or not selector.select(left):
                raise TimeoutError("regex matching ran past its time limit")
            try:
                view = view[os.write(fd, view[:65536]) :]
            except BlockingIOError:  # NOSILENT: the pipe filled between select and write; wait for room again
                continue


def _read_line_by(worker: subprocess.Popen[bytes], deadline: float) -> str:
    """Read the worker's one-line answer, raising :class:`TimeoutError` at *deadline*."""
    assert worker.stdout is not None  # noqa: S101 - Popen was given stdout=PIPE
    fd = worker.stdout.fileno()
    chunks: list[bytes] = []
    with selectors.DefaultSelector() as selector:
        selector.register(fd, selectors.EVENT_READ)
        while True:
            left = deadline - time.monotonic()
            if left <= 0 or not selector.select(left):
                raise TimeoutError("regex matching ran past its time limit")
            chunk = os.read(fd, 65536)
            chunks.append(chunk)
            if not chunk or chunk.endswith(b"\n"):
                break
    return b"".join(chunks).decode("ascii")


def _discard_worker(worker: subprocess.Popen[bytes]) -> None:
    """Kill *worker* and release its pipes; the next call starts a fresh worker."""
    global _worker  # noqa: PLW0603 - one worker per process, guarded by _lock
    worker.kill()
    worker.wait()
    for stream in (worker.stdin, worker.stdout):
        if stream is not None:
            stream.close()
    if _worker is worker:
        _worker = None
    log.warning("scrape: regex worker discarded (timed out, interrupted or broken)")
