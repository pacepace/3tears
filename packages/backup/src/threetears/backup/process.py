"""Subprocess plumbing shared by every dump driver.

Two primitives: :func:`stream_stdout` runs a command and yields its stdout in chunks (the *dump*
side — pg_dump/ysql_dump write the archive to stdout), and :func:`feed_stdin` runs a command and
streams bytes into its stdin (the *restore* side — pg_restore/ysqlsh read the archive from stdin).

Both drain stderr concurrently (so a chatty tool can't deadlock on a full stderr pipe), enforce an
optional wall-clock ``timeout``, and — critically — **never leave a child running**: if the caller
aborts, the operation times out, or the input stream errors, the child is killed before the helper
returns (otherwise a child blocked on a full stdout pipe would wedge ``proc.wait()`` forever). A
non-zero exit raises :class:`BackupToolError` carrying the captured stderr; when a restore child
dies early and breaks the stdin pipe, the exit-code diagnosis wins over the raw ``BrokenPipeError``.
"""

from __future__ import annotations

import asyncio
import os
import signal
from collections.abc import AsyncIterator, Mapping

__all__ = ["BackupToolError", "feed_stdin", "stream_stdout"]

_READ_CHUNK = 1 << 16  # 64 KiB
_TIMED_OUT = -1  # synthetic returncode used in the timeout message


class BackupToolError(RuntimeError):
    """A dump/restore subprocess exited non-zero (or timed out).

    :param tool: the command that failed (argv[0]).
    :param returncode: the process exit code (``-1`` for a timeout).
    :param stderr: the captured standard error (decoded, best-effort).
    """

    def __init__(self, tool: str, returncode: int, stderr: str) -> None:
        super().__init__(f"{tool} failed (exit {returncode}): {stderr.strip()}")
        self.tool = tool
        self.returncode = returncode
        self.stderr = stderr


async def _drain(stream: asyncio.StreamReader | None) -> bytes:
    return b"" if stream is None else await stream.read()


def _kill(proc: asyncio.subprocess.Process) -> None:
    """SIGKILL the whole child process group if it is still running.

    Killing the *group* (children are spawned with ``start_new_session=True``) ensures no
    grandchild is orphaned — and, critically, that no descendant keeps the stderr pipe's write end
    open, which would otherwise wedge the concurrent stderr drain on an abort/timeout.
    """
    if proc.returncode is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


async def stream_stdout(
    argv: list[str], *, env: Mapping[str, str] | None = None, timeout: float | None = None
) -> AsyncIterator[bytes]:
    """Run ``argv`` and yield its stdout in chunks; raise on a non-zero exit or timeout.

    :param argv: the command and its arguments.
    :param env: environment for the child (e.g. ``PGPASSWORD``); ``None`` inherits.
    :param timeout: wall-clock ceiling in seconds; ``None`` disables it.
    :raises BackupToolError: when the command exits non-zero or exceeds ``timeout``.
    """
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=dict(env) if env is not None else None,
        start_new_session=True,  # own process group, so _kill can take down the whole tree
    )
    stderr_task = asyncio.ensure_future(_drain(proc.stderr))
    deadline = None if timeout is None else asyncio.get_running_loop().time() + timeout
    assert proc.stdout is not None
    reached_eof = False
    try:
        while True:
            try:
                chunk = await _read_within(proc.stdout, deadline)
            except TimeoutError:
                _kill(proc)
                stderr = (await stderr_task).decode(errors="replace")
                raise BackupToolError(argv[0], _TIMED_OUT, f"timed out after {timeout}s. {stderr}") from None
            if not chunk:
                reached_eof = True
                break
            yield chunk
    finally:
        # if the consumer aborted (GeneratorExit) or we errored before EOF, the child may be
        # blocked writing to a now-undrained stdout pipe — kill it so wait() can't hang.
        if not reached_eof:
            _kill(proc)
        stderr_final = (await stderr_task).decode(errors="replace")
        returncode = await proc.wait()
    if returncode != 0:
        raise BackupToolError(argv[0], returncode, stderr_final)


async def _read_within(stream: asyncio.StreamReader, deadline: float | None) -> bytes:
    if deadline is None:
        return await stream.read(_READ_CHUNK)
    remaining = deadline - asyncio.get_running_loop().time()
    if remaining <= 0:
        raise TimeoutError
    return await asyncio.wait_for(stream.read(_READ_CHUNK), timeout=remaining)


async def feed_stdin(
    argv: list[str],
    source: AsyncIterator[bytes],
    *,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
) -> None:
    """Run ``argv`` and stream ``source`` into its stdin; raise on a non-zero exit or timeout.

    :param argv: the command and its arguments.
    :param source: async iterator of bytes to write to the child's stdin.
    :param env: environment for the child; ``None`` inherits.
    :param timeout: wall-clock ceiling in seconds; ``None`` disables it.
    :raises BackupToolError: when the command exits non-zero or exceeds ``timeout``.
    """
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
        env=dict(env) if env is not None else None,
        start_new_session=True,  # own process group, so _kill can take down the whole tree
    )
    stderr_task = asyncio.ensure_future(_drain(proc.stderr))
    pumped = False
    try:
        if timeout is None:
            await _pump_stdin(proc, source)
        else:
            await asyncio.wait_for(_pump_stdin(proc, source), timeout=timeout)
        pumped = True
    except TimeoutError:
        _kill(proc)
        stderr = (await stderr_task).decode(errors="replace")
        raise BackupToolError(argv[0], _TIMED_OUT, f"timed out after {timeout}s. {stderr}") from None
    finally:
        # only kill on the abnormal path (source raised / timed out); on success let the child
        # finish and exit cleanly rather than SIGKILLing it out from under a good run.
        if not pumped:
            _kill(proc)
        stderr_final = (await stderr_task).decode(errors="replace")
        returncode = await proc.wait()
    if returncode != 0:
        raise BackupToolError(argv[0], returncode, stderr_final)


async def _pump_stdin(proc: asyncio.subprocess.Process, source: AsyncIterator[bytes]) -> None:
    assert proc.stdin is not None
    try:
        async for chunk in source:
            proc.stdin.write(chunk)
            await proc.stdin.drain()
        proc.stdin.close()
        await proc.stdin.wait_closed()
    except BrokenPipeError, ConnectionResetError:
        # the child died early; it will have a non-zero exit + stderr, so let the caller's
        # returncode check surface BackupToolError (the real diagnosis) rather than this pipe error.
        pass
    # wait for the child to finish INSIDE the timed region, so ``timeout`` bounds the whole
    # restore (a child that reads its stdin fast but then processes for a long time is still capped).
    await proc.wait()
