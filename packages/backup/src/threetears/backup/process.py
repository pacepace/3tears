"""Subprocess plumbing shared by every dump driver.

Two primitives: :func:`stream_stdout` runs a command and yields its stdout in chunks (the *dump*
side — pg_dump/ysql_dump write the archive to stdout), and :func:`feed_stdin` runs a command and
streams bytes into its stdin (the *restore* side — pg_restore/ysqlsh read the archive from stdin).

Both drain stderr concurrently (so a chatty tool can't deadlock on a full stderr pipe), enforce an
optional wall-clock ``timeout``, and — critically — **never leave a child running**: if the caller
aborts, the operation times out, or the input stream errors, the child is killed before the helper
returns (otherwise a child blocked on a full stdout pipe would wedge ``proc.wait()`` forever). A
non-zero exit raises :class:`BackupToolError` carrying the captured stderr; when a restore child
dies early and breaks the stdin pipe, the exit-code diagnosis wins over the raw ``BrokenPipeError``
-- and if that child nonetheless exited ``0``, the short feed is itself raised, because an archive
that was only partly written must never be reported as a completed restore.

**The limit of that guarantee.** A short feed is detected only when the write side actually
breaks, which needs the unwritten remainder to exceed the OS pipe buffer (~64 KiB). Below that the
kernel absorbs the whole archive, a child that never reads a byte exits cleanly, and nothing here
can tell that apart from a successful restore. Detecting it by polling for a child that exited
before EOF was tried and does NOT work: ``pg_restore`` legitimately exits ``0`` at the archive's
end-of-archive marker without waiting for EOF, so that check failed every real restore. From
outside the process, "ignored the archive" and "consumed it and exited early" are the same
observation. A caller that needs certainty for small archives must verify the restored database --
which is what :mod:`threetears.backup.verify` is for.
"""

from __future__ import annotations

import asyncio
import os
import signal
from collections.abc import AsyncIterator, Mapping

from threetears.observe import get_logger

__all__ = ["BackupToolError", "feed_stdin", "stream_stdout"]

log = get_logger(__name__)

_READ_CHUNK = 1 << 16  # 64 KiB
_TIMED_OUT = -1  # synthetic returncode used in the timeout message
_SHORT_FEED = -2  # synthetic returncode: the child exited 0 without consuming the whole archive


class BackupToolError(RuntimeError):
    """A dump/restore subprocess exited non-zero (or timed out).

    :param tool: the command that failed (argv[0]).
    :param returncode: the process exit code, or a synthetic sentinel: ``-1`` for a timeout,
        ``-2`` for a child that exited without consuming the whole archive.
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
            # NOSILENT: the group is already gone, which is the state killpg was called to reach
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
    fully_fed = False
    try:
        if timeout is None:
            fully_fed = await _pump_stdin(proc, source)
        else:
            fully_fed = await asyncio.wait_for(_pump_stdin(proc, source), timeout=timeout)
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
    if not fully_fed:
        # Exit 0 on a short feed: the tool is happy, but it never received the whole archive, so
        # the restore is partial. Reporting success here would lose data with nothing to show it.
        # _SHORT_FEED rather than the real (zero) exit code: a consumer branching on
        # ``exc.returncode`` would read 0 as "no failure", which is precisely the reading this
        # error exists to prevent. Mirrors the _TIMED_OUT convention already in this module.
        raise BackupToolError(
            argv[0], _SHORT_FEED, f"stdin closed before the archive was fully written. {stderr_final}"
        )


async def _pump_stdin(proc: asyncio.subprocess.Process, source: AsyncIterator[bytes]) -> bool:
    """Write ``source`` into the child's stdin.

    :return: True if the whole source reached the child; False if the child closed the pipe first.
    """
    assert proc.stdin is not None
    fully_fed = True
    try:
        async for chunk in source:
            proc.stdin.write(chunk)
            await proc.stdin.drain()
        # The child cannot have finished reading yet: EOF is delivered by the close() below, and a
        # tool that consumes an archive reads to EOF. So an ALREADY-EXITED child here did not read
        # what we wrote.
        #
        # This is what catches a short feed the pipe buffer swallowed. Below ~64 KiB the writes sit
        # in the kernel buffer, the child never reads them, and no BrokenPipeError is ever raised --
        # so BrokenPipeError alone reported a partial restore of a small archive as a clean one,
        # which is the exact failure the guard was added for. `proc.returncode` is not usable here
        # (it stays None until the child is reaped), hence the bounded poll.
        proc.stdin.close()
        await proc.stdin.wait_closed()
    except (BrokenPipeError, ConnectionResetError) as exc:
        # The child died early. Its exit code + stderr are the better diagnosis, so the pipe error
        # itself is not raised -- but the caller is told the feed was short, because a child that
        # dies early and still exits 0 would otherwise pass off a partial archive as a full restore.
        # This is the ONLY signal available: it needs the unwritten remainder to exceed the pipe
        # buffer. See the module docstring for why the obvious alternative does not work.
        log.warning(
            "restore child closed its stdin before the archive was fully written",
            extra={"extra_data": {"pid": proc.pid, "error": str(exc), "error_type": type(exc).__name__}},
        )
        fully_fed = False
    # wait for the child to finish INSIDE the timed region, so ``timeout`` bounds the whole
    # restore (a child that reads its stdin fast but then processes for a long time is still capped).
    await proc.wait()
    return fully_fed
