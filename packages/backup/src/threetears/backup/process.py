"""Subprocess plumbing shared by every dump driver.

Two primitives: :func:`stream_stdout` runs a command and yields its stdout in chunks (the *dump*
side — pg_dump/ysql_dump write the archive to stdout), and :func:`feed_stdin` runs a command and
streams bytes into its stdin (the *restore* side — pg_restore/ysqlsh read the archive from stdin).
Both drain stderr concurrently (so a chatty tool can't deadlock on a full stderr pipe) and raise
:class:`BackupToolError` on a non-zero exit, carrying the captured stderr for diagnosis.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping

__all__ = ["BackupToolError", "feed_stdin", "stream_stdout"]

_READ_CHUNK = 1 << 16  # 64 KiB


class BackupToolError(RuntimeError):
    """A dump/restore subprocess exited non-zero.

    :param tool: the command that failed (argv[0]).
    :param returncode: the process exit code.
    :param stderr: the captured standard error (decoded, best-effort).
    """

    def __init__(self, tool: str, returncode: int, stderr: str) -> None:
        super().__init__(f"{tool} failed (exit {returncode}): {stderr.strip()}")
        self.tool = tool
        self.returncode = returncode
        self.stderr = stderr


async def _drain(stream: asyncio.StreamReader | None) -> bytes:
    return b"" if stream is None else await stream.read()


async def stream_stdout(argv: list[str], *, env: Mapping[str, str] | None = None) -> AsyncIterator[bytes]:
    """Run ``argv`` and yield its stdout in chunks; raise on a non-zero exit.

    :param argv: the command and its arguments.
    :param env: environment for the child (e.g. ``PGPASSWORD``); ``None`` inherits.
    :raises BackupToolError: when the command exits non-zero.
    """
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=dict(env) if env is not None else None,
    )
    stderr_task = asyncio.ensure_future(_drain(proc.stderr))
    assert proc.stdout is not None
    try:
        while True:
            chunk = await proc.stdout.read(_READ_CHUNK)
            if not chunk:
                break
            yield chunk
    finally:
        stderr = (await stderr_task).decode(errors="replace")
        returncode = await proc.wait()
    if returncode != 0:
        raise BackupToolError(argv[0], returncode, stderr)


async def feed_stdin(argv: list[str], source: AsyncIterator[bytes], *, env: Mapping[str, str] | None = None) -> None:
    """Run ``argv`` and stream ``source`` into its stdin; raise on a non-zero exit.

    :param argv: the command and its arguments.
    :param source: async iterator of bytes to write to the child's stdin.
    :param env: environment for the child; ``None`` inherits.
    :raises BackupToolError: when the command exits non-zero.
    """
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
        env=dict(env) if env is not None else None,
    )
    stderr_task = asyncio.ensure_future(_drain(proc.stderr))
    assert proc.stdin is not None
    try:
        async for chunk in source:
            proc.stdin.write(chunk)
            await proc.stdin.drain()
        proc.stdin.close()
    finally:
        stderr = (await stderr_task).decode(errors="replace")
        returncode = await proc.wait()
    if returncode != 0:
        raise BackupToolError(argv[0], returncode, stderr)
