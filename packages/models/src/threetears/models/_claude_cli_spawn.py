"""Start a Claude CLI subprocess without forking the worker, and without holding its event loop.

Under uvloop (uvicorn's loop whenever it is installed), ``asyncio.create_subprocess_exec`` is libuv's
``uv_spawn``: a full ``fork()`` of the calling process, made while the calling thread holds the
interpreter lock (uvloop ``handles/process.pyx``: ``PyOS_BeforeFork()`` then ``uv_spawn``), after which
the parent waits for the child's ``exec``. The fork copies the worker's page tables, so the time grows
with resident memory, and no thread of the process runs meanwhile -- not the event loop, not the thread
that answers uvicorn's health check. Measured from one process, one ``claude -v`` spawn the SDK's way
(``anyio.open_process``), longest stall of a 1 ms ticker over five spawns: uvloop 8 ms at 76 MB, 23 ms at
527 MB, 95 ms at 2 GB, on an idle host; the stdlib loop 8-10 ms at every size. In metallm's ~500 MB
production workers, on a loaded host, it was 2-12 s, and the supervisor killed the worker at 5 s
(2026-10-14..21: 134 of 208 freeze reports, the main thread in that spawn under
:meth:`threetears.models.claude_cli_pool.PooledCliSession.start`).

The stdlib's ``subprocess.Popen`` starts the child with ``vfork``/``posix_spawn`` on Linux, which copies
no page tables. :func:`open_process_without_fork` runs it in a worker thread and joins the pipes to the
running loop with public asyncio calls, returning the ``anyio.abc.Process`` that ``anyio.open_process``
would: the shape the Claude Agent SDK's transport consumes. Off asyncio (trio), it is
``anyio.open_process`` unchanged.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import threading
from collections.abc import Mapping, Sequence
from signal import Signals
from typing import IO, Any

import anyio
import anyio.abc
import anyio.lowlevel
import anyio.to_thread

__all__ = ["open_process_without_fork"]


class _PipeReceiveStream(anyio.abc.ByteReceiveStream):
    """A child's stdout or stderr pipe, read on the loop."""

    def __init__(self, reader: asyncio.StreamReader, transport: asyncio.ReadTransport) -> None:
        self._reader = reader
        self._transport = transport

    async def receive(self, max_bytes: int = 65536) -> bytes:
        """The next bytes the child wrote.

        :param max_bytes: the most to return
        :ptype max_bytes: int
        :return: at least one byte
        :rtype: bytes
        :raises anyio.EndOfStream: when the child closed the pipe
        """
        data = await self._reader.read(max_bytes)
        if not data:
            raise anyio.EndOfStream
        return data

    async def aclose(self) -> None:
        """Stop reading: a pending or later ``receive`` raises ``ClosedResourceError``."""
        self._reader.set_exception(anyio.ClosedResourceError())
        self._transport.close()
        await anyio.lowlevel.checkpoint()


class _PipeSendStream(anyio.abc.ByteSendStream):
    """A child's stdin pipe, written on the loop."""

    def __init__(self, writer: asyncio.StreamWriter) -> None:
        self._writer = writer
        self._closed = False

    async def send(self, item: bytes) -> None:
        """Write ``item`` and wait until the pipe has taken it.

        :param item: the bytes to write
        :ptype item: bytes
        :raises anyio.ClosedResourceError: when this stream was closed
        :raises anyio.BrokenResourceError: when the child closed its end
        """
        try:
            self._writer.write(item)
            await self._writer.drain()
        except (ConnectionResetError, BrokenPipeError, RuntimeError) as exc:
            if self._closed:
                raise anyio.ClosedResourceError from exc
            if self._writer.is_closing():
                raise anyio.BrokenResourceError from exc
            raise

    async def aclose(self) -> None:
        """Close the pipe: the child reads end of input."""
        self._closed = True
        self._writer.close()
        await anyio.lowlevel.checkpoint()


class _Process(anyio.abc.Process):
    """A ``subprocess.Popen`` child as an ``anyio.abc.Process`` on the running loop.

    Its exit is watched with a pidfd on the loop (Linux), so it is reaped when it exits, whether or
    not anyone waits; elsewhere ``wait`` waits in a worker thread.
    """

    def __init__(
        self,
        popen: subprocess.Popen[bytes],
        stdin: _PipeSendStream | None,
        stdout: _PipeReceiveStream | None,
        stderr: _PipeReceiveStream | None,
    ) -> None:
        self._popen = popen
        self._stdin = stdin
        self._stdout = stdout
        self._stderr = stderr
        self._exited: asyncio.Future[int] | None = None
        if hasattr(os, "pidfd_open"):
            loop = asyncio.get_running_loop()
            pidfd = os.pidfd_open(popen.pid)
            self._exited = loop.create_future()
            loop.add_reader(pidfd, self._reap, loop, pidfd)

    def _reap(self, loop: asyncio.AbstractEventLoop, pidfd: int) -> None:
        """The child exited: collect its status, which does not block now."""
        loop.remove_reader(pidfd)
        os.close(pidfd)
        returncode = self._popen.wait()
        if self._exited is not None and not self._exited.done():
            self._exited.set_result(returncode)

    async def aclose(self) -> None:
        """Close the pipes and wait for the child; a cancelled wait kills it first."""
        with anyio.CancelScope(shield=True) as scope:
            for stream in (self._stdin, self._stdout, self._stderr):
                if stream is not None:
                    await stream.aclose()
            scope.shield = False
            try:
                await self.wait()
            except BaseException:
                scope.shield = True
                self.kill()
                await self.wait()
                raise

    async def wait(self) -> int:
        """Wait for the child to exit.

        :return: its return code
        :rtype: int
        """
        if self._exited is not None:
            return await asyncio.shield(self._exited)
        return await anyio.to_thread.run_sync(self._popen.wait, abandon_on_cancel=True)

    def terminate(self) -> None:
        """Send the child SIGTERM."""
        self._popen.terminate()

    def kill(self) -> None:
        """Send the child SIGKILL."""
        self._popen.kill()

    def send_signal(self, signal: Signals) -> None:
        """Send the child ``signal``.

        :param signal: the signal
        :ptype signal: Signals
        """
        self._popen.send_signal(signal)

    @property
    def pid(self) -> int:
        """The child's process id."""
        return self._popen.pid

    @property
    def returncode(self) -> int | None:
        """The child's return code, ``None`` until it has exited and been reaped."""
        return self._popen.returncode

    @property
    def stdin(self) -> anyio.abc.ByteSendStream | None:
        """The child's stdin, when it is a pipe."""
        return self._stdin

    @property
    def stdout(self) -> anyio.abc.ByteReceiveStream | None:
        """The child's stdout, when it is a pipe."""
        return self._stdout

    @property
    def stderr(self) -> anyio.abc.ByteReceiveStream | None:
        """The child's stderr, when it is a pipe."""
        return self._stderr


async def _read_end(pipe: IO[bytes] | None) -> _PipeReceiveStream | None:
    """Join a child's output pipe to the running loop."""
    if pipe is None:
        return None
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    transport, _ = await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), pipe)
    return _PipeReceiveStream(reader, transport)


async def _write_end(pipe: IO[bytes] | None) -> _PipeSendStream | None:
    """Join a child's input pipe to the running loop."""
    if pipe is None:
        return None
    loop = asyncio.get_running_loop()
    transport, protocol = await loop.connect_write_pipe(asyncio.streams.FlowControlMixin, pipe)
    return _PipeSendStream(asyncio.StreamWriter(transport, protocol, None, loop))


def _discard(popen: subprocess.Popen[bytes]) -> None:
    """Kill a child whose start was given up, collect it and close its pipes. Blocks until it is gone."""
    popen.kill()
    popen.wait()
    for pipe in (popen.stdin, popen.stdout, popen.stderr):
        if pipe is not None:
            pipe.close()


class _Start:
    """One ``subprocess.Popen`` call in a worker thread, which its caller may give up on at any point.

    The caller's await returns at once when it is cancelled, while the thread may still be inside
    ``Popen``: whichever of the two finishes second discards the child, so a given-up start never
    leaves one running.
    """

    def __init__(self, popen_args: dict[str, Any]) -> None:
        self._popen_args = popen_args
        self._lock = threading.Lock()
        self._popen: subprocess.Popen[bytes] | None = None
        self._given_up = False

    def run(self) -> subprocess.Popen[bytes]:
        """Start the child (in a worker thread).

        :return: the child
        :rtype: subprocess.Popen[bytes]
        """
        popen: subprocess.Popen[bytes] = subprocess.Popen(**self._popen_args)  # noqa: S603 -- the caller's command, as anyio.open_process runs it
        with self._lock:
            self._popen = popen
            given_up = self._given_up
        if given_up:
            _discard(popen)
        return popen

    async def give_up(self) -> None:
        """Discard the child if it was started; one started later is discarded by :meth:`run`."""
        with self._lock:
            self._given_up = True
            popen = self._popen
        if popen is not None:
            with anyio.CancelScope(shield=True):
                await anyio.to_thread.run_sync(_discard, popen)


async def open_process_without_fork(
    command: str | bytes | Sequence[str | bytes | os.PathLike[str]],
    *,
    stdin: int | IO[Any] | None = subprocess.PIPE,
    stdout: int | IO[Any] | None = subprocess.PIPE,
    stderr: int | IO[Any] | None = subprocess.PIPE,
    cwd: str | bytes | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
    **kwargs: Any,
) -> anyio.abc.Process:
    """``anyio.open_process``, with the child started in a worker thread by ``subprocess.Popen``.

    Takes the arguments ``anyio.open_process`` takes. A string command runs through the shell, as
    there. Off asyncio it calls ``anyio.open_process`` itself.

    :param command: the program and its arguments, or a shell command line
    :ptype command: str | bytes | Sequence[str | bytes | os.PathLike[str]]
    :param stdin: ``subprocess.PIPE``, a file, a descriptor or ``None``
    :ptype stdin: int | IO[Any] | None
    :param stdout: as ``stdin``
    :ptype stdout: int | IO[Any] | None
    :param stderr: as ``stdin``
    :ptype stderr: int | IO[Any] | None
    :param cwd: the child's working directory
    :ptype cwd: str | bytes | os.PathLike[str] | None
    :param env: the child's whole environment; the parent's when ``None``
    :ptype env: Mapping[str, str] | None
    :param kwargs: the rest of ``anyio.open_process``'s keywords (``user``, ``start_new_session``, ...)
    :ptype kwargs: Any
    :return: the started process
    :rtype: anyio.abc.Process
    :raises OSError: when the program cannot be started (``FileNotFoundError`` when it does not exist)
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return await anyio.open_process(command, stdin=stdin, stdout=stdout, stderr=stderr, cwd=cwd, env=env, **kwargs)

    start = _Start(
        {
            "args": command,
            "shell": isinstance(command, (str, bytes)),
            "stdin": stdin,
            "stdout": stdout,
            "stderr": stderr,
            "cwd": cwd,
            "env": env,
            "bufsize": 0,
            **kwargs,
        }
    )
    try:
        # abandon_on_cancel: a cancelled start returns now, rather than after a spawn that may take seconds
        popen = await anyio.to_thread.run_sync(start.run, abandon_on_cancel=True)
        return _Process(
            popen, await _write_end(popen.stdin), await _read_end(popen.stdout), await _read_end(popen.stderr)
        )
    except BaseException:
        # A start cancelled or failed must not leave a child running: nothing else holds it. The
        # cancellation or failure itself is re-raised.
        await start.give_up()
        raise
