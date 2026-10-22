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
would: the shape the Claude Agent SDK's transport consumes. The child is reaped by a thread of its own,
so it is collected even when the loop that started it has closed (a pooled session can outlive its
loop: :meth:`threetears.models.claude_cli_pool.ClaudeCliPool` abandons such sessions by signal).

**Setting ``user`` (``ClaudeAgentOptions.user``) brings the stall back**: with a uid, gid or groups to
change, CPython's ``_posixsubprocess`` cannot use ``vfork`` and makes a full ``fork()``. A working
directory does not: it rules out ``posix_spawn`` only, and ``vfork`` is still used.
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

        Checkpoints as anyio's own wrapper does: a cancelled caller writes nothing, and a send that
        did not have to wait still yields to the loop.

        :param item: the bytes to write
        :ptype item: bytes
        :raises anyio.ClosedResourceError: when this stream was closed
        :raises anyio.BrokenResourceError: when the child closed its end
        """
        await anyio.lowlevel.checkpoint_if_cancelled()
        try:
            self._writer.write(item)
            await self._writer.drain()
        except (ConnectionResetError, BrokenPipeError, RuntimeError) as exc:
            if self._closed:
                raise anyio.ClosedResourceError from exc
            if self._writer.is_closing():
                raise anyio.BrokenResourceError from exc
            raise
        await anyio.lowlevel.cancel_shielded_checkpoint()

    async def aclose(self) -> None:
        """Close the pipe: the child reads end of input."""
        self._closed = True
        self._writer.close()
        await anyio.lowlevel.checkpoint()


class _Process(anyio.abc.Process):
    """A ``subprocess.Popen`` child as an ``anyio.abc.Process`` on the running loop.

    A daemon thread of its own waits for the child and reaps it the moment it exits, whether or not
    anyone waits and whether or not the loop that started it is still running; the exit is passed to
    that loop when it is. One blocked thread per live child, outside anyio's thread pool, so a long-lived
    CLI never holds a pool token.
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
        self._loop = asyncio.get_running_loop()
        self._exited: asyncio.Future[int] = self._loop.create_future()
        threading.Thread(target=self._reap, name=f"reap-cli-{popen.pid}", daemon=True).start()

    def _reap(self) -> None:
        """Wait for the child (in its own thread), reap it, and tell the loop that started it."""
        returncode = self._popen.wait()
        try:
            self._loop.call_soon_threadsafe(self._settle, returncode)
        except RuntimeError:
            # NOSILENT: the starting loop has closed, so nothing on it waits for this exit; the child
            # is reaped above all the same, and returncode reads it.
            pass

    def _settle(self, returncode: int) -> None:
        if not self._exited.done():
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
        if asyncio.get_running_loop() is self._loop:
            return await asyncio.shield(self._exited)
        # on another loop than the one that started it: the reaper thread collects it either way
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


class _Pipes:
    """A child's pipes joined to the running loop, every transport made kept until the start succeeds.

    A start given up halfway must close each transport it made, not only the pipe file under it: a
    closed file leaves its fd registered with the loop, and a later pipe given the same fd number then
    cannot be joined (or, under uvloop, a poll handle watches a closed fd).
    """

    def __init__(self) -> None:
        self.made: list[asyncio.BaseTransport] = []

    async def read_end(self, pipe: IO[bytes] | None) -> _PipeReceiveStream | None:
        """Join a child's output pipe to the running loop."""
        if pipe is None:
            return None
        loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader()
        transport, _ = await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), pipe)
        self.made.append(transport)
        return _PipeReceiveStream(reader, transport)

    async def write_end(self, pipe: IO[bytes] | None) -> _PipeSendStream | None:
        """Join a child's input pipe to the running loop."""
        if pipe is None:
            return None
        loop = asyncio.get_running_loop()
        transport, protocol = await loop.connect_write_pipe(asyncio.streams.FlowControlMixin, pipe)
        self.made.append(transport)
        return _PipeSendStream(asyncio.StreamWriter(transport, protocol, None, loop))

    def close(self) -> None:
        """Close every transport made: each fd leaves the loop now."""
        for transport in self.made:
            transport.close()


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
    command: Sequence[str | os.PathLike[str]],
    *,
    stdin: int | IO[Any] | None = subprocess.PIPE,
    stdout: int | IO[Any] | None = subprocess.PIPE,
    stderr: int | IO[Any] | None = subprocess.PIPE,
    cwd: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
    **kwargs: Any,
) -> anyio.abc.Process:
    """``anyio.open_process`` on asyncio, with the child started in a worker thread by ``subprocess.Popen``.

    Takes the arguments ``anyio.open_process`` takes, the command as a list (the SDK always passes one).

    :param command: the program and its arguments
    :ptype command: Sequence[str | os.PathLike[str]]
    :param stdin: ``subprocess.PIPE``, a file, a descriptor or ``None``
    :ptype stdin: int | IO[Any] | None
    :param stdout: as ``stdin``
    :ptype stdout: int | IO[Any] | None
    :param stderr: as ``stdin``
    :ptype stderr: int | IO[Any] | None
    :param cwd: the child's working directory
    :ptype cwd: str | os.PathLike[str] | None
    :param env: the child's whole environment; the parent's when ``None``
    :ptype env: Mapping[str, str] | None
    :param kwargs: the rest of ``anyio.open_process``'s keywords (``user`` forces a ``fork``: see above)
    :ptype kwargs: Any
    :return: the started process
    :rtype: anyio.abc.Process
    :raises OSError: when the program cannot be started (``FileNotFoundError`` when it does not exist)
    """
    start = _Start(
        {
            "args": list(command),
            "stdin": stdin,
            "stdout": stdout,
            "stderr": stderr,
            "cwd": cwd,
            "env": env,
            "bufsize": 0,
            **kwargs,
        }
    )
    pipes = _Pipes()
    try:
        # abandon_on_cancel: a cancelled start returns now, rather than after a spawn that may take seconds
        popen = await anyio.to_thread.run_sync(start.run, abandon_on_cancel=True)
        return _Process(
            popen,
            await pipes.write_end(popen.stdin),
            await pipes.read_end(popen.stdout),
            await pipes.read_end(popen.stderr),
        )
    except BaseException:
        # A start cancelled or failed must not leave a child running, nor a pipe registered with the
        # loop: nothing else holds them. The cancellation or failure itself is re-raised.
        pipes.close()
        await start.give_up()
        raise
