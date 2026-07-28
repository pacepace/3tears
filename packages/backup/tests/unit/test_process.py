"""Unit tests for the subprocess plumbing — proven with sh/printf/cat, no database.

These exercise the real streaming/error paths (a subprocess is spawned) without needing a DB:
``printf`` stands in for a dump (produces stdout), ``cat`` for a restore (consumes stdin).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from threetears.backup.process import BackupToolError, feed_stdin, stream_stdout


async def _emit(data: bytes, *, chunk: int = 4) -> AsyncIterator[bytes]:
    for i in range(0, len(data), chunk):
        yield data[i : i + chunk]


async def _collect(stream: AsyncIterator[bytes]) -> bytes:
    buf = bytearray()
    async for part in stream:
        buf += part
    return bytes(buf)


@pytest.mark.asyncio
async def test_stream_stdout_yields_command_output() -> None:
    out = await _collect(stream_stdout(["printf", "%s", "hello-dump"]))
    assert out == b"hello-dump"


@pytest.mark.asyncio
async def test_stream_stdout_raises_on_nonzero_with_stderr() -> None:
    with pytest.raises(BackupToolError) as excinfo:
        await _collect(stream_stdout(["sh", "-c", "printf oops >&2; exit 3"]))
    assert excinfo.value.returncode == 3
    assert "oops" in excinfo.value.stderr


@pytest.mark.asyncio
async def test_feed_stdin_streams_into_command(tmp_path: Path) -> None:
    sink = tmp_path / "restored.bin"
    payload = b"restore-payload-" * 100

    await feed_stdin(["sh", "-c", f"cat > {sink}"], _emit(payload))

    assert sink.read_bytes() == payload


@pytest.mark.asyncio
async def test_feed_stdin_raises_on_nonzero() -> None:
    with pytest.raises(BackupToolError) as excinfo:
        await feed_stdin(["sh", "-c", "printf nope >&2; exit 2"], _emit(b"data"))
    assert excinfo.value.returncode == 2
    assert "nope" in excinfo.value.stderr


@pytest.mark.asyncio
async def test_env_is_passed_to_child() -> None:
    out = await _collect(stream_stdout(["sh", "-c", 'printf "%s" "$SECRET"'], env={"SECRET": "from-env"}))
    assert out == b"from-env"


@pytest.mark.asyncio
async def test_stream_stdout_timeout_kills_a_hung_child() -> None:
    import asyncio

    started = asyncio.get_running_loop().time()
    with pytest.raises(BackupToolError, match="timed out"):
        await _collect(stream_stdout(["sh", "-c", "sleep 5"], timeout=0.3))
    assert asyncio.get_running_loop().time() - started < 3  # killed, not waited out


@pytest.mark.asyncio
async def test_stream_stdout_abort_does_not_hang() -> None:
    import asyncio

    # child floods stdout; consume one chunk then abort — the generator must close promptly,
    # killing the child rather than blocking on proc.wait() against a full, undrained pipe.
    gen = stream_stdout(["sh", "-c", "yes 3tears | head -c 50000000"])
    first = await anext(gen)
    assert first
    await asyncio.wait_for(gen.aclose(), timeout=5)  # regressions here would hang the suite


@pytest.mark.asyncio
async def test_feed_stdin_early_exit_surfaces_backup_tool_error_not_broken_pipe() -> None:
    # child exits non-zero WITHOUT reading a large stdin stream -> the write side breaks; the real
    # exit code + stderr must win over the BrokenPipeError.
    big = _emit(b"x" * (5 * 1024 * 1024), chunk=64 * 1024)
    with pytest.raises(BackupToolError) as excinfo:
        await feed_stdin(["sh", "-c", "printf boom >&2; exit 2"], big)
    assert excinfo.value.returncode == 2
    assert "boom" in excinfo.value.stderr


@pytest.mark.asyncio
async def test_feed_stdin_short_feed_with_exit_zero_still_raises() -> None:
    """A child that exits 0 without reading a LARGE archive is a partial restore, and must raise."""
    with pytest.raises(BackupToolError, match="stdin closed before the archive was fully written"):
        await feed_stdin(["sh", "-c", "exit 0"], _emit(b"x" * (5 * 1024 * 1024), chunk=64 * 1024))


@pytest.mark.asyncio
async def test_feed_stdin_short_feed_reports_a_synthetic_returncode() -> None:
    """Not 0: a consumer branching on ``exc.returncode`` would read the real exit code as success."""
    with pytest.raises(BackupToolError) as excinfo:
        await feed_stdin(["sh", "-c", "exit 0"], _emit(b"x" * (5 * 1024 * 1024), chunk=64 * 1024))
    assert excinfo.value.returncode != 0


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [1024, 32 * 1024], ids=["1KiB", "32KiB"])
async def test_a_short_feed_below_the_pipe_buffer_is_known_to_go_undetected(size: int) -> None:
    """Pins the documented LIMIT of the short-feed guard, so nobody mistakes it for a guarantee.

    Detection needs the write side to actually break, which needs the unwritten remainder to exceed
    the OS pipe buffer (~64 KiB). Below that the kernel absorbs the whole archive and a child that
    reads nothing exits cleanly -- indistinguishable, from outside the process, from a restore that
    worked. Polling for a child that exited before EOF was tried and does not work: ``pg_restore``
    legitimately exits 0 at the end-of-archive marker without waiting for EOF, so that check failed
    every real restore (caught by the integration tier, not by this file).

    If this test ever FAILS, the guard got better -- update the module docstring and delete this.
    """
    await feed_stdin(["sh", "-c", "exit 0"], _emit(b"x" * size, chunk=64 * 1024))


@pytest.mark.asyncio
async def test_feed_stdin_does_not_false_positive_on_a_slow_but_legitimate_reader(tmp_path: Path) -> None:
    """The early-exit poll must not misread a child that is simply slow to start consuming."""
    sink = tmp_path / "slow.bin"
    payload = b"restore-payload-" * 64

    await feed_stdin(["sh", "-c", f"sleep 0.2; cat > {sink}"], _emit(payload))

    assert sink.read_bytes() == payload


@pytest.mark.asyncio
async def test_feed_stdin_timeout_kills_a_hung_child() -> None:
    import asyncio

    started = asyncio.get_running_loop().time()
    with pytest.raises(BackupToolError, match="timed out"):
        await feed_stdin(["sh", "-c", "sleep 5"], _emit(b"tiny"), timeout=0.3)
    assert asyncio.get_running_loop().time() - started < 3
