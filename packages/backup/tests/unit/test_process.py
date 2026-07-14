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
