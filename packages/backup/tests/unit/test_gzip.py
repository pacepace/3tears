"""Unit tests for streaming gzip/gunzip."""

from __future__ import annotations

import gzip as stdlib_gzip
from collections.abc import AsyncIterator

import pytest

from threetears.backup.gzip import gunzip_stream, gzip_stream


async def _emit(data: bytes, *, chunk: int = 7) -> AsyncIterator[bytes]:
    for i in range(0, len(data), chunk):
        yield data[i : i + chunk]


async def _collect(stream: AsyncIterator[bytes]) -> bytes:
    buf = bytearray()
    async for part in stream:
        buf += part
    return bytes(buf)


@pytest.mark.parametrize("payload", [b"", b"tiny", b"CREATE TABLE t (id int);\n" * 5000, bytes(range(256)) * 40])
@pytest.mark.asyncio
async def test_round_trip(payload: bytes) -> None:
    assert await _collect(gunzip_stream(gzip_stream(_emit(payload)))) == payload


@pytest.mark.asyncio
async def test_output_is_standard_gzip() -> None:
    payload = b"interoperable-gzip-payload" * 100
    compressed = await _collect(gzip_stream(_emit(payload)))
    assert compressed[:2] == b"\x1f\x8b"  # gzip magic
    assert stdlib_gzip.decompress(compressed) == payload  # a plain `gunzip` can read it


@pytest.mark.asyncio
async def test_compresses_repetitive_data() -> None:
    payload = b"A" * 100_000
    compressed = await _collect(gzip_stream(_emit(payload)))
    assert len(compressed) < len(payload)  # actually smaller


@pytest.mark.asyncio
async def test_truncated_stream_is_rejected() -> None:
    # flush() does not raise on an incomplete gzip stream; gunzip must catch it via the eof flag.
    full = await _collect(gzip_stream(_emit(b"CREATE TABLE t (id int);\n" * 2000)))
    truncated = full[:-16]  # lop off the gzip trailer + tail

    with pytest.raises(ValueError, match="truncated"):
        await _collect(gunzip_stream(_emit(truncated)))
