"""Streaming gzip (de)compression for dump bytes.

A driver whose dump format isn't already compressed (Yugabyte's plain SQL) is gzipped on the way
to the store and gunzipped on the way back. Both directions are incremental (``zlib`` objects), so
a multi-GB dump streams through without buffering.
"""

from __future__ import annotations

import zlib
from collections.abc import AsyncIterator

__all__ = ["gunzip_stream", "gzip_stream"]

_GZIP_WBITS = 16 + zlib.MAX_WBITS  # emit a gzip header/trailer
_GUNZIP_WBITS = 32 + zlib.MAX_WBITS  # auto-detect gzip or zlib on the way back


async def gzip_stream(source: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    """Yield gzip-compressed chunks for ``source``."""
    compressor = zlib.compressobj(wbits=_GZIP_WBITS)
    async for chunk in source:
        block = compressor.compress(chunk)
        if block:
            yield block
    tail = compressor.flush()
    if tail:
        yield tail


async def gunzip_stream(source: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    """Yield decompressed chunks for a gzip stream ``source``.

    :raises ValueError: if the stream ends before the gzip trailer (truncated / incomplete).
    """
    decompressor = zlib.decompressobj(wbits=_GUNZIP_WBITS)
    async for chunk in source:
        block = decompressor.decompress(chunk)
        if block:
            yield block
    tail = decompressor.flush()
    if tail:
        yield tail
    # flush() does NOT raise on a truncated stream; eof is False until the gzip trailer is consumed.
    if not decompressor.eof:
        raise ValueError("truncated gzip stream: trailer not reached")
