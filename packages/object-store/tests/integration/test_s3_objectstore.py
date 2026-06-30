"""Live integration tests for S3ObjectStore against a running MinIO.

Marked ``integration`` so the default unit run excludes them. Defaults
target the dev MinIO from the compose stack (localhost:9000, minioadmin,
bucket ``aibots-objects``); override via ``OBJECT_STORE_*`` env vars.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import pytest

from threetears.object_store.s3 import S3ObjectStore

pytestmark = pytest.mark.integration

_ENDPOINT = os.environ.get("OBJECT_STORE_ENDPOINT", "http://localhost:9000")
_ACCESS = os.environ.get("OBJECT_STORE_ACCESS_KEY", "minioadmin")
_SECRET = os.environ.get("OBJECT_STORE_SECRET_KEY", "minioadmin")
_BUCKET = os.environ.get("OBJECT_STORE_BUCKET", "aibots-objects")


def _store(part_size_bytes: int = 8 * 1024 * 1024) -> S3ObjectStore:
    """Build a store pointed at the dev MinIO.

    :param part_size_bytes: multipart part size
    :ptype part_size_bytes: int
    :return: configured store
    :rtype: S3ObjectStore
    """
    return S3ObjectStore(
        endpoint_url=_ENDPOINT,
        access_key=_ACCESS,
        secret_key=_SECRET,
        bucket=_BUCKET,
        part_size_bytes=part_size_bytes,
    )


async def _collect(stream: AsyncIterator[bytes]) -> bytes:
    """Drain a byte stream into one buffer (test helper only).

    :param stream: async byte stream
    :ptype stream: AsyncIterator[bytes]
    :return: full content
    :rtype: bytes
    """
    out = bytearray()
    async for chunk in stream:
        out.extend(chunk)
    return bytes(out)


async def _aiter(data: bytes, chunk: int) -> AsyncIterator[bytes]:
    """Yield ``data`` in ``chunk``-sized pieces as an async iterator.

    :param data: source bytes
    :ptype data: bytes
    :param chunk: chunk size
    :ptype chunk: int
    :return: async byte stream
    :rtype: AsyncIterator[bytes]
    """
    for i in range(0, len(data), chunk):
        yield data[i : i + chunk]


@pytest.mark.asyncio
async def test_put_get_delete_roundtrip_small() -> None:
    """A small object round-trips via single PUT + streamed read + presign."""
    store = _store()
    key = "itest/small.txt"
    payload = b"hello streaming object store"

    await store.put(key, _aiter(payload, 4), content_type="text/plain")
    got = await _collect(store.open_read(key))
    assert got == payload

    url = await store.presigned_get_url(key)
    assert "itest/small.txt" in url

    keys = [k async for k in store.list_keys(prefix="itest/")]
    assert key in keys

    await store.delete(key)
    after = [k async for k in store.list_keys(prefix="itest/")]
    assert key not in after


@pytest.mark.asyncio
async def test_put_get_roundtrip_multipart_large() -> None:
    """An object larger than one part round-trips via multipart upload."""
    store = _store(part_size_bytes=5 * 1024 * 1024)
    key = "itest/large.bin"
    payload = os.urandom(12 * 1024 * 1024)  # 12 MiB -> 3 parts at 5 MiB

    await store.put(
        key,
        _aiter(payload, 1024 * 1024),
        content_type="application/octet-stream",
        size=len(payload),
    )
    got = await _collect(store.open_read(key))
    assert got == payload

    await store.delete(key)
