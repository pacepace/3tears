"""Live integration tests for S3ObjectStore against an S3 the test brings itself.

Marked ``integration`` so the default unit run excludes them.

These used to address a MinIO assumed to be already running at
``localhost:9000`` with a bucket someone had created by hand, overridable by
``OBJECT_STORE_*`` env vars. On any machine where that was not true -- every
fresh checkout, and CI -- they did not skip, they FAILED with ``NoSuchBucket``,
which teaches a reader that red is normal. The container now comes from the
``s3_container`` fixture, so the only precondition is Docker, and its absence
skips rather than fails.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import pytest

from threetears.object_store.s3 import S3ObjectStore

pytestmark = pytest.mark.integration


def _store(
    endpoint: str,
    bucket: str,
    credentials: tuple[str, str],
    part_size_bytes: int = 8 * 1024 * 1024,
) -> S3ObjectStore:
    """Build a store pointed at the session's S3 testcontainer.

    :param endpoint: container endpoint URL
    :ptype endpoint: str
    :param bucket: bucket the fixture created
    :ptype bucket: str
    :param credentials: access/secret pair the container accepts
    :ptype credentials: tuple[str, str]
    :param part_size_bytes: multipart part size
    :ptype part_size_bytes: int
    :return: configured store
    :rtype: threetears.object_store.s3.S3ObjectStore
    """
    access_key, secret_key = credentials
    return S3ObjectStore(
        endpoint_url=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        bucket=bucket,
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
async def test_put_get_delete_roundtrip_small(
    s3_container: tuple[str, str],
    s3_credentials: tuple[str, str],
) -> None:
    """A small object round-trips via single PUT + streamed read + presign.

    :param s3_container: endpoint and bucket of the session's S3 container
    :ptype s3_container: tuple[str, str]
    :param s3_credentials: access/secret pair the container accepts
    :ptype s3_credentials: tuple[str, str]
    :return: nothing
    :rtype: None
    """
    endpoint, bucket = s3_container
    store = _store(endpoint, bucket, s3_credentials)
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
async def test_put_get_roundtrip_multipart_large(
    s3_container: tuple[str, str],
    s3_credentials: tuple[str, str],
) -> None:
    """An object larger than one part round-trips via multipart upload.

    :param s3_container: endpoint and bucket of the session's S3 container
    :ptype s3_container: tuple[str, str]
    :param s3_credentials: access/secret pair the container accepts
    :ptype s3_credentials: tuple[str, str]
    :return: nothing
    :rtype: None
    """
    endpoint, bucket = s3_container
    store = _store(endpoint, bucket, s3_credentials, part_size_bytes=5 * 1024 * 1024)
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
