"""Unit tests for S3ObjectStore streaming/batching branches.

Uses an in-memory fake S3 client injected via the constructor ``session``
seam, so the critical paths the live-MinIO happy-path can't cheaply cover
run in CI: empty / exact-multiple / single-giant-chunk / abort-on-failure
uploads, >1000-key delete batching, and multi-page listing.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import pytest

from threetears.object_store.s3 import S3ObjectStore


# parity-exempt: aiobotocore StreamingBody stub -- botocore's dynamically-built response body has no importable Protocol to declare parity against; only iter_chunks is exercised
class _FakeBody:
    """Streaming body stub exposing aiobotocore's ``iter_chunks``."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    async def iter_chunks(self, size: int) -> AsyncIterator[bytes]:
        """Yield the body in ``size``-byte chunks.

        :param size: chunk size
        :ptype size: int
        :return: async byte stream
        :rtype: AsyncIterator[bytes]
        """
        for i in range(0, len(self._data), size):
            yield self._data[i : i + size]


class _S3State:
    """In-memory backend state shared across clients from one fake session."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.mtimes: dict[str, datetime] = {}
        self.delete_batches: list[list[str]] = []
        self.aborted: list[str] = []
        self.completed: list[str] = []
        self.page_size: int = 1000
        self.fail_part: int | None = None


# parity-exempt: aioboto3 S3 client stub -- a botocore-generated client with hundreds of operations and no importable Protocol; only the get/put/list/delete/presign calls S3ObjectStore makes are stubbed
class _FakeS3Client:
    """Minimal in-memory S3 client matching the calls S3ObjectStore makes."""

    def __init__(self, state: _S3State) -> None:
        self._s = state
        self._mpu: dict[str, dict[int, bytes]] = {}
        self._counter = 0

    async def __aenter__(self) -> _FakeS3Client:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def create_multipart_upload(self, *, Bucket: str, Key: str, ContentType: str | None = None) -> dict[str, Any]:
        self._counter += 1
        uid = f"mpu-{self._counter}"
        self._mpu[uid] = {}
        return {"UploadId": uid}

    async def upload_part(
        self, *, Bucket: str, Key: str, PartNumber: int, UploadId: str, Body: bytes
    ) -> dict[str, Any]:
        if self._s.fail_part is not None and PartNumber == self._s.fail_part:
            raise RuntimeError("simulated upload_part failure")
        self._mpu[UploadId][PartNumber] = bytes(Body)
        return {"ETag": f'"etag-{PartNumber}"'}

    async def complete_multipart_upload(
        self, *, Bucket: str, Key: str, UploadId: str, MultipartUpload: dict[str, Any]
    ) -> dict[str, Any]:
        stored = self._mpu.pop(UploadId)
        self._s.objects[Key] = b"".join(stored[p["PartNumber"]] for p in MultipartUpload["Parts"])
        self._s.completed.append(Key)
        return {}

    async def abort_multipart_upload(self, *, Bucket: str, Key: str, UploadId: str) -> dict[str, Any]:
        self._mpu.pop(UploadId, None)
        self._s.aborted.append(Key)
        return {}

    async def put_object(self, *, Bucket: str, Key: str, Body: bytes, ContentType: str | None = None) -> dict[str, Any]:
        self._s.objects[Key] = bytes(Body)
        return {}

    async def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        return {"Body": _FakeBody(self._s.objects[Key])}

    async def delete_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        self._s.objects.pop(Key, None)
        return {}

    async def delete_objects(self, *, Bucket: str, Delete: dict[str, Any]) -> dict[str, Any]:
        batch = [o["Key"] for o in Delete["Objects"]]
        self._s.delete_batches.append(batch)
        for k in batch:
            self._s.objects.pop(k, None)
        return {}

    async def list_objects_v2(
        self, *, Bucket: str, Prefix: str | None = None, ContinuationToken: str | None = None
    ) -> dict[str, Any]:
        matched = sorted(k for k in self._s.objects if Prefix is None or k.startswith(Prefix))
        start = int(ContinuationToken) if ContinuationToken else 0
        page = matched[start : start + self._s.page_size]
        _epoch = datetime(2020, 1, 1, tzinfo=UTC)
        resp: dict[str, Any] = {
            "Contents": [
                {
                    "Key": k,
                    "Size": len(self._s.objects[k]),
                    "LastModified": self._s.mtimes.get(k, _epoch),
                }
                for k in page
            ]
        }
        if start + self._s.page_size < len(matched):
            resp["IsTruncated"] = True
            resp["NextContinuationToken"] = str(start + self._s.page_size)
        return resp


# parity-exempt: aioboto3 Session.client() factory stub -- an external SDK context-manager factory with no importable Protocol to mirror
class _FakeSession:
    """Fake aioboto3 session handing out fresh in-memory clients."""

    def __init__(self, state: _S3State) -> None:
        self._state = state

    def client(self, *args: object, **kwargs: object) -> _FakeS3Client:
        """Return a fresh fake client over the shared state.

        :return: fake S3 client
        :rtype: _FakeS3Client
        """
        return _FakeS3Client(self._state)


def _store(state: _S3State, *, part_size_bytes: int = 5 * 1024 * 1024) -> S3ObjectStore:
    """Build a store wired to the in-memory fake session.

    :param state: shared fake backend state
    :ptype state: _S3State
    :param part_size_bytes: multipart part size
    :ptype part_size_bytes: int
    :return: store under test
    :rtype: S3ObjectStore
    """
    return S3ObjectStore(
        endpoint_url=None,
        access_key="k",
        secret_key="s",
        bucket="b",
        part_size_bytes=part_size_bytes,
        session=_FakeSession(state),
    )


async def _aiter(data: bytes, chunk: int) -> AsyncIterator[bytes]:
    """Yield ``data`` in ``chunk``-sized pieces.

    :param data: source bytes
    :ptype data: bytes
    :param chunk: chunk size
    :ptype chunk: int
    :return: async byte stream
    :rtype: AsyncIterator[bytes]
    """
    for i in range(0, len(data), chunk):
        yield data[i : i + chunk]


async def _empty() -> AsyncIterator[bytes]:
    """An empty async byte stream.

    :return: async byte stream that yields nothing
    :rtype: AsyncIterator[bytes]
    """
    if False:  # pragma: no cover
        yield b""


@pytest.mark.asyncio
async def test_put_empty_object_uses_single_put() -> None:
    """A 0-byte object stores via a single empty PUT (no multipart)."""
    state = _S3State()
    await _store(state).put("k/empty", _empty(), content_type="application/octet-stream")
    assert state.objects["k/empty"] == b""
    assert state.completed == []


@pytest.mark.asyncio
async def test_put_small_object_uses_single_put() -> None:
    """An object under one part stores via a single PUT."""
    state = _S3State()
    await _store(state).put("k/small", _aiter(b"hello", 2), content_type="text/plain")
    assert state.objects["k/small"] == b"hello"
    assert state.completed == []


@pytest.mark.asyncio
async def test_put_exact_multiple_skips_empty_final_part() -> None:
    """An object exactly N*part_size completes without a 0-byte final part."""
    part = 5 * 1024 * 1024
    state = _S3State()
    payload = b"x" * (2 * part)
    await _store(state, part_size_bytes=part).put(
        "k/exact", _aiter(payload, part), content_type="application/octet-stream"
    )
    assert state.objects["k/exact"] == payload
    assert state.completed == ["k/exact"]


@pytest.mark.asyncio
async def test_put_multipart_with_remainder() -> None:
    """A full part plus a remainder round-trips via multipart."""
    part = 5 * 1024 * 1024
    state = _S3State()
    payload = b"y" * (part + 1234)
    await _store(state, part_size_bytes=part).put(
        "k/rem", _aiter(payload, 65536), content_type="application/octet-stream"
    )
    assert state.objects["k/rem"] == payload


@pytest.mark.asyncio
async def test_put_single_chunk_larger_than_part() -> None:
    """One giant incoming chunk is drained into multiple parts."""
    part = 5 * 1024 * 1024
    state = _S3State()
    payload = b"z" * (3 * part)
    await _store(state, part_size_bytes=part).put(
        "k/big", _aiter(payload, 3 * part), content_type="application/octet-stream"
    )
    assert state.objects["k/big"] == payload


@pytest.mark.asyncio
async def test_put_aborts_multipart_on_mid_stream_failure() -> None:
    """A failure mid-upload aborts the multipart -- no orphaned parts."""
    part = 5 * 1024 * 1024
    state = _S3State()
    state.fail_part = 2
    payload = b"w" * (3 * part)
    with pytest.raises(RuntimeError, match="simulated upload_part failure"):
        await _store(state, part_size_bytes=part).put(
            "k/fail", _aiter(payload, part), content_type="application/octet-stream"
        )
    assert state.aborted == ["k/fail"]
    assert "k/fail" not in state.objects


@pytest.mark.asyncio
async def test_delete_many_batches_over_the_1000_key_limit() -> None:
    """>1000 keys are chunked into <=1000-key DeleteObjects requests."""
    state = _S3State()
    keys = [f"k/{i}" for i in range(2500)]
    await _store(state).delete_many(keys)
    assert [len(b) for b in state.delete_batches] == [1000, 1000, 500]
    assert all(len(b) <= 1000 for b in state.delete_batches)


@pytest.mark.asyncio
async def test_delete_many_empty_is_noop() -> None:
    """Deleting an empty list issues no request."""
    state = _S3State()
    await _store(state).delete_many([])
    assert state.delete_batches == []


@pytest.mark.asyncio
async def test_list_keys_paginates_and_filters_by_prefix() -> None:
    """list_keys walks every page and honors the prefix filter."""
    state = _S3State()
    state.page_size = 2
    for i in range(5):
        state.objects[f"p/{i}"] = b"x"
    state.objects["other"] = b"x"
    keys = [k async for k in _store(state).list_keys(prefix="p/")]
    assert sorted(keys) == ["p/0", "p/1", "p/2", "p/3", "p/4"]


@pytest.mark.asyncio
async def test_list_entries_carries_key_size_and_mtime() -> None:
    """list_entries paginates and yields each object's key, size, and mtime."""
    state = _S3State()
    state.page_size = 2
    older = datetime(2021, 6, 1, tzinfo=UTC)
    newer = datetime(2023, 6, 1, tzinfo=UTC)
    state.objects["p/old"] = b"abc"
    state.mtimes["p/old"] = older
    state.objects["p/new"] = b"defgh"
    state.mtimes["p/new"] = newer
    state.objects["p/mid"] = b"z"
    state.objects["other"] = b"x"
    entries = {e.key: e async for e in _store(state).list_entries(prefix="p/")}
    assert sorted(entries) == ["p/mid", "p/new", "p/old"]
    assert entries["p/old"].size_bytes == 3
    assert entries["p/old"].last_modified == older
    assert entries["p/new"].size_bytes == 5
    assert entries["p/new"].last_modified == newer
    # unset mtime falls back to the fake's epoch default, never crashes
    assert entries["p/mid"].last_modified == datetime(2020, 1, 1, tzinfo=UTC)


@pytest.mark.asyncio
async def test_open_read_streams_back_the_object() -> None:
    """open_read yields the stored bytes in chunks."""
    state = _S3State()
    state.objects["k/r"] = b"abcdefgh"
    chunks = [c async for c in _store(state).open_read("k/r")]
    assert b"".join(chunks) == b"abcdefgh"
