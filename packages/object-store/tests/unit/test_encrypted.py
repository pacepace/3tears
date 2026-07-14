"""Unit tests for :class:`EncryptedObjectStore` — streaming AEAD over any ObjectStore.

No infra: the inner store is an in-memory fake, so these assert the crypto contract
(round-trip fidelity, tamper/reorder/truncation detection, key isolation) and the
pass-through of the non-body operations.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from cryptography.exceptions import InvalidTag
from pydantic import SecretStr

from threetears.media.contracts import ObjectListing, ObjectStore
from threetears.object_store.encrypted import EncryptedObjectStore


def test_conforms_to_object_store_protocol() -> None:
    # mypy verifies the signatures structurally; runtime_checkable verifies the method set.
    store: ObjectStore = EncryptedObjectStore(_MemStore(), _PASS, scrypt_n=_FAST_N)
    assert isinstance(store, ObjectStore)


# a low scrypt factor so a per-object KDF doesn't dominate the suite (crypto is unchanged).
_FAST_N = 2**8
_PASS = SecretStr("correct horse battery staple")


class _MemStore:
    """In-memory ObjectStore fake: keys → stored bytes, with a delete/list log."""

    def __init__(self) -> None:
        self.blobs: dict[str, bytes] = {}
        self.content_types: dict[str, str] = {}
        self.deleted: list[str] = []
        self.presign_calls: list[tuple[str, int]] = []

    async def put(self, key: str, body: AsyncIterator[bytes], *, content_type: str, size: int | None = None) -> None:
        buf = bytearray()
        async for chunk in body:
            buf += chunk
        self.blobs[key] = bytes(buf)
        self.content_types[key] = content_type

    async def _read(self, key: str, *, chunk: int) -> AsyncIterator[bytes]:
        data = self.blobs[key]
        for i in range(0, len(data), chunk):
            yield data[i : i + chunk]

    def open_read(self, key: str, *, chunk: int = 7) -> AsyncIterator[bytes]:
        # a deliberately awkward chunk size (7) so the reader can't assume frame-aligned reads.
        return self._read(key, chunk=chunk)

    async def delete(self, key: str) -> None:
        self.deleted.append(key)
        self.blobs.pop(key, None)

    async def delete_many(self, keys: list[str]) -> None:
        for key in keys:
            await self.delete(key)

    async def list_keys(self, prefix: str | None = None) -> AsyncIterator[str]:
        for key in self.blobs:
            if prefix is None or key.startswith(prefix):
                yield key

    async def list_entries(self, prefix: str | None = None) -> AsyncIterator[ObjectListing]:
        from datetime import UTC, datetime

        for key, blob in self.blobs.items():
            if prefix is None or key.startswith(prefix):
                yield ObjectListing(key=key, last_modified=datetime.now(UTC), size_bytes=len(blob))

    async def presigned_get_url(self, key: str, *, expires_in: int = 300) -> str:
        self.presign_calls.append((key, expires_in))
        return f"https://example.test/{key}?exp={expires_in}"


async def _emit(data: bytes, *, chunk: int = 5) -> AsyncIterator[bytes]:
    for i in range(0, len(data), chunk):
        yield data[i : i + chunk]


async def _collect(stream: AsyncIterator[bytes]) -> bytes:
    buf = bytearray()
    async for chunk in stream:
        buf += chunk
    return bytes(buf)


def _store(inner: _MemStore, *, frame_size: int = 16) -> EncryptedObjectStore:
    return EncryptedObjectStore(inner, _PASS, frame_size=frame_size, scrypt_n=_FAST_N)


@pytest.mark.parametrize(
    "payload",
    [
        b"",  # empty — still one (final) frame
        b"short",  # under one frame
        b"exactly-16-bytes",  # exactly one frame boundary (16)
        b"a" * 16 * 4,  # four full frames, empty tail
        b"b" * (16 * 3 + 5),  # three frames + a partial final
        bytes(range(256)) * 10,  # binary, multi-frame
    ],
)
@pytest.mark.asyncio
async def test_round_trip_preserves_bytes(payload: bytes) -> None:
    inner = _MemStore()
    store = _store(inner)

    await store.put("k", _emit(payload), content_type="application/x-thing")
    got = await _collect(store.open_read("k"))

    assert got == payload


@pytest.mark.asyncio
async def test_stored_bytes_are_ciphertext_not_plaintext() -> None:
    inner = _MemStore()
    store = _store(inner)
    payload = b"the operator seed must never land in the clear" * 10

    await store.put("k", _emit(payload), content_type="text/plain")

    stored = inner.blobs["k"]
    assert payload not in stored
    assert stored.startswith(b"3TB1")  # magic header
    assert inner.content_types["k"] == "application/octet-stream"  # not the caller's type


@pytest.mark.asyncio
async def test_wrong_passphrase_fails() -> None:
    inner = _MemStore()
    await _store(inner).put("k", _emit(b"secret payload"), content_type="text/plain")

    wrong = EncryptedObjectStore(inner, SecretStr("wrong pass"), frame_size=16, scrypt_n=_FAST_N)
    with pytest.raises(InvalidTag):
        await _collect(wrong.open_read("k"))


@pytest.mark.asyncio
async def test_tampered_ciphertext_is_rejected() -> None:
    inner = _MemStore()
    store = _store(inner)
    await store.put("k", _emit(b"x" * 40), content_type="text/plain")

    blob = bytearray(inner.blobs["k"])
    blob[-1] ^= 0x01  # flip a bit in the last frame's tag
    inner.blobs["k"] = bytes(blob)

    with pytest.raises(InvalidTag):
        await _collect(store.open_read("k"))


@pytest.mark.asyncio
async def test_flipped_final_flag_is_rejected() -> None:
    # the final flag is authenticated via AAD — flipping a non-final frame to "final"
    # (to truncate) must trip the tag rather than silently return a short read.
    inner = _MemStore()
    store = _store(inner)
    await store.put("k", _emit(b"y" * 40), content_type="text/plain")  # 3 frames (16,16,8)

    blob = bytearray(inner.blobs["k"])
    # first frame header is right after MAGIC(4)+salt(16) = offset 20; byte 20 is its final-flag.
    blob[20] = 1
    inner.blobs["k"] = bytes(blob)

    with pytest.raises(InvalidTag):
        await _collect(store.open_read("k"))


@pytest.mark.asyncio
async def test_truncated_stream_is_rejected() -> None:
    inner = _MemStore()
    store = _store(inner)
    await store.put("k", _emit(b"z" * 40), content_type="text/plain")

    inner.blobs["k"] = inner.blobs["k"][:-10]  # lop off part of the final frame

    with pytest.raises(ValueError, match="truncated"):
        await _collect(store.open_read("k"))


@pytest.mark.asyncio
async def test_bad_magic_is_rejected() -> None:
    inner = _MemStore()
    inner.blobs["k"] = b"NOPE" + b"\x00" * 16 + b"garbage"
    store = _store(inner)

    with pytest.raises(ValueError, match="magic"):
        await _collect(store.open_read("k"))


@pytest.mark.asyncio
async def test_delete_and_list_pass_through() -> None:
    inner = _MemStore()
    store = _store(inner)
    await store.put("a/1", _emit(b"one"), content_type="text/plain")
    await store.put("a/2", _emit(b"two"), content_type="text/plain")
    await store.put("b/3", _emit(b"three"), content_type="text/plain")

    keys = {k async for k in store.list_keys("a/")}
    assert keys == {"a/1", "a/2"}

    entries = [e async for e in store.list_entries("a/")]
    assert {e.key for e in entries} == {"a/1", "a/2"}

    await store.delete("a/1")
    await store.delete_many(["a/2", "b/3"])
    assert inner.deleted == ["a/1", "a/2", "b/3"]


@pytest.mark.asyncio
async def test_presigned_url_passes_through() -> None:
    inner = _MemStore()
    store = _store(inner)
    await store.put("k", _emit(b"data"), content_type="text/plain")

    url = await store.presigned_get_url("k", expires_in=60)
    assert url == "https://example.test/k?exp=60"
    assert inner.presign_calls == [("k", 60)]


@pytest.mark.asyncio
async def test_rejects_nonpositive_frame_size() -> None:
    with pytest.raises(ValueError, match="frame_size"):
        EncryptedObjectStore(_MemStore(), _PASS, frame_size=0)
