"""Unit tests for :class:`FilesystemObjectStore` — the second ObjectStore driver.

No infra: a tmp dir is the whole backend. Covers round-trip, atomic publish, key-traversal
rejection, listing/deletion, and composition under :class:`EncryptedObjectStore` (the real path
the backup engine takes: encrypt → land on disk → decrypt).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from pydantic import SecretStr

from threetears.media.contracts import ObjectStore
from threetears.object_store.encrypted import EncryptedObjectStore
from threetears.object_store.filesystem import FilesystemObjectStore


def test_conforms_to_object_store_protocol(tmp_path: Path) -> None:
    # mypy verifies the signatures structurally; runtime_checkable verifies the method set.
    store: ObjectStore = FilesystemObjectStore(tmp_path)
    assert isinstance(store, ObjectStore)


async def _emit(data: bytes, *, chunk: int = 8) -> AsyncIterator[bytes]:
    for i in range(0, len(data), chunk):
        yield data[i : i + chunk]


async def _collect(stream: AsyncIterator[bytes]) -> bytes:
    buf = bytearray()
    async for part in stream:
        buf += part
    return bytes(buf)


@pytest.mark.asyncio
async def test_round_trip(tmp_path: Path) -> None:
    store = FilesystemObjectStore(tmp_path)
    payload = bytes(range(256)) * 20

    await store.put("a/b/thing.bin", _emit(payload), content_type="application/octet-stream")
    got = await _collect(store.open_read("a/b/thing.bin"))

    assert got == payload
    assert (tmp_path / "a" / "b" / "thing.bin").read_bytes() == payload


@pytest.mark.asyncio
async def test_put_creates_root_and_is_atomic(tmp_path: Path) -> None:
    # root doesn't exist yet; put must create it, and leave no visible .tmp behind.
    store = FilesystemObjectStore(tmp_path / "fresh")
    await store.put("k", _emit(b"data"), content_type="text/plain")

    keys = [k async for k in store.list_keys()]
    assert keys == ["k"]  # no temp artifact surfaced


@pytest.mark.asyncio
async def test_list_entries_carries_size(tmp_path: Path) -> None:
    store = FilesystemObjectStore(tmp_path)
    await store.put("x/1", _emit(b"abc"), content_type="text/plain")
    await store.put("x/2", _emit(b"abcdef"), content_type="text/plain")
    await store.put("y/3", _emit(b"z"), content_type="text/plain")

    entries = {e.key: e.size_bytes async for e in store.list_entries("x/")}
    assert entries == {"x/1": 3, "x/2": 6}


@pytest.mark.asyncio
async def test_delete_and_delete_many(tmp_path: Path) -> None:
    store = FilesystemObjectStore(tmp_path)
    await store.put("a", _emit(b"1"), content_type="text/plain")
    await store.put("b", _emit(b"2"), content_type="text/plain")
    await store.put("c", _emit(b"3"), content_type="text/plain")

    await store.delete("a")
    await store.delete("missing")  # no error
    await store.delete_many(["b", "c"])

    assert [k async for k in store.list_keys()] == []


@pytest.mark.asyncio
async def test_key_traversal_is_rejected(tmp_path: Path) -> None:
    store = FilesystemObjectStore(tmp_path / "root")
    with pytest.raises(ValueError, match="escapes"):
        await store.put("../escape", _emit(b"nope"), content_type="text/plain")


@pytest.mark.asyncio
async def test_presigned_url_is_file_uri(tmp_path: Path) -> None:
    store = FilesystemObjectStore(tmp_path)
    await store.put("k", _emit(b"data"), content_type="text/plain")

    url = await store.presigned_get_url("k")
    assert url.startswith("file://")
    assert url.endswith("/k")


@pytest.mark.asyncio
async def test_encrypted_over_filesystem_end_to_end(tmp_path: Path) -> None:
    # the backup path: EncryptedObjectStore streaming AEAD onto a real on-disk store.
    inner = FilesystemObjectStore(tmp_path)
    store = EncryptedObjectStore(inner, SecretStr("pw"), frame_size=64, scrypt_n=2**8)
    payload = b"backup-dump-contents\n" * 500

    await store.put("backups/2026/07/13/db.sql.gz.enc", _emit(payload, chunk=100), content_type="application/sql")
    on_disk = (tmp_path / "backups/2026/07/13/db.sql.gz.enc").read_bytes()
    got = await _collect(store.open_read("backups/2026/07/13/db.sql.gz.enc"))

    assert got == payload
    assert payload not in on_disk  # what landed on disk is ciphertext
    assert on_disk.startswith(b"3TB1")


@pytest.mark.asyncio
async def test_failed_put_leaves_no_orphan_temp_file(tmp_path: Path) -> None:
    store = FilesystemObjectStore(tmp_path)

    async def _boom() -> AsyncIterator[bytes]:
        yield b"partial-write"
        raise RuntimeError("dump failed mid-stream")

    with pytest.raises(RuntimeError, match="dump failed"):
        await store.put("k", _boom(), content_type="text/plain")

    leftovers = [p.name for p in tmp_path.rglob("*") if p.is_file()]
    assert leftovers == []  # neither the object nor a dangling .tmp remains
