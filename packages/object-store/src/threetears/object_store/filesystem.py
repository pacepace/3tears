"""A local-filesystem :class:`ObjectStore` — the second driver behind the protocol.

S3 is the first driver; this one proves the contract is backend-agnostic and gives callers a
dependency-free store for local/dev/test and for durable on-disk artifacts (e.g. an encrypted
offsite copy written under :class:`EncryptedObjectStore`). Keys map to relative paths under a
root; writes are atomic (temp file + ``os.replace``) so a partial write is never listed or read.
Blocking file I/O is offloaded to a thread so the event loop keeps turning.

There is no server-side presign for a filesystem, so :meth:`presigned_get_url` returns a
``file://`` URL and ignores ``expires_in`` (there is nothing to time-limit).
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid7

from threetears.media.contracts import ObjectListing

__all__ = ["FilesystemObjectStore"]

_READ_CHUNK = 1 << 18  # 256 KiB


class FilesystemObjectStore:
    """An :class:`ObjectStore` backed by a directory tree.

    :param root: the directory under which every key resolves; created on demand.
    """

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root).resolve()

    def _path_for(self, key: str) -> Path:
        # keys are relative posix paths; reject anything that escapes the root (traversal / absolute).
        candidate = (self._root / key).resolve()
        if candidate != self._root and self._root not in candidate.parents:
            raise ValueError(f"key escapes the store root: {key!r}")
        return candidate

    async def put(self, key: str, body: AsyncIterator[bytes], *, content_type: str, size: int | None = None) -> None:
        path = self._path_for(key)
        await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{uuid7().hex}.tmp")
        handle = await asyncio.to_thread(open, tmp, "wb")
        try:
            async for chunk in body:
                await asyncio.to_thread(handle.write, chunk)
            await asyncio.to_thread(handle.close)
            # atomic publish: a reader/lister never sees a half-written object.
            await asyncio.to_thread(os.replace, tmp, path)
        except BaseException:
            # a failed write (body errored, cancelled) must not leave an orphan temp file behind.
            await asyncio.to_thread(handle.close)
            await asyncio.to_thread(tmp.unlink, missing_ok=True)
            raise

    async def open_read(self, key: str) -> AsyncIterator[bytes]:
        path = self._path_for(key)
        handle = await asyncio.to_thread(open, path, "rb")
        try:
            while True:
                chunk = await asyncio.to_thread(handle.read, _READ_CHUNK)
                if not chunk:
                    break
                yield chunk
        finally:
            await asyncio.to_thread(handle.close)

    async def delete(self, key: str) -> None:
        await asyncio.to_thread(self._path_for(key).unlink, missing_ok=True)

    async def delete_many(self, keys: list[str]) -> None:
        for key in keys:
            await self.delete(key)

    async def list_keys(self, prefix: str | None = None) -> AsyncIterator[str]:
        async for entry in self.list_entries(prefix):
            yield entry.key

    async def list_entries(self, prefix: str | None = None) -> AsyncIterator[ObjectListing]:
        # all blocking filesystem work (walk + stat) happens in one thread hop, off the event loop.
        for entry in await asyncio.to_thread(self._scan_entries, prefix):
            yield entry

    def _scan_entries(self, prefix: str | None) -> list[ObjectListing]:
        entries: list[ObjectListing] = []
        for path in sorted(self._root.rglob("*")):
            # temp files (partial writes) start with a dot and are never surfaced.
            if not path.is_file() or path.name.endswith(".tmp"):
                continue
            key = path.relative_to(self._root).as_posix()
            if prefix is not None and not key.startswith(prefix):
                continue
            stat = path.stat()
            entries.append(
                ObjectListing(
                    key=key,
                    last_modified=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
                    size_bytes=stat.st_size,
                )
            )
        return entries

    async def presigned_get_url(self, key: str, *, expires_in: int = 300) -> str:
        # a filesystem has no server-side presign; hand back a file:// URL (expires_in is moot).
        return self._path_for(key).as_uri()
