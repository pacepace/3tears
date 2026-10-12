"""atomic file-write helper built on tmp + fsync + rename + dir-fsync.

writes content to sibling temporary file, fsyncs it, renames it into place,
and fsyncs parent directory so the rename itself is durable. interrupted
writes cannot corrupt pre-existing target file because rename is atomic on
POSIX when source and target live on same filesystem.

see https://pubs.opengroup.org/onlinepubs/9699919799/functions/rename.html
and https://lwn.net/Articles/457667/ for background on directory fsync.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from pathlib import Path
from uuid import uuid7

__all__ = [
    "atomic_write",
    "atomic_write_sync",
]


def _write_atomically(path: Path, payload: bytes, mode: int) -> None:
    """perform synchronous atomic write of payload to path.

    wrapped in :func:`asyncio.to_thread` by public async entrypoint so event
    loop is not blocked by fsync calls.

    :param path: destination filesystem path for final file
    :ptype path: Path
    :param payload: raw bytes to write
    :ptype payload: bytes
    :param mode: permission bits for the created file
    :ptype mode: int
    :return: None
    :rtype: None
    :raises OSError: if filesystem operation fails at any step
    """
    parent = path.parent
    tmp_path = parent / f"{path.name}.tmp.{uuid7().hex}"
    tmp_fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        try:
            os.write(tmp_fd, payload)
            os.fsync(tmp_fd)
        finally:
            os.close(tmp_fd)
        try:
            os.rename(tmp_path, path)
        except BaseException:
            # NOSILENT: cleanup of a temp file on a path that re-raises immediately below
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise
        dir_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except BaseException:
        # NOSILENT: cleanup of a temp file on a path that re-raises immediately below
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


async def atomic_write(path: Path, content: bytes | str, *, mode: int = 0o600) -> None:
    """write content to path atomically with fsync-rename-fsync sequence.

    coerces str content to UTF-8 bytes before writing. interrupted writes
    (between fsync and rename, or during rename) leave pre-existing target
    file untouched and clean up temporary file on best-effort basis.

    sync filesystem operations run in worker thread via
    :func:`asyncio.to_thread` so event loop remains responsive.

    :param path: destination filesystem path for final file
    :ptype path: Path
    :param content: raw bytes, or str encoded as UTF-8 before writing
    :ptype content: bytes | str
    :param mode: permission bits for the file. defaults to ``0o600`` -- the rename carries the
        TEMP file's mode onto the destination, including over an existing file, so a caller
        writing world- or group-readable content must say so explicitly rather than inherit
        whatever the target had
    :ptype mode: int
    :return: None
    :rtype: None
    :raises OSError: if filesystem operation fails at any step
    :raises UnicodeEncodeError: if str content cannot be encoded as UTF-8
    """
    payload = content.encode("utf-8") if isinstance(content, str) else content
    await asyncio.to_thread(_write_atomically, path, payload, mode)


def atomic_write_sync(path: Path, content: bytes | str, *, mode: int = 0o600) -> None:
    """write content to path atomically, BLOCKING the calling thread.

    same tmp + fsync + rename + dir-fsync sequence as :func:`atomic_write`, for callers that
    genuinely have no event loop to await on. prefer :func:`atomic_write` from async code --
    the fsync pair here is the slow part, and running it inline stalls every coroutine in the
    process.

    :param path: destination filesystem path for final file
    :ptype path: Path
    :param content: raw bytes, or str encoded as UTF-8 before writing
    :ptype content: bytes | str
    :param mode: permission bits for the file; see :func:`atomic_write`
    :ptype mode: int
    :return: None
    :rtype: None
    :raises OSError: if filesystem operation fails at any step
    :raises UnicodeEncodeError: if str content cannot be encoded as UTF-8
    """
    payload = content.encode("utf-8") if isinstance(content, str) else content
    _write_atomically(path, payload, mode)
