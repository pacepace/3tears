"""tests for threetears.core.utils.atomic_write.

covers round-trip, interrupted-write cleanup, parent-directory fsync,
and bytes-vs-str input parity.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from threetears.core.utils.atomic_write import atomic_write


class TestAtomicWriteRoundTrip:
    """round-trip content written and read back matches input exactly."""

    async def test_writes_bytes_content(self, tmp_path: Path) -> None:
        target = tmp_path / "out.bin"
        payload = b"\x00\x01\x02hello\xff"
        await atomic_write(target, payload)
        assert target.read_bytes() == payload

    async def test_writes_str_content_as_utf8(self, tmp_path: Path) -> None:
        target = tmp_path / "out.txt"
        payload = "hello world \u00e9"
        await atomic_write(target, payload)
        assert target.read_bytes() == payload.encode("utf-8")

    async def test_bytes_and_str_parity_on_ascii(self, tmp_path: Path) -> None:
        target_bytes = tmp_path / "b.txt"
        target_str = tmp_path / "s.txt"
        text = "ascii-only payload"
        await atomic_write(target_bytes, text.encode("utf-8"))
        await atomic_write(target_str, text)
        assert target_bytes.read_bytes() == target_str.read_bytes()

    async def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        target = tmp_path / "existing.txt"
        target.write_bytes(b"old contents")
        await atomic_write(target, b"new contents")
        assert target.read_bytes() == b"new contents"


class TestAtomicWriteInterrupted:
    """interrupted rename leaves original intact and cleans up temp file."""

    async def test_rename_failure_preserves_original(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        target = tmp_path / "precious.txt"
        target.write_bytes(b"original-do-not-touch")

        def _boom(src: str, dst: str) -> None:
            raise OSError("simulated rename failure")

        monkeypatch.setattr(os, "rename", _boom)

        with pytest.raises(OSError, match="simulated rename failure"):
            await atomic_write(target, b"new-content-that-never-lands")

        assert target.read_bytes() == b"original-do-not-touch"

    async def test_rename_failure_cleans_up_temp_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        target = tmp_path / "target.txt"

        def _boom(src: str, dst: str) -> None:
            raise OSError("simulated rename failure")

        monkeypatch.setattr(os, "rename", _boom)

        with pytest.raises(OSError):
            await atomic_write(target, b"payload")

        leftover = [p for p in tmp_path.iterdir() if ".tmp." in p.name]
        assert leftover == [], f"temp files not cleaned up: {leftover}"
        assert not target.exists()


class TestAtomicWriteParentDirFsync:
    """parent directory fsync is performed to durably commit the rename."""

    async def test_file_and_directory_are_both_fsynced(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fsync_kinds: list[str] = []
        live_dir_fds: set[int] = set()
        original_fsync = os.fsync
        original_open = os.open
        original_close = os.close

        def _spy_open(path: Any, flags: int, mode: int = 0o777) -> int:
            fd = original_open(path, flags, mode)
            if flags & os.O_DIRECTORY:
                live_dir_fds.add(fd)
            return fd

        def _spy_close(fd: int) -> None:
            live_dir_fds.discard(fd)
            original_close(fd)

        def _spy_fsync(fd: int) -> None:
            fsync_kinds.append("dir" if fd in live_dir_fds else "file")
            original_fsync(fd)

        monkeypatch.setattr(os, "open", _spy_open)
        monkeypatch.setattr(os, "close", _spy_close)
        monkeypatch.setattr(os, "fsync", _spy_fsync)

        target = tmp_path / "durable.txt"
        await atomic_write(target, b"payload")

        assert len(fsync_kinds) >= 2, (
            f"expected at least two fsync calls (file + directory), got {fsync_kinds}"
        )
        assert "dir" in fsync_kinds, "parent directory fd was never fsynced"
        assert "file" in fsync_kinds, "file fd was never fsynced"
