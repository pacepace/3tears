"""The SQLite L1 is one WAL file per process and name (threetears.core.cache.l1_file): where it goes, that it is shared
by name and goes with its last holder, and that no file outlives its process (a clean exit removes it; a killed
process's is swept by the next process's first L1, never a live one's)."""

from __future__ import annotations

import gc
import os
import signal
import subprocess
import sys
import textwrap
import time
import uuid
from pathlib import Path

import pytest
from sqlalchemy import Column, MetaData, String, Table

from threetears.core.cache import l1_file
from threetears.core.cache.l1_lru import L1Ledger
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.testing.l1_tables import L1_EVICTION_INFO, TEST_L1_TABLE


def _metadata() -> MetaData:
    metadata = MetaData()
    Table(
        "files_rows",
        metadata,
        Column("id", String, primary_key=True),
        Column("body", String),
        info={L1_EVICTION_INFO: TEST_L1_TABLE},
    )
    return metadata


def _backend(directory: Path, name: str | None = None) -> SQLiteBackend:
    backend = SQLiteBackend(
        db_name=name or f"files_{uuid.uuid4().hex[:8]}", directory=directory, budget=L1Ledger(1 << 30, thread=False)
    )
    backend.initialize(_metadata())
    return backend


def _ours(directory: Path) -> list[str]:
    return sorted(entry.name for entry in directory.iterdir() if entry.name.startswith("threetears-l1-"))


class TestTheFile:
    def test_it_is_a_wal_file_named_for_the_process_in_the_l1_directory(self, tmp_path: Path) -> None:
        backend = _backend(tmp_path, "named")
        path = backend.file_path
        assert path is not None and path.parent == tmp_path
        assert path.name.startswith(f"threetears-l1-{os.getpid()}-") and path.name.endswith("-named.sqlite")
        conn = backend.get_connection()
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 0
        backend.reset()

    def test_reset_removes_the_file_its_wal_and_its_lock(self, tmp_path: Path) -> None:
        backend = _backend(tmp_path)
        backend.upsert("files_rows", {"id": "a", "body": "x"})
        assert _ours(tmp_path)
        backend.reset()
        assert _ours(tmp_path) == []

    def test_backends_of_one_name_share_it_and_the_last_reset_removes_it(self, tmp_path: Path) -> None:
        first, second = _backend(tmp_path, "shared"), _backend(tmp_path, "shared")
        first.upsert("files_rows", {"id": "a", "body": "x"})
        assert second.select_by_id("files_rows", "a") == {"id": "a", "body": "x"}
        first.reset()
        assert second.select_by_id("files_rows", "a") == {"id": "a", "body": "x"}
        second.reset()
        assert _ours(tmp_path) == []

    def test_a_backend_dropped_without_reset_gives_its_file_back(self, tmp_path: Path) -> None:
        backend = _backend(tmp_path)
        backend.upsert("files_rows", {"id": "a", "body": "x"})
        del backend
        gc.collect()
        assert _ours(tmp_path) == []

    def test_the_directory_is_named_by_the_environment(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(l1_file.L1_DIR_ENV, str(tmp_path))
        assert l1_file.l1_directory() == tmp_path


_HOLD_AN_L1 = textwrap.dedent(
    """
    import sys, time
    from pathlib import Path
    from sqlalchemy import Column, MetaData, String, Table
    from threetears.core.cache.l1_lru import L1Ledger
    from threetears.core.cache.sqlite import SQLiteBackend
    from threetears.core.testing.l1_tables import L1_EVICTION_INFO, TEST_L1_TABLE
    metadata = MetaData()
    Table("files_rows", metadata, Column("id", String, primary_key=True), info={L1_EVICTION_INFO: TEST_L1_TABLE})
    backend = SQLiteBackend(db_name="child", directory=Path(sys.argv[1]), budget=L1Ledger(1 << 30, thread=False))
    backend.initialize(metadata)
    backend.upsert("files_rows", {"id": "a"})
    print("ready", flush=True)
    if sys.argv[2] == "wait":
        time.sleep(60)
    """
)


def _child(directory: Path, mode: str) -> subprocess.Popen[str]:
    child = subprocess.Popen(
        [sys.executable, "-c", _HOLD_AN_L1, str(directory), mode], stdout=subprocess.PIPE, text=True
    )
    assert child.stdout is not None and child.stdout.readline().strip() == "ready"
    return child


class TestNoFileOutlivesItsProcess:
    def test_a_clean_exit_removes_its_files(self, tmp_path: Path) -> None:
        child = _child(tmp_path, "exit")
        assert child.wait(30) == 0
        assert _ours(tmp_path) == []

    def test_a_killed_processes_files_are_swept_and_a_live_ones_kept(self, tmp_path: Path) -> None:
        killed, live = _child(tmp_path, "wait"), _child(tmp_path, "wait")
        try:
            killed.send_signal(signal.SIGKILL)
            killed.wait(30)
            left = _ours(tmp_path)
            assert any(name.startswith(f"threetears-l1-{killed.pid}-") for name in left), left
            assert l1_file.sweep(tmp_path) == 1
            after = _ours(tmp_path)
            assert not any(name.startswith(f"threetears-l1-{killed.pid}-") for name in after), after
            assert any(name.startswith(f"threetears-l1-{live.pid}-") for name in after), after
        finally:
            live.kill()
            live.wait(30)

    def test_the_first_l1_of_a_process_sweeps_its_directory(self, tmp_path: Path) -> None:
        killed = _child(tmp_path, "wait")
        killed.send_signal(signal.SIGKILL)
        killed.wait(30)
        # a new process's first L1 in the directory sweeps it
        child = _child(tmp_path, "exit")
        assert child.wait(30) == 0
        deadline = time.monotonic() + 5
        while _ours(tmp_path) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert _ours(tmp_path) == []
