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
        monkeypatch.setenv("THREETEARS_L1_DIR", str(tmp_path))
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


_COLLECTED_WHILE_OPENING = textwrap.dedent(
    """
    import gc, sys
    from pathlib import Path
    from sqlalchemy import Column, MetaData, String, Table
    from threetears.core.cache import l1_file
    from threetears.core.cache.l1_lru import L1Ledger
    from threetears.core.cache.sqlite import SQLiteBackend
    from threetears.core.testing.l1_tables import L1_EVICTION_INFO, TEST_L1_TABLE
    directory = Path(sys.argv[1])

    def backend(name):
        metadata = MetaData()
        Table("files_rows", metadata, Column("id", String, primary_key=True), info={L1_EVICTION_INFO: TEST_L1_TABLE})
        made = SQLiteBackend(db_name=name, directory=directory, budget=L1Ledger(1 << 30, thread=False))
        made.initialize(metadata)
        return made

    dropped = backend("dropped")
    dropped.itself = dropped  # a reference cycle: only a collection frees it
    del dropped
    # a collection runs wherever an allocation sets it off: here, inside the next open's hold of the files' lock
    owner_path = l1_file._owner_path
    def collecting(path):
        gc.collect()
        return owner_path(path)
    l1_file._owner_path = collecting
    opened = backend("opened")
    l1_file._owner_path = owner_path
    print("opened", flush=True)
    opened.reset()
    names = sorted(entry.name for entry in directory.iterdir() if entry.name.startswith("threetears-l1-"))
    print("left", names, flush=True)
    """
)


class TestABackendCollectedWhileAnotherOpens:
    def test_the_open_finishes_and_the_collected_file_goes_at_the_next_release(self, tmp_path: Path) -> None:
        # a backend dropped without reset gives its file back from its finalizer, which a collection runs on
        # whichever thread allocated: run inside an open's hold of the files' lock, the finalizer waited on that
        # lock, held by its own thread, for ever (a unit run's test sat 600 s, until pytest-timeout's alarm was raised
        # in the finalizer and swallowed there; 2026-11-16)
        child = subprocess.run(
            [sys.executable, "-c", _COLLECTED_WHILE_OPENING, str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert child.returncode == 0, child.stderr
        assert child.stdout.splitlines() == ["opened", "left []"], child.stdout + child.stderr


#: a finalizer that makes its thread's connection to a backend (a collected FeatureCache's R-Tree drop), run by a
#: collection inside the backend's reset, on a thread with no connection of its own, as the reset closes the others
_COLLECTED_WHILE_RESETTING = textwrap.dedent(
    """
    import gc, sys, threading, weakref
    from pathlib import Path
    from sqlalchemy import MetaData
    from threetears.core.cache.l1_lru import L1Ledger
    from threetears.core.cache.rtree import SQLiteRTree
    from threetears.core.cache.sqlite import SQLiteBackend
    gc.disable()  # the one collection is the reset's
    backend = SQLiteBackend(db_name="pool", directory=Path(sys.argv[1]), budget=L1Ledger(1 << 30, thread=False))
    backend.initialize(MetaData())
    tree = SQLiteRTree(backend, "owned_rtree", "owned_map")
    tree.create()

    class Owner:
        pass

    owner = Owner()
    owner.itself = owner  # a reference cycle: only a collection frees it
    weakref.finalize(owner, tree.drop)  # as a FeatureCache's finalizer drops its R-Tree
    del owner

    class Collecting:
        def close(self):
            gc.collect()

    backend._pooled_connections = (*backend._pooled_connections, Collecting())
    resetting = threading.Thread(target=backend.reset, daemon=True)
    resetting.start()
    resetting.join(20)
    print("reset" if not resetting.is_alive() else "hung", flush=True)
    """
)


class TestAFinalizerConnectingDuringAReset:
    def test_the_reset_finishes(self, tmp_path: Path) -> None:
        # a finalizer runs inside whatever its collection interrupted: one making its thread's connection inside the
        # reset's hold of the backend's pool lock waited on that lock, held by its own thread, for ever (2026-11-16)
        child = subprocess.run(
            [sys.executable, "-c", _COLLECTED_WHILE_RESETTING, str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert child.returncode == 0, child.stderr
        assert child.stdout.splitlines() == ["reset"], child.stdout + child.stderr


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
