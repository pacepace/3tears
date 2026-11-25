"""a write the SQLite L3 acknowledged is in the file after its process is killed with SIGKILL.

For each kind of write -- a collection's insert, update and delete, the store's many-row write and a transaction --
a child process (``sqlite_l3_child.py``) makes the write, prints that it returned, and is killed with SIGKILL: no
``close()``, no exit handler, nothing after the acknowledgement. This process then opens the file and finds the write.
While the child lives, its hold on the file refuses a second opener; its death gives the hold back.
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

from packages.core.tests.unit.backends.sqlite_l3_child import deployment
from threetears.core.backends import SqliteL3Backend
from threetears.core.backends.sqlite_file import SqliteStoreLockedError

_CHILD = Path(__file__).with_name("sqlite_l3_child.py")
#: how long the child may take to start, open the file and write
_ACK_SECONDS = 60


async def _killed_after(path: Path, kind: str, key: uuid.UUID) -> None:
    """run the child until it acknowledges ``kind``, check its hold refuses this process, then SIGKILL it."""
    child = await asyncio.create_subprocess_exec(
        sys.executable,
        str(_CHILD),
        str(path),
        kind,
        str(key),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        assert child.stdout is not None
        line = await asyncio.wait_for(child.stdout.readline(), timeout=_ACK_SECONDS)
        if line.decode().strip() != f"acked {kind}":
            assert child.stderr is not None
            raise AssertionError(f"the child did not acknowledge {kind}: {line!r} {await child.stderr.read()!r}")
        with pytest.raises(SqliteStoreLockedError):
            SqliteL3Backend(path).open()
    finally:
        if child.returncode is None:
            os.kill(child.pid, signal.SIGKILL)
        await child.wait()
    assert child.returncode == -signal.SIGKILL


async def _seed(path: Path, key: uuid.UUID) -> None:
    store, ledger = deployment(path)
    now = datetime.now(UTC)
    try:
        await ledger.save_entity(
            ledger.create({"id": key, "state": "seeded", "date_created": now, "date_updated": now})
        )
    finally:
        store.close()


async def _stored(path: Path, key: uuid.UUID) -> dict[str, object] | None:
    store, _ = deployment(path)
    try:
        return await store.fetch_one("ledger", {"id": key})
    finally:
        store.close()


async def test_an_insert_survives(tmp_path: Path) -> None:
    path, key = tmp_path / "state.sqlite", uuid.uuid4()
    await _killed_after(path, "insert", key)
    row = await _stored(path, key)
    assert row is not None
    assert row["state"] == "new"


async def test_an_update_survives(tmp_path: Path) -> None:
    path, key = tmp_path / "state.sqlite", uuid.uuid4()
    await _seed(path, key)
    await _killed_after(path, "update", key)
    row = await _stored(path, key)
    assert row is not None
    assert row["state"] == "updated"


async def test_a_delete_survives(tmp_path: Path) -> None:
    path, key = tmp_path / "state.sqlite", uuid.uuid4()
    await _seed(path, key)
    await _killed_after(path, "delete", key)
    assert await _stored(path, key) is None


async def test_a_many_row_write_survives_whole(tmp_path: Path) -> None:
    path, key = tmp_path / "state.sqlite", uuid.uuid4()
    await _killed_after(path, "many", key)
    store, _ = deployment(path)
    try:
        assert len(await store.scan("ledger", {"state": "many"})) == 100
    finally:
        store.close()


async def test_a_committed_transaction_survives_whole(tmp_path: Path) -> None:
    path, key = tmp_path / "state.sqlite", uuid.uuid4()
    await _killed_after(path, "transaction", key)
    store, _ = deployment(path)
    try:
        assert len(await store.scan("ledger", {"state": "tx"})) == 10
    finally:
        store.close()
