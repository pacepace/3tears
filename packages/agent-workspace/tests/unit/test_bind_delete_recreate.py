"""regression: bind capture-back re-creating a previously deleted path.

before the fix, ``_capture_back`` derived the next journal version from
the head cache (or a naive counter). when capture-back recorded a delete
it removed the head row but left journal history intact; a subsequent
re-create of the same path then tried to insert at version 1, colliding
with the still-present first create on the
``(workspace_id, relative_path, version)`` unique index.

the fix routes new-file version derivation through
:func:`_next_journal_version`, which scans the journal. this test locks
that contract in: delete-then-recreate of the same path inside two
sequential capture-back windows must emit monotonically increasing
version numbers.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest

from threetears.agent.workspace.materialize import _capture_back


pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# fakes: a fake asyncpg connection that honours the journal-max dispatch
# ---------------------------------------------------------------------------


def _sha256(data: bytes) -> str:
    """
    compute sha256 hex digest of ``data`` for snapshot fixtures.

    :param data: bytes to hash
    :ptype data: bytes
    :return: lowercase hex sha256 digest
    :rtype: str
    """
    return hashlib.sha256(data).hexdigest()


@dataclass
class _FakeWorkspace:
    """minimal stand-in exposing only the ``id`` attribute ``_capture_back`` reads."""

    id: UUID


@dataclass
class _FakeTransaction:
    """records transaction enter/exit against the parent connection."""

    parent: _FakeConnection
    entered: bool = False
    exited: bool = False

    async def __aenter__(self) -> _FakeTransaction:
        self.entered = True
        self.parent.transaction_open = True
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.exited = True
        self.parent.transaction_open = False
        return None


@dataclass
class _FakeConnection:
    """fake asyncpg connection: dispatches fetchrow by SQL shape.

    tracks every INSERT into ``workspace_file_versions`` so the per-path
    max version is authoritative for subsequent
    ``SELECT COALESCE(MAX(version) ...)`` lookups, independent of any
    external fixture configuration.
    """

    executions: list[tuple[str, tuple[Any, ...], bool]] = field(default_factory=list)
    fetchrows: list[tuple[str, tuple[Any, ...], bool]] = field(default_factory=list)
    transactions: list[_FakeTransaction] = field(default_factory=list)
    transaction_open: bool = False
    journal_max_by_path: dict[tuple[UUID, str], int] = field(default_factory=dict)

    def transaction(self) -> _FakeTransaction:
        tx = _FakeTransaction(parent=self)
        self.transactions.append(tx)
        return tx

    async def execute(self, query: str, *args: Any) -> str:
        self.executions.append((query, args, self.transaction_open))
        if "INSERT INTO workspace_file_versions" in query:
            workspace_id: UUID = args[1]
            relative_path: str = args[2]
            inserted_version = int(args[3])
            key = (workspace_id, relative_path)
            prior = self.journal_max_by_path.get(key, 0)
            if inserted_version > prior:
                self.journal_max_by_path[key] = inserted_version
        return "INSERT 0 1"

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        """dispatch by SQL shape: journal-max SELECT returns per-path max.

        all other SELECTs (``_capture_back`` never issues any) return None.
        """
        self.fetchrows.append((query, args, self.transaction_open))
        result: dict[str, Any] | None
        if "COALESCE(MAX(version)" in query:
            workspace_id: UUID = args[0]
            relative_path: str = args[1]
            result = {
                "max_version": self.journal_max_by_path.get(
                    (workspace_id, relative_path), 0
                )
            }
        else:
            result = None
        return result


@dataclass
class _FakeAcquireCM:
    """async-context-manager wrapper returning the configured connection."""

    conn: _FakeConnection

    async def __aenter__(self) -> _FakeConnection:
        return self.conn

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        return None


@dataclass
class _FakePool:
    """fake asyncpg pool dispatching every acquire to a single connection."""

    conn: _FakeConnection = field(default_factory=_FakeConnection)

    def acquire(self) -> _FakeAcquireCM:
        return _FakeAcquireCM(conn=self.conn)


# ---------------------------------------------------------------------------
# regression test
# ---------------------------------------------------------------------------


async def test_capture_back_delete_then_recreate_emits_monotonic_versions(
    tmp_path: Path,
) -> None:
    """delete-then-recreate of same path: second create must skip collided versions.

    seeds ``a.txt`` at journal version 1, then:

    1. runs ``_capture_back`` with an empty disk root -> delete emits
       version 2, advancing per-path max to 2.
    2. runs ``_capture_back`` again with the file restored on disk
       (snapshot reports it as absent, mirroring post-delete L3 state) ->
       create must emit version 3 (not 1), proving journal-derived
       version wins over a raw counter.
    """
    ws = _FakeWorkspace(id=uuid4())
    disk_root = tmp_path / "bind_root"
    disk_root.mkdir()
    file_path = disk_root / "a.txt"

    pool = _FakePool()
    # seed journal to reflect an already-present version 1 row for a.txt.
    # capture-back does not touch this on the snapshot fetch because
    # snapshot is supplied directly by the caller; the journal-max cache
    # emulates what L3 would return on the COALESCE query.
    pool.conn.journal_max_by_path[(ws.id, "a.txt")] = 1

    initial_sha = _sha256(b"alpha")
    snapshot_v1: dict[str, tuple[str, int]] = {"a.txt": (initial_sha, 1)}

    # phase 1: disk is empty, snapshot had the file -> delete.
    changed_delete = await _capture_back(
        workspace=ws,
        disk_root=disk_root,
        snapshot=snapshot_v1,
        workspace_file_collection=None,  # type: ignore[arg-type]
        workspace_file_version_collection=None,  # type: ignore[arg-type]
        workspace_collection=None,  # type: ignore[arg-type]
        db_pool=pool,
        actor_id=uuid4(),
        correlation_id=uuid4(),
    )
    assert changed_delete == ["a.txt"]

    # walk the journal inserts and assert the delete landed at version 2.
    delete_inserts = [
        args
        for query, args, _ in pool.conn.executions
        if "INSERT INTO workspace_file_versions" in query
    ]
    assert len(delete_inserts) == 1
    # column positions per _INSERT_WORKSPACE_FILE_VERSION_SQL:
    # id, workspace_id, relative_path, version, content, sha256, action
    assert delete_inserts[0][2] == "a.txt"
    assert delete_inserts[0][3] == 2
    assert delete_inserts[0][6] == "delete"

    # phase 2: head is now empty (delete landed), journal max for this
    # path is 2. restore the file on disk; caller snapshot reflects the
    # post-delete empty head state.
    file_path.write_bytes(b"alpha-reborn")
    snapshot_v2: dict[str, tuple[str, int]] = {}

    changed_create = await _capture_back(
        workspace=ws,
        disk_root=disk_root,
        snapshot=snapshot_v2,
        workspace_file_collection=None,  # type: ignore[arg-type]
        workspace_file_version_collection=None,  # type: ignore[arg-type]
        workspace_collection=None,  # type: ignore[arg-type]
        db_pool=pool,
        actor_id=uuid4(),
        correlation_id=uuid4(),
    )
    assert changed_create == ["a.txt"]

    all_inserts = [
        args
        for query, args, _ in pool.conn.executions
        if "INSERT INTO workspace_file_versions" in query
    ]
    # two INSERTs total: the delete at v2, then the re-create at v3.
    assert len(all_inserts) == 2
    recreate_args = all_inserts[1]
    assert recreate_args[2] == "a.txt"
    # critical regression guard: re-create must land at v3, not v1.
    assert recreate_args[3] == 3
    assert recreate_args[6] == "create"
    # final per-path journal max reflects both inserts.
    assert pool.conn.journal_max_by_path[(ws.id, "a.txt")] == 3
