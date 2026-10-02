"""regression: bind capture-back re-creating a previously deleted path.

before the fix, capture-back derived the next journal version from
the head cache (or a naive counter). when capture-back recorded a delete
it removed the head row but left journal history intact; a subsequent
re-create of the same path then tried to insert at version 1, colliding
with the still-present first create on the
``(workspace_id, relative_path, version)`` unique index.

the fix routes new-file version derivation through
:func:`_next_journal_version`, which scans the journal. this test locks
that contract in: delete-then-recreate of the same path inside two
sequential bind windows must emit monotonically increasing version
numbers.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest

from threetears.agent.workspace.bind_policy import BindConflictPolicy
from threetears.agent.workspace.materialize import bind
from packages.agent.workspace.tests._helpers.asyncpg_shims import (
    FakeAsyncpgAcquireCM,
    FakeAsyncpgConnection,
    FakeAsyncpgPool,
    FakeAsyncpgTransaction,
)
from packages.agent.workspace.tests._helpers.scripted_watch import ScriptedWatch
from packages.agent.workspace.tests._helpers.workspace_shims import (
    FakeWorkspaceCollection,
    FakeWorkspaceEntity,
    FakeWorkspaceFile,
    FakeWorkspaceFileCollection,
    FakeWorkspaceFileLease,
    FakeWorkspaceFileLeaseHandle,
    FakeWorkspaceFileVersionCollection,
    FakeWorkspaceSandbox,
)


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
class _FakeWorkspace(FakeWorkspaceEntity):
    """minimal stand-in exposing the attrs bind and capture-back read."""

    @property
    def namespace_name(self) -> str:
        """canonical workspace namespace name (WS-ACL-06)."""
        return f"workspace.{self.id}"

    id: UUID
    name: str = "ws_test"
    agent_id: UUID = field(default_factory=uuid4)
    owner_agent_id: UUID = field(default_factory=uuid4)
    date_deleted: None = None


@dataclass
class _FakeFile(FakeWorkspaceFile):
    """head-state file row exposing the fields bind reads."""

    relative_path: str
    content: bytes
    sha256: str
    version: int


class _FakeWorkspaceCollection(FakeWorkspaceCollection):
    """serves the one workspace by ``(agent_id, workspace_id)``."""

    def __init__(self, workspace: _FakeWorkspace) -> None:
        self.workspace = workspace

    async def find_by_id(self, agent_id: UUID, workspace_id: UUID) -> _FakeWorkspace | None:
        found = self.workspace.agent_id == agent_id and self.workspace.id == workspace_id
        return self.workspace if found else None


class _FakeFileCollection(FakeWorkspaceFileCollection):
    """the head-state rows L3 holds, set by the test between windows."""

    def __init__(self, files: list[_FakeFile]) -> None:
        self.files = files

    async def find_by_workspace(self, workspace_id: UUID) -> list[_FakeFile]:
        del workspace_id
        return list(self.files)


class _FakeVersionCollection(FakeWorkspaceFileVersionCollection):
    """unused by bind; capture-back writes the journal through the pool."""


class _FakeSandbox(FakeWorkspaceSandbox):
    """resolves the bind root to ``root / name``."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def resolve_fs_path(self, path: str, root_name: str) -> Path:
        del root_name
        return self.root / path


class _FakeLeaseHandle(FakeWorkspaceFileLeaseHandle):
    """an uncontended lease."""

    async def __aenter__(self) -> _FakeLeaseHandle:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        return None


class _FakeLease(FakeWorkspaceFileLease):
    """grants every acquire at once."""

    async def acquire(
        self,
        workspace_id: UUID,
        relative_path: str,
        ttl_seconds: int = 30,
        max_wait_seconds: int = 60,
    ) -> _FakeLeaseHandle:
        del workspace_id, relative_path, ttl_seconds, max_wait_seconds
        return _FakeLeaseHandle()


@dataclass
class _FakeTransaction(FakeAsyncpgTransaction):
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
class _FakeConnection(FakeAsyncpgConnection):
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

    def transaction(self, namespace: Any = None) -> _FakeTransaction:
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

        all other SELECTs (head lookups) find no row.
        """
        self.fetchrows.append((query, args, self.transaction_open))
        result: dict[str, Any] | None
        if "COALESCE(MAX(version)" in query:
            workspace_id: UUID = args[0]
            relative_path: str = args[1]
            result = {"max_version": self.journal_max_by_path.get((workspace_id, relative_path), 0)}
        else:
            result = None
        return result


@dataclass
class _FakeAcquireCM(FakeAsyncpgAcquireCM):
    """async-context-manager wrapper returning the configured connection."""

    conn: _FakeConnection

    async def __aenter__(self) -> _FakeConnection:
        return self.conn

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        return None


@dataclass
class _FakePool(FakeAsyncpgPool):
    """fake asyncpg pool dispatching every acquire to a single connection."""

    conn: _FakeConnection = field(default_factory=_FakeConnection)

    def acquire(self) -> _FakeAcquireCM:
        return _FakeAcquireCM(conn=self.conn)


# ---------------------------------------------------------------------------
# regression test
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _window(
    ws: _FakeWorkspace,
    files: _FakeFileCollection,
    pool: _FakePool,
    bind_root: Path,
) -> AsyncIterator[Path]:
    """one L3_WINS bind window over the fakes, with an idle watcher.

    :param ws: the workspace
    :ptype ws: _FakeWorkspace
    :param files: the head-state rows L3 holds as the window opens
    :ptype files: _FakeFileCollection
    :param pool: the pool capture-back writes through
    :ptype pool: _FakePool
    :param bind_root: the sandbox root the window binds under
    :ptype bind_root: Path
    :return: the bound disk root
    :rtype: AsyncIterator[Path]
    """
    async with bind(
        agent_id=ws.agent_id,
        workspace_id=ws.id,
        sandbox=_FakeSandbox(bind_root),  # type: ignore[arg-type]
        lease=_FakeLease(),  # type: ignore[arg-type]
        workspace_collection=_FakeWorkspaceCollection(ws),  # type: ignore[arg-type]
        workspace_file_collection=files,  # type: ignore[arg-type]
        workspace_file_version_collection=_FakeVersionCollection(),  # type: ignore[arg-type]
        db_pool=pool,
        actor_id=uuid4(),
        correlation_id=uuid4(),
        on_conflict=BindConflictPolicy.L3_WINS,
        watch_changes=ScriptedWatch(),
    ) as disk_root:
        yield disk_root


def _journal(pool: _FakePool) -> list[tuple[Any, ...]]:
    """every journal insert's arguments, in order."""
    return [args for query, args, _ in pool.conn.executions if "INSERT INTO workspace_file_versions" in query]


async def test_capture_back_delete_then_recreate_emits_monotonic_versions(
    tmp_path: Path,
) -> None:
    """delete-then-recreate of same path: second create must skip collided versions.

    seeds ``a.txt`` at journal version 1, then:

    1. opens a bind window, deletes the file on disk, closes it ->
       capture-back emits a delete at version 2, advancing per-path max
       to 2.
    2. opens a second window over the post-delete L3 state (no head row
       for the path), writes the file back, closes it -> the create must
       emit version 3 (not 1), proving journal-derived version wins over
       a raw counter.
    """
    ws = _FakeWorkspace(id=uuid4())
    bind_root = tmp_path / "bind_root"
    bind_root.mkdir()

    pool = _FakePool()
    # seed journal to reflect an already-present version 1 row for a.txt;
    # the journal-max cache emulates what L3 returns on the COALESCE query.
    pool.conn.journal_max_by_path[(ws.id, "a.txt")] = 1
    files = _FakeFileCollection([_FakeFile("a.txt", b"alpha", _sha256(b"alpha"), 1)])

    # phase 1: the window projects a.txt onto disk; the body deletes it.
    async with _window(ws, files, pool, bind_root) as disk_root:
        (disk_root / "a.txt").unlink()

    # walk the journal inserts and assert the delete landed at version 2.
    delete_inserts = _journal(pool)
    assert len(delete_inserts) == 1
    # column positions per _INSERT_WORKSPACE_FILE_VERSION_SQL:
    # id, workspace_id, relative_path, version, content, sha256, action
    assert delete_inserts[0][2] == "a.txt"
    assert delete_inserts[0][3] == 2
    assert delete_inserts[0][6] == "delete"

    # phase 2: head is now empty (delete landed), journal max for this
    # path is 2. the body restores the file on disk.
    files.files = []
    async with _window(ws, files, pool, bind_root) as disk_root:
        (disk_root / "a.txt").write_bytes(b"alpha-reborn")

    all_inserts = _journal(pool)
    # two INSERTs total: the delete at v2, then the re-create at v3.
    assert len(all_inserts) == 2
    recreate_args = all_inserts[1]
    assert recreate_args[2] == "a.txt"
    # critical regression guard: re-create must land at v3, not v1.
    assert recreate_args[3] == 3
    assert recreate_args[6] == "create"
    # final per-path journal max reflects both inserts.
    assert pool.conn.journal_max_by_path[(ws.id, "a.txt")] == 3
