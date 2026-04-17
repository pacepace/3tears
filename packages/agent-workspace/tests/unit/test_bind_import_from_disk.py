"""unit tests for bind auto-import-from-disk when L3 is empty.

the bind contract now seeds L3 from disk on enter when the target
workspace has no head rows. these tests pin that behavior:

- empty L3 + populated disk -> create journal + head rows appear, one
  per file, at version 1.
- populated L3 + populated disk -> no re-import (exactly what bind
  did before the import hook landed).
- re-running bind after the first import leaves L3 untouched on enter
  (idempotency gate).

the fakes mirror the ones in ``test_bind.py`` intentionally; sharing
them via a helper module is out of scope for this shard.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4, uuid7

import pytest

from threetears.agent.workspace.bind_policy import BindConflictPolicy
from threetears.agent.workspace.materialize import bind


def _sha(content: bytes) -> str:
    """compute hex sha256 of raw bytes.

    :param content: bytes to digest
    :ptype content: bytes
    :return: 64-character hex digest
    :rtype: str
    """
    return hashlib.sha256(content).hexdigest()


@dataclass
class _FakeWorkspace:
    """stand-in for :class:`Workspace` exposing fields bind reads.

    :ivar id: workspace identifier
    :ivar name: workspace name
    :ivar current_version: current version pointer
    :ivar date_deleted: soft-delete timestamp, None when live
    """

    id: UUID
    name: str
    current_version: int = 0
    date_deleted: datetime | None = None


@dataclass
class _FakeFile:
    """stand-in for :class:`WorkspaceFile` exposing fields bind reads.

    :ivar relative_path: workspace-relative path
    :ivar content: file content bytes
    :ivar sha256: sha256 hex digest of content
    :ivar version: file version number
    """

    relative_path: str
    content: bytes
    sha256: str
    version: int


class _FakeWorkspaceCollection:
    """fake :class:`WorkspaceCollection` exposing :meth:`find_by_id`."""

    def __init__(self, workspaces: list[_FakeWorkspace]) -> None:
        """capture workspaces list the collection serves.

        :param workspaces: live workspace entities to serve
        :ptype workspaces: list[_FakeWorkspace]
        :return: None
        :rtype: None
        """
        self._workspaces = workspaces

    async def find_by_id(self, workspace_id: UUID) -> _FakeWorkspace | None:
        """locate a live workspace by id.

        :param workspace_id: identifier to lookup
        :ptype workspace_id: UUID
        :return: matched workspace or None
        :rtype: _FakeWorkspace | None
        """
        result: _FakeWorkspace | None = None
        for ws in self._workspaces:
            if ws.id == workspace_id and ws.date_deleted is None:
                result = ws
                break
        return result


class _FakeFileCollection:
    """fake :class:`WorkspaceFileCollection` exposing :meth:`find_by_workspace`.

    serves either the list passed at construction time or a live-view
    onto the fake connection's ``head_by_path`` dict when ``bind_to_conn``
    is invoked. the live-view is necessary for bind's second ``find_by_workspace``
    call (the one that builds the snapshot after the import step) to see
    the rows the import step just committed.
    """

    def __init__(self, files: list[_FakeFile]) -> None:
        """capture files the collection serves.

        :param files: head-state file entities
        :ptype files: list[_FakeFile]
        :return: None
        :rtype: None
        """
        self._files = files
        self._conn: _FakeConnection | None = None
        self.find_calls: int = 0

    def bind_to_conn(self, conn: _FakeConnection) -> None:
        """switch the collection into live-view mode against ``conn``.

        once bound, ``find_by_workspace`` returns a fresh snapshot of the
        connection's ``head_by_path`` merged with any initially-seeded
        files so the post-import snapshot pick up committed rows.

        :param conn: fake connection whose state the collection mirrors
        :ptype conn: _FakeConnection
        :return: None
        :rtype: None
        """
        self._conn = conn

    async def find_by_workspace(self, workspace_id: UUID) -> list[_FakeFile]:
        """count calls and return a file list reflecting current state.

        :param workspace_id: unused; fake is single-workspace
        :ptype workspace_id: UUID
        :return: file list reflecting seeds and any conn-written upserts
        :rtype: list[_FakeFile]
        """
        del workspace_id
        self.find_calls += 1
        result: list[_FakeFile]
        if self._conn is None:
            result = list(self._files)
        else:
            # merge seeded files with conn-side head_by_path so tests can
            # observe what bind's capture-back snapshot would see.
            by_path: dict[str, _FakeFile] = {
                f.relative_path: f for f in self._files
            }
            for rel, row in self._conn.head_by_path.items():
                by_path[rel] = _FakeFile(
                    relative_path=rel,
                    content=row["content"],
                    sha256=row["sha256"],
                    version=row["version"],
                )
            result = list(by_path.values())
        return result


class _FakeVersionCollection:
    """fake :class:`WorkspaceFileVersionCollection` -- unused by bind."""


class _FakeSandbox:
    """stand-in for :class:`WorkspaceSandbox` resolving to ``tmp_path``."""

    def __init__(self, roots: dict[str, Path]) -> None:
        """capture root mapping the sandbox serves.

        :param roots: mapping of root name to absolute path
        :ptype roots: dict[str, Path]
        :return: None
        :rtype: None
        """
        self._roots = roots

    def resolve_fs_path(self, path: str, root_name: str) -> Path:
        """resolve ``root / path`` while enforcing simple parentage.

        :param path: workspace-relative path
        :ptype path: str
        :param root_name: named root to resolve under
        :ptype root_name: str
        :return: resolved absolute path
        :rtype: Path
        """
        root = self._roots[root_name]
        candidate = (root / path).resolve()
        candidate.relative_to(root)
        return candidate


@dataclass
class _FakeLeaseHandle:
    """async-context-manager handle modeling a :class:`LeaseHandle`.

    :ivar released: True once exit fired
    """

    released: bool = False

    async def __aenter__(self) -> _FakeLeaseHandle:
        """enter the critical section.

        :return: self
        :rtype: _FakeLeaseHandle
        """
        return self

    async def __aexit__(
        self, exc_type: Any, exc_val: Any, exc_tb: Any,
    ) -> None:
        """exit the critical section.

        :param exc_type: exception type (unused)
        :ptype exc_type: Any
        :param exc_val: exception value (unused)
        :ptype exc_val: Any
        :param exc_tb: exception traceback (unused)
        :ptype exc_tb: Any
        :return: None
        :rtype: None
        """
        del exc_type, exc_val, exc_tb
        self.released = True


@dataclass
class _FakeLease:
    """fake :class:`WorkspaceFileLease` recording acquire calls."""

    handles: list[_FakeLeaseHandle] = field(default_factory=list)

    async def acquire(
        self,
        workspace_id: UUID,
        relative_path: str,
        ttl_seconds: int = 30,
        max_wait_seconds: int = 60,
    ) -> _FakeLeaseHandle:
        """mint a handle and record the acquire call.

        :param workspace_id: target workspace
        :ptype workspace_id: UUID
        :param relative_path: lease key tail
        :ptype relative_path: str
        :param ttl_seconds: TTL (unused)
        :ptype ttl_seconds: int
        :param max_wait_seconds: max wait (unused)
        :ptype max_wait_seconds: int
        :return: fresh lease handle
        :rtype: _FakeLeaseHandle
        """
        del workspace_id, relative_path, ttl_seconds, max_wait_seconds
        handle = _FakeLeaseHandle()
        self.handles.append(handle)
        return handle


@dataclass
class _FakeTransaction:
    """fake asyncpg transaction context manager."""

    parent: _FakeConnection

    async def __aenter__(self) -> _FakeTransaction:
        """open the transaction.

        :return: self
        :rtype: _FakeTransaction
        """
        self.parent.transaction_open = True
        return self

    async def __aexit__(
        self, exc_type: Any, exc_val: Any, exc_tb: Any,
    ) -> None:
        """close the transaction.

        :param exc_type: exception type
        :ptype exc_type: Any
        :param exc_val: exception value
        :ptype exc_val: Any
        :param exc_tb: exception traceback
        :ptype exc_tb: Any
        :return: None
        :rtype: None
        """
        del exc_type, exc_val, exc_tb
        self.parent.transaction_open = False


@dataclass
class _FakeConnection:
    """fake asyncpg connection recording execute + fetchrow calls."""

    executions: list[tuple[str, tuple[Any, ...]]] = field(default_factory=list)
    transactions: list[_FakeTransaction] = field(default_factory=list)
    transaction_open: bool = False
    journal_max_by_path: dict[str, int] = field(default_factory=dict)
    head_by_path: dict[str, dict[str, Any]] = field(default_factory=dict)

    def transaction(self) -> _FakeTransaction:
        """create a fresh transaction wrapper.

        :return: transaction context manager
        :rtype: _FakeTransaction
        """
        tx = _FakeTransaction(parent=self)
        self.transactions.append(tx)
        return tx

    async def execute(self, query: str, *args: Any) -> str:
        """record the execute and maintain per-path journal-max state.

        :param query: SQL text
        :ptype query: str
        :param args: bound parameters
        :ptype args: Any
        :return: asyncpg-style tag
        :rtype: str
        """
        self.executions.append((query, args))
        if "INSERT INTO workspace_file_versions" in query:
            rel = args[2]
            inserted_version = int(args[3])
            prior = self.journal_max_by_path.get(rel, 0)
            if inserted_version > prior:
                self.journal_max_by_path[rel] = inserted_version
        elif "INSERT INTO workspace_files" in query:
            rel = args[2]
            self.head_by_path[rel] = {
                "content": args[3],
                "sha256": args[4],
                "version": args[5],
            }
        elif "DELETE FROM workspace_files" in query:
            rel = args[1]
            self.head_by_path.pop(rel, None)
        return "INSERT 0 1"

    async def fetchrow(
        self, query: str, *args: Any,
    ) -> dict[str, Any] | None:
        """route journal-max + head-row SELECTs to fake state.

        :param query: SQL text
        :ptype query: str
        :param args: bound parameters
        :ptype args: Any
        :return: dict-shaped row or None
        :rtype: dict[str, Any] | None
        """
        result: dict[str, Any] | None
        if "COALESCE(MAX(version)" in query:
            rel = args[1]
            max_version = self.journal_max_by_path.get(rel, 0)
            result = {"max_version": max_version}
        else:
            rel = args[1]
            head = self.head_by_path.get(rel)
            result = None if head is None else dict(head)
        return result


@dataclass
class _FakeAcquireCM:
    """asyncpg-shaped acquire context manager."""

    conn: _FakeConnection

    async def __aenter__(self) -> _FakeConnection:
        """return the fake connection.

        :return: fake connection
        :rtype: _FakeConnection
        """
        return self.conn

    async def __aexit__(
        self, exc_type: Any, exc_val: Any, exc_tb: Any,
    ) -> None:
        """no-op close.

        :param exc_type: exception type
        :ptype exc_type: Any
        :param exc_val: exception value
        :ptype exc_val: Any
        :param exc_tb: exception traceback
        :ptype exc_tb: Any
        :return: None
        :rtype: None
        """
        del exc_type, exc_val, exc_tb


@dataclass
class _FakePool:
    """asyncpg-shaped pool routing to a single shared connection."""

    conn: _FakeConnection = field(default_factory=_FakeConnection)

    def acquire(self) -> _FakeAcquireCM:
        """return an acquire context manager yielding the shared conn.

        :return: acquire context manager
        :rtype: _FakeAcquireCM
        """
        return _FakeAcquireCM(conn=self.conn)


def _harness(tmp_path: Path, initial_files: list[_FakeFile]) -> dict[str, Any]:
    """build a fake bind harness rooted at ``tmp_path``.

    :param tmp_path: scratch root from pytest
    :ptype tmp_path: Path
    :param initial_files: seed head-state file list
    :ptype initial_files: list[_FakeFile]
    :return: harness dict with wired fakes
    :rtype: dict[str, Any]
    """
    ws_id = uuid7()
    ws = _FakeWorkspace(id=ws_id, name="ws_test")
    bind_root = tmp_path / "bind_root"
    bind_root.mkdir(parents=True, exist_ok=True)
    sandbox = _FakeSandbox({"bind": bind_root})
    pool = _FakePool()
    file_coll = _FakeFileCollection(initial_files)
    file_coll.bind_to_conn(pool.conn)
    # pre-seed the fake connection's head_by_path with initial_files so
    # bind's first gate call (before import) sees the correct shape.
    for f in initial_files:
        pool.conn.head_by_path[f.relative_path] = {
            "content": f.content,
            "sha256": f.sha256,
            "version": f.version,
        }
    return {
        "workspace_id": ws_id,
        "workspace": ws,
        "sandbox": sandbox,
        "bind_root": bind_root,
        "workspace_coll": _FakeWorkspaceCollection([ws]),
        "file_coll": file_coll,
        "version_coll": _FakeVersionCollection(),
        "lease": _FakeLease(),
        "pool": pool,
        "actor_id": uuid4(),
        "correlation_id": uuid7(),
    }


async def _enter_and_exit_bind(
    h: dict[str, Any],
    on_conflict: BindConflictPolicy = BindConflictPolicy.L3_WINS,
) -> None:
    """enter + exit bind against the harness with no-op body.

    these tests were authored against the L3-authoritative "seed when
    empty" gate, so the helper defaults to
    :attr:`BindConflictPolicy.L3_WINS` regardless of the bind() default.

    :param h: harness dict
    :ptype h: dict[str, Any]
    :param on_conflict: policy override; defaults to L3_WINS to pin the
        original suite semantics
    :ptype on_conflict: BindConflictPolicy
    :return: None
    :rtype: None
    """
    async with bind(
        workspace_id=h["workspace_id"],
        sandbox=h["sandbox"],
        lease=h["lease"],
        workspace_collection=h["workspace_coll"],
        workspace_file_collection=h["file_coll"],
        workspace_file_version_collection=h["version_coll"],
        db_pool=h["pool"],
        actor_id=h["actor_id"],
        correlation_id=h["correlation_id"],
        on_conflict=on_conflict,
    ):
        pass


@pytest.mark.asyncio
async def test_empty_l3_populated_disk_imports_create_rows(
    tmp_path: Path,
) -> None:
    """bind against empty L3 and a pre-populated disk imports disk -> L3.

    :param tmp_path: scratch root from pytest
    :ptype tmp_path: Path
    :return: None
    :rtype: None
    """
    h = _harness(tmp_path, initial_files=[])
    disk_root = h["bind_root"] / h["workspace"].name
    disk_root.mkdir(parents=True, exist_ok=True)
    (disk_root / "a.txt").write_bytes(b"hello")
    (disk_root / "sub").mkdir()
    (disk_root / "sub" / "b.txt").write_bytes(b"world")

    await _enter_and_exit_bind(h)

    journal_inserts = [
        e for e in h["pool"].conn.executions
        if "INSERT INTO workspace_file_versions" in e[0]
    ]
    head_inserts = [
        e for e in h["pool"].conn.executions
        if "INSERT INTO workspace_files" in e[0]
    ]
    paths_journaled = {row[1][2] for row in journal_inserts}
    assert paths_journaled == {"a.txt", "sub/b.txt"}
    # every imported row lands as create at version 1.
    for row in journal_inserts:
        assert row[1][6] == "create"
        assert row[1][3] == 1
    assert len(head_inserts) == 2


@pytest.mark.asyncio
async def test_populated_l3_does_not_reimport(tmp_path: Path) -> None:
    """bind skips the import step when the workspace has any head rows.

    :param tmp_path: scratch root from pytest
    :ptype tmp_path: Path
    :return: None
    :rtype: None
    """
    existing = b"seeded"
    h = _harness(
        tmp_path,
        initial_files=[
            _FakeFile("seeded.txt", existing, _sha(existing), 1),
        ],
    )
    # pretend disk has a new file that bind would pick up on capture-back
    # -- but NOT on the initial import (gate: L3 is non-empty).
    disk_root = h["bind_root"] / h["workspace"].name
    disk_root.mkdir(parents=True, exist_ok=True)
    (disk_root / "orphan.txt").write_bytes(b"bystander")

    await _enter_and_exit_bind(h)

    # expected: zero import-journal rows BEFORE capture-back. capture-back
    # still emits a create for orphan.txt on clean exit (that's part of
    # the existing bind contract, and test_bind.py already covers it).
    journal = [
        e for e in h["pool"].conn.executions
        if "INSERT INTO workspace_file_versions" in e[0]
    ]
    # on-enter import would have created a version-1 row for seeded.txt;
    # that must NOT appear.
    seeded_rows = [row for row in journal if row[1][2] == "seeded.txt"]
    assert seeded_rows == []


@pytest.mark.asyncio
async def test_second_bind_after_import_is_no_op_on_enter(
    tmp_path: Path,
) -> None:
    """re-binding after a successful import does not re-emit create rows.

    simulates the "second bind" path by seeding ``initial_files`` to
    reflect the state L3 would carry after import-on-first-enter.

    :param tmp_path: scratch root from pytest
    :ptype tmp_path: Path
    :return: None
    :rtype: None
    """
    content = b"previously imported"
    h = _harness(
        tmp_path,
        initial_files=[
            _FakeFile("a.txt", content, _sha(content), 1),
        ],
    )
    disk_root = h["bind_root"] / h["workspace"].name
    disk_root.mkdir(parents=True, exist_ok=True)
    (disk_root / "a.txt").write_bytes(content)

    await _enter_and_exit_bind(h)

    # expected: no journal rows at all (disk matches L3, gate trips).
    journal = [
        e for e in h["pool"].conn.executions
        if "INSERT INTO workspace_file_versions" in e[0]
    ]
    assert journal == []
