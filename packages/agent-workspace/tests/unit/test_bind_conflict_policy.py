"""unit tests for the bind L3-vs-disk conflict policy.

covers both ``BindConflictPolicy.L3_WINS`` (default) and
``BindConflictPolicy.DISK_WINS`` across the two surfaces where the
policy applies:

- seed-on-enter: :func:`_seed_l3_from_disk` via the public :func:`bind`
  context manager. exercises the noop-when-populated gate for L3_WINS
  and the full create/update/delete mirror for DISK_WINS.
- during-window live watcher: :func:`_handle_watch_batch` driven
  directly with synthesized :func:`watchfiles.awatch` batches. the
  production :func:`_watch_loop` forwards awatch's output into the
  same helper so direct calls are faithful to the runtime path.

the fakes mirror the style of :mod:`test_bind_import_from_disk` -- a
live-view ``find_by_workspace`` keyed to the fake connection's
``head_by_path`` so the post-seed second scan observes the rows the
seed transaction just committed.
"""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4, uuid7

import pytest
from watchfiles import Change

from threetears.agent.workspace.bind_policy import BindConflictPolicy
from threetears.agent.workspace.materialize import (
    _handle_watch_batch,
    bind,
)


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

    @property
    def namespace_name(self) -> str:
        """canonical workspace namespace name (WS-ACL-06)."""
        return f"workspace.{self.id}"


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
    """fake :class:`WorkspaceFileCollection` with live-view mode.

    :ivar _files: seed head-state file list
    :ivar _conn: optional bound connection for live-view semantics
    :ivar find_calls: running count of ``find_by_workspace`` invocations
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
            by_path: dict[str, _FakeFile] = {
                f.relative_path: f for f in self._files if f.relative_path in self._conn.head_by_path
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
        """resolve ``root / path`` while enforcing parentage.

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
    """async-context-manager handle modeling :class:`LeaseHandle`.

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
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any,
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
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any,
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

    def transaction(self, namespace: Any = None) -> _FakeTransaction:
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
        self,
        query: str,
        *args: Any,
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
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any,
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


def _harness(
    tmp_path: Path,
    initial_files: list[_FakeFile],
) -> dict[str, Any]:
    """build a fake bind harness rooted at ``tmp_path``.

    seeds the fake connection's ``head_by_path`` with every entry in
    ``initial_files`` so bind's first scan and the watcher's head
    probe both see the correct shape.

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
    *,
    on_conflict: BindConflictPolicy,
) -> None:
    """enter + exit bind against the harness with no-op body.

    :param h: harness dict
    :ptype h: dict[str, Any]
    :param on_conflict: policy to pass into :func:`bind`
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


# ---------------------------------------------------------------------------
# seed-on-enter tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_l3_wins_seed_noop_when_l3_has_files(tmp_path: Path) -> None:
    """L3_WINS bind against populated L3 + different disk keeps L3 untouched.

    the seed-if-empty gate must short-circuit: no journal rows emitted
    for either the L3 path or the divergent disk path during the seed
    phase.

    :param tmp_path: scratch root from pytest
    :ptype tmp_path: Path
    :return: None
    :rtype: None
    """
    l3_content = b"from-l3"
    h = _harness(
        tmp_path,
        initial_files=[
            _FakeFile("a.txt", l3_content, _sha(l3_content), 1),
        ],
    )
    disk_root = h["bind_root"] / h["workspace"].name
    disk_root.mkdir(parents=True, exist_ok=True)
    (disk_root / "a.txt").write_bytes(b"from-disk-different")

    await _enter_and_exit_bind(h, on_conflict=BindConflictPolicy.L3_WINS)

    # the seed-from-disk step must not emit any journal rows; the
    # watcher is only live while the body yields, and the body is a
    # no-op here. capture-back may notice that the disk bytes differ
    # from the snapshot (bind projected L3 -> disk on enter so they
    # match), so no rows come from there either.
    journal = [e for e in h["pool"].conn.executions if "INSERT INTO workspace_file_versions" in e[0]]
    # L3 path remains the same version 1 (no reseed, no update).
    a_rows = [row for row in journal if row[1][2] == "a.txt"]
    assert a_rows == []
    # head row still carries L3 content exactly (bind re-projected on
    # enter via atomic_write, so disk matches L3 by exit).
    head = h["pool"].conn.head_by_path["a.txt"]
    assert head["content"] == l3_content


@pytest.mark.asyncio
async def test_l3_wins_seed_imports_when_l3_empty(tmp_path: Path) -> None:
    """L3_WINS bind against empty L3 imports disk files as create rows.

    matches the historical ``_import_disk_to_l3_if_empty`` behavior:
    every disk file is a ``create`` at version 1.

    :param tmp_path: scratch root from pytest
    :ptype tmp_path: Path
    :return: None
    :rtype: None
    """
    h = _harness(tmp_path, initial_files=[])
    disk_root = h["bind_root"] / h["workspace"].name
    disk_root.mkdir(parents=True, exist_ok=True)
    (disk_root / "seed.txt").write_bytes(b"hello")

    await _enter_and_exit_bind(h, on_conflict=BindConflictPolicy.L3_WINS)

    journal = [e for e in h["pool"].conn.executions if "INSERT INTO workspace_file_versions" in e[0]]
    seed_rows = [row for row in journal if row[1][2] == "seed.txt"]
    assert len(seed_rows) == 1
    assert seed_rows[0][1][6] == "create"
    assert seed_rows[0][1][3] == 1


@pytest.mark.asyncio
async def test_disk_wins_seed_always_imports_from_disk(
    tmp_path: Path,
) -> None:
    """DISK_WINS bind clobbers L3: update divergent, delete L3-only paths.

    pre-populates L3 with two files ``a.txt`` and ``old.txt``; disk
    has a different ``a.txt`` content (forces update) and a new
    ``b.txt`` (forces create) but no ``old.txt`` (forces delete).

    :param tmp_path: scratch root from pytest
    :ptype tmp_path: Path
    :return: None
    :rtype: None
    """
    a_l3 = b"old-a-content"
    old_l3 = b"will-be-deleted"
    h = _harness(
        tmp_path,
        initial_files=[
            _FakeFile("a.txt", a_l3, _sha(a_l3), 1),
            _FakeFile("old.txt", old_l3, _sha(old_l3), 1),
        ],
    )
    disk_root = h["bind_root"] / h["workspace"].name
    disk_root.mkdir(parents=True, exist_ok=True)
    (disk_root / "a.txt").write_bytes(b"new-a-content")
    (disk_root / "b.txt").write_bytes(b"new-b-content")

    await _enter_and_exit_bind(h, on_conflict=BindConflictPolicy.DISK_WINS)

    journal = [e for e in h["pool"].conn.executions if "INSERT INTO workspace_file_versions" in e[0]]
    # a.txt: update row from the seed step (disk differs from L3).
    a_seed_rows = [row for row in journal if row[1][2] == "a.txt" and row[1][6] == "update"]
    assert len(a_seed_rows) == 1
    # b.txt: create row from seed.
    b_seed_rows = [row for row in journal if row[1][2] == "b.txt" and row[1][6] == "create"]
    assert len(b_seed_rows) == 1
    # old.txt: delete row from seed.
    old_delete_rows = [row for row in journal if row[1][2] == "old.txt" and row[1][6] == "delete"]
    assert len(old_delete_rows) == 1
    # head_by_path reflects the seed: a.txt updated, b.txt created,
    # old.txt removed.
    assert h["pool"].conn.head_by_path["a.txt"]["content"] == b"new-a-content"
    assert h["pool"].conn.head_by_path["b.txt"]["content"] == b"new-b-content"
    assert "old.txt" not in h["pool"].conn.head_by_path


# ---------------------------------------------------------------------------
# during-window watcher tests
# ---------------------------------------------------------------------------


async def _call_watch_batch(
    h: dict[str, Any],
    *,
    batch: set[tuple[Change, str]],
    on_conflict: BindConflictPolicy,
    just_wrote: deque[tuple[str, str]] | None = None,
) -> list[str]:
    """drive :func:`_handle_watch_batch` directly against the harness.

    :param h: harness dict
    :ptype h: dict[str, Any]
    :param batch: synthesized awatch event set
    :ptype batch: set[tuple[Change, str]]
    :param on_conflict: policy to pass through
    :ptype on_conflict: BindConflictPolicy
    :param just_wrote: bounded round-trip guard deque
    :ptype just_wrote: deque[tuple[str, str]] | None
    :return: list of mutated relative paths
    :rtype: list[str]
    """
    disk_root = h["bind_root"] / h["workspace"].name
    disk_root.mkdir(parents=True, exist_ok=True)
    if just_wrote is None:
        just_wrote = deque(maxlen=256)
    return await _handle_watch_batch(
        batch=batch,
        workspace=h["workspace"],
        disk_root=disk_root,
        resolved_root=disk_root.resolve(),
        db_pool=h["pool"],
        actor_id=h["actor_id"],
        correlation_id=h["correlation_id"],
        just_wrote=just_wrote,
        on_conflict=on_conflict,
    )


@pytest.mark.asyncio
async def test_l3_wins_watch_skips_modify_events(tmp_path: Path) -> None:
    """L3_WINS watcher ignores Change.modified events for L3-present files.

    :param tmp_path: scratch root from pytest
    :ptype tmp_path: Path
    :return: None
    :rtype: None
    """
    content = b"l3-canonical"
    h = _harness(
        tmp_path,
        initial_files=[_FakeFile("m.txt", content, _sha(content), 1)],
    )
    disk_root = h["bind_root"] / h["workspace"].name
    disk_root.mkdir(parents=True, exist_ok=True)
    target = disk_root / "m.txt"
    target.write_bytes(b"externally-rewritten")

    changed = await _call_watch_batch(
        h,
        batch={(Change.modified, str(target))},
        on_conflict=BindConflictPolicy.L3_WINS,
    )
    assert changed == []
    # no journal insert from the watcher.
    journal = [e for e in h["pool"].conn.executions if "INSERT INTO workspace_file_versions" in e[0]]
    assert journal == []


@pytest.mark.asyncio
async def test_l3_wins_watch_imports_added_for_new_paths(
    tmp_path: Path,
) -> None:
    """L3_WINS watcher imports Change.added for paths NOT in L3.

    :param tmp_path: scratch root from pytest
    :ptype tmp_path: Path
    :return: None
    :rtype: None
    """
    h = _harness(tmp_path, initial_files=[])
    disk_root = h["bind_root"] / h["workspace"].name
    disk_root.mkdir(parents=True, exist_ok=True)
    target = disk_root / "fresh.txt"
    target.write_bytes(b"brand-new")

    changed = await _call_watch_batch(
        h,
        batch={(Change.added, str(target))},
        on_conflict=BindConflictPolicy.L3_WINS,
    )
    assert changed == ["fresh.txt"]
    journal = [e for e in h["pool"].conn.executions if "INSERT INTO workspace_file_versions" in e[0]]
    assert len(journal) == 1
    assert journal[0][1][2] == "fresh.txt"
    assert journal[0][1][6] == "create"


@pytest.mark.asyncio
async def test_l3_wins_watch_skips_added_for_existing_paths(
    tmp_path: Path,
) -> None:
    """L3_WINS watcher skips Change.added events for paths already in L3.

    simulates a disk-echo: L3 already has a row, disk has a different
    payload, the OS delivers an ``added`` event. the watcher must
    skip (L3 remains authoritative).

    :param tmp_path: scratch root from pytest
    :ptype tmp_path: Path
    :return: None
    :rtype: None
    """
    l3_content = b"l3-canonical"
    h = _harness(
        tmp_path,
        initial_files=[_FakeFile("e.txt", l3_content, _sha(l3_content), 1)],
    )
    disk_root = h["bind_root"] / h["workspace"].name
    disk_root.mkdir(parents=True, exist_ok=True)
    target = disk_root / "e.txt"
    target.write_bytes(b"external-overwrite")

    changed = await _call_watch_batch(
        h,
        batch={(Change.added, str(target))},
        on_conflict=BindConflictPolicy.L3_WINS,
    )
    assert changed == []
    journal = [e for e in h["pool"].conn.executions if "INSERT INTO workspace_file_versions" in e[0]]
    assert journal == []


@pytest.mark.asyncio
async def test_l3_wins_watch_skips_deleted_events(tmp_path: Path) -> None:
    """L3_WINS watcher ignores Change.deleted for L3-present files.

    :param tmp_path: scratch root from pytest
    :ptype tmp_path: Path
    :return: None
    :rtype: None
    """
    content = b"dont-delete-me"
    h = _harness(
        tmp_path,
        initial_files=[_FakeFile("d.txt", content, _sha(content), 1)],
    )
    disk_root = h["bind_root"] / h["workspace"].name
    disk_root.mkdir(parents=True, exist_ok=True)
    target = disk_root / "d.txt"
    target.write_bytes(content)
    target.unlink()

    changed = await _call_watch_batch(
        h,
        batch={(Change.deleted, str(target))},
        on_conflict=BindConflictPolicy.L3_WINS,
    )
    assert changed == []
    journal = [e for e in h["pool"].conn.executions if "INSERT INTO workspace_file_versions" in e[0]]
    assert journal == []
    # head row still present: L3 is authoritative.
    assert "d.txt" in h["pool"].conn.head_by_path


@pytest.mark.asyncio
async def test_disk_wins_watch_imports_all(tmp_path: Path) -> None:
    """DISK_WINS watcher imports add / modify / delete events wholesale.

    verifies the pre-policy baseline behavior is preserved when the
    caller opts into ``DISK_WINS``: a modify produces an update row,
    a delete produces a delete row.

    :param tmp_path: scratch root from pytest
    :ptype tmp_path: Path
    :return: None
    :rtype: None
    """
    content_a = b"v1"
    content_b = b"will-die"
    h = _harness(
        tmp_path,
        initial_files=[
            _FakeFile("a.txt", content_a, _sha(content_a), 1),
            _FakeFile("b.txt", content_b, _sha(content_b), 1),
        ],
    )
    disk_root = h["bind_root"] / h["workspace"].name
    disk_root.mkdir(parents=True, exist_ok=True)
    # modify a.txt on disk.
    target_a = disk_root / "a.txt"
    target_a.write_bytes(b"v2")
    # delete b.txt on disk.
    target_b = disk_root / "b.txt"
    target_b.write_bytes(content_b)
    target_b.unlink()

    changed = await _call_watch_batch(
        h,
        batch={
            (Change.modified, str(target_a)),
            (Change.deleted, str(target_b)),
        },
        on_conflict=BindConflictPolicy.DISK_WINS,
    )
    assert set(changed) == {"a.txt", "b.txt"}
    journal = [e for e in h["pool"].conn.executions if "INSERT INTO workspace_file_versions" in e[0]]
    update_rows = [row for row in journal if row[1][6] == "update"]
    delete_rows = [row for row in journal if row[1][6] == "delete"]
    assert len(update_rows) == 1 and update_rows[0][1][2] == "a.txt"
    assert len(delete_rows) == 1 and delete_rows[0][1][2] == "b.txt"
    # head_by_path reflects: a.txt updated, b.txt removed.
    assert h["pool"].conn.head_by_path["a.txt"]["sha256"] == _sha(b"v2")
    assert "b.txt" not in h["pool"].conn.head_by_path
