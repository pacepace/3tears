"""unit tests for ``threetears.agent.workspace.materialize.bind`` + :func:`recover`.

the tests wire the async context manager against:

- a real :class:`pathlib.Path` tree under ``tmp_path``,
- a fake sandbox whose ``resolve_fs_path(name, root)`` computes
  ``tmp_path / root / name`` (so escape/KeyError branches of the real
  sandbox are out of scope here — they're covered by sandbox tests),
- a fake workspace / file / version collection bundle backed by plain
  dicts,
- a fake asyncpg pool that records every ``execute`` call and the
  transaction nesting so capture-back can be verified,
- a fake :class:`WorkspaceFileLease` that records acquire/release and
  stamps the lease key on a shared counter so the integration-style
  race tests in ``test_bind_lease_race.py`` can reuse the same fake.

the fake pool does NOT actually persist rows; the tests assert against
the recorded ``execute`` calls and the order they arrive. this matches
the fs_write unit-test style already in the tree.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4, uuid7

import pytest

from threetears.agent.workspace.bind_policy import BindConflictPolicy
from threetears.agent.workspace.materialize import bind, recover


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


@dataclass
class _FakeWorkspace:
    """stand-in for :class:`Workspace` exposing the fields bind reads."""

    id: UUID
    name: str
    current_version: int = 0
    date_deleted: datetime | None = None


@dataclass
class _FakeFile:
    """stand-in for :class:`WorkspaceFile` exposing the fields bind reads."""

    relative_path: str
    content: bytes
    sha256: str
    version: int


class _FakeWorkspaceCollection:
    """fake :class:`WorkspaceCollection` exposing :meth:`find_by_id`."""

    def __init__(self, workspaces: list[_FakeWorkspace]) -> None:
        self._workspaces = workspaces
        self.find_by_id_calls: list[UUID] = []

    async def find_by_id(self, workspace_id: UUID) -> _FakeWorkspace | None:
        self.find_by_id_calls.append(workspace_id)
        for ws in self._workspaces:
            if ws.id == workspace_id and ws.date_deleted is None:
                return ws
        return None


class _FakeFileCollection:
    """fake :class:`WorkspaceFileCollection` exposing :meth:`find_by_workspace`."""

    def __init__(self, files: list[_FakeFile]) -> None:
        self._files = files

    async def find_by_workspace(self, workspace_id: UUID) -> list[_FakeFile]:
        del workspace_id
        return list(self._files)


class _FakeVersionCollection:
    """fake :class:`WorkspaceFileVersionCollection` -- unused by bind/capture."""


class _FakeSandbox:
    """stand-in for :class:`WorkspaceSandbox`; resolve -> ``tmp_path / root / name``."""

    def __init__(self, roots: dict[str, Path]) -> None:
        self._roots = roots

    def resolve_fs_path(self, path: str, root_name: str) -> Path:
        root = self._roots[root_name]
        candidate = (root / path).resolve()
        candidate.relative_to(root)
        return candidate


@dataclass
class _FakeLeaseHandle:
    """async-context-manager handle modelling :class:`LeaseHandle`."""

    key: str
    holder: str
    released: bool = False
    body_executing_flag: dict[str, Any] | None = None

    async def __aenter__(self) -> _FakeLeaseHandle:
        if self.body_executing_flag is not None:
            assert self.body_executing_flag.get("count", 0) == 0, (
                "another lease holder already inside the critical section"
            )
            self.body_executing_flag["count"] = 1
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.body_executing_flag is not None:
            self.body_executing_flag["count"] = 0
        self.released = True
        return None


@dataclass
class _FakeLease:
    """fake :class:`WorkspaceFileLease` recording acquire calls."""

    acquired: list[tuple[UUID, str, int, int]] = field(default_factory=list)
    handles: list[_FakeLeaseHandle] = field(default_factory=list)
    body_executing_flag: dict[str, Any] | None = None

    async def acquire(
        self,
        workspace_id: UUID,
        relative_path: str,
        ttl_seconds: int = 30,
        max_wait_seconds: int = 60,
    ) -> _FakeLeaseHandle:
        self.acquired.append((workspace_id, relative_path, ttl_seconds, max_wait_seconds))
        handle = _FakeLeaseHandle(
            key=f"workspace:{workspace_id.hex}:{relative_path}",
            holder=f"pod-{len(self.handles)}",
            body_executing_flag=self.body_executing_flag,
        )
        self.handles.append(handle)
        return handle


@dataclass
class _FakeTransaction:
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
    executions: list[tuple[str, tuple[Any, ...], bool]] = field(default_factory=list)
    fetchrows: list[tuple[str, tuple[Any, ...], bool]] = field(default_factory=list)
    transactions: list[_FakeTransaction] = field(default_factory=list)
    transaction_open: bool = False
    head_row: dict[str, Any] | None = None
    journal_max_version: int = 0
    journal_max_by_path: dict[str, int] = field(default_factory=dict)

    def transaction(self) -> _FakeTransaction:
        tx = _FakeTransaction(parent=self)
        self.transactions.append(tx)
        return tx

    async def execute(self, query: str, *args: Any) -> str:
        self.executions.append((query, args, self.transaction_open))
        # keep per-path journal-max monotonic so a create followed by a
        # delete followed by a re-create in the same transaction picks up
        # the next version rather than colliding on the prior one.
        if "INSERT INTO workspace_file_versions" in query:
            rel = args[2]
            inserted_version = int(args[3])
            prior = self.journal_max_by_path.get(rel, 0)
            if inserted_version > prior:
                self.journal_max_by_path[rel] = inserted_version
        return "INSERT 0 1"

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        """dispatch by SQL shape: journal-max SELECT returns a row with
        ``max_version`` (per-path if recorded, else the global fallback);
        head SELECT (and any fallback) returns ``head_row``.
        """
        self.fetchrows.append((query, args, self.transaction_open))
        result: dict[str, Any] | None
        if "COALESCE(MAX(version)" in query:
            rel = args[1]
            per_path = self.journal_max_by_path.get(rel)
            max_version = per_path if per_path is not None else self.journal_max_version
            result = {"max_version": max_version}
        else:
            result = self.head_row
        return result


@dataclass
class _FakeAcquireCM:
    conn: _FakeConnection

    async def __aenter__(self) -> _FakeConnection:
        return self.conn

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        return None


@dataclass
class _FakePool:
    conn: _FakeConnection = field(default_factory=_FakeConnection)

    def acquire(self) -> _FakeAcquireCM:
        return _FakeAcquireCM(conn=self.conn)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _build_harness(
    tmp_path: Path,
    *,
    workspace_name: str = "ws_test",
    initial_files: list[_FakeFile] | None = None,
) -> dict[str, Any]:
    """assemble a fake bind environment wired against ``tmp_path``.

    seeds ``pool.conn.journal_max_by_path`` from ``initial_files`` so
    ``_next_journal_version`` returns realistic prior+1 values. in
    production every head row has at least one matching journal row,
    so a fake harness claiming head version N without a journal entry
    at N would diverge from the real DB state.
    """
    ws_id = uuid7()
    ws = _FakeWorkspace(id=ws_id, name=workspace_name)
    bind_root = tmp_path / "bind_root"
    bind_root.mkdir(parents=True, exist_ok=True)
    sandbox = _FakeSandbox({"bind": bind_root})
    workspace_coll = _FakeWorkspaceCollection([ws])
    files_seed = list(initial_files or [])
    file_coll = _FakeFileCollection(files_seed)
    version_coll = _FakeVersionCollection()
    lease = _FakeLease()
    pool = _FakePool()
    for seed_file in files_seed:
        pool.conn.journal_max_by_path[seed_file.relative_path] = seed_file.version
    return {
        "workspace_id": ws_id,
        "workspace": ws,
        "sandbox": sandbox,
        "bind_root": bind_root,
        "workspace_coll": workspace_coll,
        "file_coll": file_coll,
        "version_coll": version_coll,
        "lease": lease,
        "pool": pool,
        "actor_id": uuid4(),
        "correlation_id": uuid7(),
    }


async def _call_bind(
    harness: dict[str, Any],
    root_name: str = "bind",
    on_conflict: BindConflictPolicy = BindConflictPolicy.L3_WINS,
) -> Any:
    return bind(
        workspace_id=harness["workspace_id"],
        sandbox=harness["sandbox"],
        lease=harness["lease"],
        workspace_collection=harness["workspace_coll"],
        workspace_file_collection=harness["file_coll"],
        workspace_file_version_collection=harness["version_coll"],
        db_pool=harness["pool"],
        actor_id=harness["actor_id"],
        correlation_id=harness["correlation_id"],
        root_name=root_name,
        on_conflict=on_conflict,
    )


# ---------------------------------------------------------------------------
# tests -- happy path, each capture kind
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bind_sync_l3_to_disk_on_enter(tmp_path: Path) -> None:
    """entering bind writes every workspace file to the sandboxed disk root."""
    content = b"hello\n"
    harness = _build_harness(
        tmp_path,
        initial_files=[_FakeFile("a.txt", content, _sha(content), 1)],
    )
    async with await _call_bind(harness) as disk_root:
        assert (disk_root / "a.txt").read_bytes() == content


@pytest.mark.asyncio
async def test_bind_body_updates_file_emits_update_journal(
    tmp_path: Path,
) -> None:
    """body overwrites existing file -> capture writes one update journal row."""
    initial = b"v1"
    harness = _build_harness(
        tmp_path,
        initial_files=[_FakeFile("notes.txt", initial, _sha(initial), 1)],
    )
    async with await _call_bind(harness) as disk_root:
        (disk_root / "notes.txt").write_bytes(b"v2")

    executions = harness["pool"].conn.executions
    assert len(harness["pool"].conn.transactions) == 1
    tx = harness["pool"].conn.transactions[0]
    assert tx.entered is True and tx.exited is True

    journal_rows = [e for e in executions if "INSERT INTO workspace_file_versions" in e[0]]
    head_rows = [e for e in executions if "INSERT INTO workspace_files" in e[0]]
    ws_rows = [e for e in executions if "UPDATE workspaces" in e[0]]
    assert len(journal_rows) == 1
    assert journal_rows[0][1][3] == 2  # new version is prior+1
    assert journal_rows[0][1][6] == "update"
    assert journal_rows[0][2] is True  # inside transaction
    assert len(head_rows) == 1
    assert len(ws_rows) == 1


@pytest.mark.asyncio
async def test_bind_body_creates_new_file_emits_create_journal(
    tmp_path: Path,
) -> None:
    """body creates a file not in snapshot -> capture writes one create journal row."""
    harness = _build_harness(tmp_path, initial_files=[])
    async with await _call_bind(harness) as disk_root:
        (disk_root / "new.txt").write_bytes(b"brand new")

    executions = harness["pool"].conn.executions
    journal_rows = [e for e in executions if "INSERT INTO workspace_file_versions" in e[0]]
    assert len(journal_rows) == 1
    assert journal_rows[0][1][3] == 1  # new version starts at 1
    assert journal_rows[0][1][6] == "create"


@pytest.mark.asyncio
async def test_bind_body_deletes_file_emits_delete_journal(
    tmp_path: Path,
) -> None:
    """body deletes an existing disk file -> capture writes delete journal row."""
    content = b"goodbye"
    harness = _build_harness(
        tmp_path,
        initial_files=[_FakeFile("bye.txt", content, _sha(content), 3)],
    )
    async with await _call_bind(harness) as disk_root:
        (disk_root / "bye.txt").unlink()

    executions = harness["pool"].conn.executions
    journal_rows = [e for e in executions if "INSERT INTO workspace_file_versions" in e[0]]
    delete_head = [e for e in executions if "DELETE FROM workspace_files" in e[0]]
    assert len(journal_rows) == 1
    assert journal_rows[0][1][3] == 4  # prior+1
    assert journal_rows[0][1][6] == "delete"
    assert len(delete_head) == 1


@pytest.mark.asyncio
async def test_bind_body_no_changes_opens_no_transaction(
    tmp_path: Path,
) -> None:
    """body touches nothing -> no transaction, no execute calls for writes."""
    content = b"stable"
    harness = _build_harness(
        tmp_path,
        initial_files=[_FakeFile("fixed.txt", content, _sha(content), 1)],
    )
    async with await _call_bind(harness):
        pass

    assert harness["pool"].conn.transactions == []
    assert harness["pool"].conn.executions == []


# ---------------------------------------------------------------------------
# tests -- exception path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bind_body_exception_skips_capture_and_releases_lease(
    tmp_path: Path,
) -> None:
    """body raises -> lease released, NO capture-back, exception propagates."""
    content = b"v1"
    harness = _build_harness(
        tmp_path,
        initial_files=[_FakeFile("a.txt", content, _sha(content), 1)],
    )

    class _Boom(RuntimeError):
        pass

    with pytest.raises(_Boom):
        async with await _call_bind(harness) as disk_root:
            (disk_root / "a.txt").write_bytes(b"mutated-but-rolled-back")
            raise _Boom("body crashed")

    assert harness["lease"].handles[0].released is True
    assert harness["pool"].conn.transactions == []
    assert harness["pool"].conn.executions == []


# ---------------------------------------------------------------------------
# tests -- lease semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bind_acquires_lease_with_root_scoped_key(tmp_path: Path) -> None:
    """lease key is ``bind:{root_name}`` -- default root yields ``bind:bind``."""
    harness = _build_harness(tmp_path, initial_files=[])
    async with await _call_bind(harness):
        pass
    assert len(harness["lease"].acquired) == 1
    ws_id, rel_path, ttl, max_wait = harness["lease"].acquired[0]
    assert ws_id == harness["workspace_id"]
    assert rel_path == "bind:bind"
    assert ttl == 30
    assert max_wait == 60


@pytest.mark.asyncio
async def test_bind_different_root_name_yields_different_lease_key(
    tmp_path: Path,
) -> None:
    """custom root_name produces a distinct lease key."""
    harness = _build_harness(tmp_path, initial_files=[])
    harness["sandbox"]._roots["secondary"] = harness["bind_root"]
    async with await _call_bind(harness, root_name="secondary"):
        pass
    assert harness["lease"].acquired[0][1] == "bind:secondary"


@pytest.mark.asyncio
async def test_bind_soft_deleted_workspace_raises(tmp_path: Path) -> None:
    """bind on a workspace with ``date_deleted is not None`` raises ValueError."""
    harness = _build_harness(tmp_path, initial_files=[])
    harness["workspace"].date_deleted = datetime.now(UTC)
    with pytest.raises(ValueError):
        async with await _call_bind(harness):
            pass


@pytest.mark.asyncio
async def test_bind_unknown_workspace_raises(tmp_path: Path) -> None:
    """bind on a workspace id the collection does not resolve raises ValueError."""
    harness = _build_harness(tmp_path, initial_files=[])
    harness["workspace_coll"]._workspaces = []
    with pytest.raises(ValueError):
        async with await _call_bind(harness):
            pass


# ---------------------------------------------------------------------------
# tests -- recover
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recover_writes_disk_diffs_back_to_l3(tmp_path: Path) -> None:
    """recover after simulated crash: writes diffs, returns changed list."""
    existing = b"original"
    harness = _build_harness(
        tmp_path,
        initial_files=[_FakeFile("live.txt", existing, _sha(existing), 2)],
    )

    # pre-populate disk root to simulate what a crashed bind left behind
    disk_root = harness["bind_root"] / harness["workspace"].name
    disk_root.mkdir(parents=True, exist_ok=True)
    (disk_root / "live.txt").write_bytes(b"mutated")
    (disk_root / "brand_new.txt").write_bytes(b"new stuff")

    changed = await recover(
        workspace_id=harness["workspace_id"],
        sandbox=harness["sandbox"],
        workspace_collection=harness["workspace_coll"],
        workspace_file_collection=harness["file_coll"],
        workspace_file_version_collection=harness["version_coll"],
        db_pool=harness["pool"],
        actor_id=harness["actor_id"],
        correlation_id=harness["correlation_id"],
    )
    assert set(changed) == {"live.txt", "brand_new.txt"}
    executions = harness["pool"].conn.executions
    actions = [e[1][6] for e in executions if "INSERT INTO workspace_file_versions" in e[0]]
    assert sorted(actions) == ["create", "update"]


@pytest.mark.asyncio
async def test_recover_unchanged_file_skipped(tmp_path: Path) -> None:
    """file whose sha matches L3 head is not rewritten."""
    content = b"same"
    harness = _build_harness(
        tmp_path,
        initial_files=[_FakeFile("keep.txt", content, _sha(content), 1)],
    )
    disk_root = harness["bind_root"] / harness["workspace"].name
    disk_root.mkdir(parents=True, exist_ok=True)
    (disk_root / "keep.txt").write_bytes(content)

    changed = await recover(
        workspace_id=harness["workspace_id"],
        sandbox=harness["sandbox"],
        workspace_collection=harness["workspace_coll"],
        workspace_file_collection=harness["file_coll"],
        workspace_file_version_collection=harness["version_coll"],
        db_pool=harness["pool"],
        actor_id=harness["actor_id"],
        correlation_id=harness["correlation_id"],
    )
    assert changed == []
    assert harness["pool"].conn.transactions == []


@pytest.mark.asyncio
async def test_recover_unknown_workspace_raises(tmp_path: Path) -> None:
    """recover on unknown workspace id raises ValueError."""
    harness = _build_harness(tmp_path, initial_files=[])
    harness["workspace_coll"]._workspaces = []
    with pytest.raises(ValueError):
        await recover(
            workspace_id=harness["workspace_id"],
            sandbox=harness["sandbox"],
            workspace_collection=harness["workspace_coll"],
            workspace_file_collection=harness["file_coll"],
            workspace_file_version_collection=harness["version_coll"],
            db_pool=harness["pool"],
            actor_id=harness["actor_id"],
            correlation_id=harness["correlation_id"],
        )


# ---------------------------------------------------------------------------
# tests -- audit publish on capture-back
# ---------------------------------------------------------------------------


@dataclass
class _FakeNats:
    """records ``publish`` calls for audit envelope assertions."""

    published: list[tuple[str, bytes]] = field(default_factory=list)
    raise_on_publish: BaseException | None = None

    async def publish(self, subject: str, payload: bytes) -> None:
        self.published.append((subject, payload))
        if self.raise_on_publish is not None:
            raise self.raise_on_publish


@pytest.mark.asyncio
async def test_bind_emits_one_audit_event_per_changed_file(
    tmp_path: Path,
) -> None:
    """capture-back: one ``workspace.bind`` audit event per changed file."""
    import json

    initial = b"v1"
    harness = _build_harness(
        tmp_path,
        initial_files=[_FakeFile("a.txt", initial, _sha(initial), 1)],
    )
    nats = _FakeNats()
    async with bind(
        workspace_id=harness["workspace_id"],
        sandbox=harness["sandbox"],
        lease=harness["lease"],
        workspace_collection=harness["workspace_coll"],
        workspace_file_collection=harness["file_coll"],
        workspace_file_version_collection=harness["version_coll"],
        db_pool=harness["pool"],
        actor_id=harness["actor_id"],
        correlation_id=harness["correlation_id"],
        root_name="bind",
        nats_client=nats,
        namespace="ns",
    ) as disk_root:
        (disk_root / "a.txt").write_bytes(b"v2")
        (disk_root / "b.txt").write_bytes(b"new file")

    # two changes (update + create) -> two audit events
    assert len(nats.published) == 2
    subjects = {s for s, _p in nats.published}
    assert subjects == {"ns.audit.workspace.bind"}
    envelopes = [json.loads(p.decode("utf-8")) for _s, p in nats.published]
    for env in envelopes:
        assert env["event_type"] == "workspace.bind"
        assert env["action"] == "bind"
        assert env["resource_type"] == "workspace_file"
        assert env["details"]["root_name"] == "bind"
    kinds = {env["details"]["change_kind"] for env in envelopes}
    assert kinds == {"update", "create"}


@pytest.mark.asyncio
async def test_bind_audit_publish_failure_does_not_break_capture_back(
    tmp_path: Path,
) -> None:
    """raising NATS publish does not undo a successful capture-back commit."""
    harness = _build_harness(tmp_path, initial_files=[])
    nats = _FakeNats(raise_on_publish=RuntimeError("nats offline"))
    # bind body should complete cleanly even though publish raises
    async with bind(
        workspace_id=harness["workspace_id"],
        sandbox=harness["sandbox"],
        lease=harness["lease"],
        workspace_collection=harness["workspace_coll"],
        workspace_file_collection=harness["file_coll"],
        workspace_file_version_collection=harness["version_coll"],
        db_pool=harness["pool"],
        actor_id=harness["actor_id"],
        correlation_id=harness["correlation_id"],
        root_name="bind",
        nats_client=nats,
        namespace="ns",
    ) as disk_root:
        (disk_root / "new.txt").write_bytes(b"hello")
    # one journal+head row committed
    journal = [e for e in harness["pool"].conn.executions if "INSERT INTO workspace_file_versions" in e[0]]
    assert len(journal) == 1


@pytest.mark.asyncio
async def test_bind_no_changes_publishes_no_audit_events(tmp_path: Path) -> None:
    """empty capture-back set means zero audit events even with NATS wired."""
    content = b"stable"
    harness = _build_harness(
        tmp_path,
        initial_files=[_FakeFile("fixed.txt", content, _sha(content), 1)],
    )
    nats = _FakeNats()
    async with bind(
        workspace_id=harness["workspace_id"],
        sandbox=harness["sandbox"],
        lease=harness["lease"],
        workspace_collection=harness["workspace_coll"],
        workspace_file_collection=harness["file_coll"],
        workspace_file_version_collection=harness["version_coll"],
        db_pool=harness["pool"],
        actor_id=harness["actor_id"],
        correlation_id=harness["correlation_id"],
        root_name="bind",
        nats_client=nats,
        namespace="ns",
    ):
        pass
    assert nats.published == []


# ---------------------------------------------------------------------------
# tests -- live watcher during bind window
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bind_watch_batch_imports_new_disk_file(tmp_path: Path) -> None:
    """feeding a simulated watch batch to the helper writes a create row.

    exercising the watcher task via :func:`watchfiles.awatch` inside a
    unit test is flaky -- OS-native events deliver after unpredictable
    delays. we drive :func:`_handle_watch_batch` directly with a
    synthesized batch; the production :func:`_watch_loop` forwards
    awatch's batches into this same helper so the coverage is faithful
    to the runtime path.
    """
    from collections import deque

    from watchfiles import Change

    from threetears.agent.workspace.materialize import _handle_watch_batch

    harness = _build_harness(tmp_path, initial_files=[])
    async with await _call_bind(harness) as disk_root:
        # builder writes a file mid-bind.
        target = disk_root / "mid_bind.txt"
        target.write_bytes(b"written by external process")
        # simulate a watchfiles event batch delivered for that change.
        batch = {(Change.added, str(target))}
        just_wrote: deque[tuple[str, str]] = deque(maxlen=256)
        changed = await _handle_watch_batch(
            batch=batch,
            workspace=harness["workspace"],
            disk_root=disk_root,
            resolved_root=disk_root.resolve(),
            db_pool=harness["pool"],
            actor_id=harness["actor_id"],
            correlation_id=harness["correlation_id"],
            just_wrote=just_wrote,
        )
        assert changed == ["mid_bind.txt"]

    journal = [e for e in harness["pool"].conn.executions if "INSERT INTO workspace_file_versions" in e[0]]
    mid_rows = [row for row in journal if row[1][2] == "mid_bind.txt"]
    # at least one create from the live-watcher call. the fake file
    # collection used here does NOT reflect conn-side writes, so
    # capture-back re-journals the same file via the existing snapshot-
    # diff path -- a second row is acceptable. the first row is a
    # create.
    assert len(mid_rows) >= 1
    assert mid_rows[0][1][6] == "create"
    assert mid_rows[0][1][3] == 1  # first version starts at 1


@pytest.mark.asyncio
async def test_watch_batch_deleted_change_emits_delete(tmp_path: Path) -> None:
    """Change.deleted on a tracked file emits delete journal + head delete.

    the delete path requires disk-wins semantics: under the default
    l3-wins policy external deletions are ignored because L3 is
    authoritative. this test explicitly passes disk_wins so the
    emit-delete branch is exercised.
    """
    from collections import deque

    from watchfiles import Change

    from threetears.agent.workspace.bind_policy import BindConflictPolicy
    from threetears.agent.workspace.materialize import _handle_watch_batch

    content = b"to-be-deleted"
    harness = _build_harness(
        tmp_path,
        initial_files=[_FakeFile("gone.txt", content, _sha(content), 2)],
    )
    async with await _call_bind(harness) as disk_root:
        target = disk_root / "gone.txt"
        # seed the fake head row so the watcher's delete branch finds it.
        harness["pool"].conn.head_row = {
            "content": content,
            "sha256": _sha(content),
            "version": 2,
        }
        # have the deletion actually occur on disk first so capture-back
        # also notices the file is gone. the watcher batch is delivered
        # after the file is already removed from the FS.
        target.unlink()
        batch = {(Change.deleted, str(target))}
        just_wrote: deque[tuple[str, str]] = deque(maxlen=256)
        changed = await _handle_watch_batch(
            batch=batch,
            workspace=harness["workspace"],
            disk_root=disk_root,
            resolved_root=disk_root.resolve(),
            db_pool=harness["pool"],
            actor_id=harness["actor_id"],
            correlation_id=harness["correlation_id"],
            just_wrote=just_wrote,
            on_conflict=BindConflictPolicy.DISK_WINS,
        )
        assert changed == ["gone.txt"]

    executions = harness["pool"].conn.executions
    delete_head = [e for e in executions if "DELETE FROM workspace_files" in e[0]]
    journal_deletes = [e for e in executions if "INSERT INTO workspace_file_versions" in e[0] and e[1][6] == "delete"]
    assert len(journal_deletes) >= 1
    assert len(delete_head) >= 1


@pytest.mark.asyncio
async def test_watch_batch_just_wrote_suppresses_roundtrip(
    tmp_path: Path,
) -> None:
    """change whose (rel, sha) is in just_wrote set is skipped.

    models the bind-side round-trip: when the initial L3->disk sync wrote
    a file and the OS delivers an ``added`` event for it, we must NOT
    journal that as an external change.
    """
    from collections import deque

    from watchfiles import Change

    from threetears.agent.workspace.materialize import _handle_watch_batch

    harness = _build_harness(tmp_path, initial_files=[])
    async with await _call_bind(harness) as disk_root:
        target = disk_root / "own_write.txt"
        payload = b"our own bytes"
        target.write_bytes(payload)
        batch = {(Change.added, str(target))}
        just_wrote: deque[tuple[str, str]] = deque(maxlen=256)
        just_wrote.append(("own_write.txt", _sha(payload)))
        changed = await _handle_watch_batch(
            batch=batch,
            workspace=harness["workspace"],
            disk_root=disk_root,
            resolved_root=disk_root.resolve(),
            db_pool=harness["pool"],
            actor_id=harness["actor_id"],
            correlation_id=harness["correlation_id"],
            just_wrote=just_wrote,
        )
        assert changed == []
        # helper emitted no rows of its own; exact row count is
        # asserted AFTER bind exits so we can observe the final state.
        journal_during = [
            e
            for e in harness["pool"].conn.executions
            if "INSERT INTO workspace_file_versions" in e[0] and e[1][2] == "own_write.txt"
        ]
        assert journal_during == []
    # OUTSIDE the bind: capture-back emitted exactly one create.
    journal_after = [
        e
        for e in harness["pool"].conn.executions
        if "INSERT INTO workspace_file_versions" in e[0] and e[1][2] == "own_write.txt"
    ]
    assert len(journal_after) == 1
    assert journal_after[0][1][6] == "create"
