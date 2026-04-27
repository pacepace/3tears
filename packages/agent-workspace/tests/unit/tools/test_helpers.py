"""tests for :mod:`threetears.agent.workspace.tools.helpers`.

covers all three primitives:

- :func:`_resolve_workspace` -- explicit name, pin-fallback, no-pin,
  unknown name, unknown pin, soft-deleted.
- :func:`_write_file_atomic` -- new file (create), existing file
  (update), OCC success, OCC failure, three-row transaction ordering,
  GREATEST-style workspace version update.
- :func:`_resolve_ref` -- ``"head"``, int, digit-string, checkpoint
  label, and miss.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest

from threetears.agent.workspace.tools import helpers as helpers_module
from threetears.agent.workspace.tools.helpers import (
    NoWorkspacePinned,
    Sha256Mismatch,
    WorkspaceNotFound,
    _resolve_ref,
    _resolve_workspace,
    _write_file_atomic,
)


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeWorkspaceEntity:
    """minimal stand-in for :class:`Workspace` exposing id, name, delete flag."""

    id: UUID
    name: str
    agent_id: UUID = field(default_factory=uuid4)
    date_deleted: datetime | None = None

    @property
    def namespace_name(self) -> str:
        """canonical workspace namespace name (WS-ACL-06)."""
        return f"workspace.{self.id}"


class _FakeWorkspaceCollection:
    """records lookups and serves entities by either name or id."""

    def __init__(
        self,
        entities: list[_FakeWorkspaceEntity] | None = None,
    ) -> None:
        self._entities = entities or []
        self.by_name_calls: list[tuple[UUID, str]] = []
        self.by_id_calls: list[tuple[UUID, UUID]] = []

    async def find_by_agent_and_name(self, agent_id: UUID, name: str) -> _FakeWorkspaceEntity | None:
        self.by_name_calls.append((agent_id, name))
        for e in self._entities:
            if e.name == name:
                return e
        return None

    async def find_by_id_and_agent(self, workspace_id: UUID, agent_id: UUID) -> _FakeWorkspaceEntity | None:
        self.by_id_calls.append((workspace_id, agent_id))
        for e in self._entities:
            if e.id == workspace_id:
                return e
        return None


class _FakeContext:
    """sentinel context object passed into _resolve_workspace."""


@dataclass
class _FakePin:
    workspace_id: UUID
    workspace_name: str


# ---------------------------------------------------------------------------
# _resolve_workspace
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_workspace_explicit_name_hits() -> None:
    """non-empty workspace_arg drives a by-name lookup."""
    agent_id = uuid4()
    target = _FakeWorkspaceEntity(id=uuid4(), name="alpha")
    workspaces = _FakeWorkspaceCollection([target])

    result = await _resolve_workspace("alpha", _FakeContext(), workspaces, agent_id)

    assert result is target
    assert workspaces.by_name_calls == [(agent_id, "alpha")]
    assert workspaces.by_id_calls == []


@pytest.mark.asyncio
async def test_resolve_workspace_explicit_name_miss_raises() -> None:
    """unknown workspace_arg raises WorkspaceNotFound."""
    workspaces = _FakeWorkspaceCollection([])
    with pytest.raises(WorkspaceNotFound) as excinfo:
        await _resolve_workspace("ghost", _FakeContext(), workspaces, uuid4())
    assert "ghost" in str(excinfo.value)


@pytest.mark.asyncio
async def test_resolve_workspace_no_arg_no_pin_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """empty workspace_arg + no pin raises NoWorkspacePinned."""

    async def _stub_get_pin(context: Any) -> Any:
        return None

    monkeypatch.setattr(helpers_module.pin_module, "get_pin", _stub_get_pin)
    workspaces = _FakeWorkspaceCollection([])
    with pytest.raises(NoWorkspacePinned):
        await _resolve_workspace(None, _FakeContext(), workspaces, uuid4())


@pytest.mark.asyncio
async def test_resolve_workspace_pin_fallback_hits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """empty workspace_arg + pin -> by-id lookup under agent."""
    agent_id = uuid4()
    pinned = _FakeWorkspaceEntity(id=uuid4(), name="pinned")
    workspaces = _FakeWorkspaceCollection([pinned])

    async def _stub_get_pin(context: Any) -> _FakePin:
        return _FakePin(workspace_id=pinned.id, workspace_name=pinned.name)

    monkeypatch.setattr(helpers_module.pin_module, "get_pin", _stub_get_pin)

    result = await _resolve_workspace(None, _FakeContext(), workspaces, agent_id)

    assert result is pinned
    assert workspaces.by_id_calls == [(pinned.id, agent_id)]
    assert workspaces.by_name_calls == []


@pytest.mark.asyncio
async def test_resolve_workspace_pin_stale_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pin pointing at deleted/unknown workspace raises WorkspaceNotFound."""
    workspaces = _FakeWorkspaceCollection([])

    async def _stub_get_pin(context: Any) -> _FakePin:
        return _FakePin(workspace_id=uuid4(), workspace_name="gone")

    monkeypatch.setattr(helpers_module.pin_module, "get_pin", _stub_get_pin)

    with pytest.raises(WorkspaceNotFound) as excinfo:
        await _resolve_workspace(None, _FakeContext(), workspaces, uuid4())
    assert "gone" in str(excinfo.value)


@pytest.mark.asyncio
async def test_resolve_workspace_soft_deleted_raises() -> None:
    """soft-deleted entity resolved by name raises WorkspaceNotFound."""
    deleted = _FakeWorkspaceEntity(id=uuid4(), name="rip", date_deleted=datetime.now(UTC))
    workspaces = _FakeWorkspaceCollection([deleted])

    with pytest.raises(WorkspaceNotFound) as excinfo:
        await _resolve_workspace("rip", _FakeContext(), workspaces, uuid4())
    assert "rip" in str(excinfo.value)


# ---------------------------------------------------------------------------
# _write_file_atomic -- fake pool/conn
# ---------------------------------------------------------------------------


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


@dataclass
class _FakeConnection:
    """records execute + fetchrow calls and captures transaction membership."""

    head_row: dict[str, Any] | None = None
    journal_max_version: int = 0
    executions: list[tuple[str, tuple[Any, ...], bool]] = field(default_factory=list)
    fetchrows: list[tuple[str, tuple[Any, ...], bool]] = field(default_factory=list)
    transactions: list[_FakeTransaction] = field(default_factory=list)
    transaction_open: bool = False

    def transaction(self, namespace: Any = None) -> _FakeTransaction:
        tx = _FakeTransaction(parent=self)
        self.transactions.append(tx)
        return tx

    async def execute(self, query: str, *args: Any) -> str:
        self.executions.append((query, args, self.transaction_open))
        return "INSERT 0 1"

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        """dispatch by SQL shape: head SELECT returns head_row; journal-max
        SELECT returns a row with ``max_version``; all other queries return
        ``head_row`` for signature compatibility.
        """
        self.fetchrows.append((query, args, self.transaction_open))
        result: dict[str, Any] | None
        if "COALESCE(MAX(version)" in query:
            result = {"max_version": self.journal_max_version}
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


def _fake_workspace() -> _FakeWorkspaceEntity:
    return _FakeWorkspaceEntity(id=uuid4(), name="ws")


@pytest.mark.asyncio
async def test_write_file_atomic_new_file_inserts_at_version_one() -> None:
    """no existing head -> version 1, action create, three in-tx writes."""
    pool = _FakePool()
    ws = _fake_workspace()
    actor = uuid4()
    corr = uuid4()

    new_version, new_sha = await _write_file_atomic(
        db_pool=pool,
        workspace=ws,
        relative_path="a/b.txt",
        content=b"hello",
        action="create",
        actor_id=actor,
        correlation_id=corr,
        expected_sha256=None,
        workspace_file_collection=None,
        workspace_file_version_collection=None,
        workspace_collection=None,
    )

    assert new_version == 1
    # sha256("hello")
    assert new_sha == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    # two fetchrows (head SELECT + journal-max SELECT), both in tx;
    # three executes (journal insert, head upsert, workspace update).
    assert len(pool.conn.fetchrows) == 2
    for _sql, _args, in_tx in pool.conn.fetchrows:
        assert in_tx is True
    assert len(pool.conn.executions) == 3
    for _sql, _args, in_tx in pool.conn.executions:
        assert in_tx is True
    # ordering: journal INSERT, head UPSERT, workspace UPDATE
    order = [e[0] for e in pool.conn.executions]
    assert "INSERT INTO workspace_file_versions" in order[0]
    assert "INSERT INTO workspace_files" in order[1]
    assert "UPDATE workspaces" in order[2]
    # transaction entered + exited exactly once
    assert len(pool.conn.transactions) == 1
    assert pool.conn.transactions[0].entered is True
    assert pool.conn.transactions[0].exited is True


@pytest.mark.asyncio
async def test_write_file_atomic_existing_file_bumps_version() -> None:
    """existing head at version 3 -> new version 4, action update."""
    pool = _FakePool()
    pool.conn.head_row = {
        "content": b"old",
        "sha256": "a" * 64,
        "version": 3,
    }
    pool.conn.journal_max_version = 3
    ws = _fake_workspace()
    new_version, _new_sha = await _write_file_atomic(
        db_pool=pool,
        workspace=ws,
        relative_path="doc.md",
        content=b"new",
        action="update",
        actor_id=uuid4(),
        correlation_id=uuid4(),
        expected_sha256=None,
        workspace_file_collection=None,
        workspace_file_version_collection=None,
        workspace_collection=None,
    )
    assert new_version == 4
    # journal row insert used action="update"
    journal_sql, journal_args, _ = pool.conn.executions[0]
    assert "INSERT INTO workspace_file_versions" in journal_sql
    # args positions: id, workspace_id, relative_path, version, content,
    # sha256, action, label, actor_id, correlation_id, date_created
    assert journal_args[6] == "update"
    assert journal_args[3] == 4


@pytest.mark.asyncio
async def test_write_file_atomic_occ_success_passes_sha() -> None:
    """expected_sha256 matching current head -> write proceeds."""
    pool = _FakePool()
    pool.conn.head_row = {
        "content": b"old",
        "sha256": "b" * 64,
        "version": 7,
    }
    pool.conn.journal_max_version = 7
    ws = _fake_workspace()
    new_version, _ = await _write_file_atomic(
        db_pool=pool,
        workspace=ws,
        relative_path="c.txt",
        content=b"ok",
        action="update",
        actor_id=uuid4(),
        correlation_id=uuid4(),
        expected_sha256="b" * 64,
        workspace_file_collection=None,
        workspace_file_version_collection=None,
        workspace_collection=None,
    )
    assert new_version == 8
    assert len(pool.conn.executions) == 3


@pytest.mark.asyncio
async def test_write_file_atomic_occ_mismatch_raises_and_aborts_tx() -> None:
    """expected sha != current sha -> Sha256Mismatch; no execute calls."""
    pool = _FakePool()
    pool.conn.head_row = {
        "content": b"old",
        "sha256": "c" * 64,
        "version": 2,
    }
    ws = _fake_workspace()
    with pytest.raises(Sha256Mismatch) as excinfo:
        await _write_file_atomic(
            db_pool=pool,
            workspace=ws,
            relative_path="c.txt",
            content=b"ignored",
            action="update",
            actor_id=uuid4(),
            correlation_id=uuid4(),
            expected_sha256="d" * 64,
            workspace_file_collection=None,
            workspace_file_version_collection=None,
            workspace_collection=None,
        )
    assert excinfo.value.expected == "d" * 64
    assert excinfo.value.current == "c" * 64
    # fetchrow happened inside tx, but no executes ran
    assert len(pool.conn.fetchrows) == 1
    assert pool.conn.executions == []
    # transaction context-manager exited (asyncpg rolls back on exc)
    assert len(pool.conn.transactions) == 1
    assert pool.conn.transactions[0].entered is True
    assert pool.conn.transactions[0].exited is True


@pytest.mark.asyncio
async def test_write_file_atomic_occ_mismatch_absent_reports_none_current() -> None:
    """expected set but no head row -> Sha256Mismatch with current=None."""
    pool = _FakePool()  # head_row = None
    ws = _fake_workspace()
    with pytest.raises(Sha256Mismatch) as excinfo:
        await _write_file_atomic(
            db_pool=pool,
            workspace=ws,
            relative_path="absent.txt",
            content=b"x",
            action="create",
            actor_id=uuid4(),
            correlation_id=uuid4(),
            expected_sha256="e" * 64,
            workspace_file_collection=None,
            workspace_file_version_collection=None,
            workspace_collection=None,
        )
    assert excinfo.value.current is None
    assert excinfo.value.expected == "e" * 64


@pytest.mark.asyncio
async def test_write_file_atomic_workspace_update_uses_greatest_semantics() -> None:
    """workspace UPDATE SQL advances current_version via GREATEST."""
    pool = _FakePool()
    ws = _fake_workspace()
    await _write_file_atomic(
        db_pool=pool,
        workspace=ws,
        relative_path="x",
        content=b"",
        action="create",
        actor_id=uuid4(),
        correlation_id=uuid4(),
        expected_sha256=None,
        workspace_file_collection=None,
        workspace_file_version_collection=None,
        workspace_collection=None,
    )
    update_sql, update_args, _ = pool.conn.executions[2]
    assert "UPDATE workspaces" in update_sql
    assert "GREATEST(current_version" in update_sql
    # args positions: new_version, now, workspace_id
    assert update_args[0] == 1
    assert update_args[2] == ws.id


def test_sha256_mismatch_exposes_expected_and_current_attrs() -> None:
    """Sha256Mismatch carries expected/current as attributes for tool wrappers."""
    exc = Sha256Mismatch(expected="a" * 64, current="b" * 64)
    assert exc.expected == "a" * 64
    assert exc.current == "b" * 64
    assert "expected" in str(exc)
    assert "current" in str(exc)


# ---------------------------------------------------------------------------
# _resolve_ref
# ---------------------------------------------------------------------------


class _RefQueryConnection:
    """records SQL issued by _resolve_ref and returns a scripted row."""

    def __init__(self, row: dict[str, Any] | None) -> None:
        self._row = row
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        self.calls.append((query, args))
        return self._row


@pytest.mark.asyncio
async def test_resolve_ref_head_selects_newest_version_row() -> None:
    """ref=='head' triggers ORDER BY version DESC LIMIT 1 query."""
    workspace_id = uuid4()
    row = {"version": 5, "content": b"latest", "action": "update"}
    conn = _RefQueryConnection(row)
    result = await _resolve_ref(conn, workspace_id, "a.txt", "head")
    assert result == row
    assert len(conn.calls) == 1
    sql, args = conn.calls[0]
    assert "ORDER BY version DESC" in sql
    assert "LIMIT 1" in sql
    assert args == (workspace_id, "a.txt")


@pytest.mark.asyncio
async def test_resolve_ref_integer_selects_by_exact_version() -> None:
    """int ref triggers WHERE ... AND version = $3 lookup."""
    workspace_id = uuid4()
    row = {"version": 3, "content": b"v3"}
    conn = _RefQueryConnection(row)
    result = await _resolve_ref(conn, workspace_id, "a.txt", 3)
    assert result == row
    sql, args = conn.calls[0]
    assert "AND version = $3" in sql
    assert args == (workspace_id, "a.txt", 3)


@pytest.mark.asyncio
async def test_resolve_ref_digit_string_treated_as_int() -> None:
    """digit-only string ref is parsed as integer version."""
    workspace_id = uuid4()
    row = {"version": 7, "content": b"v7"}
    conn = _RefQueryConnection(row)
    result = await _resolve_ref(conn, workspace_id, "a.txt", "7")
    assert result == row
    sql, args = conn.calls[0]
    assert "AND version = $3" in sql
    assert args == (workspace_id, "a.txt", 7)


@pytest.mark.asyncio
async def test_resolve_ref_non_numeric_string_treated_as_checkpoint_label() -> None:
    """non-head, non-digit string ref queries action='checkpoint' AND label."""
    workspace_id = uuid4()
    row = {"version": 2, "content": b"cp", "action": "checkpoint", "label": "v1-release"}
    conn = _RefQueryConnection(row)
    result = await _resolve_ref(conn, workspace_id, "a.txt", "v1-release")
    assert result == row
    sql, args = conn.calls[0]
    assert "action = 'checkpoint'" in sql
    assert "label = $3" in sql
    assert args == (workspace_id, "a.txt", "v1-release")


@pytest.mark.asyncio
async def test_resolve_ref_missing_returns_none() -> None:
    """no matching row returns None (caller treats as skip or clean error)."""
    conn = _RefQueryConnection(None)
    result = await _resolve_ref(conn, uuid4(), "missing.txt", 99)
    assert result is None
