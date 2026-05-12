"""tests for ``threetears.workspace.fs_write`` -- FsWriteTool."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from threetears.agent.tools.base_tool import MCPToolDefinition
from threetears.core.security import SandboxDenied

from threetears.agent.workspace.tools.fs_write import FsWriteTool
from _helpers.asyncpg_shims import FakeAsyncpgAcquireCM, FakeAsyncpgConnection, FakeAsyncpgPool, FakeAsyncpgTransaction
from _helpers.workspace_shims import (
    FakeWorkspaceCollection,
    FakeWorkspaceContext,
    FakeWorkspaceEntity,
    FakeWorkspaceFile,
    FakeWorkspaceFileCollection,
    FakeWorkspaceFileVersionCollection,
    FakeWorkspaceSandbox,
)


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeWorkspaceEntity(FakeWorkspaceEntity):
    id: UUID
    name: str
    date_deleted: datetime | None = None
    agent_id: UUID = field(default_factory=uuid4)

    @property
    def namespace_name(self) -> str:
        """canonical workspace namespace name (WS-ACL-06)."""
        return f"workspace.{self.id}"


class _FakeWorkspaceCollection(FakeWorkspaceCollection):
    def __init__(self, entities: list[_FakeWorkspaceEntity]) -> None:
        self._entities = entities

    async def find_by_agent_and_name(self, agent_id: UUID, name: str) -> _FakeWorkspaceEntity | None:
        for e in self._entities:
            if e.name == name:
                return e
        return None

    async def find_by_id_and_agent(self, workspace_id: UUID, agent_id: UUID) -> _FakeWorkspaceEntity | None:
        for e in self._entities:
            if e.id == workspace_id:
                return e
        return None


@dataclass
class _FakeFileEntity(FakeWorkspaceFile):
    relative_path: str
    content: bytes
    sha256: str
    version: int
    date_updated: datetime = datetime.now(UTC)


class _FakeFileCollection(FakeWorkspaceFileCollection):
    def __init__(self, files: list[_FakeFileEntity] | None = None) -> None:
        self._files = files or []

    async def find_by_workspace_and_relative_path(
        self, workspace_id: UUID, relative_path: str
    ) -> _FakeFileEntity | None:
        for f in self._files:
            if f.relative_path == relative_path:
                return f
        return None


class _FakeVersionCollection(FakeWorkspaceFileVersionCollection):
    pass


class _RecordingSandbox:
    def __init__(self, deny_writes: list[str] | None = None) -> None:
        self._deny_writes = set(deny_writes or [])
        self.syntax_calls: list[str] = []

    def validate_syntax(self, target: str) -> None:
        self.syntax_calls.append(target)
        if target in self._deny_writes:
            raise SandboxDenied("access", target, "syntactic deny (test fixture)")


class _FakeContext(FakeWorkspaceContext):
    pass


@dataclass
class _FakeTransaction(FakeAsyncpgTransaction):
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
class _FakeConnection(FakeAsyncpgConnection):
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
        """dispatch by SQL shape: journal-max SELECT returns a row with
        ``max_version``; head SELECT (and any fallback) returns ``head_row``.
        """
        self.fetchrows.append((query, args, self.transaction_open))
        result: dict[str, Any] | None
        if "COALESCE(MAX(version)" in query:
            result = {"max_version": self.journal_max_version}
        else:
            result = self.head_row
        return result


@dataclass
class _FakeAcquireCM(FakeAsyncpgAcquireCM):
    conn: _FakeConnection

    async def __aenter__(self) -> _FakeConnection:
        return self.conn

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        return None


@dataclass
class _FakePool(FakeAsyncpgPool):
    conn: _FakeConnection = field(default_factory=_FakeConnection)

    def acquire(self) -> _FakeAcquireCM:
        return _FakeAcquireCM(conn=self.conn)


# ---------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------


def _build_tool(
    *,
    acl_cache: Any,
    workspace_entities: list[_FakeWorkspaceEntity] | None = None,
    files: list[_FakeFileEntity] | None = None,
    head_row: dict[str, Any] | None = None,
    deny_writes: list[str] | None = None,
    agent_id: UUID | None = None,
) -> tuple[FsWriteTool, _FakePool, _RecordingSandbox, UUID]:
    agent_id = agent_id or uuid4()
    ws_entity = workspace_entities[0] if workspace_entities else _FakeWorkspaceEntity(id=uuid4(), name="ws")
    workspaces = _FakeWorkspaceCollection(workspace_entities or [ws_entity])
    file_coll = _FakeFileCollection(files)
    sandbox = _RecordingSandbox(deny_writes=deny_writes)
    pool = _FakePool()
    pool.conn.head_row = head_row
    tool = FsWriteTool(
        workspace_collection=workspaces,  # type: ignore[arg-type]
        workspace_file_collection=file_coll,  # type: ignore[arg-type]
        workspace_file_version_collection=_FakeVersionCollection(),  # type: ignore[arg-type]
        sandbox=sandbox,  # type: ignore[arg-type]
        context_provider=lambda: _FakeContext(),
        agent_id=agent_id,
        db_pool=pool,
        acl_cache=acl_cache,
    )
    return tool, pool, sandbox, agent_id


# ---------------------------------------------------------------------------
# create (no existing head row)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fs_write_create_new_file_at_version_one(
    permissive_acl_cache: MagicMock,
) -> None:
    """no existing head -> action=create, version=1, sha matches content."""
    tool, pool, sandbox, _ = _build_tool(acl_cache=permissive_acl_cache)
    result = await tool.execute(
        relative_path="notes/hello.md",
        content="hello",
        workspace="ws",
    )
    assert result.success is True, result.error
    # enforce happened and was a write
    assert sandbox.syntax_calls == ["notes/hello.md"]
    # three executes in tx; two fetchrows (head + journal-max) in tx
    assert len(pool.conn.executions) == 3
    for _sql, _args, in_tx in pool.conn.executions:
        assert in_tx is True
    assert len(pool.conn.fetchrows) == 2
    # journal action is create, version is 1
    journal_sql, journal_args, _ = pool.conn.executions[0]
    assert "workspace_file_versions" in journal_sql
    assert journal_args[6] == "create"
    assert journal_args[3] == 1
    # sha matches actual content hash
    expected_sha = hashlib.sha256(b"hello").hexdigest()
    assert expected_sha in result.content
    # metadata bundled
    assert result.metadata is not None
    assert result.metadata["sha256"] == expected_sha
    assert result.metadata["version"] == 1
    assert result.metadata["bytes_written"] == 5


@pytest.mark.asyncio
async def test_fs_write_writes_inside_single_transaction(
    permissive_acl_cache: MagicMock,
) -> None:
    """every execute statement occurs while transaction_open is True."""
    tool, pool, _sandbox, _ = _build_tool(acl_cache=permissive_acl_cache)
    await tool.execute(relative_path="a.md", content="x", workspace="ws")
    assert len(pool.conn.transactions) == 1
    assert pool.conn.transactions[0].entered is True
    assert pool.conn.transactions[0].exited is True
    assert all(in_tx for _s, _a, in_tx in pool.conn.executions)


# ---------------------------------------------------------------------------
# update (existing head row)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fs_write_update_existing_file_bumps_version_and_sha(
    permissive_acl_cache: MagicMock,
) -> None:
    """existing head -> action=update, version incremented, sha changes."""
    existing = _FakeFileEntity(
        relative_path="a.md",
        content=b"old",
        sha256="old" * 21 + "o",
        version=5,
    )
    head = {"content": b"old", "sha256": "old" * 21 + "o", "version": 5}
    tool, pool, _sandbox, _ = _build_tool(files=[existing], head_row=head, acl_cache=permissive_acl_cache)
    pool.conn.journal_max_version = 5

    result = await tool.execute(relative_path="a.md", content="fresh", workspace="ws")
    assert result.success is True, result.error
    # journal action=update, version=6
    journal_args = pool.conn.executions[0][1]
    assert journal_args[6] == "update"
    assert journal_args[3] == 6
    # sha256 is the new content's hash
    expected_sha = hashlib.sha256(b"fresh").hexdigest()
    assert result.metadata is not None
    assert result.metadata["sha256"] == expected_sha
    assert result.metadata["version"] == 6


# ---------------------------------------------------------------------------
# OCC failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fs_write_stale_expected_sha_returns_mismatch_error(
    permissive_acl_cache: MagicMock,
) -> None:
    """expected_sha256 != current sha -> error names current sha and advises retry."""
    existing = _FakeFileEntity(
        relative_path="a.md",
        content=b"old",
        sha256="c" * 64,
        version=2,
    )
    head = {"content": b"old", "sha256": "c" * 64, "version": 2}
    tool, pool, _sandbox, _ = _build_tool(files=[existing], head_row=head, acl_cache=permissive_acl_cache)

    result = await tool.execute(
        relative_path="a.md",
        content="new",
        expected_sha256="d" * 64,
        workspace="ws",
    )
    assert result.success is False
    assert result.error is not None
    assert "sha256 mismatch" in result.error
    assert "c" * 64 in result.error  # current sha in message
    assert "re-read" in result.error
    # no executes happened on OCC fail; fetchrow did
    assert pool.conn.executions == []
    assert len(pool.conn.fetchrows) == 1


# ---------------------------------------------------------------------------
# sandbox gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fs_write_sandbox_denied_returns_clean_error_no_writes(
    permissive_acl_cache: MagicMock,
) -> None:
    """SandboxDenied on write -> clean error, no pool acquire."""
    tool, pool, sandbox, _ = _build_tool(deny_writes=["secret.env"], acl_cache=permissive_acl_cache)
    result = await tool.execute(relative_path="secret.env", content="x", workspace="ws")
    assert result.success is False
    assert result.error is not None
    assert "secret.env" in result.error
    # enforce invoked once for write
    assert sandbox.syntax_calls == ["secret.env"]
    # no fetchrow, no execute -- gate-then-act held
    assert pool.conn.fetchrows == []
    assert pool.conn.executions == []


# ---------------------------------------------------------------------------
# journal row fields
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fs_write_journal_row_captures_actor_and_correlation(
    permissive_acl_cache: MagicMock,
) -> None:
    """journal row carries agent_id as actor_id; correlation_id is a UUID."""
    agent_id = uuid4()
    tool, pool, _sandbox, _ = _build_tool(agent_id=agent_id, acl_cache=permissive_acl_cache)
    await tool.execute(relative_path="a.md", content="x", workspace="ws")
    journal_args = pool.conn.executions[0][1]
    # args positions per SQL in helpers:
    # id, workspace_id, relative_path, version, content, sha256,
    # action, label, actor_id, correlation_id, date_created
    assert journal_args[8] == agent_id
    assert isinstance(journal_args[9], UUID)


# ---------------------------------------------------------------------------
# MCP surface
# ---------------------------------------------------------------------------


def test_fs_write_mcp_name_is_exact_string(
    permissive_acl_cache: MagicMock,
) -> None:
    tool, _, _, _ = _build_tool(acl_cache=permissive_acl_cache)
    assert tool.mcp_name() == "threetears.workspace.fs_write"


def test_fs_write_mcp_schema_requires_path_and_content(
    permissive_acl_cache: MagicMock,
) -> None:
    tool, _, _, _ = _build_tool(acl_cache=permissive_acl_cache)
    defn = tool.mcp_schema()
    assert isinstance(defn, MCPToolDefinition)
    assert defn.input_schema["required"] == ["relative_path", "content"]
    assert defn.input_schema["additionalProperties"] is False
    props = defn.input_schema["properties"]
    assert set(props.keys()) == {
        "relative_path",
        "content",
        "expected_sha256",
        "workspace",
    }
