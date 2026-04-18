"""tests for ``threetears.workspace.fs_edit`` -- FsEditTool."""

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

from threetears.agent.workspace.tools.fs_edit import FsEditTool


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeWorkspaceEntity:
    id: UUID
    name: str
    date_deleted: datetime | None = None

    @property
    def namespace_name(self) -> str:
        """canonical workspace namespace name (WS-ACL-06)."""
        return f"workspace.{self.id}"


class _FakeWorkspaceCollection:
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
class _FakeFileEntity:
    relative_path: str
    content: bytes
    sha256: str
    version: int
    date_updated: datetime = datetime.now(UTC)


class _FakeFileCollection:
    def __init__(self, files: list[_FakeFileEntity] | None = None) -> None:
        self._files = files or []

    async def find_by_workspace_and_relative_path(
        self, workspace_id: UUID, relative_path: str
    ) -> _FakeFileEntity | None:
        for f in self._files:
            if f.relative_path == relative_path:
                return f
        return None


class _FakeVersionCollection:
    pass


class _RecordingSandbox:
    def __init__(self, deny_writes: list[str] | None = None) -> None:
        self._deny_writes = set(deny_writes or [])
        self.enforce_calls: list[tuple[str, str]] = []

    def enforce(self, action: str, target: str) -> None:
        self.enforce_calls.append((action, target))
        if action == "write" and target in self._deny_writes:
            raise SandboxDenied(action, target, "not in write globs")


class _FakeContext:
    pass


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


def _build_tool(
    *,
    acl_cache: Any,
    files: list[_FakeFileEntity] | None = None,
    head_row: dict[str, Any] | None = None,
    deny_writes: list[str] | None = None,
) -> tuple[FsEditTool, _FakePool, _RecordingSandbox]:
    ws = _FakeWorkspaceEntity(id=uuid4(), name="ws")
    workspaces = _FakeWorkspaceCollection([ws])
    file_coll = _FakeFileCollection(files)
    sandbox = _RecordingSandbox(deny_writes=deny_writes)
    pool = _FakePool()
    pool.conn.head_row = head_row
    tool = FsEditTool(
        workspace_collection=workspaces,  # type: ignore[arg-type]
        workspace_file_collection=file_coll,  # type: ignore[arg-type]
        workspace_file_version_collection=_FakeVersionCollection(),  # type: ignore[arg-type]
        sandbox=sandbox,  # type: ignore[arg-type]
        context_provider=lambda: _FakeContext(),
        agent_id=uuid4(),
        db_pool=pool,
        acl_cache=acl_cache,
    )
    return tool, pool, sandbox


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fs_edit_happy_replaces_all_occurrences_and_writes(
    permissive_acl_cache: MagicMock,
) -> None:
    """find replaces ALL occurrences; three writes land under one transaction."""
    text = "foo bar foo baz foo"
    existing = _FakeFileEntity(
        relative_path="a.md",
        content=text.encode("utf-8"),
        sha256="a" * 64,
        version=1,
    )
    head = {"content": text.encode("utf-8"), "sha256": "a" * 64, "version": 1}
    tool, pool, sandbox = _build_tool(
        files=[existing], head_row=head, acl_cache=permissive_acl_cache
    )
    pool.conn.journal_max_version = 1

    result = await tool.execute(
        relative_path="a.md",
        find="foo",
        replace="QUX",
        workspace="ws",
    )
    assert result.success is True, result.error
    assert "replaced 3 occurrence" in result.content
    # journal INSERT content carries fully-substituted text
    journal_sql, journal_args, _ = pool.conn.executions[0]
    assert "workspace_file_versions" in journal_sql
    assert journal_args[4] == b"QUX bar QUX baz QUX"
    # sandbox write enforced BEFORE any transaction started
    assert sandbox.enforce_calls == [("write", "a.md")]
    # three executes, all in tx
    assert len(pool.conn.executions) == 3
    for _s, _a, in_tx in pool.conn.executions:
        assert in_tx is True
    # sha is of the new content
    expected_sha = hashlib.sha256(b"QUX bar QUX baz QUX").hexdigest()
    assert result.metadata is not None
    assert result.metadata["sha256"] == expected_sha
    assert result.metadata["occurrences"] == 3


# ---------------------------------------------------------------------------
# error paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fs_edit_missing_file_returns_clean_error(
    permissive_acl_cache: MagicMock,
) -> None:
    tool, pool, _ = _build_tool(files=[], acl_cache=permissive_acl_cache)
    result = await tool.execute(relative_path="missing.md", find="x", replace="y", workspace="ws")
    assert result.success is False
    assert result.error is not None
    assert "missing.md" in result.error
    # no fetchrow or execute occurred
    assert pool.conn.fetchrows == []
    assert pool.conn.executions == []


@pytest.mark.asyncio
async def test_fs_edit_find_empty_returns_clean_error(
    permissive_acl_cache: MagicMock,
) -> None:
    existing = _FakeFileEntity(relative_path="a.md", content=b"x", sha256="a" * 64, version=1)
    tool, pool, _ = _build_tool(files=[existing], acl_cache=permissive_acl_cache)
    result = await tool.execute(relative_path="a.md", find="", replace="y", workspace="ws")
    assert result.success is False
    assert result.error is not None
    assert "empty" in result.error
    assert pool.conn.executions == []


@pytest.mark.asyncio
async def test_fs_edit_find_missing_from_content_returns_clean_error(
    permissive_acl_cache: MagicMock,
) -> None:
    existing = _FakeFileEntity(
        relative_path="a.md",
        content=b"hello world",
        sha256="a" * 64,
        version=1,
    )
    tool, pool, _ = _build_tool(files=[existing], acl_cache=permissive_acl_cache)
    result = await tool.execute(relative_path="a.md", find="missing", replace="x", workspace="ws")
    assert result.success is False
    assert result.error is not None
    assert "find string not found" in result.error
    assert pool.conn.executions == []


@pytest.mark.asyncio
async def test_fs_edit_binary_file_returns_clean_error(
    permissive_acl_cache: MagicMock,
) -> None:
    """non-UTF-8 content is refused with explicit binary error."""
    existing = _FakeFileEntity(
        relative_path="logo.png",
        content=b"\x89PNG\xff\xfe",
        sha256="a" * 64,
        version=1,
    )
    tool, pool, _ = _build_tool(files=[existing], acl_cache=permissive_acl_cache)
    result = await tool.execute(relative_path="logo.png", find="x", replace="y", workspace="ws")
    assert result.success is False
    assert result.error is not None
    assert "binary" in result.error
    assert pool.conn.executions == []


@pytest.mark.asyncio
async def test_fs_edit_stale_sha256_returns_mismatch_error(
    permissive_acl_cache: MagicMock,
) -> None:
    """stale expected_sha256 on a good find -> clean mismatch error."""
    text = "hello"
    existing = _FakeFileEntity(
        relative_path="a.md",
        content=text.encode("utf-8"),
        sha256="c" * 64,
        version=1,
    )
    head = {"content": text.encode("utf-8"), "sha256": "c" * 64, "version": 1}
    tool, pool, _ = _build_tool(
        files=[existing], head_row=head, acl_cache=permissive_acl_cache
    )

    result = await tool.execute(
        relative_path="a.md",
        find="hello",
        replace="world",
        expected_sha256="d" * 64,
        workspace="ws",
    )
    assert result.success is False
    assert result.error is not None
    assert "sha256 mismatch" in result.error
    assert "c" * 64 in result.error
    # fetchrow happened inside helper tx; no executes
    assert pool.conn.executions == []


@pytest.mark.asyncio
async def test_fs_edit_sandbox_denied_never_reads_or_writes(
    permissive_acl_cache: MagicMock,
) -> None:
    """SandboxDenied on write -> gate-then-act; no collection or pool touches."""
    existing = _FakeFileEntity(
        relative_path="secret.env",
        content=b"TOKEN=xyz",
        sha256="a" * 64,
        version=1,
    )
    tool, pool, sandbox = _build_tool(
        files=[existing], deny_writes=["secret.env"], acl_cache=permissive_acl_cache
    )
    result = await tool.execute(
        relative_path="secret.env",
        find="TOKEN=",
        replace="TOKEN=abc",
        workspace="ws",
    )
    assert result.success is False
    assert result.error is not None
    assert "secret.env" in result.error
    assert sandbox.enforce_calls == [("write", "secret.env")]
    assert pool.conn.fetchrows == []
    assert pool.conn.executions == []


# ---------------------------------------------------------------------------
# MCP surface
# ---------------------------------------------------------------------------


def test_fs_edit_mcp_name_is_exact_string(
    permissive_acl_cache: MagicMock,
) -> None:
    tool, _, _ = _build_tool(acl_cache=permissive_acl_cache)
    assert tool.mcp_name() == "threetears.workspace.fs_edit"


def test_fs_edit_mcp_schema_declares_required_fields(
    permissive_acl_cache: MagicMock,
) -> None:
    tool, _, _ = _build_tool(acl_cache=permissive_acl_cache)
    defn = tool.mcp_schema()
    assert isinstance(defn, MCPToolDefinition)
    assert defn.input_schema["required"] == [
        "relative_path",
        "find",
        "replace",
    ]
    assert defn.input_schema["additionalProperties"] is False
