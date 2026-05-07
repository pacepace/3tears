"""tests for ``threetears.workspace.delete`` -- WorkspaceDeleteTool."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from threetears.agent.tools.base_tool import MCPToolDefinition

from threetears.agent.workspace.tools import workspace_delete as workspace_delete_module
from threetears.agent.workspace.tools.workspace_delete import WorkspaceDeleteTool
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
    """minimal stand-in for :class:`Workspace` for delete-target lookups."""

    @property
    def namespace_name(self) -> str:
        """canonical workspace namespace name (WS-ACL-06)."""
        return f"workspace.{self.id}"

    id: UUID
    name: str
    agent_id: UUID = field(default_factory=uuid4)
    date_deleted: Any = None


class _FakeWorkspaceCollection(FakeWorkspaceCollection):
    def __init__(self, entities: list[_FakeWorkspaceEntity]) -> None:
        self._entities = entities

    async def find_by_agent_and_name(self, agent_id: UUID, name: str) -> _FakeWorkspaceEntity | None:
        for e in self._entities:
            if e.name == name:
                return e
        return None


class _FakeFileCollection(FakeWorkspaceFileCollection):
    """unused by delete; satisfies factory contract."""


class _FakeVersionCollection(FakeWorkspaceFileVersionCollection):
    """unused by delete; satisfies factory contract."""


class _FakeSandbox(FakeWorkspaceSandbox):
    """unused by delete; satisfies factory contract."""

    def resolve_fs_path(self, path: str, root_name: str) -> Any:
        raise KeyError(root_name)

    def validate_syntax(self, target: str) -> None:
        return None


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
    executions: list[tuple[str, tuple[Any, ...], bool]] = field(default_factory=list)
    transaction_calls: int = 0
    transactions: list[_FakeTransaction] = field(default_factory=list)
    transaction_open: bool = False

    def transaction(self, namespace: Any = None) -> _FakeTransaction:
        self.transaction_calls += 1
        tx = _FakeTransaction(parent=self)
        self.transactions.append(tx)
        return tx

    async def execute(self, query: str, *args: Any) -> str:
        self.executions.append((query, args, self.transaction_open))
        return "UPDATE 1"


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
    acquire_calls: int = 0

    def acquire(self) -> _FakeAcquireCM:
        self.acquire_calls += 1
        return _FakeAcquireCM(conn=self.conn)


@dataclass
class _PinnedSnapshot:
    workspace_id: UUID
    workspace_name: str


class _FakeContext(FakeWorkspaceContext):
    pass


def _build_tool(
    *,
    workspace_collection: Any,
    db_pool: Any,
    acl_cache: Any,
    agent_id: UUID | None = None,
    context_provider: Any = None,
) -> WorkspaceDeleteTool:
    return WorkspaceDeleteTool(
        workspace_collection=workspace_collection,
        workspace_file_collection=_FakeFileCollection(),
        workspace_file_version_collection=_FakeVersionCollection(),
        sandbox=_FakeSandbox(),
        context_provider=context_provider or (lambda: _FakeContext()),
        agent_id=agent_id or uuid4(),
        db_pool=db_pool,
        acl_cache=acl_cache,
    )


# ---------------------------------------------------------------------------
# happy paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_soft_deletes_workspace_in_transaction(
    monkeypatch: pytest.MonkeyPatch,
    permissive_acl_cache: MagicMock,
) -> None:
    """delete sets date_deleted via UPDATE inside a single transaction."""
    ws_id = uuid4()
    workspace = _FakeWorkspaceEntity(id=ws_id, name="bye")

    async def _get_pin(_ctx: Any) -> Any:
        return None

    cleared: list[Any] = []

    async def _clear_pin(_ctx: Any) -> None:
        cleared.append(_ctx)

    monkeypatch.setattr(workspace_delete_module.pin, "get_pin", _get_pin)
    monkeypatch.setattr(workspace_delete_module.pin, "clear_pin", _clear_pin)

    pool = _FakePool()
    tool = _build_tool(
        workspace_collection=_FakeWorkspaceCollection([workspace]),
        db_pool=pool,
        acl_cache=permissive_acl_cache,
    )

    result = await tool.execute(name="bye")

    assert result.success is True
    assert result.error is None
    assert "deleted workspace 'bye'" in result.content

    assert pool.acquire_calls == 1
    assert pool.conn.transaction_calls == 1
    assert pool.conn.transactions[0].entered is True
    assert pool.conn.transactions[0].exited is True

    updates = [e for e in pool.conn.executions if "UPDATE workspaces" in e[0]]
    assert len(updates) == 1
    sql, args, in_tx = updates[0]
    assert "SET date_deleted = $1" in sql
    assert "date_updated = $1" in sql
    assert "WHERE id = $2" in sql
    assert in_tx is True
    assert isinstance(args[0], datetime)
    assert args[0].tzinfo == UTC
    assert args[1] == ws_id

    # no pin to clear
    assert cleared == []


@pytest.mark.asyncio
async def test_delete_clears_pin_when_pinned_workspace_matches(
    monkeypatch: pytest.MonkeyPatch,
    permissive_acl_cache: MagicMock,
) -> None:
    """pinned workspace's pin is cleared by the same delete call."""
    ws_id = uuid4()
    workspace = _FakeWorkspaceEntity(id=ws_id, name="seed")
    fake_ctx = _FakeContext()

    async def _get_pin(ctx: Any) -> _PinnedSnapshot:
        assert ctx is fake_ctx
        return _PinnedSnapshot(workspace_id=ws_id, workspace_name="seed")

    cleared: list[Any] = []

    async def _clear_pin(ctx: Any) -> None:
        cleared.append(ctx)

    monkeypatch.setattr(workspace_delete_module.pin, "get_pin", _get_pin)
    monkeypatch.setattr(workspace_delete_module.pin, "clear_pin", _clear_pin)

    pool = _FakePool()
    tool = _build_tool(
        workspace_collection=_FakeWorkspaceCollection([workspace]),
        db_pool=pool,
        context_provider=lambda: fake_ctx,
        acl_cache=permissive_acl_cache,
    )

    result = await tool.execute(name="seed")
    assert result.success is True
    assert cleared == [fake_ctx]


@pytest.mark.asyncio
async def test_delete_does_not_clear_pin_for_different_workspace(
    monkeypatch: pytest.MonkeyPatch,
    permissive_acl_cache: MagicMock,
) -> None:
    """a different pinned workspace is left untouched when deleting another."""
    ws_id = uuid4()
    other_id = uuid4()
    workspace = _FakeWorkspaceEntity(id=ws_id, name="bye")

    async def _get_pin(_ctx: Any) -> _PinnedSnapshot:
        return _PinnedSnapshot(workspace_id=other_id, workspace_name="other")

    cleared: list[Any] = []

    async def _clear_pin(ctx: Any) -> None:
        cleared.append(ctx)

    monkeypatch.setattr(workspace_delete_module.pin, "get_pin", _get_pin)
    monkeypatch.setattr(workspace_delete_module.pin, "clear_pin", _clear_pin)

    pool = _FakePool()
    tool = _build_tool(
        workspace_collection=_FakeWorkspaceCollection([workspace]),
        db_pool=pool,
        acl_cache=permissive_acl_cache,
    )
    result = await tool.execute(name="bye")
    assert result.success is True
    assert cleared == []


# ---------------------------------------------------------------------------
# error paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_unknown_workspace_returns_error(
    permissive_acl_cache: MagicMock,
) -> None:
    """missing workspace yields clean error and writes nothing."""
    pool = _FakePool()
    tool = _build_tool(
        workspace_collection=_FakeWorkspaceCollection([]),
        db_pool=pool,
        acl_cache=permissive_acl_cache,
    )
    result = await tool.execute(name="ghost")
    assert result.success is False
    assert result.error is not None
    assert "'ghost'" in result.error
    assert pool.conn.executions == []


@pytest.mark.asyncio
async def test_delete_already_deleted_returns_error_idempotently(
    permissive_acl_cache: MagicMock,
) -> None:
    """deleting an already-soft-deleted workspace surfaces same not-found error."""
    workspace = _FakeWorkspaceEntity(
        id=uuid4(),
        name="ghost",
        date_deleted=datetime(2026, 4, 16, 9, 0, 0, tzinfo=UTC),
    )
    pool = _FakePool()
    tool = _build_tool(
        workspace_collection=_FakeWorkspaceCollection([workspace]),
        db_pool=pool,
        acl_cache=permissive_acl_cache,
    )
    result = await tool.execute(name="ghost")
    assert result.success is False
    assert result.error is not None
    assert "'ghost'" in result.error
    assert pool.conn.executions == []


@pytest.mark.asyncio
async def test_delete_traps_unexpected_exceptions_as_data(
    permissive_acl_cache: MagicMock,
) -> None:
    """unexpected runtime errors surface as ToolResult(success=False)."""

    class _Boom:
        async def find_by_agent_and_name(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("storage offline")

    pool = _FakePool()
    tool = _build_tool(workspace_collection=_Boom(), db_pool=pool, acl_cache=permissive_acl_cache)
    result = await tool.execute(name="x")
    assert result.success is False
    assert result.error is not None
    assert "storage offline" in result.error


# ---------------------------------------------------------------------------
# MCP surface
# ---------------------------------------------------------------------------


def test_mcp_name_is_exact_string(
    permissive_acl_cache: MagicMock,
) -> None:
    pool = _FakePool()
    tool = _build_tool(
        workspace_collection=_FakeWorkspaceCollection([]),
        db_pool=pool,
        acl_cache=permissive_acl_cache,
    )
    assert tool.mcp_name() == "threetears.workspace.delete"


def test_mcp_version_is_semver_string(
    permissive_acl_cache: MagicMock,
) -> None:
    pool = _FakePool()
    tool = _build_tool(
        workspace_collection=_FakeWorkspaceCollection([]),
        db_pool=pool,
        acl_cache=permissive_acl_cache,
    )
    assert tool.mcp_version() == "1.0"


def test_mcp_schema_declares_required_name(
    permissive_acl_cache: MagicMock,
) -> None:
    pool = _FakePool()
    tool = _build_tool(
        workspace_collection=_FakeWorkspaceCollection([]),
        db_pool=pool,
        acl_cache=permissive_acl_cache,
    )
    definition = tool.mcp_schema()
    assert isinstance(definition, MCPToolDefinition)
    assert definition.name == "threetears.workspace.delete"
    schema = definition.input_schema
    assert "name" in schema["properties"]
    assert schema["required"] == ["name"]
    assert schema["additionalProperties"] is False
