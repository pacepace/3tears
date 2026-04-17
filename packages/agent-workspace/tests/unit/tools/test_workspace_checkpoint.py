"""tests for ``threetears.workspace.checkpoint`` -- WorkspaceCheckpointTool."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

import pytest

from threetears.agent.tools.base_tool import MCPToolDefinition

from threetears.agent.workspace.tools.workspace_checkpoint import (
    WorkspaceCheckpointTool,
)


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeWorkspaceEntity:
    id: UUID
    name: str
    date_deleted: Any = None


class _FakeWorkspaceCollection:
    def __init__(self, entities: list[_FakeWorkspaceEntity]) -> None:
        self._entities = entities

    async def find_by_agent_and_name(
        self, agent_id: UUID, name: str
    ) -> _FakeWorkspaceEntity | None:
        for e in self._entities:
            if e.name == name:
                return e
        return None

    async def find_by_id_and_agent(
        self, workspace_id: UUID, agent_id: UUID
    ) -> _FakeWorkspaceEntity | None:
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


class _FakeFileCollection:
    def __init__(self, files: list[_FakeFileEntity]) -> None:
        self._files = files
        self.calls: list[UUID] = []

    async def find_by_workspace(
        self, workspace_id: UUID
    ) -> list[_FakeFileEntity]:
        self.calls.append(workspace_id)
        return list(self._files)


class _FakeVersionCollection:
    """placeholder -- checkpoint writes journal rows through the pool conn."""


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
    executions: list[tuple[str, tuple[Any, ...], bool]] = field(default_factory=list)
    transactions: list[_FakeTransaction] = field(default_factory=list)
    transaction_open: bool = False

    def transaction(self) -> _FakeTransaction:
        tx = _FakeTransaction(parent=self)
        self.transactions.append(tx)
        return tx

    async def execute(self, query: str, *args: Any) -> str:
        self.executions.append((query, args, self.transaction_open))
        return "INSERT 0 1"


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


class _FakeContext:
    pass


def _build_tool(
    *,
    workspace_entities: list[_FakeWorkspaceEntity],
    files: list[_FakeFileEntity],
    agent_id: UUID | None = None,
) -> tuple[WorkspaceCheckpointTool, _FakePool, UUID]:
    workspaces = _FakeWorkspaceCollection(workspace_entities)
    file_coll = _FakeFileCollection(files)
    pool = _FakePool()
    aid = agent_id or uuid4()
    tool = WorkspaceCheckpointTool(
        workspace_collection=workspaces,  # type: ignore[arg-type]
        workspace_file_collection=file_coll,  # type: ignore[arg-type]
        workspace_file_version_collection=_FakeVersionCollection(),  # type: ignore[arg-type]
        context_provider=lambda: _FakeContext(),
        agent_id=aid,
        db_pool=pool,
    )
    return tool, pool, aid


# ---------------------------------------------------------------------------
# happy paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_checkpoint_inserts_one_journal_row_per_current_file() -> None:
    """N head files -> N journal rows with action='checkpoint' and label."""
    ws = _FakeWorkspaceEntity(id=uuid4(), name="ws")
    files = [
        _FakeFileEntity(
            relative_path="a.txt",
            content=b"alpha",
            sha256="a" * 64,
            version=3,
        ),
        _FakeFileEntity(
            relative_path="b.md",
            content=b"# beta",
            sha256="b" * 64,
            version=1,
        ),
    ]
    tool, pool, aid = _build_tool(workspace_entities=[ws], files=files)
    result = await tool.execute(label="v1.0", workspace="ws")
    assert result.success is True, result.error
    assert "2 files" in result.content
    assert "'v1.0'" in result.content

    # exactly one transaction wrapped all writes
    assert len(pool.conn.transactions) == 1
    assert pool.conn.transactions[0].entered is True
    assert pool.conn.transactions[0].exited is True
    # every execute was inside the transaction
    assert len(pool.conn.executions) == 2
    for _sql, _args, in_tx in pool.conn.executions:
        assert in_tx is True

    # each row is action=checkpoint, label=v1.0, actor_id=agent
    paths_seen = set()
    for _sql, args, _ in pool.conn.executions:
        # args positions: id, workspace_id, relative_path, version, content,
        # sha256, action, label, actor_id, correlation_id, date_created
        assert args[6] == "checkpoint"
        assert args[7] == "v1.0"
        assert args[8] == aid
        assert args[1] == ws.id
        paths_seen.add(args[2])
    assert paths_seen == {"a.txt", "b.md"}


@pytest.mark.asyncio
async def test_checkpoint_empty_workspace_reports_zero_files() -> None:
    """no head files -> zero journal rows, success=True with zero-count text."""
    ws = _FakeWorkspaceEntity(id=uuid4(), name="ws")
    tool, pool, _ = _build_tool(workspace_entities=[ws], files=[])
    result = await tool.execute(label="empty", workspace="ws")
    assert result.success is True, result.error
    assert "0 files" in result.content
    assert "'empty'" in result.content
    # transaction opened even with zero writes
    assert len(pool.conn.transactions) == 1
    assert pool.conn.executions == []


# ---------------------------------------------------------------------------
# error paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_checkpoint_rejects_empty_label() -> None:
    """empty label returns clean error; no pool activity."""
    ws = _FakeWorkspaceEntity(id=uuid4(), name="ws")
    tool, pool, _ = _build_tool(workspace_entities=[ws], files=[])
    result = await tool.execute(label="", workspace="ws")
    assert result.success is False
    assert result.error is not None
    assert "label" in result.error
    assert pool.conn.executions == []


@pytest.mark.asyncio
async def test_checkpoint_unknown_workspace_returns_clean_error() -> None:
    """unknown workspace name -> clean error."""
    tool, _pool, _ = _build_tool(workspace_entities=[], files=[])
    result = await tool.execute(label="v1", workspace="ghost")
    assert result.success is False
    assert result.error is not None
    assert "ghost" in result.error


# ---------------------------------------------------------------------------
# anti-pattern: checkpoint MUST NOT accept sandbox in its constructor
# ---------------------------------------------------------------------------


def test_checkpoint_constructor_has_no_sandbox_parameter() -> None:
    """WorkspaceCheckpointTool deliberately omits sandbox from __init__."""
    signature = inspect.signature(WorkspaceCheckpointTool.__init__)
    assert "sandbox" not in signature.parameters


# ---------------------------------------------------------------------------
# MCP surface
# ---------------------------------------------------------------------------


def test_checkpoint_mcp_name_is_exact_string() -> None:
    tool, _, _ = _build_tool(
        workspace_entities=[_FakeWorkspaceEntity(id=uuid4(), name="ws")],
        files=[],
    )
    assert tool.mcp_name() == "threetears.workspace.checkpoint"


def test_checkpoint_mcp_schema_shape() -> None:
    tool, _, _ = _build_tool(
        workspace_entities=[_FakeWorkspaceEntity(id=uuid4(), name="ws")],
        files=[],
    )
    defn = tool.mcp_schema()
    assert isinstance(defn, MCPToolDefinition)
    schema = defn.input_schema
    assert schema["required"] == ["label"]
    assert schema["additionalProperties"] is False
    props = schema["properties"]
    assert set(props.keys()) == {"label", "workspace"}
    assert props["label"]["minLength"] == 1
    assert props["label"]["maxLength"] == 255
