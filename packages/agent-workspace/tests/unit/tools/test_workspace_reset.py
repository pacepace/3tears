"""tests for ``threetears.workspace.reset`` -- WorkspaceResetTool."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest

from threetears.agent.tools.base_tool import MCPToolDefinition

from threetears.agent.workspace.tools import workspace_reset as workspace_reset_module
from threetears.agent.workspace.tools.workspace_reset import WorkspaceResetTool


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeWorkspaceEntity:
    """minimal stand-in for :class:`Workspace` for reset-target lookups."""

    id: UUID
    name: str
    template_name: str | None
    current_version: int = 0
    date_deleted: Any = None


@dataclass
class _FakeFileEntity:
    """minimal stand-in for :class:`WorkspaceFile`."""

    relative_path: str
    content: bytes
    sha256: str


class _FakeWorkspaceCollection:
    def __init__(self, entities: list[_FakeWorkspaceEntity]) -> None:
        self._entities = entities

    async def find_by_agent_and_name(self, agent_id: UUID, name: str) -> _FakeWorkspaceEntity | None:
        for e in self._entities:
            if e.name == name:
                return e
        return None


class _FakeFileCollection:
    def __init__(self, files: list[_FakeFileEntity]) -> None:
        self._files = files
        self.find_calls: list[UUID] = []

    async def find_by_workspace(self, workspace_id: UUID) -> list[_FakeFileEntity]:
        self.find_calls.append(workspace_id)
        return list(self._files)


class _FakeVersionCollection:
    """placeholder; tool uses pool directly."""


class _RecordingSandbox:
    def __init__(self, templates_root: Path) -> None:
        self._templates_root = templates_root
        self.resolve_calls: list[tuple[str, str]] = []
        self.enforce_calls: list[tuple[str, str]] = []

    def resolve_fs_path(self, path: str, root_name: str) -> Path:
        self.resolve_calls.append((path, root_name))
        return self._templates_root

    def enforce(self, action: str, target: str) -> None:
        self.enforce_calls.append((action, target))


class _NoTemplateSandbox:
    def resolve_fs_path(self, path: str, root_name: str) -> Path:
        raise KeyError(root_name)

    def enforce(self, action: str, target: str) -> None:
        return None


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
    transaction_calls: int = 0
    transactions: list[_FakeTransaction] = field(default_factory=list)
    transaction_open: bool = False

    def transaction(self) -> _FakeTransaction:
        self.transaction_calls += 1
        tx = _FakeTransaction(parent=self)
        self.transactions.append(tx)
        return tx

    async def execute(self, query: str, *args: Any) -> str:
        self.executions.append((query, args, self.transaction_open))
        return "OK"


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
    acquire_calls: int = 0

    def acquire(self) -> _FakeAcquireCM:
        self.acquire_calls += 1
        return _FakeAcquireCM(conn=self.conn)


class _FakeContext:
    pass


@dataclass
class _PinnedSnapshot:
    workspace_id: UUID
    workspace_name: str


def _build_tool(
    *,
    workspace_collection: Any,
    workspace_file_collection: Any,
    sandbox: Any,
    db_pool: Any,
    agent_id: UUID | None = None,
    context_provider: Any = None,
) -> WorkspaceResetTool:
    return WorkspaceResetTool(
        workspace_collection=workspace_collection,
        workspace_file_collection=workspace_file_collection,
        workspace_file_version_collection=_FakeVersionCollection(),
        sandbox=sandbox,
        context_provider=context_provider or (lambda: _FakeContext()),
        agent_id=agent_id or uuid4(),
        db_pool=db_pool,
    )


# ---------------------------------------------------------------------------
# happy paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reset_happy_path_reverts_files_and_advances_version(
    tmp_path: Path,
) -> None:
    """reset re-issues template content for paths in both, journals revert, advances version."""
    template_dir = tmp_path / "starter"
    template_dir.mkdir()
    (template_dir / "README.md").write_bytes(b"# original\n")
    (template_dir / "main.py").write_bytes(b"original = True\n")

    ws_id = uuid4()
    workspace = _FakeWorkspaceEntity(id=ws_id, name="seed", template_name="starter", current_version=4)
    current_files = [
        _FakeFileEntity(relative_path="README.md", content=b"# tampered\n", sha256="x" * 64),
        _FakeFileEntity(relative_path="main.py", content=b"hacked = True\n", sha256="y" * 64),
    ]
    pool = _FakePool()
    tool = _build_tool(
        workspace_collection=_FakeWorkspaceCollection([workspace]),
        workspace_file_collection=_FakeFileCollection(current_files),
        sandbox=_RecordingSandbox(template_dir),
        db_pool=pool,
    )

    result = await tool.execute(name="seed")

    assert result.success is True, result.error
    assert "reset workspace 'seed'" in result.content
    assert "to template 'starter'" in result.content
    assert "2 files affected" in result.content

    # transaction wrapped all writes
    assert pool.conn.transaction_calls == 1
    assert pool.conn.transactions[0].entered is True
    assert pool.conn.transactions[0].exited is True
    assert all(in_tx for _sql, _args, in_tx in pool.conn.executions)

    # 2 file upserts, 2 journal revert rows, 1 version update
    file_upserts = [e for e in pool.conn.executions if "INSERT INTO workspace_files" in e[0]]
    journal_inserts = [e for e in pool.conn.executions if "INSERT INTO workspace_file_versions" in e[0]]
    version_updates = [e for e in pool.conn.executions if "UPDATE workspaces SET current_version" in e[0]]
    assert len(file_upserts) == 2
    assert len(journal_inserts) == 2
    assert len(version_updates) == 1

    # all journal rows have action="revert" and version = 5
    for _sql, args, _intx in journal_inserts:
        assert args[6] == "revert"
        assert args[3] == 5

    # version update sets current_version=5 for our workspace
    upd_args = version_updates[0][1]
    assert upd_args[0] == 5
    assert upd_args[2] == ws_id


@pytest.mark.asyncio
async def test_reset_drops_files_template_no_longer_has(tmp_path: Path) -> None:
    """files present in workspace but absent from template are DELETED with delete journal row."""
    template_dir = tmp_path / "starter"
    template_dir.mkdir()
    (template_dir / "keep.md").write_bytes(b"keep me\n")

    ws_id = uuid4()
    workspace = _FakeWorkspaceEntity(id=ws_id, name="seed", template_name="starter", current_version=1)
    current_files = [
        _FakeFileEntity(relative_path="keep.md", content=b"old\n", sha256="a" * 64),
        _FakeFileEntity(relative_path="dropped.md", content=b"orphan\n", sha256="b" * 64),
    ]
    pool = _FakePool()
    tool = _build_tool(
        workspace_collection=_FakeWorkspaceCollection([workspace]),
        workspace_file_collection=_FakeFileCollection(current_files),
        sandbox=_RecordingSandbox(template_dir),
        db_pool=pool,
    )

    result = await tool.execute(name="seed")

    assert result.success is True, result.error
    deletes = [e for e in pool.conn.executions if "DELETE FROM workspace_files" in e[0]]
    assert len(deletes) == 1
    assert deletes[0][1][1] == "dropped.md"

    journal_inserts = [e for e in pool.conn.executions if "INSERT INTO workspace_file_versions" in e[0]]
    actions = sorted(args[6] for _sql, args, _intx in journal_inserts)
    assert actions == ["delete", "revert"]


@pytest.mark.asyncio
async def test_reset_creates_files_template_added(tmp_path: Path) -> None:
    """files in template but absent from workspace get INSERTed with create journal row."""
    template_dir = tmp_path / "starter"
    template_dir.mkdir()
    (template_dir / "old.md").write_bytes(b"old\n")
    (template_dir / "new.md").write_bytes(b"new\n")

    ws_id = uuid4()
    workspace = _FakeWorkspaceEntity(id=ws_id, name="seed", template_name="starter", current_version=2)
    current_files = [
        _FakeFileEntity(relative_path="old.md", content=b"old\n", sha256="o" * 64),
    ]
    pool = _FakePool()
    tool = _build_tool(
        workspace_collection=_FakeWorkspaceCollection([workspace]),
        workspace_file_collection=_FakeFileCollection(current_files),
        sandbox=_RecordingSandbox(template_dir),
        db_pool=pool,
    )

    result = await tool.execute(name="seed")

    assert result.success is True
    journal_inserts = [e for e in pool.conn.executions if "INSERT INTO workspace_file_versions" in e[0]]
    by_action = {args[6]: args for _sql, args, _intx in journal_inserts}
    assert "create" in by_action
    assert by_action["create"][2] == "new.md"
    assert "revert" in by_action


@pytest.mark.asyncio
async def test_reset_uses_pin_when_name_omitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """omitting name falls back to the pinned workspace via pin.get_pin."""
    template_dir = tmp_path / "starter"
    template_dir.mkdir()
    (template_dir / "x.md").write_bytes(b"x\n")

    ws_id = uuid4()
    workspace = _FakeWorkspaceEntity(id=ws_id, name="pinned", template_name="starter", current_version=0)

    async def _get_pin(_ctx: Any) -> _PinnedSnapshot:
        return _PinnedSnapshot(workspace_id=ws_id, workspace_name="pinned")

    monkeypatch.setattr(workspace_reset_module.pin, "get_pin", _get_pin)

    pool = _FakePool()
    tool = _build_tool(
        workspace_collection=_FakeWorkspaceCollection([workspace]),
        workspace_file_collection=_FakeFileCollection([]),
        sandbox=_RecordingSandbox(template_dir),
        db_pool=pool,
    )

    result = await tool.execute()
    assert result.success is True, result.error
    assert "pinned" in result.content


# ---------------------------------------------------------------------------
# error paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reset_no_name_and_no_pin_returns_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """omitting name with no pin returns clean error."""

    async def _get_pin(_ctx: Any) -> Any:
        return None

    monkeypatch.setattr(workspace_reset_module.pin, "get_pin", _get_pin)

    pool = _FakePool()
    tool = _build_tool(
        workspace_collection=_FakeWorkspaceCollection([]),
        workspace_file_collection=_FakeFileCollection([]),
        sandbox=_NoTemplateSandbox(),
        db_pool=pool,
    )
    result = await tool.execute()
    assert result.success is False
    assert result.error is not None
    assert "no workspace name provided and none pinned" in result.error
    assert pool.conn.executions == []


@pytest.mark.asyncio
async def test_reset_workspace_not_found_returns_error() -> None:
    """missing workspace yields clean error and no writes."""
    pool = _FakePool()
    tool = _build_tool(
        workspace_collection=_FakeWorkspaceCollection([]),
        workspace_file_collection=_FakeFileCollection([]),
        sandbox=_NoTemplateSandbox(),
        db_pool=pool,
    )
    result = await tool.execute(name="ghost")
    assert result.success is False
    assert "'ghost'" in result.error  # type: ignore[operator]
    assert pool.conn.executions == []


@pytest.mark.asyncio
async def test_reset_workspace_without_template_returns_error() -> None:
    """workspace without template_name yields clean error and no writes."""
    workspace = _FakeWorkspaceEntity(id=uuid4(), name="empty", template_name=None, current_version=0)
    pool = _FakePool()
    tool = _build_tool(
        workspace_collection=_FakeWorkspaceCollection([workspace]),
        workspace_file_collection=_FakeFileCollection([]),
        sandbox=_NoTemplateSandbox(),
        db_pool=pool,
    )
    result = await tool.execute(name="empty")
    assert result.success is False
    assert result.error is not None
    assert "no template" in result.error
    assert pool.conn.executions == []


@pytest.mark.asyncio
async def test_reset_traps_unexpected_exceptions_as_data() -> None:
    """unexpected runtime errors surface as ToolResult(success=False)."""

    class _Boom:
        async def find_by_agent_and_name(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("storage offline")

    pool = _FakePool()
    tool = _build_tool(
        workspace_collection=_Boom(),
        workspace_file_collection=_FakeFileCollection([]),
        sandbox=_NoTemplateSandbox(),
        db_pool=pool,
    )
    result = await tool.execute(name="x")
    assert result.success is False
    assert result.error is not None
    assert "storage offline" in result.error


# ---------------------------------------------------------------------------
# MCP surface
# ---------------------------------------------------------------------------


def test_mcp_name_is_exact_string() -> None:
    pool = _FakePool()
    tool = _build_tool(
        workspace_collection=_FakeWorkspaceCollection([]),
        workspace_file_collection=_FakeFileCollection([]),
        sandbox=_NoTemplateSandbox(),
        db_pool=pool,
    )
    assert tool.mcp_name() == "threetears.workspace.reset"


def test_mcp_version_is_semver_string() -> None:
    pool = _FakePool()
    tool = _build_tool(
        workspace_collection=_FakeWorkspaceCollection([]),
        workspace_file_collection=_FakeFileCollection([]),
        sandbox=_NoTemplateSandbox(),
        db_pool=pool,
    )
    assert tool.mcp_version() == "1.0"


def test_mcp_schema_declares_optional_name() -> None:
    pool = _FakePool()
    tool = _build_tool(
        workspace_collection=_FakeWorkspaceCollection([]),
        workspace_file_collection=_FakeFileCollection([]),
        sandbox=_NoTemplateSandbox(),
        db_pool=pool,
    )
    definition = tool.mcp_schema()
    assert isinstance(definition, MCPToolDefinition)
    assert definition.name == "threetears.workspace.reset"
    schema = definition.input_schema
    assert "name" in schema["properties"]
    assert schema["required"] == []
    assert schema["additionalProperties"] is False
