"""tests for ``threetears.workspace.fs_read`` -- FsReadTool."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from threetears.agent.tools.base_tool import MCPToolDefinition
from threetears.core.security import SandboxDecision, SandboxDenied

from threetears.agent.workspace.tools import helpers as helpers_module
from threetears.agent.workspace.tools.fs_read import FsReadTool
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
        self.find_calls: list[tuple[UUID, str]] = []

    async def find_by_workspace_and_relative_path(
        self, workspace_id: UUID, relative_path: str
    ) -> _FakeFileEntity | None:
        self.find_calls.append((workspace_id, relative_path))
        for f in self._files:
            if f.relative_path == relative_path:
                return f
        return None


class _RecordingSandbox:
    """records validate_syntax calls; raises on syntactically-invalid targets.

    namespace-task-01 phase 7: the glob-driven enforce / check_relative_key
    surface is retired from workspace tools. tests still need a stand-in
    for the sandbox so they can assert the tool called validate_syntax
    before reaching the rbac gate. denylist entries simulate the syntactic
    rejection surface (empty, absolute, ``..``, control chars); the
    path-level authorization decision lives in the injected acl_cache.
    """

    def __init__(self, deny_reads: list[str] | None = None) -> None:
        self._deny_syntax = set(deny_reads or [])
        self.syntax_calls: list[str] = []

    def validate_syntax(self, target: str) -> None:
        self.syntax_calls.append(target)
        if target in self._deny_syntax:
            raise SandboxDenied("access", target, "syntactic deny (test fixture)")


class _FakeContext(FakeWorkspaceContext):
    pass


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _build_tool(
    *,
    acl_cache: Any,
    workspace_entities: list[_FakeWorkspaceEntity] | None = None,
    files: list[_FakeFileEntity] | None = None,
    deny_reads: list[str] | None = None,
    agent_id: UUID | None = None,
) -> tuple[FsReadTool, _FakeFileCollection, _RecordingSandbox, UUID]:
    agent_id = agent_id or uuid4()
    workspaces = _FakeWorkspaceCollection(workspace_entities or [_FakeWorkspaceEntity(id=uuid4(), name="ws")])
    file_coll = _FakeFileCollection(files)
    sandbox = _RecordingSandbox(deny_reads=deny_reads)
    tool = FsReadTool(
        workspace_collection=workspaces,  # type: ignore[arg-type]
        workspace_file_collection=file_coll,  # type: ignore[arg-type]
        sandbox=sandbox,  # type: ignore[arg-type]
        context_provider=lambda: _FakeContext(),
        agent_id=agent_id,
        acl_cache=acl_cache,
    )
    return tool, file_coll, sandbox, agent_id


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fs_read_happy_returns_content_sha_version(
    permissive_acl_cache: MagicMock,
) -> None:
    """explicit workspace + allowed read returns text + metadata."""
    ws = _FakeWorkspaceEntity(id=uuid4(), name="ws")
    file_entity = _FakeFileEntity(
        relative_path="docs/readme.md",
        content=b"hello world",
        sha256="a" * 64,
        version=2,
    )
    tool, files, sandbox, _ = _build_tool(workspace_entities=[ws], files=[file_entity], acl_cache=permissive_acl_cache)

    result = await tool.execute(relative_path="docs/readme.md", workspace="ws")

    assert result.success is True, result.error
    assert result.content == "hello world"
    assert result.metadata == {
        "sha256": "a" * 64,
        "version": 2,
        "is_binary": False,
    }
    assert sandbox.syntax_calls == ["docs/readme.md"]
    assert files.find_calls == [(ws.id, "docs/readme.md")]


@pytest.mark.asyncio
async def test_fs_read_binary_returns_base64_with_is_binary_true(
    permissive_acl_cache: MagicMock,
) -> None:
    """non-UTF-8 content returns base64 and is_binary=True."""
    raw = b"\x89PNG\r\n\x1a\n\xff\xfe"
    ws = _FakeWorkspaceEntity(id=uuid4(), name="ws")
    file_entity = _FakeFileEntity(
        relative_path="img/logo.png",
        content=raw,
        sha256="b" * 64,
        version=1,
    )
    tool, _files, _sandbox, _ = _build_tool(
        workspace_entities=[ws], files=[file_entity], acl_cache=permissive_acl_cache
    )

    result = await tool.execute(relative_path="img/logo.png", workspace="ws")

    assert result.success is True, result.error
    assert result.content == base64.b64encode(raw).decode("ascii")
    assert result.metadata is not None
    assert result.metadata["is_binary"] is True


# ---------------------------------------------------------------------------
# error paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fs_read_missing_file_returns_clean_error(
    permissive_acl_cache: MagicMock,
) -> None:
    """file not in head-state returns ToolResult.success=False."""
    ws = _FakeWorkspaceEntity(id=uuid4(), name="ws")
    tool, _files, _sandbox, _ = _build_tool(workspace_entities=[ws], files=[], acl_cache=permissive_acl_cache)
    result = await tool.execute(relative_path="missing.md", workspace="ws")
    assert result.success is False
    assert result.error is not None
    assert "missing.md" in result.error
    assert "'ws'" in result.error


@pytest.mark.asyncio
async def test_fs_read_sandbox_denied_returns_clean_error(
    permissive_acl_cache: MagicMock,
) -> None:
    """SandboxDenied becomes ToolResult with error text (no raise)."""
    ws = _FakeWorkspaceEntity(id=uuid4(), name="ws")
    tool, files, sandbox, _ = _build_tool(
        workspace_entities=[ws],
        files=[
            _FakeFileEntity(
                relative_path="secret.env",
                content=b"TOKEN=xyz",
                sha256="c" * 64,
                version=1,
            )
        ],
        deny_reads=["secret.env"],
        acl_cache=permissive_acl_cache,
    )
    result = await tool.execute(relative_path="secret.env", workspace="ws")
    assert result.success is False
    assert result.error is not None
    assert "secret.env" in result.error
    # enforce was called BEFORE any file lookup (gate-then-act)
    assert sandbox.syntax_calls == ["secret.env"]
    assert files.find_calls == []


@pytest.mark.asyncio
async def test_fs_read_unknown_workspace_name_returns_clean_error(
    permissive_acl_cache: MagicMock,
) -> None:
    """unknown workspace name returns ToolResult.success=False."""
    tool, _files, _sandbox, _ = _build_tool(workspace_entities=[], acl_cache=permissive_acl_cache)
    result = await tool.execute(relative_path="x", workspace="ghost")
    assert result.success is False
    assert result.error is not None
    assert "ghost" in result.error


@pytest.mark.asyncio
async def test_fs_read_no_workspace_and_no_pin_returns_clean_error(
    monkeypatch: pytest.MonkeyPatch,
    permissive_acl_cache: MagicMock,
) -> None:
    """omitted workspace + no pin returns ToolResult.success=False."""

    async def _no_pin(context: Any) -> Any:
        return None

    monkeypatch.setattr(helpers_module.pin_module, "get_pin", _no_pin)
    tool, _files, _sandbox, _ = _build_tool(acl_cache=permissive_acl_cache)
    result = await tool.execute(relative_path="x")
    assert result.success is False
    assert result.error is not None
    assert "pinned" in result.error.lower()


# ---------------------------------------------------------------------------
# MCP surface
# ---------------------------------------------------------------------------


def test_fs_read_mcp_name_is_exact_string(
    permissive_acl_cache: MagicMock,
) -> None:
    tool, _, _, _ = _build_tool(acl_cache=permissive_acl_cache)
    assert tool.mcp_name() == "threetears.workspace.fs_read"


def test_fs_read_mcp_schema_declares_required_relative_path(
    permissive_acl_cache: MagicMock,
) -> None:
    tool, _, _, _ = _build_tool(acl_cache=permissive_acl_cache)
    defn = tool.mcp_schema()
    assert isinstance(defn, MCPToolDefinition)
    assert defn.input_schema["required"] == ["relative_path"]
    assert defn.input_schema["additionalProperties"] is False
    props = defn.input_schema["properties"]
    assert set(props.keys()) == {"relative_path", "workspace"}
