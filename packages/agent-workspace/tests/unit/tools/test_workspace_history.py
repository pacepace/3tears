"""tests for ``threetears.workspace.history`` -- WorkspaceHistoryTool."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from threetears.agent.tools.base_tool import MCPToolDefinition
from threetears.core.security import SandboxDecision, SandboxDenied

from threetears.agent.workspace.tools.workspace_history import (
    WorkspaceHistoryTool,
)


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeWorkspaceEntity:
    id: UUID
    name: str
    date_deleted: Any = None

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
class _FakeVersionRow:
    relative_path: str
    version: int
    action: str
    label: str | None
    actor_id: UUID
    correlation_id: UUID
    date_created: datetime
    sha256: str
    content: bytes


class _FakeFileCollection:
    """placeholder: history tool does not query this collection."""


class _FakeVersionCollection:
    def __init__(self, rows: list[_FakeVersionRow]) -> None:
        self._rows = rows
        self.by_workspace_calls: list[tuple[UUID, int]] = []
        self.by_path_calls: list[tuple[UUID, str, int]] = []

    async def find_by_workspace(self, workspace_id: UUID, limit: int) -> list[_FakeVersionRow]:
        self.by_workspace_calls.append((workspace_id, limit))
        ordered = sorted(self._rows, key=lambda r: r.date_created, reverse=True)
        return ordered[:limit]

    async def find_by_workspace_and_path(
        self, workspace_id: UUID, relative_path: str, limit: int
    ) -> list[_FakeVersionRow]:
        self.by_path_calls.append((workspace_id, relative_path, limit))
        filtered = [r for r in self._rows if r.relative_path == relative_path]
        ordered = sorted(filtered, key=lambda r: r.date_created, reverse=True)
        return ordered[:limit]


class _RecordingSandbox:
    """validate_syntax raises for syntactically-denied paths.

    namespace-task-01 phase 7: the glob-driven enforce / check_relative_key
    surface is retired. the rbac per-file gate is stubbed via the
    ``stub_authorize_workspace_file_access`` autouse fixture in
    ``conftest.py``. tests that want to simulate per-path denial
    override that fixture locally; this sandbox stand-in only fields
    syntactic rejections.
    """

    def __init__(
        self,
        deny_reads: list[str] | None = None,
    ) -> None:
        self._deny_reads = set(deny_reads or [])
        self.syntax_calls: list[str] = []

    def validate_syntax(self, target: str) -> None:
        self.syntax_calls.append(target)
        if target in self._deny_reads:
            raise SandboxDenied("access", target, "syntactic deny (test fixture)")


class _FakeContext:
    pass


def _row(
    *,
    relative_path: str,
    version: int,
    action: str = "update",
    label: str | None = None,
    content: bytes = b"payload",
    offset_seconds: int = 0,
) -> _FakeVersionRow:
    return _FakeVersionRow(
        relative_path=relative_path,
        version=version,
        action=action,
        label=label,
        actor_id=uuid4(),
        correlation_id=uuid4(),
        date_created=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=offset_seconds),
        sha256="s" * 64,
        content=content,
    )


def _build_tool(
    *,
    workspace_entities: list[_FakeWorkspaceEntity],
    version_rows: list[_FakeVersionRow],
    acl_cache: Any,
    deny_reads: list[str] | None = None,
) -> tuple[WorkspaceHistoryTool, _FakeVersionCollection, _RecordingSandbox]:
    workspaces = _FakeWorkspaceCollection(workspace_entities)
    versions = _FakeVersionCollection(version_rows)
    sandbox = _RecordingSandbox(deny_reads=deny_reads)
    tool = WorkspaceHistoryTool(
        workspace_collection=workspaces,  # type: ignore[arg-type]
        workspace_file_collection=_FakeFileCollection(),  # type: ignore[arg-type]
        workspace_file_version_collection=versions,  # type: ignore[arg-type]
        sandbox=sandbox,  # type: ignore[arg-type]
        context_provider=lambda: _FakeContext(),
        agent_id=uuid4(),
        acl_cache=acl_cache,
    )
    return tool, versions, sandbox


# ---------------------------------------------------------------------------
# happy paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_history_workspace_wide_returns_newest_first_without_content(
    permissive_acl_cache: MagicMock,
) -> None:
    """no relative_path: all journal rows newest-first, no content blob in output."""
    ws = _FakeWorkspaceEntity(id=uuid4(), name="ws")
    rows = [
        _row(
            relative_path="a.txt",
            version=1,
            content=b"hello",
            offset_seconds=0,
        ),
        _row(
            relative_path="b.md",
            version=1,
            content=b"world-long",
            offset_seconds=10,
        ),
        _row(
            relative_path="a.txt",
            version=2,
            content=b"hi",
            offset_seconds=20,
        ),
    ]
    tool, versions, _ = _build_tool(workspace_entities=[ws], version_rows=rows, acl_cache=permissive_acl_cache)
    result = await tool.execute(workspace="ws")
    assert result.success is True, result.error
    payload = json.loads(result.content)
    assert len(payload) == 3
    # newest-first
    assert payload[0]["relative_path"] == "a.txt"
    assert payload[0]["version"] == 2
    # no content field
    for entry in payload:
        assert "content" not in entry
        assert "size_bytes" in entry
    # size_bytes is computed from content length
    assert payload[0]["size_bytes"] == len(b"hi")
    assert versions.by_workspace_calls == [(ws.id, 50)]


@pytest.mark.asyncio
async def test_history_per_path_calls_enforce_and_narrow_query(
    permissive_acl_cache: MagicMock,
) -> None:
    """relative_path triggers sandbox.enforce('read', path) and path-scoped query."""
    ws = _FakeWorkspaceEntity(id=uuid4(), name="ws")
    rows = [
        _row(relative_path="a.txt", version=1, offset_seconds=0),
        _row(relative_path="b.md", version=1, offset_seconds=5),
        _row(relative_path="a.txt", version=2, offset_seconds=10),
    ]
    tool, versions, sandbox = _build_tool(workspace_entities=[ws], version_rows=rows, acl_cache=permissive_acl_cache)
    result = await tool.execute(relative_path="a.txt", workspace="ws")
    assert result.success is True, result.error
    payload = json.loads(result.content)
    assert {entry["relative_path"] for entry in payload} == {"a.txt"}
    assert len(payload) == 2
    assert sandbox.syntax_calls == ["a.txt"]
    assert versions.by_path_calls == [(ws.id, "a.txt", 50)]
    assert versions.by_workspace_calls == []


@pytest.mark.asyncio
async def test_history_filters_sandbox_denied_rows_when_no_path(
    permissive_acl_cache: MagicMock,
) -> None:
    """workspace-wide history drops rows whose path fails sandbox read."""
    ws = _FakeWorkspaceEntity(id=uuid4(), name="ws")
    rows = [
        _row(relative_path="public.md", version=1, offset_seconds=0),
        _row(relative_path="secret.env", version=1, offset_seconds=5),
        _row(relative_path="public.md", version=2, offset_seconds=10),
    ]
    tool, _, _ = _build_tool(
        workspace_entities=[ws],
        version_rows=rows,
        deny_reads=["secret.env"],
        acl_cache=permissive_acl_cache,
    )
    result = await tool.execute(workspace="ws")
    assert result.success is True
    payload = json.loads(result.content)
    assert {entry["relative_path"] for entry in payload} == {"public.md"}


@pytest.mark.asyncio
async def test_history_sandbox_denied_relative_path_returns_clean_error(
    permissive_acl_cache: MagicMock,
) -> None:
    """per-path history with a denied read path surfaces a clean error."""
    ws = _FakeWorkspaceEntity(id=uuid4(), name="ws")
    tool, _versions, _sandbox = _build_tool(
        workspace_entities=[ws],
        version_rows=[_row(relative_path="secret.env", version=1)],
        deny_reads=["secret.env"],
        acl_cache=permissive_acl_cache,
    )
    result = await tool.execute(relative_path="secret.env", workspace="ws")
    assert result.success is False
    assert result.error is not None
    assert "secret.env" in result.error


@pytest.mark.asyncio
async def test_history_limit_is_capped_at_max(
    permissive_acl_cache: MagicMock,
) -> None:
    """out-of-range limit is clamped to max (500)."""
    ws = _FakeWorkspaceEntity(id=uuid4(), name="ws")
    tool, versions, _ = _build_tool(
        workspace_entities=[ws],
        version_rows=[_row(relative_path="a", version=1)],
        acl_cache=permissive_acl_cache,
    )
    result = await tool.execute(workspace="ws", limit=10_000)
    assert result.success is True
    assert versions.by_workspace_calls == [(ws.id, 500)]


@pytest.mark.asyncio
async def test_history_limit_is_clamped_to_at_least_one(
    permissive_acl_cache: MagicMock,
) -> None:
    """zero/negative limit is clamped to the floor of 1."""
    ws = _FakeWorkspaceEntity(id=uuid4(), name="ws")
    tool, versions, _ = _build_tool(
        workspace_entities=[ws],
        version_rows=[_row(relative_path="a", version=1)],
        acl_cache=permissive_acl_cache,
    )
    result = await tool.execute(workspace="ws", limit=0)
    assert result.success is True
    assert versions.by_workspace_calls == [(ws.id, 1)]


@pytest.mark.asyncio
async def test_history_unknown_workspace_returns_clean_error(
    permissive_acl_cache: MagicMock,
) -> None:
    """unknown workspace name yields clean error."""
    tool, _versions, _ = _build_tool(workspace_entities=[], version_rows=[], acl_cache=permissive_acl_cache)
    result = await tool.execute(workspace="ghost")
    assert result.success is False
    assert result.error is not None
    assert "ghost" in result.error


# ---------------------------------------------------------------------------
# MCP surface
# ---------------------------------------------------------------------------


def test_history_mcp_name_is_exact_string(
    permissive_acl_cache: MagicMock,
) -> None:
    tool, _, _ = _build_tool(
        workspace_entities=[_FakeWorkspaceEntity(id=uuid4(), name="ws")],
        version_rows=[],
        acl_cache=permissive_acl_cache,
    )
    assert tool.mcp_name() == "threetears.workspace.history"


def test_history_mcp_schema_shape(
    permissive_acl_cache: MagicMock,
) -> None:
    tool, _, _ = _build_tool(
        workspace_entities=[_FakeWorkspaceEntity(id=uuid4(), name="ws")],
        version_rows=[],
        acl_cache=permissive_acl_cache,
    )
    defn = tool.mcp_schema()
    assert isinstance(defn, MCPToolDefinition)
    schema = defn.input_schema
    assert schema["required"] == []
    assert schema["additionalProperties"] is False
    props = schema["properties"]
    assert set(props.keys()) == {"relative_path", "limit", "workspace"}
    assert props["limit"]["minimum"] == 1
    assert props["limit"]["maximum"] == 500
