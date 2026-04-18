"""tests for ``threetears.workspace.fs_list`` -- FsListTool."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from threetears.agent.tools.base_tool import MCPToolDefinition
from threetears.core.security import SandboxDecision

from threetears.agent.workspace.tools.fs_list import FsListTool


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeWorkspaceEntity:
    id: UUID
    name: str
    date_deleted: datetime | None = None


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
    content: bytes = b""
    sha256: str = "a" * 64
    version: int = 1
    date_updated: datetime = datetime(2026, 4, 16, 12, 0, 0, tzinfo=UTC)


class _FakeFileCollection:
    def __init__(self, files: list[_FakeFileEntity]) -> None:
        self._files = files

    async def find_by_workspace(self, workspace_id: UUID) -> list[_FakeFileEntity]:
        return list(self._files)


class _FilteringSandbox:
    """allows everything by default; denies reads on a configurable set."""

    def __init__(self, deny_reads: list[str] | None = None) -> None:
        self._deny_reads = set(deny_reads or [])

    def check_relative_key(self, key: str, mode: str) -> SandboxDecision:
        if mode == "read" and key in self._deny_reads:
            return SandboxDecision.DENY
        return SandboxDecision.ALLOW


class _FakeContext:
    pass


def _build_tool(
    *,
    files: list[_FakeFileEntity],
    acl_cache: Any,
    deny_reads: list[str] | None = None,
) -> FsListTool:
    ws = _FakeWorkspaceEntity(id=uuid4(), name="ws")
    return FsListTool(
        workspace_collection=_FakeWorkspaceCollection([ws]),  # type: ignore[arg-type]
        workspace_file_collection=_FakeFileCollection(files),  # type: ignore[arg-type]
        sandbox=_FilteringSandbox(deny_reads=deny_reads),  # type: ignore[arg-type]
        context_provider=lambda: _FakeContext(),
        agent_id=uuid4(),
        acl_cache=acl_cache,
    )


# ---------------------------------------------------------------------------
# happy paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fs_list_no_glob_returns_all_files(
    permissive_acl_cache: MagicMock,
) -> None:
    """without glob, every allowed file shows up."""
    tool = _build_tool(
        files=[
            _FakeFileEntity(relative_path="a.md"),
            _FakeFileEntity(relative_path="b.yaml"),
            _FakeFileEntity(relative_path="src/main.py"),
        ],
        acl_cache=permissive_acl_cache,
    )
    result = await tool.execute(workspace="ws")
    assert result.success is True, result.error
    entries = json.loads(result.content)
    paths = sorted(e["relative_path"] for e in entries)
    assert paths == ["a.md", "b.yaml", "src/main.py"]
    assert result.metadata == {"count": 3}


@pytest.mark.asyncio
async def test_fs_list_glob_filters_recursively_with_double_star(
    permissive_acl_cache: MagicMock,
) -> None:
    """``**/*.yaml`` matches single- and multi-segment paths."""
    tool = _build_tool(
        files=[
            _FakeFileEntity(relative_path="top.yaml"),
            _FakeFileEntity(relative_path="docs/a.yaml"),
            _FakeFileEntity(relative_path="docs/nested/b.yaml"),
            _FakeFileEntity(relative_path="readme.md"),
        ],
        acl_cache=permissive_acl_cache,
    )
    result = await tool.execute(glob="**/*.yaml", workspace="ws")
    assert result.success is True, result.error
    paths = sorted(e["relative_path"] for e in json.loads(result.content))
    assert paths == ["docs/a.yaml", "docs/nested/b.yaml", "top.yaml"]


@pytest.mark.asyncio
async def test_fs_list_sandbox_read_filters_out_denied(
    permissive_acl_cache: MagicMock,
) -> None:
    """files denied by sandbox read are silently dropped (not errors)."""
    tool = _build_tool(
        files=[
            _FakeFileEntity(relative_path="public.md"),
            _FakeFileEntity(relative_path="secret.env"),
        ],
        deny_reads=["secret.env"],
        acl_cache=permissive_acl_cache,
    )
    result = await tool.execute(workspace="ws")
    assert result.success is True, result.error
    paths = [e["relative_path"] for e in json.loads(result.content)]
    assert paths == ["public.md"]


@pytest.mark.asyncio
async def test_fs_list_entries_carry_sha_version_and_iso_date(
    permissive_acl_cache: MagicMock,
) -> None:
    """each entry carries relative_path, sha256, version, date_updated (iso)."""
    file_entity = _FakeFileEntity(
        relative_path="x",
        sha256="1" * 64,
        version=3,
        date_updated=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
    )
    tool = _build_tool(files=[file_entity], acl_cache=permissive_acl_cache)
    result = await tool.execute(workspace="ws")
    entries = json.loads(result.content)
    assert entries[0] == {
        "relative_path": "x",
        "sha256": "1" * 64,
        "version": 3,
        "date_updated": "2026-01-02T03:04:05+00:00",
    }


# ---------------------------------------------------------------------------
# error paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fs_list_unknown_workspace_returns_clean_error(
    permissive_acl_cache: MagicMock,
) -> None:
    tool = _build_tool(files=[], acl_cache=permissive_acl_cache)
    result = await tool.execute(workspace="ghost")
    assert result.success is False
    assert result.error is not None
    assert "ghost" in result.error


# ---------------------------------------------------------------------------
# MCP surface
# ---------------------------------------------------------------------------


def test_fs_list_mcp_name_is_exact_string(
    permissive_acl_cache: MagicMock,
) -> None:
    tool = _build_tool(files=[], acl_cache=permissive_acl_cache)
    assert tool.mcp_name() == "threetears.workspace.fs_list"


def test_fs_list_mcp_schema_no_required_fields(
    permissive_acl_cache: MagicMock,
) -> None:
    tool = _build_tool(files=[], acl_cache=permissive_acl_cache)
    defn = tool.mcp_schema()
    assert isinstance(defn, MCPToolDefinition)
    assert defn.input_schema["required"] == []
    assert defn.input_schema["additionalProperties"] is False
    props = defn.input_schema["properties"]
    assert set(props.keys()) == {"glob", "workspace"}
