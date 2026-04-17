"""tests for ``threetears.workspace.list`` -- WorkspaceListTool."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest

from threetears.agent.tools.base_tool import MCPToolDefinition

from threetears.agent.workspace.tools.workspace_list import WorkspaceListTool


@dataclass
class _FakeWorkspaceEntity:
    """minimal stand-in for :class:`Workspace` exposing only what the tool reads."""

    name: str
    description: str | None
    date_updated: datetime


class _FakeCollection:
    """records the agent_id passed to find_by_agent and returns a fixed list."""

    def __init__(self, entities: list[_FakeWorkspaceEntity]) -> None:
        self._entities = entities
        self.calls: list[UUID] = []

    async def find_by_agent(self, agent_id: UUID) -> list[_FakeWorkspaceEntity]:
        self.calls.append(agent_id)
        return self._entities


class _FailingCollection:
    """raises on find_by_agent so we can confirm error trapping."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def find_by_agent(self, agent_id: UUID) -> list[_FakeWorkspaceEntity]:
        raise self._exc


@pytest.mark.asyncio
async def test_execute_returns_json_array_for_populated_agent() -> None:
    """populated agent yields JSON array of {name, description, date_updated}."""
    agent_id = uuid4()
    when = datetime(2026, 4, 16, 12, 0, 0, tzinfo=UTC)
    entities = [
        _FakeWorkspaceEntity(name="alpha", description="first", date_updated=when),
        _FakeWorkspaceEntity(name="beta", description=None, date_updated=when),
    ]
    coll = _FakeCollection(entities)
    tool = WorkspaceListTool(workspace_collection=coll, agent_id=agent_id)

    result = await tool.execute()

    assert result.success is True
    assert result.error is None
    payload: list[dict[str, Any]] = json.loads(result.content)
    assert payload == [
        {"name": "alpha", "description": "first", "date_updated": when.isoformat()},
        {"name": "beta", "description": "", "date_updated": when.isoformat()},
    ]
    assert coll.calls == [agent_id]


@pytest.mark.asyncio
async def test_execute_returns_empty_array_for_empty_agent() -> None:
    """empty agent yields ``"[]"`` content with success True."""
    coll = _FakeCollection([])
    tool = WorkspaceListTool(workspace_collection=coll, agent_id=uuid4())

    result = await tool.execute()

    assert result.success is True
    assert result.content == "[]"
    assert result.error is None


@pytest.mark.asyncio
async def test_execute_traps_collection_errors_as_data() -> None:
    """collection failures surface as ToolResult(success=False, error=...)."""
    coll = _FailingCollection(RuntimeError("pool exploded"))
    tool = WorkspaceListTool(workspace_collection=coll, agent_id=uuid4())

    result = await tool.execute()

    assert result.success is False
    assert result.error is not None
    assert "list failed" in result.error
    assert "pool exploded" in result.error


@pytest.mark.asyncio
async def test_execute_dates_use_iso_format() -> None:
    """date_updated values are emitted as ISO-8601 strings with tzinfo."""
    when = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    coll = _FakeCollection([_FakeWorkspaceEntity(name="x", description=None, date_updated=when)])
    tool = WorkspaceListTool(workspace_collection=coll, agent_id=uuid4())

    result = await tool.execute()

    payload = json.loads(result.content)
    assert payload[0]["date_updated"] == when.isoformat()
    parsed = datetime.fromisoformat(payload[0]["date_updated"])
    assert parsed == when


def test_mcp_name_is_exact_string() -> None:
    """mcp_name must equal ``threetears.workspace.list`` exactly."""
    tool = WorkspaceListTool(workspace_collection=_FakeCollection([]), agent_id=uuid4())

    assert tool.mcp_name() == "threetears.workspace.list"


def test_mcp_version_is_semver_string() -> None:
    """mcp_version returns a non-empty version string."""
    tool = WorkspaceListTool(workspace_collection=_FakeCollection([]), agent_id=uuid4())

    assert tool.mcp_version() == "1.0"


def test_mcp_schema_returns_definition_with_empty_object_input() -> None:
    """mcp_schema returns MCPToolDefinition with the tool name and empty input schema."""
    tool = WorkspaceListTool(workspace_collection=_FakeCollection([]), agent_id=uuid4())

    definition = tool.mcp_schema()

    assert isinstance(definition, MCPToolDefinition)
    assert definition.name == "threetears.workspace.list"
    assert definition.version == "1.0"
    assert isinstance(definition.description, str)
    assert definition.description
    assert definition.input_schema["type"] == "object"
    assert definition.input_schema["properties"] == {}
    assert definition.input_schema["additionalProperties"] is False
