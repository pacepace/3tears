"""tests for ``threetears.workspace.current`` -- WorkspaceCurrentTool."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import pytest

from threetears.agent.tools.base_tool import MCPToolDefinition
from threetears.agent.workspace.pin import PinnedWorkspace

from threetears.agent.workspace.tools import workspace_current as workspace_current_module
from threetears.agent.workspace.tools.workspace_current import WorkspaceCurrentTool


class _FakeContext:
    """sentinel context object returned by the provider closure."""


@pytest.mark.asyncio
async def test_execute_returns_pin_snapshot_when_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pinned conversation yields JSON with workspace_id, workspace_name, date_pinned."""
    workspace_id = UUID("11111111-1111-1111-1111-111111111111")
    actor_id = UUID("22222222-2222-2222-2222-222222222222")
    when = datetime(2026, 4, 16, 12, 0, 0, tzinfo=UTC)
    snapshot = PinnedWorkspace(
        workspace_id=workspace_id,
        workspace_name="main",
        date_pinned=when,
        pinned_by_actor_id=actor_id,
    )

    received_contexts: list[Any] = []

    async def _fake_get_pin(context: Any) -> PinnedWorkspace | None:
        received_contexts.append(context)
        return snapshot

    monkeypatch.setattr(workspace_current_module.pin, "get_pin", _fake_get_pin)

    fake_ctx = _FakeContext()
    tool = WorkspaceCurrentTool(context_provider=lambda: fake_ctx)

    result = await tool.execute()

    assert result.success is True
    assert result.error is None
    payload = json.loads(result.content)
    assert payload == {
        "workspace_id": str(workspace_id),
        "workspace_name": "main",
        "date_pinned": when.isoformat(),
    }
    assert received_contexts == [fake_ctx]


@pytest.mark.asyncio
async def test_execute_returns_null_pin_message_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """unset pin yields {"pin": null, "message": ...} with helpful hint."""

    async def _fake_get_pin(context: Any) -> PinnedWorkspace | None:
        return None

    monkeypatch.setattr(workspace_current_module.pin, "get_pin", _fake_get_pin)

    tool = WorkspaceCurrentTool(context_provider=lambda: _FakeContext())

    result = await tool.execute()

    assert result.success is True
    assert result.error is None
    payload = json.loads(result.content)
    assert payload["pin"] is None
    assert "workspace.use" in payload["message"]


@pytest.mark.asyncio
async def test_execute_traps_get_pin_errors_as_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """get_pin failures surface as ToolResult(success=False, error=...)."""

    async def _boom(context: Any) -> PinnedWorkspace | None:
        raise RuntimeError("context unavailable")

    monkeypatch.setattr(workspace_current_module.pin, "get_pin", _boom)

    tool = WorkspaceCurrentTool(context_provider=lambda: _FakeContext())

    result = await tool.execute()

    assert result.success is False
    assert result.error is not None
    assert "current failed" in result.error
    assert "context unavailable" in result.error


def test_mcp_name_is_exact_string() -> None:
    """mcp_name must equal ``threetears.workspace.current`` exactly."""
    tool = WorkspaceCurrentTool(context_provider=lambda: _FakeContext())

    assert tool.mcp_name() == "threetears.workspace.current"


def test_mcp_version_is_semver_string() -> None:
    """mcp_version returns a non-empty version string."""
    tool = WorkspaceCurrentTool(context_provider=lambda: _FakeContext())

    assert tool.mcp_version() == "1.0"


def test_mcp_schema_returns_definition_with_empty_object_input() -> None:
    """mcp_schema returns MCPToolDefinition with empty object input schema."""
    tool = WorkspaceCurrentTool(context_provider=lambda: _FakeContext())

    definition = tool.mcp_schema()

    assert isinstance(definition, MCPToolDefinition)
    assert definition.name == "threetears.workspace.current"
    assert definition.version == "1.0"
    assert definition.input_schema["type"] == "object"
    assert definition.input_schema["properties"] == {}
    assert definition.input_schema["additionalProperties"] is False
