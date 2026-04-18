"""tests for ``threetears.workspace.current`` -- WorkspaceCurrentTool.

workspace-task-19 Phase 5 rewrote this tool to verify pin visibility
via the ``workspace.discover`` subject. tests here exercise:

- pinned + caller owns the workspace -> discovery returns the row -> success
- pinned + caller has no grant -> discovery returns empty -> WorkspaceAccessDenied path
- unset pin -> null-pin message
- discovery / context failures -> errors-as-data
- mcp metadata surfaces
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest

from threetears.agent.tools.base_tool import MCPToolDefinition
from threetears.agent.tools.call_scope import ToolCallScope, enter_call_scope
from threetears.agent.tools.context_envelope import CallContext
from threetears.agent.workspace.discovery_client import (
    DiscoveryClientError,
    WorkspaceDiscoverySummary,
)
from threetears.agent.workspace.pin import PinnedWorkspace

from threetears.agent.workspace.tools import workspace_current as workspace_current_module
from threetears.agent.workspace.tools.workspace_current import WorkspaceCurrentTool


class _FakeContext:
    """sentinel context object returned by the provider closure."""


@dataclass
class _FakeDiscoveryClient:
    """stand-in for :class:`WorkspaceDiscoveryClient` returning fixed items."""

    items: list[WorkspaceDiscoverySummary]
    raise_exc: Exception | None = None

    async def discover(
        self,
        *,
        correlation_id: UUID,
        agent_id: UUID,
        customer_id: UUID,
        user_id: UUID | None,
    ) -> list[WorkspaceDiscoverySummary]:
        if self.raise_exc is not None:
            raise self.raise_exc
        return list(self.items)


def _make_summary(workspace_id: UUID, owner_agent: UUID, customer: UUID) -> WorkspaceDiscoverySummary:
    """build a discovery summary for the given workspace id."""
    return WorkspaceDiscoverySummary(
        id=workspace_id,
        name=f"workspace.{workspace_id}",
        owner_agent_id=owner_agent,
        customer_id=customer,
    )


def _make_scope(customer_id: UUID | None = None, user_id: UUID | None = None) -> ToolCallScope:
    """build a ToolCallScope with identity dims."""
    ctx = CallContext(
        agent_id=uuid4(),
        user_id=user_id or uuid4(),
        customer_id=customer_id or uuid4(),
    )
    return ToolCallScope(context=ctx)


@pytest.mark.asyncio
async def test_execute_returns_pin_snapshot_when_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pinned + visible-in-discovery yields the snapshot JSON."""
    workspace_id = UUID("11111111-1111-1111-1111-111111111111")
    actor_id = UUID("22222222-2222-2222-2222-222222222222")
    agent_id = uuid4()
    customer_id = uuid4()
    when = datetime(2026, 4, 16, 12, 0, 0, tzinfo=UTC)
    snapshot = PinnedWorkspace(
        workspace_id=workspace_id,
        workspace_name="main",
        date_pinned=when,
        pinned_by_actor_id=actor_id,
    )

    async def _fake_get_pin(context: Any) -> PinnedWorkspace | None:
        return snapshot

    monkeypatch.setattr(workspace_current_module.pin, "get_pin", _fake_get_pin)

    client = _FakeDiscoveryClient(
        items=[_make_summary(workspace_id, agent_id, customer_id)],
    )
    tool = WorkspaceCurrentTool(
        context_provider=lambda: _FakeContext(),
        discovery_client=client,  # type: ignore[arg-type]
        agent_id=agent_id,
    )

    scope = _make_scope(customer_id=customer_id)
    async with enter_call_scope(scope):
        result = await tool.execute()

    assert result.success is True
    payload = json.loads(result.content)
    assert payload["workspace_id"] == str(workspace_id)
    assert payload["workspace_name"] == "main"


@pytest.mark.asyncio
async def test_execute_denies_when_pin_not_in_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pinned workspace absent from discovery -> access denied error."""
    workspace_id = uuid4()
    when = datetime(2026, 4, 16, 12, 0, 0, tzinfo=UTC)
    snapshot = PinnedWorkspace(
        workspace_id=workspace_id,
        workspace_name="shared",
        date_pinned=when,
        pinned_by_actor_id=uuid4(),
    )

    async def _fake_get_pin(context: Any) -> PinnedWorkspace | None:
        return snapshot

    monkeypatch.setattr(workspace_current_module.pin, "get_pin", _fake_get_pin)

    client = _FakeDiscoveryClient(items=[])  # empty -> not visible
    tool = WorkspaceCurrentTool(
        context_provider=lambda: _FakeContext(),
        discovery_client=client,  # type: ignore[arg-type]
        agent_id=uuid4(),
    )

    async with enter_call_scope(_make_scope()):
        result = await tool.execute()

    assert result.success is False
    assert result.error is not None
    assert "not visible" in result.error


@pytest.mark.asyncio
async def test_execute_returns_null_pin_message_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """unset pin yields {"pin": null, "message": ...}."""

    async def _fake_get_pin(context: Any) -> PinnedWorkspace | None:
        return None

    monkeypatch.setattr(workspace_current_module.pin, "get_pin", _fake_get_pin)

    tool = WorkspaceCurrentTool(
        context_provider=lambda: _FakeContext(),
        discovery_client=_FakeDiscoveryClient(items=[]),  # type: ignore[arg-type]
        agent_id=uuid4(),
    )

    async with enter_call_scope(_make_scope()):
        result = await tool.execute()

    assert result.success is True
    payload = json.loads(result.content)
    assert payload["pin"] is None
    assert "workspace.use" in payload["message"]


@pytest.mark.asyncio
async def test_execute_traps_discovery_errors_as_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """discovery transport failures surface as ToolResult(success=False)."""
    snapshot = PinnedWorkspace(
        workspace_id=uuid4(),
        workspace_name="any",
        date_pinned=datetime.now(UTC),
        pinned_by_actor_id=uuid4(),
    )

    async def _fake_get_pin(context: Any) -> PinnedWorkspace | None:
        return snapshot

    monkeypatch.setattr(workspace_current_module.pin, "get_pin", _fake_get_pin)

    client = _FakeDiscoveryClient(
        items=[],
        raise_exc=DiscoveryClientError("nats down"),
    )
    tool = WorkspaceCurrentTool(
        context_provider=lambda: _FakeContext(),
        discovery_client=client,  # type: ignore[arg-type]
        agent_id=uuid4(),
    )

    async with enter_call_scope(_make_scope()):
        result = await tool.execute()

    assert result.success is False
    assert "current failed" in (result.error or "")


def test_mcp_name_is_exact_string() -> None:
    """mcp_name must equal ``threetears.workspace.current`` exactly."""
    tool = WorkspaceCurrentTool(
        context_provider=lambda: _FakeContext(),
        discovery_client=_FakeDiscoveryClient(items=[]),  # type: ignore[arg-type]
        agent_id=uuid4(),
    )
    assert tool.mcp_name() == "threetears.workspace.current"


def test_mcp_version_is_semver_string() -> None:
    """mcp_version returns a non-empty version string."""
    tool = WorkspaceCurrentTool(
        context_provider=lambda: _FakeContext(),
        discovery_client=_FakeDiscoveryClient(items=[]),  # type: ignore[arg-type]
        agent_id=uuid4(),
    )
    assert tool.mcp_version() == "1.0"


def test_mcp_schema_returns_definition_with_empty_object_input() -> None:
    """mcp_schema returns MCPToolDefinition with empty object input schema."""
    tool = WorkspaceCurrentTool(
        context_provider=lambda: _FakeContext(),
        discovery_client=_FakeDiscoveryClient(items=[]),  # type: ignore[arg-type]
        agent_id=uuid4(),
    )
    definition = tool.mcp_schema()
    assert isinstance(definition, MCPToolDefinition)
    assert definition.name == "threetears.workspace.current"
    assert definition.version == "1.0"
    assert definition.input_schema["properties"] == {}
