"""tests for ``threetears.workspace.list`` -- WorkspaceListTool.

workspace-task-19 Phase 5 rewrote the list tool to issue a NATS
request to the broker's ``{ns}.workspace.discover`` subject instead
of scanning the caller's agent schema. these tests exercise the
rewritten tool against a fake discovery client.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
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
from threetears.agent.workspace.tools.workspace_list import WorkspaceListTool


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


def _make_scope(customer_id: UUID | None = None, user_id: UUID | None = None) -> ToolCallScope:
    """build a ToolCallScope with identity dims."""
    ctx = CallContext(
        agent_id=uuid4(),
        user_id=user_id or uuid4(),
        customer_id=customer_id or uuid4(),
    )
    return ToolCallScope(context=ctx)


@pytest.mark.asyncio
async def test_execute_returns_discovered_summaries() -> None:
    """populated discovery yields JSON array of name/owner/customer entries."""
    agent_id = uuid4()
    customer_id = uuid4()
    other_agent = uuid4()
    items = [
        WorkspaceDiscoverySummary(
            id=uuid4(),
            name="workspace.alpha",
            owner_agent_id=agent_id,
            customer_id=customer_id,
        ),
        WorkspaceDiscoverySummary(
            id=uuid4(),
            name="workspace.beta",
            owner_agent_id=other_agent,
            customer_id=customer_id,
        ),
    ]
    client = _FakeDiscoveryClient(items=items)
    tool = WorkspaceListTool(discovery_client=client, agent_id=agent_id)  # type: ignore[arg-type]

    async with enter_call_scope(_make_scope(customer_id=customer_id)):
        result = await tool.execute()

    assert result.success is True
    payload: list[dict[str, Any]] = json.loads(result.content)
    assert len(payload) == 2
    assert payload[0]["name"] == "workspace.alpha"
    assert payload[1]["owner_agent_id"] == str(other_agent)


@pytest.mark.asyncio
async def test_execute_returns_empty_array_for_empty_discovery() -> None:
    """empty discovery set yields ``"[]"`` content with success True."""
    client = _FakeDiscoveryClient(items=[])
    tool = WorkspaceListTool(discovery_client=client, agent_id=uuid4())  # type: ignore[arg-type]

    async with enter_call_scope(_make_scope()):
        result = await tool.execute()

    assert result.success is True
    assert result.content == "[]"


@pytest.mark.asyncio
async def test_execute_traps_discovery_errors_as_data() -> None:
    """discovery transport failures surface as ToolResult(success=False)."""
    client = _FakeDiscoveryClient(
        items=[],
        raise_exc=DiscoveryClientError("nats timeout"),
    )
    tool = WorkspaceListTool(discovery_client=client, agent_id=uuid4())  # type: ignore[arg-type]

    async with enter_call_scope(_make_scope()):
        result = await tool.execute()

    assert result.success is False
    assert "list failed" in (result.error or "")
    assert "nats timeout" in (result.error or "")


@pytest.mark.asyncio
async def test_execute_requires_customer_on_scope() -> None:
    """call without customer_id on scope yields a clean errors-as-data message."""
    client = _FakeDiscoveryClient(items=[])
    tool = WorkspaceListTool(discovery_client=client, agent_id=uuid4())  # type: ignore[arg-type]

    ctx = CallContext(agent_id=uuid4(), user_id=uuid4(), customer_id=None)
    async with enter_call_scope(ToolCallScope(context=ctx)):
        result = await tool.execute()

    assert result.success is False
    assert "customer_id" in (result.error or "")


def test_mcp_name_is_exact_string() -> None:
    """mcp_name must equal ``threetears.workspace.list`` exactly."""
    tool = WorkspaceListTool(
        discovery_client=_FakeDiscoveryClient(items=[]),  # type: ignore[arg-type]
        agent_id=uuid4(),
    )
    assert tool.mcp_name() == "threetears.workspace.list"


def test_mcp_version_is_semver_string() -> None:
    """mcp_version returns a non-empty version string."""
    tool = WorkspaceListTool(
        discovery_client=_FakeDiscoveryClient(items=[]),  # type: ignore[arg-type]
        agent_id=uuid4(),
    )
    assert tool.mcp_version() == "1.0"


def test_mcp_schema_returns_definition_with_empty_object_input() -> None:
    """mcp_schema returns MCPToolDefinition with empty object input schema."""
    tool = WorkspaceListTool(
        discovery_client=_FakeDiscoveryClient(items=[]),  # type: ignore[arg-type]
        agent_id=uuid4(),
    )
    definition = tool.mcp_schema()
    assert isinstance(definition, MCPToolDefinition)
    assert definition.name == "threetears.workspace.list"
    assert definition.input_schema["properties"] == {}
