"""tests for ``threetears.workspace.list`` -- WorkspaceListTool.

workspace-task-19 Phase 5 rewrote the list tool to issue a NATS
request instead of scanning the caller's agent schema. namespace-
task-01 Phase 1 generalized that subject from
``{ns}.workspace.discover`` to ``{ns}.namespace.discover`` with a
``namespace_type`` filter; the tool now passes
``namespace_type="workspace"`` explicitly on every call. these tests
exercise the tool against a fake namespace-discovery client that
records the filter it was asked for.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

import pytest

from threetears.agent.tools.base_tool import MCPToolDefinition
from threetears.agent.tools.call_scope import ToolCallScope, enter_call_scope
from threetears.agent.tools.context_envelope import CallContext

from threetears.agent.tools.namespace_discovery_client import (
    DiscoveryClientError,
    NamespaceDiscoverySummary,
)
from threetears.agent.workspace.tools.workspace_list import WorkspaceListTool


@dataclass
# parity-exempt: workspace discovery-client subset for the workspace_list tool unit; same shape as workspace_current's discovery-client fake above
class _FakeDiscoveryClient:
    """stand-in for :class:`NamespaceDiscoveryClient` returning fixed items.

    records the ``namespace_type`` filter the tool passed so the tests
    can assert the tool is asking for ``"workspace"`` specifically.
    """

    items: list[NamespaceDiscoverySummary]
    raise_exc: Exception | None = None
    last_filter: str | None = field(default=None, init=False)
    last_identity_token: str | None = field(default=None, init=False)
    last_user_identity_token: str | None = field(default=None, init=False)

    async def discover(
        self,
        *,
        correlation_id: UUID,
        identity_token: str | None = None,
        user_identity_token: str | None = None,
        namespace_type: str | None = None,
    ) -> list[NamespaceDiscoverySummary]:
        if self.raise_exc is not None:
            raise self.raise_exc
        self.last_filter = namespace_type
        self.last_identity_token = identity_token
        self.last_user_identity_token = user_identity_token
        return list(self.items)


def _make_scope(customer_id: UUID | None = None, user_id: UUID | None = None) -> ToolCallScope:
    """build a ToolCallScope carrying the tokens the tool forwards.

    the identity dims stay on the context because the rest of the
    dispatch reads them; what the discovery call now carries is the
    TOKENS, which is what the broker verifies.
    """
    ctx = CallContext(
        agent_id=uuid4(),
        user_id=user_id or uuid4(),
        customer_id=customer_id or uuid4(),
        identity_token="agent.token",
        user_identity_token="user.assertion",
    )
    return ToolCallScope(context=ctx)


@pytest.mark.asyncio
async def test_execute_returns_discovered_summaries() -> None:
    """populated discovery yields JSON array of name/owner/customer entries."""
    agent_id = uuid4()
    customer_id = uuid4()
    other_agent = uuid4()
    items = [
        NamespaceDiscoverySummary(
            id=uuid4(),
            name="workspace.alpha",
            namespace_type="workspace",
            owner_agent_id=agent_id,
            customer_id=customer_id,
        ),
        NamespaceDiscoverySummary(
            id=uuid4(),
            name="workspace.beta",
            namespace_type="workspace",
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
    # tool must ask the broker for workspace-type rows specifically
    assert client.last_filter == "workspace"


@pytest.mark.asyncio
async def test_execute_returns_empty_array_for_empty_discovery() -> None:
    """empty discovery set yields ``"[]"`` content with success True."""
    client = _FakeDiscoveryClient(items=[])
    tool = WorkspaceListTool(discovery_client=client, agent_id=uuid4())  # type: ignore[arg-type]

    async with enter_call_scope(_make_scope()):
        result = await tool.execute()

    assert result.success is True
    assert result.content == "[]"
    assert client.last_filter == "workspace"


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
async def test_execute_requires_an_identity_token_on_scope() -> None:
    """a scope carrying neither token yields a clean errors-as-data message."""
    client = _FakeDiscoveryClient(items=[])
    tool = WorkspaceListTool(discovery_client=client, agent_id=uuid4())  # type: ignore[arg-type]

    ctx = CallContext(agent_id=uuid4(), user_id=uuid4(), customer_id=None)
    async with enter_call_scope(ToolCallScope(context=ctx)):
        result = await tool.execute()

    assert result.success is False
    assert "identity token" in (result.error or "")


@pytest.mark.asyncio
async def test_execute_forwards_both_tokens_to_discovery() -> None:
    """the tool states no identity of its own; it forwards what it was given."""
    client = _FakeDiscoveryClient(items=[])
    tool = WorkspaceListTool(discovery_client=client, agent_id=uuid4())  # type: ignore[arg-type]

    async with enter_call_scope(_make_scope()):
        result = await tool.execute()

    assert result.success is True
    assert client.last_identity_token == "agent.token"
    assert client.last_user_identity_token == "user.assertion"


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
