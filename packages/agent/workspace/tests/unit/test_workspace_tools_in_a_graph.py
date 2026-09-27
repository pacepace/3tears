"""A workspace tool run inside a LangGraph graph gets the same call scope it gets over NATS.

Wrapped by :func:`~threetears.agent.tools.langchain_adapter.to_langchain_tool`, a tool runs inside
the :class:`~threetears.agent.tools.call_scope.ToolCallScope` the ToolServer installs around a
dispatch, built from the graph config's ``call_context``. It used to run in none, so a workspace
tool -- which resolves the conversation's context and the caller's tokens from the scope -- could
not run in a graph at all.

These live outside ``tests/unit/tools/`` on purpose: that directory's conftest installs a scope
around every test, and a graph running a tool has no scope but the one the adapter installs.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest

from threetears.agent.tools.call_scope import tool_context_provider
from threetears.agent.tools.context_envelope import CallContext
from threetears.agent.tools.langchain_adapter import to_langchain_tool
from threetears.agent.tools.namespace_discovery_client import NamespaceDiscoverySummary
from threetears.agent.workspace.pin import PinnedWorkspace
from threetears.agent.workspace.tools import workspace_current as workspace_current_module
from threetears.agent.workspace.tools.workspace_current import WorkspaceCurrentTool


# parity-exempt: the one NamespaceDiscoveryClient method workspace_current calls, recording the tokens it forwards
class _Discovery:
    """answers ``discover`` with fixed rows and records the identity token it was handed."""

    def __init__(self, items: list[NamespaceDiscoverySummary]) -> None:
        """hold the rows to answer with.

        :param items: the discovery rows
        :ptype items: list[NamespaceDiscoverySummary]
        :return: nothing
        :rtype: None
        """
        self.items = items
        self.identity_token: str | None = None

    async def discover(
        self,
        *,
        correlation_id: UUID,
        identity_token: str | None = None,
        user_identity_token: str | None = None,
        namespace_type: str | None = None,
    ) -> list[NamespaceDiscoverySummary]:
        """the fixed rows, recording the token.

        :param correlation_id: the call's correlation id
        :ptype correlation_id: UUID
        :param identity_token: the calling agent's token
        :ptype identity_token: str | None
        :param user_identity_token: the invoking user's assertion
        :ptype user_identity_token: str | None
        :param namespace_type: the namespace type filter
        :ptype namespace_type: str | None
        :return: the rows
        :rtype: list[NamespaceDiscoverySummary]
        """
        self.identity_token = identity_token
        return list(self.items)


def _tool(discovery: _Discovery, agent_id: UUID, **adapter: Any) -> Any:
    """``workspace_current`` wrapped for LangGraph, reading its context through the call scope.

    :param discovery: the discovery double
    :ptype discovery: _Discovery
    :param agent_id: the calling agent
    :ptype agent_id: UUID
    :param adapter: keyword arguments for :func:`to_langchain_tool`
    :ptype adapter: Any
    :return: the wrapped tool
    :rtype: Any
    """
    return to_langchain_tool(
        WorkspaceCurrentTool(
            context_provider=tool_context_provider,
            discovery_client=discovery,  # type: ignore[arg-type]
            agent_id=agent_id,
        ),
        **adapter,
    )


@pytest.mark.asyncio
async def test_the_tool_resolves_its_conversation_and_tokens_from_the_call_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """the conversation reaches the tool through the host's factory, the tokens through the scope.

    :param monkeypatch: pytest monkeypatch
    :ptype monkeypatch: pytest.MonkeyPatch
    :return: none
    :rtype: None
    """
    workspace_id, agent_id, customer_id = uuid4(), uuid4(), uuid4()
    conversation_id, user_id = uuid4(), uuid4()
    conversation_context = object()
    handed: list[Any] = []

    async def _get_pin(context: Any) -> PinnedWorkspace:
        handed.append(context)
        return PinnedWorkspace(
            workspace_id=workspace_id,
            workspace_name="main",
            date_pinned=datetime(2026, 4, 16, 12, 0, 0, tzinfo=UTC),
            pinned_by_actor_id=uuid4(),
        )

    async def _factory(conversation: UUID, user: UUID) -> Any:
        assert (conversation, user) == (conversation_id, user_id)
        return conversation_context

    monkeypatch.setattr(workspace_current_module.pin, "get_pin", _get_pin)
    discovery = _Discovery(
        [
            NamespaceDiscoverySummary(
                id=workspace_id,
                name=f"workspace.{workspace_id}",
                namespace_type="workspace",
                owner_agent_id=agent_id,
                customer_id=customer_id,
            )
        ]
    )
    tool = _tool(discovery, agent_id, context_factory=_factory)
    call_context = CallContext(
        conversation_id=conversation_id,
        user_id=user_id,
        customer_id=customer_id,
        agent_id=agent_id,
        identity_token="agent.token",
        user_identity_token="user.assertion",
    )

    message = await tool.ainvoke(
        {"type": "tool_call", "id": "w1", "name": tool.name, "args": {}},
        {"configurable": {"call_context": call_context}},
    )

    assert message.status == "success"
    assert json.loads(message.content)["workspace_id"] == str(workspace_id)
    assert handed == [conversation_context]
    assert discovery.identity_token == "agent.token"


@pytest.mark.asyncio
async def test_without_a_call_context_the_error_names_the_config_key() -> None:
    """no call context: the tool cannot know its conversation, and its failure says what to supply.

    ``workspace_current`` answers every failure as data, so the named cause arrives on a failed
    tool message rather than as an exception.

    :return: none
    :rtype: None
    """
    tool = _tool(_Discovery([]), uuid4())
    message = await tool.ainvoke({"type": "tool_call", "id": "w2", "name": tool.name, "args": {}}, {"configurable": {}})
    assert message.status == "error"
    assert "config['configurable']['call_context']" in message.content
