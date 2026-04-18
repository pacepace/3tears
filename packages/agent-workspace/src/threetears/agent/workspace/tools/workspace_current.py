"""``threetears.workspace.current`` -- report the conversation's pinned workspace.

workspace-task-19 Phase 5 extends the pin-snapshot path with discovery
so a conversation pinned to a workspace owned by a different agent
(same customer, grant in place) resolves cleanly. when the pin's
workspace_id is NOT in the caller's agent-owned set, the tool issues a
``workspace.discover`` request to verify the caller can see it under a
grant; if the grant is absent, the tool raises
:class:`~threetears.agent.workspace.authorize.WorkspaceAccessDenied`
so the conversation surfaces a clear recovery message.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from uuid import UUID, uuid7

from threetears.agent.tools.base_tool import (
    MCPToolDefinition,
    TearsTool,
    ToolResult,
)
from threetears.agent.tools.call_scope import current_scope
from threetears.agent.tools.context import ToolContextManager
from threetears.observe import get_logger

from threetears.agent.workspace import pin
from threetears.agent.workspace.authorize import WorkspaceAccessDenied
from threetears.agent.workspace.discovery_client import (
    DiscoveryClientError,
    WorkspaceDiscoveryClient,
)
from threetears.agent.workspace.factory import register_tool_builder

__all__ = [
    "WorkspaceCurrentTool",
]

log = get_logger(__name__)


_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}


class WorkspaceCurrentTool(TearsTool):
    """report which workspace is pinned to the current conversation.

    reads :func:`pin.get_pin` and serializes the snapshot. when the
    pinned workspace is not owned by the calling agent, verifies
    visibility via the discovery subject; a missing grant surfaces as
    :class:`WorkspaceAccessDenied` so the pin stays in the conversation
    state but the LLM sees a clear denial.
    """

    def __init__(
        self,
        context_provider: Callable[[], ToolContextManager],
        discovery_client: WorkspaceDiscoveryClient,
        agent_id: UUID,
    ) -> None:
        """
        binds tool to per-conversation context, discovery client, and agent.

        :param context_provider: zero-arg callable returning the current
            conversation's ToolContextManager
        :ptype context_provider: Callable[[], ToolContextManager]
        :param discovery_client: NATS client for ``workspace.discover``
        :ptype discovery_client: WorkspaceDiscoveryClient
        :param agent_id: identifier of calling agent
        :ptype agent_id: UUID
        """
        self._context_provider = context_provider
        self._discovery = discovery_client
        self._agent_id = agent_id

    async def execute(self, **kwargs: Any) -> ToolResult:
        """
        returns the conversation's pin snapshot or a null-pin response.

        :param kwargs: ignored, schema declares no inputs
        :ptype kwargs: Any
        :return: tool result with JSON content or error message
        :rtype: ToolResult
        """
        result: ToolResult
        try:
            snapshot = await pin.get_pin(self._context_provider())
            if snapshot is None:
                payload: dict[str, Any] = {
                    "pin": None,
                    "message": "no workspace pinned; call workspace.use(name) to set",
                }
                result = ToolResult(success=True, content=json.dumps(payload))
            else:
                await self._verify_visibility(snapshot.workspace_id)
                payload = {
                    "workspace_id": str(snapshot.workspace_id),
                    "workspace_name": snapshot.workspace_name,
                    "date_pinned": snapshot.date_pinned.isoformat(),
                }
                result = ToolResult(success=True, content=json.dumps(payload))
        except WorkspaceAccessDenied as exc:
            result = ToolResult(
                success=False,
                content="",
                error=f"pinned workspace not visible: {exc}",
            )
        except DiscoveryClientError as exc:
            result = ToolResult(
                success=False,
                content="",
                error=f"current failed: {exc}",
            )
        except Exception as exc:
            log.exception("workspace_current failed: %s", exc)
            result = ToolResult(
                success=False,
                content="",
                error=f"current failed: {exc}",
            )
        return result

    async def _verify_visibility(self, workspace_id: UUID) -> None:
        """confirm the caller can see ``workspace_id`` under their scope.

        owner-path short-circuits: when discovery returns the workspace
        with ``owner_agent_id == self._agent_id``, access is fine. when
        a non-owner workspace appears in the discovery set, a grant is
        in place. when the workspace is NOT in the discovery set, no
        grant exists and the caller must not leak its existence --
        raise :class:`WorkspaceAccessDenied`.

        :param workspace_id: pinned workspace identifier
        :ptype workspace_id: UUID
        :return: nothing
        :rtype: None
        :raises WorkspaceAccessDenied: on missing customer or missing grant
        :raises DiscoveryClientError: on broker failure
        """
        scope = current_scope()
        customer_id: UUID | None = None if scope is None else scope.context.customer_id
        user_id: UUID | None = None if scope is None else scope.context.user_id
        correlation_id: UUID = (
            scope.context.correlation_id
            if scope is not None and scope.context.correlation_id is not None
            else uuid7()
        )
        if customer_id is None:
            raise WorkspaceAccessDenied(
                "workspace.current requires a customer_id on the call scope",
            )
        items = await self._discovery.discover(
            correlation_id=correlation_id,
            agent_id=self._agent_id,
            customer_id=customer_id,
            user_id=user_id,
        )
        visible = any(item.id == workspace_id for item in items)
        if not visible:
            raise WorkspaceAccessDenied(
                f"pinned workspace {workspace_id} not in discovery set "
                "for calling agent + user + customer",
            )

    def mcp_schema(self) -> MCPToolDefinition:
        """
        returns the MCP definition for this tool.

        pure: no side effects, safe for repeated discovery calls.

        :return: MCP-compatible tool definition
        :rtype: MCPToolDefinition
        """
        return MCPToolDefinition(
            name=self.mcp_name(),
            version=self.mcp_version(),
            description="return the workspace pinned to the current conversation",
            input_schema=_INPUT_SCHEMA,
        )

    def mcp_name(self) -> str:
        """
        returns the namespaced tool name advertised to MCP clients.

        :return: tool name
        :rtype: str
        """
        return "threetears.workspace.current"

    def mcp_version(self) -> str:
        """
        returns the semver-compatible tool version.

        :return: version string
        :rtype: str
        """
        return "1.0"


def _build(**kwargs: Any) -> WorkspaceCurrentTool:
    """
    constructs a :class:`WorkspaceCurrentTool` from the factory dep bundle.

    consumes ``context_provider``, ``nats_client``, ``namespace``, and
    ``agent_id`` to build a :class:`WorkspaceDiscoveryClient`. ignores
    the rest. registered with :mod:`threetears.agent.workspace.factory`
    on import so :func:`build_workspace_tools` emits this tool.

    :param kwargs: full factory dependency bundle
    :ptype kwargs: Any
    :return: constructed tool
    :rtype: WorkspaceCurrentTool
    """
    client = WorkspaceDiscoveryClient(
        nats_client=kwargs.get("nats_client"),
        namespace=kwargs.get("namespace") or "",
    )
    return WorkspaceCurrentTool(
        context_provider=kwargs["context_provider"],
        discovery_client=client,
        agent_id=kwargs["agent_id"],
    )


register_tool_builder(_build)
