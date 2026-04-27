"""``threetears.workspace.delete`` -- soft-delete a workspace.

soft-delete sets ``date_deleted`` on the workspaces row so downstream
queries (``list``) hide it while history queries can still traverse the
journal. when the deleted workspace is the one currently pinned to this
conversation, the pin is also cleared; pins in other conversations are
left alone -- those readers will surface a "no longer exists" error and
self-correct on the next call.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid7

from threetears.agent.acl import AclCache
from threetears.agent.audit import AuditEvent, publish_audit
from threetears.agent.tools.base_tool import (
    MCPToolDefinition,
    TearsTool,
    ToolResult,
)
from threetears.agent.tools.context import ToolContextManager
from threetears.observe import get_logger

from threetears.agent.workspace import pin
from threetears.agent.workspace.authorize import (
    WorkspaceAccessDenied,
)
from threetears.agent.workspace.collections import (
    WorkspaceCollection,
    WorkspaceFileCollection,
    WorkspaceFileVersionCollection,
)
from threetears.agent.workspace.factory import register_tool_builder
from threetears.agent.workspace.sandbox import WorkspaceSandbox
from threetears.agent.workspace.tools.helpers import (
    authorize_workspace,
    workspace_audit_identity,
)

__all__ = [
    "WorkspaceDeleteTool",
]

log = get_logger(__name__)


_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "human-readable workspace name to soft-delete",
        },
    },
    "required": ["name"],
    "additionalProperties": False,
}


_SOFT_DELETE_WORKSPACE_SQL = (
    "UPDATE workspaces SET date_deleted = $1, date_updated = $1 WHERE id = $2 AND agent_id = $3"
)


class WorkspaceDeleteTool(TearsTool):
    """soft-delete a workspace by name; clear conversation pin if matched.

    soft-delete preserves the journal so history queries continue to
    work. deleting an already-soft-deleted workspace returns the same
    "not found" error to keep the surface single-shaped for callers.
    """

    def __init__(
        self,
        workspace_collection: WorkspaceCollection,
        workspace_file_collection: WorkspaceFileCollection,
        workspace_file_version_collection: WorkspaceFileVersionCollection,
        sandbox: WorkspaceSandbox,
        context_provider: Callable[[], ToolContextManager],
        agent_id: UUID,
        db_pool: Any,
        acl_cache: AclCache,
        nats_client: Any = None,
        namespace: str | None = None,
    ) -> None:
        """
        binds tool to workspace collection, conversation context, and pool.

        the file/version collections and sandbox are accepted for
        symmetry with the other lifecycle tools and to keep the factory
        bundle uniform; this tool itself does not touch them because
        soft-delete only writes to the workspaces row.

        :param workspace_collection: collection providing find_by_agent_and_name
        :ptype workspace_collection: WorkspaceCollection
        :param workspace_file_collection: accepted for factory symmetry
        :ptype workspace_file_collection: WorkspaceFileCollection
        :param workspace_file_version_collection: accepted for factory symmetry
        :ptype workspace_file_version_collection: WorkspaceFileVersionCollection
        :param sandbox: accepted for factory symmetry
        :ptype sandbox: WorkspaceSandbox
        :param context_provider: zero-arg callable returning conversation context
        :ptype context_provider: Callable[[], ToolContextManager]
        :param agent_id: identifier of agent owning workspace
        :ptype agent_id: UUID
        :param db_pool: asyncpg pool supplying acquire+transaction
        :ptype db_pool: Any
        """
        self._workspaces = workspace_collection
        self._files = workspace_file_collection
        self._versions = workspace_file_version_collection
        self._sandbox = sandbox
        self._context_provider = context_provider
        self._agent_id = agent_id
        self._db_pool = db_pool
        self._nats_client = nats_client
        self._namespace = namespace
        self._acl_cache = acl_cache

    async def execute(self, **kwargs: Any) -> ToolResult:
        """
        soft-delete workspace ``name`` and clear pin when matching.

        returns the same "not found" error for missing or already-
        deleted rows so the LLM-facing surface is single-shaped. all
        failures arrive as :class:`ToolResult` with ``success=False``;
        this method never raises.

        :param kwargs: must include ``name``: workspace to delete
        :ptype kwargs: Any
        :return: tool result reporting delete or error
        :rtype: ToolResult
        """
        name = kwargs.get("name", "")

        correlation_id = uuid7()
        result: ToolResult
        try:
            workspace = await self._workspaces.find_by_agent_and_name(self._agent_id, name)
            if workspace is None or workspace.date_deleted is not None:
                result = ToolResult(
                    success=False,
                    content="",
                    error=f"workspace {name!r} not found",
                )
            else:
                await authorize_workspace(
                    workspace,
                    "write",
                    db_pool=self._db_pool,
                    acl_cache=self._acl_cache,
                )
                now = datetime.now(UTC)
                # WS-ACL-06: bind the tx to the workspace's namespace
                # so grantee-agent deletes of shared workspaces would
                # land in the owner's schema. today owner-only callers
                # are the production path (delete is owner-capability),
                # but routing through the namespace keeps the code
                # uniform and lets cross-agent tooling evolve without
                # a branch.
                async with self._db_pool.acquire() as conn:
                    async with conn.transaction(namespace=workspace.namespace_name):
                        await conn.execute(
                            _SOFT_DELETE_WORKSPACE_SQL,
                            now,
                            workspace.id,
                            workspace.agent_id,
                        )

                ctx = self._context_provider()
                snapshot = await pin.get_pin(ctx)
                if snapshot is not None and snapshot.workspace_id == workspace.id:
                    await pin.clear_pin(ctx)

                # defense-in-depth: additive per-tool event on top of
                # the baseline ``tool.call`` emitted by ToolServer.
                try:
                    if self._namespace is not None:
                        identity = workspace_audit_identity(workspace)
                        event = AuditEvent(
                            id=uuid7(),
                            timestamp=datetime.now(UTC),
                            event_type="workspace.delete",
                            actor_user_id=identity.actor_user_id,
                            calling_agent_id=identity.calling_agent_id,
                            owner_agent_id=identity.owner_agent_id,
                            customer_id=identity.customer_id,
                            resource_namespace_id=identity.namespace_id,
                            resource_namespace_type="workspace",
                            action="delete",
                            outcome="success",
                            correlation_id=correlation_id,
                            details={
                                "workspace_resource_id": str(workspace.id),
                                "name": name,
                            },
                        )
                        await publish_audit(
                            event,
                            nats_client=self._nats_client,
                            namespace=self._namespace,
                        )
                # NOSILENT: audit failure never taints delete
                except Exception as audit_exc:
                    log.exception(
                        "workspace_delete audit publish swallow caught: %s",
                        audit_exc,
                    )

                result = ToolResult(
                    success=True,
                    content=f"deleted workspace {name!r}",
                )
        except WorkspaceAccessDenied as exc:
            result = ToolResult(success=False, content="", error=str(exc))
        except Exception as exc:
            log.exception("workspace_delete failed: %s", exc)
            result = ToolResult(
                success=False,
                content="",
                error=f"delete failed: {exc}",
            )
        return result

    def mcp_schema(self) -> MCPToolDefinition:
        """
        returns the MCP definition for this tool.

        :return: MCP-compatible tool definition
        :rtype: MCPToolDefinition
        """
        return MCPToolDefinition(
            name=self.mcp_name(),
            version=self.mcp_version(),
            description="soft-delete a workspace by name",
            input_schema=_INPUT_SCHEMA,
        )

    def mcp_name(self) -> str:
        """
        returns the namespaced tool name advertised to MCP clients.

        :return: tool name
        :rtype: str
        """
        return "threetears.workspace.delete"

    def mcp_version(self) -> str:
        """
        returns the semver-compatible tool version.

        :return: version string
        :rtype: str
        """
        return "1.0"


def _build(**kwargs: Any) -> WorkspaceDeleteTool:
    """
    constructs a :class:`WorkspaceDeleteTool` from the factory dep bundle.

    consumes ``workspace_collection``, ``workspace_file_collection``,
    ``workspace_file_version_collection``, ``sandbox``,
    ``context_provider``, ``agent_id``, and ``db_pool``; ignores the
    rest. registered with :mod:`threetears.agent.workspace.factory` on
    import so :func:`build_workspace_tools` emits this tool.

    :param kwargs: full factory dependency bundle
    :ptype kwargs: Any
    :return: constructed tool
    :rtype: WorkspaceDeleteTool
    """
    return WorkspaceDeleteTool(
        workspace_collection=kwargs["workspace_collection"],
        workspace_file_collection=kwargs["workspace_file_collection"],
        workspace_file_version_collection=kwargs["workspace_file_version_collection"],
        sandbox=kwargs["sandbox"],
        context_provider=kwargs["context_provider"],
        agent_id=kwargs["agent_id"],
        db_pool=kwargs["db_pool"],
        nats_client=kwargs.get("nats_client"),
        namespace=kwargs.get("namespace"),
        acl_cache=kwargs["acl_cache"],
    )


register_tool_builder(_build)
