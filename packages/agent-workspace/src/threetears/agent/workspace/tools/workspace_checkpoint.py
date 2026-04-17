"""``threetears.workspace.checkpoint`` -- tag workspace state with a label.

checkpoint writes one ``action='checkpoint'`` journal row per current
head file, all sharing the supplied ``label`` and the file's current
version/sha/content. it does **not** mutate ``workspace_files`` head
state or ``workspaces.current_version`` -- checkpoints are *tags*, not
new versions. later diff/rollback can address the labelled rows via
the checkpoint's label value.

this is a workspace-level operation; it deliberately does **not** call
``sandbox.enforce``. the sandbox dep is not even in the constructor so
the anti-pattern cannot be introduced without a signature change, which
the shard-18 AST test flags.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid7

from threetears.agent.tools.base_tool import (
    MCPToolDefinition,
    TearsTool,
    ToolResult,
)
from threetears.agent.tools.context import ToolContextManager
from threetears.observe import get_logger

from threetears.agent.workspace.collections import (
    WorkspaceCollection,
    WorkspaceFileCollection,
    WorkspaceFileVersionCollection,
)
from threetears.agent.workspace.factory import register_tool_builder
from threetears.agent.workspace.tools.helpers import (
    NoWorkspacePinned,
    WorkspaceNotFound,
    _resolve_workspace,
)

log = get_logger(__name__)


_INSERT_WORKSPACE_FILE_VERSION_SQL = """
INSERT INTO workspace_file_versions (
    id, workspace_id, relative_path, version, content,
    sha256, action, label, actor_id, correlation_id, date_created
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
"""


_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "minLength": 1,
            "maxLength": 255,
            "description": (
                "label for the checkpoint tag; 1-255 chars, unique per "
                "workspace by convention"
            ),
        },
        "workspace": {
            "type": "string",
            "description": "workspace name; defaults to pinned workspace",
        },
    },
    "required": ["label"],
    "additionalProperties": False,
}


class WorkspaceCheckpointTool(TearsTool):
    """snapshot the current head of every file in the workspace under a label.

    constructor deliberately does NOT accept a ``sandbox`` dependency:
    checkpoint is a workspace-level operation, not a per-file one. the
    absence of ``sandbox`` makes an accidental per-file ``enforce`` call
    impossible without an obvious signature change (shard-18 AST test).
    """

    def __init__(
        self,
        workspace_collection: WorkspaceCollection,
        workspace_file_collection: WorkspaceFileCollection,
        workspace_file_version_collection: WorkspaceFileVersionCollection,
        context_provider: Callable[[], ToolContextManager],
        agent_id: UUID,
        db_pool: Any,
    ) -> None:
        """
        binds tool to collections, context, owning agent, and pool.

        :param workspace_collection: collection providing workspace resolution
        :ptype workspace_collection: WorkspaceCollection
        :param workspace_file_collection: collection providing
            ``find_by_workspace`` for the current head file set
        :ptype workspace_file_collection: WorkspaceFileCollection
        :param workspace_file_version_collection: accepted for dep
            symmetry; checkpoint inserts journal rows directly via the
            pool connection so they share the enclosing transaction
        :ptype workspace_file_version_collection: WorkspaceFileVersionCollection
        :param context_provider: zero-arg callable returning conversation context
        :ptype context_provider: Callable[[], ToolContextManager]
        :param agent_id: identifier of agent owning workspace; recorded as
            ``actor_id`` on every checkpoint journal row
        :ptype agent_id: UUID
        :param db_pool: asyncpg pool supplying acquire+transaction for the
            multi-row checkpoint write
        :ptype db_pool: Any
        """
        self._workspaces = workspace_collection
        self._files = workspace_file_collection
        self._versions = workspace_file_version_collection
        self._context_provider = context_provider
        self._agent_id = agent_id
        self._db_pool = db_pool

    async def execute(self, **kwargs: Any) -> ToolResult:
        """
        emit one ``action='checkpoint'`` journal row per current head file.

        all failures arrive as :class:`ToolResult` with ``success=False``;
        this method never raises. head state and workspace version are
        not touched -- checkpoint is a metadata tag, not a new version.

        :param kwargs: must include ``label``; optional ``workspace``
        :ptype kwargs: Any
        :return: tool result summarizing rows written under the label
        :rtype: ToolResult
        """
        label = kwargs.get("label", "")
        workspace_arg = kwargs.get("workspace")

        # guard clause (entry-time input validation) per CLAUDE.md rule.
        if not isinstance(label, str) or not label:
            return ToolResult(
                success=False,
                content="",
                error="label is required and must be a non-empty string",
            )

        result: ToolResult
        try:
            workspace = await _resolve_workspace(
                workspace_arg,
                self._context_provider(),
                self._workspaces,
                self._agent_id,
            )
            head_files = await self._files.find_by_workspace(workspace.id)
            now = datetime.now(UTC)
            correlation_id = uuid7()
            async with self._db_pool.acquire() as conn:
                async with conn.transaction():
                    for file_entity in head_files:
                        await conn.execute(
                            _INSERT_WORKSPACE_FILE_VERSION_SQL,
                            uuid7(),
                            workspace.id,
                            file_entity.relative_path,
                            file_entity.version,
                            file_entity.content,
                            file_entity.sha256,
                            "checkpoint",
                            label,
                            self._agent_id,
                            correlation_id,
                            now,
                        )
            result = ToolResult(
                success=True,
                content=(
                    f"checkpointed {len(head_files)} files with label {label!r}"
                ),
                metadata={"n_files": len(head_files), "label": label},
            )
        except (WorkspaceNotFound, NoWorkspacePinned) as exc:
            result = ToolResult(success=False, content="", error=str(exc))
        except Exception as exc:
            log.exception("workspace_checkpoint failed: %s", exc)
            result = ToolResult(
                success=False,
                content="",
                error=f"checkpoint failed: {exc}",
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
            description=(
                "tag the current workspace state with a checkpoint label; "
                "writes one journal row per head file without changing head "
                "state"
            ),
            input_schema=_INPUT_SCHEMA,
        )

    def mcp_name(self) -> str:
        """
        returns the namespaced tool name advertised to MCP clients.

        :return: tool name
        :rtype: str
        """
        return "threetears.workspace.checkpoint"

    def mcp_version(self) -> str:
        """
        returns the semver-compatible tool version.

        :return: version string
        :rtype: str
        """
        return "1.0"


def _build(**kwargs: Any) -> WorkspaceCheckpointTool:
    """
    constructs a :class:`WorkspaceCheckpointTool` from the factory dep bundle.

    consumes workspace, file, and journal collections, ``context_provider``,
    ``agent_id``, and ``db_pool``. deliberately does NOT consume ``sandbox``;
    see class-level docstring.

    :param kwargs: full factory dependency bundle
    :ptype kwargs: Any
    :return: constructed tool
    :rtype: WorkspaceCheckpointTool
    """
    return WorkspaceCheckpointTool(
        workspace_collection=kwargs["workspace_collection"],
        workspace_file_collection=kwargs["workspace_file_collection"],
        workspace_file_version_collection=kwargs["workspace_file_version_collection"],
        context_provider=kwargs["context_provider"],
        agent_id=kwargs["agent_id"],
        db_pool=kwargs["db_pool"],
    )


register_tool_builder(_build)
