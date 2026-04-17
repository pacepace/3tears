"""``threetears.workspace.create`` -- create a new workspace.

three create modes share a single transactional sequence: empty (no
files), from_template (copy bytes from an image-shipped template
directory), or from_workspace (fork from an existing workspace by name).
in every mode the workspace row, file head rows, and journal rows are
inserted under one ``conn.transaction()`` so partial failure cannot
leave orphan files or journal-without-head inconsistencies. on success
the new workspace is auto-pinned to the current conversation; on
``UniqueViolationError`` (duplicate name within the agent) the tool
returns a clean error rather than raising.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid7

import asyncpg

from threetears.agent.tools.base_tool import (
    MCPToolDefinition,
    TearsTool,
    ToolResult,
)
from threetears.agent.tools.context import ToolContextManager
from threetears.observe import get_logger

from threetears.agent.workspace import audit, pin
from threetears.agent.workspace.collections import (
    WorkspaceCollection,
    WorkspaceFileCollection,
    WorkspaceFileVersionCollection,
)
from threetears.agent.workspace.config import ValidatorEntry
from threetears.agent.workspace.factory import register_tool_builder
from threetears.agent.workspace.sandbox import WorkspaceSandbox
from threetears.agent.workspace.tools.helpers import _resolve_validators
from threetears.agent.workspace.validators import (
    WorkspaceValidationError,
    dispatch_validators,
)

log = get_logger(__name__)


_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "unique workspace name within this agent",
        },
        "description": {
            "type": "string",
            "description": "optional human-readable description",
        },
        "from_template": {
            "type": "string",
            "description": "name of an image-shipped template directory under templates_dir",
        },
        "from_workspace": {
            "type": "string",
            "description": "name of an existing workspace to fork from",
        },
    },
    "required": ["name"],
    "additionalProperties": False,
}


_INSERT_WORKSPACE_SQL = """
INSERT INTO workspaces (
    id, agent_id, name, description, template_name,
    created_by, current_version, date_created, date_updated, date_deleted
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NULL)
"""

_INSERT_WORKSPACE_FILE_SQL = """
INSERT INTO workspace_files (
    id, workspace_id, relative_path, content, sha256, version, date_updated
) VALUES ($1, $2, $3, $4, $5, $6, $7)
"""

_INSERT_WORKSPACE_FILE_VERSION_SQL = """
INSERT INTO workspace_file_versions (
    id, workspace_id, relative_path, version, content,
    sha256, action, label, actor_id, correlation_id, date_created
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
"""


class WorkspaceCreateTool(TearsTool):
    """create a new workspace from a template, an existing workspace, or empty.

    sources are mutually exclusive. on success the new workspace becomes
    the conversation's pinned workspace (one trip through
    :func:`pin.set_pin`). all multi-row inserts run under a single
    asyncpg transaction so a failure anywhere rolls back to a consistent
    state.
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
        nats_client: Any = None,
        namespace: str | None = None,
        validators: list[ValidatorEntry] | None = None,
    ) -> None:
        """
        binds tool to collections, sandbox, conversation context, and pool.

        :param workspace_collection: collection providing find_by_agent_and_name
        :ptype workspace_collection: WorkspaceCollection
        :param workspace_file_collection: collection providing find_by_workspace
        :ptype workspace_file_collection: WorkspaceFileCollection
        :param workspace_file_version_collection: collection accepted for
            symmetry with sibling tools; transactional inserts go directly
            through the pool to share one transaction
        :ptype workspace_file_version_collection: WorkspaceFileVersionCollection
        :param sandbox: workspace sandbox enforcing template-read constraints
        :ptype sandbox: WorkspaceSandbox
        :param context_provider: zero-arg callable returning the current
            conversation's ToolContextManager
        :ptype context_provider: Callable[[], ToolContextManager]
        :param agent_id: identifier of agent owning the new workspace
        :ptype agent_id: UUID
        :param db_pool: asyncpg pool (or pool-like) supplying acquire+transaction
        :ptype db_pool: Any
        :param nats_client: NATS client for audit publish; None in tests /
            bootstrap to skip the audit step
        :ptype nats_client: Any
        :param namespace: NATS subject namespace for audit subject
        :ptype namespace: str | None
        :param validators: per-pattern validator entries; every seeded
            file (from template or source workspace) is validated before
            the batch INSERTs run. first failure aborts the create
            transaction so no partial workspace is left behind.
        :ptype validators: list[ValidatorEntry] | None
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
        self._validators = validators

    async def execute(self, **kwargs: Any) -> ToolResult:
        """
        create new workspace per kwargs; auto-pin on success.

        accepts ``name`` (required), optional ``description``, and at most
        one of ``from_template`` / ``from_workspace``. all failures arrive
        as :class:`ToolResult` with ``success=False``; this method never
        raises.

        :param kwargs: must include ``name``; optional ``description``,
            ``from_template``, ``from_workspace``
        :ptype kwargs: Any
        :return: tool result reporting create or error
        :rtype: ToolResult
        """
        name = kwargs.get("name", "")
        description = kwargs.get("description")
        from_template = kwargs.get("from_template")
        from_workspace = kwargs.get("from_workspace")

        # guard clause (entry-time input validation) per CLAUDE.md rule.
        if from_template and from_workspace:
            return ToolResult(
                success=False,
                content="",
                error=(
                    "from_template and from_workspace are mutually exclusive; "
                    "specify at most one source"
                ),
            )

        result: ToolResult
        correlation_id = uuid7()
        effective_template: str | None = None
        files_count = 0
        workspace_id: UUID | None = None
        try:
            files, source_template_name = await self._resolve_source_files(
                from_template, from_workspace
            )
            effective_template = from_template
            if effective_template is None and from_workspace is not None:
                effective_template = source_template_name
            files_count = len(files)
            workspace_id = await self._insert_all(
                name=name,
                description=description,
                template_name=effective_template,
                files=files,
                correlation_id=correlation_id,
            )
            try:
                await pin.set_pin(
                    self._context_provider(),
                    workspace_id=workspace_id,
                    workspace_name=name,
                    pinned_by_actor_id=self._agent_id,
                )
            except Exception as pin_exc:
                log.exception(
                    "workspace_create pin failed after insert: %s", pin_exc,
                )
                result = ToolResult(
                    success=False,
                    content="",
                    error=f"create succeeded but pin failed: {pin_exc}",
                )
            else:
                # defense-in-depth: publish_workspace_event swallows its own
                # publish failures, but wrap again so any unforeseen error
                # from the helper itself cannot taint a successful return.
                try:
                    if self._namespace is not None:
                        await audit.publish_workspace_event(
                            nats_client=self._nats_client,
                            namespace=self._namespace,
                            event_type="workspace.create",
                            actor_id=self._agent_id,
                            agent_id=self._agent_id,
                            resource_type="workspace",
                            resource_id=str(workspace_id),
                            action="create",
                            details={
                                "name": name,
                                "template_name": effective_template,
                                "files_changed": files_count,
                            },
                            correlation_id=correlation_id,
                        )
                # NOSILENT: audit failure must never taint a successful create
                except Exception as audit_exc:
                    log.exception(
                        "workspace_create audit publish swallow caught: %s",
                        audit_exc,
                    )
                result = ToolResult(
                    success=True,
                    content=(
                        f"created workspace {name!r} (workspace_id={workspace_id})"
                    ),
                )
        except _CreateError as exc:
            result = ToolResult(success=False, content="", error=str(exc))
        except WorkspaceValidationError as exc:
            result = ToolResult(
                success=False,
                content="",
                error=f"validation failed: {exc.pattern} -> {exc.reason}",
            )
        except asyncpg.exceptions.UniqueViolationError:
            result = ToolResult(
                success=False,
                content="",
                error=f"workspace {name!r} already exists for this agent",
            )
        except Exception as exc:
            log.exception("workspace_create failed: %s", exc)
            result = ToolResult(
                success=False,
                content="",
                error=f"create failed: {exc}",
            )
        return result

    async def _resolve_source_files(
        self,
        from_template: str | None,
        from_workspace: str | None,
    ) -> tuple[list[tuple[str, bytes, str]], str | None]:
        """
        resolve seed files from the requested source; empty list when neither.

        :param from_template: template directory name under templates root
        :ptype from_template: str | None
        :param from_workspace: existing workspace name to fork from
        :ptype from_workspace: str | None
        :return: pair of (file triples, inherited template_name) where the
            inherited template_name is set only when forking and the source
            workspace itself was created from a template
        :rtype: tuple[list[tuple[str, bytes, str]], str | None]
        :raises _CreateError: if from_workspace name is unknown
        """
        files: list[tuple[str, bytes, str]] = []
        inherited_template: str | None = None
        if from_template:
            files = await self._read_template_files(from_template)
        elif from_workspace:
            files, inherited_template = await self._read_source_workspace_files(
                from_workspace
            )
        result = (files, inherited_template)
        return result

    async def _read_template_files(
        self, template_name: str,
    ) -> list[tuple[str, bytes, str]]:
        """
        walk the named template directory and gate every file via sandbox.

        sandbox enforcement must run on the event loop (the sandbox is a
        plain Python object and raising from a worker thread is fine, but
        keeping enforce calls on the main loop keeps the failure path
        cheap and allows future async-sandbox extension). blocking
        filesystem walk + read_bytes is dispatched to :func:`asyncio.to_thread`
        so the event loop stays responsive on large templates.

        :param template_name: template directory name under the templates root
        :ptype template_name: str
        :return: list of (relative_path, content_bytes, sha256_hex) triples
        :rtype: list[tuple[str, bytes, str]]
        """
        templates_root = self._sandbox.resolve_fs_path(template_name, "templates")
        # collect (relative, path) pairs in thread, then enforce sandbox
        # on each relative key on the main loop (sandbox.enforce may raise
        # and surface cleanly that way).
        candidates = await asyncio.to_thread(
            _collect_template_paths, templates_root,
        )
        for relative, _path in candidates:
            self._sandbox.enforce("read", relative)
        triples = await asyncio.to_thread(
            _read_template_bytes, candidates,
        )
        return triples

    async def _read_source_workspace_files(
        self,
        source_name: str,
    ) -> tuple[list[tuple[str, bytes, str]], str | None]:
        """
        copy file rows from an existing workspace to seed the new one.

        :param source_name: existing workspace name to fork from
        :ptype source_name: str
        :return: pair of (file triples, source template_name) so the
            new workspace can inherit reset behavior from its parent
        :rtype: tuple[list[tuple[str, bytes, str]], str | None]
        :raises _CreateError: if source workspace does not exist
        """
        source = await self._workspaces.find_by_agent_and_name(
            self._agent_id, source_name
        )
        if source is None:
            raise _CreateError(f"source workspace {source_name!r} not found")
        rows = await self._files.find_by_workspace(source.id)
        files = [(f.relative_path, f.content, f.sha256) for f in rows]
        result = (files, source.template_name)
        return result

    async def _insert_all(
        self,
        *,
        name: str,
        description: str | None,
        template_name: str | None,
        files: list[tuple[str, bytes, str]],
        correlation_id: UUID,
    ) -> UUID:
        """
        insert workspace + file head + journal rows in one transaction.

        carries the inherited ``template_name`` from the source workspace
        when the caller forked one; otherwise uses the explicit
        ``from_template`` value. ``current_version`` lands at 0 for an
        empty workspace and 1 when at least one file is seeded.

        :param name: unique workspace name within agent
        :ptype name: str
        :param description: optional description text
        :ptype description: str | None
        :param template_name: template name to record on the new row
        :ptype template_name: str | None
        :param files: pre-resolved file triples to seed the workspace
        :ptype files: list[tuple[str, bytes, str]]
        :return: identifier of the newly inserted workspace
        :rtype: UUID
        :raises asyncpg.exceptions.UniqueViolationError: on duplicate name
        """
        workspace_id = uuid7()
        now = datetime.now(UTC)
        current_version = 1 if files else 0

        # validator dispatch runs BEFORE any INSERT so a rejection leaves
        # no partial workspace. runs outside the transaction because
        # validators are pure predicates over bytes; keeping them out of
        # the tx avoids holding a connection while a validator performs
        # slow imports on first call.
        if self._validators:
            for relative, content, _sha in files:
                dispatch_validators(self._validators, relative, content)

        async with self._db_pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    _INSERT_WORKSPACE_SQL,
                    workspace_id,
                    self._agent_id,
                    name,
                    description,
                    template_name,
                    self._agent_id,  # created_by = agent
                    current_version,
                    now,
                    now,
                )
                for relative, content, sha in files:
                    file_id = uuid7()
                    version_id = uuid7()
                    await conn.execute(
                        _INSERT_WORKSPACE_FILE_SQL,
                        file_id,
                        workspace_id,
                        relative,
                        content,
                        sha,
                        1,
                        now,
                    )
                    await conn.execute(
                        _INSERT_WORKSPACE_FILE_VERSION_SQL,
                        version_id,
                        workspace_id,
                        relative,
                        1,
                        content,
                        sha,
                        "create",
                        None,
                        self._agent_id,
                        correlation_id,
                        now,
                    )
        return workspace_id

    def mcp_schema(self) -> MCPToolDefinition:
        """
        returns the MCP definition for this tool.

        :return: MCP-compatible tool definition
        :rtype: MCPToolDefinition
        """
        return MCPToolDefinition(
            name=self.mcp_name(),
            version=self.mcp_version(),
            description="create a new workspace from a template, an existing workspace, or empty",
            input_schema=_INPUT_SCHEMA,
        )

    def mcp_name(self) -> str:
        """
        returns the namespaced tool name advertised to MCP clients.

        :return: tool name
        :rtype: str
        """
        return "threetears.workspace.create"

    def mcp_version(self) -> str:
        """
        returns the semver-compatible tool version.

        :return: version string
        :rtype: str
        """
        return "1.0"


def _collect_template_paths(
    templates_root: Path,
) -> list[tuple[str, Path]]:
    """sync walker: enumerate template-root regular files as ``(relative, path)``.

    split from :meth:`WorkspaceCreateTool._read_template_files` so the
    blocking walk runs inside :func:`asyncio.to_thread` while sandbox
    enforcement stays on the main event loop. return value is sorted
    for deterministic validator / sandbox ordering.

    :param templates_root: sandbox-resolved template directory
    :ptype templates_root: Path
    :return: sorted list of ``(relative_path, full_path)`` tuples
    :rtype: list[tuple[str, Path]]
    """
    pairs: list[tuple[str, Path]] = []
    for path in sorted(templates_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(templates_root).as_posix()
        pairs.append((relative, path))
    return pairs


def _read_template_bytes(
    pairs: list[tuple[str, Path]],
) -> list[tuple[str, bytes, str]]:
    """sync reader: dereference ``(relative, path)`` pairs into byte triples.

    dispatched via :func:`asyncio.to_thread` by callers so the blocking
    ``read_bytes`` + sha256 work happens off the event loop.

    :param pairs: ``(relative, full_path)`` tuples from walker
    :ptype pairs: list[tuple[str, Path]]
    :return: list of ``(relative_path, content_bytes, sha256_hex)`` triples
    :rtype: list[tuple[str, bytes, str]]
    """
    triples: list[tuple[str, bytes, str]] = []
    for relative, path in pairs:
        content = path.read_bytes()
        sha = hashlib.sha256(content).hexdigest()
        triples.append((relative, content, sha))
    return triples


class _CreateError(Exception):
    """internal sentinel for clean error-as-data branches inside execute."""


def _build(**kwargs: Any) -> WorkspaceCreateTool:
    """
    constructs a :class:`WorkspaceCreateTool` from the factory dep bundle.

    consumes ``workspace_collection``, ``workspace_file_collection``,
    ``workspace_file_version_collection``, ``sandbox``,
    ``context_provider``, ``agent_id``, and ``db_pool``; ignores the
    rest. registered with :mod:`threetears.agent.workspace.factory` on
    import so :func:`build_workspace_tools` emits this tool.

    :param kwargs: full factory dependency bundle
    :ptype kwargs: Any
    :return: constructed tool
    :rtype: WorkspaceCreateTool
    """
    return WorkspaceCreateTool(
        workspace_collection=kwargs["workspace_collection"],
        workspace_file_collection=kwargs["workspace_file_collection"],
        workspace_file_version_collection=kwargs["workspace_file_version_collection"],
        sandbox=kwargs["sandbox"],
        context_provider=kwargs["context_provider"],
        agent_id=kwargs["agent_id"],
        db_pool=kwargs["db_pool"],
        nats_client=kwargs.get("nats_client"),
        namespace=kwargs.get("namespace"),
        validators=_resolve_validators(kwargs),
    )


register_tool_builder(_build)
