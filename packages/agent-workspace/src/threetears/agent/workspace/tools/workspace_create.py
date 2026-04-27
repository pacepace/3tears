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
from pydantic import BaseModel

# namespace-task-01 follow-up (post-emit re-materialization wave): the
# paired ``platform.namespaces`` write for every workspace-create now
# rides a NATS event published on
# :meth:`threetears.nats.Subjects.workspaces_create`. the hub-side
# :class:`aibots.hub.workspace.namespace_emitter
# .WorkspaceNamespaceEmitter` subscribes (no queue group, every
# replica observes) and upserts the row via
# :class:`HubNamespaceCollection`. the agent-side L3 proxy routes
# writes to the agent's own ``agent_<hex>`` schema, which has no
# ``namespaces`` table -- the hub owns direct DB access and is the
# SOLE writer of platform-scoped catalog rows. mirrors the tool-pod
# registration path (``ToolNamespaceEmitter`` on
# ``{ns}.tools.register``).

from threetears.agent.audit import AuditEvent, publish_audit
from threetears.core.namespaces import PLURAL_PREFIX_WORKSPACE, build_namespace_name
from threetears.agent.tools.base_tool import (
    MCPToolDefinition,
    TearsTool,
    ToolResult,
)
from threetears.agent.tools.context import ToolContextManager
from threetears.nats import Subjects
from threetears.observe import get_logger

from threetears.agent.workspace import pin
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

__all__ = [
    "WorkspaceCreateEvent",
    "WorkspaceCreateTool",
]


class WorkspaceCreateEvent(BaseModel):
    """payload published to ``{ns}.workspaces.create`` after a
    successful agent-side workspace insert.

    carries the minimum identity + naming shape the hub-side
    :class:`aibots.hub.workspace.namespace_emitter
    .WorkspaceNamespaceEmitter` needs to upsert one ``workspace``-type
    row in ``platform.namespaces``. the field set mirrors the entity
    payload the agent used to assemble locally; the emitter stamps
    ``date_created`` / ``date_updated`` server-side from
    :func:`datetime.now(UTC)` so the hub is the timestamp authority.

    :param workspace_id: deterministic workspace UUID minted at insert
        time; reused as the namespace row's primary key so the hub
        upsert is idempotent under retry
    :ptype workspace_id: UUID
    :param namespace_name: pre-built canonical namespace name (e.g.
        ``workspaces.<workspace_id>``) produced by
        :func:`threetears.core.namespaces.build_namespace_name` on the
        agent side; the hub respects the agent-supplied name verbatim
    :ptype namespace_name: str
    :param schema_name: per-agent schema where the workspace's tables
        live (``agent_<hex>``); stamped on the namespace row so
        broker-side authorization can resolve the right backing
        schema for downstream tool calls
    :ptype schema_name: str
    :param owner_agent_id: owning agent UUID (the agent that created
        the workspace); always set, since workspace_create runs under
        an agent's tool-server
    :ptype owner_agent_id: UUID
    :param customer_id: owning customer UUID (resolved at create time
        from the live ToolCallScope or the constructor fallback); MAY
        be ``None`` when the create flow runs outside any conversation
        scope and the constructor was supplied no fallback -- the
        emitter still upserts the row but the hub authorize helper
        treats ``NULL`` customer_id as unroutable, so downstream tool
        calls deny until a backfill lands
    :ptype customer_id: UUID | None
    """

    workspace_id: UUID
    namespace_name: str
    schema_name: str
    owner_agent_id: UUID
    customer_id: UUID | None = None

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
        *,
        nats_client: Any,
        namespace: str,
        validators: list[ValidatorEntry] | None = None,
        customer_id: UUID | None = None,
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
        :param nats_client: connected canonical NATS wrapper client.
            REQUIRED. carries two distinct publishes per successful
            create: (a) the per-tool ``workspace.create`` audit event
            on ``{ns}.audit.workspace.>``, and (b) the
            :class:`WorkspaceCreateEvent` on ``{ns}.workspaces.create``
            consumed by the hub's
            :class:`aibots.hub.workspace.namespace_emitter
            .WorkspaceNamespaceEmitter` to upsert the paired
            ``platform.namespaces`` row of type ``workspace``. the tool
            will not degrade silently when it is omitted -- a
            misconfigured wiring fails loudly.
        :ptype nats_client: Any
        :param namespace: NATS subject namespace prefix. REQUIRED for
            both the audit subject and the workspace-create event
            subject; passed through to :class:`Subjects` for subject
            construction
        :ptype namespace: str
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
        self._customer_id = customer_id

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
                error=("from_template and from_workspace are mutually exclusive; specify at most one source"),
            )

        result: ToolResult
        correlation_id = uuid7()
        effective_template: str | None = None
        files_count = 0
        workspace_id: UUID | None = None
        try:
            files, source_template_name = await self._resolve_source_files(from_template, from_workspace)
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
                    "workspace_create pin failed after insert: %s",
                    pin_exc,
                )
                result = ToolResult(
                    success=False,
                    content="",
                    error=f"create succeeded but pin failed: {pin_exc}",
                )
            else:
                # defense-in-depth: additive per-tool event on top of
                # the baseline ``tool.call`` emitted by ToolServer.
                #
                # workspace_create is the one tool that publishes before a
                # Workspace entity exists to hand to workspace_audit_identity
                # (we're still INSIDE the creation flow). we read the call
                # scope directly for actor/calling/customer, and source
                # owner_agent_id + resource_namespace_id from the values we
                # just inserted. any missing dimension is a wiring bug,
                # raised by the guard below and surfaced by the outer
                # swallow.
                try:
                    if self._namespace is not None:
                        from threetears.agent.tools.call_scope import (
                            current_scope as _current_scope,
                        )

                        _scope = _current_scope()
                        if _scope is None:
                            raise RuntimeError(
                                "workspace_create audit: no ToolCallScope "
                                "installed; every tool dispatch must run "
                                "under enter_call_scope."
                            )
                        _ctx = _scope.context
                        if _ctx.user_id is None or _ctx.agent_id is None:
                            raise RuntimeError(
                                "workspace_create audit: scope missing "
                                "user_id or agent_id; cannot publish the "
                                "identity tuple."
                            )
                        _audit_customer = _ctx.customer_id or self._customer_id
                        if _audit_customer is None:
                            raise RuntimeError(
                                "workspace_create audit: no customer_id in "
                                "scope or constructor; workspace was created "
                                "without an owning customer."
                            )
                        event = AuditEvent(
                            id=uuid7(),
                            timestamp=datetime.now(UTC),
                            event_type="workspace.create",
                            actor_user_id=_ctx.user_id,
                            calling_agent_id=_ctx.agent_id,
                            owner_agent_id=self._agent_id,
                            customer_id=_audit_customer,
                            resource_namespace_id=workspace_id,
                            resource_namespace_type="workspace",
                            action="create",
                            outcome="success",
                            correlation_id=correlation_id,
                            details={
                                "workspace_resource_id": str(workspace_id),
                                "name": name,
                                "template_name": effective_template,
                                "files_changed": files_count,
                            },
                        )
                        await publish_audit(
                            event,
                            nats_client=self._nats_client,
                            namespace=self._namespace,
                        )
                # NOSILENT: audit failure must never taint a successful create
                except Exception as audit_exc:
                    log.exception(
                        "workspace_create audit publish swallow caught: %s",
                        audit_exc,
                    )
                result = ToolResult(
                    success=True,
                    content=(f"created workspace {name!r} (workspace_id={workspace_id})"),
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
            files, inherited_template = await self._read_source_workspace_files(from_workspace)
        result = (files, inherited_template)
        return result

    async def _read_template_files(
        self,
        template_name: str,
    ) -> list[tuple[str, bytes, str]]:
        """
        walk the named template directory and enforce syntactic sanity.

        template files are rooted under ``templates_dir`` and were
        validated at packaging time; the agent creating the workspace
        is implicitly its owner. namespace-task-01 phase 7 retires the
        per-path rbac glob check here because no workspace namespace
        exists yet (the call SITE is ``create``). syntactic validation
        (absolute-path, parent-ref, control char) still runs via
        :meth:`WorkspaceSandbox.validate_syntax` so malformed template
        keys cannot slip into the fresh workspace.

        blocking filesystem walk + read_bytes is dispatched to
        :func:`asyncio.to_thread` so the event loop stays responsive on
        large templates.

        :param template_name: template directory name under the templates root
        :ptype template_name: str
        :return: list of (relative_path, content_bytes, sha256_hex) triples
        :rtype: list[tuple[str, bytes, str]]
        """
        templates_root = self._sandbox.resolve_fs_path(template_name, "templates")
        candidates = await asyncio.to_thread(
            _collect_template_paths,
            templates_root,
        )
        for relative, _path in candidates:
            self._sandbox.validate_syntax(relative)
        triples = await asyncio.to_thread(
            _read_template_bytes,
            candidates,
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
        source = await self._workspaces.find_by_agent_and_name(self._agent_id, source_name)
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

        # customer_id source for the paired namespace row: prefer the
        # live ToolCallScope (set by the runtime for real calls) over
        # the constructor kwarg (supplied for tests and bootstrap).
        # when neither is present we fall back to None -- the namespace
        # row ends up with a NULL customer_id, which the authorize
        # helper treats as unroutable, and downstream tool calls will
        # deny until the migration backfill lands the right value.
        from threetears.agent.tools.call_scope import current_scope

        scope = current_scope()
        customer_id: UUID | None = (
            scope.context.customer_id
            if scope is not None and scope.context.customer_id is not None
            else self._customer_id
        )
        schema_name = f"agent_{self._agent_id.hex}"
        namespace_name = build_namespace_name(
            PLURAL_PREFIX_WORKSPACE, str(workspace_id)
        )

        # workspace_create is owner-only by construction: it owns the
        # physical rows it is about to materialize. the workspace row
        # + file rows land under one ``conn.transaction()`` on the
        # dedicated ``db_pool``; the paired ``platform.namespaces``
        # row rides the agent's main NATS-proxy pool via
        # :meth:`NamespaceCollection.save_entity` (three-tier-task-01
        # phase F). the two writes cannot share one transaction
        # because the Collection proxies through a different broker
        # path, but the idempotent ``ON CONFLICT (id) DO UPDATE``
        # semantics on the namespace id (equal to ``workspace_id``)
        # let any retry converge on the same row.
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

        # workspace + files committed. publish the paired-namespace
        # event for the hub-side emitter to upsert
        # ``platform.namespaces``. customer_id may be None when the
        # create runs outside any conversation scope and the
        # constructor was supplied no fallback -- the emitter still
        # upserts the row but the hub authorize helper treats NULL
        # customer_id as unroutable, denying downstream tool calls
        # until a backfill lands.
        event = WorkspaceCreateEvent(
            workspace_id=workspace_id,
            namespace_name=namespace_name,
            schema_name=schema_name,
            owner_agent_id=self._agent_id,
            customer_id=customer_id,
        )
        await self._nats_client.publish(
            subject=Subjects.workspaces_create(),
            message=event,
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
    ``context_provider``, ``agent_id``, ``db_pool``, ``nats_client``,
    and ``namespace``; ignores the rest. registered with
    :mod:`threetears.agent.workspace.factory` on import so
    :func:`build_workspace_tools` emits this tool.

    ``nats_client`` and ``namespace`` are required keys -- the tool
    publishes one :class:`WorkspaceCreateEvent` on
    ``{ns}.workspaces.create`` per successful create so the hub-side
    emitter can upsert the paired namespace row. dropping them would
    leave the workspace catalog out of sync; the factory raises
    :class:`KeyError` rather than silently degrading.

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
        nats_client=kwargs["nats_client"],
        namespace=kwargs["namespace"],
        validators=_resolve_validators(kwargs),
        customer_id=kwargs.get("customer_id"),
    )


register_tool_builder(_build)
