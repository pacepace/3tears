"""workspace collections -- three-tier CRUD for workspace entities.

all three collections (workspaces, workspace_files, workspace_file_versions)
are :class:`~threetears.core.collections.schema_backed.SchemaBackedCollection`
subclasses. CRUD comes from the declarative :class:`TableSchema`;
domain queries (``find_by_agent``, ``find_by_workspace``,
``find_by_workspace_and_relative_path``, history-shaped selects) stay
on subclasses because their query shape is per-collection.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from threetears.core.collections.flush import WriteBuffer
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import (
    BYTES_TYPE,
    DATETIME_TYPE,
    INT_TYPE,
    STRING_TYPE,
    UUID_TYPE,
    Column,
    SchemaBackedCollection,
    TableSchema,
)
from threetears.core.config import CoreConfig
from threetears.observe import get_logger

from threetears.agent.workspace.entities import Workspace, WorkspaceFile, WorkspaceFileVersion

__all__ = [
    "WorkspaceCollection",
    "WorkspaceFileCollection",
    "WorkspaceFileVersionCollection",
]

log = get_logger(__name__)


class WorkspaceCollection(SchemaBackedCollection[Workspace]):
    """collection for Workspace entities with three-tier caching.

    CRUD is generated from :attr:`schema`. no CAS column -- the v034
    workspace write path relies on table-level SERIALIZABLE semantics
    composed at the transaction level by the calling tool, not
    row-level OCC on this collection.

    three-tier-task-01 phase F: the paired ``platform.namespaces``
    write that lived on the old hand-rolled save_to_postgres moved to
    :class:`WorkspaceCreateTool._insert_all`, which persists the
    namespace via :meth:`NamespaceCollection.save_entity` after the
    workspace transaction commits.
    """

    primary_key_column: str = "id"
    schema = TableSchema(
        name="workspaces",
        primary_key="id",
        columns=[
            Column("id", UUID_TYPE),
            Column("agent_id", UUID_TYPE, immutable=True),
            Column("name", STRING_TYPE),
            Column("description", STRING_TYPE, nullable=True),
            Column("template_name", STRING_TYPE, nullable=True),
            Column("created_by", UUID_TYPE, immutable=True),
            Column("current_version", INT_TYPE),
            Column("date_created", DATETIME_TYPE, immutable=True),
            Column("date_updated", DATETIME_TYPE),
            Column("date_deleted", DATETIME_TYPE, nullable=True),
        ],
    )

    def __init__(
        self,
        registry: CollectionRegistry,
        config: CoreConfig,
        postgres_pool: Any,
        nats_client: Any = None,
        write_buffer: WriteBuffer | None = None,
    ) -> None:
        """initialize collection with required dependencies.

        consumers compose multi-row transactions (see shard 11/12/14);
        collection exposes CRUD only. the ``postgres_pool`` kwarg is
        stored onto ``self.l3_pool`` so the generic CRUD path finds
        the pool uniformly with registry-resolved siblings.

        :param registry: collection registry for dependency injection
        :ptype registry: CollectionRegistry
        :param config: core configuration for flush strategy and caching
        :ptype config: CoreConfig
        :param postgres_pool: asyncpg pool bound to per-agent schema via search_path
        :ptype postgres_pool: Any
        :param nats_client: NATS client for L2 and invalidation signaling
        :ptype nats_client: Any
        :param write_buffer: optional deferred-write buffer for batched flushes
        :ptype write_buffer: WriteBuffer | None
        """
        super().__init__(registry, config, nats_client, write_buffer)
        # override the registry-resolved pool with the ctor-injected one.
        # callers that construct directly (tests, cross-agent tooling)
        # bypass the registry's l3 binding; this keeps the public
        # ``postgres_pool`` ctor surface without a second pool field
        self.l3_pool = postgres_pool

    @property
    def table_name(self) -> str:
        """returns database table name for this collection.

        :return: table name
        :rtype: str
        """
        return "workspaces"

    @property
    def entity_class(self) -> type[Workspace]:
        """returns entity class for this collection.

        :return: Workspace entity class
        :rtype: type[Workspace]
        """
        return Workspace

    async def find_by_agent(
        self,
        agent_id: UUID,
        *,
        include_deleted: bool = False,
    ) -> list[Workspace]:
        """
        fetches workspaces owned by agent, newest update first.

        defaults to live workspaces only -- rows with non-null
        date_deleted are excluded so list-style consumers (e.g. the
        ``threetears.workspace.list`` tool) never surface soft-deleted
        rows. pass ``include_deleted=True`` for admin or recovery flows
        that need the full set. promotes hits to L2 so peer pods see
        fresh rows on next miss; hydrated entities bind to this
        collection so subsequent mutations route through the same cache
        stack.

        :param agent_id: identifier of agent whose workspaces to list
        :ptype agent_id: UUID
        :param include_deleted: when True, return soft-deleted rows too
        :ptype include_deleted: bool
        :return: list of workspace entities, newest update first
        :rtype: list[Workspace]
        """
        if include_deleted:
            sql = "SELECT * FROM workspaces WHERE agent_id = $1 ORDER BY date_updated DESC"
        else:
            sql = "SELECT * FROM workspaces WHERE agent_id = $1 AND date_deleted IS NULL ORDER BY date_updated DESC"
        rows = await self.l3_pool.fetch(sql, agent_id)
        entities: list[Workspace] = []
        for row in rows:
            data = dict(row)
            entity = self.entity_class(data, is_new=False, collection=self)
            await self._save_to_l2(data["id"], data)
            entities.append(entity)
        return entities

    async def find_by_id(
        self,
        agent_id: UUID,
        workspace_id: UUID,
    ) -> Workspace | None:
        """
        fetches a live workspace by (agent_id, workspace_id).

        used by low-level runtime primitives (bind, materialize, recover)
        that hold a workspace_id directly and resolve the authoritative
        row. filters on ``date_deleted IS NULL`` so soft-deleted
        workspaces are not bindable -- a bind onto a tombstone would
        silently write capture rows against a row list operations no
        longer surface, which is a correctness hazard. callers that
        need the soft-deleted row (admin / recovery flows) use
        :meth:`find_by_id_and_agent` directly.

        ``agent_id`` is the partition column on the ``workspaces``
        table; the caller supplies it explicitly so a poisoned
        ``workspace_id`` cannot leak across agents.

        :param agent_id: agent partition the workspace belongs to
        :ptype agent_id: UUID
        :param workspace_id: identifier of workspace to fetch
        :ptype workspace_id: UUID
        :return: matching live workspace entity or None
        :rtype: Workspace | None
        """
        row = await self.l3_pool.fetchrow(
            "SELECT * FROM workspaces WHERE id = $1 AND agent_id = $2 AND date_deleted IS NULL",
            workspace_id,
            agent_id,
        )
        result: Workspace | None = None
        if row is not None:
            data = dict(row)
            entity = self.entity_class(data, is_new=False, collection=self)
            await self._save_to_l2(data["id"], data)
            result = entity
        return result

    async def find_by_id_and_agent(self, workspace_id: UUID, agent_id: UUID) -> Workspace | None:
        """
        fetches workspace by id under ownership constraint of agent.

        used by callers (e.g. fs_* tools resolving a pinned workspace) that
        hold a workspace_id from a pin snapshot and need the authoritative
        row without depending on the name (which may have been renamed).
        filters on agent_id so a poisoned pin cannot leak across agents.
        does NOT filter by date_deleted: callers decide whether to accept
        soft-deleted workspaces by inspecting the returned entity.

        :param workspace_id: identifier of workspace to fetch
        :ptype workspace_id: UUID
        :param agent_id: identifier of owning agent
        :ptype agent_id: UUID
        :return: matching workspace entity or None
        :rtype: Workspace | None
        """
        row = await self.l3_pool.fetchrow(
            "SELECT * FROM workspaces WHERE id = $1 AND agent_id = $2",
            workspace_id,
            agent_id,
        )
        result: Workspace | None = None
        if row is not None:
            data = dict(row)
            entity = self.entity_class(data, is_new=False, collection=self)
            await self._save_to_l2(data["id"], data)
            result = entity
        return result

    async def find_by_agent_and_name(self, agent_id: UUID, name: str) -> Workspace | None:
        """
        fetches a single workspace by owning agent and human name.

        names are unique within an agent (enforced by schema); returns
        None when no row matches. does NOT filter by date_deleted -- this
        is a low-level lookup used by callers (e.g. delete tool, history
        queries) that may need to find a soft-deleted workspace; callers
        that want only live workspaces must check date_deleted on the
        returned entity. hit promotes to L2 so a peer pod warm-loads on
        next miss.

        :param agent_id: identifier of owning agent
        :ptype agent_id: UUID
        :param name: workspace human name to look up
        :ptype name: str
        :return: matching workspace entity or None
        :rtype: Workspace | None
        """
        row = await self.l3_pool.fetchrow(
            "SELECT * FROM workspaces WHERE agent_id = $1 AND name = $2",
            agent_id,
            name,
        )
        result: Workspace | None = None
        if row is not None:
            data = dict(row)
            entity = self.entity_class(data, is_new=False, collection=self)
            await self._save_to_l2(data["id"], data)
            result = entity
        return result


class WorkspaceFileCollection(SchemaBackedCollection[WorkspaceFile]):
    """collection for WorkspaceFile head-state entities with three-tier caching.

    CRUD is generated from :attr:`schema`. no CAS column at this level
    -- writers needing optimistic concurrency compose it at the
    transaction level (fs_write, fs_edit, doc_* read-within-transaction
    + direct conn.fetchrow). content is BYTES_TYPE so base64 round-trip
    through L2 preserves arbitrary bytes including NULs.
    """

    primary_key_column: str = "id"
    schema = TableSchema(
        name="workspace_files",
        primary_key="id",
        columns=[
            Column("id", UUID_TYPE),
            Column("workspace_id", UUID_TYPE, immutable=True),
            Column("relative_path", STRING_TYPE, immutable=True),
            Column("content", BYTES_TYPE),
            Column("sha256", STRING_TYPE),
            Column("version", INT_TYPE),
            Column("date_updated", DATETIME_TYPE),
        ],
    )

    def __init__(
        self,
        registry: CollectionRegistry,
        config: CoreConfig,
        postgres_pool: Any,
        nats_client: Any = None,
        write_buffer: WriteBuffer | None = None,
    ) -> None:
        """initialize collection with required dependencies.

        consumers compose multi-row transactions that pair head-state
        upsert with journal insert; collection exposes CRUD only.

        :param registry: collection registry for dependency injection
        :ptype registry: CollectionRegistry
        :param config: core configuration for flush strategy and caching
        :ptype config: CoreConfig
        :param postgres_pool: asyncpg pool bound to per-agent schema via search_path
        :ptype postgres_pool: Any
        :param nats_client: NATS client for L2 and invalidation signaling
        :ptype nats_client: Any
        :param write_buffer: optional deferred-write buffer for batched flushes
        :ptype write_buffer: WriteBuffer | None
        """
        super().__init__(registry, config, nats_client, write_buffer)
        self.l3_pool = postgres_pool

    @property
    def table_name(self) -> str:
        """returns database table name for this collection.

        :return: table name
        :rtype: str
        """
        return "workspace_files"

    @property
    def entity_class(self) -> type[WorkspaceFile]:
        """returns entity class for this collection.

        :return: WorkspaceFile entity class
        :rtype: type[WorkspaceFile]
        """
        return WorkspaceFile

    async def find_by_workspace_and_relative_path(self, workspace_id: UUID, relative_path: str) -> WorkspaceFile | None:
        """
        fetches head-state file row for given (workspace_id, relative_path).

        used by fs_read / fs_list filters / fs_edit pre-reads outside an
        active transaction. writers that need OCC (fs_write, fs_edit,
        doc_*) read within their own transaction via direct conn.fetchrow
        so read-and-write share SERIALIZABLE semantics; this method
        targets the lookup-only path. returns None when no row matches.

        :param workspace_id: identifier of parent workspace
        :ptype workspace_id: UUID
        :param relative_path: workspace-relative path to look up
        :ptype relative_path: str
        :return: matching file entity or None
        :rtype: WorkspaceFile | None
        """
        row = await self.l3_pool.fetchrow(
            "SELECT * FROM workspace_files WHERE workspace_id = $1 AND relative_path = $2",
            workspace_id,
            relative_path,
        )
        result: WorkspaceFile | None = None
        if row is not None:
            data = dict(row)
            entity = self.entity_class(data, is_new=False, collection=self)
            await self._save_to_l2(data["id"], data)
            result = entity
        return result

    async def find_by_workspace(self, workspace_id: UUID) -> list[WorkspaceFile]:
        """
        fetches every head-state file row for given workspace.

        used by lifecycle composers (create-from-workspace, reset, diff)
        that need the current file set without going through the journal.
        promotes each row to L2 so peer pods warm on next miss; hydrated
        entities bind to this collection so subsequent mutations route
        through the same cache stack.

        :param workspace_id: identifier of parent workspace
        :ptype workspace_id: UUID
        :return: list of file entities for workspace
        :rtype: list[WorkspaceFile]
        """
        rows = await self.l3_pool.fetch(
            "SELECT * FROM workspace_files WHERE workspace_id = $1",
            workspace_id,
        )
        entities: list[WorkspaceFile] = []
        for row in rows:
            data = dict(row)
            entity = self.entity_class(data, is_new=False, collection=self)
            await self._save_to_l2(data["id"], data)
            entities.append(entity)
        return entities


class WorkspaceFileVersionCollection(SchemaBackedCollection[WorkspaceFileVersion]):
    """collection for WorkspaceFileVersion append-only journal entities.

    journal is strictly append-only; duplicate ``(workspace_id,
    relative_path, version)`` inserts raise via the UNIQUE constraint
    surfaced by asyncpg. no update or soft-delete semantics exist on
    this collection. delete by primary key is provided for test and
    retention cleanup only; production callers do not rewrite history.

    CRUD is generated from :attr:`schema` with ``on_conflict="raise"`` so
    the generator emits plain INSERT -- no ON CONFLICT clause, no DO
    UPDATE SET -- matching the journal semantics. duplicate ``(workspace_id,
    relative_path, version)`` inserts surface as
    :class:`asyncpg.exceptions.UniqueViolationError` from asyncpg.
    """

    primary_key_column: str = "id"
    schema = TableSchema(
        name="workspace_file_versions",
        primary_key="id",
        columns=[
            Column("id", UUID_TYPE),
            Column("workspace_id", UUID_TYPE, immutable=True),
            Column("relative_path", STRING_TYPE, immutable=True),
            Column("version", INT_TYPE, immutable=True),
            Column("content", BYTES_TYPE, immutable=True),
            Column("sha256", STRING_TYPE, immutable=True),
            Column("action", STRING_TYPE, immutable=True),
            Column("label", STRING_TYPE, nullable=True, immutable=True),
            Column("actor_id", UUID_TYPE, immutable=True),
            Column("correlation_id", UUID_TYPE, immutable=True),
            Column("date_created", DATETIME_TYPE, immutable=True),
        ],
        on_conflict="raise",
    )

    def __init__(
        self,
        registry: CollectionRegistry,
        config: CoreConfig,
        postgres_pool: Any,
        nats_client: Any = None,
        write_buffer: WriteBuffer | None = None,
    ) -> None:
        """initialize collection with required dependencies.

        :param registry: collection registry for dependency injection
        :ptype registry: CollectionRegistry
        :param config: core configuration for flush strategy and caching
        :ptype config: CoreConfig
        :param postgres_pool: asyncpg pool bound to per-agent schema via search_path
        :ptype postgres_pool: Any
        :param nats_client: NATS client for L2 and invalidation signaling
        :ptype nats_client: Any
        :param write_buffer: optional deferred-write buffer for batched flushes
        :ptype write_buffer: WriteBuffer | None
        """
        super().__init__(registry, config, nats_client, write_buffer)
        self.l3_pool = postgres_pool

    @property
    def table_name(self) -> str:
        """returns database table name for this collection.

        :return: table name
        :rtype: str
        """
        return "workspace_file_versions"

    @property
    def entity_class(self) -> type[WorkspaceFileVersion]:
        """
        returns entity class for this collection.

        :return: WorkspaceFileVersion entity class
        :rtype: type[WorkspaceFileVersion]
        """
        return WorkspaceFileVersion

    async def find_by_workspace(self, workspace_id: UUID, limit: int) -> list[WorkspaceFileVersion]:
        """
        fetches journal rows for workspace, newest-first, bounded by limit.

        backs ``threetears.workspace.history`` when no ``relative_path``
        is supplied: one ``SELECT ... ORDER BY date_created DESC LIMIT N``
        over the ``idx_workspace_file_versions_history`` index. hydrates
        into entities so the tool can surface metadata (action, label,
        actor, correlation, size) without re-loading content through a
        second query; callers filter the byte blob out of the response
        payload themselves.

        :param workspace_id: identifier of parent workspace
        :ptype workspace_id: UUID
        :param limit: maximum rows to return, enforced server-side
        :ptype limit: int
        :return: newest-first list of journal entities bounded by limit
        :rtype: list[WorkspaceFileVersion]
        """
        rows = await self.l3_pool.fetch(
            "SELECT * FROM workspace_file_versions WHERE workspace_id = $1 ORDER BY date_created DESC LIMIT $2",
            workspace_id,
            limit,
        )
        entities: list[WorkspaceFileVersion] = []
        for row in rows:
            data = dict(row)
            entity = self.entity_class(data, is_new=False, collection=self)
            entities.append(entity)
        return entities

    async def find_by_workspace_and_path(
        self, workspace_id: UUID, relative_path: str, limit: int
    ) -> list[WorkspaceFileVersion]:
        """
        fetches journal rows for one path in a workspace, newest-first.

        backs ``threetears.workspace.history`` when ``relative_path`` is
        supplied: same ordering/limit contract as
        :meth:`find_by_workspace`, narrowed to a single path key.

        :param workspace_id: identifier of parent workspace
        :ptype workspace_id: UUID
        :param relative_path: path key to filter on
        :ptype relative_path: str
        :param limit: maximum rows to return, enforced server-side
        :ptype limit: int
        :return: newest-first list of journal entities bounded by limit
        :rtype: list[WorkspaceFileVersion]
        """
        rows = await self.l3_pool.fetch(
            "SELECT * FROM workspace_file_versions "
            "WHERE workspace_id = $1 AND relative_path = $2 "
            "ORDER BY date_created DESC LIMIT $3",
            workspace_id,
            relative_path,
            limit,
        )
        entities: list[WorkspaceFileVersion] = []
        for row in rows:
            data = dict(row)
            entity = self.entity_class(data, is_new=False, collection=self)
            entities.append(entity)
        return entities
