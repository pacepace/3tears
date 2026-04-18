"""workspace collections -- three-tier CRUD for workspace entities."""

from __future__ import annotations

import base64
import json
import types
from datetime import datetime
from enum import Enum
from typing import Any, get_args, get_origin
from uuid import UUID

from threetears.core.collections.base import BaseCollection
from threetears.core.collections.flush import WriteBuffer
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import CoreConfig
from threetears.observe import get_logger

from threetears.agent.workspace.entities import Workspace, WorkspaceFile, WorkspaceFileVersion

__all__ = [
    "WorkspaceCollection",
    "WorkspaceFileCollection",
    "WorkspaceFileVersionCollection",
]

log = get_logger(__name__)


_WORKSPACE_FIELD_TYPES: dict[str, Any] = {
    "id": UUID,
    "agent_id": UUID,
    "name": str,
    "description": str | None,
    "template_name": str | None,
    "created_by": UUID,
    "current_version": int,
    "date_created": datetime,
    "date_updated": datetime,
    "date_deleted": datetime | None,
}

_WORKSPACE_FILE_FIELD_TYPES: dict[str, Any] = {
    "id": UUID,
    "workspace_id": UUID,
    "relative_path": str,
    "content": bytes,
    "sha256": str,
    "version": int,
    "date_updated": datetime,
}

_WORKSPACE_FILE_VERSION_FIELD_TYPES: dict[str, Any] = {
    "id": UUID,
    "workspace_id": UUID,
    "relative_path": str,
    "version": int,
    "content": bytes,
    "sha256": str,
    "action": str,
    "label": str | None,
    "actor_id": UUID,
    "correlation_id": UUID,
    "date_created": datetime,
}


def _json_serializer(obj: object) -> str | int | float | bool | None:
    """
    serializes non-JSON-native types for json.dumps at L2 boundary.

    :param obj: value requiring custom serialization
    :ptype obj: object
    :return: JSON-compatible scalar representation
    :rtype: str | int | float | bool | None
    :raises TypeError: if obj type is not handled
    """
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, bytes):
        return base64.b64encode(obj).decode("ascii")
    if isinstance(obj, Enum):
        result: str | int | float | bool | None = obj.value
        return result
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _resolve_base_type(type_hint: Any) -> type | None:
    """
    extracts concrete type from possibly-Optional or generic type hint.

    :param type_hint: Python type hint from _FIELD_TYPES mapping
    :ptype type_hint: Any
    :return: resolved base type or None when hint cannot be resolved
    :rtype: type | None
    """
    origin = get_origin(type_hint)
    result: type | None
    if origin is None:
        result = type_hint if isinstance(type_hint, type) else None
    elif origin is types.UnionType:
        args = get_args(type_hint)
        non_none = [a for a in args if a is not type(None)]
        if not non_none:
            result = None
        else:
            inner = non_none[0]
            inner_origin = get_origin(inner)
            result = inner_origin if inner_origin is not None else inner
    else:
        result = origin
    return result


def _deserialize_field(key: str, value: Any, field_types: dict[str, Any]) -> Any:
    """
    converts one JSON-decoded value back to its native Python type.

    :param key: field name used to look up declared type
    :ptype key: str
    :param value: JSON-decoded raw value
    :ptype value: Any
    :param field_types: mapping of field name to declared type
    :ptype field_types: dict[str, Any]
    :return: value converted to native Python type
    :rtype: Any
    """
    if value is None:
        return None
    base_type = _resolve_base_type(field_types.get(key))
    result: Any
    if base_type is UUID and isinstance(value, str):
        result = UUID(value)
    elif base_type is datetime and isinstance(value, str):
        result = datetime.fromisoformat(value)
    elif base_type is bytes and isinstance(value, str):
        result = base64.b64decode(value.encode("ascii"))
    elif base_type is bool and isinstance(value, (bool, int)):
        result = bool(value)
    elif base_type is int and isinstance(value, int):
        result = value
    else:
        result = value
    return result


def _deserialize_row(data: bytes, field_types: dict[str, Any]) -> dict[str, Any]:
    """
    decodes JSON bytes to row dict with typed field coercion.

    :param data: JSON-encoded bytes from L2 cache
    :ptype data: bytes
    :param field_types: mapping of field name to declared type
    :ptype field_types: dict[str, Any]
    :return: row dict with native Python types
    :rtype: dict[str, Any]
    """
    raw: dict[str, Any] = json.loads(data.decode("utf-8"))
    result: dict[str, Any] = {}
    for key, value in raw.items():
        result[key] = _deserialize_field(key, value, field_types)
    return result


class WorkspaceCollection(BaseCollection[Workspace]):
    """collection for Workspace entities with three-tier caching."""

    primary_key_column: str = "id"

    def __init__(
        self,
        registry: CollectionRegistry,
        config: CoreConfig,
        postgres_pool: Any,
        nats_client: Any = None,
        write_buffer: WriteBuffer | None = None,
    ) -> None:
        """
        initializes collection with required dependencies.

        consumers compose multi-row transactions (see shard 11/12/14);
        collection exposes CRUD only.

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
        self._postgres_pool = postgres_pool
        super().__init__(registry, config, nats_client, write_buffer)

    @property
    def table_name(self) -> str:
        """
        returns database table name for this collection.

        :return: table name
        :rtype: str
        """
        return "workspaces"

    @property
    def entity_class(self) -> type[Workspace]:
        """
        returns entity class for this collection.

        :return: Workspace entity class
        :rtype: type[Workspace]
        """
        return Workspace

    async def _fetch_from_postgres(self, entity_id: Any) -> dict[str, Any] | None:
        """
        reads workspace row by primary key.

        :param entity_id: workspace UUID to fetch
        :ptype entity_id: Any
        :return: row dict or None when missing
        :rtype: dict[str, Any] | None
        """
        row = await self._postgres_pool.fetchrow(
            "SELECT * FROM workspaces WHERE id = $1",
            entity_id,
        )
        result: dict[str, Any] | None = None if row is None else dict(row)
        return result

    async def _save_to_postgres(
        self,
        data: dict[str, Any],
        original_timestamp: datetime | None = None,
    ) -> int:
        """
        upserts workspace row with explicit column list.

        :param data: row dict keyed by column name
        :ptype data: dict[str, Any]
        :param original_timestamp: optional prior date_updated for OCC
        :ptype original_timestamp: datetime | None
        :return: affected row count
        :rtype: int
        """
        result = await self._postgres_pool.execute(
            """
            INSERT INTO workspaces (
                id, agent_id, name, description, template_name,
                created_by, current_version, date_created, date_updated, date_deleted
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (id) DO UPDATE SET
                name = EXCLUDED.name,
                description = EXCLUDED.description,
                template_name = EXCLUDED.template_name,
                current_version = EXCLUDED.current_version,
                date_updated = EXCLUDED.date_updated,
                date_deleted = EXCLUDED.date_deleted
            """,
            data["id"],
            data["agent_id"],
            data["name"],
            data.get("description"),
            data.get("template_name"),
            data["created_by"],
            data["current_version"],
            data["date_created"],
            data["date_updated"],
            data.get("date_deleted"),
        )
        affected: int = int(result.split()[-1])
        return affected

    async def _delete_from_postgres(self, entity_id: Any) -> None:
        """
        removes workspace row by primary key; FK cascade drops children.

        :param entity_id: workspace UUID to delete
        :ptype entity_id: Any
        """
        await self._postgres_pool.execute(
            "DELETE FROM workspaces WHERE id = $1",
            entity_id,
        )

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
        rows = await self._postgres_pool.fetch(sql, agent_id)
        entities: list[Workspace] = []
        for row in rows:
            data = dict(row)
            entity = self.entity_class(data, is_new=False, collection=self)
            await self._save_to_l2(data["id"], data)
            entities.append(entity)
        return entities

    async def find_by_id(
        self,
        workspace_id: UUID,
        *,
        agent_id: UUID | None = None,
    ) -> Workspace | None:
        """
        fetches workspace by id, optionally scoped to owning agent.

        used by low-level runtime primitives (bind, materialize, recover)
        that hold a workspace_id directly and resolve the authoritative
        row without context for an owning agent. filters on
        ``date_deleted IS NULL`` so soft-deleted workspaces are not
        bindable -- a bind onto a tombstone would silently write capture
        rows against a row list operations no longer surface, which is a
        correctness hazard. callers that need the soft-deleted row
        (admin/recovery flows) use :meth:`find_by_id_and_agent` directly.

        when ``agent_id`` is supplied, the query adds an ``AND agent_id =
        $2`` clause so a poisoned workspace_id cannot leak across agents.
        callers that already trust the id (internal bind/materialize)
        pass None and get the unscoped lookup.

        :param workspace_id: identifier of workspace to fetch
        :ptype workspace_id: UUID
        :param agent_id: optional owning-agent scope; when None, no
            ownership filter is applied
        :ptype agent_id: UUID | None
        :return: matching live workspace entity or None
        :rtype: Workspace | None
        """
        if agent_id is None:
            row = await self._postgres_pool.fetchrow(
                "SELECT * FROM workspaces WHERE id = $1 AND date_deleted IS NULL",
                workspace_id,
            )
        else:
            row = await self._postgres_pool.fetchrow(
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
        row = await self._postgres_pool.fetchrow(
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
        row = await self._postgres_pool.fetchrow(
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

    def _serialize(self, data: dict[str, Any]) -> bytes:
        """
        encodes row dict to JSON bytes for L2 storage.

        :param data: row dict keyed by column name
        :ptype data: dict[str, Any]
        :return: JSON-encoded bytes
        :rtype: bytes
        """
        return json.dumps(data, default=_json_serializer).encode("utf-8")

    def _deserialize(self, data: bytes) -> dict[str, Any]:
        """
        decodes JSON bytes from L2 into typed row dict.

        :param data: JSON-encoded bytes
        :ptype data: bytes
        :return: row dict with native Python types
        :rtype: dict[str, Any]
        """
        return _deserialize_row(data, _WORKSPACE_FIELD_TYPES)


class WorkspaceFileCollection(BaseCollection[WorkspaceFile]):
    """collection for WorkspaceFile head-state entities with three-tier caching."""

    primary_key_column: str = "id"

    def __init__(
        self,
        registry: CollectionRegistry,
        config: CoreConfig,
        postgres_pool: Any,
        nats_client: Any = None,
        write_buffer: WriteBuffer | None = None,
    ) -> None:
        """
        initializes collection with required dependencies.

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
        self._postgres_pool = postgres_pool
        super().__init__(registry, config, nats_client, write_buffer)

    @property
    def table_name(self) -> str:
        """
        returns database table name for this collection.

        :return: table name
        :rtype: str
        """
        return "workspace_files"

    @property
    def entity_class(self) -> type[WorkspaceFile]:
        """
        returns entity class for this collection.

        :return: WorkspaceFile entity class
        :rtype: type[WorkspaceFile]
        """
        return WorkspaceFile

    async def _fetch_from_postgres(self, entity_id: Any) -> dict[str, Any] | None:
        """
        reads workspace_file head row by primary key.

        :param entity_id: workspace_file UUID to fetch
        :ptype entity_id: Any
        :return: row dict or None when missing
        :rtype: dict[str, Any] | None
        """
        row = await self._postgres_pool.fetchrow(
            "SELECT * FROM workspace_files WHERE id = $1",
            entity_id,
        )
        result: dict[str, Any] | None = None if row is None else dict(row)
        return result

    async def _save_to_postgres(
        self,
        data: dict[str, Any],
        original_timestamp: datetime | None = None,
    ) -> int:
        """
        upserts workspace_file head row by primary key.

        :param data: row dict keyed by column name
        :ptype data: dict[str, Any]
        :param original_timestamp: optional prior date_updated for OCC
        :ptype original_timestamp: datetime | None
        :return: affected row count
        :rtype: int
        """
        result = await self._postgres_pool.execute(
            """
            INSERT INTO workspace_files (
                id, workspace_id, relative_path, content,
                sha256, version, date_updated
            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (id) DO UPDATE SET
                content = EXCLUDED.content,
                sha256 = EXCLUDED.sha256,
                version = EXCLUDED.version,
                date_updated = EXCLUDED.date_updated
            """,
            data["id"],
            data["workspace_id"],
            data["relative_path"],
            data["content"],
            data["sha256"],
            data["version"],
            data["date_updated"],
        )
        affected: int = int(result.split()[-1])
        return affected

    async def _delete_from_postgres(self, entity_id: Any) -> None:
        """
        removes workspace_file head row by primary key.

        :param entity_id: workspace_file UUID to delete
        :ptype entity_id: Any
        """
        await self._postgres_pool.execute(
            "DELETE FROM workspace_files WHERE id = $1",
            entity_id,
        )

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
        row = await self._postgres_pool.fetchrow(
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
        rows = await self._postgres_pool.fetch(
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

    def _serialize(self, data: dict[str, Any]) -> bytes:
        """
        encodes row dict to JSON bytes for L2 storage with bytes base64-encoded.

        :param data: row dict keyed by column name
        :ptype data: dict[str, Any]
        :return: JSON-encoded bytes
        :rtype: bytes
        """
        return json.dumps(data, default=_json_serializer).encode("utf-8")

    def _deserialize(self, data: bytes) -> dict[str, Any]:
        """
        decodes JSON bytes from L2 into typed row dict.

        :param data: JSON-encoded bytes
        :ptype data: bytes
        :return: row dict with native Python types
        :rtype: dict[str, Any]
        """
        return _deserialize_row(data, _WORKSPACE_FILE_FIELD_TYPES)


class WorkspaceFileVersionCollection(BaseCollection[WorkspaceFileVersion]):
    """collection for WorkspaceFileVersion append-only journal entities.

    journal is strictly append-only; duplicate (workspace_id, relative_path,
    version) inserts raise via UNIQUE constraint. no update or soft-delete
    semantics exist on this collection. delete by primary key is provided for
    test and retention cleanup only; production callers do not rewrite history.
    """

    primary_key_column: str = "id"

    def __init__(
        self,
        registry: CollectionRegistry,
        config: CoreConfig,
        postgres_pool: Any,
        nats_client: Any = None,
        write_buffer: WriteBuffer | None = None,
    ) -> None:
        """
        initializes collection with required dependencies.

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
        self._postgres_pool = postgres_pool
        super().__init__(registry, config, nats_client, write_buffer)

    @property
    def table_name(self) -> str:
        """
        returns database table name for this collection.

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

    async def _fetch_from_postgres(self, entity_id: Any) -> dict[str, Any] | None:
        """
        reads workspace_file_versions journal row by primary key.

        :param entity_id: journal row UUID to fetch
        :ptype entity_id: Any
        :return: row dict or None when missing
        :rtype: dict[str, Any] | None
        """
        row = await self._postgres_pool.fetchrow(
            "SELECT * FROM workspace_file_versions WHERE id = $1",
            entity_id,
        )
        result: dict[str, Any] | None = None if row is None else dict(row)
        return result

    async def _save_to_postgres(
        self,
        data: dict[str, Any],
        original_timestamp: datetime | None = None,
    ) -> int:
        """
        inserts journal row; raises on UNIQUE conflict (journal is append-only).

        no ON CONFLICT clause: duplicate (workspace_id, relative_path, version)
        attempts are rejected by the UNIQUE constraint. original_timestamp is
        accepted for signature compatibility with BaseCollection but ignored
        because journal rows are never updated.

        :param data: row dict keyed by column name
        :ptype data: dict[str, Any]
        :param original_timestamp: accepted for signature compatibility, unused
        :ptype original_timestamp: datetime | None
        :return: affected row count, always 1 on success
        :rtype: int
        :raises asyncpg.UniqueViolationError: on duplicate triple insert
        """
        result = await self._postgres_pool.execute(
            """
            INSERT INTO workspace_file_versions (
                id, workspace_id, relative_path, version, content,
                sha256, action, label, actor_id, correlation_id, date_created
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """,
            data["id"],
            data["workspace_id"],
            data["relative_path"],
            data["version"],
            data["content"],
            data["sha256"],
            data["action"],
            data.get("label"),
            data["actor_id"],
            data["correlation_id"],
            data["date_created"],
        )
        affected: int = int(result.split()[-1])
        return affected

    async def _delete_from_postgres(self, entity_id: Any) -> None:
        """
        removes journal row by primary key; reserved for retention and tests.

        production callers do not rewrite history; this is provided only so
        the BaseCollection abstract contract is satisfied and retention jobs
        can prune rows under operator control.

        :param entity_id: journal row UUID to delete
        :ptype entity_id: Any
        """
        await self._postgres_pool.execute(
            "DELETE FROM workspace_file_versions WHERE id = $1",
            entity_id,
        )

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
        rows = await self._postgres_pool.fetch(
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
        rows = await self._postgres_pool.fetch(
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

    def _serialize(self, data: dict[str, Any]) -> bytes:
        """
        encodes row dict to JSON bytes for L2 storage with bytes base64-encoded.

        :param data: row dict keyed by column name
        :ptype data: dict[str, Any]
        :return: JSON-encoded bytes
        :rtype: bytes
        """
        return json.dumps(data, default=_json_serializer).encode("utf-8")

    def _deserialize(self, data: bytes) -> dict[str, Any]:
        """
        decodes JSON bytes from L2 into typed row dict.

        :param data: JSON-encoded bytes
        :ptype data: bytes
        :return: row dict with native Python types
        :rtype: dict[str, Any]
        """
        return _deserialize_row(data, _WORKSPACE_FILE_VERSION_FIELD_TYPES)
