"""Memories collection -- three-tier CRUD for memory entities."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID

from threetears.core.collections.base import BaseCollection
from threetears.core.collections.flush import WriteBuffer
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import CoreConfig
from threetears.observe import get_logger

from threetears.agent.memory.authorize import (
    ACTION_MEMORY_READ,
    ACTION_MEMORY_WRITE,
    MemoryAuthorizerDependencies,
    authorize_memory_access,
    ensure_memory_owner_assignment,
)
from threetears.agent.memory.entities import MemoryEntity

__all__ = [
    "MemoriesCollection",
]

log = get_logger(__name__)

# Field type mapping for JSON serialization/deserialization
_FIELD_TYPES: dict[str, Any] = {
    "memory_id": UUID,
    "agent_id": UUID,
    "customer_id": UUID,
    "user_id": UUID,
    "conversation_id": UUID,
    "message_id_source": UUID,
    "type_memory": str,
    "content": str,
    "embedding": list[float],
    "is_deleted": bool,
    "media_id": UUID | None,
    "date_created": datetime,
    "date_deleted": datetime | None,
    "date_updated": datetime | None,
}


def _encode_embedding(value: object) -> str | None:
    """Encode an embedding value for the pgvector ``vector`` column.

    asyncpg has no native pgvector codec; values must be passed as
    the literal textual representation ``[v1, v2, ...]``. Lists
    produced by entity setters or extraction code get JSON-encoded
    here at the WRITE boundary. Already-encoded strings pass through.
    ``None`` passes through too so callers that omit embeddings stay
    working.

    :param value: list of floats, string, or None
    :ptype value: object
    :return: bracketed textual vector, passthrough string, or None
    :rtype: str | None
    """
    if value is None:
        result: str | None = None
    elif isinstance(value, str):
        result = value
    elif isinstance(value, list):
        result = json.dumps(value)
    else:
        result = str(value)
    return result


def _to_naive_utc(value: datetime | None) -> datetime | None:
    """Convert a timezone-aware datetime to naive UTC for TIMESTAMP columns.

    YugabyteDB TIMESTAMP columns (see memory migrations v001/v004) are
    timezone-naive. Per CLAUDE.md "Datetime Handling", aware datetimes
    must be converted at the WRITE boundary. Naive inputs are returned
    as-is; ``None`` passes through.

    :param value: aware or naive datetime, or None
    :ptype value: datetime | None
    :return: naive UTC datetime or None
    :rtype: datetime | None
    """
    if value is None:
        result: datetime | None = None
    elif value.tzinfo is None:
        result = value
    else:
        result = value.astimezone(UTC).replace(tzinfo=None)
    return result


def _json_serializer(obj: object) -> str | int | float | bool | None:
    """Serialize non-JSON-native types for json.dumps."""
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value  # type: ignore[no-any-return]
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _resolve_base_type(type_hint: Any) -> type | None:
    """Extract the concrete type from a possibly-Optional type hint."""
    import types
    from typing import get_args, get_origin

    origin = get_origin(type_hint)
    if origin is not None:
        if origin is types.UnionType:
            args = get_args(type_hint)
            non_none = [a for a in args if a is not type(None)]
            if non_none:
                inner = non_none[0]
                inner_origin = get_origin(inner)
                return inner_origin if inner_origin is not None else inner  # type: ignore[no-any-return]
            return None
        return origin  # type: ignore[no-any-return]
    return type_hint  # type: ignore[no-any-return]


class MemoriesCollection(BaseCollection[MemoryEntity]):
    """Collection for memory entities with three-tier caching."""

    primary_key_column: str = "memory_id"

    def __init__(
        self,
        registry: CollectionRegistry,
        config: CoreConfig,
        postgres_pool: Any,
        authorizer: MemoryAuthorizerDependencies,
        nats_client: Any = None,
        write_buffer: WriteBuffer | None = None,
    ) -> None:
        """initialize memory collection with required rbac authorizer.

        :param registry: shared collection registry
        :ptype registry: CollectionRegistry
        :param config: core configuration governing flush behaviour
        :ptype config: CoreConfig
        :param postgres_pool: asyncpg-shape pool pinned to agent schema
        :ptype postgres_pool: Any
        :param authorizer: rbac authorizer dependency bundle; required.
            every ``caller_user_id``-bearing read / write goes through
            :func:`authorize_memory_access` against this bundle. tests
            inject a permissive fixture from ``conftest.py``; production
            wiring builds the real bundle from hub-side loaders +
            namespace resolver + first-write assignment ensurer
        :ptype authorizer: MemoryAuthorizerDependencies
        :param nats_client: L2 NATS KV client for cache promotion
        :ptype nats_client: Any
        :param write_buffer: optional shared write buffer
        :ptype write_buffer: WriteBuffer | None
        """
        self._postgres_pool = postgres_pool
        self._authorizer = authorizer
        super().__init__(registry, config, nats_client, write_buffer)

    @property
    def table_name(self) -> str:
        """Return the database table name for this collection."""
        return "memories"

    @property
    def entity_class(self) -> type[MemoryEntity]:
        """Return the entity class for this collection."""
        return MemoryEntity

    async def _fetch_from_postgres(self, entity_id: Any) -> dict[str, Any] | None:
        row = await self._postgres_pool.fetchrow("SELECT * FROM memories WHERE memory_id = $1", entity_id)
        if row is None:
            return None
        return dict(row)

    async def _save_to_postgres(self, data: dict[str, Any], original_timestamp: datetime | None = None) -> int:
        if original_timestamp is None:
            result = await self._postgres_pool.execute(
                """
                INSERT INTO memories (
                    memory_id, agent_id, customer_id, user_id,
                    conversation_id, message_id_source,
                    type_memory, content, embedding, is_deleted,
                    media_id, date_created, date_deleted, date_updated
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::vector, $10, $11, $12, $13, $14)
                ON CONFLICT (memory_id) DO UPDATE SET
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    is_deleted = EXCLUDED.is_deleted,
                    date_deleted = EXCLUDED.date_deleted,
                    date_updated = EXCLUDED.date_updated
                """,
                data["memory_id"],
                data.get("agent_id"),
                data.get("customer_id"),
                data["user_id"],
                data["conversation_id"],
                data["message_id_source"],
                data["type_memory"],
                data["content"],
                _encode_embedding(data["embedding"]),
                data["is_deleted"],
                data.get("media_id"),
                _to_naive_utc(data["date_created"]),
                _to_naive_utc(data.get("date_deleted")),
                _to_naive_utc(data.get("date_updated")),
            )
        else:
            result = await self._postgres_pool.execute(
                """
                UPDATE memories SET
                    content = $2,
                    embedding = $3::vector,
                    is_deleted = $4,
                    date_deleted = $5,
                    date_updated = $6
                WHERE memory_id = $1 AND date_updated = $7
                """,
                data["memory_id"],
                data["content"],
                _encode_embedding(data["embedding"]),
                data["is_deleted"],
                _to_naive_utc(data.get("date_deleted")),
                _to_naive_utc(data.get("date_updated")),
                _to_naive_utc(original_timestamp),
            )
        return int(result.split()[-1])

    async def _delete_from_postgres(self, entity_id: Any) -> None:
        await self._postgres_pool.execute("DELETE FROM memories WHERE memory_id = $1", entity_id)

    def _serialize(self, data: dict[str, Any]) -> bytes:
        return json.dumps(data, default=_json_serializer).encode("utf-8")

    def _deserialize(self, data: bytes) -> dict[str, Any]:
        raw: dict[str, Any] = json.loads(data.decode("utf-8"))
        result: dict[str, Any] = {}
        for key, value in raw.items():
            if value is None:
                result[key] = None
                continue
            base_type = _resolve_base_type(_FIELD_TYPES.get(key))
            if base_type is UUID and isinstance(value, str):
                result[key] = UUID(value)
            elif base_type is datetime and isinstance(value, str):
                result[key] = datetime.fromisoformat(value)
            elif base_type is bool and isinstance(value, (bool, int)):
                result[key] = bool(value)
            elif base_type is list and isinstance(value, list):
                result[key] = value
            else:
                result[key] = value
        return result

    async def find_by_user(
        self,
        user_id: UUID,
        include_deleted: bool = False,
        *,
        agent_id: UUID | None = None,
        customer_id: UUID | None = None,
        caller_user_id: UUID | None = None,
        caller_agent_id: UUID | None = None,
    ) -> list[MemoryEntity]:
        """fetch memories for user from L3, enforcing rbac on user reads.

        when ``caller_user_id`` is provided the rbac evaluator decides
        ``memory.read`` on the ``(agent_id, customer_id)`` memory
        namespace before the SQL runs. the owner short-circuit fires
        when ``caller_agent_id == agent_id``; a mismatched pair
        surfaces :class:`MemoryAccessDenied` from
        :func:`authorize_memory_access`. the row-level
        ``user_id = $1`` filter is kept as a belt-and-suspenders cut
        against grants that resolve to broad type_customer scope but
        should still respect the per-row owner column.

        passing ``caller_user_id`` without ``agent_id`` + ``customer_id``
        is a programming error — the evaluator cannot resolve the
        memory namespace without both. internal agent-only callers
        that want row-filter-only access pass ``caller_user_id=None``.

        :param user_id: user whose memories to fetch (row filter)
        :ptype user_id: UUID
        :param include_deleted: whether to include soft-deleted memories
        :ptype include_deleted: bool
        :param agent_id: owning agent UUID (memory namespace owner);
            required when ``caller_user_id`` is set
        :ptype agent_id: UUID | None
        :param customer_id: owning customer UUID; required when
            ``caller_user_id`` is set
        :ptype customer_id: UUID | None
        :param caller_user_id: invoking user UUID for evaluator
        :ptype caller_user_id: UUID | None
        :param caller_agent_id: invoking agent UUID for evaluator
        :ptype caller_agent_id: UUID | None
        :return: list of memory entities belonging to user
        :rtype: list[MemoryEntity]
        :raises MemoryAccessDenied: when rbac enforcement denies
        :raises ValueError: when ``caller_user_id`` is set without
            both ``agent_id`` and ``customer_id``
        """
        if caller_user_id is not None:
            if agent_id is None or customer_id is None:
                raise ValueError(
                    "find_by_user with caller_user_id requires agent_id + customer_id",
                )
            await authorize_memory_access(
                action=ACTION_MEMORY_READ,
                agent_id=agent_id,
                customer_id=customer_id,
                caller_user_id=caller_user_id,
                caller_agent_id=caller_agent_id,
                deps=self._authorizer,
            )

        if include_deleted:
            rows = await self._postgres_pool.fetch(
                "SELECT * FROM memories WHERE user_id = $1 ORDER BY date_created DESC",
                user_id,
            )
        else:
            rows = await self._postgres_pool.fetch(
                "SELECT * FROM memories WHERE user_id = $1 AND is_deleted = false ORDER BY date_created DESC",
                user_id,
            )
        entities: list[MemoryEntity] = []
        for row in rows:
            data = dict(row)
            entity = self.entity_class(data, is_new=False, collection=self)
            entity.original_date_updated = data.get("date_updated")
            entity_id = data["memory_id"]
            # Promote to L2
            await self._save_to_l2(entity_id, data)
            entities.append(entity)
        return entities

    async def find_by_scope(
        self,
        agent_id: UUID,
        customer_id: UUID | None = None,
        user_id: UUID | None = None,
        include_deleted: bool = False,
        *,
        caller_user_id: UUID | None = None,
        caller_agent_id: UUID | None = None,
    ) -> list[MemoryEntity]:
        """fetch memories scoped by agent, enforcing rbac on user reads.

        when ``caller_user_id`` is provided the rbac evaluator
        decides ``memory.read`` on the ``(agent_id, customer_id)``
        memory namespace before the SQL runs. ``customer_id`` is
        required for evaluator invocation — single-argument
        ``find_by_scope(agent_id)`` is an agent-internal row-scan
        path (no user dimension) and runs without evaluation.

        :param agent_id: agent ID scope (required)
        :ptype agent_id: UUID
        :param customer_id: optional customer ID to further narrow
            scope; required when ``caller_user_id`` is set
        :ptype customer_id: UUID | None
        :param user_id: optional user ID to further narrow scope
        :ptype user_id: UUID | None
        :param include_deleted: whether to include soft-deleted memories
        :ptype include_deleted: bool
        :param caller_user_id: invoking user UUID for evaluator
        :ptype caller_user_id: UUID | None
        :param caller_agent_id: invoking agent UUID for evaluator
        :ptype caller_agent_id: UUID | None
        :return: list of memory entities matching scope
        :rtype: list[MemoryEntity]
        :raises MemoryAccessDenied: when rbac enforcement denies
        :raises ValueError: when ``caller_user_id`` is set without
            ``customer_id``
        """
        if caller_user_id is not None:
            if customer_id is None:
                raise ValueError(
                    "find_by_scope with caller_user_id requires customer_id",
                )
            await authorize_memory_access(
                action=ACTION_MEMORY_READ,
                agent_id=agent_id,
                customer_id=customer_id,
                caller_user_id=caller_user_id,
                caller_agent_id=caller_agent_id,
                deps=self._authorizer,
            )
        conditions = ["agent_id = $1"]
        params: list[object] = [agent_id]
        param_idx = 2

        if customer_id is not None:
            conditions.append(f"customer_id = ${param_idx}")
            params.append(customer_id)
            param_idx += 1

        if user_id is not None:
            conditions.append(f"user_id = ${param_idx}")
            params.append(user_id)
            param_idx += 1

        if not include_deleted:
            conditions.append("is_deleted = false")

        where_clause = " AND ".join(conditions)
        query = f"SELECT * FROM memories WHERE {where_clause} ORDER BY date_created DESC"

        rows = await self._postgres_pool.fetch(query, *params)

        entities: list[MemoryEntity] = []
        for row in rows:
            data = dict(row)
            entity = self.entity_class(data, is_new=False, collection=self)
            entity.original_date_updated = data.get("date_updated")
            entity_id = data["memory_id"]
            await self._save_to_l2(entity_id, data)
            entities.append(entity)
        return entities

    async def save_memory(
        self,
        entity: MemoryEntity,
        *,
        agent_id: UUID,
        customer_id: UUID,
        caller_user_id: UUID | None,
        caller_agent_id: UUID | None,
    ) -> None:
        """persist a memory after evaluating ``memory.write`` for the caller.

        user-initiated writes land here. the evaluator decides
        ``memory.write`` for the caller against the ``(agent_id,
        customer_id)`` memory namespace; owner short-circuit
        applies when ``caller_agent_id == agent_id`` (agent-
        internal writes). on a successful user write (evaluator
        passed AND caller_user_id is set AND the agent-owner
        short-circuit did NOT fire), the per-user ``MemoryOwner``
        assignment is ensured via
        :func:`ensure_memory_owner_assignment` so subsequent reads
        from the same user hit a cached grant.

        the agent-internal extractor path bypasses this method and
        calls :meth:`save_entity` directly because it runs under
        the agent-owner short-circuit; the explicit call site gives
        operators a clean audit distinction between agent-emitted
        and user-explicit memory rows.

        rbac enforcement is unconditional: the collection is
        constructed with a required :class:`MemoryAuthorizerDependencies`
        bundle, and every call evaluates :func:`authorize_memory_access`
        before the write.

        :param entity: memory entity to persist
        :ptype entity: MemoryEntity
        :param agent_id: owning agent UUID
        :ptype agent_id: UUID
        :param customer_id: owning customer UUID
        :ptype customer_id: UUID
        :param caller_user_id: invoking user UUID (``None`` for
            agent-internal writes — owner short-circuit path)
        :ptype caller_user_id: UUID | None
        :param caller_agent_id: invoking agent UUID
        :ptype caller_agent_id: UUID | None
        :return: nothing
        :rtype: None
        :raises MemoryAccessDenied: on evaluator deny
        """
        ns_entity = await authorize_memory_access(
            action=ACTION_MEMORY_WRITE,
            agent_id=agent_id,
            customer_id=customer_id,
            caller_user_id=caller_user_id,
            caller_agent_id=caller_agent_id,
            deps=self._authorizer,
        )

        await self.save_entity(entity)

        # auto-assignment on first user-write: bind the user's
        # per-user memory-owner group to the MemoryOwner role scoped
        # to this memory namespace. idempotent on replay —
        # :func:`ensure_memory_owner_assignment` drives every row
        # through deterministic :func:`uuid5` ids + Collection
        # ON CONFLICT semantics. skip when the caller is the owning
        # agent (no user identity to bind the group to).
        is_owner_shortcut = (
            caller_agent_id is not None and caller_agent_id == agent_id
        )
        if caller_user_id is not None and not is_owner_shortcut:
            await ensure_memory_owner_assignment(
                user_id=caller_user_id,
                namespace=ns_entity,
                deps=self._authorizer,
            )
        return None

    async def soft_delete(self, entity: MemoryEntity) -> None:
        """Soft-delete a memory by setting is_deleted and date_deleted."""
        entity.is_deleted = True
        entity.date_deleted = datetime.now(UTC)
        await self.save_entity(entity)
