"""Memories collection -- three-tier CRUD for memory entities.

Also wires the sibling collections for tables the memory package owns
but previously lacked Collection coverage: :class:`MediaCollection`,
:class:`MediaContentCollection`, :class:`MemoryChunkCollection` (adopted
under namespace-task-01 phase 8.5b). Each Collection resolves its L3
pool via the registry (``self.l3_pool``) — the bespoke
``_postgres_pool`` field is retired in favour of the registry pattern
:class:`ConversationCollection` already uses.

Complex hybrid-search queries (vector + FTS + MMR) live as methods on
these Collections with documented ``# cache-bypass:`` comments — the
Collection stays the single entry point for memory-table SQL even when
the query shape is not primary-key addressable and therefore cannot
benefit from L1 row caching.
"""

from __future__ import annotations

import asyncio
import json
import math
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
from threetears.agent.memory.entities import (
    MediaContentEntity,
    MediaEntity,
    MemoryChunkEntity,
    MemoryEntity,
    MemoryRefEntity,
)

__all__ = [
    "MediaCollection",
    "MediaContentCollection",
    "MemoriesCollection",
    "MemoryChunkCollection",
    "MemoryRefsCollection",
]

log = get_logger(__name__)

# Field type mapping for JSON serialization/deserialization
_MEMORY_FIELD_TYPES: dict[str, Any] = {
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


_MEDIA_FIELD_TYPES: dict[str, Any] = {
    "media_id": UUID,
    "agent_id": UUID | None,
    "customer_id": UUID | None,
    "user_id": UUID,
    "media_category": str,
    "metadata_json": dict,
    "date_created": datetime,
    "date_updated": datetime,
}


_MEDIA_CONTENT_FIELD_TYPES: dict[str, Any] = {
    "content_id": UUID,
    "media_id": UUID,
    "agent_id": UUID | None,
    "customer_id": UUID | None,
    "user_id": UUID,
    "content_type": str,
    "content": str,
    "summary": str | None,
    "embedding": list[float],
    "date_created": datetime,
}


_MEMORY_CHUNK_FIELD_TYPES: dict[str, Any] = {
    "chunk_id": UUID,
    "media_id": UUID | None,
    "agent_id": UUID | None,
    "customer_id": UUID | None,
    "user_id": UUID,
    "content": str,
    "summary": str | None,
    "heading_context": str | None,
    "page_number": int,
    "embedding": list[float],
    "date_created": datetime,
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
    """Serialize non-JSON-native types for json.dumps.

    :param obj: value that ``json.dumps`` could not encode natively
    :ptype obj: object
    :return: JSON-compatible representation
    :rtype: str | int | float | bool | None
    :raises TypeError: when ``obj`` cannot be serialized
    """
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value  # type: ignore[no-any-return]
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _resolve_base_type(type_hint: Any) -> type | None:
    """Extract the concrete type from a possibly-Optional type hint.

    :param type_hint: python type hint, may be wrapped in ``| None``
    :ptype type_hint: Any
    :return: underlying concrete type, or ``None``
    :rtype: type | None
    """
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


def _deserialize_with_types(
    raw: dict[str, Any], field_types: dict[str, Any],
) -> dict[str, Any]:
    """Map JSON-decoded dict back to native python types per ``field_types``.

    :param raw: decoded dict from JSON payload
    :ptype raw: dict[str, Any]
    :param field_types: map of column name to declared type hint
    :ptype field_types: dict[str, Any]
    :return: dict with values coerced to their typed form
    :rtype: dict[str, Any]
    """
    result: dict[str, Any] = {}
    for key, value in raw.items():
        if value is None:
            result[key] = None
            continue
        base_type = _resolve_base_type(field_types.get(key))
        if base_type is UUID and isinstance(value, str):
            result[key] = UUID(value)
        elif base_type is datetime and isinstance(value, str):
            result[key] = datetime.fromisoformat(value)
        elif base_type is bool and isinstance(value, (bool, int)):
            result[key] = bool(value)
        elif base_type is int and isinstance(value, (int, float)):
            result[key] = int(value)
        elif base_type is list and isinstance(value, list):
            result[key] = value
        else:
            result[key] = value
    return result


def _build_fts_text(user_text: str, min_len: int = 3, max_len: int = 500) -> str | None:
    """Prepare free-text query for ``websearch_to_tsquery``.

    :param user_text: raw user text
    :ptype user_text: str
    :param min_len: minimum length after stripping
    :ptype min_len: int
    :param max_len: cap on returned length
    :ptype max_len: int
    :return: cleaned query string, or ``None`` when too short
    :rtype: str | None
    """
    text = user_text.strip()
    if len(text) < min_len:
        return None
    return text[:max_len]


def _build_user_scope_clause(
    user_id: UUID,
    *,
    agent_id: UUID | None = None,
    customer_id: UUID | None = None,
    start_param: int = 2,
    table_prefix: str = "",
) -> tuple[str, list[UUID], int]:
    """Build WHERE fragments scoping by user/agent/customer.

    shared across the four Collections' hybrid-search methods: the
    same signature :mod:`retrieval.py` + :mod:`tools.py` already use
    so callers that migrate onto the Collection API keep their
    parameterisation identical.

    :param user_id: user ID (always included)
    :ptype user_id: UUID
    :param agent_id: optional agent ID scope
    :ptype agent_id: UUID | None
    :param customer_id: optional customer ID scope
    :ptype customer_id: UUID | None
    :param start_param: starting positional parameter index
    :ptype start_param: int
    :param table_prefix: optional table alias prefix
    :ptype table_prefix: str
    :return: tuple of (conditions string, param values, last param
        index used)
    :rtype: tuple[str, list[UUID], int]
    """
    prefix = f"{table_prefix}." if table_prefix else ""
    conditions: list[str] = []
    params: list[UUID] = []
    idx = start_param

    if agent_id is not None:
        conditions.append(f"{prefix}agent_id = ${idx}")
        params.append(agent_id)
        idx += 1

    if customer_id is not None:
        conditions.append(f"{prefix}customer_id = ${idx}")
        params.append(customer_id)
        idx += 1

    conditions.append(f"{prefix}user_id = ${idx}")
    params.append(user_id)

    return " AND ".join(conditions), params, idx


def _normalize_scores(candidates: list[dict[str, Any]], key: str) -> None:
    """Min-max normalize scores in-place to [0, 1].

    :param candidates: candidate rows (mutated)
    :ptype candidates: list[dict[str, Any]]
    :param key: score column name
    :ptype key: str
    :return: nothing
    :rtype: None
    """
    scores = [c.get(key, 0.0) for c in candidates]
    lo, hi = min(scores), max(scores)
    span = hi - lo
    for c in candidates:
        raw = c.get(key, 0.0)
        if span > 0:
            c[key] = (raw - lo) / span
        elif hi > 0:
            c[key] = 1.0


def _recency_weight(created: datetime, half_life_hours: float) -> float:
    """Exponential recency decay (1.0 = just created, ~0.37 at half-life).

    :param created: creation timestamp (naive UTC accepted, treated as UTC)
    :ptype created: datetime
    :param half_life_hours: half-life in hours
    :ptype half_life_hours: float
    :return: decay factor in (0, 1]
    :rtype: float
    """
    now = datetime.now(UTC)
    if created.tzinfo is None:
        created = created.replace(tzinfo=UTC)
    hours_ago = max((now - created).total_seconds() / 3600, 0.0)
    return math.exp(-hours_ago / half_life_hours)


class MemoriesCollection(BaseCollection[MemoryEntity]):
    """Collection for memory entities with three-tier caching.

    CRUD goes through :meth:`get` / :meth:`save_entity` / :meth:`delete`
    so L1 / L2 / L3 tiers stay coherent. Hybrid-search methods
    (:meth:`hybrid_search`, :meth:`search_by_ids`,
    :meth:`find_similar_for_dedup`, :meth:`count_by_user`) absorb the
    vector + FTS + MMR queries that used to live raw on the pool; they
    carry ``# cache-bypass:`` justification because the query shape is
    not primary-key-addressable and therefore cannot benefit from the
    L1 row cache — but keeping them on the Collection preserves the
    single-entry-point contract enforcement test walker #3 relies on.
    """

    primary_key_column: str = "memory_id"

    def __init__(
        self,
        registry: CollectionRegistry,
        config: CoreConfig,
        authorizer: MemoryAuthorizerDependencies,
        nats_client: Any = None,
        write_buffer: WriteBuffer | None = None,
    ) -> None:
        """initialize memory collection with required rbac authorizer.

        :param registry: shared collection registry; ``l3_pool`` is
            resolved through :meth:`CollectionRegistry.get_l3_pool`
            (same pattern :class:`ConversationCollection` uses), so
            callers must bind the target agent pool via
            :meth:`CollectionRegistry.configure` or
            :meth:`CollectionRegistry.bind_table` before construction
        :ptype registry: CollectionRegistry
        :param config: core configuration governing flush behaviour
        :ptype config: CoreConfig
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
        self._authorizer = authorizer
        super().__init__(registry, config, nats_client, write_buffer)

    @property
    def table_name(self) -> str:
        """Return the database table name for this collection.

        :return: table name
        :rtype: str
        """
        return "memories"

    @property
    def entity_class(self) -> type[MemoryEntity]:
        """Return the entity class for this collection.

        :return: entity class
        :rtype: type[MemoryEntity]
        """
        return MemoryEntity

    async def fetch_from_postgres(self, entity_id: Any) -> dict[str, Any] | None:
        """Fetch one memory row from L3 by primary key.

        :param entity_id: memory primary-key value
        :ptype entity_id: Any
        :return: row dict or ``None``
        :rtype: dict[str, Any] | None
        """
        if self.l3_pool is None:
            return None
        row = await self.l3_pool.fetchrow(
            "SELECT * FROM memories WHERE memory_id = $1", entity_id,
        )
        result: dict[str, Any] | None = dict(row) if row is not None else None
        return result

    async def save_to_postgres(
        self, data: dict[str, Any], original_timestamp: datetime | None = None,
    ) -> int:
        """Upsert one memory row into L3.

        :param data: row data to persist
        :ptype data: dict[str, Any]
        :param original_timestamp: optimistic-concurrency guard
        :ptype original_timestamp: datetime | None
        :return: rows affected
        :rtype: int
        """
        if self.l3_pool is None:
            return 0
        if original_timestamp is None:
            result = await self.l3_pool.execute(
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
            result = await self.l3_pool.execute(
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

    async def delete_from_postgres(self, entity_id: Any) -> None:
        """Hard-delete a memory row from L3.

        :param entity_id: memory primary-key value
        :ptype entity_id: Any
        :return: nothing
        :rtype: None
        """
        if self.l3_pool is None:
            return
        await self.l3_pool.execute(
            "DELETE FROM memories WHERE memory_id = $1", entity_id,
        )

    def serialize(self, data: dict[str, Any]) -> bytes:
        """Serialize a row dict for L2 storage.

        :param data: row data
        :ptype data: dict[str, Any]
        :return: JSON-encoded bytes
        :rtype: bytes
        """
        return json.dumps(data, default=_json_serializer).encode("utf-8")

    def deserialize(self, data: bytes) -> dict[str, Any]:
        """Deserialize L2 payload back into a row dict.

        :param data: JSON-encoded bytes
        :ptype data: bytes
        :return: row data
        :rtype: dict[str, Any]
        """
        raw: dict[str, Any] = json.loads(data.decode("utf-8"))
        return _deserialize_with_types(raw, _MEMORY_FIELD_TYPES)

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

        if self.l3_pool is None:
            return []
        if include_deleted:
            # cache-bypass: multi-row scan by user_id is not primary-key
            # addressable; L1 row cache would not help. method on
            # Collection preserves single entry point.
            rows = await self.l3_pool.fetch(
                "SELECT * FROM memories WHERE user_id = $1 ORDER BY date_created DESC",
                user_id,
            )
        else:
            # cache-bypass: same as above — multi-row scan path.
            rows = await self.l3_pool.fetch(
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
        if self.l3_pool is None:
            return []
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

        # cache-bypass: multi-row scan by agent scope is not primary-
        # key addressable; L1 row cache would not help. method on
        # Collection preserves single entry point + rbac gating.
        rows = await self.l3_pool.fetch(query, *params)

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
        """Soft-delete a memory by setting is_deleted and date_deleted.

        :param entity: entity to soft-delete
        :ptype entity: MemoryEntity
        :return: nothing
        :rtype: None
        """
        entity.is_deleted = True
        entity.date_deleted = datetime.now(UTC)
        await self.save_entity(entity)

    async def count_by_user(self, user_id: UUID) -> bool:
        """check whether any memory row exists for the given user.

        returns a boolean rather than an exact count — every caller
        today uses the existence flag (tools.py first-write ensure
        gate). keeps the query cheap (``SELECT EXISTS(...)``) and
        side-steps full row-count pagination concerns.

        :param user_id: user UUID to probe
        :ptype user_id: UUID
        :return: ``True`` iff user has at least one memory row
        :rtype: bool
        """
        if self.l3_pool is None:
            return False
        # cache-bypass: existence probe is not primary-key-addressable;
        # L1 row cache would not help. method on Collection preserves
        # single entry point.
        value = await self.l3_pool.fetchval(
            "SELECT EXISTS(SELECT 1 FROM memories WHERE user_id = $1)",
            user_id,
        )
        return bool(value)

    async def find_similar_for_dedup(
        self,
        *,
        user_id: UUID,
        embedding: list[float],
        top_k: int,
        threshold: float,
    ) -> list[dict[str, Any]]:
        """vector search for near-duplicate memories by embedding.

        used by :class:`MemoryExtractor._get_similar_memories` +
        :class:`add_memory` tool dedup guard. returns memories whose
        cosine similarity to ``embedding`` exceeds ``threshold``,
        capped at ``top_k``.

        :param user_id: owning user UUID (row filter)
        :ptype user_id: UUID
        :param embedding: query embedding vector
        :ptype embedding: list[float]
        :param top_k: maximum candidates to consider
        :ptype top_k: int
        :param threshold: minimum cosine similarity to surface
        :ptype threshold: float
        :return: list of ``{memory_id, content, type_memory, similarity}``
        :rtype: list[dict[str, Any]]
        """
        if self.l3_pool is None:
            return []
        embedding_str = json.dumps(embedding)
        # cache-bypass: vector-distance search is not primary-key-
        # addressable; L1 row cache cannot serve. keeping the query
        # on the Collection preserves single entry point for uniformity
        # + audit hooks + future observability.
        rows = await self.l3_pool.fetch(
            """
            SELECT memory_id, content, type_memory,
                   1 - (embedding <=> $1::vector) AS similarity
            FROM memories
            WHERE user_id = $2 AND is_deleted = false
            ORDER BY embedding <=> $1::vector
            LIMIT $3
            """,
            embedding_str,
            user_id,
            top_k,
        )
        return [
            {
                "memory_id": row["memory_id"],
                "content": row["content"],
                "type_memory": row["type_memory"],
                "similarity": float(row["similarity"]),
            }
            for row in rows
            if float(row["similarity"]) >= threshold
        ]

    async def hybrid_search(
        self,
        *,
        user_id: UUID,
        embedding: list[float],
        user_text: str,
        top_k: int,
        candidate_limit: int,
        similarity_threshold: float,
        recency_half_life_hours: float,
        signal_weights: dict[str, float],
        agent_id: UUID | None = None,
        customer_id: UUID | None = None,
        fts_min_len: int = 3,
        fts_max_len: int = 500,
    ) -> list[dict[str, Any]]:
        """parallel vector + FTS hybrid search across the memories table.

        absorbs the SQL that used to live in
        :meth:`MemoryRetriever._query_memories`. three-signal
        ranking (semantic / keyword / recency) + cosine-distance
        ordering; candidates are merged across the two parallel
        queries on ``memory_id``, recency-decayed, score-combined, and
        threshold-filtered.

        :param user_id: owning user UUID (row filter)
        :ptype user_id: UUID
        :param embedding: query embedding vector
        :ptype embedding: list[float]
        :param user_text: raw query text for FTS
        :ptype user_text: str
        :param top_k: number of candidates to return after ranking
        :ptype top_k: int
        :param candidate_limit: per-query candidate pool size
        :ptype candidate_limit: int
        :param similarity_threshold: floor on hybrid score
        :ptype similarity_threshold: float
        :param recency_half_life_hours: exponential decay half-life
        :ptype recency_half_life_hours: float
        :param signal_weights: mapping ``{"semantic", "keyword",
            "recency"}`` to weights
        :ptype signal_weights: dict[str, float]
        :param agent_id: optional agent scope
        :ptype agent_id: UUID | None
        :param customer_id: optional customer scope
        :ptype customer_id: UUID | None
        :param fts_min_len: minimum query length for FTS activation
        :ptype fts_min_len: int
        :param fts_max_len: truncation length for FTS queries
        :ptype fts_max_len: int
        :return: ranked candidate list, top_k entries
        :rtype: list[dict[str, Any]]
        """
        if self.l3_pool is None:
            return []
        embedding_str = json.dumps(embedding)

        scope_conditions, scope_params, param_offset = _build_user_scope_clause(
            user_id,
            agent_id=agent_id,
            customer_id=customer_id,
            start_param=2,
        )
        vec_where = f"WHERE {scope_conditions} AND is_deleted = false"
        limit_param = f"${param_offset + 1}"

        # cache-bypass: vector-distance search is not primary-key-
        # addressable; see :meth:`find_similar_for_dedup` for the same
        # justification. method on Collection preserves single entry
        # point for rbac + audit.
        vec_coro = self.l3_pool.fetch(
            f"""
            SELECT memory_id, content, summary, type_memory, date_created,
                   embedding,
                   1 - (embedding <=> $1::vector) AS similarity
            FROM memories
            {vec_where}
            ORDER BY embedding <=> $1::vector
            LIMIT {limit_param}
            """,
            embedding_str,
            *scope_params,
            candidate_limit,
        )

        fts_text = _build_fts_text(user_text, fts_min_len, fts_max_len)
        if fts_text:
            fts_scope_conditions, fts_scope_params, fts_param_offset = (
                _build_user_scope_clause(
                    user_id,
                    agent_id=agent_id,
                    customer_id=customer_id,
                    start_param=2,
                )
            )
            fts_where = f"WHERE {fts_scope_conditions} AND is_deleted = false"
            fts_limit_param = f"${fts_param_offset + 1}"
            # cache-bypass: FTS rank query is not primary-key-
            # addressable. See :meth:`hybrid_search` docstring.
            fts_coro = self.l3_pool.fetch(
                f"""
                SELECT memory_id, content, summary, type_memory, date_created,
                       embedding,
                       ts_rank_cd(search_vector, websearch_to_tsquery('english', $1)) AS fts_rank
                FROM memories
                {fts_where}
                  AND search_vector @@ websearch_to_tsquery('english', $1)
                ORDER BY fts_rank DESC
                LIMIT {fts_limit_param}
                """,
                fts_text,
                *fts_scope_params,
                candidate_limit,
            )
            vec_rows, fts_rows = await asyncio.gather(vec_coro, fts_coro)
        else:
            vec_rows = await vec_coro
            fts_rows = []

        merged: dict[Any, dict[str, Any]] = {}
        for row in vec_rows:
            mid = row["memory_id"]
            emb = row["embedding"]
            if isinstance(emb, str):
                emb = json.loads(emb)
            merged[mid] = {
                "memory_id": mid,
                "content": row["content"],
                "summary": row["summary"],
                "type_memory": row["type_memory"],
                "date_created": row["date_created"],
                "similarity": float(row["similarity"]),
                "fts_rank": 0.0,
                "embedding": emb,
            }
        for row in fts_rows:
            mid = row["memory_id"]
            if mid in merged:
                merged[mid]["fts_rank"] = float(row["fts_rank"])
            else:
                emb = row["embedding"]
                if isinstance(emb, str):
                    emb = json.loads(emb)
                merged[mid] = {
                    "memory_id": mid,
                    "content": row["content"],
                    "summary": row["summary"],
                    "type_memory": row["type_memory"],
                    "date_created": row["date_created"],
                    "similarity": 0.0,
                    "fts_rank": float(row["fts_rank"]),
                    "embedding": emb,
                }

        candidates = list(merged.values())
        if not candidates:
            return []

        _normalize_scores(candidates, "fts_rank")

        for c in candidates:
            recency = _recency_weight(c["date_created"], recency_half_life_hours)
            c["recency"] = round(recency, 4)
            c["hybrid_score"] = round(
                signal_weights["semantic"] * c["similarity"]
                + signal_weights["keyword"] * c["fts_rank"]
                + signal_weights["recency"] * recency,
                4,
            )

        filtered = [c for c in candidates if c["hybrid_score"] > similarity_threshold]
        filtered.sort(key=lambda m: m["hybrid_score"], reverse=True)
        return filtered[:top_k]

    async def search_by_ids(
        self,
        memory_ids: list[UUID],
        user_id: UUID,
    ) -> list[dict[str, Any]]:
        """batch lookup of memories by primary key, scoped to a user.

        absorbs the ``SELECT ... FROM memories WHERE memory_id = ANY(...)``
        leg of :func:`_search_by_ids` in :mod:`tools.py`. scoped by
        ``user_id`` so a leaked memory_id from another user does not
        surface content.

        :param memory_ids: primary-key values to fetch
        :ptype memory_ids: list[UUID]
        :param user_id: owning user UUID (row filter)
        :ptype user_id: UUID
        :return: list of row dicts (content + metadata)
        :rtype: list[dict[str, Any]]
        """
        if self.l3_pool is None or not memory_ids:
            return []
        # cache-bypass: batch IN(...) query is primary-key addressable
        # in principle but the existing L1 backend offers only per-row
        # ``select_by_id`` / ``select_batch`` and does not mix with
        # the row-level ``user_id`` filter we enforce here. keeping
        # the batch on the Collection preserves the single entry point
        # + rbac guard; the walker tags this call as inside the
        # Collection class so enforcement test #3 stays clean.
        rows = await self.l3_pool.fetch(
            """
            SELECT memory_id, type_memory, content, date_created
            FROM memories
            WHERE memory_id = ANY($1::uuid[])
              AND user_id = $2
              AND is_deleted = false
            """,
            memory_ids,
            user_id,
        )
        return [dict(row) for row in rows]

    async def search_by_semantic(
        self,
        *,
        user_id: UUID,
        embedding: list[float],
        max_results: int,
        similarity_threshold: float,
        type_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """semantic (vector-only) search for the add/search memory tools.

        absorbs the ``_search_memories`` path from :mod:`tools.py`
        (the LangChain ``memory_search`` tool's vector leg). returns
        rows whose cosine similarity exceeds ``similarity_threshold``,
        capped at ``max_results``. ``type_filter`` is validated by the
        caller; this method trusts the value and applies it as an
        equality predicate.

        :param user_id: owning user UUID (row filter)
        :ptype user_id: UUID
        :param embedding: query embedding vector
        :ptype embedding: list[float]
        :param max_results: cap on returned rows
        :ptype max_results: int
        :param similarity_threshold: similarity floor
        :ptype similarity_threshold: float
        :param type_filter: optional ``type_memory`` equality filter
        :ptype type_filter: str | None
        :return: list of row dicts with ``similarity`` field
        :rtype: list[dict[str, Any]]
        """
        if self.l3_pool is None:
            return []
        embedding_str = json.dumps(embedding)
        params: list[Any] = [embedding_str, user_id]
        conditions = ["user_id = $2", "is_deleted = false"]
        param_idx = 3
        if type_filter:
            conditions.append(f"type_memory = ${param_idx}")
            params.append(type_filter)
            param_idx += 1
        where_clause = " AND ".join(conditions)
        query_sql = f"""
            SELECT memory_id, type_memory, content, date_created,
                   1 - (embedding <=> $1::vector) AS similarity
            FROM memories
            WHERE {where_clause}
            ORDER BY embedding <=> $1::vector
            LIMIT ${param_idx}
        """
        params.append(max_results)
        # cache-bypass: vector-distance ordered lookup; see
        # :meth:`hybrid_search` docstring.
        rows = await self.l3_pool.fetch(query_sql, *params)
        return [
            {
                "memory_id": str(row["memory_id"]),
                "type": row["type_memory"],
                "content": row["content"],
                "date_created": row["date_created"],
                "similarity": float(row["similarity"]),
            }
            for row in rows
            if float(row["similarity"]) > similarity_threshold
        ]

    async def search_by_fts(
        self,
        *,
        user_id: UUID,
        fts_text: str,
        max_results: int,
        type_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """FTS keyword search for the add/search memory tools.

        complements :meth:`search_by_semantic` — run both in parallel
        and merge on ``memory_id``.

        :param user_id: owning user UUID (row filter)
        :ptype user_id: UUID
        :param fts_text: FTS query text
        :ptype fts_text: str
        :param max_results: cap on returned rows
        :ptype max_results: int
        :param type_filter: optional ``type_memory`` equality filter
        :ptype type_filter: str | None
        :return: list of row dicts
        :rtype: list[dict[str, Any]]
        """
        if self.l3_pool is None:
            return []
        conditions = [
            "user_id = $2",
            "is_deleted = false",
            "search_vector @@ websearch_to_tsquery('english', $1)",
        ]
        params: list[Any] = [fts_text, user_id]
        idx = 3
        if type_filter:
            conditions.append(f"type_memory = ${idx}")
            params.append(type_filter)
            idx += 1
        where = " AND ".join(conditions)
        query_sql = f"""
            SELECT memory_id, type_memory, content, date_created,
                   ts_rank_cd(search_vector, websearch_to_tsquery('english', $1)) AS fts_rank
            FROM memories
            WHERE {where}
            ORDER BY fts_rank DESC
            LIMIT {max_results}
        """
        # cache-bypass: FTS rank query is not primary-key-addressable;
        # see :meth:`hybrid_search` docstring.
        rows = await self.l3_pool.fetch(query_sql, *params)
        return [
            {
                "memory_id": str(row["memory_id"]),
                "type": row["type_memory"],
                "content": row["content"],
                "date_created": row["date_created"],
            }
            for row in rows
        ]

    async def fetch_content_for_recall(
        self,
        *,
        memory_id: UUID,
        user_id: UUID,
    ) -> str | None:
        """fetch just the ``content`` field for the recall_memory tool.

        scoped by ``user_id`` so a leaked ID does not cross users.

        :param memory_id: memory primary-key value
        :ptype memory_id: UUID
        :param user_id: owning user UUID (row filter)
        :ptype user_id: UUID
        :return: content text or ``None`` if not found
        :rtype: str | None
        """
        if self.l3_pool is None:
            return None
        # cache-bypass: the L1 cache would serve this lookup when the
        # row has already been warmed, but the ``user_id`` + ``is_deleted``
        # guard here is a SECURITY control (cross-user leak prevention)
        # that the L1 row cache cannot enforce. the authoritative read
        # must stay at the database.
        row = await self.l3_pool.fetchrow(
            "SELECT content FROM memories WHERE memory_id = $1 AND user_id = $2 AND is_deleted = false",
            memory_id,
            user_id,
        )
        if row is None:
            return None
        result: str = row["content"]
        return result


class MediaCollection(BaseCollection[MediaEntity]):
    """three-tier collection for :class:`MediaEntity` (table ``media``).

    the media parent record carries a category discriminator and a
    JSONB metadata blob; child rows live in
    :class:`MediaContentCollection` and :class:`MemoryChunkCollection`.
    adopted under namespace-task-01 phase 8.5b — the v006 migration
    had no Collection before now.
    """

    primary_key_column: str = "media_id"

    @property
    def table_name(self) -> str:
        """Return the database table name for this collection.

        :return: table name
        :rtype: str
        """
        return "media"

    @property
    def entity_class(self) -> type[MediaEntity]:
        """Return the entity class for this collection.

        :return: entity class
        :rtype: type[MediaEntity]
        """
        return MediaEntity

    async def fetch_from_postgres(self, entity_id: Any) -> dict[str, Any] | None:
        """fetch one media row from L3 by primary key.

        :param entity_id: media primary-key value
        :ptype entity_id: Any
        :return: row dict or ``None``
        :rtype: dict[str, Any] | None
        """
        if self.l3_pool is None:
            return None
        row = await self.l3_pool.fetchrow(
            "SELECT * FROM media WHERE media_id = $1", entity_id,
        )
        result: dict[str, Any] | None = dict(row) if row is not None else None
        return result

    async def save_to_postgres(
        self, data: dict[str, Any], original_timestamp: datetime | None = None,
    ) -> int:
        """upsert one media row into L3.

        :param data: row data to persist
        :ptype data: dict[str, Any]
        :param original_timestamp: ignored for media (no CAS column
            distinct from ``date_updated``)
        :ptype original_timestamp: datetime | None
        :return: rows affected
        :rtype: int
        """
        _ = original_timestamp
        if self.l3_pool is None:
            return 0
        metadata_value = data.get("metadata_json")
        if metadata_value is not None and not isinstance(metadata_value, str):
            metadata_value = json.dumps(metadata_value, default=_json_serializer)
        result = await self.l3_pool.execute(
            """
            INSERT INTO media (
                media_id, agent_id, customer_id, user_id,
                media_category, metadata_json,
                date_created, date_updated
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (media_id) DO UPDATE SET
                agent_id = EXCLUDED.agent_id,
                customer_id = EXCLUDED.customer_id,
                user_id = EXCLUDED.user_id,
                media_category = EXCLUDED.media_category,
                metadata_json = EXCLUDED.metadata_json,
                date_updated = EXCLUDED.date_updated
            """,
            data["media_id"],
            data.get("agent_id"),
            data.get("customer_id"),
            data["user_id"],
            data["media_category"],
            metadata_value,
            _to_naive_utc(data["date_created"]),
            _to_naive_utc(data.get("date_updated") or data["date_created"]),
        )
        return int(result.split()[-1])

    async def delete_from_postgres(self, entity_id: Any) -> None:
        """hard-delete a media row from L3.

        :param entity_id: media primary-key value
        :ptype entity_id: Any
        :return: nothing
        :rtype: None
        """
        if self.l3_pool is None:
            return
        await self.l3_pool.execute(
            "DELETE FROM media WHERE media_id = $1", entity_id,
        )

    def serialize(self, data: dict[str, Any]) -> bytes:
        """serialize a row dict for L2 storage.

        :param data: row data
        :ptype data: dict[str, Any]
        :return: JSON-encoded bytes
        :rtype: bytes
        """
        return json.dumps(data, default=_json_serializer).encode("utf-8")

    def deserialize(self, data: bytes) -> dict[str, Any]:
        """deserialize L2 payload back into a row dict.

        :param data: JSON-encoded bytes
        :ptype data: bytes
        :return: row data
        :rtype: dict[str, Any]
        """
        raw: dict[str, Any] = json.loads(data.decode("utf-8"))
        return _deserialize_with_types(raw, _MEDIA_FIELD_TYPES)


class MediaContentCollection(BaseCollection[MediaContentEntity]):
    """three-tier collection for :class:`MediaContentEntity`.

    carries content rows attached to :class:`MediaEntity` through
    ``media_id``. hybrid-search surface
    (:meth:`hybrid_search`, :meth:`search_by_ids`) lives on this
    collection because vector/FTS queries are the primary way callers
    probe this table; by-ID pull-through is rare.
    """

    primary_key_column: str = "content_id"

    @property
    def table_name(self) -> str:
        """Return the database table name for this collection.

        :return: table name
        :rtype: str
        """
        return "media_content"

    @property
    def entity_class(self) -> type[MediaContentEntity]:
        """Return the entity class for this collection.

        :return: entity class
        :rtype: type[MediaContentEntity]
        """
        return MediaContentEntity

    async def fetch_from_postgres(self, entity_id: Any) -> dict[str, Any] | None:
        """fetch one media_content row from L3 by primary key.

        :param entity_id: content primary-key value
        :ptype entity_id: Any
        :return: row dict or ``None``
        :rtype: dict[str, Any] | None
        """
        if self.l3_pool is None:
            return None
        row = await self.l3_pool.fetchrow(
            "SELECT * FROM media_content WHERE content_id = $1", entity_id,
        )
        result: dict[str, Any] | None = dict(row) if row is not None else None
        return result

    async def save_to_postgres(
        self, data: dict[str, Any], original_timestamp: datetime | None = None,
    ) -> int:
        """upsert one media_content row into L3.

        :param data: row data to persist
        :ptype data: dict[str, Any]
        :param original_timestamp: ignored for media_content (no CAS
            column distinct from ``date_created``)
        :ptype original_timestamp: datetime | None
        :return: rows affected
        :rtype: int
        """
        _ = original_timestamp
        if self.l3_pool is None:
            return 0
        result = await self.l3_pool.execute(
            """
            INSERT INTO media_content (
                content_id, media_id, agent_id, customer_id,
                user_id, content_type, content, summary,
                embedding, date_created
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::vector, $10)
            ON CONFLICT (content_id) DO UPDATE SET
                content_type = EXCLUDED.content_type,
                content = EXCLUDED.content,
                summary = EXCLUDED.summary,
                embedding = EXCLUDED.embedding
            """,
            data["content_id"],
            data["media_id"],
            data.get("agent_id"),
            data.get("customer_id"),
            data["user_id"],
            data["content_type"],
            data["content"],
            data.get("summary"),
            _encode_embedding(data.get("embedding")),
            _to_naive_utc(data["date_created"]),
        )
        return int(result.split()[-1])

    async def delete_from_postgres(self, entity_id: Any) -> None:
        """hard-delete a media_content row from L3.

        :param entity_id: content primary-key value
        :ptype entity_id: Any
        :return: nothing
        :rtype: None
        """
        if self.l3_pool is None:
            return
        await self.l3_pool.execute(
            "DELETE FROM media_content WHERE content_id = $1", entity_id,
        )

    def serialize(self, data: dict[str, Any]) -> bytes:
        """serialize a row dict for L2 storage.

        :param data: row data
        :ptype data: dict[str, Any]
        :return: JSON-encoded bytes
        :rtype: bytes
        """
        return json.dumps(data, default=_json_serializer).encode("utf-8")

    def deserialize(self, data: bytes) -> dict[str, Any]:
        """deserialize L2 payload back into a row dict.

        :param data: JSON-encoded bytes
        :ptype data: bytes
        :return: row data
        :rtype: dict[str, Any]
        """
        raw: dict[str, Any] = json.loads(data.decode("utf-8"))
        return _deserialize_with_types(raw, _MEDIA_CONTENT_FIELD_TYPES)

    async def hybrid_search(
        self,
        *,
        user_id: UUID,
        embedding: list[float],
        user_text: str,
        top_k: int,
        candidate_limit: int,
        similarity_threshold: float,
        recency_half_life_hours: float,
        signal_weights: dict[str, float],
        agent_id: UUID | None = None,
        customer_id: UUID | None = None,
        fts_min_len: int = 3,
        fts_max_len: int = 500,
    ) -> list[dict[str, Any]]:
        """parallel vector + FTS hybrid search joining media_content to media.

        absorbs :meth:`MemoryRetriever._query_media_content` from the
        retrieval module.

        :param user_id: owning user UUID (row filter)
        :ptype user_id: UUID
        :param embedding: query embedding vector
        :ptype embedding: list[float]
        :param user_text: raw query text for FTS
        :ptype user_text: str
        :param top_k: number of candidates to return after ranking
        :ptype top_k: int
        :param candidate_limit: per-query candidate pool size
        :ptype candidate_limit: int
        :param similarity_threshold: floor on hybrid score
        :ptype similarity_threshold: float
        :param recency_half_life_hours: exponential decay half-life
        :ptype recency_half_life_hours: float
        :param signal_weights: mapping ``{"semantic", "keyword",
            "recency"}`` to weights
        :ptype signal_weights: dict[str, float]
        :param agent_id: optional agent scope
        :ptype agent_id: UUID | None
        :param customer_id: optional customer scope
        :ptype customer_id: UUID | None
        :param fts_min_len: minimum query length for FTS activation
        :ptype fts_min_len: int
        :param fts_max_len: truncation length for FTS queries
        :ptype fts_max_len: int
        :return: ranked candidate list, top_k entries
        :rtype: list[dict[str, Any]]
        """
        if self.l3_pool is None:
            return []
        embedding_str = json.dumps(embedding)

        scope_conditions, scope_params, param_offset = _build_user_scope_clause(
            user_id,
            agent_id=agent_id,
            customer_id=customer_id,
            start_param=2,
            table_prefix="mc",
        )
        limit_param = f"${param_offset + 1}"

        # cache-bypass: vector-distance search joining media_content ->
        # media. not primary-key-addressable on either side; L1 row
        # cache cannot help. method on Collection preserves single
        # entry point + audit hooks.
        vec_coro = self.l3_pool.fetch(
            f"""
            SELECT mc.content_id, mc.content, mc.summary, mc.content_type,
                   mc.media_id, mc.date_created, mc.embedding,
                   med.media_category, med.metadata_json,
                   1 - (mc.embedding <=> $1::vector) AS similarity
            FROM media_content mc
            JOIN media med ON mc.media_id = med.media_id
            WHERE {scope_conditions} AND mc.embedding IS NOT NULL
            ORDER BY mc.embedding <=> $1::vector
            LIMIT {limit_param}
            """,
            embedding_str,
            *scope_params,
            candidate_limit,
        )

        fts_text = _build_fts_text(user_text, fts_min_len, fts_max_len)
        if fts_text:
            fts_scope_conditions, fts_scope_params, fts_param_offset = (
                _build_user_scope_clause(
                    user_id,
                    agent_id=agent_id,
                    customer_id=customer_id,
                    start_param=2,
                    table_prefix="mc",
                )
            )
            fts_limit_param = f"${fts_param_offset + 1}"
            # cache-bypass: FTS rank query joining media_content ->
            # media. see vec_coro justification above.
            fts_coro = self.l3_pool.fetch(
                f"""
                SELECT mc.content_id, mc.content, mc.summary, mc.content_type,
                       mc.media_id, mc.date_created, mc.embedding,
                       med.media_category, med.metadata_json,
                       ts_rank_cd(mc.search_vector, websearch_to_tsquery('english', $1)) AS fts_rank
                FROM media_content mc
                JOIN media med ON mc.media_id = med.media_id
                WHERE {fts_scope_conditions} AND mc.embedding IS NOT NULL
                  AND mc.search_vector @@ websearch_to_tsquery('english', $1)
                ORDER BY fts_rank DESC
                LIMIT {fts_limit_param}
                """,
                fts_text,
                *fts_scope_params,
                candidate_limit,
            )
            vec_rows, fts_rows = await asyncio.gather(vec_coro, fts_coro)
        else:
            vec_rows = await vec_coro
            fts_rows = []

        merged: dict[Any, dict[str, Any]] = {}
        for row in vec_rows:
            cid = row["content_id"]
            meta = row["metadata_json"]
            if isinstance(meta, str):
                meta = json.loads(meta)
            title = (
                (meta or {}).get("document_title")
                or (meta or {}).get("original_filename")
                or (meta or {}).get("title")
            )
            emb = row["embedding"]
            if isinstance(emb, str):
                emb = json.loads(emb)
            merged[cid] = {
                "content_id": cid,
                "content": row["content"],
                "summary": row["summary"],
                "content_type": row["content_type"],
                "media_id": str(row["media_id"]),
                "media_category": row["media_category"],
                "title": title,
                "date_created": row["date_created"],
                "similarity": float(row["similarity"]),
                "fts_rank": 0.0,
                "embedding": emb,
            }
        for row in fts_rows:
            cid = row["content_id"]
            if cid in merged:
                merged[cid]["fts_rank"] = float(row["fts_rank"])
            else:
                meta = row["metadata_json"]
                if isinstance(meta, str):
                    meta = json.loads(meta)
                title = (
                    (meta or {}).get("document_title")
                    or (meta or {}).get("original_filename")
                    or (meta or {}).get("title")
                )
                emb = row["embedding"]
                if isinstance(emb, str):
                    emb = json.loads(emb)
                merged[cid] = {
                    "content_id": cid,
                    "content": row["content"],
                    "summary": row["summary"],
                    "content_type": row["content_type"],
                    "media_id": str(row["media_id"]),
                    "media_category": row["media_category"],
                    "title": title,
                    "date_created": row["date_created"],
                    "similarity": 0.0,
                    "fts_rank": float(row["fts_rank"]),
                    "embedding": emb,
                }

        candidates = list(merged.values())
        if not candidates:
            return []

        _normalize_scores(candidates, "fts_rank")

        for c in candidates:
            recency = _recency_weight(c["date_created"], recency_half_life_hours)
            c["recency"] = round(recency, 4)
            c["hybrid_score"] = round(
                signal_weights["semantic"] * c["similarity"]
                + signal_weights["keyword"] * c["fts_rank"]
                + signal_weights["recency"] * recency,
                4,
            )

        filtered = [c for c in candidates if c["hybrid_score"] > similarity_threshold]
        filtered.sort(key=lambda c: c["hybrid_score"], reverse=True)
        return filtered[:top_k]

    async def search_by_ids(
        self,
        content_ids: list[UUID],
        user_id: UUID,
    ) -> list[dict[str, Any]]:
        """batch lookup of media_content rows by primary key.

        joins to the parent :class:`MediaEntity` for ``media_category``
        + ``metadata_json``, scoped to ``user_id``.

        :param content_ids: primary-key values to fetch
        :ptype content_ids: list[UUID]
        :param user_id: owning user UUID (row filter)
        :ptype user_id: UUID
        :return: list of row dicts
        :rtype: list[dict[str, Any]]
        """
        if self.l3_pool is None or not content_ids:
            return []
        # cache-bypass: batch IN(...) join spanning media_content ->
        # media. see :meth:`hybrid_search` for the detailed rationale.
        rows = await self.l3_pool.fetch(
            """
            SELECT mc.content_id, mc.content, mc.content_type, mc.media_id,
                   med.media_category, med.metadata_json, med.date_created
            FROM media_content mc
            JOIN media med ON mc.media_id = med.media_id
            WHERE mc.content_id = ANY($1::uuid[]) AND mc.user_id = $2
            """,
            content_ids,
            user_id,
        )
        return [dict(row) for row in rows]

    async def search_by_semantic(
        self,
        *,
        user_id: UUID,
        embedding: list[float],
        max_results: int,
        similarity_threshold: float,
    ) -> list[dict[str, Any]]:
        """vector-only semantic search (memory_search tool leg).

        :param user_id: owning user UUID
        :ptype user_id: UUID
        :param embedding: query embedding vector
        :ptype embedding: list[float]
        :param max_results: cap on returned rows
        :ptype max_results: int
        :param similarity_threshold: similarity floor
        :ptype similarity_threshold: float
        :return: list of row dicts
        :rtype: list[dict[str, Any]]
        """
        if self.l3_pool is None:
            return []
        embedding_str = json.dumps(embedding)
        # cache-bypass: vector-distance search joining media_content ->
        # media. see :meth:`hybrid_search`.
        rows = await self.l3_pool.fetch(
            """
            SELECT mc.content_id, mc.content, mc.content_type,
                   mc.media_id, med.media_category, med.metadata_json,
                   med.date_created,
                   1 - (mc.embedding <=> $1::vector) AS similarity
            FROM media_content mc
            JOIN media med ON mc.media_id = med.media_id
            WHERE mc.user_id = $2
              AND mc.embedding IS NOT NULL
            ORDER BY mc.embedding <=> $1::vector
            LIMIT $3
            """,
            embedding_str,
            user_id,
            max_results,
        )
        result: list[dict[str, Any]] = []
        for row in rows:
            sim = float(row["similarity"])
            if sim <= similarity_threshold:
                continue
            result.append(
                {
                    "content_id": str(row["content_id"]),
                    "content": row["content"],
                    "content_type": row["content_type"],
                    "media_id": str(row["media_id"]),
                    "media_category": row["media_category"],
                    "metadata_json": row["metadata_json"],
                    "date_created": row["date_created"],
                    "similarity": sim,
                },
            )
        return result

    async def search_by_fts(
        self,
        *,
        user_id: UUID,
        fts_text: str,
        max_results: int,
    ) -> list[dict[str, Any]]:
        """FTS keyword search for the memory_search tool.

        :param user_id: owning user UUID
        :ptype user_id: UUID
        :param fts_text: FTS query text
        :ptype fts_text: str
        :param max_results: cap on returned rows
        :ptype max_results: int
        :return: list of row dicts
        :rtype: list[dict[str, Any]]
        """
        if self.l3_pool is None:
            return []
        # cache-bypass: FTS rank query joining media_content -> media.
        rows = await self.l3_pool.fetch(
            """
            SELECT mc.content_id, mc.content, mc.content_type,
                   mc.media_id, med.media_category, med.metadata_json,
                   med.date_created,
                   ts_rank_cd(mc.search_vector, websearch_to_tsquery('english', $1)) AS fts_rank
            FROM media_content mc
            JOIN media med ON mc.media_id = med.media_id
            WHERE mc.user_id = $2
              AND mc.embedding IS NOT NULL
              AND mc.search_vector @@ websearch_to_tsquery('english', $1)
            ORDER BY fts_rank DESC
            LIMIT $3
            """,
            fts_text,
            user_id,
            max_results,
        )
        return [
            {
                "content_id": str(row["content_id"]),
                "content": row["content"],
                "content_type": row["content_type"],
                "media_id": str(row["media_id"]),
                "media_category": row["media_category"],
                "metadata_json": row["metadata_json"],
                "date_created": row["date_created"],
            }
            for row in rows
        ]

    async def fetch_content_for_recall(
        self,
        *,
        content_id: UUID,
        user_id: UUID,
    ) -> str | None:
        """fetch ``content`` for the recall_memory tool (media leg).

        :param content_id: media_content primary-key value
        :ptype content_id: UUID
        :param user_id: owning user UUID (row filter)
        :ptype user_id: UUID
        :return: content text or ``None``
        :rtype: str | None
        """
        if self.l3_pool is None:
            return None
        # cache-bypass: by-ID fetch scoped by user_id — the user_id
        # guard is a security predicate the L1 cache cannot enforce.
        row = await self.l3_pool.fetchrow(
            "SELECT content FROM media_content WHERE content_id = $1 AND user_id = $2",
            content_id,
            user_id,
        )
        if row is None:
            return None
        result: str = row["content"]
        return result


class MemoryChunkCollection(BaseCollection[MemoryChunkEntity]):
    """three-tier collection for :class:`MemoryChunkEntity`.

    document-style chunks with heading / page metadata; optional
    parent :class:`MediaEntity` through ``media_id``.
    """

    primary_key_column: str = "chunk_id"

    @property
    def table_name(self) -> str:
        """Return the database table name for this collection.

        :return: table name
        :rtype: str
        """
        return "memory_chunks"

    @property
    def entity_class(self) -> type[MemoryChunkEntity]:
        """Return the entity class for this collection.

        :return: entity class
        :rtype: type[MemoryChunkEntity]
        """
        return MemoryChunkEntity

    async def fetch_from_postgres(self, entity_id: Any) -> dict[str, Any] | None:
        """fetch one memory_chunks row from L3 by primary key.

        :param entity_id: chunk primary-key value
        :ptype entity_id: Any
        :return: row dict or ``None``
        :rtype: dict[str, Any] | None
        """
        if self.l3_pool is None:
            return None
        row = await self.l3_pool.fetchrow(
            "SELECT * FROM memory_chunks WHERE chunk_id = $1", entity_id,
        )
        result: dict[str, Any] | None = dict(row) if row is not None else None
        return result

    async def save_to_postgres(
        self, data: dict[str, Any], original_timestamp: datetime | None = None,
    ) -> int:
        """upsert one memory_chunks row into L3.

        :param data: row data to persist
        :ptype data: dict[str, Any]
        :param original_timestamp: ignored for chunks (no CAS column
            distinct from ``date_created``)
        :ptype original_timestamp: datetime | None
        :return: rows affected
        :rtype: int
        """
        _ = original_timestamp
        if self.l3_pool is None:
            return 0
        result = await self.l3_pool.execute(
            """
            INSERT INTO memory_chunks (
                chunk_id, media_id, agent_id, customer_id,
                user_id, content, summary, heading_context,
                page_number, embedding, date_created
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::vector, $11)
            ON CONFLICT (chunk_id) DO UPDATE SET
                content = EXCLUDED.content,
                summary = EXCLUDED.summary,
                heading_context = EXCLUDED.heading_context,
                page_number = EXCLUDED.page_number,
                embedding = EXCLUDED.embedding
            """,
            data["chunk_id"],
            data.get("media_id"),
            data.get("agent_id"),
            data.get("customer_id"),
            data["user_id"],
            data["content"],
            data.get("summary"),
            data.get("heading_context"),
            data.get("page_number"),
            _encode_embedding(data.get("embedding")),
            _to_naive_utc(data["date_created"]),
        )
        return int(result.split()[-1])

    async def delete_from_postgres(self, entity_id: Any) -> None:
        """hard-delete a memory_chunks row from L3.

        :param entity_id: chunk primary-key value
        :ptype entity_id: Any
        :return: nothing
        :rtype: None
        """
        if self.l3_pool is None:
            return
        await self.l3_pool.execute(
            "DELETE FROM memory_chunks WHERE chunk_id = $1", entity_id,
        )

    def serialize(self, data: dict[str, Any]) -> bytes:
        """serialize a row dict for L2 storage.

        :param data: row data
        :ptype data: dict[str, Any]
        :return: JSON-encoded bytes
        :rtype: bytes
        """
        return json.dumps(data, default=_json_serializer).encode("utf-8")

    def deserialize(self, data: bytes) -> dict[str, Any]:
        """deserialize L2 payload back into a row dict.

        :param data: JSON-encoded bytes
        :ptype data: bytes
        :return: row data
        :rtype: dict[str, Any]
        """
        raw: dict[str, Any] = json.loads(data.decode("utf-8"))
        return _deserialize_with_types(raw, _MEMORY_CHUNK_FIELD_TYPES)

    async def hybrid_search(
        self,
        *,
        user_id: UUID,
        embedding: list[float],
        user_text: str,
        candidate_k: int,
        similarity_threshold: float,
        chunk_signal_weights: dict[str, float],
        agent_id: UUID | None = None,
        customer_id: UUID | None = None,
        fts_min_len: int = 3,
        fts_max_len: int = 500,
    ) -> list[dict[str, Any]]:
        """parallel vector + FTS search against memory_chunks + media.

        absorbs :meth:`MemoryRetriever._query_chunks`. two-signal
        ranking (semantic + keyword); no recency decay because chunks
        come from documents whose freshness signal is the upload
        event, not the chunk's own lifecycle.

        :param user_id: owning user UUID
        :ptype user_id: UUID
        :param embedding: query embedding vector
        :ptype embedding: list[float]
        :param user_text: raw query text for FTS
        :ptype user_text: str
        :param candidate_k: per-query candidate pool size
        :ptype candidate_k: int
        :param similarity_threshold: floor on hybrid score
        :ptype similarity_threshold: float
        :param chunk_signal_weights: mapping ``{"semantic",
            "keyword"}`` to weights
        :ptype chunk_signal_weights: dict[str, float]
        :param agent_id: optional agent scope
        :ptype agent_id: UUID | None
        :param customer_id: optional customer scope
        :ptype customer_id: UUID | None
        :param fts_min_len: minimum query length for FTS activation
        :ptype fts_min_len: int
        :param fts_max_len: truncation length for FTS queries
        :ptype fts_max_len: int
        :return: ranked candidate list
        :rtype: list[dict[str, Any]]
        """
        if self.l3_pool is None:
            return []
        embedding_str = json.dumps(embedding)

        scope_conditions, scope_params, param_offset = _build_user_scope_clause(
            user_id,
            agent_id=agent_id,
            customer_id=customer_id,
            start_param=2,
            table_prefix="mc",
        )
        limit_param = f"${param_offset + 1}"

        # cache-bypass: vector-distance search joining memory_chunks ->
        # media. not primary-key-addressable; L1 row cache cannot help.
        # method on Collection preserves single entry point.
        vec_coro = self.l3_pool.fetch(
            f"""
            SELECT mc.chunk_id, mc.content, mc.summary, mc.heading_context,
                   mc.page_number, mc.media_id,
                   mc.embedding, med.metadata_json,
                   1 - (mc.embedding <=> $1::vector) AS similarity
            FROM memory_chunks mc
            LEFT JOIN media med ON mc.media_id = med.media_id
            WHERE {scope_conditions}
            ORDER BY mc.embedding <=> $1::vector
            LIMIT {limit_param}
            """,
            embedding_str,
            *scope_params,
            candidate_k,
        )

        fts_text = _build_fts_text(user_text, fts_min_len, fts_max_len)
        if fts_text:
            fts_scope_conditions, fts_scope_params, fts_param_offset = (
                _build_user_scope_clause(
                    user_id,
                    agent_id=agent_id,
                    customer_id=customer_id,
                    start_param=2,
                    table_prefix="mc",
                )
            )
            fts_limit_param = f"${fts_param_offset + 1}"
            # cache-bypass: FTS rank query joining memory_chunks -> media.
            fts_coro = self.l3_pool.fetch(
                f"""
                SELECT mc.chunk_id, mc.content, mc.summary, mc.heading_context,
                       mc.page_number, mc.media_id,
                       mc.embedding, med.metadata_json,
                       ts_rank_cd(mc.search_vector, websearch_to_tsquery('english', $1)) AS fts_rank
                FROM memory_chunks mc
                LEFT JOIN media med ON mc.media_id = med.media_id
                WHERE {fts_scope_conditions}
                  AND mc.search_vector @@ websearch_to_tsquery('english', $1)
                ORDER BY fts_rank DESC
                LIMIT {fts_limit_param}
                """,
                fts_text,
                *fts_scope_params,
                candidate_k,
            )
            vec_rows, fts_rows = await asyncio.gather(vec_coro, fts_coro)
        else:
            vec_rows = await vec_coro
            fts_rows = []

        merged: dict[Any, dict[str, Any]] = {}
        for row in vec_rows:
            ckid = row["chunk_id"]
            title = None
            meta = row["metadata_json"]
            if meta:
                if isinstance(meta, str):
                    meta = json.loads(meta)
                title = meta.get("document_title") or meta.get("original_filename")
            emb = row["embedding"]
            if isinstance(emb, str):
                emb = json.loads(emb)
            merged[ckid] = {
                "chunk_id": ckid,
                "content": row["content"],
                "summary": row["summary"],
                "heading_context": row["heading_context"],
                "page_number": row["page_number"],
                "media_id": str(row["media_id"]) if row["media_id"] else None,
                "title": title,
                "similarity": float(row["similarity"]),
                "fts_rank": 0.0,
                "embedding": emb,
            }
        for row in fts_rows:
            ckid = row["chunk_id"]
            if ckid in merged:
                merged[ckid]["fts_rank"] = float(row["fts_rank"])
            else:
                title = None
                meta = row["metadata_json"]
                if meta:
                    if isinstance(meta, str):
                        meta = json.loads(meta)
                    title = meta.get("document_title") or meta.get("original_filename")
                emb = row["embedding"]
                if isinstance(emb, str):
                    emb = json.loads(emb)
                merged[ckid] = {
                    "chunk_id": ckid,
                    "content": row["content"],
                    "summary": row["summary"],
                    "heading_context": row["heading_context"],
                    "page_number": row["page_number"],
                    "media_id": str(row["media_id"]) if row["media_id"] else None,
                    "title": title,
                    "similarity": 0.0,
                    "fts_rank": float(row["fts_rank"]),
                    "embedding": emb,
                }

        candidates = list(merged.values())
        if not candidates:
            return []

        _normalize_scores(candidates, "fts_rank")

        for c in candidates:
            c["hybrid_score"] = round(
                chunk_signal_weights["semantic"] * c["similarity"]
                + chunk_signal_weights["keyword"] * c["fts_rank"],
                4,
            )

        filtered = [c for c in candidates if c["hybrid_score"] > similarity_threshold]
        filtered.sort(key=lambda c: c["hybrid_score"], reverse=True)
        return filtered

    async def search_by_ids(
        self,
        chunk_ids: list[UUID],
        user_id: UUID,
    ) -> list[dict[str, Any]]:
        """batch lookup of memory_chunks rows by primary key.

        :param chunk_ids: primary-key values to fetch
        :ptype chunk_ids: list[UUID]
        :param user_id: owning user UUID (row filter)
        :ptype user_id: UUID
        :return: list of row dicts joined to media
        :rtype: list[dict[str, Any]]
        """
        if self.l3_pool is None or not chunk_ids:
            return []
        # cache-bypass: batch IN(...) left-join spanning memory_chunks
        # -> media. see :meth:`hybrid_search`.
        rows = await self.l3_pool.fetch(
            """
            SELECT mc.chunk_id, mc.content, mc.heading_context, mc.page_number,
                   med.metadata_json
            FROM memory_chunks mc
            LEFT JOIN media med ON mc.media_id = med.media_id
            WHERE mc.chunk_id = ANY($1::uuid[]) AND mc.user_id = $2
            """,
            chunk_ids,
            user_id,
        )
        return [dict(row) for row in rows]

    async def search_by_semantic(
        self,
        *,
        user_id: UUID,
        embedding: list[float],
        max_results: int,
        similarity_threshold: float,
    ) -> list[dict[str, Any]]:
        """vector-only semantic search (memory_search tool leg).

        :param user_id: owning user UUID
        :ptype user_id: UUID
        :param embedding: query embedding vector
        :ptype embedding: list[float]
        :param max_results: cap on returned rows
        :ptype max_results: int
        :param similarity_threshold: similarity floor
        :ptype similarity_threshold: float
        :return: list of row dicts
        :rtype: list[dict[str, Any]]
        """
        if self.l3_pool is None:
            return []
        embedding_str = json.dumps(embedding)
        # cache-bypass: vector-distance search joining memory_chunks
        # -> media. see :meth:`hybrid_search`.
        rows = await self.l3_pool.fetch(
            """
            SELECT mc.chunk_id, mc.content, mc.heading_context, mc.page_number,
                   med.metadata_json,
                   1 - (mc.embedding <=> $1::vector) AS similarity
            FROM memory_chunks mc
            LEFT JOIN media med ON mc.media_id = med.media_id
            WHERE mc.user_id = $2
            ORDER BY mc.embedding <=> $1::vector
            LIMIT $3
            """,
            embedding_str,
            user_id,
            max_results,
        )
        result: list[dict[str, Any]] = []
        for row in rows:
            sim = float(row["similarity"])
            if sim <= similarity_threshold:
                continue
            result.append(
                {
                    "chunk_id": str(row["chunk_id"]),
                    "content": row["content"],
                    "heading_context": row["heading_context"],
                    "page_number": row["page_number"],
                    "metadata_json": row["metadata_json"],
                    "similarity": sim,
                },
            )
        return result

    async def fetch_content_for_recall(
        self,
        *,
        chunk_id: UUID,
        user_id: UUID,
    ) -> str | None:
        """fetch ``content`` for the recall_memory tool (chunk leg).

        :param chunk_id: chunk primary-key value
        :ptype chunk_id: UUID
        :param user_id: owning user UUID (row filter)
        :ptype user_id: UUID
        :return: chunk text or ``None``
        :rtype: str | None
        """
        if self.l3_pool is None:
            return None
        # cache-bypass: by-ID fetch scoped by user_id — security
        # predicate the L1 cache cannot enforce.
        row = await self.l3_pool.fetchrow(
            "SELECT content FROM memory_chunks WHERE chunk_id = $1 AND user_id = $2",
            chunk_id,
            user_id,
        )
        if row is None:
            return None
        result: str = row["content"]
        return result


_MEMORY_REF_FIELD_TYPES: dict[str, Any] = {
    "conversation_id": UUID,
    "item_id": UUID,
    "item_type": str,
    "short_desc": str,
    "date_added": datetime,
}


def _coerce_uuid_fields(row: dict[str, Any]) -> dict[str, Any]:
    """coerce pgproto UUID instances in a row dict to stdlib UUID.

    asyncpg returns UUID columns as ``asyncpg.pgproto.pgproto.UUID``,
    which SQLite's parameter binder rejects when the row is written
    into L1 even though pgproto UUID subclasses stdlib UUID (the
    binder special-cases stdlib UUID and errors on subclasses). the
    ``type() is not UUID`` check forces conversion for any subclass.
    converting at the fetch boundary lets every downstream tier
    accept the values uniformly.

    :param row: row dict as returned by asyncpg
    :ptype row: dict[str, Any]
    :return: same dict with pk UUID columns coerced
    :rtype: dict[str, Any]
    """
    for key in ("conversation_id", "item_id"):
        value = row.get(key)
        if value is not None and type(value) is not UUID:
            row[key] = UUID(str(value))
    return row


class MemoryRefsCollection(BaseCollection[MemoryRefEntity]):
    """three-tier collection for :class:`MemoryRefEntity`.

    namespace-task-01 phase 8.5l-2: retires :class:`MemoryLedger` — the
    bespoke wrapper that sat on top of ``SQLiteBackend`` with hand-rolled
    pool.fetch / pool.execute against ``conversation_memory_refs``. on
    top of 8.5l-1's composite-pk :class:`BaseCollection` support, the
    table fits the Collection contract natively:
    ``primary_key_column = ("conversation_id", "item_id")``. the L2
    invalidation envelope carries ``ids: [<conversation_id>,
    <item_id>]`` per 8.5l-1's wire format.

    tier configuration: L1 (SQLite) + L2 (NATS KV) + L3 (postgres /
    ``NatsProxyL3Backend``). full three-tier CRUD via
    :meth:`save_entity` / :meth:`get` / :meth:`delete`. per-conversation
    multi-row scans land on :meth:`find_by_conversation` with an
    explicit ``# cache-bypass:`` annotation documenting the query is
    not primary-key-addressable.
    """

    primary_key_column: str | tuple[str, ...] = ("conversation_id", "item_id")

    @property
    def table_name(self) -> str:
        """return database table name.

        :return: table name
        :rtype: str
        """
        return "conversation_memory_refs"

    @property
    def entity_class(self) -> type[MemoryRefEntity]:
        """return entity class for this collection.

        :return: entity class
        :rtype: type[MemoryRefEntity]
        """
        return MemoryRefEntity

    async def fetch_from_postgres(self, entity_id: Any) -> dict[str, Any] | None:
        """fetch one row from L3 by composite pk.

        :param entity_id: ``(conversation_id, item_id)`` tuple; scalar
            inputs raise in :meth:`BaseCollection.normalize_pk`
        :ptype entity_id: Any
        :return: row dict on hit, ``None`` on miss
        :rtype: dict[str, Any] | None
        """
        if self.l3_pool is None:
            return None
        key = self.normalize_pk(entity_id)
        row = await self.l3_pool.fetchrow(
            """
            SELECT conversation_id, item_id, item_type, short_desc, date_added
            FROM conversation_memory_refs
            WHERE conversation_id = $1 AND item_id = $2
            """,
            key[0],
            key[1],
        )
        if row is None:
            return None
        return _coerce_uuid_fields(dict(row))

    async def save_to_postgres(
        self, data: dict[str, Any], original_timestamp: datetime | None = None,
    ) -> int:
        """upsert one row into L3 via composite-pk ON CONFLICT.

        the table has no ``date_updated`` column — optimistic-
        concurrency CAS does not apply and ``original_timestamp`` is
        ignored. truncate ``short_desc`` to 150 chars to match the
        migration-v002 VARCHAR(150) bound.

        :param data: row data; must contain both pk columns plus
            ``item_type`` / ``short_desc`` / ``date_added``
        :ptype data: dict[str, Any]
        :param original_timestamp: ignored (no CAS column)
        :ptype original_timestamp: datetime | None
        :return: rows affected (1 on success, 0 on failure)
        :rtype: int
        """
        _ = original_timestamp
        if self.l3_pool is None:
            return 0
        desc_value: str = data["short_desc"]
        if len(desc_value) > 150:
            desc_value = desc_value[:150]
        status = await self.l3_pool.execute(
            """
            INSERT INTO conversation_memory_refs (
                conversation_id, item_id, item_type, short_desc, date_added
            ) VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (conversation_id, item_id) DO UPDATE SET
                item_type = EXCLUDED.item_type,
                short_desc = EXCLUDED.short_desc,
                date_added = EXCLUDED.date_added
            """,
            data["conversation_id"],
            data["item_id"],
            data["item_type"],
            desc_value,
            _to_naive_utc(data["date_added"]),
        )
        return 1 if status else 0

    async def delete_from_postgres(self, entity_id: Any) -> None:
        """hard-delete one row from L3 by composite pk.

        :param entity_id: ``(conversation_id, item_id)`` tuple
        :ptype entity_id: Any
        :return: nothing
        :rtype: None
        """
        if self.l3_pool is None:
            return
        key = self.normalize_pk(entity_id)
        await self.l3_pool.execute(
            "DELETE FROM conversation_memory_refs WHERE conversation_id = $1 AND item_id = $2",
            key[0],
            key[1],
        )

    def serialize(self, data: dict[str, Any]) -> bytes:
        """serialize a row dict for L2 storage.

        :param data: row data
        :ptype data: dict[str, Any]
        :return: JSON-encoded bytes
        :rtype: bytes
        """
        return json.dumps(data, default=_json_serializer).encode("utf-8")

    def deserialize(self, data: bytes) -> dict[str, Any]:
        """deserialize L2 payload back into a row dict.

        :param data: JSON-encoded bytes
        :ptype data: bytes
        :return: row data with typed columns
        :rtype: dict[str, Any]
        """
        raw: dict[str, Any] = json.loads(data.decode("utf-8"))
        return _deserialize_with_types(raw, _MEMORY_REF_FIELD_TYPES)

    async def find_by_conversation(
        self, conversation_id: UUID,
    ) -> list[MemoryRefEntity]:
        """fetch every ref for a conversation, ordered by ``date_added`` asc.

        absorbs the former ``MemoryLedger.load(pool, conversation_id)``
        SQL into a Collection method. multi-row scan is not primary-
        key-addressable: the query touches every row sharing the
        ``conversation_id`` prefix of the composite pk, so L1 row-level
        cache does not apply. each hit is promoted into L2 so other
        pods starting cold can resolve the row without an L3 round-
        trip.

        :param conversation_id: conversation UUID to scan for
        :ptype conversation_id: UUID
        :return: list of ref entities in chronological order
        :rtype: list[MemoryRefEntity]
        """
        if self.l3_pool is None:
            return []
        # cache-bypass: multi-row scan by ``conversation_id`` (first
        # column of the composite pk) is not primary-key-addressable.
        # L1 row cache serves per-``(conversation_id, item_id)`` lookups
        # only; a prefix scan would not hit. keeping the query on the
        # Collection preserves the single entry point + lets callers
        # share L2 invalidation channels.
        rows = await self.l3_pool.fetch(
            """
            SELECT conversation_id, item_id, item_type, short_desc, date_added
            FROM conversation_memory_refs
            WHERE conversation_id = $1
            ORDER BY date_added ASC
            """,
            conversation_id,
        )
        entities: list[MemoryRefEntity] = []
        for row in rows:
            data = _coerce_uuid_fields(dict(row))
            entity = self.entity_class(data, is_new=False, collection=self)
            entity_id = (data["conversation_id"], data["item_id"])
            await self._save_to_l2(entity_id, data)
            entities.append(entity)
        return entities
