"""
ConversationsCollection -- three-tier CRUD for :class:`Conversation`.

mirrors the shape of :class:`~threetears.agent.memory.collections.
MemoriesCollection`: L1 SQLite (pod-local) in front of L2 NATS KV in
front of L3 YugabyteDB, with a :class:`WriteBuffer` batching writes.
the collection is agent-scoped; the underlying asyncpg pool is
expected to have ``search_path`` already set to the per-agent
schema by the L3 broker before the collection is constructed.
"""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from threetears.conversations.entity import Conversation
from threetears.core.collections.base import BaseCollection
from threetears.core.collections.flush import WriteBuffer
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import CoreConfig
from threetears.observe import get_logger

log = get_logger(__name__)


# field type mapping for JSON round-trips through L2.
_FIELD_TYPES: dict[str, Any] = {
    "id": UUID,
    "agent_id": UUID,
    "customer_id": UUID,
    "user_id": UUID,
    "channel_type": str,
    "conversation_ref": str,
    "status": str,
    "summary": str,
    "date_created": datetime,
    "date_updated": datetime,
    "date_last_message": datetime,
    "metadata": dict,
}


def _json_serializer(obj: object) -> str | int | float | bool | None:
    """
    JSON encoder hook covering UUID, datetime, and Enum fields.

    :param obj: value json.dumps could not encode natively
    :ptype obj: object
    :return: JSON-serializable representation
    :rtype: str | int | float | bool | None
    :raises TypeError: if obj is not a supported non-native type
    """
    if isinstance(obj, UUID):
        result: str | int | float | bool | None = str(obj)
        return result
    if isinstance(obj, datetime):
        result = obj.isoformat()
        return result
    if isinstance(obj, Enum):
        result = obj.value  # type: ignore[assignment]
        return result
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _resolve_base_type(type_hint: Any) -> type | None:
    """
    resolve a possibly-``Optional`` / generic type hint to its origin.

    :param type_hint: entry from :data:`_FIELD_TYPES`
    :ptype type_hint: Any
    :return: concrete type or ``None`` when the hint is ``Optional[None]``
    :rtype: type | None
    """
    import types
    from typing import get_args, get_origin

    origin = get_origin(type_hint)
    result: type | None
    if origin is None:
        result = type_hint
    elif origin is types.UnionType:
        args = get_args(type_hint)
        non_none = [a for a in args if a is not type(None)]
        if non_none:
            inner = non_none[0]
            inner_origin = get_origin(inner)
            result = inner_origin if inner_origin is not None else inner
        else:
            result = None
    else:
        result = origin
    return result


class ConversationsCollection(BaseCollection[Conversation]):
    """
    three-tier collection for :class:`Conversation` entities.

    the collection is the sole writer to the ``conversations`` table:
    memory and agent-tools packages read conversations through it (or
    via their own foreign-key joins) but never mutate the table
    directly. inserts are upserts keyed on ``id`` so re-ingest is
    safe.

    :param registry: shared collection registry providing L1 / L3
        handles
    :ptype registry: CollectionRegistry
    :param config: :class:`CoreConfig` controlling flush strategy and
        cache behaviour
    :ptype config: CoreConfig
    :param postgres_pool: asyncpg pool bound to the per-agent schema
    :ptype postgres_pool: Any
    :param nats_client: connected NATS client for L2 propagation, or
        ``None`` in test harnesses
    :ptype nats_client: Any
    :param write_buffer: optional shared :class:`WriteBuffer` for
        bounded-concurrency flushing
    :ptype write_buffer: WriteBuffer | None
    """

    _primary_key_column: str = "id"

    def __init__(
        self,
        registry: CollectionRegistry,
        config: CoreConfig,
        postgres_pool: Any,
        nats_client: Any = None,
        write_buffer: WriteBuffer | None = None,
    ) -> None:
        """
        initialize the collection and register it with the registry.

        :param registry: shared collection registry
        :ptype registry: CollectionRegistry
        :param config: core config driving flush behaviour
        :ptype config: CoreConfig
        :param postgres_pool: asyncpg pool bound to the agent schema
        :ptype postgres_pool: Any
        :param nats_client: optional connected NATS client
        :ptype nats_client: Any
        :param write_buffer: optional shared write buffer
        :ptype write_buffer: WriteBuffer | None
        """
        self._postgres_pool = postgres_pool
        super().__init__(registry, config, nats_client, write_buffer)

    @property
    def table_name(self) -> str:
        """
        return the table name for this collection.

        :return: ``"conversations"``
        :rtype: str
        """
        return "conversations"

    @property
    def entity_class(self) -> type[Conversation]:
        """
        return the entity class this collection produces.

        :return: :class:`Conversation`
        :rtype: type[Conversation]
        """
        return Conversation

    async def _fetch_from_postgres(self, entity_id: Any) -> dict[str, Any] | None:
        """
        read a single row by primary key from L3.

        :param entity_id: conversation UUID
        :ptype entity_id: Any
        :return: row dict or ``None`` if missing
        :rtype: dict[str, Any] | None
        """
        row = await self._postgres_pool.fetchrow(
            "SELECT * FROM conversations WHERE id = $1", entity_id
        )
        result: dict[str, Any] | None = None if row is None else dict(row)
        return result

    async def _save_to_postgres(
        self, data: dict[str, Any], original_timestamp: datetime | None = None
    ) -> int:
        """
        upsert one conversation row in L3 with optimistic concurrency.

        new rows take the full insert path; updates use the stored
        ``date_updated`` as a fence so concurrent writers race
        correctly rather than silently overwriting each other.

        :param data: fully populated conversation row dict
        :ptype data: dict[str, Any]
        :param original_timestamp: previous ``date_updated`` for
            optimistic concurrency (``None`` on insert)
        :ptype original_timestamp: datetime | None
        :return: number of rows affected
        :rtype: int
        """
        if original_timestamp is None:
            sql_insert = (
                "INSERT INTO conversations ("
                "id, agent_id, customer_id, user_id, channel_type, "
                "conversation_ref, status, summary, date_created, "
                "date_updated, date_last_message, metadata"
                ") VALUES ("
                "$1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12"
                ") ON CONFLICT (id) DO UPDATE SET "
                "status = EXCLUDED.status, "
                "summary = EXCLUDED.summary, "
                "date_updated = EXCLUDED.date_updated, "
                "date_last_message = EXCLUDED.date_last_message, "
                "metadata = EXCLUDED.metadata"
            )
            result = await self._postgres_pool.execute(
                sql_insert,
                data["id"],
                data["agent_id"],
                data["customer_id"],
                data["user_id"],
                data["channel_type"],
                data.get("conversation_ref"),
                data["status"],
                data.get("summary"),
                data["date_created"],
                data["date_updated"],
                data.get("date_last_message"),
                json.dumps(data.get("metadata")) if data.get("metadata") is not None else None,
            )
        else:
            sql_update = (
                "UPDATE conversations SET "
                "status = $2, summary = $3, date_updated = $4, "
                "date_last_message = $5, metadata = $6 "
                "WHERE id = $1 AND date_updated = $7"
            )
            result = await self._postgres_pool.execute(
                sql_update,
                data["id"],
                data["status"],
                data.get("summary"),
                data["date_updated"],
                data.get("date_last_message"),
                json.dumps(data.get("metadata")) if data.get("metadata") is not None else None,
                original_timestamp,
            )
        affected = int(result.split()[-1])
        return affected

    async def _delete_from_postgres(self, entity_id: Any) -> None:
        """
        delete a conversation row from L3.

        :param entity_id: conversation UUID
        :ptype entity_id: Any
        """
        await self._postgres_pool.execute(
            "DELETE FROM conversations WHERE id = $1", entity_id
        )

    def _serialize(self, data: dict[str, Any]) -> bytes:
        """
        serialize a row dict for L2 storage.

        :param data: row dict
        :ptype data: dict[str, Any]
        :return: JSON-encoded bytes
        :rtype: bytes
        """
        return json.dumps(data, default=_json_serializer).encode("utf-8")

    def _deserialize(self, data: bytes) -> dict[str, Any]:
        """
        deserialize an L2 row payload, coercing UUIDs and timestamps.

        :param data: JSON-encoded bytes
        :ptype data: bytes
        :return: row dict with strongly typed UUID / datetime values
        :rtype: dict[str, Any]
        """
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
            else:
                result[key] = value
        return result

    async def find_by_user(
        self, user_id: UUID, include_closed: bool = False
    ) -> list[Conversation]:
        """
        fetch every conversation owned by the given user.

        results come from L3 (the source of truth for historical rows)
        and are promoted into L2 so subsequent reads hit the cache
        tier. ordering is newest-first.

        :param user_id: user whose conversations to fetch
        :ptype user_id: UUID
        :param include_closed: include closed / archived conversations
        :ptype include_closed: bool
        :return: conversations owned by ``user_id``
        :rtype: list[Conversation]
        """
        if include_closed:
            rows = await self._postgres_pool.fetch(
                "SELECT * FROM conversations WHERE user_id = $1 "
                "ORDER BY date_created DESC",
                user_id,
            )
        else:
            rows = await self._postgres_pool.fetch(
                "SELECT * FROM conversations WHERE user_id = $1 AND status != $2 "
                "ORDER BY date_created DESC",
                user_id,
                "closed",
            )
        entities: list[Conversation] = []
        for row in rows:
            data = dict(row)
            entity = self.entity_class(data, is_new=False, collection=self)
            entity._original_date_updated = data.get("date_updated")
            entity_id = data["id"]
            await self._save_to_l2(entity_id, data)
            entities.append(entity)
        return entities
