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
from threetears.core.logging import get_logger

from threetears.agent.memory.entities import MemoryEntity

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

    _primary_key_column: str = "memory_id"

    def __init__(
        self,
        registry: CollectionRegistry,
        config: CoreConfig,
        postgres_pool: Any,
        nats_client: Any = None,
        write_buffer: WriteBuffer | None = None,
    ) -> None:
        self._postgres_pool = postgres_pool
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
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
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
                data["embedding"],
                data["is_deleted"],
                data.get("media_id"),
                data["date_created"],
                data.get("date_deleted"),
                data.get("date_updated"),
            )
        else:
            result = await self._postgres_pool.execute(
                """
                UPDATE memories SET
                    content = $2,
                    embedding = $3,
                    is_deleted = $4,
                    date_deleted = $5,
                    date_updated = $6
                WHERE memory_id = $1 AND date_updated = $7
                """,
                data["memory_id"],
                data["content"],
                data["embedding"],
                data["is_deleted"],
                data.get("date_deleted"),
                data.get("date_updated"),
                original_timestamp,
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

    async def find_by_user(self, user_id: UUID, include_deleted: bool = False) -> list[MemoryEntity]:
        """Fetch all memories for user from L3, promote to caches.

        :param user_id: user whose memories to fetch
        :ptype user_id: UUID
        :param include_deleted: whether to include soft-deleted memories
        :ptype include_deleted: bool
        :return: list of memory entities belonging to user
        :rtype: list[MemoryEntity]
        """
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
            entity._original_date_updated = data.get("date_updated")
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
    ) -> list[MemoryEntity]:
        """Fetch memories scoped by agent, optionally narrowed by customer and user.

        :param agent_id: agent ID scope (required)
        :ptype agent_id: UUID
        :param customer_id: optional customer ID to further narrow scope
        :ptype customer_id: UUID | None
        :param user_id: optional user ID to further narrow scope
        :ptype user_id: UUID | None
        :param include_deleted: whether to include soft-deleted memories
        :ptype include_deleted: bool
        :return: list of memory entities matching scope
        :rtype: list[MemoryEntity]
        """
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
            entity._original_date_updated = data.get("date_updated")
            entity_id = data["memory_id"]
            await self._save_to_l2(entity_id, data)
            entities.append(entity)
        return entities

    async def soft_delete(self, entity: MemoryEntity) -> None:
        """Soft-delete a memory by setting is_deleted and date_deleted."""
        entity.is_deleted = True
        entity.date_deleted = datetime.now(UTC)
        await self.save_entity(entity)
