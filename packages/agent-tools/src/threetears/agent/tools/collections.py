"""Context items collection — three-tier CRUD for conversation context.

Provides persistent, cross-pod storage for variables, tool results, and
media slots via the standard L1 (SQLite) → L2 (NATS KV) → L3 (PostgreSQL)
caching path.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import Column, DateTime, MetaData, Table, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PgUUID

from threetears.core.collections.base import BaseCollection
from threetears.core.logging import get_logger
from threetears.core.serialization import deserialize_from_json, serialize_to_json

from threetears.agent.tools.entities import ContextItemEntity

log = get_logger(__name__)

_FIELD_TYPES: dict[str, Any] = {
    "context_id": UUID,
    "conversation_id": UUID,
    "context_type": str,
    "key": str,
    "short_desc": str,
    "long_desc": str,
    "content": str,
    "metadata": dict,
    "date_accessed": datetime,
    "date_created": datetime,
    "date_updated": datetime,
}


def context_items_table(metadata: MetaData) -> Table:
    """Register the ``context_items`` table on the given SA metadata.

    Call this before ``SQLiteBackend.initialize(metadata)`` so the L1
    cache gets the correct schema.  Safe to call multiple times — returns
    the existing table if already registered.
    """
    if "context_items" in metadata.tables:
        return metadata.tables["context_items"]
    return Table(
        "context_items",
        metadata,
        Column("context_id", PgUUID(as_uuid=True), primary_key=True),
        Column("conversation_id", PgUUID(as_uuid=True), nullable=False),
        Column("context_type", Text(), nullable=False),
        Column("key", Text(), nullable=False),
        Column("short_desc", Text(), nullable=False),
        Column("long_desc", Text(), nullable=False, server_default=""),
        Column("content", Text(), nullable=False),
        Column("metadata", JSONB(), nullable=True),
        Column("date_accessed", DateTime(timezone=True), nullable=False),
        Column("date_created", DateTime(timezone=True), nullable=False),
        Column("date_updated", DateTime(timezone=True), nullable=False),
    )


class ContextItemCollection(BaseCollection[ContextItemEntity]):
    """Three-tier collection for conversation context items.

    Stores variables, tool results, and media slots in a single
    ``context_items`` table.  Provides conversation-scoped queries
    and LRU eviction for tool results.
    """

    _primary_key_column: str = "context_id"

    @property
    def table_name(self) -> str:
        return "context_items"

    @property
    def entity_class(self) -> type[ContextItemEntity]:
        return ContextItemEntity

    # -- Standard BaseCollection abstract methods --

    async def _fetch_from_postgres(self, entity_id: Any) -> dict[str, Any] | None:
        row = await self._l3_pool.fetchrow(
            "SELECT * FROM context_items WHERE context_id = $1",
            entity_id if isinstance(entity_id, UUID) else UUID(str(entity_id)),
        )
        return dict(row) if row else None

    async def _save_to_postgres(self, data: dict[str, Any], original_timestamp: datetime | None = None) -> int:
        context_id = data["context_id"]
        if not isinstance(context_id, UUID):
            context_id = UUID(str(context_id))
        conversation_id = data["conversation_id"]
        if not isinstance(conversation_id, UUID):
            conversation_id = UUID(str(conversation_id))

        metadata_val = data.get("metadata")
        if isinstance(metadata_val, dict):
            metadata_val = json.dumps(metadata_val)

        if original_timestamp is None:
            result = await self._l3_pool.execute(
                """
                INSERT INTO context_items (
                    context_id, conversation_id, context_type, key,
                    short_desc, long_desc, content, metadata,
                    date_accessed, date_created, date_updated
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9, $10, $11)
                ON CONFLICT (context_id) DO UPDATE SET
                    short_desc = EXCLUDED.short_desc,
                    long_desc = EXCLUDED.long_desc,
                    content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata,
                    date_accessed = EXCLUDED.date_accessed,
                    date_updated = EXCLUDED.date_updated
                """,
                context_id,
                conversation_id,
                data["context_type"],
                data["key"],
                data["short_desc"],
                data.get("long_desc", ""),
                data["content"],
                metadata_val,
                data["date_accessed"],
                data["date_created"],
                data["date_updated"],
            )
        else:
            result = await self._l3_pool.execute(
                """
                UPDATE context_items SET
                    short_desc = $2, long_desc = $3, content = $4, metadata = $5::jsonb,
                    date_accessed = $6, date_updated = $7
                WHERE context_id = $1 AND date_updated = $8
                """,
                context_id,
                data["short_desc"],
                data.get("long_desc", ""),
                data["content"],
                metadata_val,
                data["date_accessed"],
                data["date_updated"],
                original_timestamp,
            )
        return int(result.split()[-1])

    async def _delete_from_postgres(self, entity_id: Any) -> None:
        await self._l3_pool.execute(
            "DELETE FROM context_items WHERE context_id = $1",
            entity_id if isinstance(entity_id, UUID) else UUID(str(entity_id)),
        )

    def _serialize(self, data: dict[str, Any]) -> bytes:
        return serialize_to_json(data)

    def _deserialize(self, data: bytes) -> dict[str, Any]:
        return deserialize_from_json(data, _FIELD_TYPES)

    # -- Conversation-scoped queries --

    async def find_by_conversation(self, conversation_id: str | UUID) -> list[ContextItemEntity]:
        """Load all context items for a conversation from L3, populate L1."""
        cid = conversation_id if isinstance(conversation_id, UUID) else UUID(str(conversation_id))
        rows = await self._l3_pool.fetch(
            """
            SELECT * FROM context_items
            WHERE conversation_id = $1
            ORDER BY date_created ASC
            """,
            cid,
        )
        entities: list[ContextItemEntity] = []
        for row in rows:
            data = dict(row)
            entity = self.entity_class(data, is_new=False, collection=self)
            entity._original_date_updated = data.get("date_updated")
            self._write_to_cache_sync(data)
            entities.append(entity)
        return entities

    async def upsert_variable(self, data: dict[str, Any]) -> UUID:
        """Upsert a variable using the partial unique index.

        Returns the context_id (may differ from input on conflict).
        """
        context_id = data["context_id"]
        if not isinstance(context_id, UUID):
            context_id = UUID(str(context_id))
        conversation_id = data["conversation_id"]
        if not isinstance(conversation_id, UUID):
            conversation_id = UUID(str(conversation_id))

        metadata_val = data.get("metadata")
        if isinstance(metadata_val, dict):
            metadata_val = json.dumps(metadata_val)

        row = await self._l3_pool.fetchrow(
            """
            INSERT INTO context_items (
                context_id, conversation_id, context_type, key,
                short_desc, long_desc, content, metadata,
                date_accessed, date_created, date_updated
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9, $10, $11)
            ON CONFLICT (conversation_id, key) WHERE context_type = 'variable'
            DO UPDATE SET
                short_desc = EXCLUDED.short_desc,
                long_desc = EXCLUDED.long_desc,
                content = EXCLUDED.content,
                metadata = EXCLUDED.metadata,
                date_accessed = EXCLUDED.date_accessed,
                date_updated = EXCLUDED.date_updated
            RETURNING context_id
            """,
            context_id,
            conversation_id,
            "variable",
            data["key"],
            data["short_desc"],
            data.get("long_desc", ""),
            data["content"],
            metadata_val,
            data["date_accessed"],
            data["date_created"],
            data["date_updated"],
        )
        returned_id: UUID = row["context_id"]

        # Update L1 cache
        cache_data = dict(data)
        cache_data["context_id"] = returned_id
        self._write_to_cache_sync(cache_data)
        await self._save_to_l2(returned_id, cache_data)
        await self._publish_invalidation(returned_id)

        return returned_id

    async def touch(self, context_id: str | UUID) -> None:
        """Update ``date_accessed`` for LRU tracking.

        Writes to L1 synchronously, propagates to L2/L3 asynchronously.
        """
        cid = context_id if isinstance(context_id, UUID) else UUID(str(context_id))
        now = datetime.now(UTC)

        # Update L1 immediately
        if self._l1 is not None:
            row = self._l1.select_by_id(self.table_name, str(cid), self._primary_key_column)
            if row is not None:
                row["date_accessed"] = now
                self._l1.upsert(self.table_name, row, self._primary_key_column)

        # Propagate to L3 (fire-and-forget via background)
        try:
            await self._l3_pool.execute(
                "UPDATE context_items SET date_accessed = $2 WHERE context_id = $1",
                cid,
                now,
            )
        except Exception as exc:
            log.warning(
                "Failed to update date_accessed in L3",
                extra={"extra_data": {"context_id": str(cid), "error": str(exc)}},
            )

    async def count_results(self, conversation_id: str | UUID) -> int:
        """Count tool_result items for a conversation."""
        cid = conversation_id if isinstance(conversation_id, UUID) else UUID(str(conversation_id))
        row = await self._l3_pool.fetchrow(
            """
            SELECT COUNT(*) AS cnt FROM context_items
            WHERE conversation_id = $1 AND context_type = 'tool_result'
            """,
            cid,
        )
        return int(row["cnt"]) if row else 0

    async def evict_lru(self, conversation_id: str | UUID, result_limit: int) -> int:
        """Evict oldest tool_result items exceeding the limit.

        Only evicts ``context_type = 'tool_result'``.  Variables and
        media slots are never evicted.  Returns the number of items evicted.
        """
        cid = conversation_id if isinstance(conversation_id, UUID) else UUID(str(conversation_id))

        count = await self.count_results(cid)
        if count <= result_limit:
            return 0

        to_evict = count - result_limit
        evict_rows = await self._l3_pool.fetch(
            """
            SELECT context_id FROM context_items
            WHERE conversation_id = $1 AND context_type = 'tool_result'
            ORDER BY date_accessed ASC
            LIMIT $2
            """,
            cid,
            to_evict,
        )

        evicted = 0
        for row in evict_rows:
            eid = row["context_id"]
            await self._delete_from_postgres(eid)
            if self._l1 is not None:
                self._l1.delete_by_id(self.table_name, str(eid), self._primary_key_column)
            await self._delete_from_l2(eid)
            await self._publish_invalidation(eid)
            evicted += 1

        if evicted:
            log.debug(
                "LRU eviction completed",
                extra={
                    "extra_data": {
                        "conversation_id": str(cid),
                        "evicted": evicted,
                        "result_limit": result_limit,
                    }
                },
            )

        return evicted
