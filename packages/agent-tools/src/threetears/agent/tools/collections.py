"""Context items collection — three-tier CRUD for conversation context.

Provides persistent, cross-pod storage for variables, tool results, and
media slots via the standard L1 (SQLite) → L2 (NATS KV) → L3 (PostgreSQL)
caching path. CRUD is handled by :class:`SchemaBackedCollection`; this
module carries only the domain-specific queries (variable upsert, LRU
eviction, conversation scan) and the SQLAlchemy metadata helper used
by the L1 cache bootstrap.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import Column as SAColumn
from sqlalchemy import DateTime, MetaData, Table, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PgUUID

from threetears.core.collections.schema_backed import (
    DATETIME_TYPE,
    JSONB_TYPE,
    STRING_TYPE,
    UUID_TYPE,
    Column,
    SchemaBackedCollection,
    TableSchema,
)
from threetears.observe import get_logger

from threetears.agent.tools.entities import ContextItemEntity

__all__ = [
    "ContextItemCollection",
    "context_items_table",
    "migrate_context_items_schema",
]

log = get_logger(__name__)


def context_items_table(metadata: MetaData) -> Table:
    """register the ``context_items`` table on the given SA metadata.

    call this before ``SQLiteBackend.initialize(metadata)`` so the L1
    cache gets the correct schema. safe to call multiple times -- returns
    the existing table if already registered. composite primary key on
    ``(conversation_id, context_id)`` mirrors the L3 partition layout
    so cache rows are isolated per partition.

    :param metadata: SQLAlchemy metadata to attach the table to
    :ptype metadata: MetaData
    :return: the ``context_items`` :class:`Table`
    :rtype: Table
    """
    if "context_items" in metadata.tables:
        return metadata.tables["context_items"]
    return Table(
        "context_items",
        metadata,
        SAColumn("conversation_id", PgUUID(as_uuid=True), primary_key=True, nullable=False),
        SAColumn("context_id", PgUUID(as_uuid=True), primary_key=True, nullable=False),
        SAColumn("context_type", Text(), nullable=False),
        SAColumn("key", Text(), nullable=False),
        SAColumn("short_desc", Text(), nullable=False),
        SAColumn("long_desc", Text(), nullable=False, server_default=""),
        SAColumn("content", Text(), nullable=False),
        SAColumn("metadata", JSONB(), nullable=True),
        SAColumn("date_accessed", DateTime(timezone=True), nullable=False),
        SAColumn("date_created", DateTime(timezone=True), nullable=False),
        SAColumn("date_updated", DateTime(timezone=True), nullable=False),
    )


class ContextItemCollection(SchemaBackedCollection[ContextItemEntity]):
    """three-tier collection for conversation context items.

    stores variables, tool results, and media slots in a single
    ``context_items`` table. CRUD comes from
    :class:`SchemaBackedCollection`; domain methods
    (``find_by_conversation``, ``upsert_variable``, ``touch``,
    ``count_results``, ``evict_lru``) stay here because their query
    shape is per-collection.
    """

    primary_key_column: str | tuple[str, ...] = ("conversation_id", "context_id")
    schema = TableSchema(
        name="context_items",
        primary_key=("conversation_id", "context_id"),
        columns=[
            Column("conversation_id", UUID_TYPE, partition=True),
            Column("context_id", UUID_TYPE),
            Column("context_type", STRING_TYPE, immutable=True),
            Column("key", STRING_TYPE, immutable=True),
            Column("short_desc", STRING_TYPE),
            Column("long_desc", STRING_TYPE, nullable=True),
            Column("content", STRING_TYPE),
            Column("metadata", JSONB_TYPE, nullable=True),
            Column("date_accessed", DATETIME_TYPE),
            Column("date_created", DATETIME_TYPE, immutable=True),
            Column("date_updated", DATETIME_TYPE),
        ],
        cas_column="date_updated",
    )

    @property
    def table_name(self) -> str:
        """return the database table name.

        :return: table name
        :rtype: str
        """
        return "context_items"

    @property
    def entity_class(self) -> type[ContextItemEntity]:
        """return the entity class for this collection.

        :return: entity class
        :rtype: type[ContextItemEntity]
        """
        return ContextItemEntity

    # -- Conversation-scoped queries --

    async def find_by_conversation(self, conversation_id: str | UUID) -> list[ContextItemEntity]:
        """load all context items for a conversation from L3, populate L1.

        :param conversation_id: conversation UUID (accepts string or UUID)
        :ptype conversation_id: str | UUID
        :return: list of entities in chronological order
        :rtype: list[ContextItemEntity]
        """
        cid = conversation_id if isinstance(conversation_id, UUID) else UUID(str(conversation_id))
        rows = await self.l3_pool.fetch(
            """
            SELECT * FROM context_items
            WHERE conversation_id = $1
            ORDER BY date_created ASC
            """,
            cid,
        )
        entities: list[ContextItemEntity] = []
        for row in rows:
            data = self._coerce_row(dict(row))
            entity = self.entity_class(data, is_new=False, collection=self)
            entity.original_date_updated = data.get("date_updated")
            self.write_to_cache_sync(data)
            entities.append(entity)
        return entities

    async def upsert_variable(self, conversation_id: UUID, data: dict[str, Any]) -> UUID:
        """upsert a variable using the partial unique index.

        returns the context_id (may differ from input on conflict). uses
        the ``(conversation_id, key) WHERE context_type = 'variable'``
        partial unique index so a name collision returns the existing
        row's id rather than inserting a duplicate. ``conversation_id``
        is the partition column and is also read from ``data`` for
        backwards compatibility with existing callers; the explicit
        positional argument satisfies the partition-column contract.

        :param conversation_id: conversation partition the variable lives in
        :ptype conversation_id: UUID
        :param data: row dict keyed by column name
        :ptype data: dict[str, Any]
        :return: authoritative context_id (existing on conflict, new on insert)
        :rtype: UUID
        """
        context_id = data["context_id"]
        if not isinstance(context_id, UUID):
            context_id = UUID(str(context_id))
        if data.get("conversation_id") != conversation_id:
            data = dict(data)
            data["conversation_id"] = conversation_id

        metadata_val = data.get("metadata")
        if isinstance(metadata_val, dict):
            metadata_val = json.dumps(metadata_val)

        row = await self.l3_pool.fetchrow(
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

        # Update L1 cache. composite pk on (conversation_id, context_id)
        # so L2 / invalidation addressing must be the tuple form.
        cache_data = dict(data)
        cache_data["context_id"] = returned_id
        cache_data["conversation_id"] = conversation_id
        pk = (conversation_id, returned_id)
        self.write_to_cache_sync(cache_data)
        await self._save_to_l2(pk, cache_data)
        await self._publish_invalidation(pk)

        return returned_id

    async def touch(self, conversation_id: UUID, context_id: str | UUID) -> None:
        """update ``date_accessed`` for LRU tracking.

        writes to L1 synchronously, propagates to L3. L2 stays coherent
        via the standard invalidation path on subsequent writes.
        ``conversation_id`` is required because the table is partitioned
        on it; cache-row addressing uses the composite ``(conversation_id,
        context_id)`` tuple.

        :param conversation_id: conversation partition the row lives in
        :ptype conversation_id: UUID
        :param context_id: entity identifier (accepts string or UUID)
        :ptype context_id: str | UUID
        :return: nothing
        :rtype: None
        """
        cid = context_id if isinstance(context_id, UUID) else UUID(str(context_id))
        now = datetime.now(UTC)

        # Update L1 immediately. composite-pk row keyed on
        # (conversation_id, context_id) tuple per the partition layout.
        if self._l1 is not None:
            row = self._l1.select_by_id(
                self.table_name,
                (conversation_id, cid),
                self.primary_key_columns,
            )
            if row is not None:
                row["date_accessed"] = now
                self._l1.upsert(self.table_name, row, self.primary_key_columns)

        # Propagate to L3. partition predicate on conversation_id keeps
        # the UPDATE inside one partition and satisfies the SQL-level
        # partition-column enforcement.
        try:
            await self.l3_pool.execute(
                "UPDATE context_items SET date_accessed = $3 "
                "WHERE conversation_id = $1 AND context_id = $2",
                conversation_id,
                cid,
                now,
            )
        except Exception as exc:
            log.warning(
                "Failed to update date_accessed in L3",
                extra={
                    "extra_data": {
                        "conversation_id": str(conversation_id),
                        "context_id": str(cid),
                        "error": str(exc),
                    },
                },
            )

    async def count_results(self, conversation_id: str | UUID) -> int:
        """count tool_result items for a conversation.

        :param conversation_id: conversation UUID (accepts string or UUID)
        :ptype conversation_id: str | UUID
        :return: count of ``context_type = 'tool_result'`` rows
        :rtype: int
        """
        cid = conversation_id if isinstance(conversation_id, UUID) else UUID(str(conversation_id))
        row = await self.l3_pool.fetchrow(
            """
            SELECT COUNT(*) AS cnt FROM context_items
            WHERE conversation_id = $1 AND context_type = 'tool_result'
            """,
            cid,
        )
        return int(row["cnt"]) if row else 0

    async def evict_lru(self, conversation_id: str | UUID, result_limit: int) -> int:
        """evict oldest tool_result items exceeding the limit.

        only evicts ``context_type = 'tool_result'``. variables and
        media slots are never evicted.

        :param conversation_id: conversation UUID (accepts string or UUID)
        :ptype conversation_id: str | UUID
        :param result_limit: maximum tool_result rows to retain
        :ptype result_limit: int
        :return: number of items evicted
        :rtype: int
        """
        cid = conversation_id if isinstance(conversation_id, UUID) else UUID(str(conversation_id))

        count = await self.count_results(cid)
        if count <= result_limit:
            return 0

        to_evict = count - result_limit
        evict_rows = await self.l3_pool.fetch(
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
            pk = (cid, eid)
            await self.delete_from_postgres(pk)
            if self._l1 is not None:
                self._l1.delete_by_id(self.table_name, pk, self.primary_key_columns)
            await self._delete_from_l2(pk)
            await self._publish_invalidation(pk)
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


async def migrate_context_items_schema(pool: Any) -> bool:
    """migrate context_items table from legacy schema to v0.5.0 schema.

    detects old column names (``summary``, ``value``) and renames them
    to the current schema (``short_desc``, ``long_desc``, ``content``).
    backfills ``long_desc`` from the first 1000 chars of ``content``.

    safe to call on every startup -- detects whether migration is needed
    by probing the column list, and is a no-op if already up to date.
    idempotent: uses IF EXISTS / IF NOT EXISTS throughout.

    :param pool: asyncpg connection pool
    :ptype pool: Any
    :returns: True if columns were migrated, False if already current
    :rtype: bool
    """
    # Probe current columns
    cols = await pool.fetch(
        """
        SELECT column_name FROM information_schema.columns
        WHERE table_name = 'context_items'
        """
    )
    if not cols:
        return False  # Table doesn't exist yet

    col_names = {row["column_name"] for row in cols}

    needs_migration = "summary" in col_names or ("value" in col_names and "content" not in col_names)
    if not needs_migration:
        return False

    log.info("Migrating context_items schema to v0.5.0")

    async with pool.acquire() as conn:
        async with conn.transaction():
            # Rename summary → short_desc
            if "summary" in col_names and "short_desc" not in col_names:
                await conn.execute("ALTER TABLE context_items RENAME COLUMN summary TO short_desc")
                log.info("Renamed context_items.summary → short_desc")

            # Rename value → content
            if "value" in col_names and "content" not in col_names:
                await conn.execute("ALTER TABLE context_items RENAME COLUMN value TO content")
                log.info("Renamed context_items.value → content")

            # Add long_desc if missing, backfill from content
            if "long_desc" not in col_names:
                await conn.execute(
                    "ALTER TABLE context_items ADD COLUMN IF NOT EXISTS long_desc TEXT NOT NULL DEFAULT ''"
                )
                await conn.execute("UPDATE context_items SET long_desc = LEFT(content, 1000) WHERE long_desc = ''")
                log.info("Added context_items.long_desc, backfilled from content")

            # Drop legacy check constraint (allow any context_type)
            await conn.execute("ALTER TABLE context_items DROP CONSTRAINT IF EXISTS ck_context_items_type")

    log.info("context_items schema migration complete")
    return True
