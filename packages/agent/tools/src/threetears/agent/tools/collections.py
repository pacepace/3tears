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
from typing import Any, cast
from uuid import UUID

from sqlalchemy import MetaData, Table

from threetears.core.collections.schema_backed import (
    DATETIMETZ_TYPE,
    JSONB_TYPE,
    STRING_TYPE,
    UUID_TYPE,
    Column,
    ForeignKey as SchemaForeignKey,
    Index as SchemaIndex,
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

# context_types that have a ``(conversation_id, key)`` partial unique
# index and may therefore be upserted on key. ``variable`` ships from
# v003; ``tool_result`` from v004. Membership is checked before the
# value reaches the literal-only ON CONFLICT predicate in
# :meth:`ContextItemCollection._upsert_keyed`.
_UPSERTABLE_CONTEXT_TYPES: frozenset[str] = frozenset({"variable", "tool_result"})


def context_items_table(metadata: MetaData) -> Table:
    """register the ``context_items`` table on the given SA metadata.

    v0.8.0: schema declaration is now the single source of truth. This
    factory is a thin idempotency wrapper around
    :meth:`ContextItemCollection.schema.to_sqlalchemy_table`. Call this
    before ``SQLiteBackend.initialize(metadata)`` so the L1 cache gets
    the correct schema. composite primary key on ``(conversation_id,
    context_id)`` mirrors the L3 partition layout so cache rows are
    isolated per partition.

    :param metadata: SQLAlchemy metadata to attach the table to
    :ptype metadata: MetaData
    :return: the ``context_items`` :class:`Table`
    :rtype: Table
    """
    return cast(Table, ContextItemCollection.schema.to_sqlalchemy_table(metadata))


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
    # v0.8.0 enrichment: ``long_desc`` carries ``server_default="''"``
    # to match prod (the v001 migration declared this default; prod
    # ``information_schema`` confirms). ``long_desc`` is also
    # non-nullable in prod (NULL is the default value the server
    # substitutes when the caller omits it). The FK on
    # ``conversation_id`` matches prod's
    # ``fk_context_items_conversation`` (CASCADE on parent
    # conversation delete) -- declared at table level because the
    # inline 2-tuple form does not carry ``on_delete=``. Indexes
    # mirror the v001 migration + the ``ix_context_items_var_key``
    # partial-unique that ``upsert_variable`` requires for its
    # ``ON CONFLICT (conversation_id, key) WHERE context_type =
    # 'variable'`` clause + the ``ix_context_items_lru`` LRU index
    # that ``evict_lru`` reads.
    schema = TableSchema(
        name="context_items",
        primary_key=("conversation_id", "context_id"),
        columns=[
            Column("conversation_id", UUID_TYPE, partition=True),
            Column("context_id", UUID_TYPE),
            Column("context_type", STRING_TYPE, immutable=True),
            Column("key", STRING_TYPE, immutable=True),
            Column("short_desc", STRING_TYPE),
            Column("long_desc", STRING_TYPE, server_default="''::text"),
            Column("content", STRING_TYPE),
            Column("metadata", JSONB_TYPE, nullable=True),
            Column("date_accessed", DATETIMETZ_TYPE),
            Column("date_created", DATETIMETZ_TYPE, immutable=True),
            Column("date_updated", DATETIMETZ_TYPE),
        ],
        cas_column="date_updated",
        foreign_keys=(
            SchemaForeignKey(
                "conversation_id",
                "conversations",
                "conversation_id",
                on_delete="CASCADE",
            ),
        ),
        indexes=(
            SchemaIndex("ix_context_items_conv", "conversation_id"),
            SchemaIndex(
                "ix_context_items_type",
                "conversation_id",
                "context_type",
            ),
            SchemaIndex(
                "ix_context_items_lru",
                "conversation_id",
                "date_accessed",
            ),
            SchemaIndex(
                "ix_context_items_var_key",
                "conversation_id",
                "key",
                unique=True,
                where="context_type = 'variable'",
            ),
            # Dedup index for tool_results (v004 migration). Mirrors the
            # variable key index so upsert_tool_result's ON CONFLICT has a
            # matching partial-unique index.
            SchemaIndex(
                "ix_context_items_tool_result_key",
                "conversation_id",
                "key",
                unique=True,
                where="context_type = 'tool_result'",
            ),
        ),
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

    async def _upsert_keyed(
        self,
        conversation_id: UUID,
        context_type: str,
        data: dict[str, Any],
    ) -> UUID:
        """upsert a keyed context item via its partial unique index.

        Shared codepath for :meth:`upsert_variable` and
        :meth:`upsert_tool_result`. ``context_type`` selects the matching
        ``(conversation_id, key) WHERE context_type = '<type>'`` partial
        unique index, so a key collision refreshes the existing row's
        content / metadata / ``date_accessed`` instead of inserting a
        duplicate, and returns that row's id. ``context_type`` is checked
        against a fixed allow-list before it reaches the (literal-only)
        ON CONFLICT predicate, so it can never carry SQL.

        :param conversation_id: conversation partition the item lives in
        :ptype conversation_id: UUID
        :param context_type: ``"variable"`` or ``"tool_result"``
        :ptype context_type: str
        :param data: row dict keyed by column name
        :ptype data: dict[str, Any]
        :return: authoritative context_id (existing on conflict, new on insert)
        :rtype: UUID
        :raises ValueError: when ``context_type`` has no dedup index
        """
        if context_type not in _UPSERTABLE_CONTEXT_TYPES:
            msg = f"context_type must be one of {sorted(_UPSERTABLE_CONTEXT_TYPES)}; got {context_type!r}"
            raise ValueError(msg)

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
            f"""
            INSERT INTO context_items (
                context_id, conversation_id, context_type, key,
                short_desc, long_desc, content, metadata,
                date_accessed, date_created, date_updated
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9, $10, $11)
            ON CONFLICT (conversation_id, key) WHERE context_type = '{context_type}'
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
            context_type,
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
        return await self._upsert_keyed(conversation_id, "variable", data)

    async def upsert_tool_result(
        self,
        conversation_id: UUID,
        data: dict[str, Any],
    ) -> UUID:
        """upsert a tool_result using the partial unique index.

        Mirrors :meth:`upsert_variable` for ``context_type =
        'tool_result'`` (the ``ix_context_items_tool_result_key`` partial
        unique index, agent-tools v004). The caller keys the row
        ``tool_name + ':' + hash(input)`` so a repeat call with the same
        input refreshes the existing row (bumping ``date_accessed`` back
        to the top of the LRU) instead of appending a duplicate -- the
        dedup half of the context-bloat fix.

        :param conversation_id: conversation partition the result lives in
        :ptype conversation_id: UUID
        :param data: row dict keyed by column name (``key`` already
            encodes tool name + input hash)
        :ptype data: dict[str, Any]
        :return: authoritative context_id (existing on conflict, new on insert)
        :rtype: UUID
        """
        return await self._upsert_keyed(conversation_id, "tool_result", data)

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
                "UPDATE context_items SET date_accessed = $3 WHERE conversation_id = $1 AND context_id = $2",
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
