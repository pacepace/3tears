"""
ConversationsCollection -- three-tier CRUD for :class:`Conversation`.

mirrors the shape of :class:`~threetears.agent.memory.collections.
MemoriesCollection`: L1 SQLite (pod-local) in front of L2 NATS KV in
front of L3 YugabyteDB, with a :class:`WriteBuffer` batching writes.
the collection is agent-scoped; the underlying asyncpg pool is
expected to have ``search_path`` already set to the per-agent
schema by the L3 broker before the collection is constructed.

CRUD is generated from :attr:`ConversationsCollection.schema` via
:class:`SchemaBackedCollection`. the CAS-fenced UPDATE path uses the
``date_updated`` column so concurrent writers race correctly rather
than silently overwriting each other; the insert path is a
``INSERT ... ON CONFLICT (id) DO UPDATE`` upsert so re-ingest of a
known conversation is safe.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from threetears.conversations.entity import Conversation
from threetears.core.collections.flush import WriteBuffer
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import (
    DATETIME_TYPE,
    JSONB_TYPE,
    STRING_TYPE,
    UUID_TYPE,
    Column,
    SchemaBackedCollection,
    TableSchema,
)
from threetears.core.config import CoreConfig
from threetears.observe import get_logger

__all__ = [
    "ConversationsCollection",
]

log = get_logger(__name__)


class ConversationsCollection(SchemaBackedCollection[Conversation]):
    """three-tier collection for :class:`Conversation` entities.

    the collection is the sole writer to the ``conversations`` table:
    memory and agent-tools packages read conversations through it (or
    via their own foreign-key joins) but never mutate the table
    directly. inserts are upserts keyed on ``id`` so re-ingest is
    safe. CRUD comes from the declarative :class:`TableSchema`;
    domain query :meth:`find_by_user` stays on the subclass because
    its filtered SELECT/ORDER is per-collection.

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

    primary_key_column: str = "id"
    schema = TableSchema(
        name="conversations",
        primary_key="id",
        columns=[
            Column("id", UUID_TYPE),
            Column("agent_id", UUID_TYPE, immutable=True),
            Column("customer_id", UUID_TYPE, immutable=True),
            Column("user_id", UUID_TYPE, immutable=True),
            Column("channel_type", STRING_TYPE, immutable=True),
            Column("conversation_ref", STRING_TYPE, nullable=True, immutable=True),
            Column("status", STRING_TYPE),
            Column("summary", STRING_TYPE, nullable=True),
            Column("date_created", DATETIME_TYPE, immutable=True),
            Column("date_updated", DATETIME_TYPE),
            Column("date_last_message", DATETIME_TYPE, nullable=True),
            Column("metadata", JSONB_TYPE, nullable=True),
        ],
        cas_column="date_updated",
    )

    def __init__(
        self,
        registry: CollectionRegistry,
        config: CoreConfig,
        postgres_pool: Any,
        nats_client: Any = None,
        write_buffer: WriteBuffer | None = None,
    ) -> None:
        """initialize the collection and register it with the registry.

        the ``postgres_pool`` kwarg is stored onto ``self.l3_pool`` so
        the generic CRUD path finds the pool uniformly with siblings
        that resolve the pool through the registry.

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
        super().__init__(registry, config, nats_client, write_buffer)
        self.l3_pool = postgres_pool

    @property
    def table_name(self) -> str:
        """return the table name for this collection.

        :return: ``"conversations"``
        :rtype: str
        """
        return "conversations"

    @property
    def entity_class(self) -> type[Conversation]:
        """return the entity class this collection produces.

        :return: :class:`Conversation`
        :rtype: type[Conversation]
        """
        return Conversation

    async def find_by_user(
        self, user_id: UUID, include_closed: bool = False,
    ) -> list[Conversation]:
        """fetch every conversation owned by the given user.

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
            rows = await self.l3_pool.fetch(
                "SELECT * FROM conversations WHERE user_id = $1 "
                "ORDER BY date_created DESC",
                user_id,
            )
        else:
            rows = await self.l3_pool.fetch(
                "SELECT * FROM conversations WHERE user_id = $1 AND status != $2 "
                "ORDER BY date_created DESC",
                user_id,
                "closed",
            )
        entities: list[Conversation] = []
        for row in rows:
            data = self._coerce_row(dict(row))
            entity = self.entity_class(data, is_new=False, collection=self)
            entity.original_date_updated = data.get("date_updated")
            entity_id = data["id"]
            await self._save_to_l2(entity_id, data)
            entities.append(entity)
        return entities
