"""
FolderCollection -- three-tier CRUD for :class:`Folder`.

mirrors :class:`~threetears.conversations.collection.ConversationsCollection`:
L1 SQLite (pod-local) in front of L2 NATS KV in front of L3
YugabyteDB. the collection is agent-scoped; the underlying asyncpg
pool is expected to have ``search_path`` already set to the per-agent
schema by the L3 broker before the collection is constructed.

CRUD is generated from :attr:`FolderCollection.schema` via
:class:`SchemaBackedCollection`. the CAS-fenced UPDATE path uses the
``date_updated`` column so concurrent writers race correctly rather
than silently overwriting each other; the insert path is an
``INSERT ... ON CONFLICT DO UPDATE`` upsert so re-ingest of a known
folder is safe.

the folder is the app-agnostic peer of the conversation: it groups
conversations under a per-owner named container. app-specific bits
(color, sort order, icon) live in the ``metadata`` JSONB blob so the
canonical shape never grows per-product columns.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from threetears.conversations.folder_entity import Folder
from threetears.core.collections.flush import WriteBuffer
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import (
    DATETIMETZ_TYPE,
    JSONB_TYPE,
    STRING_TYPE,
    UUID_TYPE,
    Column,
    Index as SchemaIndex,
    SchemaBackedCollection,
    TableSchema,
)
from threetears.core.config import CoreConfig
from threetears.observe import get_logger

__all__ = [
    "FolderCollection",
]

log = get_logger(__name__)


class FolderCollection(SchemaBackedCollection[Folder]):
    """three-tier collection for :class:`Folder` entities.

    the collection is the sole writer to the ``folders`` table:
    consumer apps create / rename / delete folders through it and read
    them back via :meth:`find_by_user`. inserts are upserts keyed on
    the composite ``(agent_id, folder_id)`` pk so re-ingest is safe.
    CRUD comes from the declarative :class:`TableSchema`; the domain
    query :meth:`find_by_user` stays on the subclass because its
    filtered SELECT/ORDER is per-collection.

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

    # composite-pk partition on ``agent_id`` (matches the conversations
    # table). ``customer_id`` / ``user_id`` / ``date_created`` are
    # immutable scope/identity columns; ``name`` / ``metadata`` /
    # ``date_updated`` are mutable. app-specific presentation bits
    # (color, sort_order, icon) deliberately stay out of the column set
    # and live in ``metadata`` so the canonical shape is app-agnostic.
    primary_key_column: str | tuple[str, ...] = ("agent_id", "folder_id")
    schema = TableSchema(
        name="folders",
        primary_key=("agent_id", "folder_id"),
        columns=[
            Column("agent_id", UUID_TYPE, partition=True),
            Column("folder_id", UUID_TYPE),
            Column("customer_id", UUID_TYPE, immutable=True),
            Column("user_id", UUID_TYPE, immutable=True),
            Column("name", STRING_TYPE),
            Column("metadata", JSONB_TYPE, nullable=True),
            Column("date_created", DATETIMETZ_TYPE, immutable=True),
            Column("date_updated", DATETIMETZ_TYPE),
        ],
        cas_column="date_updated",
        indexes=(SchemaIndex("idx_folders_user", "agent_id", "user_id"),),
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

        :return: ``"folders"``
        :rtype: str
        """
        return "folders"

    @property
    def entity_class(self) -> type[Folder]:
        """return the entity class this collection produces.

        :return: :class:`Folder`
        :rtype: type[Folder]
        """
        return Folder

    async def find_by_user(
        self,
        agent_id: UUID,
        user_id: UUID,
    ) -> list[Folder]:
        """fetch every folder owned by the given user under one agent.

        results come from L3 (the source of truth for historical rows)
        and are promoted into L2 so subsequent reads hit the cache
        tier. ordering is by ``name`` ascending -- a stable, generic
        default; consumers that present folders in a custom order (e.g.
        a ``sort_order`` carried in ``metadata``) re-sort the returned
        list themselves. ``agent_id`` is the partition column on the
        ``folders`` table; the caller supplies it explicitly so the
        lookup stays inside one agent's data slice and the partition
        predicate is enforced at the SQL boundary.

        :param agent_id: agent partition the folders belong to
        :ptype agent_id: UUID
        :param user_id: user whose folders to fetch
        :ptype user_id: UUID
        :return: folders owned by ``user_id`` under ``agent_id``,
            ordered by name
        :rtype: list[Folder]
        """
        rows = await self.l3_pool.fetch(
            "SELECT * FROM folders WHERE agent_id = $1 AND user_id = $2 ORDER BY name ASC",
            agent_id,
            user_id,
        )
        entities: list[Folder] = []
        for row in rows:
            data = self._coerce_row(dict(row))
            entity = self.entity_class(data, is_new=False, collection=self)
            entity.original_date_updated = data.get("date_updated")
            pk = (data["agent_id"], data["folder_id"])
            await self._save_to_l2(pk, data)
            entities.append(entity)
        return entities
