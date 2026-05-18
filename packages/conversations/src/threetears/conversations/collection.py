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

from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar
from uuid import UUID

from threetears.conversations.entity import Conversation
from threetears.core.collections.flush import WriteBuffer
from threetears.core.collections.registry import CollectionRegistry

if TYPE_CHECKING:
    from threetears.conversations.buffer import ConversationWriteBuffer
from threetears.core.collections.schema_backed import (
    DATETIMETZ_TYPE,
    INT_TYPE,
    JSONB_TYPE,
    STRING_TYPE,
    TSVECTOR_TYPE,
    UUID_TYPE,
    Column,
    Index as SchemaIndex,
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

    # rationale: ``attach_write_buffer`` wires a back-reference to a
    # pre-constructed ``ConversationWriteBuffer``; it neither reads
    # nor writes table rows, so the partition-column gate has nothing
    # to enforce.
    _partition_exempt_methods: ClassVar[frozenset[str]] = frozenset(
        {"attach_write_buffer"},
    )
    # v0.8.0 hygiene enrichment: ``search_vector`` (TSVECTOR,
    # immutable, trigger-maintained per v005 migration);
    # ``language`` server default ``'english'`` matches v006
    # migration. Indexes mirror the v001 / v005 migrations:
    # ``idx_conv_user`` / ``idx_conv_customer`` (composite by
    # date_created) + ``idx_conv_status`` + ``idx_conversations_search_vector``
    # (GIN -- can't be expressed in v0.8.0 IndexDef, kept Alembic-side
    # for now). Standard btree indexes are declared here.
    # v0.8.0 shard 04.6: the bare-``id`` PK column was renamed to
    # ``conversation_id`` to standardize on ``<entity>_id`` across all
    # entity tables (matches metallm prod + JSON API contract).
    primary_key_column: str | tuple[str, ...] = ("agent_id", "conversation_id")
    schema = TableSchema(
        name="conversations",
        primary_key=("agent_id", "conversation_id"),
        columns=[
            Column("agent_id", UUID_TYPE, partition=True),
            Column("conversation_id", UUID_TYPE),
            Column("customer_id", UUID_TYPE, immutable=True),
            Column("user_id", UUID_TYPE, immutable=True),
            Column("channel_type", STRING_TYPE, immutable=True),
            Column("conversation_ref", STRING_TYPE, nullable=True, immutable=True),
            Column("name", STRING_TYPE, nullable=True),
            Column("status", STRING_TYPE),
            Column("summary", STRING_TYPE, nullable=True),
            Column(
                "search_vector",
                TSVECTOR_TYPE,
                nullable=True,
                immutable=True,
            ),
            Column("date_created", DATETIMETZ_TYPE, immutable=True),
            Column("date_updated", DATETIMETZ_TYPE),
            Column("date_last_message", DATETIMETZ_TYPE, nullable=True),
            Column("metadata", JSONB_TYPE, nullable=True),
            Column("message_count", INT_TYPE),
            # v006: postgres FTS tokenizer config for the conversation
            # name (and any future name-derived tsvector signals).
            # Default ``'english'``; valid values are any installed
            # ``pg_ts_config`` entry (``simple``, ``spanish``, ``french``,
            # ``german``, etc.). The trigger function rebuilds the
            # search_vector on UPDATE OF this column too, so flipping
            # language re-tokenizes lazily.
            Column("language", STRING_TYPE, server_default="'english'::text"),
        ],
        cas_column="date_updated",
        indexes=(
            SchemaIndex("idx_conv_user", "user_id", "date_created"),
            SchemaIndex(
                "idx_conv_customer",
                "customer_id",
                "date_created",
            ),
            SchemaIndex("idx_conv_status", "status"),
        ),
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
        # ConversationWriteBuffer for cross-conversation pod-wide
        # batched writes (data-layer-task-01 sub-task 3). attached
        # lazily by callers via :meth:`attach_write_buffer`; remains
        # ``None`` in pure-CRUD test harnesses that do not need
        # batching. distinct from BaseCollection._write_buffer (the
        # generic flush primitive); this attribute is the optional
        # cross-conversation batcher.
        self._conversation_write_buffer: ConversationWriteBuffer | None = None

    def attach_write_buffer(
        self,
        buffer: ConversationWriteBuffer,
    ) -> None:
        """attach a :class:`ConversationWriteBuffer` for delegated batching.

        callers that constructed the collection then constructed the
        buffer (because the buffer takes the collection in its
        constructor) wire the back-reference here so
        :meth:`enqueue_message_recorded` has a target. constructor
        circular-init avoidance.

        :param buffer: cross-conversation write buffer
        :ptype buffer: ConversationWriteBuffer
        :return: nothing
        :rtype: None
        """
        self._conversation_write_buffer = buffer

    async def enqueue_message_recorded(
        self,
        *,
        agent_id: UUID,
        conversation_id: UUID,
        at: datetime,
        role: str,
    ) -> None:
        """delegate one observed message to the attached write buffer.

        no-op when no buffer is attached. the buffer accumulates
        deltas across every conversation the pod is multiplexing and
        flushes opportunistically (timer / threshold / shutdown);
        callers that need an immediate write should construct the
        entity, call :meth:`Conversation.record_message`, and call
        :meth:`save_entity` directly.

        :param agent_id: agent partition the conversation lives in
        :ptype agent_id: UUID
        :param conversation_id: conversation UUID
        :ptype conversation_id: UUID
        :param at: timestamp the message was observed at
        :ptype at: datetime
        :param role: short actor token (``user`` / ``assistant`` / ...)
        :ptype role: str
        :return: nothing
        :rtype: None
        """
        if self._conversation_write_buffer is None:
            return
        await self._conversation_write_buffer.enqueue(
            agent_id=agent_id,
            conversation_id=conversation_id,
            at=at,
            role=role,
        )

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

    async def search(
        self,
        agent_id: UUID,
        user_id: UUID,
        query: str,
        *,
        limit: int = 20,
        offset: int = 0,
        include_closed: bool = False,
        query_language: str = "english",
    ) -> list[Conversation]:
        """full-text search across conversations the user participates in.

        delegates to postgres' ``search_vector @@ websearch_to_tsquery``
        on the ``conversations.search_vector`` column populated by the
        v005 trigger. results come from L3 ordered by ``ts_rank_cd``
        descending (most relevant first) with ``date_updated`` desc as
        a stable secondary order.

        the FTS query uses ``websearch_to_tsquery`` (not
        ``plainto_tsquery`` or ``to_tsquery``) so user-typed input
        survives the boundary without breaking on operators -- spaces
        become AND, ``OR`` becomes OR, leading ``-`` becomes NOT,
        double-quoted phrases become phrase queries. matches postgres'
        documented "search-engine-style" parser. queries shorter than
        the minimum useful tokenization length (2 chars) return empty;
        callers that want substring matching on shorter inputs use
        their own ILIKE fallback.

        scope: the WHERE clause pins ``user_id = $2`` so cross-user
        leakage is impossible at the SQL boundary. ``agent_id`` is the
        partition column on ``conversations``; supplying it explicitly
        keeps the lookup inside one agent's slice (matches
        :meth:`find_by_user`). closed conversations are excluded by
        default to match the typical "search my open chats" UX; pass
        ``include_closed=True`` to widen the scan.

        product consumers that need a wider search (also FTS into
        message bodies, hybrid with vector similarity, etc.) compose
        this method's result with their own per-product joins -- 3tears
        does not pin a canonical ``messages`` table so the framework
        cannot do the join itself.

        :param agent_id: agent partition the conversations belong to
        :ptype agent_id: UUID
        :param user_id: user whose conversations to scope the search to
        :ptype user_id: UUID
        :param query: free-text search query (passed to
            ``websearch_to_tsquery``)
        :ptype query: str
        :param limit: maximum rows to return (default 20)
        :ptype limit: int
        :param offset: pagination offset (default 0)
        :ptype offset: int
        :param include_closed: include closed / archived conversations
            in the search scope (default False)
        :ptype include_closed: bool
        :param query_language: postgres FTS config name used to
            tokenize the query (``'english'``, ``'spanish'``, ``'french'``,
            etc.; default ``'english'``). The conversations' stored
            search_vector is tokenized per-row via the language column
            (v006 migration). For mono-language deployments leave the
            default and both sides match. For polyglot deployments the
            caller passes the user's preferred language; matches
            against same-language rows are precise, cross-language
            matches are tokenized differently and may degrade in
            quality but don't crash. The parameter is intentionally
            not validated against ``pg_ts_config`` here -- a typo
            surfaces as a clean postgres error rather than a silent
            wrong-tokenization match
        :ptype query_language: str
        :return: matching conversations, most-relevant-first
        :rtype: list[Conversation]
        """
        # Defensive: an empty / whitespace-only query would match every
        # row via FTS' empty-query semantics. Return [] explicitly so
        # callers do not accidentally surface every conversation.
        if not query or not query.strip():
            return []

        if include_closed:
            status_predicate = ""
            params: tuple[Any, ...] = (
                agent_id,
                user_id,
                query,
                limit,
                offset,
                query_language,
            )
        else:
            status_predicate = " AND status != $7"
            params = (
                agent_id,
                user_id,
                query,
                limit,
                offset,
                query_language,
                "closed",
            )

        rows = await self.l3_pool.fetch(
            "SELECT *, ts_rank_cd(search_vector, websearch_to_tsquery($6::regconfig, $3)) AS rank "
            "FROM conversations "
            "WHERE agent_id = $1 "
            "  AND user_id = $2 "
            "  AND search_vector @@ websearch_to_tsquery($6::regconfig, $3)"
            + status_predicate
            + " ORDER BY rank DESC, date_updated DESC "
            "LIMIT $4 OFFSET $5",
            *params,
        )
        entities: list[Conversation] = []
        for row in rows:
            data = self._coerce_row(dict(row))
            # ``rank`` is a derived projection from the SELECT, not a
            # column on the entity -- strip it before the entity builds
            # itself so a future entity-level audit doesn't think
            # ``rank`` is supposed to be there.
            data.pop("rank", None)
            entity = self.entity_class(data, is_new=False, collection=self)
            entity.original_date_updated = data.get("date_updated")
            pk = (data["agent_id"], data["conversation_id"])
            # Populate L2 only, not L1. Search results are derived
            # (caller's intent is "find conversations matching X",
            # not "warm the L1 row cache"), and the typical follow-up
            # is the user clicking through to one specific result --
            # a single L1 miss + L2 pull-through is the right cost
            # profile for that access pattern. Filling L1 with every
            # search result would evict actively-used rows for cold
            # data the user may never revisit. Callers that need an
            # L1-hot row for one specific result do ``get(pk)`` after
            # picking from the search hits, which warms L1 lazily.
            await self._save_to_l2(pk, data)
            entities.append(entity)
        return entities

    async def find_by_user(
        self,
        agent_id: UUID,
        user_id: UUID,
        include_closed: bool = False,
    ) -> list[Conversation]:
        """fetch every conversation owned by the given user under one agent.

        results come from L3 (the source of truth for historical rows)
        and are promoted into L2 so subsequent reads hit the cache
        tier. ordering is newest-first. ``agent_id`` is the partition
        column on the ``conversations`` table; the caller supplies it
        explicitly so the lookup stays inside one agent's data slice
        and the partition predicate is enforced at the SQL boundary.

        :param agent_id: agent partition the conversations belong to
        :ptype agent_id: UUID
        :param user_id: user whose conversations to fetch
        :ptype user_id: UUID
        :param include_closed: include closed / archived conversations
        :ptype include_closed: bool
        :return: conversations owned by ``user_id`` under ``agent_id``
        :rtype: list[Conversation]
        """
        if include_closed:
            rows = await self.l3_pool.fetch(
                "SELECT * FROM conversations WHERE agent_id = $1 AND user_id = $2 ORDER BY date_created DESC",
                agent_id,
                user_id,
            )
        else:
            rows = await self.l3_pool.fetch(
                "SELECT * FROM conversations "
                "WHERE agent_id = $1 AND user_id = $2 AND status != $3 "
                "ORDER BY date_created DESC",
                agent_id,
                user_id,
                "closed",
            )
        entities: list[Conversation] = []
        for row in rows:
            data = self._coerce_row(dict(row))
            entity = self.entity_class(data, is_new=False, collection=self)
            entity.original_date_updated = data.get("date_updated")
            pk = (data["agent_id"], data["conversation_id"])
            await self._save_to_l2(pk, data)
            entities.append(entity)
        return entities
