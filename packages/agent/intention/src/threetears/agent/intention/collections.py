"""Intentions collection -- three-tier CRUD for standing-want records.

:class:`IntentionsCollection` is the single entry point for
``intentions``-table SQL. CRUD is generated from :attr:`schema` and goes
through :meth:`get` / :meth:`save_entity` / :meth:`delete` so the L1 / L2
/ L3 tiers stay coherent; ``date_updated`` is the CAS fence so concurrent
writers race correctly.

User isolation is a ``user_id`` WHERE clause, NOT RBAC: every metallm
user shares one ``agent_id`` (the partition), so the partition isolates
nothing and the agent-owner RBAC short-circuit would see every user's
wants. The user-facing read :meth:`find_by_user` therefore takes
``user_id`` as a **required** parameter and filters on it (mirroring
:meth:`MemoriesCollection.find_by_user`).
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, ClassVar, Final, cast
from uuid import UUID

from sqlalchemy import MetaData, Table

from threetears.core.collections.salience import SalienceDecay
from threetears.core.collections.schema_backed import (
    DATETIMETZ_TYPE,
    ENUM_TYPE,
    NUMERIC_TYPE,
    STRING_TYPE,
    UUID_TYPE,
    VECTOR_TYPE,
    Column,
    Index as SchemaIndex,
    SchemaBackedCollection,
    TableSchema,
)
from threetears.core.data.embedding_width import DEFAULT_EMBEDDING_DIM, schema_with_embedding_dim
from threetears.core.keyed import AnyOf, Fragment, Nearest, Selected, Sort, read_by, read_nearest, update_keyed
from threetears.observe import get_logger

from threetears.agent.intention.entities import IntentionEntity
from threetears.agent.intention.types import INTENTION_STATUS_VALUES, IntentionStatus
from threetears.core.tenancy import TENANT_BY_SCHEMA
from threetears.core.eviction import Declared, Policy

__all__ = [
    "IntentionsCollection",
    "intentions_table",
]

log = get_logger(__name__)


# Embedding dimension the intentions table is declared with: the default of
# the deployment setting, the width agent-memory defaults to as well, so a
# shared embedding provider serves both corpora. The collection and table
# factory take ``embedding_dim`` and derive their schema at that width.
_INTENTION_VECTOR_DIM = DEFAULT_EMBEDDING_DIM


# Explicit column list for multi-row reads: the pgvector ``embedding``
# column is cast ``::text`` so asyncpg returns it without a registered
# vector codec (dedup/semantic paths, added in B2, decode it explicitly).
#: the columns each read of whole wants selects, the embedding as its text
_INTENTION_COLUMNS: Final = (
    "intention_id",
    "agent_id",
    "customer_id",
    "user_id",
    "status",
    "content",
    Selected("t.embedding::text", "embedding"),
    "salience",
    "last_decayed_at",
    "last_surfaced_at",
    "source_memory_id",
    "source_conversation_id",
    "date_created",
    "date_updated",
)


def intentions_table(metadata: MetaData, *, embedding_dim: int = DEFAULT_EMBEDDING_DIM) -> Table:
    """Register the ``intentions`` table on the given SA metadata.

    Thin idempotency wrapper around
    :meth:`IntentionsCollection.schema.to_sqlalchemy_table`. Call before
    ``SQLiteBackend.initialize(metadata)`` so the L1 cache builds with
    the full schema, and before Alembic ``target_metadata`` reflection so
    auto-generate sees the same shape.

    :param metadata: SQLAlchemy metadata to attach the table to
    :ptype metadata: MetaData
    :param embedding_dim: the ``embedding`` column's width, the deployment's; default 1024
    :ptype embedding_dim: int
    :return: the ``intentions`` :class:`Table`
    :rtype: Table
    """
    schema = schema_with_embedding_dim(IntentionsCollection.schema, embedding_dim)
    return cast(Table, schema.to_sqlalchemy_table(metadata))


class IntentionsCollection(SchemaBackedCollection[IntentionEntity]):
    """Collection for standing-want entities with three-tier caching.

    CRUD is generated from :attr:`schema`: ``embedding`` is
    ``VECTOR_TYPE`` (pgvector), ``date_updated`` is the CAS fence, and
    the scope + provenance columns (agent/customer/date_created) are
    marked immutable so the ``DO UPDATE SET`` clause narrows to the
    mutable want fields (``status`` / ``content`` / ``salience`` /
    ``embedding`` / the decay + cooldown anchors).

    :meth:`find_by_user` carries a ``# cache-bypass:`` justification
    because the multi-row scan is not primary-key addressable; keeping it
    on the Collection preserves the single-entry-point contract.
    """

    tenancy = TENANT_BY_SCHEMA

    #: salience decay on read (:class:`~threetears.core.collections.salience.SalienceDecay`), off by default: with
    #: it, the ranked reads order by effective salience and a reinforcement stamps the decay so far into the base
    salience_decay: ClassVar[SalienceDecay | None] = None

    primary_key_column: str | tuple[str, ...] = ("agent_id", "intention_id")
    schema = TableSchema(
        name="intentions",
        primary_key=("agent_id", "intention_id"),
        columns=[
            Column("intention_id", UUID_TYPE),
            Column("agent_id", UUID_TYPE, partition=True),
            # customer_id / user_id are nullable scope grains (like memory
            # after v024). metallm enforces NOT NULL + the user_id filter
            # at its own consumer layer; a null here is an agent-internal
            # / global want.
            Column("customer_id", UUID_TYPE, immutable=True, nullable=True),
            # user_id is a soft ref (no FK): the primitive supports
            # agent-internal wants and deployments without a users table,
            # and avoids a cross-package teardown-order constraint.
            # Isolation is the WHERE clause on this column, not a FK.
            # ``immutable=True`` (matches memory's user_id): user_id is the
            # SOLE user-isolation boundary (agent_id is shared across all
            # metallm users), so it is write-once -- excluded from the
            # entity-UPDATE generator so a stray setter can never move a
            # want across users.
            Column("user_id", UUID_TYPE, nullable=True, immutable=True),
            # a fresh PG enum -- no shared-memory_type ALTER pain. Default
            # 'open' on log; mutable as the want walks its lifecycle.
            Column(
                "status",
                ENUM_TYPE,
                enum_type=INTENTION_STATUS_VALUES,
                enum_name="intention_status",
                server_default=f"'{IntentionStatus.OPEN.value}'",
            ),
            Column("content", STRING_TYPE),
            # dedup on log + future semantic recall; nullable until embedded.
            Column(
                "embedding",
                VECTOR_TYPE,
                vector_dim=_INTENTION_VECTOR_DIM,
                nullable=True,
            ),
            # reuses the memory decay substrate: NUMERIC(5,4) seeded 0.5.
            # ``immutable=True`` (mirrors memory): salience is written ONLY
            # by the raw decay / bump passes (bump_salience; decay is read, SalienceDecay),
            # never by a full-entity save -- so a mark_surfaced / refresh save
            # can't carry a stale cached salience back and revert a scheduled
            # decay. Insert still applies the value / server default.
            Column(
                "salience",
                NUMERIC_TYPE,
                precision=5,
                scale=4,
                nullable=False,
                server_default="0.5",
                immutable=True,
            ),
            # decay anchor: age is measured from the last decay run so
            # total decay over a period is cadence-safe. ``immutable=True``:
            # written only by the raw decay pass, excluded from the entity
            # UPDATE so a save can't roll the anchor back (which would make
            # the next pass double-count the elapsed age).
            Column("last_decayed_at", DATETIMETZ_TYPE, nullable=True, immutable=True),
            # cooldown anchor: the read-path filter excludes a want
            # surfaced within intention_cooldown_days (enforced in B2).
            Column("last_surfaced_at", DATETIMETZ_TYPE, nullable=True),
            # soft-ref provenance (no FK): where the want came from.
            Column("source_memory_id", UUID_TYPE, immutable=True, nullable=True),
            Column("source_conversation_id", UUID_TYPE, immutable=True, nullable=True),
            Column("date_created", DATETIMETZ_TYPE, immutable=True),
            Column("date_updated", DATETIMETZ_TYPE, nullable=True),
        ],
        cas_column="date_updated",
        indexes=(
            # the deliberation hot path: rank a user's open wants by
            # salience. Partial on status='open' so the index stays small.
            SchemaIndex(
                "idx_intentions_open_ranked",
                "agent_id",
                "user_id",
                "salience",
                where="status = 'open'",
                desc="salience",
            ),
            # a person's every want, whatever its status (v003): find_by_user, find_similar_for_dedup
            SchemaIndex("ix_intentions_agent_user", "agent_id", "user_id"),
            # HNSW over the embedding for the log-time dedup lookup (since v0.15.0). KEEP (coordinator, 2026-11-14, on
            # that history reason): an owner with many open wants is read through it, a small one is sorted exactly
            # off idx_intentions_open_ranked; a platform pool's iterative scan keeps the small owner's k rows.
            SchemaIndex(
                "ix_intentions_embedding_hnsw",
                "embedding",
                using="hnsw",
                ops={"embedding": "vector_cosine_ops"},
                purpose=(
                    "packages/agent/intention/src/threetears/agent/intention/collections.py"
                    "::IntentionsCollection.find_similar_for_dedup"
                ),
                plan_test=(
                    "packages/agent/intention/tests/integration/test_index_plans.py"
                    "::test_ix_intentions_embedding_hnsw_serves_a_heavy_owners_dedup_read"
                ),
            ),
        ),
        l1_eviction=Declared(Policy.EVICTABLE, "rows cached on a read by key; a miss reads L2/L3 and refills"),
    )

    def __init__(self, *args: Any, embedding_dim: int = DEFAULT_EMBEDDING_DIM, **kwargs: Any) -> None:
        """construct the collection with its ``embedding`` column ``embedding_dim`` wide.

        :param args: positional args forwarded to :class:`SchemaBackedCollection`
            (``registry``, ``config``, ``nats_client``, ``write_buffer``)
        :ptype args: Any
        :param embedding_dim: the ``embedding`` column's width, the deployment's (default 1024,
            agent-memory's width too); a write carrying an embedding of another width is refused
            before any statement runs
        :ptype embedding_dim: int
        :param kwargs: keyword args forwarded to :class:`SchemaBackedCollection`
        :ptype kwargs: Any
        :return: nothing
        :rtype: None
        :raises ValueError: when ``embedding_dim`` is below 1 or above 16,000
        """
        self.schema = schema_with_embedding_dim(type(self).schema, embedding_dim)
        super().__init__(*args, **kwargs)

    @property
    def table_name(self) -> str:
        """Return the database table name for this collection.

        :return: table name
        :rtype: str
        """
        return "intentions"

    @property
    def entity_class(self) -> type[IntentionEntity]:
        """Return the entity class for this collection.

        :return: entity class
        :rtype: type[IntentionEntity]
        """
        return IntentionEntity

    async def find_by_user(
        self,
        user_id: UUID,
        *,
        agent_id: UUID,
    ) -> list[IntentionEntity]:
        """fetch every intention for ``(agent_id, user_id)`` from L3.

        ``agent_id`` is the partition column (required on every read);
        ``user_id`` is the isolation boundary and is a **required**
        parameter -- there is no user-agnostic list path, because every
        metallm user shares one ``agent_id``. Results are salience-ranked
        (the deliberation ordering), then most-recent-first.

        The status / cooldown filtering that ``intention_list`` layers on
        top lands in B2; this method is the isolation-enforcing substrate
        it builds on.

        :param user_id: owning user whose wants to fetch (row filter)
        :ptype user_id: UUID
        :param agent_id: partition column on intentions; required
        :ptype agent_id: UUID
        :return: the user's intention entities, salience-ranked
        :rtype: list[IntentionEntity]
        """
        if self.l3_pool is None:
            return []
        # cache-bypass: the multi-row scan by (agent_id, user_id) is not
        # primary-key addressable, so the L1 row cache would not serve it;
        # keeping it on the Collection preserves the single entry point.
        # the person's rows by ix_intentions_agent_user (agent_id, user_id); no index holds the rank (see
        # find_open_for_deliberation), so the sort is of that one person's wants
        rows = await read_by(
            self.l3_pool,
            "intentions",
            by={"agent_id": agent_id, "user_id": user_id},
            columns=_INTENTION_COLUMNS,
            order=self._ranked_order(),
            records=True,
        )
        entities: list[IntentionEntity] = []
        for row in rows:
            data = dict(row)
            # collection=self warms L1 + keeps the entity save-able (the
            # memory find_by_user template); a collection-less entity is
            # detached and B2's mark_surfaced/decay could not persist it.
            entity = self.entity_class(data, is_new=False, collection=self)
            entity.original_date_updated = data.get("date_updated")
            entities.append(entity)
        return entities

    async def find_open_for_deliberation(
        self,
        user_id: UUID,
        *,
        agent_id: UUID,
        cooldown_cutoff: datetime,
    ) -> list[IntentionEntity]:
        """fetch the user's deliberation candidate set (the ``intention_list`` substrate).

        Restraint brakes #1 (cooldown) and #2 (decay-sink) are enforced
        in the query, not by convention: the WHERE clause keeps only
        ``open`` wants that have not been surfaced within the cooldown
        window (``last_surfaced_at`` is null OR older than
        ``cooldown_cutoff``), and the ``salience DESC`` ordering sinks
        abandoned (decayed) wants to the bottom. ``user_id`` is the
        **required** isolation boundary (every metallm user shares one
        ``agent_id``); ``agent_id`` is the partition column.

        :param user_id: owning user whose open wants to rank (row filter)
        :ptype user_id: UUID
        :param agent_id: partition column on intentions; required
        :ptype agent_id: UUID
        :param cooldown_cutoff: exclude wants surfaced at/after this
            instant (``now - intention_cooldown_days``)
        :ptype cooldown_cutoff: datetime
        :return: open, outside-cooldown wants, salience-ranked
        :rtype: list[IntentionEntity]
        """
        if self.l3_pool is None:
            return []
        # cache-bypass: the ranked multi-row scan is not primary-key
        # addressable, so it stays on the Collection (single entry point).
        # WHY no index holds the order (coordinator ruling 2026-11-14): with
        # decay on, the rank is ln(salience) + ln2/half_life * epoch(stamp);
        # EXTRACT(EPOCH FROM timestamptz) is STABLE, not IMMUTABLE, and
        # half_life is per-class config, so neither a stored generated column
        # nor an expression index can hold it (only an app-maintained column
        # written on every salience write could). There is no LIMIT: the read
        # is one user's open wants, which idx_intentions_open_ranked's
        # (agent_id, user_id) prefix under status = 'open' bounds, so the sort
        # is of that small set alone.
        rows = await read_by(
            self.l3_pool,
            "intentions",
            # the partial index's predicate (status = 'open'), pinned, so the read states it
            by={"agent_id": agent_id, "user_id": user_id, "status": IntentionStatus.OPEN.value},
            columns=_INTENTION_COLUMNS,
            where=[Fragment("(t.last_surfaced_at IS NULL OR t.last_surfaced_at < $1)", (cooldown_cutoff,))],
            order=self._ranked_order(),
            records=True,
        )
        entities: list[IntentionEntity] = []
        for row in rows:
            data = dict(row)
            entity = self.entity_class(data, is_new=False, collection=self)
            entity.original_date_updated = data.get("date_updated")
            entities.append(entity)
        return entities

    async def find_similar_for_dedup(
        self,
        *,
        user_id: UUID,
        agent_id: UUID,
        embedding: list[float],
        top_k: int,
        threshold: float,
    ) -> list[dict[str, Any]]:
        """vector search for near-duplicate OPEN wants by embedding.

        Used by ``intention_log`` to refresh a near-duplicate open want
        instead of creating a second row for the same standing want.
        Only ``open`` wants are candidates -- a granted / dropped want
        must not suppress a fresh log of the same intent. Mirrors
        :meth:`MemoriesCollection.find_similar_for_dedup`; ``agent_id``
        is the partition column and ``user_id`` the isolation boundary,
        both required.

        :param user_id: owning user UUID (row filter)
        :ptype user_id: UUID
        :param agent_id: partition column on intentions; required
        :ptype agent_id: UUID
        :param embedding: query embedding vector
        :ptype embedding: list[float]
        :param top_k: maximum candidates to consider
        :ptype top_k: int
        :param threshold: minimum cosine similarity to surface
        :ptype threshold: float
        :return: list of ``{intention_id, content, salience, similarity}``
        :rtype: list[dict[str, Any]]
        """
        if self.l3_pool is None:
            return []
        # cache-bypass: vector-distance search is not primary-key
        # addressable; the L1 row cache cannot serve it. ``embedding IS
        # NOT NULL`` guards the ``float(similarity)`` cast (``NULL <=>
        # vector`` is NULL). Only OPEN wants are dedup candidates (the
        # status pin is written as its literal, 'open'). read_nearest
        # re-sorts the rows by distance: a platform pool's HNSW scan runs
        # in relaxed order (the comment at read_nearest_statement).
        rows = await read_nearest(
            self.l3_pool,
            "intentions",
            by={"agent_id": agent_id, "user_id": user_id, "status": IntentionStatus.OPEN.value},
            where=[Fragment("t.embedding IS NOT NULL", ())],
            nearest=Nearest("embedding", json.dumps(embedding), similarity="similarity"),
            limit=top_k,
            columns=("intention_id", "content", "salience"),
            records=True,
        )
        return [
            {
                "intention_id": row["intention_id"],
                "content": row["content"],
                "salience": float(row["salience"]),
                "similarity": float(row["similarity"]),
            }
            for row in rows
            if float(row["similarity"]) >= threshold
        ]

    async def bump_salience(
        self,
        intention_ids: list[UUID],
        *,
        agent_id: UUID,
        access_bump: float,
    ) -> None:
        """reinforce salience for re-logged wants (mirrors memory's raw bump).

        ``intention_log``'s dedup-refresh calls this to raise a
        near-duplicate open want's salience by ``access_bump`` (clamped to
        1.0). ``salience`` is ``immutable`` to the entity-UPDATE generator,
        so this raw pass is its only writer -- keeping decay and
        reinforcement on the same raw path means a full-entity save (which
        the refresh also does, for ``last_surfaced_at`` + ``embedding``)
        can never carry a stale salience back and revert a scheduled decay.

        Partition-scoped by ``agent_id``. Invalidates each bumped row's
        L1/L2 entry so a subsequent ``get()`` re-reads the fresh salience.

        :param intention_ids: primary-key ids to reinforce
        :ptype intention_ids: list[UUID]
        :param agent_id: partition column; required
        :ptype agent_id: UUID
        :param access_bump: increment added to salience (clamped to 1.0)
        :ptype access_bump: float
        :return: nothing
        :rtype: None
        """
        if self.l3_pool is None or not intention_ids:
            return None
        async with self.bypassing_write(*[(agent_id, intention_id) for intention_id in intention_ids]):
            await update_keyed(
                self.l3_pool,
                "intentions",
                values=self._reinforced(access_bump),
                by={"agent_id": agent_id, "intention_id": AnyOf(intention_ids, cast="uuid")},
            )
        return None

    def _ranked_order(self) -> list[Sort]:
        """the ranked reads' order: the salience rank (:meth:`_salience_rank`) descending, then the newest first.

        :return: the sort
        :rtype: list[Sort]
        """
        return [
            Sort("salience_rank", expression=self._salience_rank(), descending=True),
            Sort("date_created", descending=True),
        ]

    def _salience_rank(self) -> str:
        """what the ranked reads order by: the stored salience, or with decay on read its rank expression.

        :return: the SQL expression
        :rtype: str
        """
        decay = type(self).salience_decay
        return "salience" if decay is None else decay.rank_sql()

    def _reinforced(self, bump: float) -> dict[str, Fragment]:
        """a reinforcement's values: the stored salience plus the bump, or with decay on read the effective
        salience plus the bump, stamped now (:meth:`SalienceDecay.reinforced_values`).

        :param bump: the increment
        :ptype bump: float
        :return: column -> its new value
        :rtype: dict[str, Fragment]
        """
        decay = type(self).salience_decay
        if decay is None:
            return {"salience": Fragment("LEAST(1.0, salience + $1)", (bump,))}
        return decay.reinforced_values(bump)
