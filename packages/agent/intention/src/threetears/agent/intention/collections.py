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
from typing import Any, cast
from uuid import UUID

from sqlalchemy import MetaData, Table

from threetears.core.collections.salience import apply_salience_decay
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
    spans_partitions,
)
from threetears.observe import get_logger

from threetears.agent.intention.entities import IntentionEntity
from threetears.agent.intention.types import INTENTION_STATUS_VALUES, IntentionStatus

__all__ = [
    "IntentionsCollection",
    "intentions_table",
]

log = get_logger(__name__)


# Embedding dimension carried by the intentions table. Matches memory's
# 1024-dim vector so a shared embedding provider serves both corpora.
_INTENTION_VECTOR_DIM = 1024


# Explicit column list for multi-row reads: the pgvector ``embedding``
# column is cast ``::text`` so asyncpg returns it without a registered
# vector codec (dedup/semantic paths, added in B2, decode it explicitly).
_INTENTIONS_SELECT_COLUMNS = (
    "intention_id, agent_id, customer_id, user_id, status, content, "
    "embedding::text AS embedding, salience, last_decayed_at, "
    "last_surfaced_at, source_memory_id, source_conversation_id, "
    "date_created, date_updated"
)


def intentions_table(metadata: MetaData) -> Table:
    """Register the ``intentions`` table on the given SA metadata.

    Thin idempotency wrapper around
    :meth:`IntentionsCollection.schema.to_sqlalchemy_table`. Call before
    ``SQLiteBackend.initialize(metadata)`` so the L1 cache builds with
    the full schema, and before Alembic ``target_metadata`` reflection so
    auto-generate sees the same shape.

    :param metadata: SQLAlchemy metadata to attach the table to
    :ptype metadata: MetaData
    :return: the ``intentions`` :class:`Table`
    :rtype: Table
    """
    return cast(Table, IntentionsCollection.schema.to_sqlalchemy_table(metadata))


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
            Column("user_id", UUID_TYPE, nullable=True),
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
            Column(
                "salience",
                NUMERIC_TYPE,
                precision=5,
                scale=4,
                nullable=False,
                server_default="0.5",
            ),
            # decay anchor: age is measured from the last decay run so
            # total decay over a period is cadence-safe.
            Column("last_decayed_at", DATETIMETZ_TYPE, nullable=True),
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
            # (Column order / DESC live in the raw v001 DDL; the DSL only
            # needs the column set + WHERE for parity + L1 parity.)
            SchemaIndex(
                "idx_intentions_open_ranked",
                "agent_id",
                "user_id",
                "salience",
                where=f"status = '{IntentionStatus.OPEN.value}'",
            ),
            # the cooldown filter reads by last-surfaced recency.
            SchemaIndex(
                "idx_intentions_last_surfaced",
                "agent_id",
                "last_surfaced_at",
            ),
            # HNSW over the embedding for the log-time dedup lookup.
            SchemaIndex(
                "ix_intentions_embedding_hnsw",
                "embedding",
                using="hnsw",
                ops={"embedding": "vector_cosine_ops"},
            ),
        ),
    )

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
        rows = await self.l3_pool.fetch(
            f"SELECT {_INTENTIONS_SELECT_COLUMNS} FROM intentions "
            "WHERE agent_id = $1 AND user_id = $2 "
            "ORDER BY salience DESC, date_created DESC",
            agent_id,
            user_id,
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
        rows = await self.l3_pool.fetch(
            f"SELECT {_INTENTIONS_SELECT_COLUMNS} FROM intentions "
            "WHERE agent_id = $1 AND user_id = $2 "
            f"AND status = '{IntentionStatus.OPEN.value}' "
            "AND (last_surfaced_at IS NULL OR last_surfaced_at < $3) "
            "ORDER BY salience DESC, date_created DESC",
            agent_id,
            user_id,
            cooldown_cutoff,
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
        embedding_str = json.dumps(embedding)
        # cache-bypass: vector-distance search is not primary-key
        # addressable; the L1 row cache cannot serve it. ``embedding IS
        # NOT NULL`` guards the ``float(similarity)`` cast (``NULL <=>
        # vector`` is NULL). Only OPEN wants are dedup candidates.
        rows = await self.l3_pool.fetch(
            "SELECT intention_id, content, salience, "
            "1 - (embedding OPERATOR(public.<=>) $1::text::public.vector) AS similarity "
            "FROM intentions "
            "WHERE agent_id = $2 AND user_id = $3 AND embedding IS NOT NULL "
            f"AND status = '{IntentionStatus.OPEN.value}' "
            "ORDER BY embedding OPERATOR(public.<=>) $1::text::public.vector "
            "LIMIT $4",
            embedding_str,
            agent_id,
            user_id,
            top_k,
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

    @spans_partitions(marker_only=True)
    async def decay_salience(
        self,
        *,
        half_life_days: float,
        floor: float,
    ) -> int:
        """decay stored salience for every want (restraint brake #2).

        The scheduled maintenance pass: ``salience`` sinks toward
        ``floor`` on a ``half_life_days`` half-life, anchored on
        ``last_decayed_at`` so total decay over a period is
        cadence-independent. Delegates to the shared
        :func:`apply_salience_decay` (factored in A2) so ``agent/memory``
        and ``agent/intention`` run identical decay; this method stays
        the single entry point for the ``intentions`` table's SQL.

        ``skip_evergreen=False`` because ``intentions`` carries no
        ``evergreen`` pin column -- a standing want has no "never decays"
        concept (design §6.5), so the pass ages every want.

        Cache coherence: the raw L3 decay leaves L1/L2 holding the
        pre-decay salience, so each decayed pk is invalidated (via
        :meth:`invalidate_cache`) -- otherwise ``intention_log``'s
        dedup-refresh, which reads the want via ``get()`` before bumping
        it, could re-persist a stale salience and undo the decay.

        Abandoned wants sink out of the salience-ranked deliberation list
        without being deleted -- a re-log refreshes them.

        :param half_life_days: decay half-life in days
        :ptype half_life_days: float
        :param floor: salience asymptote; never decays below this
        :ptype floor: float
        :return: number of wants decayed
        :rtype: int
        """
        if self.l3_pool is None:
            return 0
        # __SPANS_PARTITIONS__: salience decay is a global maintenance
        # sweep with no partition to scope to -- it ages every row the
        # pool holds. The SQL literal lives in the shared helper (which
        # carries no ``intentions`` literal), so the partition-enforcement
        # walker is satisfied and this method holds no raw table SQL.
        result = await apply_salience_decay(
            self.l3_pool,
            table="intentions",
            half_life_seconds=half_life_days * 86400.0,
            floor=floor,
            skip_evergreen=False,
            returning_columns=self.primary_key_columns,
        )
        decayed_pks = result if isinstance(result, list) else []
        for pk in decayed_pks:
            await self.invalidate_cache(pk)
        return len(decayed_pks)
