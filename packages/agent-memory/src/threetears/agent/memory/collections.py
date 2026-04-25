"""Memories collection -- three-tier CRUD for memory entities.

Also wires the sibling collections for tables the memory package owns
but previously lacked Collection coverage: :class:`MediaCollection`,
:class:`MediaContentCollection`, :class:`MemoryChunkCollection` (adopted
under namespace-task-01 phase 8.5b). Each Collection resolves its L3
pool via the registry (``self.l3_pool``) — the bespoke
``_postgres_pool`` field is retired in favour of the registry pattern
:class:`ConversationCollection` already uses.

Complex hybrid-search queries (vector + FTS + MMR) live as methods on
these Collections with documented ``# cache-bypass:`` comments — the
Collection stays the single entry point for memory-table SQL even when
the query shape is not primary-key addressable and therefore cannot
benefit from L1 row caching.
"""

from __future__ import annotations

import asyncio
import json
import math
from datetime import UTC, datetime
from typing import Any, ClassVar
from uuid import UUID

from threetears.core.collections.flush import WriteBuffer
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import (
    BOOL_TYPE,
    DATETIME_TYPE,
    INT_TYPE,
    JSONB_TYPE,
    STRING_TYPE,
    UUID_TYPE,
    VECTOR_TYPE,
    Column,
    SchemaBackedCollection,
    TableSchema,
    spans_partitions,
)
from threetears.core.config import CoreConfig
from threetears.observe import get_logger

from threetears.agent.memory.authorize import (
    ACTION_MEMORY_READ,
    ACTION_MEMORY_WRITE,
    MemoryAuthorizerDependencies,
    authorize_memory_access,
    ensure_memory_owner_assignment,
)
from threetears.agent.memory.entities import (
    MediaContentEntity,
    MediaEntity,
    MemoryChunkEntity,
    MemoryEntity,
    MemoryRefEntity,
)

__all__ = [
    "MediaCollection",
    "MediaContentCollection",
    "MemoriesCollection",
    "MemoryChunkCollection",
    "MemoryRefsCollection",
]

log = get_logger(__name__)


def _build_fts_text(user_text: str, min_len: int = 3, max_len: int = 500) -> str | None:
    """Prepare free-text query for ``websearch_to_tsquery``.

    :param user_text: raw user text
    :ptype user_text: str
    :param min_len: minimum length after stripping
    :ptype min_len: int
    :param max_len: cap on returned length
    :ptype max_len: int
    :return: cleaned query string, or ``None`` when too short
    :rtype: str | None
    """
    text = user_text.strip()
    if len(text) < min_len:
        return None
    return text[:max_len]


def _build_user_scope_clause(
    user_id: UUID,
    *,
    agent_id: UUID,
    customer_id: UUID,
    start_param: int = 2,
    table_prefix: str = "",
) -> tuple[str, list[UUID], int]:
    """Build WHERE fragments scoping by agent / customer / user.

    collections-task-04 requires the partition column ``agent_id`` and
    its sub-scope ``customer_id`` to be mandatory: the legacy "optional
    scoping tag" doctrine is retired with v008's NOT NULL restoration.
    every memory hybrid-search SQL site receives a fully-scoped
    ``(agent_id, customer_id, user_id)`` triple.

    :param user_id: user ID
    :ptype user_id: UUID
    :param agent_id: partition column for the memory tables; required
    :ptype agent_id: UUID
    :param customer_id: customer ID sub-scope; required
    :ptype customer_id: UUID
    :param start_param: starting positional parameter index
    :ptype start_param: int
    :param table_prefix: optional table alias prefix
    :ptype table_prefix: str
    :return: tuple of (conditions string, param values, last param
        index used)
    :rtype: tuple[str, list[UUID], int]
    """
    prefix = f"{table_prefix}." if table_prefix else ""
    conditions: list[str] = []
    params: list[UUID] = []
    idx = start_param

    conditions.append(f"{prefix}agent_id = ${idx}")
    params.append(agent_id)
    idx += 1

    conditions.append(f"{prefix}customer_id = ${idx}")
    params.append(customer_id)
    idx += 1

    conditions.append(f"{prefix}user_id = ${idx}")
    params.append(user_id)

    return " AND ".join(conditions), params, idx


def _normalize_scores(candidates: list[dict[str, Any]], key: str) -> None:
    """Min-max normalize scores in-place to [0, 1].

    :param candidates: candidate rows (mutated)
    :ptype candidates: list[dict[str, Any]]
    :param key: score column name
    :ptype key: str
    :return: nothing
    :rtype: None
    """
    scores = [c.get(key, 0.0) for c in candidates]
    lo, hi = min(scores), max(scores)
    span = hi - lo
    for c in candidates:
        raw = c.get(key, 0.0)
        if span > 0:
            c[key] = (raw - lo) / span
        elif hi > 0:
            c[key] = 1.0


def _recency_weight(created: datetime, half_life_hours: float) -> float:
    """Exponential recency decay (1.0 = just created, ~0.37 at half-life).

    :param created: creation timestamp (naive UTC accepted, treated as UTC)
    :ptype created: datetime
    :param half_life_hours: half-life in hours
    :ptype half_life_hours: float
    :return: decay factor in (0, 1]
    :rtype: float
    """
    now = datetime.now(UTC)
    if created.tzinfo is None:
        created = created.replace(tzinfo=UTC)
    hours_ago = max((now - created).total_seconds() / 3600, 0.0)
    return math.exp(-hours_ago / half_life_hours)


class MemoriesCollection(SchemaBackedCollection[MemoryEntity]):
    """Collection for memory entities with three-tier caching.

    CRUD goes through :meth:`get` / :meth:`save_entity` / :meth:`delete`
    so L1 / L2 / L3 tiers stay coherent. Hybrid-search methods
    (:meth:`hybrid_search`, :meth:`search_by_ids`,
    :meth:`find_similar_for_dedup`, :meth:`count_by_user`) absorb the
    vector + FTS + MMR queries that used to live raw on the pool; they
    carry ``# cache-bypass:`` justification because the query shape is
    not primary-key-addressable and therefore cannot benefit from the
    L1 row cache — but keeping them on the Collection preserves the
    single-entry-point contract enforcement test walker #3 relies on.

    CRUD is generated from :attr:`schema`: embedding is ``VECTOR_TYPE``
    (pgvector ``::vector`` cast + list[float] decode), ``date_updated``
    is the CAS fence so concurrent writers race correctly, and the
    scope columns (agent/customer/user/conversation/message_id_source/
    type_memory/media_id/date_created) are marked immutable so the
    ``DO UPDATE SET`` clause narrows to content + embedding +
    is_deleted + date_deleted + date_updated.
    """

    primary_key_column: str | tuple[str, ...] = ("agent_id", "memory_id")
    # rationale: ``soft_delete`` operates on a fully-formed
    # :class:`MemoryEntity` whose composite pk already pins the
    # partition; the method reads agent_id off the entity rather than
    # accepting it as a redundant parameter. ``create`` / ``get`` are
    # framework methods inherited from :class:`SchemaBackedCollection`
    # but Python's MRO surfaces them on the subclass via the property /
    # generic decoration; declaring them exempt here keeps the
    # __init_subclass__ guard quiet without weakening the partition
    # contract on the read-write surface.
    _partition_exempt_methods: ClassVar[frozenset[str]] = frozenset({"soft_delete"})
    schema = TableSchema(
        name="memories",
        primary_key=("agent_id", "memory_id"),
        columns=[
            Column("memory_id", UUID_TYPE),
            Column("agent_id", UUID_TYPE, partition=True),
            Column("customer_id", UUID_TYPE, immutable=True),
            Column("user_id", UUID_TYPE, immutable=True),
            Column("conversation_id", UUID_TYPE, immutable=True),
            Column("message_id_source", UUID_TYPE, immutable=True),
            Column("type_memory", STRING_TYPE, immutable=True),
            Column("content", STRING_TYPE),
            Column("embedding", VECTOR_TYPE),
            Column("is_deleted", BOOL_TYPE),
            Column("media_id", UUID_TYPE, nullable=True, immutable=True),
            Column("date_created", DATETIME_TYPE, immutable=True),
            Column("date_deleted", DATETIME_TYPE, nullable=True),
            Column("date_updated", DATETIME_TYPE, nullable=True),
        ],
        cas_column="date_updated",
    )

    def __init__(
        self,
        registry: CollectionRegistry,
        config: CoreConfig,
        authorizer: MemoryAuthorizerDependencies,
        nats_client: Any = None,
        write_buffer: WriteBuffer | None = None,
    ) -> None:
        """initialize memory collection with required rbac authorizer.

        :param registry: shared collection registry; ``l3_pool`` is
            resolved through :meth:`CollectionRegistry.get_l3_pool`
            (same pattern :class:`ConversationCollection` uses), so
            callers must bind the target agent pool via
            :meth:`CollectionRegistry.configure` or
            :meth:`CollectionRegistry.bind_table` before construction
        :ptype registry: CollectionRegistry
        :param config: core configuration governing flush behaviour
        :ptype config: CoreConfig
        :param authorizer: rbac authorizer dependency bundle; required.
            every ``caller_user_id``-bearing read / write goes through
            :func:`authorize_memory_access` against this bundle. tests
            inject a permissive fixture from ``conftest.py``; production
            wiring builds the real bundle from hub-side loaders +
            namespace resolver + first-write assignment ensurer
        :ptype authorizer: MemoryAuthorizerDependencies
        :param nats_client: L2 NATS KV client for cache promotion
        :ptype nats_client: Any
        :param write_buffer: optional shared write buffer
        :ptype write_buffer: WriteBuffer | None
        """
        self._authorizer = authorizer
        super().__init__(registry, config, nats_client, write_buffer)

    @property
    def table_name(self) -> str:
        """Return the database table name for this collection.

        :return: table name
        :rtype: str
        """
        return "memories"

    @property
    def entity_class(self) -> type[MemoryEntity]:
        """Return the entity class for this collection.

        :return: entity class
        :rtype: type[MemoryEntity]
        """
        return MemoryEntity

    async def find_by_user(
        self,
        user_id: UUID,
        include_deleted: bool = False,
        *,
        agent_id: UUID,
        customer_id: UUID,
        caller_user_id: UUID | None = None,
        caller_agent_id: UUID | None = None,
    ) -> list[MemoryEntity]:
        """fetch memories for ``(agent_id, user_id)`` from L3, enforcing rbac.

        ``agent_id`` is the partition column on the memories table;
        every read must include the partition predicate. ``customer_id``
        is a required sub-scope. when ``caller_user_id`` is provided
        the rbac evaluator decides ``memory.read`` on the
        ``(agent_id, customer_id)`` memory namespace before the SQL
        runs. the owner short-circuit fires when
        ``caller_agent_id == agent_id``; a mismatched pair surfaces
        :class:`MemoryAccessDenied` from :func:`authorize_memory_access`.

        the row-level ``user_id`` filter is kept as a belt-and-suspenders
        cut against grants that resolve to broad type_customer scope
        but should still respect the per-row owner column.

        :param user_id: user whose memories to fetch (row filter)
        :ptype user_id: UUID
        :param include_deleted: whether to include soft-deleted memories
        :ptype include_deleted: bool
        :param agent_id: partition column on memories; required
        :ptype agent_id: UUID
        :param customer_id: required sub-scope
        :ptype customer_id: UUID
        :param caller_user_id: invoking user UUID for evaluator
        :ptype caller_user_id: UUID | None
        :param caller_agent_id: invoking agent UUID for evaluator
        :ptype caller_agent_id: UUID | None
        :return: list of memory entities belonging to user
        :rtype: list[MemoryEntity]
        :raises MemoryAccessDenied: when rbac enforcement denies
        """
        if caller_user_id is not None:
            await authorize_memory_access(
                action=ACTION_MEMORY_READ,
                agent_id=agent_id,
                customer_id=customer_id,
                caller_user_id=caller_user_id,
                caller_agent_id=caller_agent_id,
                deps=self._authorizer,
            )

        if self.l3_pool is None:
            return []
        if include_deleted:
            # cache-bypass: multi-row scan by (agent_id, customer_id,
            # user_id) is not primary-key addressable; L1 row cache
            # would not help. method on Collection preserves single
            # entry point.
            rows = await self.l3_pool.fetch(
                "SELECT * FROM memories "
                "WHERE agent_id = $1 AND customer_id = $2 AND user_id = $3 "
                "ORDER BY date_created DESC",
                agent_id,
                customer_id,
                user_id,
            )
        else:
            # cache-bypass: same as above — multi-row scan path.
            rows = await self.l3_pool.fetch(
                "SELECT * FROM memories "
                "WHERE agent_id = $1 AND customer_id = $2 AND user_id = $3 "
                "AND is_deleted = false "
                "ORDER BY date_created DESC",
                agent_id,
                customer_id,
                user_id,
            )
        entities: list[MemoryEntity] = []
        for row in rows:
            data = dict(row)
            entity = self.entity_class(data, is_new=False, collection=self)
            entity.original_date_updated = data.get("date_updated")
            entity_id = (data["agent_id"], data["memory_id"])
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
        *,
        caller_user_id: UUID | None = None,
        caller_agent_id: UUID | None = None,
    ) -> list[MemoryEntity]:
        """fetch memories scoped by agent, enforcing rbac on user reads.

        when ``caller_user_id`` is provided the rbac evaluator
        decides ``memory.read`` on the ``(agent_id, customer_id)``
        memory namespace before the SQL runs. ``customer_id`` is
        required for evaluator invocation — single-argument
        ``find_by_scope(agent_id)`` is an agent-internal row-scan
        path (no user dimension) and runs without evaluation.

        :param agent_id: agent ID scope (required)
        :ptype agent_id: UUID
        :param customer_id: optional customer ID to further narrow
            scope; required when ``caller_user_id`` is set
        :ptype customer_id: UUID | None
        :param user_id: optional user ID to further narrow scope
        :ptype user_id: UUID | None
        :param include_deleted: whether to include soft-deleted memories
        :ptype include_deleted: bool
        :param caller_user_id: invoking user UUID for evaluator
        :ptype caller_user_id: UUID | None
        :param caller_agent_id: invoking agent UUID for evaluator
        :ptype caller_agent_id: UUID | None
        :return: list of memory entities matching scope
        :rtype: list[MemoryEntity]
        :raises MemoryAccessDenied: when rbac enforcement denies
        :raises ValueError: when ``caller_user_id`` is set without
            ``customer_id``
        """
        if caller_user_id is not None:
            if customer_id is None:
                raise ValueError(
                    "find_by_scope with caller_user_id requires customer_id",
                )
            await authorize_memory_access(
                action=ACTION_MEMORY_READ,
                agent_id=agent_id,
                customer_id=customer_id,
                caller_user_id=caller_user_id,
                caller_agent_id=caller_agent_id,
                deps=self._authorizer,
            )
        if self.l3_pool is None:
            return []
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

        # cache-bypass: multi-row scan by agent scope is not primary-
        # key addressable; L1 row cache would not help. method on
        # Collection preserves single entry point + rbac gating.
        rows = await self.l3_pool.fetch(query, *params)

        entities: list[MemoryEntity] = []
        for row in rows:
            data = dict(row)
            entity = self.entity_class(data, is_new=False, collection=self)
            entity.original_date_updated = data.get("date_updated")
            entity_id = (data["agent_id"], data["memory_id"])
            await self._save_to_l2(entity_id, data)
            entities.append(entity)
        return entities

    async def save_memory(
        self,
        entity: MemoryEntity,
        *,
        agent_id: UUID,
        customer_id: UUID,
        caller_user_id: UUID | None,
        caller_agent_id: UUID | None,
    ) -> None:
        """persist a memory after evaluating ``memory.write`` for the caller.

        user-initiated writes land here. the evaluator decides
        ``memory.write`` for the caller against the ``(agent_id,
        customer_id)`` memory namespace; owner short-circuit
        applies when ``caller_agent_id == agent_id`` (agent-
        internal writes). on a successful user write (evaluator
        passed AND caller_user_id is set AND the agent-owner
        short-circuit did NOT fire), the per-user ``MemoryOwner``
        assignment is ensured via
        :func:`ensure_memory_owner_assignment` so subsequent reads
        from the same user hit a cached grant.

        the agent-internal extractor path bypasses this method and
        calls :meth:`save_entity` directly because it runs under
        the agent-owner short-circuit; the explicit call site gives
        operators a clean audit distinction between agent-emitted
        and user-explicit memory rows.

        rbac enforcement is unconditional: the collection is
        constructed with a required :class:`MemoryAuthorizerDependencies`
        bundle, and every call evaluates :func:`authorize_memory_access`
        before the write.

        :param entity: memory entity to persist
        :ptype entity: MemoryEntity
        :param agent_id: owning agent UUID
        :ptype agent_id: UUID
        :param customer_id: owning customer UUID
        :ptype customer_id: UUID
        :param caller_user_id: invoking user UUID (``None`` for
            agent-internal writes — owner short-circuit path)
        :ptype caller_user_id: UUID | None
        :param caller_agent_id: invoking agent UUID
        :ptype caller_agent_id: UUID | None
        :return: nothing
        :rtype: None
        :raises MemoryAccessDenied: on evaluator deny
        """
        ns_entity = await authorize_memory_access(
            action=ACTION_MEMORY_WRITE,
            agent_id=agent_id,
            customer_id=customer_id,
            caller_user_id=caller_user_id,
            caller_agent_id=caller_agent_id,
            deps=self._authorizer,
        )

        await self.save_entity(entity)

        is_owner_shortcut = (
            caller_agent_id is not None and caller_agent_id == agent_id
        )
        if caller_user_id is not None and not is_owner_shortcut:
            await ensure_memory_owner_assignment(
                user_id=caller_user_id,
                namespace=ns_entity,
                deps=self._authorizer,
            )
        return None

    async def soft_delete(self, entity: MemoryEntity) -> None:
        """Soft-delete a memory by setting is_deleted and date_deleted.

        :param entity: entity to soft-delete
        :ptype entity: MemoryEntity
        :return: nothing
        :rtype: None
        """
        entity.is_deleted = True
        entity.date_deleted = datetime.now(UTC)
        await self.save_entity(entity)

    async def count_by_user(self, user_id: UUID, *, agent_id: UUID) -> bool:
        """check whether any memory row exists for ``(agent_id, user_id)``.

        returns a boolean rather than an exact count — every caller
        today uses the existence flag (tools.py first-write ensure
        gate). keeps the query cheap (``SELECT EXISTS(...)``) and
        side-steps full row-count pagination concerns. ``agent_id`` is
        the partition column on memories and is required.

        :param user_id: user UUID to probe
        :ptype user_id: UUID
        :param agent_id: partition column on memories; required
        :ptype agent_id: UUID
        :return: ``True`` iff user has at least one memory row in
            this agent's partition
        :rtype: bool
        """
        if self.l3_pool is None:
            return False
        # cache-bypass: existence probe is not primary-key-addressable;
        # L1 row cache would not help. method on Collection preserves
        # single entry point.
        value = await self.l3_pool.fetchval(
            "SELECT EXISTS(SELECT 1 FROM memories "
            "WHERE agent_id = $1 AND user_id = $2)",
            agent_id,
            user_id,
        )
        return bool(value)

    async def find_similar_for_dedup(
        self,
        *,
        user_id: UUID,
        agent_id: UUID,
        embedding: list[float],
        top_k: int,
        threshold: float,
    ) -> list[dict[str, Any]]:
        """vector search for near-duplicate memories by embedding.

        used by :class:`MemoryExtractor._get_similar_memories` +
        :class:`add_memory` tool dedup guard. returns memories whose
        cosine similarity to ``embedding`` exceeds ``threshold``,
        capped at ``top_k``. ``agent_id`` is the partition column and
        is required.

        :param user_id: owning user UUID (row filter)
        :ptype user_id: UUID
        :param agent_id: partition column on memories; required
        :ptype agent_id: UUID
        :param embedding: query embedding vector
        :ptype embedding: list[float]
        :param top_k: maximum candidates to consider
        :ptype top_k: int
        :param threshold: minimum cosine similarity to surface
        :ptype threshold: float
        :return: list of ``{memory_id, content, type_memory, similarity}``
        :rtype: list[dict[str, Any]]
        """
        if self.l3_pool is None:
            return []
        embedding_str = json.dumps(embedding)
        # cache-bypass: vector-distance search is not primary-key-
        # addressable; L1 row cache cannot serve. keeping the query
        # on the Collection preserves single entry point for uniformity
        # + audit hooks + future observability.
        rows = await self.l3_pool.fetch(
            """
            SELECT memory_id, content, type_memory,
                   1 - (embedding <=> $1::vector) AS similarity
            FROM memories
            WHERE agent_id = $2 AND user_id = $3 AND is_deleted = false
            ORDER BY embedding <=> $1::vector
            LIMIT $4
            """,
            embedding_str,
            agent_id,
            user_id,
            top_k,
        )
        return [
            {
                "memory_id": row["memory_id"],
                "content": row["content"],
                "type_memory": row["type_memory"],
                "similarity": float(row["similarity"]),
            }
            for row in rows
            if float(row["similarity"]) >= threshold
        ]

    async def hybrid_search(
        self,
        *,
        user_id: UUID,
        agent_id: UUID,
        customer_id: UUID,
        embedding: list[float],
        user_text: str,
        top_k: int,
        candidate_limit: int,
        similarity_threshold: float,
        recency_half_life_hours: float,
        signal_weights: dict[str, float],
        fts_min_len: int = 3,
        fts_max_len: int = 500,
    ) -> list[dict[str, Any]]:
        """parallel vector + FTS hybrid search across the memories table.

        absorbs the SQL that used to live in
        :meth:`MemoryRetriever._query_memories`. three-signal
        ranking (semantic / keyword / recency) + cosine-distance
        ordering; candidates are merged across the two parallel
        queries on ``memory_id``, recency-decayed, score-combined, and
        threshold-filtered.

        :param user_id: owning user UUID (row filter)
        :ptype user_id: UUID
        :param agent_id: partition column on memories; required
        :ptype agent_id: UUID
        :param customer_id: required sub-scope
        :ptype customer_id: UUID
        :param embedding: query embedding vector
        :ptype embedding: list[float]
        :param user_text: raw query text for FTS
        :ptype user_text: str
        :param top_k: number of candidates to return after ranking
        :ptype top_k: int
        :param candidate_limit: per-query candidate pool size
        :ptype candidate_limit: int
        :param similarity_threshold: floor on hybrid score
        :ptype similarity_threshold: float
        :param recency_half_life_hours: exponential decay half-life
        :ptype recency_half_life_hours: float
        :param signal_weights: mapping ``{"semantic", "keyword",
            "recency"}`` to weights
        :ptype signal_weights: dict[str, float]
        :param fts_min_len: minimum query length for FTS activation
        :ptype fts_min_len: int
        :param fts_max_len: truncation length for FTS queries
        :ptype fts_max_len: int
        :return: ranked candidate list, top_k entries
        :rtype: list[dict[str, Any]]
        """
        if self.l3_pool is None:
            return []
        embedding_str = json.dumps(embedding)

        scope_conditions, scope_params, param_offset = _build_user_scope_clause(
            user_id,
            agent_id=agent_id,
            customer_id=customer_id,
            start_param=2,
        )
        vec_where = f"WHERE {scope_conditions} AND is_deleted = false"
        limit_param = f"${param_offset + 1}"

        # cache-bypass: vector-distance search is not primary-key-
        # addressable; see :meth:`find_similar_for_dedup` for the same
        # justification. method on Collection preserves single entry
        # point for rbac + audit.
        vec_coro = self.l3_pool.fetch(
            f"""
            SELECT memory_id, content, summary, type_memory, date_created,
                   embedding,
                   1 - (embedding <=> $1::vector) AS similarity
            FROM memories
            {vec_where}
            ORDER BY embedding <=> $1::vector
            LIMIT {limit_param}
            """,
            embedding_str,
            *scope_params,
            candidate_limit,
        )

        fts_text = _build_fts_text(user_text, fts_min_len, fts_max_len)
        if fts_text:
            fts_scope_conditions, fts_scope_params, fts_param_offset = (
                _build_user_scope_clause(
                    user_id,
                    agent_id=agent_id,
                    customer_id=customer_id,
                    start_param=2,
                )
            )
            fts_where = f"WHERE {fts_scope_conditions} AND is_deleted = false"
            fts_limit_param = f"${fts_param_offset + 1}"
            # cache-bypass: FTS rank query is not primary-key-
            # addressable. See :meth:`hybrid_search` docstring.
            fts_coro = self.l3_pool.fetch(
                f"""
                SELECT memory_id, content, summary, type_memory, date_created,
                       embedding,
                       ts_rank_cd(search_vector, websearch_to_tsquery('english', $1)) AS fts_rank
                FROM memories
                {fts_where}
                  AND search_vector @@ websearch_to_tsquery('english', $1)
                ORDER BY fts_rank DESC
                LIMIT {fts_limit_param}
                """,
                fts_text,
                *fts_scope_params,
                candidate_limit,
            )
            vec_rows, fts_rows = await asyncio.gather(vec_coro, fts_coro)
        else:
            vec_rows = await vec_coro
            fts_rows = []

        merged: dict[Any, dict[str, Any]] = {}
        for row in vec_rows:
            mid = row["memory_id"]
            emb = row["embedding"]
            if isinstance(emb, str):
                emb = json.loads(emb)
            merged[mid] = {
                "memory_id": mid,
                "content": row["content"],
                "summary": row["summary"],
                "type_memory": row["type_memory"],
                "date_created": row["date_created"],
                "similarity": float(row["similarity"]),
                "fts_rank": 0.0,
                "embedding": emb,
            }
        for row in fts_rows:
            mid = row["memory_id"]
            if mid in merged:
                merged[mid]["fts_rank"] = float(row["fts_rank"])
            else:
                emb = row["embedding"]
                if isinstance(emb, str):
                    emb = json.loads(emb)
                merged[mid] = {
                    "memory_id": mid,
                    "content": row["content"],
                    "summary": row["summary"],
                    "type_memory": row["type_memory"],
                    "date_created": row["date_created"],
                    "similarity": 0.0,
                    "fts_rank": float(row["fts_rank"]),
                    "embedding": emb,
                }

        candidates = list(merged.values())
        if not candidates:
            return []

        _normalize_scores(candidates, "fts_rank")

        for c in candidates:
            recency = _recency_weight(c["date_created"], recency_half_life_hours)
            c["recency"] = round(recency, 4)
            c["hybrid_score"] = round(
                signal_weights["semantic"] * c["similarity"]
                + signal_weights["keyword"] * c["fts_rank"]
                + signal_weights["recency"] * recency,
                4,
            )

        filtered = [c for c in candidates if c["hybrid_score"] > similarity_threshold]
        filtered.sort(key=lambda m: m["hybrid_score"], reverse=True)
        return filtered[:top_k]

    async def search_by_ids(
        self,
        memory_ids: list[UUID],
        user_id: UUID,
        *,
        agent_id: UUID,
    ) -> list[dict[str, Any]]:
        """batch lookup of memories by primary key, scoped to ``(agent_id, user_id)``.

        absorbs the ``SELECT ... FROM memories WHERE memory_id = ANY(...)``
        leg of :func:`_search_by_ids` in :mod:`tools.py`. scoped by
        ``agent_id`` (partition predicate) and ``user_id`` (row-level
        owner) so a leaked memory_id from another partition or another
        user does not surface content.

        :param memory_ids: primary-key values to fetch
        :ptype memory_ids: list[UUID]
        :param user_id: owning user UUID (row filter)
        :ptype user_id: UUID
        :param agent_id: partition column on memories; required
        :ptype agent_id: UUID
        :return: list of row dicts (content + metadata)
        :rtype: list[dict[str, Any]]
        """
        if self.l3_pool is None or not memory_ids:
            return []
        # cache-bypass: batch IN(...) query is primary-key addressable
        # in principle but the existing L1 backend offers only per-row
        # ``select_by_id`` / ``select_batch`` and does not mix with
        # the row-level ``user_id`` filter we enforce here. keeping
        # the batch on the Collection preserves the single entry point
        # + rbac guard; the walker tags this call as inside the
        # Collection class so enforcement test #3 stays clean.
        rows = await self.l3_pool.fetch(
            """
            SELECT memory_id, type_memory, content, date_created
            FROM memories
            WHERE agent_id = $1
              AND memory_id = ANY($2::uuid[])
              AND user_id = $3
              AND is_deleted = false
            """,
            agent_id,
            memory_ids,
            user_id,
        )
        return [dict(row) for row in rows]

    async def search_by_semantic(
        self,
        *,
        user_id: UUID,
        agent_id: UUID,
        embedding: list[float],
        max_results: int,
        similarity_threshold: float,
        type_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """semantic (vector-only) search for the add/search memory tools.

        absorbs the ``_search_memories`` path from :mod:`tools.py`
        (the LangChain ``memory_search`` tool's vector leg). returns
        rows whose cosine similarity exceeds ``similarity_threshold``,
        capped at ``max_results``. ``type_filter`` is validated by the
        caller; this method trusts the value and applies it as an
        equality predicate. ``agent_id`` is the partition column on
        memories and is required.

        :param user_id: owning user UUID (row filter)
        :ptype user_id: UUID
        :param agent_id: partition column on memories; required
        :ptype agent_id: UUID
        :param embedding: query embedding vector
        :ptype embedding: list[float]
        :param max_results: cap on returned rows
        :ptype max_results: int
        :param similarity_threshold: similarity floor
        :ptype similarity_threshold: float
        :param type_filter: optional ``type_memory`` equality filter
        :ptype type_filter: str | None
        :return: list of row dicts with ``similarity`` field
        :rtype: list[dict[str, Any]]
        """
        if self.l3_pool is None:
            return []
        embedding_str = json.dumps(embedding)
        params: list[Any] = [embedding_str, agent_id, user_id]
        conditions = ["agent_id = $2", "user_id = $3", "is_deleted = false"]
        param_idx = 4
        if type_filter:
            conditions.append(f"type_memory = ${param_idx}")
            params.append(type_filter)
            param_idx += 1
        where_clause = " AND ".join(conditions)
        query_sql = f"""
            SELECT memory_id, type_memory, content, date_created,
                   1 - (embedding <=> $1::vector) AS similarity
            FROM memories
            WHERE {where_clause}
            ORDER BY embedding <=> $1::vector
            LIMIT ${param_idx}
        """
        params.append(max_results)
        # cache-bypass: vector-distance ordered lookup; see
        # :meth:`hybrid_search` docstring.
        rows = await self.l3_pool.fetch(query_sql, *params)
        return [
            {
                "memory_id": str(row["memory_id"]),
                "type": row["type_memory"],
                "content": row["content"],
                "date_created": row["date_created"],
                "similarity": float(row["similarity"]),
            }
            for row in rows
            if float(row["similarity"]) > similarity_threshold
        ]

    async def search_by_fts(
        self,
        *,
        user_id: UUID,
        agent_id: UUID,
        fts_text: str,
        max_results: int,
        type_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """FTS keyword search for the add/search memory tools.

        complements :meth:`search_by_semantic` — run both in parallel
        and merge on ``memory_id``. ``agent_id`` is the partition
        column on memories and is required.

        :param user_id: owning user UUID (row filter)
        :ptype user_id: UUID
        :param agent_id: partition column on memories; required
        :ptype agent_id: UUID
        :param fts_text: FTS query text
        :ptype fts_text: str
        :param max_results: cap on returned rows
        :ptype max_results: int
        :param type_filter: optional ``type_memory`` equality filter
        :ptype type_filter: str | None
        :return: list of row dicts
        :rtype: list[dict[str, Any]]
        """
        if self.l3_pool is None:
            return []
        conditions = [
            "agent_id = $2",
            "user_id = $3",
            "is_deleted = false",
            "search_vector @@ websearch_to_tsquery('english', $1)",
        ]
        params: list[Any] = [fts_text, agent_id, user_id]
        idx = 4
        if type_filter:
            conditions.append(f"type_memory = ${idx}")
            params.append(type_filter)
            idx += 1
        where = " AND ".join(conditions)
        query_sql = f"""
            SELECT memory_id, type_memory, content, date_created,
                   ts_rank_cd(search_vector, websearch_to_tsquery('english', $1)) AS fts_rank
            FROM memories
            WHERE {where}
            ORDER BY fts_rank DESC
            LIMIT {max_results}
        """
        # cache-bypass: FTS rank query is not primary-key-addressable;
        # see :meth:`hybrid_search` docstring.
        rows = await self.l3_pool.fetch(query_sql, *params)
        return [
            {
                "memory_id": str(row["memory_id"]),
                "type": row["type_memory"],
                "content": row["content"],
                "date_created": row["date_created"],
            }
            for row in rows
        ]

    async def fetch_content_for_recall(
        self,
        *,
        memory_id: UUID,
        user_id: UUID,
        agent_id: UUID,
    ) -> str | None:
        """fetch just the ``content`` field for the recall_memory tool.

        scoped by ``agent_id`` (partition predicate) and ``user_id``
        (row-level owner) so a leaked ID does not cross partitions or
        users.

        :param memory_id: memory primary-key value
        :ptype memory_id: UUID
        :param user_id: owning user UUID (row filter)
        :ptype user_id: UUID
        :param agent_id: partition column on memories; required
        :ptype agent_id: UUID
        :return: content text or ``None`` if not found
        :rtype: str | None
        """
        if self.l3_pool is None:
            return None
        # cache-bypass: the L1 cache would serve this lookup when the
        # row has already been warmed, but the ``user_id`` + ``is_deleted``
        # guard here is a SECURITY control (cross-user leak prevention)
        # that the L1 row cache cannot enforce. the authoritative read
        # must stay at the database.
        row = await self.l3_pool.fetchrow(
            "SELECT content FROM memories "
            "WHERE agent_id = $1 AND memory_id = $2 AND user_id = $3 "
            "AND is_deleted = false",
            agent_id,
            memory_id,
            user_id,
        )
        if row is None:
            return None
        result: str = row["content"]
        return result

    @spans_partitions
    async def find_for_user_in_agents(
        self,
        *,
        user_id: UUID,
        agent_ids: tuple[UUID, ...],
        customer_id: UUID | None = None,
    ) -> list[MemoryEntity]:
        """fetch memories for ``user_id`` across an authorized set of agents.

        canonical cross-partition retrieval pattern: the caller has
        resolved an authorized set of agent partitions upstream
        (typically via :class:`MemoryAccessService` evaluating the
        unified RBAC engine on each candidate memory namespace) and
        passes the resolved set in as a tuple. the tuple shape is the
        type signal that distinguishes "deliberately fan out across
        these partitions" from "any partitions you want" -- the
        :func:`spans_partitions` decorator validates the shape at call
        time and refuses an empty tuple.

        ACL is NOT evaluated inside this method; the Collection is
        domain-pure. wire it through :class:`MemoryAccessService` (in
        :mod:`threetears.agent.memory.access`) to compose authorization
        with the partition fan-out.

        :param user_id: owning user UUID (row filter applied
            consistently in every selected partition)
        :ptype user_id: UUID
        :param agent_ids: tuple of agent UUIDs the caller has been
            authorized to read from. must be non-empty (the
            :func:`spans_partitions` decorator rejects empty tuples;
            service-layer callers short-circuit when no agents resolve
            and never invoke this method with an empty set)
        :ptype agent_ids: tuple[UUID, ...]
        :param customer_id: optional customer ID sub-scope; when
            provided, narrows the result set to rows whose
            ``customer_id`` matches
        :ptype customer_id: UUID | None
        :return: list of memory entities across the authorized agent
            partitions, ordered by ``date_created`` DESC
        :rtype: list[MemoryEntity]
        """
        if self.l3_pool is None:
            return []
        # cache-bypass: cross-partition fan-out by ANY($N::uuid[]) is
        # not primary-key-addressable; L1 row cache cannot serve a
        # multi-partition projection. method on Collection preserves
        # single entry point and is decorated @spans_partitions so the
        # AST walker treats agent_id as deliberately fanned-out.
        if customer_id is None:
            rows = await self.l3_pool.fetch(
                "SELECT * FROM memories "
                "WHERE agent_id = ANY($1::uuid[]) AND user_id = $2 "
                "AND is_deleted = false "
                "ORDER BY date_created DESC",
                list(agent_ids),
                user_id,
            )
        else:
            rows = await self.l3_pool.fetch(
                "SELECT * FROM memories "
                "WHERE agent_id = ANY($1::uuid[]) AND customer_id = $2 "
                "AND user_id = $3 AND is_deleted = false "
                "ORDER BY date_created DESC",
                list(agent_ids),
                customer_id,
                user_id,
            )
        entities: list[MemoryEntity] = []
        for row in rows:
            data = dict(row)
            entity = self.entity_class(data, is_new=False, collection=self)
            entity.original_date_updated = data.get("date_updated")
            entity_id = (data["agent_id"], data["memory_id"])
            await self._save_to_l2(entity_id, data)
            entities.append(entity)
        return entities


class MediaCollection(SchemaBackedCollection[MediaEntity]):
    """three-tier collection for :class:`MediaEntity` (table ``media``).

    the media parent record carries a category discriminator and a
    JSONB metadata blob; child rows live in
    :class:`MediaContentCollection` and :class:`MemoryChunkCollection`.
    adopted under namespace-task-01 phase 8.5b — the v006 migration
    had no Collection before now. CRUD is generated from
    :attr:`schema` via :class:`SchemaBackedCollection`; no CAS path
    because the table has no ``date_updated`` fence column distinct
    from ``date_created``.
    """

    primary_key_column: str | tuple[str, ...] = ("agent_id", "media_id")
    schema = TableSchema(
        name="media",
        primary_key=("agent_id", "media_id"),
        columns=[
            Column("media_id", UUID_TYPE),
            Column("agent_id", UUID_TYPE, partition=True),
            Column("customer_id", UUID_TYPE),
            Column("user_id", UUID_TYPE),
            Column("media_category", STRING_TYPE),
            Column("metadata_json", JSONB_TYPE, nullable=True),
            Column("date_created", DATETIME_TYPE, immutable=True),
            Column("date_updated", DATETIME_TYPE),
        ],
    )

    @property
    def table_name(self) -> str:
        """return the database table name for this collection.

        :return: table name
        :rtype: str
        """
        return "media"

    @property
    def entity_class(self) -> type[MediaEntity]:
        """return the entity class for this collection.

        :return: entity class
        :rtype: type[MediaEntity]
        """
        return MediaEntity


class MediaContentCollection(SchemaBackedCollection[MediaContentEntity]):
    """three-tier collection for :class:`MediaContentEntity`.

    carries content rows attached to :class:`MediaEntity` through
    ``media_id``. hybrid-search surface
    (:meth:`hybrid_search`, :meth:`search_by_ids`) lives on this
    collection because vector/FTS queries are the primary way callers
    probe this table; by-ID pull-through is rare. CRUD is generated
    from :attr:`schema`; the embedding column uses ``VECTOR_TYPE`` so
    the generator emits ``$N::vector`` on the INSERT path and decodes
    the textual response back to ``list[float]`` on read.
    """

    primary_key_column: str | tuple[str, ...] = ("agent_id", "content_id")
    schema = TableSchema(
        name="media_content",
        primary_key=("agent_id", "content_id"),
        columns=[
            Column("content_id", UUID_TYPE),
            Column("media_id", UUID_TYPE, immutable=True),
            Column("agent_id", UUID_TYPE, partition=True),
            Column("customer_id", UUID_TYPE, immutable=True),
            Column("user_id", UUID_TYPE, immutable=True),
            Column("content_type", STRING_TYPE),
            Column("content", STRING_TYPE),
            Column("summary", STRING_TYPE, nullable=True),
            Column("embedding", VECTOR_TYPE, nullable=True),
            Column("date_created", DATETIME_TYPE, immutable=True),
        ],
    )

    @property
    def table_name(self) -> str:
        """return the database table name for this collection.

        :return: table name
        :rtype: str
        """
        return "media_content"

    @property
    def entity_class(self) -> type[MediaContentEntity]:
        """return the entity class for this collection.

        :return: entity class
        :rtype: type[MediaContentEntity]
        """
        return MediaContentEntity

    async def hybrid_search(
        self,
        *,
        user_id: UUID,
        agent_id: UUID,
        customer_id: UUID,
        embedding: list[float],
        user_text: str,
        top_k: int,
        candidate_limit: int,
        similarity_threshold: float,
        recency_half_life_hours: float,
        signal_weights: dict[str, float],
        fts_min_len: int = 3,
        fts_max_len: int = 500,
    ) -> list[dict[str, Any]]:
        """parallel vector + FTS hybrid search joining media_content to media.

        absorbs :meth:`MemoryRetriever._query_media_content` from the
        retrieval module. ``agent_id`` is the partition column on both
        media_content and media; the JOIN matches on the composite
        ``(agent_id, media_id)`` so the relationship cannot stretch
        across partitions. ``customer_id`` is a required sub-scope.

        :param user_id: owning user UUID (row filter)
        :ptype user_id: UUID
        :param agent_id: partition column on media_content + media; required
        :ptype agent_id: UUID
        :param customer_id: required sub-scope
        :ptype customer_id: UUID
        :param embedding: query embedding vector
        :ptype embedding: list[float]
        :param user_text: raw query text for FTS
        :ptype user_text: str
        :param top_k: number of candidates to return after ranking
        :ptype top_k: int
        :param candidate_limit: per-query candidate pool size
        :ptype candidate_limit: int
        :param similarity_threshold: floor on hybrid score
        :ptype similarity_threshold: float
        :param recency_half_life_hours: exponential decay half-life
        :ptype recency_half_life_hours: float
        :param signal_weights: mapping ``{"semantic", "keyword",
            "recency"}`` to weights
        :ptype signal_weights: dict[str, float]
        :param fts_min_len: minimum query length for FTS activation
        :ptype fts_min_len: int
        :param fts_max_len: truncation length for FTS queries
        :ptype fts_max_len: int
        :return: ranked candidate list, top_k entries
        :rtype: list[dict[str, Any]]
        """
        if self.l3_pool is None:
            return []
        embedding_str = json.dumps(embedding)

        scope_conditions, scope_params, param_offset = _build_user_scope_clause(
            user_id,
            agent_id=agent_id,
            customer_id=customer_id,
            start_param=2,
            table_prefix="mc",
        )
        limit_param = f"${param_offset + 1}"

        # cache-bypass: vector-distance search joining media_content ->
        # media. not primary-key-addressable on either side; L1 row
        # cache cannot help. method on Collection preserves single
        # entry point + audit hooks. composite JOIN on (agent_id,
        # media_id) keeps the relationship inside one partition.
        vec_coro = self.l3_pool.fetch(
            f"""
            SELECT mc.content_id, mc.content, mc.summary, mc.content_type,
                   mc.media_id, mc.date_created, mc.embedding,
                   med.media_category, med.metadata_json,
                   1 - (mc.embedding <=> $1::vector) AS similarity
            FROM media_content mc
            JOIN media med
              ON mc.agent_id = med.agent_id
             AND mc.media_id = med.media_id
            WHERE {scope_conditions} AND mc.embedding IS NOT NULL
            ORDER BY mc.embedding <=> $1::vector
            LIMIT {limit_param}
            """,
            embedding_str,
            *scope_params,
            candidate_limit,
        )

        fts_text = _build_fts_text(user_text, fts_min_len, fts_max_len)
        if fts_text:
            fts_scope_conditions, fts_scope_params, fts_param_offset = (
                _build_user_scope_clause(
                    user_id,
                    agent_id=agent_id,
                    customer_id=customer_id,
                    start_param=2,
                    table_prefix="mc",
                )
            )
            fts_limit_param = f"${fts_param_offset + 1}"
            # cache-bypass: FTS rank query joining media_content ->
            # media. see vec_coro justification above.
            fts_coro = self.l3_pool.fetch(
                f"""
                SELECT mc.content_id, mc.content, mc.summary, mc.content_type,
                       mc.media_id, mc.date_created, mc.embedding,
                       med.media_category, med.metadata_json,
                       ts_rank_cd(mc.search_vector, websearch_to_tsquery('english', $1)) AS fts_rank
                FROM media_content mc
                JOIN media med
                  ON mc.agent_id = med.agent_id
                 AND mc.media_id = med.media_id
                WHERE {fts_scope_conditions} AND mc.embedding IS NOT NULL
                  AND mc.search_vector @@ websearch_to_tsquery('english', $1)
                ORDER BY fts_rank DESC
                LIMIT {fts_limit_param}
                """,
                fts_text,
                *fts_scope_params,
                candidate_limit,
            )
            vec_rows, fts_rows = await asyncio.gather(vec_coro, fts_coro)
        else:
            vec_rows = await vec_coro
            fts_rows = []

        merged: dict[Any, dict[str, Any]] = {}
        for row in vec_rows:
            cid = row["content_id"]
            meta = row["metadata_json"]
            if isinstance(meta, str):
                meta = json.loads(meta)
            title = (
                (meta or {}).get("document_title")
                or (meta or {}).get("original_filename")
                or (meta or {}).get("title")
            )
            emb = row["embedding"]
            if isinstance(emb, str):
                emb = json.loads(emb)
            merged[cid] = {
                "content_id": cid,
                "content": row["content"],
                "summary": row["summary"],
                "content_type": row["content_type"],
                "media_id": str(row["media_id"]),
                "media_category": row["media_category"],
                "title": title,
                "date_created": row["date_created"],
                "similarity": float(row["similarity"]),
                "fts_rank": 0.0,
                "embedding": emb,
            }
        for row in fts_rows:
            cid = row["content_id"]
            if cid in merged:
                merged[cid]["fts_rank"] = float(row["fts_rank"])
            else:
                meta = row["metadata_json"]
                if isinstance(meta, str):
                    meta = json.loads(meta)
                title = (
                    (meta or {}).get("document_title")
                    or (meta or {}).get("original_filename")
                    or (meta or {}).get("title")
                )
                emb = row["embedding"]
                if isinstance(emb, str):
                    emb = json.loads(emb)
                merged[cid] = {
                    "content_id": cid,
                    "content": row["content"],
                    "summary": row["summary"],
                    "content_type": row["content_type"],
                    "media_id": str(row["media_id"]),
                    "media_category": row["media_category"],
                    "title": title,
                    "date_created": row["date_created"],
                    "similarity": 0.0,
                    "fts_rank": float(row["fts_rank"]),
                    "embedding": emb,
                }

        candidates = list(merged.values())
        if not candidates:
            return []

        _normalize_scores(candidates, "fts_rank")

        for c in candidates:
            recency = _recency_weight(c["date_created"], recency_half_life_hours)
            c["recency"] = round(recency, 4)
            c["hybrid_score"] = round(
                signal_weights["semantic"] * c["similarity"]
                + signal_weights["keyword"] * c["fts_rank"]
                + signal_weights["recency"] * recency,
                4,
            )

        filtered = [c for c in candidates if c["hybrid_score"] > similarity_threshold]
        filtered.sort(key=lambda c: c["hybrid_score"], reverse=True)
        return filtered[:top_k]

    async def search_by_ids(
        self,
        content_ids: list[UUID],
        user_id: UUID,
        *,
        agent_id: UUID,
    ) -> list[dict[str, Any]]:
        """batch lookup of media_content rows by primary key.

        joins to the parent :class:`MediaEntity` for ``media_category``
        + ``metadata_json``, scoped to ``(agent_id, user_id)``.

        :param content_ids: primary-key values to fetch
        :ptype content_ids: list[UUID]
        :param user_id: owning user UUID (row filter)
        :ptype user_id: UUID
        :param agent_id: partition column on media_content + media; required
        :ptype agent_id: UUID
        :return: list of row dicts
        :rtype: list[dict[str, Any]]
        """
        if self.l3_pool is None or not content_ids:
            return []
        # cache-bypass: batch IN(...) join spanning media_content ->
        # media. see :meth:`hybrid_search` for the detailed rationale.
        rows = await self.l3_pool.fetch(
            """
            SELECT mc.content_id, mc.content, mc.content_type, mc.media_id,
                   med.media_category, med.metadata_json, med.date_created
            FROM media_content mc
            JOIN media med
              ON mc.agent_id = med.agent_id
             AND mc.media_id = med.media_id
            WHERE mc.agent_id = $1
              AND mc.content_id = ANY($2::uuid[])
              AND mc.user_id = $3
            """,
            agent_id,
            content_ids,
            user_id,
        )
        return [dict(row) for row in rows]

    async def search_by_semantic(
        self,
        *,
        user_id: UUID,
        agent_id: UUID,
        embedding: list[float],
        max_results: int,
        similarity_threshold: float,
    ) -> list[dict[str, Any]]:
        """vector-only semantic search (memory_search tool leg).

        :param user_id: owning user UUID
        :ptype user_id: UUID
        :param agent_id: partition column on media_content + media; required
        :ptype agent_id: UUID
        :param embedding: query embedding vector
        :ptype embedding: list[float]
        :param max_results: cap on returned rows
        :ptype max_results: int
        :param similarity_threshold: similarity floor
        :ptype similarity_threshold: float
        :return: list of row dicts
        :rtype: list[dict[str, Any]]
        """
        if self.l3_pool is None:
            return []
        embedding_str = json.dumps(embedding)
        # cache-bypass: vector-distance search joining media_content ->
        # media. see :meth:`hybrid_search`. composite JOIN on
        # (agent_id, media_id) keeps the relationship inside one
        # partition.
        rows = await self.l3_pool.fetch(
            """
            SELECT mc.content_id, mc.content, mc.content_type,
                   mc.media_id, med.media_category, med.metadata_json,
                   med.date_created,
                   1 - (mc.embedding <=> $1::vector) AS similarity
            FROM media_content mc
            JOIN media med
              ON mc.agent_id = med.agent_id
             AND mc.media_id = med.media_id
            WHERE mc.agent_id = $2
              AND mc.user_id = $3
              AND mc.embedding IS NOT NULL
            ORDER BY mc.embedding <=> $1::vector
            LIMIT $4
            """,
            embedding_str,
            agent_id,
            user_id,
            max_results,
        )
        result: list[dict[str, Any]] = []
        for row in rows:
            sim = float(row["similarity"])
            if sim <= similarity_threshold:
                continue
            result.append(
                {
                    "content_id": str(row["content_id"]),
                    "content": row["content"],
                    "content_type": row["content_type"],
                    "media_id": str(row["media_id"]),
                    "media_category": row["media_category"],
                    "metadata_json": row["metadata_json"],
                    "date_created": row["date_created"],
                    "similarity": sim,
                },
            )
        return result

    async def search_by_fts(
        self,
        *,
        user_id: UUID,
        agent_id: UUID,
        fts_text: str,
        max_results: int,
    ) -> list[dict[str, Any]]:
        """FTS keyword search for the memory_search tool.

        :param user_id: owning user UUID
        :ptype user_id: UUID
        :param agent_id: partition column on media_content + media; required
        :ptype agent_id: UUID
        :param fts_text: FTS query text
        :ptype fts_text: str
        :param max_results: cap on returned rows
        :ptype max_results: int
        :return: list of row dicts
        :rtype: list[dict[str, Any]]
        """
        if self.l3_pool is None:
            return []
        # cache-bypass: FTS rank query joining media_content -> media.
        rows = await self.l3_pool.fetch(
            """
            SELECT mc.content_id, mc.content, mc.content_type,
                   mc.media_id, med.media_category, med.metadata_json,
                   med.date_created,
                   ts_rank_cd(mc.search_vector, websearch_to_tsquery('english', $1)) AS fts_rank
            FROM media_content mc
            JOIN media med
              ON mc.agent_id = med.agent_id
             AND mc.media_id = med.media_id
            WHERE mc.agent_id = $2
              AND mc.user_id = $3
              AND mc.embedding IS NOT NULL
              AND mc.search_vector @@ websearch_to_tsquery('english', $1)
            ORDER BY fts_rank DESC
            LIMIT $4
            """,
            fts_text,
            agent_id,
            user_id,
            max_results,
        )
        return [
            {
                "content_id": str(row["content_id"]),
                "content": row["content"],
                "content_type": row["content_type"],
                "media_id": str(row["media_id"]),
                "media_category": row["media_category"],
                "metadata_json": row["metadata_json"],
                "date_created": row["date_created"],
            }
            for row in rows
        ]

    async def fetch_content_for_recall(
        self,
        *,
        content_id: UUID,
        user_id: UUID,
        agent_id: UUID,
    ) -> str | None:
        """fetch ``content`` for the recall_memory tool (media leg).

        :param content_id: media_content primary-key value
        :ptype content_id: UUID
        :param user_id: owning user UUID (row filter)
        :ptype user_id: UUID
        :param agent_id: partition column on media_content; required
        :ptype agent_id: UUID
        :return: content text or ``None``
        :rtype: str | None
        """
        if self.l3_pool is None:
            return None
        # cache-bypass: by-ID fetch scoped by (agent_id, user_id) —
        # both are security predicates the L1 cache cannot enforce.
        row = await self.l3_pool.fetchrow(
            "SELECT content FROM media_content "
            "WHERE agent_id = $1 AND content_id = $2 AND user_id = $3",
            agent_id,
            content_id,
            user_id,
        )
        if row is None:
            return None
        result: str = row["content"]
        return result


class MemoryChunkCollection(SchemaBackedCollection[MemoryChunkEntity]):
    """three-tier collection for :class:`MemoryChunkEntity`.

    document-style chunks with heading / page metadata; optional
    parent :class:`MediaEntity` through ``media_id``. CRUD is generated
    from :attr:`schema`; the embedding column uses ``VECTOR_TYPE`` for
    pgvector cast on write + ``list[float]`` decode on read.
    """

    primary_key_column: str | tuple[str, ...] = ("agent_id", "chunk_id")
    schema = TableSchema(
        name="memory_chunks",
        primary_key=("agent_id", "chunk_id"),
        columns=[
            Column("chunk_id", UUID_TYPE),
            Column("media_id", UUID_TYPE, nullable=True, immutable=True),
            Column("agent_id", UUID_TYPE, partition=True),
            Column("customer_id", UUID_TYPE, immutable=True),
            Column("user_id", UUID_TYPE, immutable=True),
            Column("content", STRING_TYPE),
            Column("summary", STRING_TYPE, nullable=True),
            Column("heading_context", STRING_TYPE, nullable=True),
            Column("page_number", INT_TYPE, nullable=True),
            Column("embedding", VECTOR_TYPE, nullable=True),
            Column("date_created", DATETIME_TYPE, immutable=True),
        ],
    )

    @property
    def table_name(self) -> str:
        """return the database table name for this collection.

        :return: table name
        :rtype: str
        """
        return "memory_chunks"

    @property
    def entity_class(self) -> type[MemoryChunkEntity]:
        """return the entity class for this collection.

        :return: entity class
        :rtype: type[MemoryChunkEntity]
        """
        return MemoryChunkEntity

    async def hybrid_search(
        self,
        *,
        user_id: UUID,
        agent_id: UUID,
        customer_id: UUID,
        embedding: list[float],
        user_text: str,
        candidate_k: int,
        similarity_threshold: float,
        chunk_signal_weights: dict[str, float],
        fts_min_len: int = 3,
        fts_max_len: int = 500,
    ) -> list[dict[str, Any]]:
        """parallel vector + FTS search against memory_chunks + media.

        absorbs :meth:`MemoryRetriever._query_chunks`. two-signal
        ranking (semantic + keyword); no recency decay because chunks
        come from documents whose freshness signal is the upload
        event, not the chunk's own lifecycle. ``agent_id`` is the
        partition column on memory_chunks + media; ``customer_id`` is
        a required sub-scope.

        :param user_id: owning user UUID
        :ptype user_id: UUID
        :param agent_id: partition column; required
        :ptype agent_id: UUID
        :param customer_id: required sub-scope
        :ptype customer_id: UUID
        :param embedding: query embedding vector
        :ptype embedding: list[float]
        :param user_text: raw query text for FTS
        :ptype user_text: str
        :param candidate_k: per-query candidate pool size
        :ptype candidate_k: int
        :param similarity_threshold: floor on hybrid score
        :ptype similarity_threshold: float
        :param chunk_signal_weights: mapping ``{"semantic",
            "keyword"}`` to weights
        :ptype chunk_signal_weights: dict[str, float]
        :param fts_min_len: minimum query length for FTS activation
        :ptype fts_min_len: int
        :param fts_max_len: truncation length for FTS queries
        :ptype fts_max_len: int
        :return: ranked candidate list
        :rtype: list[dict[str, Any]]
        """
        if self.l3_pool is None:
            return []
        embedding_str = json.dumps(embedding)

        scope_conditions, scope_params, param_offset = _build_user_scope_clause(
            user_id,
            agent_id=agent_id,
            customer_id=customer_id,
            start_param=2,
            table_prefix="mc",
        )
        limit_param = f"${param_offset + 1}"

        # cache-bypass: vector-distance search joining memory_chunks ->
        # media. not primary-key-addressable; L1 row cache cannot help.
        # method on Collection preserves single entry point. composite
        # LEFT JOIN on (agent_id, media_id) keeps the optional
        # relationship inside one partition when present.
        vec_coro = self.l3_pool.fetch(
            f"""
            SELECT mc.chunk_id, mc.content, mc.summary, mc.heading_context,
                   mc.page_number, mc.media_id,
                   mc.embedding, med.metadata_json,
                   1 - (mc.embedding <=> $1::vector) AS similarity
            FROM memory_chunks mc
            LEFT JOIN media med
              ON mc.agent_id = med.agent_id
             AND mc.media_id = med.media_id
            WHERE {scope_conditions}
            ORDER BY mc.embedding <=> $1::vector
            LIMIT {limit_param}
            """,
            embedding_str,
            *scope_params,
            candidate_k,
        )

        fts_text = _build_fts_text(user_text, fts_min_len, fts_max_len)
        if fts_text:
            fts_scope_conditions, fts_scope_params, fts_param_offset = (
                _build_user_scope_clause(
                    user_id,
                    agent_id=agent_id,
                    customer_id=customer_id,
                    start_param=2,
                    table_prefix="mc",
                )
            )
            fts_limit_param = f"${fts_param_offset + 1}"
            # cache-bypass: FTS rank query joining memory_chunks -> media.
            fts_coro = self.l3_pool.fetch(
                f"""
                SELECT mc.chunk_id, mc.content, mc.summary, mc.heading_context,
                       mc.page_number, mc.media_id,
                       mc.embedding, med.metadata_json,
                       ts_rank_cd(mc.search_vector, websearch_to_tsquery('english', $1)) AS fts_rank
                FROM memory_chunks mc
                LEFT JOIN media med
                  ON mc.agent_id = med.agent_id
                 AND mc.media_id = med.media_id
                WHERE {fts_scope_conditions}
                  AND mc.search_vector @@ websearch_to_tsquery('english', $1)
                ORDER BY fts_rank DESC
                LIMIT {fts_limit_param}
                """,
                fts_text,
                *fts_scope_params,
                candidate_k,
            )
            vec_rows, fts_rows = await asyncio.gather(vec_coro, fts_coro)
        else:
            vec_rows = await vec_coro
            fts_rows = []

        merged: dict[Any, dict[str, Any]] = {}
        for row in vec_rows:
            ckid = row["chunk_id"]
            title = None
            meta = row["metadata_json"]
            if meta:
                if isinstance(meta, str):
                    meta = json.loads(meta)
                title = meta.get("document_title") or meta.get("original_filename")
            emb = row["embedding"]
            if isinstance(emb, str):
                emb = json.loads(emb)
            merged[ckid] = {
                "chunk_id": ckid,
                "content": row["content"],
                "summary": row["summary"],
                "heading_context": row["heading_context"],
                "page_number": row["page_number"],
                "media_id": str(row["media_id"]) if row["media_id"] else None,
                "title": title,
                "similarity": float(row["similarity"]),
                "fts_rank": 0.0,
                "embedding": emb,
            }
        for row in fts_rows:
            ckid = row["chunk_id"]
            if ckid in merged:
                merged[ckid]["fts_rank"] = float(row["fts_rank"])
            else:
                title = None
                meta = row["metadata_json"]
                if meta:
                    if isinstance(meta, str):
                        meta = json.loads(meta)
                    title = meta.get("document_title") or meta.get("original_filename")
                emb = row["embedding"]
                if isinstance(emb, str):
                    emb = json.loads(emb)
                merged[ckid] = {
                    "chunk_id": ckid,
                    "content": row["content"],
                    "summary": row["summary"],
                    "heading_context": row["heading_context"],
                    "page_number": row["page_number"],
                    "media_id": str(row["media_id"]) if row["media_id"] else None,
                    "title": title,
                    "similarity": 0.0,
                    "fts_rank": float(row["fts_rank"]),
                    "embedding": emb,
                }

        candidates = list(merged.values())
        if not candidates:
            return []

        _normalize_scores(candidates, "fts_rank")

        for c in candidates:
            c["hybrid_score"] = round(
                chunk_signal_weights["semantic"] * c["similarity"]
                + chunk_signal_weights["keyword"] * c["fts_rank"],
                4,
            )

        filtered = [c for c in candidates if c["hybrid_score"] > similarity_threshold]
        filtered.sort(key=lambda c: c["hybrid_score"], reverse=True)
        return filtered

    async def search_by_ids(
        self,
        chunk_ids: list[UUID],
        user_id: UUID,
        *,
        agent_id: UUID,
    ) -> list[dict[str, Any]]:
        """batch lookup of memory_chunks rows by primary key.

        :param chunk_ids: primary-key values to fetch
        :ptype chunk_ids: list[UUID]
        :param user_id: owning user UUID (row filter)
        :ptype user_id: UUID
        :param agent_id: partition column on memory_chunks; required
        :ptype agent_id: UUID
        :return: list of row dicts joined to media
        :rtype: list[dict[str, Any]]
        """
        if self.l3_pool is None or not chunk_ids:
            return []
        # cache-bypass: batch IN(...) left-join spanning memory_chunks
        # -> media. see :meth:`hybrid_search`. composite LEFT JOIN on
        # (agent_id, media_id) keeps optional parent in same partition.
        rows = await self.l3_pool.fetch(
            """
            SELECT mc.chunk_id, mc.content, mc.heading_context, mc.page_number,
                   med.metadata_json
            FROM memory_chunks mc
            LEFT JOIN media med
              ON mc.agent_id = med.agent_id
             AND mc.media_id = med.media_id
            WHERE mc.agent_id = $1
              AND mc.chunk_id = ANY($2::uuid[])
              AND mc.user_id = $3
            """,
            agent_id,
            chunk_ids,
            user_id,
        )
        return [dict(row) for row in rows]

    async def search_by_semantic(
        self,
        *,
        user_id: UUID,
        agent_id: UUID,
        embedding: list[float],
        max_results: int,
        similarity_threshold: float,
    ) -> list[dict[str, Any]]:
        """vector-only semantic search (memory_search tool leg).

        :param user_id: owning user UUID
        :ptype user_id: UUID
        :param agent_id: partition column on memory_chunks + media; required
        :ptype agent_id: UUID
        :param embedding: query embedding vector
        :ptype embedding: list[float]
        :param max_results: cap on returned rows
        :ptype max_results: int
        :param similarity_threshold: similarity floor
        :ptype similarity_threshold: float
        :return: list of row dicts
        :rtype: list[dict[str, Any]]
        """
        if self.l3_pool is None:
            return []
        embedding_str = json.dumps(embedding)
        # cache-bypass: vector-distance search joining memory_chunks
        # -> media. see :meth:`hybrid_search`.
        rows = await self.l3_pool.fetch(
            """
            SELECT mc.chunk_id, mc.content, mc.heading_context, mc.page_number,
                   med.metadata_json,
                   1 - (mc.embedding <=> $1::vector) AS similarity
            FROM memory_chunks mc
            LEFT JOIN media med
              ON mc.agent_id = med.agent_id
             AND mc.media_id = med.media_id
            WHERE mc.agent_id = $2
              AND mc.user_id = $3
            ORDER BY mc.embedding <=> $1::vector
            LIMIT $4
            """,
            embedding_str,
            agent_id,
            user_id,
            max_results,
        )
        result: list[dict[str, Any]] = []
        for row in rows:
            sim = float(row["similarity"])
            if sim <= similarity_threshold:
                continue
            result.append(
                {
                    "chunk_id": str(row["chunk_id"]),
                    "content": row["content"],
                    "heading_context": row["heading_context"],
                    "page_number": row["page_number"],
                    "metadata_json": row["metadata_json"],
                    "similarity": sim,
                },
            )
        return result

    async def fetch_content_for_recall(
        self,
        *,
        chunk_id: UUID,
        user_id: UUID,
        agent_id: UUID,
    ) -> str | None:
        """fetch ``content`` for the recall_memory tool (chunk leg).

        :param chunk_id: chunk primary-key value
        :ptype chunk_id: UUID
        :param user_id: owning user UUID (row filter)
        :ptype user_id: UUID
        :param agent_id: partition column on memory_chunks; required
        :ptype agent_id: UUID
        :return: chunk text or ``None``
        :rtype: str | None
        """
        if self.l3_pool is None:
            return None
        # cache-bypass: by-ID fetch scoped by (agent_id, user_id) —
        # both are security predicates the L1 cache cannot enforce.
        row = await self.l3_pool.fetchrow(
            "SELECT content FROM memory_chunks "
            "WHERE agent_id = $1 AND chunk_id = $2 AND user_id = $3",
            agent_id,
            chunk_id,
            user_id,
        )
        if row is None:
            return None
        result: str = row["content"]
        return result


_MEMORY_REF_SHORT_DESC_MAX = 150


class MemoryRefsCollection(SchemaBackedCollection[MemoryRefEntity]):
    """three-tier collection for :class:`MemoryRefEntity`.

    namespace-task-01 phase 8.5l-2: retires :class:`MemoryLedger` — the
    bespoke wrapper that sat on top of ``SQLiteBackend`` with hand-rolled
    pool.fetch / pool.execute against ``conversation_memory_refs``. on
    top of 8.5l-1's composite-pk :class:`BaseCollection` support, the
    table fits the Collection contract natively:
    ``primary_key_column = ("conversation_id", "item_id")``. the L2
    invalidation envelope carries ``ids: [<conversation_id>,
    <item_id>]`` per 8.5l-1's wire format.

    tier configuration: L1 (SQLite) + L2 (NATS KV) + L3 (postgres /
    ``NatsProxyL3Backend``). full three-tier CRUD via
    :meth:`save_entity` / :meth:`get` / :meth:`delete`. per-conversation
    multi-row scans land on :meth:`find_by_conversation` with an
    explicit ``# cache-bypass:`` annotation documenting the query is
    not primary-key-addressable.

    CRUD is generated from :attr:`schema` with composite pk
    ``(conversation_id, item_id)`` and no CAS column. ``short_desc``
    is domain-truncated to 150 chars in :meth:`save_to_postgres` to
    match the migration-v002 VARCHAR(150) bound.
    """

    primary_key_column: str | tuple[str, ...] = ("conversation_id", "item_id")
    # rationale: ``save_to_postgres`` is a framework override that
    # truncates ``short_desc`` to the migration-v002 VARCHAR(150) bound;
    # the underlying ``data`` dict already carries ``conversation_id``
    # (the partition column) by construction since
    # :class:`SchemaBackedCollection` enforces required-column presence
    # before this override runs. exempting the framework override
    # keeps the partition contract on the read surface
    # (``find_by_conversation``) without weakening the static guard.
    _partition_exempt_methods: ClassVar[frozenset[str]] = frozenset({"save_to_postgres"})
    schema = TableSchema(
        name="conversation_memory_refs",
        primary_key=("conversation_id", "item_id"),
        columns=[
            Column("conversation_id", UUID_TYPE, partition=True),
            Column("item_id", UUID_TYPE),
            Column("item_type", STRING_TYPE),
            Column("short_desc", STRING_TYPE),
            Column("date_added", DATETIME_TYPE),
        ],
    )

    @property
    def table_name(self) -> str:
        """return database table name.

        :return: table name
        :rtype: str
        """
        return "conversation_memory_refs"

    @property
    def entity_class(self) -> type[MemoryRefEntity]:
        """return entity class for this collection.

        :return: entity class
        :rtype: type[MemoryRefEntity]
        """
        return MemoryRefEntity

    async def save_to_postgres(
        self, data: dict[str, Any], original_timestamp: datetime | None = None,
    ) -> int:
        """upsert one row into L3 via composite-pk ON CONFLICT.

        truncate ``short_desc`` to :data:`_MEMORY_REF_SHORT_DESC_MAX`
        (150) chars to match the migration-v002 VARCHAR(150) bound.
        the truncation is collection-specific domain logic, not
        primitive-level coercion, so it lives on this override rather
        than inside :class:`SchemaBackedCollection`.

        :param data: row data; must contain both pk columns plus
            ``item_type`` / ``short_desc`` / ``date_added``
        :ptype data: dict[str, Any]
        :param original_timestamp: ignored (no CAS column on this table)
        :ptype original_timestamp: datetime | None
        :return: rows affected (1 on success, 0 on failure)
        :rtype: int
        """
        desc_value = data.get("short_desc")
        if isinstance(desc_value, str) and len(desc_value) > _MEMORY_REF_SHORT_DESC_MAX:
            data = dict(data)
            data["short_desc"] = desc_value[:_MEMORY_REF_SHORT_DESC_MAX]
        return await super().save_to_postgres(data, original_timestamp)

    async def find_by_conversation(
        self, conversation_id: UUID,
    ) -> list[MemoryRefEntity]:
        """fetch every ref for a conversation, ordered by ``date_added`` asc.

        absorbs the former ``MemoryLedger.load(pool, conversation_id)``
        SQL into a Collection method. multi-row scan is not primary-
        key-addressable: the query touches every row sharing the
        ``conversation_id`` prefix of the composite pk, so L1 row-level
        cache does not apply. each hit is promoted into L2 so other
        pods starting cold can resolve the row without an L3 round-
        trip.

        :param conversation_id: conversation UUID to scan for
        :ptype conversation_id: UUID
        :return: list of ref entities in chronological order
        :rtype: list[MemoryRefEntity]
        """
        if self.l3_pool is None:
            return []
        # cache-bypass: multi-row scan by ``conversation_id`` (first
        # column of the composite pk) is not primary-key-addressable.
        # L1 row cache serves per-``(conversation_id, item_id)`` lookups
        # only; a prefix scan would not hit. keeping the query on the
        # Collection preserves the single entry point + lets callers
        # share L2 invalidation channels.
        rows = await self.l3_pool.fetch(
            """
            SELECT conversation_id, item_id, item_type, short_desc, date_added
            FROM conversation_memory_refs
            WHERE conversation_id = $1
            ORDER BY date_added ASC
            """,
            conversation_id,
        )
        entities: list[MemoryRefEntity] = []
        for row in rows:
            data = self._coerce_row(dict(row))
            entity = self.entity_class(data, is_new=False, collection=self)
            entity_id = (data["conversation_id"], data["item_id"])
            await self._save_to_l2(entity_id, data)
            entities.append(entity)
        return entities
