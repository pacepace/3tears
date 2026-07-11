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
from typing import Any, ClassVar, cast
from uuid import UUID

from sqlalchemy import MetaData, Table

from threetears.core.collections.flush import WriteBuffer
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import (
    BOOL_TYPE,
    DATETIMETZ_TYPE,
    ENUM_TYPE,
    INT_TYPE,
    JSONB_TYPE,
    NUMERIC_TYPE,
    STRING_TYPE,
    TSVECTOR_TYPE,
    UUID_TYPE,
    VECTOR_TYPE,
    Column,
    ForeignKey as SchemaForeignKey,
    Index as SchemaIndex,
    SchemaBackedCollection,
    TableSchema,
    UniqueConstraint as SchemaUniqueConstraint,
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
    "conversation_memory_refs_table",
    "media_content_table",
    "media_table",
    "memories_table",
    "memory_chunks_table",
]

log = get_logger(__name__)


def conversation_memory_refs_table(metadata: MetaData) -> Table:
    """Register the ``conversation_memory_refs`` table on ``metadata``.

    v0.8.0: schema declaration is now the single source of truth. This
    factory is a thin idempotency wrapper around
    :meth:`MemoryRefsCollection.schema.to_sqlalchemy_table`. Call this
    factory before ``SQLiteCacheManager.initialize(metadata)`` so the
    L1 cache builds with the full schema; without it,
    :class:`MemoryRefsCollection.save_entity` fails at the L1 boundary
    with ``no such table: conversation_memory_refs`` even though the
    L3 Postgres write succeeds.

    :param metadata: SQLAlchemy metadata to attach the table to
    :ptype metadata: MetaData
    :return: the ``conversation_memory_refs`` :class:`Table`
    :rtype: Table
    """
    return cast(Table, MemoryRefsCollection.schema.to_sqlalchemy_table(metadata))


# ---------------------------------------------------------------------------
# v0.8.0 cross-package table factories.
#
# Each factory below is a thin idempotency wrapper around
# ``<Collection>.schema.to_sqlalchemy_table(metadata)``. The
# ``TableSchema`` declaration on the Collection is the single source
# of truth for the SQLAlchemy shape (columns, types, primary key,
# foreign keys, indexes, enum constraints, vector dimensions, server
# defaults).
#
# Before v0.8.0, each factory hand-wrote the SQLAlchemy
# ``Table(...)`` declaration alongside the Collection's
# ``TableSchema`` — two declarations of the same shape inside one
# file. v0.8.0 enriched ``TableSchema`` (shards 01-03) so the factory
# bodies can delegate to ``to_sqlalchemy_table`` (shard 04), closing
# the duplication trap that the v0.7.5 factories themselves were
# introduced to close at the cross-package boundary.
#
# Public factory signatures are unchanged so host applications
# (current and future consumers) keep calling ``memories_table(metadata)``
# etc. without modification.
# ---------------------------------------------------------------------------


# Single source of truth for the embedding dimension carried by memory
# tables. The value matches every Vector column in this file -- bump
# here to bump everywhere if an embedding provider with a different
# native dim is ever adopted. The 1024 value matches Voyage AI's
# voyage-4 default + the pre-v0.7.5 hardcoded ``Vector(1024)`` in
# the upstream declarations.
_MEMORY_VECTOR_DIM = 1024


def memories_table(metadata: MetaData) -> Table:
    """Register the ``memories`` table on the given SA metadata.

    v0.8.0: schema declaration is now the single source of truth. This
    factory is a thin idempotency wrapper around
    :meth:`MemoriesCollection.schema.to_sqlalchemy_table`. Call this
    before ``SQLiteBackend.initialize(metadata)`` so the L1 cache gets
    the correct schema, and before Alembic ``target_metadata``
    reflection so auto-generate sees the same shape.

    :param metadata: SQLAlchemy metadata to attach the table to
    :ptype metadata: MetaData
    :return: the ``memories`` :class:`Table`
    :rtype: Table
    """
    return cast(Table, MemoriesCollection.schema.to_sqlalchemy_table(metadata))


def media_table(metadata: MetaData) -> Table:
    """Register the ``media`` table on the given SA metadata.

    v0.8.0: schema declaration is now the single source of truth. This
    factory is a thin idempotency wrapper around
    :meth:`MediaCollection.schema.to_sqlalchemy_table`. Includes the
    v0.14.0-unified ``memory_id`` FK (every media row attaches to a
    memory) with CASCADE-on-memory-delete and the four indexes.

    :param metadata: SQLAlchemy metadata to attach the table to
    :ptype metadata: MetaData
    :return: the ``media`` :class:`Table`
    :rtype: Table
    """
    return cast(Table, MediaCollection.schema.to_sqlalchemy_table(metadata))


def media_content_table(metadata: MetaData) -> Table:
    """Register the ``media_content`` table on the given SA metadata.

    v0.8.0: schema declaration is now the single source of truth. This
    factory is a thin idempotency wrapper around
    :meth:`MediaContentCollection.schema.to_sqlalchemy_table`. Carries
    derived content (descriptions, transcripts, OCR text) for ``media``
    rows with its own embedding + search_vector columns.

    :param metadata: SQLAlchemy metadata to attach the table to
    :ptype metadata: MetaData
    :return: the ``media_content`` :class:`Table`
    :rtype: Table
    """
    return cast(Table, MediaContentCollection.schema.to_sqlalchemy_table(metadata))


def memory_chunks_table(metadata: MetaData) -> Table:
    """Register the ``memory_chunks`` table on the given SA metadata.

    v0.8.0: schema declaration is now the single source of truth. This
    factory is a thin idempotency wrapper around
    :meth:`MemoryChunkCollection.schema.to_sqlalchemy_table`. Carries
    chunked text + embeddings parented to a ``memories`` row, plus the
    v0.14.0 transcript-chunk ``message_id_start`` / ``message_id_end``
    provenance columns.

    :param metadata: SQLAlchemy metadata to attach the table to
    :ptype metadata: MetaData
    :return: the ``memory_chunks`` :class:`Table`
    :rtype: Table
    """
    return cast(Table, MemoryChunkCollection.schema.to_sqlalchemy_table(metadata))


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

    :param created: creation timestamp; must be timezone-aware UTC
        (every datetime in the platform is aware-UTC after
        collections-task-05; passing naive raises TypeError on the
        ``now - created`` subtract)
    :ptype created: datetime
    :param half_life_hours: half-life in hours
    :ptype half_life_hours: float
    :return: decay factor in (0, 1]
    :rtype: float
    """
    now = datetime.now(UTC)
    hours_ago = max((now - created).total_seconds() / 3600, 0.0)
    return math.exp(-hours_ago / half_life_hours)


def _chunk_row_to_dict(row: Any, score_key: str, score_value: float) -> dict[str, Any]:
    """shape a single chunk SQL row into the canonical result dict.

    Called from both :meth:`MemoryChunkCollection.hybrid_search` and
    :meth:`MemoryChunkCollection.hybrid_search_within_memory`. The
    result shape carries the chunk's own columns plus the optional
    grandparent ``media_id`` + ``metadata_json`` from the LEFT JOIN.

    :param row: asyncpg Record from the chunk SQL
    :ptype row: Any
    :param score_key: which score field to seed (``similarity`` or
        ``fts_rank``); the other defaults to 0.0 so the merge step
        can fold the symmetric query in.
    :ptype score_key: str
    :param score_value: numeric value for ``score_key``
    :ptype score_value: float
    :return: result dict with chunk + media columns + the two scoring
        fields
    :rtype: dict[str, Any]
    """
    title = None
    meta = row["metadata_json"]
    if meta:
        if isinstance(meta, str):
            meta = json.loads(meta)
        title = meta.get("document_title") or meta.get("original_filename")
    emb = row["embedding"]
    if isinstance(emb, str):
        emb = json.loads(emb)
    return {
        "chunk_id": row["chunk_id"],
        "content": row["content"],
        "summary": row["summary"],
        "heading_context": row["heading_context"],
        "page_number": row["page_number"],
        "memory_id": str(row["memory_id"]),
        "media_id": str(row["media_id"]) if row["media_id"] else None,
        "message_id_start": (str(row["message_id_start"]) if row["message_id_start"] else None),
        "message_id_end": (str(row["message_id_end"]) if row["message_id_end"] else None),
        "title": title,
        "similarity": score_value if score_key == "similarity" else 0.0,
        "fts_rank": score_value if score_key == "fts_rank" else 0.0,
        "embedding": emb,
    }


def _merge_chunk_search_rows(
    *,
    vec_rows: list[Any],
    fts_rows: list[Any],
    chunk_signal_weights: dict[str, float],
    similarity_threshold: float,
) -> list[dict[str, Any]]:
    """fold the parallel vec + FTS chunk query outputs into a ranked list.

    Used by :meth:`MemoryChunkCollection.hybrid_search` and
    :meth:`MemoryChunkCollection.hybrid_search_within_memory`. Two-
    signal ranking (semantic + keyword); no recency decay because
    chunks come from documents / transcripts whose freshness signal
    is the source artifact, not the chunk's own lifecycle.

    :param vec_rows: rows from the vector-distance query
    :ptype vec_rows: list[Any]
    :param fts_rows: rows from the FTS rank query (may be empty)
    :ptype fts_rows: list[Any]
    :param chunk_signal_weights: mapping ``{"semantic", "keyword"}``
        to combination weights
    :ptype chunk_signal_weights: dict[str, float]
    :param similarity_threshold: floor on ``hybrid_score`` for the
        final list
    :ptype similarity_threshold: float
    :return: ranked list of chunk dicts, threshold-filtered, sorted
        by hybrid_score DESC
    :rtype: list[dict[str, Any]]
    """
    merged: dict[Any, dict[str, Any]] = {}
    for row in vec_rows:
        merged[row["chunk_id"]] = _chunk_row_to_dict(row, score_key="similarity", score_value=float(row["similarity"]))
    for row in fts_rows:
        ckid = row["chunk_id"]
        if ckid in merged:
            merged[ckid]["fts_rank"] = float(row["fts_rank"])
        else:
            merged[ckid] = _chunk_row_to_dict(row, score_key="fts_rank", score_value=float(row["fts_rank"]))

    candidates = list(merged.values())
    if not candidates:
        return []

    _normalize_scores(candidates, "fts_rank")

    for c in candidates:
        c["hybrid_score"] = round(
            chunk_signal_weights["semantic"] * c["similarity"] + chunk_signal_weights["keyword"] * c["fts_rank"],
            4,
        )

    filtered = [c for c in candidates if c["hybrid_score"] > similarity_threshold]
    filtered.sort(key=lambda c: c["hybrid_score"], reverse=True)
    return filtered


# explicit column list for raw SELECTs over the ``memories`` table. the
# ``embedding`` (pgvector) column is cast to ``::text`` so asyncpg can decode
# it on the no-codec L3 pool -- a bare ``SELECT *`` returns the raw ``vector``
# type (OID 8078) that asyncpg has no codec for and raises
# ``UnsupportedClientFeatureError``. the entity deserializer parses the
# bracketed-text form. mirrors the schema-backed generated SELECT's ::text
# cast; keep in sync with the memories migration column set.
_MEMORIES_SELECT_COLUMNS = (
    "memory_id, agent_id, customer_id, user_id, type_memory, content, "
    "date_created, date_updated, embedding::text AS embedding, "
    "conversation_id, message_id_source, summary, search_vector, alias, "
    # v024 salience substrate — keep in sync with the memories migration
    # column set so raw-SQL read paths hydrate the full entity.
    "salience, last_decayed_at, last_accessed, evergreen, superseded_by"
)


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
    (pgvector ``::text::public.vector`` cast + list[float] decode), ``date_updated``
    is the CAS fence so concurrent writers race correctly, and the
    scope columns (agent/customer/user/conversation/message_id_source/
    type_memory/date_created) are marked immutable so the
    ``DO UPDATE SET`` clause narrows to content + embedding +
    date_updated.

    Under the unified data model (v015 - v019) deletion is hard only:
    the legacy ``is_deleted`` / ``date_deleted`` columns were removed
    in v018, the ``media_id`` reverse-direction FK column was removed
    in v018 (media now parents to memory), and ``conversation_id``
    became NOT NULL in v019. ``soft_delete`` was retired with the
    columns; callers use ``delete`` for hard removal (CASCADE
    propagates to chunks + media via v017 FKs).
    """

    primary_key_column: str | tuple[str, ...] = ("agent_id", "memory_id")
    # v0.8.0 enrichment: TableSchema is now the single source of truth
    # for the SQLAlchemy registration. Mirrors the v0.7.5
    # ``memories_table`` factory output plus the ``ix_memories_user_alias``
    # partial unique index relocated from upstream alembic 088 (the
    # per-user uniqueness on ``alias`` is intrinsic to the column
    # contract, not a deployment choice). The ``summary`` and
    # ``search_vector`` columns mirror prod -- ``search_vector`` is
    # populated server-side by the memories FTS trigger (3tears
    # migration v005); declared ``immutable=True`` so the UPDATE
    # generators exclude it from SET clauses.
    schema = TableSchema(
        name="memories",
        primary_key=("agent_id", "memory_id"),
        columns=[
            Column("memory_id", UUID_TYPE),
            Column("agent_id", UUID_TYPE, partition=True),
            # customer_id / user_id relaxed to nullable in v024 so the
            # memory primitive supports all three scope grains (agent /
            # customer / user). metallm enforces NOT NULL at its own
            # consumer layer; a null here means an agent- or customer-
            # scoped row (e.g. a shared-knowledge gist).
            Column("customer_id", UUID_TYPE, immutable=True, nullable=True),
            Column(
                "user_id",
                UUID_TYPE,
                immutable=True,
                nullable=True,
                foreign_key=("users", "user_id"),
            ),
            Column("conversation_id", UUID_TYPE, immutable=True),
            # message_id_source FK lives at table level with
            # on_delete="SET NULL" to match prod. The inline
            # ``foreign_key=`` 2-tuple form emits NO ACTION which
            # diverges from prod and would surface as a parity-gate
            # phantom. Per v0.8.0 locked decision: inline form for
            # NO ACTION, table-level for everything else.
            Column(
                "message_id_source",
                UUID_TYPE,
                immutable=True,
                nullable=True,
            ),
            Column(
                "type_memory",
                ENUM_TYPE,
                immutable=True,
                enum_type=(
                    "preference",
                    "fact",
                    "decision",
                    "topical_context",
                    "relational_context",
                ),
                enum_name="memory_type",
            ),
            Column("content", STRING_TYPE),
            Column("summary", STRING_TYPE, nullable=True),
            Column(
                "embedding",
                VECTOR_TYPE,
                vector_dim=_MEMORY_VECTOR_DIM,
                nullable=True,
            ),
            Column(
                "search_vector",
                TSVECTOR_TYPE,
                nullable=True,
                immutable=True,
            ),
            # v0.7.5: optional named anchor for direct lookup. Per-user
            # unique on the upstream DB side via alembic 088 (partial
            # unique index ``ix_memories_user_alias ON
            # memories(agent_id, user_id, alias) WHERE alias IS NOT
            # NULL``); relocated into 3tears in v0.8.0 (see indexes
            # below).
            Column("alias", STRING_TYPE, nullable=True),
            Column("date_created", DATETIMETZ_TYPE, immutable=True),
            Column("date_updated", DATETIMETZ_TYPE, nullable=True),
            # v024 (presence/aliveness): stored salience substrate.
            # NUMERIC(5,4) with a server default so existing INSERTs that
            # omit it apply 0.5 (metadata-only add, no table rewrite);
            # mutable so the reinforcement bump + decay pass can update it.
            Column(
                "salience",
                NUMERIC_TYPE,
                precision=5,
                scale=4,
                nullable=False,
                server_default="0.5",
            ),
            # decay anchor: age is measured from the last decay run, not
            # last access, so total decay over a period is cadence-safe.
            Column("last_decayed_at", DATETIMETZ_TYPE, nullable=True),
            # reinforcement telemetry: stamped on ambient retrieval.
            Column("last_accessed", DATETIMETZ_TYPE, nullable=True),
            # pin for core identity facts: excluded from decay AND bump.
            Column(
                "evergreen",
                BOOL_TYPE,
                nullable=False,
                server_default="false",
            ),
            # soft ref (no FK) to a consolidation gist; ambient retrieval
            # excludes non-null, direct recall still finds it.
            Column("superseded_by", UUID_TYPE, nullable=True),
        ],
        cas_column="date_updated",
        foreign_keys=(
            # message_id_source -> messages(message_id) ON DELETE
            # SET NULL: matches prod. Table-level because the
            # inline 2-tuple form cannot express on_delete. v0.8.0
            # locked decision: inline for NO ACTION, table-level for
            # everything else.
            SchemaForeignKey(
                "message_id_source",
                "messages",
                "message_id",
                on_delete="SET NULL",
            ),
        ),
        indexes=(
            SchemaIndex("ix_memories_user_date", "user_id", "date_created"),
            SchemaIndex(
                "ix_memories_user_alias",
                "agent_id",
                "user_id",
                "alias",
                unique=True,
                where="alias IS NOT NULL",
            ),
            # v0.8.1: parity-gate enrichments relocated from upstream
            # alembic. tenancy-scope btree composites used by the
            # cross-tenant access guards.
            SchemaIndex("idx_memories_agent_user", "agent_id", "user_id"),
            SchemaIndex(
                "idx_memories_agent_customer_user",
                "agent_id",
                "customer_id",
                "user_id",
            ),
            # v0.8.1: GIN over the trigger-maintained ``search_vector``
            # column drives FTS path on memories.
            SchemaIndex(
                "idx_memories_search_vector",
                "search_vector",
                using="gin",
            ),
            # v0.8.1: HNSW vector-similarity index with the
            # ``vector_cosine_ops`` opclass; ``m`` / ``ef_construction``
            # parameters mirror prod (upstream alembic). these are
            # strings to match the textual ``WITH (key = value)`` DDL
            # syntax pgvector emits.
            SchemaIndex(
                "ix_memories_embedding_hnsw",
                "embedding",
                using="hnsw",
                ops={"embedding": "vector_cosine_ops"},
                pg_with={"m": "16", "ef_construction": "64"},
            ),
        ),
        # v0.8.1: global uniqueness on ``memory_id`` (stronger than the
        # composite ``(agent_id, memory_id)`` PK so cross-agent leaks of
        # the same UUID surface as a write-time conflict). Modelled as a
        # UNIQUE CONSTRAINT (not a unique index) because prod
        # alembic 064 declares it via ``ALTER TABLE ... ADD CONSTRAINT
        # uq_memories_memory_id UNIQUE``; Alembic auto-gen distinguishes
        # the two via ``information_schema.table_constraints``.
        unique_constraints=(SchemaUniqueConstraint("uq_memories_memory_id", "memory_id"),),
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
        # cache-bypass: multi-row scan by (agent_id, customer_id,
        # user_id) is not primary-key addressable; L1 row cache
        # would not help. method on Collection preserves single
        # entry point.
        rows = await self.l3_pool.fetch(
            f"SELECT {_MEMORIES_SELECT_COLUMNS} FROM memories "
            "WHERE agent_id = $1 AND customer_id = $2 AND user_id = $3 "
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

        where_clause = " AND ".join(conditions)
        query = f"SELECT {_MEMORIES_SELECT_COLUMNS} FROM memories WHERE {where_clause} ORDER BY date_created DESC"

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

        is_owner_shortcut = caller_agent_id is not None and caller_agent_id == agent_id
        if caller_user_id is not None and not is_owner_shortcut:
            await ensure_memory_owner_assignment(
                user_id=caller_user_id,
                namespace=ns_entity,
                deps=self._authorizer,
            )
        return None

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
            "SELECT EXISTS(SELECT 1 FROM memories WHERE agent_id = $1 AND user_id = $2)",
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
        :class:`memory_add` tool dedup guard. returns memories whose
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
        # ``embedding IS NOT NULL`` filter for the same reason as
        # :meth:`hybrid_search`: ``NULL <=> vector`` is NULL, which
        # breaks the downstream ``float(similarity)`` cast.
        rows = await self.l3_pool.fetch(
            """
            SELECT memory_id, content, type_memory,
                   1 - (embedding OPERATOR(public.<=>) $1::text::public.vector) AS similarity
            FROM memories
            WHERE agent_id = $2 AND user_id = $3 AND embedding IS NOT NULL
            ORDER BY embedding OPERATOR(public.<=>) $1::text::public.vector
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

    async def find_by_alias(
        self,
        *,
        user_id: UUID,
        agent_id: UUID,
        alias: str,
    ) -> dict[str, Any] | None:
        """Look up a memory by its named alias (v0.7.5).

        Per-user unique on the upstream DB side (alembic 088 adds a
        partial unique index on ``(agent_id, user_id, alias) WHERE
        alias IS NOT NULL``), so the lookup returns at most one row.
        """
        if self.l3_pool is None or not alias:
            return None
        row = await self.l3_pool.fetchrow(
            """
            SELECT memory_id, agent_id, customer_id, user_id, conversation_id,
                   message_id_source, type_memory, content, alias,
                   date_created, date_updated
            FROM memories
            WHERE agent_id = $1 AND user_id = $2 AND alias = $3
            """,
            agent_id,
            user_id,
            alias,
        )
        return dict(row) if row else None

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
        date_after: datetime | None = None,
        date_before: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """parallel vector + FTS hybrid search across the memories table.

        absorbs the SQL that used to live in
        :meth:`MemoryRetriever._query_memories`. three-signal
        ranking (semantic / keyword / recency) + cosine-distance
        ordering; candidates are merged across the two parallel
        queries on ``memory_id``, recency-decayed, score-combined, and
        threshold-filtered.

        v0.7.5: optional ``date_after`` / ``date_before`` (inclusive)
        narrow the candidate pool to memories created within the
        range. The filter applies to BOTH the vector and the FTS
        sub-queries so paginated retrieval stays consistent.

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
        # v0.7.5 date filter: append ``AND date_created >= $X`` /
        # ``AND date_created <= $Y`` to both the vector and FTS sub-
        # queries. Bound parameters are appended after the existing
        # scope params + the limit param. Each filter uses its own
        # parameter slot so both can fire independently.
        vec_date_clause = ""
        vec_extra_params: list[Any] = []
        if date_after is not None:
            vec_date_clause += f" AND date_created >= ${param_offset + 2}"
            vec_extra_params.append(date_after)
        if date_before is not None:
            vec_date_clause += f" AND date_created <= ${param_offset + 2 + len(vec_extra_params)}"
            vec_extra_params.append(date_before)
        vec_where = f"WHERE {scope_conditions}"
        limit_param = f"${param_offset + 1}"

        # cache-bypass: vector-distance search is not primary-key-
        # addressable; see :meth:`find_similar_for_dedup` for the same
        # justification. method on Collection preserves single entry
        # point for rbac + audit.
        # Filter out rows with NULL ``embedding`` before computing
        # cosine distance: ``NULL <=> vector`` is NULL, and the
        # downstream ``float(row["similarity"])`` chokes on it. Rows
        # with NULL embeddings exist when the embedding provider was
        # absent / failed at write time -- they're recoverable on a
        # re-embed pass, but until then they must not enter the
        # similarity-ranked candidate set.
        vec_coro = self.l3_pool.fetch(
            f"""
            SELECT memory_id, content, summary, type_memory, date_created,
                   embedding::text AS embedding,
                   1 - (embedding OPERATOR(public.<=>) $1::text::public.vector) AS similarity
            FROM memories
            {vec_where}
              AND embedding IS NOT NULL{vec_date_clause}
            ORDER BY embedding OPERATOR(public.<=>) $1::text::public.vector
            LIMIT {limit_param}
            """,
            embedding_str,
            *scope_params,
            candidate_limit,
            *vec_extra_params,
        )

        fts_text = _build_fts_text(user_text, fts_min_len, fts_max_len)
        if fts_text:
            fts_scope_conditions, fts_scope_params, fts_param_offset = _build_user_scope_clause(
                user_id,
                agent_id=agent_id,
                customer_id=customer_id,
                start_param=2,
            )
            # v0.7.5 date filter on FTS sub-query.
            fts_date_clause = ""
            fts_extra_params: list[Any] = []
            if date_after is not None:
                fts_date_clause += f" AND date_created >= ${fts_param_offset + 2}"
                fts_extra_params.append(date_after)
            if date_before is not None:
                fts_date_clause += f" AND date_created <= ${fts_param_offset + 2 + len(fts_extra_params)}"
                fts_extra_params.append(date_before)
            fts_where = f"WHERE {fts_scope_conditions}"
            fts_limit_param = f"${fts_param_offset + 1}"
            # cache-bypass: FTS rank query is not primary-key-
            # addressable. See :meth:`hybrid_search` docstring.
            fts_coro = self.l3_pool.fetch(
                f"""
                SELECT memory_id, content, summary, type_memory, date_created,
                       embedding::text AS embedding,
                       ts_rank_cd(search_vector, websearch_to_tsquery('english', $1)) AS fts_rank
                FROM memories
                {fts_where}
                  AND embedding IS NOT NULL
                  AND search_vector @@ websearch_to_tsquery('english', $1){fts_date_clause}
                ORDER BY fts_rank DESC
                LIMIT {fts_limit_param}
                """,
                fts_text,
                *fts_scope_params,
                candidate_limit,
                *fts_extra_params,
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
        date_after: datetime | None = None,
        date_before: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """semantic (vector-only) search for the add/search memory tools.

        absorbs the ``_search_memories`` path from :mod:`tools.py`
        (the LangChain ``memory_search`` tool's vector leg). returns
        rows whose cosine similarity exceeds ``similarity_threshold``,
        capped at ``max_results``. ``type_filter`` is validated by the
        caller; this method trusts the value and applies it as an
        equality predicate. ``agent_id`` is the partition column on
        memories and is required.

        v0.7.5: ``date_after`` / ``date_before`` (inclusive) narrow the
        candidate pool by ``date_created``; either may be supplied
        independently.

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
        :param date_after: inclusive lower bound on ``date_created``
        :ptype date_after: datetime | None
        :param date_before: inclusive upper bound on ``date_created``
        :ptype date_before: datetime | None
        :return: list of row dicts with ``similarity`` field
        :rtype: list[dict[str, Any]]
        """
        if self.l3_pool is None:
            return []
        embedding_str = json.dumps(embedding)
        params: list[Any] = [embedding_str, agent_id, user_id]
        conditions = ["agent_id = $2", "user_id = $3"]
        param_idx = 4
        if type_filter:
            conditions.append(f"type_memory = ${param_idx}")
            params.append(type_filter)
            param_idx += 1
        if date_after is not None:
            conditions.append(f"date_created >= ${param_idx}")
            params.append(date_after)
            param_idx += 1
        if date_before is not None:
            conditions.append(f"date_created <= ${param_idx}")
            params.append(date_before)
            param_idx += 1
        conditions.append("embedding IS NOT NULL")
        where_clause = " AND ".join(conditions)
        # ``embedding IS NOT NULL`` guard: ``NULL <=> vector`` is NULL,
        # which trips ``float(similarity)`` downstream.
        query_sql = f"""
            SELECT memory_id, type_memory, content, date_created,
                   1 - (embedding OPERATOR(public.<=>) $1::text::public.vector) AS similarity
            FROM memories
            WHERE {where_clause}
            ORDER BY embedding OPERATOR(public.<=>) $1::text::public.vector
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
        date_after: datetime | None = None,
        date_before: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """FTS keyword search for the add/search memory tools.

        complements :meth:`search_by_semantic` — run both in parallel
        and merge on ``memory_id``. ``agent_id`` is the partition
        column on memories and is required.

        v0.7.5: ``date_after`` / ``date_before`` (inclusive) narrow by
        ``date_created`` — mirrors the equivalent filter on
        :meth:`search_by_semantic` so the two legs return a consistent
        window when run in parallel.

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
        :param date_after: inclusive lower bound on ``date_created``
        :ptype date_after: datetime | None
        :param date_before: inclusive upper bound on ``date_created``
        :ptype date_before: datetime | None
        :return: list of row dicts
        :rtype: list[dict[str, Any]]
        """
        if self.l3_pool is None:
            return []
        conditions = [
            "agent_id = $2",
            "user_id = $3",
            "search_vector @@ websearch_to_tsquery('english', $1)",
        ]
        params: list[Any] = [fts_text, agent_id, user_id]
        idx = 4
        if type_filter:
            conditions.append(f"type_memory = ${idx}")
            params.append(type_filter)
            idx += 1
        if date_after is not None:
            conditions.append(f"date_created >= ${idx}")
            params.append(date_after)
            idx += 1
        if date_before is not None:
            conditions.append(f"date_created <= ${idx}")
            params.append(date_before)
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
        """fetch just the ``content`` field for the memory_recall tool.

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
        # row has already been warmed, but the ``user_id`` guard here
        # is a SECURITY control (cross-user leak prevention) that the
        # L1 row cache cannot enforce. the authoritative read must
        # stay at the database.
        row = await self.l3_pool.fetchrow(
            "SELECT content FROM memories WHERE agent_id = $1 AND memory_id = $2 AND user_id = $3",
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
                f"SELECT {_MEMORIES_SELECT_COLUMNS} FROM memories WHERE agent_id = ANY($1::uuid[]) AND user_id = $2 ORDER BY date_created DESC",
                list(agent_ids),
                user_id,
            )
        else:
            rows = await self.l3_pool.fetch(
                f"SELECT {_MEMORIES_SELECT_COLUMNS} FROM memories "
                "WHERE agent_id = ANY($1::uuid[]) AND customer_id = $2 "
                "AND user_id = $3 "
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
    # v0.8.0 enrichment: full prod shape (matches v0.7.5 ``media_table``
    # factory + prod ``information_schema.columns``). Columns added in
    # v0.8.0: ``s3_key``, ``mime_type``, ``size_bytes``, ``source``,
    # ``generation_prompt``, ``thumbnail_s3_key``, ``cloud_connection_id``,
    # ``cloud_file_id``, ``cloud_file_url``, ``extraction_status``.
    # The ``date_updated`` column was REMOVED to match prod -- the
    # v0.7.5 factory only declares ``date_created`` and prod has no
    # ``date_updated`` column. The composite FK ``(agent_id, memory_id)
    # → memories(agent_id, memory_id) ON DELETE CASCADE`` is the
    # unified-model parent FK (3tears migration v017) and is required
    # by the parity gate even though prod's upstream Alembic side only
    # carries the single-column ``memory_id → memories.memory_id``
    # variant; declared as a composite to encode the partition-aware
    # relationship in 3tears.
    schema = TableSchema(
        name="media",
        primary_key=("agent_id", "media_id"),
        columns=[
            Column("agent_id", UUID_TYPE, partition=True),
            Column("media_id", UUID_TYPE),
            Column("customer_id", UUID_TYPE),
            Column(
                "user_id",
                UUID_TYPE,
                foreign_key=("users", "user_id"),
            ),
            Column("s3_key", STRING_TYPE, nullable=True),
            Column("mime_type", STRING_TYPE),
            Column("size_bytes", INT_TYPE),
            Column("source", STRING_TYPE),
            Column(
                "metadata_json",
                JSONB_TYPE,
                server_default="'{}'::jsonb",
            ),
            Column("generation_prompt", STRING_TYPE, nullable=True),
            Column(
                "media_category",
                STRING_TYPE,
                server_default="'image'::text",
            ),
            Column(
                "extraction_status",
                STRING_TYPE,
                server_default="'none'::text",
            ),
            Column("thumbnail_s3_key", STRING_TYPE, nullable=True),
            Column("cloud_connection_id", UUID_TYPE, nullable=True),
            Column("cloud_file_id", STRING_TYPE, nullable=True),
            Column("cloud_file_url", STRING_TYPE, nullable=True),
            Column("memory_id", UUID_TYPE, immutable=True),
            Column("date_created", DATETIMETZ_TYPE, immutable=True),
            # date_updated is trigger-maintained server-side: the v021
            # migration installs a BEFORE UPDATE trigger that resets
            # the column to now() on every row update. Declared
            # ``immutable=True`` so the Collection's UPDATE generator
            # excludes it from SET clauses; ``server_default="now()"``
            # so INSERTs that omit the value get the right shape from
            # Postgres.
            Column(
                "date_updated",
                DATETIMETZ_TYPE,
                immutable=True,
                server_default="now()",
            ),
        ],
        foreign_keys=(
            SchemaForeignKey(
                "cloud_connection_id",
                "cloud_connections",
                "cloud_connection_id",
                on_delete="SET NULL",
            ),
            SchemaForeignKey(
                "memory_id",
                "memories",
                "memory_id",
                on_delete="CASCADE",
            ),
        ),
        indexes=(
            SchemaIndex("ix_media_user_date", "user_id", "date_created"),
            SchemaIndex("ix_media_mime_type", "mime_type"),
            SchemaIndex("ix_media_memory_id", "memory_id"),
            SchemaIndex(
                "uq_media_cloud_connection_file",
                "cloud_connection_id",
                "cloud_file_id",
                unique=True,
            ),
            # v0.8.1: parity-gate enrichments relocated from upstream
            # alembic. tenancy-scope btree composite.
            SchemaIndex("idx_media_agent_user", "agent_id", "user_id"),
            # v0.8.1: partial btree over rows still awaiting extraction;
            # selective enough to keep the queue-scan cheap.
            SchemaIndex(
                "ix_media_extraction_pending",
                "extraction_status",
                where="extraction_status = 'pending'",
            ),
        ),
        # v0.8.1: global uniqueness on ``media_id`` (stronger than the
        # composite ``(agent_id, media_id)`` PK). Modelled as a UNIQUE
        # CONSTRAINT (not a unique index) -- prod creates it via
        # ``ALTER TABLE ... ADD CONSTRAINT uq_media_media_id UNIQUE`` in
        # upstream alembic 064.
        unique_constraints=(SchemaUniqueConstraint("uq_media_media_id", "media_id"),),
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
    the generator emits ``$N::text::public.vector`` on the INSERT path and decodes
    the textual response back to ``list[float]`` on read.
    """

    primary_key_column: str | tuple[str, ...] = ("agent_id", "content_id")
    # v0.8.0 enrichment: full prod shape (matches v0.7.5
    # ``media_content_table`` factory + prod). Added: ``model_id``,
    # ``provider_id``, ``model_name``, ``provider_name``,
    # ``token_count_prompt``, ``token_count_completion``,
    # ``cost`` (NUMERIC(12, 8)), ``metadata_json``, ``search_vector``.
    # Composite FK ``(agent_id, media_id) → media(agent_id, media_id)
    # ON DELETE CASCADE`` is the v017 partition-aware parent FK; the
    # factory documented but did not emit it in v0.7.5 (the comment at
    # the factory body says SQLAlchemy expresses composite FKs at
    # table level -- v0.8.0 expresses it explicitly).
    schema = TableSchema(
        name="media_content",
        primary_key=("agent_id", "content_id"),
        columns=[
            Column("agent_id", UUID_TYPE, partition=True),
            Column("content_id", UUID_TYPE),
            Column("customer_id", UUID_TYPE, immutable=True),
            Column("media_id", UUID_TYPE, immutable=True),
            Column(
                "user_id",
                UUID_TYPE,
                immutable=True,
                foreign_key=("users", "user_id"),
            ),
            Column("content_type", STRING_TYPE),
            Column("content", STRING_TYPE),
            Column("summary", STRING_TYPE, nullable=True),
            Column(
                "embedding",
                VECTOR_TYPE,
                nullable=True,
                vector_dim=_MEMORY_VECTOR_DIM,
            ),
            Column(
                "search_vector",
                TSVECTOR_TYPE,
                nullable=True,
                immutable=True,
            ),
            Column(
                "model_id",
                UUID_TYPE,
                nullable=True,
                foreign_key=("models", "model_id"),
            ),
            Column(
                "provider_id",
                UUID_TYPE,
                nullable=True,
                foreign_key=("providers", "provider_id"),
            ),
            Column("model_name", STRING_TYPE, nullable=True),
            Column("provider_name", STRING_TYPE, nullable=True),
            Column("token_count_prompt", INT_TYPE, nullable=True),
            Column("token_count_completion", INT_TYPE, nullable=True),
            Column(
                "cost",
                NUMERIC_TYPE,
                precision=12,
                scale=8,
                nullable=True,
            ),
            Column("metadata_json", JSONB_TYPE, nullable=True),
            Column("date_created", DATETIMETZ_TYPE, immutable=True),
        ],
        foreign_keys=(
            SchemaForeignKey(
                ("agent_id", "media_id"),
                "media",
                ("agent_id", "media_id"),
                on_delete="CASCADE",
            ),
        ),
        indexes=(
            SchemaIndex(
                "ix_media_content_media_type",
                "media_id",
                "content_type",
            ),
            SchemaIndex("ix_media_content_user", "user_id"),
            # v0.8.1: parity-gate enrichments relocated from upstream
            # alembic. tenancy-scope btree composite.
            SchemaIndex(
                "idx_media_content_agent_user",
                "agent_id",
                "user_id",
            ),
            # v0.8.1: GIN over the trigger-maintained ``search_vector``
            # column drives the FTS half of the hybrid-search path.
            SchemaIndex(
                "idx_media_content_search_vector",
                "search_vector",
                using="gin",
            ),
            # v0.8.1: HNSW vector-similarity index with the
            # ``vector_cosine_ops`` opclass. Prod does NOT carry a
            # ``WITH`` clause here (the index was built before the
            # upstream migration started parametrising hnsw), so this
            # declaration has no ``pg_with=``.
            SchemaIndex(
                "ix_media_content_embedding",
                "embedding",
                using="hnsw",
                ops={"embedding": "vector_cosine_ops"},
            ),
        ),
        # v0.8.1: global uniqueness on ``content_id``. Modelled as a
        # UNIQUE CONSTRAINT (not a unique index) -- prod creates it via
        # ``ALTER TABLE ... ADD CONSTRAINT uq_media_content_content_id
        # UNIQUE`` in upstream alembic 064.
        unique_constraints=(SchemaUniqueConstraint("uq_media_content_content_id", "content_id"),),
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
                   mc.media_id, mc.date_created, mc.embedding::text AS embedding,
                   med.media_category, med.metadata_json,
                   1 - (mc.embedding OPERATOR(public.<=>) $1::text::public.vector) AS similarity
            FROM media_content mc
            JOIN media med
              ON mc.agent_id = med.agent_id
             AND mc.media_id = med.media_id
            WHERE {scope_conditions} AND mc.embedding IS NOT NULL
            ORDER BY mc.embedding OPERATOR(public.<=>) $1::text::public.vector
            LIMIT {limit_param}
            """,
            embedding_str,
            *scope_params,
            candidate_limit,
        )

        fts_text = _build_fts_text(user_text, fts_min_len, fts_max_len)
        if fts_text:
            fts_scope_conditions, fts_scope_params, fts_param_offset = _build_user_scope_clause(
                user_id,
                agent_id=agent_id,
                customer_id=customer_id,
                start_param=2,
                table_prefix="mc",
            )
            fts_limit_param = f"${fts_param_offset + 1}"
            # cache-bypass: FTS rank query joining media_content ->
            # media. see vec_coro justification above.
            fts_coro = self.l3_pool.fetch(
                f"""
                SELECT mc.content_id, mc.content, mc.summary, mc.content_type,
                       mc.media_id, mc.date_created, mc.embedding::text AS embedding,
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
                (meta or {}).get("document_title") or (meta or {}).get("original_filename") or (meta or {}).get("title")
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
                   1 - (mc.embedding OPERATOR(public.<=>) $1::text::public.vector) AS similarity
            FROM media_content mc
            JOIN media med
              ON mc.agent_id = med.agent_id
             AND mc.media_id = med.media_id
            WHERE mc.agent_id = $2
              AND mc.user_id = $3
              AND mc.embedding IS NOT NULL
            ORDER BY mc.embedding OPERATOR(public.<=>) $1::text::public.vector
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
        """fetch ``content`` for the memory_recall tool (media leg).

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
            "SELECT content FROM media_content WHERE agent_id = $1 AND content_id = $2 AND user_id = $3",
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

    Chunks are the verbatim source layer under the unified memory
    model. Every chunk parents to exactly one :class:`MemoryEntity`
    via ``memory_id`` (NOT NULL FK with CASCADE after v017). Document
    chunks carry ``heading_context`` / ``page_number`` from the source
    artifact; transcript chunks carry ``message_id_start`` /
    ``message_id_end`` backlinking the summarized message range.

    Hybrid-search joins LEFT JOIN to :class:`MediaEntity` on
    ``(agent_id, mc.memory_id = med.memory_id)`` so document chunks
    surface their source media's ``metadata_json`` (title, filename)
    in retrieval; transcript chunks naturally LEFT JOIN to NULL
    because their parent memory has no media child.

    CRUD is generated from :attr:`schema`; the embedding column uses
    ``VECTOR_TYPE`` for pgvector cast on write + ``list[float]``
    decode on read.
    """

    primary_key_column: str | tuple[str, ...] = ("agent_id", "chunk_id")
    # v0.8.0 enrichment: full prod shape. Added: ``chunk_index``
    # (required NOT NULL int4 in prod, drives ordering),
    # ``token_count`` (required NOT NULL int4), ``search_vector``
    # (trigger-maintained tsvector). Composite FK ``(agent_id,
    # memory_id) → memories(agent_id, memory_id) ON DELETE CASCADE``
    # is the v017 unified-model parent FK.
    schema = TableSchema(
        name="memory_chunks",
        primary_key=("agent_id", "chunk_id"),
        columns=[
            Column("agent_id", UUID_TYPE, partition=True),
            Column("chunk_id", UUID_TYPE),
            Column("customer_id", UUID_TYPE, immutable=True),
            Column("memory_id", UUID_TYPE, immutable=True),
            Column(
                "user_id",
                UUID_TYPE,
                immutable=True,
                foreign_key=("users", "user_id"),
            ),
            Column("chunk_index", INT_TYPE),
            Column("content", STRING_TYPE),
            Column("summary", STRING_TYPE, nullable=True),
            Column("heading_context", STRING_TYPE, nullable=True),
            Column("page_number", INT_TYPE, nullable=True),
            Column("token_count", INT_TYPE),
            Column(
                "embedding",
                VECTOR_TYPE,
                nullable=True,
                vector_dim=_MEMORY_VECTOR_DIM,
            ),
            Column(
                "search_vector",
                TSVECTOR_TYPE,
                nullable=True,
                immutable=True,
            ),
            Column("message_id_start", UUID_TYPE, nullable=True, immutable=True),
            Column("message_id_end", UUID_TYPE, nullable=True, immutable=True),
            Column("date_created", DATETIMETZ_TYPE, immutable=True),
        ],
        foreign_keys=(
            SchemaForeignKey(
                ("agent_id", "memory_id"),
                "memories",
                ("agent_id", "memory_id"),
                on_delete="CASCADE",
            ),
        ),
        indexes=(
            SchemaIndex(
                "ix_memory_chunks_memory",
                "memory_id",
                "chunk_index",
            ),
            SchemaIndex("ix_memory_chunks_user", "user_id"),
            # v0.8.1: parity-gate enrichments relocated from upstream
            # alembic. partial composite covering the agent-scoped
            # conversation-replay lookup that follows
            # ``message_id_start`` chains.
            SchemaIndex(
                "idx_chunks_message_id_start",
                "agent_id",
                "message_id_start",
                where="message_id_start IS NOT NULL",
            ),
            # v0.8.1: tenancy-scope btree composite.
            SchemaIndex(
                "idx_memory_chunks_agent_user",
                "agent_id",
                "user_id",
            ),
            # v0.8.1: GIN over the trigger-maintained ``search_vector``
            # column drives the FTS half of chunks hybrid search.
            SchemaIndex(
                "idx_memory_chunks_search_vector",
                "search_vector",
                using="gin",
            ),
            # v0.8.1: HNSW vector-similarity index with
            # ``vector_cosine_ops``. Prod does NOT carry a ``WITH``
            # clause here (mirror of ``ix_media_content_embedding``).
            SchemaIndex(
                "ix_memory_chunks_embedding",
                "embedding",
                using="hnsw",
                ops={"embedding": "vector_cosine_ops"},
            ),
        ),
        # v0.8.1: global uniqueness on ``chunk_id``. Modelled as a
        # UNIQUE CONSTRAINT (not a unique index) -- prod creates it via
        # ``ALTER TABLE ... ADD CONSTRAINT uq_memory_chunks_chunk_id
        # UNIQUE`` in upstream alembic 064.
        unique_constraints=(SchemaUniqueConstraint("uq_memory_chunks_chunk_id", "chunk_id"),),
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
        chunk_id_after: UUID | None = None,
        chunk_id_before: UUID | None = None,
    ) -> list[dict[str, Any]]:
        """parallel vector + FTS search against memory_chunks + media.

        absorbs :meth:`MemoryRetriever._query_chunks`. two-signal
        ranking (semantic + keyword); no recency decay because chunks
        come from documents whose freshness signal is the upload
        event, not the chunk's own lifecycle. ``agent_id`` is the
        partition column on memory_chunks + media; ``customer_id`` is
        a required sub-scope.

        Cursor paging (transcript-chunks-task-A): pass
        ``chunk_id_after`` to restrict the candidate pool to chunks
        whose ``chunk_id`` is strictly greater than the cursor, or
        ``chunk_id_before`` for the symmetric backward direction.
        The cursor predicate is applied AFTER the auth filter — auth
        scoping is non-negotiable, cursor is just for ordering within
        the auth-scoped result set. Passing both raises ValueError.

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
        :param chunk_id_after: optional cursor — restrict candidate
            pool to chunks with chunk_id strictly greater
        :ptype chunk_id_after: UUID | None
        :param chunk_id_before: optional cursor — restrict to chunks
            with chunk_id strictly less. Mutually exclusive with
            ``chunk_id_after``
        :ptype chunk_id_before: UUID | None
        :return: ranked candidate list
        :rtype: list[dict[str, Any]]
        :raises ValueError: if both cursors are provided
        """
        if chunk_id_after is not None and chunk_id_before is not None:
            raise ValueError("hybrid_search: pass at most one of chunk_id_after / chunk_id_before")
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
        cursor_clause = ""
        cursor_value: UUID | None = None
        if chunk_id_after is not None:
            cursor_clause = f"AND mc.chunk_id > ${param_offset + 1}"
            cursor_value = chunk_id_after
        elif chunk_id_before is not None:
            cursor_clause = f"AND mc.chunk_id < ${param_offset + 1}"
            cursor_value = chunk_id_before
        cursor_params: list[Any] = [cursor_value] if cursor_value is not None else []
        limit_param = f"${param_offset + 1 + len(cursor_params)}"

        # cache-bypass: vector-distance search joining memory_chunks ->
        # media (through the new memory parent FK). not primary-key-
        # addressable; L1 row cache cannot help. method on Collection
        # preserves single entry point. The LEFT JOIN matches media
        # rows whose ``memory_id`` equals the chunk's ``memory_id`` —
        # i.e. the chunk's parent memory IS the media's parent memory.
        # Transcript chunks naturally LEFT JOIN to NULL because their
        # parent memory has no media child.
        # ``mc.embedding IS NOT NULL`` guard: ``NULL <=> vector`` is
        # NULL, which trips ``float(similarity)`` downstream.
        vec_coro = self.l3_pool.fetch(
            f"""
            SELECT mc.chunk_id, mc.content, mc.summary, mc.heading_context,
                   mc.page_number, mc.memory_id,
                   mc.message_id_start, mc.message_id_end,
                   mc.embedding::text AS embedding, med.metadata_json, med.media_id,
                   1 - (mc.embedding OPERATOR(public.<=>) $1::text::public.vector) AS similarity
            FROM memory_chunks mc
            LEFT JOIN media med
              ON mc.agent_id = med.agent_id
             AND mc.memory_id = med.memory_id
            WHERE {scope_conditions}
              AND mc.embedding IS NOT NULL
              {cursor_clause}
            ORDER BY mc.embedding OPERATOR(public.<=>) $1::text::public.vector
            LIMIT {limit_param}
            """,
            embedding_str,
            *scope_params,
            *cursor_params,
            candidate_k,
        )

        fts_text = _build_fts_text(user_text, fts_min_len, fts_max_len)
        if fts_text:
            fts_scope_conditions, fts_scope_params, fts_param_offset = _build_user_scope_clause(
                user_id,
                agent_id=agent_id,
                customer_id=customer_id,
                start_param=2,
                table_prefix="mc",
            )
            fts_cursor_clause = ""
            fts_cursor_params: list[Any] = []
            if cursor_value is not None:
                op = ">" if chunk_id_after is not None else "<"
                fts_cursor_clause = f"AND mc.chunk_id {op} ${fts_param_offset + 1}"
                fts_cursor_params.append(cursor_value)
            fts_limit_param = f"${fts_param_offset + 1 + len(fts_cursor_params)}"
            # cache-bypass: FTS rank query joining memory_chunks ->
            # media through the new memory_id parent FK.
            fts_coro = self.l3_pool.fetch(
                f"""
                SELECT mc.chunk_id, mc.content, mc.summary, mc.heading_context,
                       mc.page_number, mc.memory_id,
                       mc.message_id_start, mc.message_id_end,
                       mc.embedding::text AS embedding, med.metadata_json, med.media_id,
                       ts_rank_cd(mc.search_vector, websearch_to_tsquery('english', $1)) AS fts_rank
                FROM memory_chunks mc
                LEFT JOIN media med
                  ON mc.agent_id = med.agent_id
                 AND mc.memory_id = med.memory_id
                WHERE {fts_scope_conditions}
                  AND mc.embedding IS NOT NULL
                  AND mc.search_vector @@ websearch_to_tsquery('english', $1)
                  {fts_cursor_clause}
                ORDER BY fts_rank DESC
                LIMIT {fts_limit_param}
                """,
                fts_text,
                *fts_scope_params,
                *fts_cursor_params,
                candidate_k,
            )
            vec_rows, fts_rows = await asyncio.gather(vec_coro, fts_coro)
        else:
            vec_rows = await vec_coro
            fts_rows = []

        return _merge_chunk_search_rows(
            vec_rows=vec_rows,
            fts_rows=fts_rows,
            chunk_signal_weights=chunk_signal_weights,
            similarity_threshold=similarity_threshold,
        )

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
        # -> media via the memory_id parent FK. see :meth:`hybrid_search`
        # for the JOIN-shape rationale (LEFT JOIN matches media whose
        # parent memory == the chunk's parent memory). transcript chunks
        # naturally LEFT JOIN to NULL.
        rows = await self.l3_pool.fetch(
            """
            SELECT mc.chunk_id, mc.content, mc.heading_context, mc.page_number,
                   mc.memory_id, mc.message_id_start, mc.message_id_end,
                   med.metadata_json, med.media_id
            FROM memory_chunks mc
            LEFT JOIN media med
              ON mc.agent_id = med.agent_id
             AND mc.memory_id = med.memory_id
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
        # -> media via the memory_id parent FK. see :meth:`hybrid_search`.
        # ``mc.embedding IS NOT NULL`` guard: same NULL-cosine
        # rationale as :meth:`hybrid_search`.
        rows = await self.l3_pool.fetch(
            """
            SELECT mc.chunk_id, mc.content, mc.heading_context, mc.page_number,
                   mc.memory_id, mc.message_id_start, mc.message_id_end,
                   med.metadata_json,
                   1 - (mc.embedding OPERATOR(public.<=>) $1::text::public.vector) AS similarity
            FROM memory_chunks mc
            LEFT JOIN media med
              ON mc.agent_id = med.agent_id
             AND mc.memory_id = med.memory_id
            WHERE mc.agent_id = $2
              AND mc.user_id = $3
              AND mc.embedding IS NOT NULL
            ORDER BY mc.embedding OPERATOR(public.<=>) $1::text::public.vector
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
                    "memory_id": str(row["memory_id"]),
                    "content": row["content"],
                    "heading_context": row["heading_context"],
                    "page_number": row["page_number"],
                    "message_id_start": (str(row["message_id_start"]) if row["message_id_start"] else None),
                    "message_id_end": (str(row["message_id_end"]) if row["message_id_end"] else None),
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
    ) -> tuple[str, UUID] | None:
        """fetch ``content`` + parent ``memory_id`` for the recall tools.

        returns both columns in one SELECT so the ``chunk_recall``
        tool can render ``[parent memory: <memory_id>]`` footer
        without a second by-ID fetch (the prior shape returned just
        ``content`` and the tool followed up with its own
        ``SELECT memory_id`` -- two round-trips for one row).

        callers that need only the content (the ``memory_recall``
        tool's chunks leg) read ``result[0]`` and discard
        ``result[1]``; the extra column is essentially free since the
        same row is being fetched either way.

        :param chunk_id: chunk primary-key value
        :ptype chunk_id: UUID
        :param user_id: owning user UUID (row filter)
        :ptype user_id: UUID
        :param agent_id: partition column on memory_chunks; required
        :ptype agent_id: UUID
        :return: ``(content, memory_id)`` tuple or ``None`` if not found
        :rtype: tuple[str, UUID] | None
        """
        if self.l3_pool is None:
            return None
        # cache-bypass: by-ID fetch scoped by (agent_id, user_id) —
        # both are security predicates the L1 cache cannot enforce.
        row = await self.l3_pool.fetchrow(
            "SELECT content, memory_id FROM memory_chunks WHERE agent_id = $1 AND chunk_id = $2 AND user_id = $3",
            agent_id,
            chunk_id,
            user_id,
        )
        if row is None:
            return None
        return (row["content"], row["memory_id"])

    async def find_by_memory_id(
        self,
        memory_id: UUID,
        *,
        user_id: UUID,
        agent_id: UUID,
        customer_id: UUID,
        limit: int = 50,
        chunk_id_after: UUID | None = None,
        chunk_id_before: UUID | None = None,
    ) -> list[dict[str, Any]]:
        """fetch chunks parented to ``memory_id``, ordered by chunk_id ASC.

        UUIDv7 ``chunk_id`` is byte-ordered with the timestamp prefix
        so ``ORDER BY chunk_id ASC`` matches chronological creation
        order. Within a single ``conversation_summarize`` call the
        emitted chunks are also message-range-ordered because the
        agent processes the range start-to-end.

        Cursor paging: pass ``chunk_id_after`` to advance forward
        (chunks with ``chunk_id > cursor``) or ``chunk_id_before`` to
        page backward (chunks with ``chunk_id < cursor``, returned
        DESC then reversed so client order stays ASC). Passing both
        raises :class:`ValueError`.

        Auth scoping: every query carries the full ``(user_id,
        agent_id, customer_id)`` triple. The triple matches the
        partition + sub-scope + row-owner triple every other memory
        SQL site enforces — skipping any one is a cross-tenant data
        leak.

        :param memory_id: parent memory UUID
        :ptype memory_id: UUID
        :param user_id: owning user UUID
        :ptype user_id: UUID
        :param agent_id: partition column on memory_chunks; required
        :ptype agent_id: UUID
        :param customer_id: required sub-scope
        :ptype customer_id: UUID
        :param limit: max chunks to return
        :ptype limit: int
        :param chunk_id_after: cursor — return chunks with chunk_id
            strictly greater
        :ptype chunk_id_after: UUID | None
        :param chunk_id_before: cursor — return chunks with chunk_id
            strictly less. Mutually exclusive with ``chunk_id_after``
        :ptype chunk_id_before: UUID | None
        :return: chunk row dicts in chunk_id ASC order
        :rtype: list[dict[str, Any]]
        :raises ValueError: if both cursors are provided
        """
        if chunk_id_after is not None and chunk_id_before is not None:
            raise ValueError("find_by_memory_id: pass at most one of chunk_id_after / chunk_id_before")
        if self.l3_pool is None:
            return []

        # cache-bypass: multi-row scan by memory_id within an agent
        # partition. composite index idx_chunks_memory_id_chunk_id
        # backs the ORDER BY chunk_id within memory_id. auth triple
        # enforced on every row.
        params: list[Any] = [agent_id, memory_id, user_id, customer_id]
        cursor_clause = ""
        if chunk_id_after is not None:
            cursor_clause = "AND mc.chunk_id > $5"
            params.append(chunk_id_after)
            order_dir = "ASC"
        elif chunk_id_before is not None:
            cursor_clause = "AND mc.chunk_id < $5"
            params.append(chunk_id_before)
            order_dir = "DESC"
        else:
            order_dir = "ASC"
        limit_param = f"${len(params) + 1}"
        params.append(limit)

        rows = await self.l3_pool.fetch(
            f"""
            SELECT mc.chunk_id, mc.memory_id, mc.content, mc.summary,
                   mc.heading_context, mc.page_number,
                   mc.message_id_start, mc.message_id_end,
                   mc.date_created
            FROM memory_chunks mc
            WHERE mc.agent_id = $1
              AND mc.memory_id = $2
              AND mc.user_id = $3
              AND mc.customer_id = $4
              {cursor_clause}
            ORDER BY mc.chunk_id {order_dir}
            LIMIT {limit_param}
            """,
            *params,
        )
        result = [dict(row) for row in rows]
        # client-facing order is always ASC; flip the DESC backward
        # page so callers get a stable left-to-right list.
        if order_dir == "DESC":
            result.reverse()
        return result

    async def find_by_conversation_id(
        self,
        conversation_id: UUID,
        *,
        user_id: UUID,
        agent_id: UUID,
        customer_id: UUID,
        limit: int = 50,
        chunk_id_after: UUID | None = None,
        chunk_id_before: UUID | None = None,
    ) -> list[dict[str, Any]]:
        """fetch transcript chunks for ``conversation_id``, ordered by message_id_start ASC.

        Joins chunks to memories on (agent_id, memory_id) and filters
        to memories whose ``conversation_id`` matches. Document
        chunks have ``message_id_start`` NULL and are excluded from
        this query (the cross-memory scope is narrative-order, which
        document chunks don't participate in). Auth triple enforced
        on the joined memories row.

        Cursor paging semantics match :meth:`find_by_memory_id`.

        :param conversation_id: conversation scope key
        :ptype conversation_id: UUID
        :param user_id: owning user UUID
        :ptype user_id: UUID
        :param agent_id: partition column; required
        :ptype agent_id: UUID
        :param customer_id: required sub-scope
        :ptype customer_id: UUID
        :param limit: max chunks to return
        :ptype limit: int
        :param chunk_id_after: cursor — chunks with chunk_id > cursor
        :ptype chunk_id_after: UUID | None
        :param chunk_id_before: cursor — chunks with chunk_id < cursor
            (returned DESC then reversed)
        :ptype chunk_id_before: UUID | None
        :return: chunk row dicts in narrative order
        :rtype: list[dict[str, Any]]
        :raises ValueError: if both cursors are provided
        """
        if chunk_id_after is not None and chunk_id_before is not None:
            raise ValueError("find_by_conversation_id: pass at most one of chunk_id_after / chunk_id_before")
        if self.l3_pool is None:
            return []

        # cache-bypass: JOIN scan across memory_chunks + memories
        # filtered by memories.conversation_id. backed by the
        # idx_chunks_message_id_start partial index for the ORDER BY.
        # auth triple enforced on the joined memories row.
        params: list[Any] = [agent_id, conversation_id, user_id, customer_id]
        cursor_clause = ""
        if chunk_id_after is not None:
            cursor_clause = "AND mc.chunk_id > $5"
            params.append(chunk_id_after)
            order_dir = "ASC"
        elif chunk_id_before is not None:
            cursor_clause = "AND mc.chunk_id < $5"
            params.append(chunk_id_before)
            order_dir = "DESC"
        else:
            order_dir = "ASC"
        limit_param = f"${len(params) + 1}"
        params.append(limit)

        rows = await self.l3_pool.fetch(
            f"""
            SELECT mc.chunk_id, mc.memory_id, mc.content, mc.summary,
                   mc.heading_context, mc.page_number,
                   mc.message_id_start, mc.message_id_end,
                   mc.date_created
            FROM memory_chunks mc
            JOIN memories m
              ON m.agent_id = mc.agent_id
             AND m.memory_id = mc.memory_id
            WHERE mc.agent_id = $1
              AND m.conversation_id = $2
              AND mc.user_id = $3
              AND mc.customer_id = $4
              AND mc.message_id_start IS NOT NULL
              {cursor_clause}
            ORDER BY mc.message_id_start {order_dir}, mc.chunk_id {order_dir}
            LIMIT {limit_param}
            """,
            *params,
        )
        result = [dict(row) for row in rows]
        if order_dir == "DESC":
            result.reverse()
        return result

    async def hybrid_search_within_memory(
        self,
        *,
        memory_id: UUID,
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
        """hybrid (vector + FTS) chunk search restricted to one memory's chunks.

        Same ranking + parallel-query shape as :meth:`hybrid_search`;
        the only addition is a ``WHERE mc.memory_id = :memory_id``
        predicate that narrows the candidate pool to chunks under
        one parent memory. Auth scoping is enforced verbatim from
        the cross-memory path — the memory_id filter is purely
        additive, never a replacement for the auth triple.

        :param memory_id: parent memory UUID to scope the search to
        :ptype memory_id: UUID
        :param user_id: owning user UUID (row filter)
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
        :param chunk_signal_weights: mapping ``{"semantic", "keyword"}``
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
        memory_id_param = f"${param_offset + 1}"
        limit_param = f"${param_offset + 2}"

        # cache-bypass: vector-distance search joining memory_chunks
        # to media (via memory_id parent FK) scoped to one parent
        # memory. auth triple enforced. memory_id predicate purely
        # additive to the auth filter.
        vec_coro = self.l3_pool.fetch(
            f"""
            SELECT mc.chunk_id, mc.content, mc.summary, mc.heading_context,
                   mc.page_number, mc.memory_id,
                   mc.message_id_start, mc.message_id_end,
                   mc.embedding::text AS embedding, med.metadata_json, med.media_id,
                   1 - (mc.embedding OPERATOR(public.<=>) $1::text::public.vector) AS similarity
            FROM memory_chunks mc
            LEFT JOIN media med
              ON mc.agent_id = med.agent_id
             AND mc.memory_id = med.memory_id
            WHERE {scope_conditions}
              AND mc.memory_id = {memory_id_param}
            ORDER BY mc.embedding OPERATOR(public.<=>) $1::text::public.vector
            LIMIT {limit_param}
            """,
            embedding_str,
            *scope_params,
            memory_id,
            candidate_k,
        )

        fts_text = _build_fts_text(user_text, fts_min_len, fts_max_len)
        if fts_text:
            fts_scope_conditions, fts_scope_params, fts_param_offset = _build_user_scope_clause(
                user_id,
                agent_id=agent_id,
                customer_id=customer_id,
                start_param=2,
                table_prefix="mc",
            )
            fts_memory_id_param = f"${fts_param_offset + 1}"
            fts_limit_param = f"${fts_param_offset + 2}"
            # cache-bypass: FTS rank query joining chunks -> media via
            # memory_id, scoped to one parent memory.
            fts_coro = self.l3_pool.fetch(
                f"""
                SELECT mc.chunk_id, mc.content, mc.summary, mc.heading_context,
                       mc.page_number, mc.memory_id,
                       mc.message_id_start, mc.message_id_end,
                       mc.embedding::text AS embedding, med.metadata_json, med.media_id,
                       ts_rank_cd(mc.search_vector, websearch_to_tsquery('english', $1)) AS fts_rank
                FROM memory_chunks mc
                LEFT JOIN media med
                  ON mc.agent_id = med.agent_id
                 AND mc.memory_id = med.memory_id
                WHERE {fts_scope_conditions}
                  AND mc.memory_id = {fts_memory_id_param}
                  AND mc.search_vector @@ websearch_to_tsquery('english', $1)
                ORDER BY fts_rank DESC
                LIMIT {fts_limit_param}
                """,
                fts_text,
                *fts_scope_params,
                memory_id,
                candidate_k,
            )
            vec_rows, fts_rows = await asyncio.gather(vec_coro, fts_coro)
        else:
            vec_rows = await vec_coro
            fts_rows = []

        return _merge_chunk_search_rows(
            vec_rows=vec_rows,
            fts_rows=fts_rows,
            chunk_signal_weights=chunk_signal_weights,
            similarity_threshold=similarity_threshold,
        )


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
    is domain-truncated to 150 chars in :meth:`save_to_store` to
    match the migration-v002 VARCHAR(150) bound.
    """

    primary_key_column: str | tuple[str, ...] = ("conversation_id", "item_id")
    # rationale: ``save_to_store`` is a framework override that
    # truncates ``short_desc`` to the migration-v002 VARCHAR(150) bound;
    # the underlying ``data`` dict already carries ``conversation_id``
    # (the partition column) by construction since
    # :class:`SchemaBackedCollection` enforces required-column presence
    # before this override runs. exempting the framework override
    # keeps the partition contract on the read surface
    # (``find_by_conversation``) without weakening the static guard.
    _partition_exempt_methods: ClassVar[frozenset[str]] = frozenset({"save_to_store"})
    # v0.8.0 enrichment: ``date_created`` carries ``server_default="now()"``
    # to match prod (prod ``information_schema`` confirms the default).
    # ``date_created`` is also immutable per the standard 3tears
    # convention. The FK on ``conversation_id`` matches the prod
    # constraint ``conversation_memory_refs_conversation_id_fkey``
    # (CASCADE on parent conversation delete) -- declared at table
    # level because the inline 2-tuple form does not carry
    # ``on_delete=``. The lookup index ``ix_conversation_memory_refs_cid``
    # is declared in 3tears so the parity gate stays clean.
    schema = TableSchema(
        name="conversation_memory_refs",
        primary_key=("conversation_id", "item_id"),
        columns=[
            Column("conversation_id", UUID_TYPE, partition=True),
            Column("item_id", UUID_TYPE),
            Column("item_type", STRING_TYPE),
            Column("short_desc", STRING_TYPE),
            # v014 renamed date_added -> date_created and added
            # date_updated to align with the standard 3tears
            # (date_created, date_updated) convention and to satisfy
            # BaseCollection.save's L1-write contract.
            Column(
                "date_created",
                DATETIMETZ_TYPE,
                immutable=True,
                server_default="now()",
            ),
            Column("date_updated", DATETIMETZ_TYPE),
        ],
        foreign_keys=(
            SchemaForeignKey(
                "conversation_id",
                "conversations",
                "conversation_id",
                on_delete="CASCADE",
            ),
        ),
        indexes=(
            SchemaIndex(
                "ix_conversation_memory_refs_cid",
                "conversation_id",
            ),
        ),
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

    async def save_to_store(
        self,
        data: dict[str, Any],
        original_timestamp: datetime | None = None,
        *,
        conn: Any = None,
    ) -> int:
        """upsert one row into L3 via composite-pk ON CONFLICT.

        truncate ``short_desc`` to :data:`_MEMORY_REF_SHORT_DESC_MAX`
        (150) chars to match the migration-v002 VARCHAR(150) bound.
        the truncation is collection-specific domain logic, not
        primitive-level coercion, so it lives on this override rather
        than inside :class:`SchemaBackedCollection`.

        :param data: row data; must contain both pk columns plus
            ``item_type`` / ``short_desc`` / ``date_created`` /
            ``date_updated``
        :ptype data: dict[str, Any]
        :param original_timestamp: ignored (no CAS column on this table)
        :ptype original_timestamp: datetime | None
        :param conn: optional asyncpg-compatible connection; forwarded
            to :class:`SchemaBackedCollection` so the framework's
            transactional save_entity path stays atomic
        :ptype conn: Any
        :return: rows affected (1 on success, 0 on failure)
        :rtype: int
        """
        desc_value = data.get("short_desc")
        if isinstance(desc_value, str) and len(desc_value) > _MEMORY_REF_SHORT_DESC_MAX:
            data = dict(data)
            data["short_desc"] = desc_value[:_MEMORY_REF_SHORT_DESC_MAX]
        return await super().save_to_store(data, original_timestamp, conn=conn)

    async def find_by_conversation(
        self,
        conversation_id: UUID,
    ) -> list[MemoryRefEntity]:
        """fetch every ref for a conversation, ordered by ``date_created`` asc.

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
            SELECT conversation_id, item_id, item_type, short_desc,
                   date_created, date_updated
            FROM conversation_memory_refs
            WHERE conversation_id = $1
            ORDER BY date_created ASC
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
