"""Memories collection -- three-tier CRUD for memory entities.

Also wires the sibling collections for tables the memory package owns
but previously lacked Collection coverage: :class:`MediaCollection`,
:class:`MediaContentCollection`, :class:`MemoryChunkCollection` (adopted
under namespace-task-01 phase 8.5b). Each Collection resolves its L3
pool via the registry (``self.l3_pool``) -- the bespoke
``_postgres_pool`` field is retired in favour of the registry pattern
:class:`ConversationCollection` already uses.

Complex hybrid-search queries (vector + FTS + MMR) live as methods on
these Collections with documented ``# cache-bypass:`` comments -- the
Collection stays the single entry point for memory-table SQL even when
the query shape is not primary-key addressable and therefore cannot
benefit from L1 row caching.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import math
from collections.abc import Awaitable, Callable, Iterable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar, Final, cast
from uuid import UUID

from sqlalchemy import MetaData, Table
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
    MemoryConsolidationEntity,
    MemoryEntity,
    MemoryRefEntity,
)
from threetears.agent.memory.read_filter import ReadFilter, ReadFiltered, filter_clause, filter_guards
from threetears.agent.memory.retrieval_scope import (
    RetrievalScope,
    build_scope_conditions,
    scope_matches_nothing,
)
from threetears.agent.memory.revisions import RETRACTED_BECAUSE_PREFIX, RETRACTED_SALIENCE_PREFIX
from threetears.core.backends.schema_sql import decode_vector
from threetears.core.collections.flush import WriteBuffer
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.salience import SalienceDecay
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
    SchemaBackedCollection,
    TableSchema,
    spans_partitions,
)
from threetears.core.collections.schema_backed import (
    ForeignKey as SchemaForeignKey,
)
from threetears.core.collections.schema_backed import (
    Index as SchemaIndex,
)
from threetears.core.collections.schema_backed import (
    UniqueConstraint as SchemaUniqueConstraint,
)
from threetears.core.config import CoreConfig
from threetears.core.data.embedding_width import DEFAULT_EMBEDDING_DIM, schema_with_embedding_dim
from threetears.core.data.gin import gin_filter
from threetears.core.keyed import (
    AnyOf,
    Count,
    Fragment,
    Nearest,
    Selected,
    Joined,
    Sort,
    TextRank,
    aggregate_keyed,
    clause_guards,
    holds_any,
    keyset_page,
    read_by,
    read_by_joined,
    read_nearest,
    read_row,
    update_keyed,
)
from threetears.core.tenancy import TENANT_BY_SCHEMA
from threetears.observe import get_logger

__all__ = [
    "MEMORY_REFS_PAGE",
    "ConsolidationCycleError",
    "MediaCollection",
    "MediaContentCollection",
    "MemoriesCollection",
    "MemoryChunkCollection",
    "MemoryConsolidationsCollection",
    "MemoryRefsCollection",
    "assert_no_consolidation_cycle",
    "conversation_memory_refs_table",
    "media_content_table",
    "media_table",
    "memories_table",
    "memory_chunks_table",
    "memory_consolidations_table",
]

log = get_logger(__name__)


#: the keyword-match guard every memory, chunk and media-content FTS read filters on, over the keyed helper's row
#: ``t``, the query text its one parameter. ``websearch_to_tsquery`` turns an "or" or a leading "-" in the text into
#: OR / NOT, which YugabyteDB's GIN index refuses outright, so the predicate filters the scoped rows instead of
#: scanning the index (:func:`threetears.core.data.gin.gin_filter`). ranking (:class:`TextRank`) uses the same tsquery.
_FTS_MATCH: Final[str] = gin_filter("t.search_vector @@ websearch_to_tsquery('english', $1)")


def _as_pk_uuid(value: object) -> UUID:
    """Coerce a memory/edge id to a stdlib :class:`UUID`.

    Normalises across the two UUID types that flow through the edge
    graph: asyncpg yields stdlib ``uuid.UUID`` on reads, while the app
    passes ``uuid_utils.UUID`` (uuid7). The two are distinct types with
    distinct hashing, so the cycle-guard walk must compare a single
    canonical form or set-membership and ``==`` silently miss.
    """
    if type(value) is UUID:
        return value
    return UUID(str(value))


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
# ``TableSchema`` -- two declarations of the same shape inside one
# file. v0.8.0 enriched ``TableSchema`` (shards 01-03) so the factory
# bodies can delegate to ``to_sqlalchemy_table`` (shard 04), closing
# the duplication trap that the v0.7.5 factories themselves were
# introduced to close at the cross-package boundary.
#
# Public factory signatures are unchanged so host applications
# (current and future consumers) keep calling ``memories_table(metadata)``
# etc. without modification.
# ---------------------------------------------------------------------------


# The embedding width the memory tables are declared with: every Vector
# column in this file. It is the default of the deployment setting, not a
# fixed shape -- the collections and table factories take ``embedding_dim``
# and derive their schema at that width
# (:func:`~threetears.core.data.embedding_width.schema_with_embedding_dim`),
# and the migrations create the columns at the width they are registered
# with. 1024 is Voyage AI's voyage-4, the provider the tables were built for.
_MEMORY_VECTOR_DIM = DEFAULT_EMBEDDING_DIM


def memories_table(metadata: MetaData, *, embedding_dim: int = DEFAULT_EMBEDDING_DIM) -> Table:
    """Register the ``memories`` table on the given SA metadata.

    v0.8.0: schema declaration is now the single source of truth. This
    factory is a thin idempotency wrapper around
    :meth:`MemoriesCollection.schema.to_sqlalchemy_table`. Call this
    before ``SQLiteBackend.initialize(metadata)`` so the L1 cache gets
    the correct schema, and before Alembic ``target_metadata``
    reflection so auto-generate sees the same shape.

    :param metadata: SQLAlchemy metadata to attach the table to
    :ptype metadata: MetaData
    :param embedding_dim: the ``embedding`` column's width, the deployment's; default 1024
    :ptype embedding_dim: int
    :return: the ``memories`` :class:`Table`
    :rtype: Table
    """
    schema = schema_with_embedding_dim(MemoriesCollection.schema, embedding_dim)
    return cast(Table, schema.to_sqlalchemy_table(metadata))


def media_table(metadata: MetaData) -> Table:
    """Register the ``media`` table on the given SA metadata.

    v0.8.0: schema declaration is now the single source of truth. This
    factory is a thin idempotency wrapper around
    :meth:`MediaCollection.schema.to_sqlalchemy_table`. Includes the
    composite ``(agent_id, memory_id)`` key to ``memories`` (every media
    row attaches to a memory in its own agent partition) with
    CASCADE-on-memory-delete.

    :param metadata: SQLAlchemy metadata to attach the table to
    :ptype metadata: MetaData
    :return: the ``media`` :class:`Table`
    :rtype: Table
    """
    return cast(Table, MediaCollection.schema.to_sqlalchemy_table(metadata))


def media_content_table(metadata: MetaData, *, embedding_dim: int = DEFAULT_EMBEDDING_DIM) -> Table:
    """Register the ``media_content`` table on the given SA metadata.

    v0.8.0: schema declaration is now the single source of truth. This
    factory is a thin idempotency wrapper around
    :meth:`MediaContentCollection.schema.to_sqlalchemy_table`. Carries
    derived content (descriptions, transcripts, OCR text) for ``media``
    rows with its own embedding + search_vector columns.

    :param metadata: SQLAlchemy metadata to attach the table to
    :ptype metadata: MetaData
    :param embedding_dim: the ``embedding`` column's width, the deployment's; default 1024
    :ptype embedding_dim: int
    :return: the ``media_content`` :class:`Table`
    :rtype: Table
    """
    schema = schema_with_embedding_dim(MediaContentCollection.schema, embedding_dim)
    return cast(Table, schema.to_sqlalchemy_table(metadata))


def memory_chunks_table(metadata: MetaData, *, embedding_dim: int = DEFAULT_EMBEDDING_DIM) -> Table:
    """Register the ``memory_chunks`` table on the given SA metadata.

    v0.8.0: schema declaration is now the single source of truth. This
    factory is a thin idempotency wrapper around
    :meth:`MemoryChunkCollection.schema.to_sqlalchemy_table`. Carries
    chunked text + embeddings parented to a ``memories`` row, plus the
    v0.14.0 transcript-chunk ``message_id_start`` / ``message_id_end``
    provenance columns.

    :param metadata: SQLAlchemy metadata to attach the table to
    :ptype metadata: MetaData
    :param embedding_dim: the ``embedding`` column's width, the deployment's; default 1024
    :ptype embedding_dim: int
    :return: the ``memory_chunks`` :class:`Table`
    :rtype: Table
    """
    schema = schema_with_embedding_dim(MemoryChunkCollection.schema, embedding_dim)
    return cast(Table, schema.to_sqlalchemy_table(metadata))


def memory_consolidations_table(metadata: MetaData) -> Table:
    """Register the ``memory_consolidations`` edge table on ``metadata``.

    Presence/aliveness program (v026). Thin idempotency wrapper around
    :meth:`MemoryConsolidationsCollection.schema.to_sqlalchemy_table`.
    Call before ``SQLiteCacheManager.initialize(metadata)`` so the L1
    cache builds with the composite-pk edge table, and before Alembic
    ``target_metadata`` reflection so auto-generate sees the same shape.

    :param metadata: SQLAlchemy metadata to attach the table to
    :ptype metadata: MetaData
    :return: the ``memory_consolidations`` :class:`Table`
    :rtype: Table
    """
    return cast(Table, MemoryConsolidationsCollection.schema.to_sqlalchemy_table(metadata))


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


#: the most refs :meth:`MemoryRefsCollection.find_by_conversation` reads by default: the surfaced-refs FIFO a
#: conversation keeps per agent (``threetears.agent.tools.context``) holds this many and deletes the oldest past it
MEMORY_REFS_PAGE: Final[int] = 50

#: the columns :meth:`MemoriesCollection.hybrid_search` reads of each candidate, the embedding as its text
_HYBRID_MEMORY_COLUMNS: Final = (
    "memory_id",
    "content",
    "summary",
    "type_memory",
    "date_created",
    Selected("t.embedding::text", "embedding"),
)


#: the columns :meth:`MediaContentCollection.hybrid_search` reads of each candidate, the embedding as its text
_HYBRID_CONTENT_COLUMNS: Final = (
    "content_id",
    "content",
    "summary",
    "content_type",
    "media_id",
    "date_created",
    Selected("t.embedding::text", "embedding"),
)


#: the columns :meth:`MemoryChunkCollection.hybrid_search` and :meth:`MemoryChunkCollection.hybrid_search_within_memory`
#: read of each candidate, the embedding as its text
_HYBRID_CHUNK_COLUMNS: Final = (
    "chunk_id",
    "content",
    "summary",
    "heading_context",
    "page_number",
    "memory_id",
    "message_id_start",
    "message_id_end",
    Selected("t.embedding::text", "embedding"),
)


def _date_guards(column: str, after: datetime | None, before: datetime | None) -> tuple[Fragment, ...]:
    """an inclusive date window on ``column`` as a keyed helper's ``where=`` guards, each bound on its own.

    :param column: the column, as the guard names it (``t.date_created``), a TRUSTED identifier
    :ptype column: str
    :param after: inclusive lower bound; none when None
    :ptype after: datetime | None
    :param before: inclusive upper bound; none when None
    :ptype before: datetime | None
    :return: the guards
    :rtype: tuple[Fragment, ...]
    """
    return (
        *(() if after is None else (Fragment(f"{column} >= $1", (after,)),)),
        *(() if before is None else (Fragment(f"{column} <= $1", (before,)),)),
    )


def _file_window_guards(after: datetime | None, before: datetime | None) -> tuple[Fragment, ...]:
    """a media-content read's window on its file's ``date_created``, as the one guard: the content row's file, by its
    key, inside the inclusive window; none when neither bound is set (the foreign key gives every content row a file).

    :param after: inclusive lower bound on the file's date; none when None
    :ptype after: datetime | None
    :param before: inclusive upper bound on the file's date; none when None
    :ptype before: datetime | None
    :return: the guard, or none
    :rtype: tuple[Fragment, ...]
    """
    if after is None and before is None:
        return ()
    return (
        Fragment(
            "EXISTS (SELECT 1 FROM media med WHERE med.agent_id = t.agent_id AND med.media_id = t.media_id"
            " AND ($1::timestamptz IS NULL OR med.date_created >= $1) AND ($2::timestamptz IS NULL OR med.date_created <= $2))",
            (after, before),
        ),
    )


#: a scope's every-tag test (:attr:`RetrievalScope.tags_all`), the containment ``idx_memories_tags`` serves: each
#: search spreads it into its read itself (``*((Fragment(_EVERY_TAG, ...),) if ... else ())``), so the no-whole-reads
#: walk sees the read the index's purpose names; :func:`_scope_guards` leaves it out. The tags go as the RAW list: the
#: family's jsonb codec dumps it (see :func:`build_scope_conditions`)
_EVERY_TAG = "t.tags @> $1::jsonb"


def _scope_guards(scope: RetrievalScope | None) -> tuple[Fragment, ...]:
    """a :class:`RetrievalScope` as a keyed helper's ``where=`` guard (:func:`build_scope_conditions`, numbered from
    ``$1``), all but its every-tag test (:data:`_EVERY_TAG`, which each search writes itself); none for no scope.
    callers check :func:`scope_matches_nothing` first.

    :param scope: the scope, possibly ``None``
    :ptype scope: RetrievalScope | None
    :return: the guard, or none
    :rtype: tuple[Fragment, ...]
    """
    rest = None if scope is None else dataclasses.replace(scope, tags_all=None)
    conditions, params, _next = build_scope_conditions(rest, next_param=1)
    return (Fragment(" AND ".join(conditions), tuple(params)),) if conditions else ()


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


# Set on a merged candidate that reached the pool through the FTS leg. Private to
# this module: stripped in :func:`_admit_and_rank` before rows reach a caller.
_FTS_MATCHED_KEY = "_fts_matched"


def _admit_and_rank(
    candidates: list[dict[str, Any]],
    *,
    similarity_threshold: float,
) -> list[dict[str, Any]]:
    """Admit candidates on evidence of relevance, then order them by the composite.

    Admission and ranking answer different questions, and conflating them
    silently un-recalls old memories.

    ``hybrid_score`` is a weighted sum over semantic / keyword / recency. Gating
    admission on that sum means **a signal that is absent lowers the attainable
    ceiling**: a query whose lexemes match no row contributes a keyword term of
    exactly 0, and a memory older than a few multiples of the recency half-life
    contributes a recency term indistinguishable from 0. Only the semantic weight
    is left to clear a threshold calibrated as though all three could fire, so a
    perfectly relevant month-old memory is unreachable at any similarity --
    ``0.55 x 0.50 = 0.275`` against a 0.4 gate, for weights and a cosine range
    that are both ordinary.

    Recency is the sharper half of the error: how recently something was written
    is not evidence that it answers *this* query. It is a tiebreaker among
    relevant candidates, so it ranks here and never admits.

    Admission therefore consults only per-query evidence:

    - semantic -- ``similarity >= similarity_threshold``
    - keyword -- the row actually matched the FTS query

    The keyword arm matters because an FTS-only row carries ``similarity`` 0.0 as
    a *placeholder* rather than a measurement (it never entered the vector
    query's top-N, so no distance was computed for it). Admitting on similarity
    alone would drop every keyword-only hit.

    :param candidates: merged candidate rows, each carrying ``similarity`` and a
        precomputed ``hybrid_score``; mutated to strip the private FTS marker
    :ptype candidates: list[dict[str, Any]]
    :param similarity_threshold: floor on semantic similarity for admission
    :ptype similarity_threshold: float
    :return: admitted rows, sorted by ``hybrid_score`` DESC
    :rtype: list[dict[str, Any]]
    """
    admitted = [c for c in candidates if c["similarity"] >= similarity_threshold or c.get(_FTS_MATCHED_KEY, False)]
    # Strip from every candidate, not just the admitted ones: `or` short-circuits,
    # so a row admitted on similarity never evaluates the second arm.
    for c in candidates:
        c.pop(_FTS_MATCHED_KEY, None)
    admitted.sort(key=lambda c: c["hybrid_score"], reverse=True)
    return admitted


#: What a candidate scored on this query alone, kept on the row for
#: :func:`_cap_recent` and stripped before rows reach a caller.
_EVIDENCE_KEY = "_evidence"


def _cap_recent(
    ranked: list[dict[str, Any]],
    *,
    top_k: int,
    recent_cap: int | None,
    recent_window_hours: float,
) -> list[dict[str, Any]]:
    """The top ``top_k`` of ``ranked``, holding at most ``recent_cap`` memories written inside the window.

    The composite ranks: its order is kept. What the cap decides is which
    recent memories may be in the list at all, and it decides that on evidence
    that the memory answers this query (semantic and keyword), never on how new
    it is. Chosen by the composite, the capped places go to whatever was
    written last, and a five-day-old memory that answers the message exactly
    loses its place to three from this morning that barely touch it.

    A memory older than the window is never dropped here, and with fewer
    recent candidates than the cap nothing is.

    :param ranked: admitted candidates, best composite first, each carrying
        ``date_created`` and the private evidence score
    :ptype ranked: list[dict[str, Any]]
    :param top_k: how many to return
    :ptype top_k: int
    :param recent_cap: the most recent memories allowed; ``None`` is no cap
    :ptype recent_cap: int | None
    :param recent_window_hours: how old a memory may be and still count as recent
    :ptype recent_window_hours: float
    :return: the list, best composite first
    :rtype: list[dict[str, Any]]
    """
    kept = ranked
    if recent_cap is not None:
        now = datetime.now(UTC)
        recent = [c for c in ranked if (now - c["date_created"]).total_seconds() / 3600 <= recent_window_hours]
        if len(recent) > recent_cap:
            # stable: among equal evidence the better composite keeps its place
            allowed = sorted(recent, key=lambda c: c.get(_EVIDENCE_KEY, 0.0), reverse=True)[: max(recent_cap, 0)]
            allowed_ids = {id(c) for c in allowed}
            recent_ids = {id(c) for c in recent}
            kept = [c for c in ranked if id(c) not in recent_ids or id(c) in allowed_ids]
    for c in ranked:
        c.pop(_EVIDENCE_KEY, None)
    return kept[:top_k]


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
    :param similarity_threshold: floor on semantic similarity for
        admission; see :func:`_admit_and_rank` for why the composite
        does not gate
    :ptype similarity_threshold: float
    :return: ranked list of chunk dicts, admission-filtered, sorted
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
        merged[ckid][_FTS_MATCHED_KEY] = True

    candidates = list(merged.values())
    if not candidates:
        return []

    _normalize_scores(candidates, "fts_rank")

    for c in candidates:
        c["hybrid_score"] = round(
            chunk_signal_weights["semantic"] * c["similarity"] + chunk_signal_weights["keyword"] * c["fts_rank"],
            4,
        )

    return _admit_and_rank(candidates, similarity_threshold=similarity_threshold)


# explicit column list (a keyed helper's ``columns=``) for whole ``memories`` rows read on the raw pool. the
# ``embedding`` (pgvector) column is cast to ``::text`` so asyncpg can decode
# it on the no-codec L3 pool -- a bare ``SELECT *`` returns the raw ``vector``
# type (OID 8078) that asyncpg has no codec for and raises
# ``UnsupportedClientFeatureError``. the entity deserializer parses the
# bracketed-text form. tags arrive as their JSON string (no codec) and the
# entity accessor decodes them. keep in sync with the memories migration column set.
_MEMORIES_SELECTED: tuple[str | Selected, ...] = (
    "memory_id",
    "agent_id",
    "customer_id",
    "user_id",
    "type_memory",
    "content",
    "date_created",
    "date_updated",
    Selected("t.embedding::text", "embedding"),
    "conversation_id",
    "message_id_source",
    "summary",
    "search_vector",
    "alias",
    "salience",
    "last_decayed_at",
    "last_accessed",
    "evergreen",
    "superseded_by",
    "tags",
)
#: newest first, as every owner listing of memories reads them
_NEWEST_FIRST = (Sort("date_created", descending=True),)


#: A retracted memory (:func:`~threetears.agent.memory.revisions.retract`) no
#: longer holds: no search, dedup or dream reads it, and only a recall by its
#: id still opens it. ``?`` is the GIN-indexed key test on the tags array.
_NOT_RETRACTED = "NOT (COALESCE(tags, '[]'::jsonb) ? 'retracted')"

#: What a memory line is marked from, selected beside the memory by every read
#: that shows one to an agent: its name, whether it is permanent, its tags
#: (retired, and why), what replaced it and when, and a correction written
#: beside it. ``superseded_at`` is the replacement's own date and is NULL when
#: the replacement no longer exists, which is how a reader tells a memory that
#: is replaced from one whose replacement is gone. ``corrected_by`` is a linked
#: memory written beside one that was not replaced (a permanent memory's
#: correction). Each read selects them as its own columns (the two dates and the
#: correction as correlated ``Selected`` sub-selects over its row ``t``) and
#: answers them through :func:`_marks`.
_MEMORY_MARK_KEYS = ("alias", "evergreen", "tags", "superseded_by", "superseded_at", "corrected_by")

#: the columns :meth:`MemoriesCollection.find_active_for_consolidation` loads for each candidate
_CANDIDATE_COLUMNS: tuple[str | Selected, ...] = (
    "memory_id",
    "customer_id",
    "user_id",
    "conversation_id",
    "type_memory",
    "content",
    Selected("t.embedding::text", "embedding"),
    "date_created",
    "tags",
)
#: what every consolidation candidate also is, after its owner: not pinned, embedded, not superseded by a live gist,
#: not retracted
_CANDIDATE_GUARDS: tuple[Fragment, ...] = (
    Fragment("NOT t.evergreen", ()),
    Fragment("t.embedding IS NOT NULL", ()),
    Fragment(
        "(t.superseded_by IS NULL OR NOT EXISTS ("
        "SELECT 1 FROM memories g WHERE g.agent_id = t.agent_id AND g.memory_id = t.superseded_by))",
        (),
    ),
    Fragment("NOT (COALESCE(t.tags, '[]'::jsonb) ? 'retracted')", ()),
)
#: oldest first, completed by the key (rows created at one instant come in one order on every read)
_OLDEST_FIRST = (Sort("date_created"), Sort("memory_id"))

#: The tags :meth:`MemoriesCollection.mark_retracted` writes and
#: :meth:`MemoriesCollection.clear_retracted` takes off again.
_RETRACTED_BECAUSE = RETRACTED_BECAUSE_PREFIX
_RETRACTED_SALIENCE = RETRACTED_SALIENCE_PREFIX


def _marks(row: Any) -> dict[str, Any]:
    """the mark columns of a memory row, as a dict a caller can merge into its own."""
    return {key: row[key] for key in _MEMORY_MARK_KEYS}


def _chunk_parent_clause(read_filter: ReadFilter | None, chunk: str, args: list[Any]) -> str:
    """the host's read filter for a chunk read, decided on the chunk's parent memory's tags.

    :param read_filter: the host's filter, or ``None``
    :ptype read_filter: ReadFilter | None
    :param chunk: the ``memory_chunks`` row's alias or table name in the query
    :ptype chunk: str
    :param args: the query's positional values so far; the filter's are appended
    :ptype args: list[Any]
    :return: an ``AND EXISTS`` over the parent memory, or ``""`` when nothing restricts this read
    :rtype: str
    """
    probe = filter_clause(read_filter, "pm.tags", args)
    if not probe:
        return ""
    return (
        f" AND EXISTS (SELECT 1 FROM memories pm WHERE pm.agent_id = {chunk}.agent_id "
        f"AND pm.memory_id = {chunk}.memory_id{probe})"
    )


def _chunk_parent_guards(read_filter: ReadFilter | None) -> tuple[Fragment, ...]:
    """:func:`_chunk_parent_clause` over a keyed helper's chunk row (``t``), as its ``where=`` guards.

    :param read_filter: the host's filter, or ``None``
    :ptype read_filter: ReadFilter | None
    :return: the guards; none when nothing restricts this read
    :rtype: tuple[Fragment, ...]
    """
    return clause_guards(_chunk_parent_clause, read_filter, "t")


def _media_parent_guards(read_filter: ReadFilter | None) -> tuple[Fragment, ...]:
    """:func:`_media_parent_clause` over a keyed helper's content row (``t``), as its ``where=`` guards.

    :param read_filter: the host's filter, or ``None``
    :ptype read_filter: ReadFilter | None
    :return: the guards; none when nothing restricts this read
    :rtype: tuple[Fragment, ...]
    """
    return clause_guards(_media_parent_clause, read_filter, "t")


def _media_parent_clause(read_filter: ReadFilter | None, content: str, args: list[Any]) -> str:
    """the host's read filter for a media-content read, decided on the tags of its media row's memory.

    A content row whose media row or paired memory cannot be found matches no
    ``EXISTS`` and is excluded while a filter applies.

    :param read_filter: the host's filter, or ``None``
    :ptype read_filter: ReadFilter | None
    :param content: the ``media_content`` row's alias or table name in the query
    :ptype content: str
    :param args: the query's positional values so far; the filter's are appended
    :ptype args: list[Any]
    :return: an ``AND EXISTS`` over the paired memory, or ``""`` when nothing restricts this read
    :rtype: str
    """
    probe = filter_clause(read_filter, "pm.tags", args)
    if not probe:
        return ""
    return (
        " AND EXISTS (SELECT 1 FROM media pmed JOIN memories pm "
        "ON pm.agent_id = pmed.agent_id AND pm.memory_id = pmed.memory_id "
        f"WHERE pmed.agent_id = {content}.agent_id AND pmed.media_id = {content}.media_id{probe})"
    )


class MemoriesCollection(SchemaBackedCollection[MemoryEntity], ReadFiltered):
    """Collection for memory entities with three-tier caching.

    CRUD goes through :meth:`get` / :meth:`save_entity` / :meth:`delete`
    so L1 / L2 / L3 tiers stay coherent. Hybrid-search methods
    (:meth:`hybrid_search`, :meth:`search_by_ids`,
    :meth:`find_similar_for_dedup`, :meth:`count_by_user`) absorb the
    vector + FTS + MMR queries that used to live raw on the pool; they
    carry ``# cache-bypass:`` justification because the query shape is
    not primary-key-addressable and therefore cannot benefit from the
    L1 row cache -- but keeping them on the Collection preserves the
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

    tenancy = TENANT_BY_SCHEMA

    #: salience decay on read (:class:`~threetears.core.collections.salience.SalienceDecay`, pinned by ``evergreen``),
    #: off by default: with it, the ambient reads' salience floor compares the effective salience and a reinforcement
    #: stamps the decay so far into the base
    salience_decay: ClassVar[SalienceDecay | None] = None

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
            # NOT NULL since v001; every save stamps it (it is the CAS fence).
            Column("date_updated", DATETIMETZ_TYPE),
            # v024 (presence/aliveness): stored salience substrate.
            # NUMERIC(5,4) with a server default so existing INSERTs that
            # omit it apply 0.5 (metadata-only add, no table rewrite).
            # ``immutable=True`` excludes it from the entity-save UPDATE
            # generator: salience is written ONLY by the raw decay / bump
            # SQL (bump_salience; decay is read, SalienceDecay), never by a full-entity
            # save. Without this, an unrelated content UPDATE would carry a
            # stale cached salience and revert a scheduled decay (design §2
            # "the UPDATE narrows to content + embedding + date_updated").
            # Insert still applies the value / server default.
            Column(
                "salience",
                NUMERIC_TYPE,
                precision=5,
                scale=4,
                nullable=False,
                server_default="0.5",
                immutable=True,
            ),
            # decay anchor: age is measured from the last decay run, not
            # last access, so total decay over a period is cadence-safe.
            # ``immutable=True`` for the same reason as salience -- written
            # only by the raw decay pass, excluded from the entity UPDATE.
            Column("last_decayed_at", DATETIMETZ_TYPE, nullable=True, immutable=True),
            # reinforcement telemetry: stamped on ambient retrieval by the
            # raw bump pass; excluded from the entity UPDATE (immutable).
            Column("last_accessed", DATETIMETZ_TYPE, nullable=True, immutable=True),
            # pin for core identity facts: excluded from decay AND bump.
            # Mutable via the entity path (an admin pin/unpin toggles it).
            Column(
                "evergreen",
                BOOL_TYPE,
                nullable=False,
                server_default="false",
            ),
            # soft ref (no FK) to a consolidation gist; ambient retrieval
            # excludes non-null, direct recall still finds it. Written only
            # by the raw mark_superseded pass -> ``immutable=True`` so an
            # entity save can't revert it.
            Column("superseded_by", UUID_TYPE, nullable=True, immutable=True),
            # v025 (presence/aliveness): nullable JSONB label set (a JSON
            # array of strings). Mutable; GIN-indexed below for
            # containment / existence queries.
            Column("tags", JSONB_TYPE, nullable=True),
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
            # v036: one user's memories under one agent, newest first: every per-user read and count, and the
            # history pool's date bound read off it. it replaced idx_memories_agent_user (agent_id, user_id) and
            # ix_memories_user_date (user_id, date_created), which YugabyteDB passed over for the owner index
            SchemaIndex(
                "ix_memories_agent_user_date",
                "agent_id",
                "user_id",
                "date_created",
                desc="date_created",
                purpose=(
                    "packages/agent/memory/src/threetears/agent/memory/collections.py"
                    "::MemoriesCollection.count_by_user; "
                    "packages/agent/memory/src/threetears/agent/memory/collections.py"
                    "::MemoriesCollection.find_similar_for_dedup; "
                    "packages/agent/memory/src/threetears/agent/memory/collections.py"
                    "::MemoriesCollection.search_by_semantic; "
                    "packages/agent/memory/src/threetears/agent/memory/collections.py"
                    "::MemoriesCollection.search_by_fts; "
                    "packages/agent/memory/src/threetears/agent/memory/collections.py"
                    "::MemoriesCollection.list_for_user; "
                    "packages/agent/memory/src/threetears/agent/memory/collections.py"
                    "::MemoriesCollection.find_by_scope; "
                    "packages/agent/memory/src/threetears/agent/memory/collections.py"
                    "::MemoriesCollection.find_for_user_in_agents; "
                    "packages/agent/memory/src/threetears/agent/memory/collections.py"
                    "::MemoriesCollection.hybrid_search._nearest; "
                    "packages/core/src/threetears/core/collections/merge.py::repoint_user_rows"
                ),
                plan_test="packages/agent/memory/tests/integration/test_index_plans.py::test_memories_agent_user",
                reason=(
                    "memories carries seven indexes: one user's memories under one agent, every per-user read and count"
                ),
            ),
            # v038: the memories one or two source messages wrote, for a writer's same-event lookup. Before it the
            # lookup read the whole owner (a 20,000-memory owner: PostgreSQL scanned memories sequentially,
            # YugabyteDB walked the owner's primary-key range; K1's probe and S2's red plan test, 2026-11-15).
            # Led by the message so the enforced foreign key's ON DELETE SET NULL finds a deleted message's
            # memories by it too; agent and user after it so YugabyteDB, which takes the index binding the most
            # equality pins, prefers it to the owner index.
            SchemaIndex(
                "ix_memories_message_source",
                "message_id_source",
                "agent_id",
                "user_id",
                purpose=(
                    "packages/agent/memory/src/threetears/agent/memory/collections.py"
                    "::MemoriesCollection.find_same_event"
                ),
                plan_test=(
                    "packages/agent/memory/tests/integration/test_vector_index_plans.py"
                    "::test_ix_memories_message_source_serves_a_writers_same_event_lookup"
                ),
                reason=(
                    "memories carries seven indexes: the few memories a turn's source messages wrote, before a "
                    "writer stores the same event again"
                ),
            ),
            # v004: every memory a conversation produced, for the
            # conversation-scoped reads and the delete cascade from a
            # conversation. no other index leads with conversation_id.
            SchemaIndex(
                "idx_mem_conversation",
                "conversation_id",
                purpose=(
                    "packages/agent/memory/src/threetears/agent/memory/collections.py"
                    "::MemoryChunkCollection.find_by_conversation_id"
                ),
                plan_test="packages/agent/memory/tests/integration/test_index_plans.py::test_mem_conversation",
                reason="memories carries seven indexes: the memories one conversation produced, for its chunks",
            ),
            SchemaIndex(
                "ix_memories_user_alias",
                "agent_id",
                "user_id",
                "alias",
                unique=True,
                where="alias IS NOT NULL",
                purpose=(
                    "packages/agent/memory/src/threetears/agent/memory/collections.py::MemoriesCollection.find_by_alias; "
                    "packages/agent/memory/src/threetears/agent/memory/merge.py::lock_colliding_memories; "
                    "packages/agent/memory/src/threetears/agent/memory/merge.py::delete_colliding_memories"
                ),
                plan_test="packages/agent/memory/tests/integration/test_index_plans.py::test_memories_user_alias",
                reason="memories carries seven indexes: a memory by its name, unique per agent and user",
            ),
            # v0.8.1: parity-gate enrichments relocated from upstream
            # alembic. tenancy-scope btree composites used by the
            # cross-tenant access guards.
            SchemaIndex(
                "idx_memories_agent_customer_user",
                "agent_id",
                "customer_id",
                "user_id",
                purpose=(
                    "packages/agent/memory/src/threetears/agent/memory/collections.py"
                    "::MemoriesCollection.find_by_user; "
                    "packages/agent/memory/src/threetears/agent/memory/collections.py"
                    "::MemoriesCollection._ids_by_tags; "
                    "packages/agent/memory/src/threetears/agent/memory/collections.py"
                    "::MemoriesCollection.find_by_scope; "
                    "packages/agent/memory/src/threetears/agent/memory/collections.py"
                    "::MemoriesCollection.find_active_for_consolidation; "
                    "packages/agent/memory/src/threetears/agent/memory/collections.py"
                    "::MemoriesCollection.hybrid_search; "
                    "packages/agent/memory/src/threetears/agent/memory/collections.py"
                    "::MemoriesCollection.hybrid_search._nearest; "
                    "packages/agent/memory/src/threetears/agent/memory/collections.py"
                    "::MemoriesCollection.find_for_user_in_agents"
                ),
                plan_test=(
                    "packages/agent/memory/tests/integration/test_index_plans.py::test_memories_agent_customer_user"
                ),
                reason=(
                    "memories carries seven indexes: one owner's memories by the whole owner triple, every ambient read"
                ),
            ),
            # no GIN over ``search_vector``: v027 dropped it. the keyword
            # predicate goes through ``gin_filter`` (ybgin refuses a
            # multi-entry scan), so the index had no reader.
            # v0.8.1: HNSW vector-similarity index with the
            # ``vector_cosine_ops`` opclass; ``m`` / ``ef_construction``
            # parameters mirror prod (upstream alembic). these are
            # strings to match the textual ``WITH (key = value)`` DDL
            # syntax pgvector emits.
            # KEEP (coordinator, 2026-11-15; v001 idx_mem_embedding, renamed in 8060d75d7): an owner with many
            # memories is read nearest-first through it on both engines (an owner of 20,000: PostgreSQL and
            # YugabyteDB walk it for every read named below), a small one is sorted exactly off the owner index
            # (test_vector_index_plans holds both). find_same_event is no reader: ix_memories_message_source serves it.
            # Keep or drop on dev's owner sizes is a later performance call, not this declaration's.
            # KNOWN FOLLOW-UP, not fixed here (owner #101, 2026-11-14; a 50-row owner beside a 20,000-row one,
            # forced through the index): on YugabyteDB a small owner read through ybhnsw took ~100 ms at k=10
            # (k=50: 1.6 ms), old and new code alike. Both planners keep a small owner on the owner index today
            # (test_vector_index_plans), so it bites only when a plan turns to ybhnsw for one.
            SchemaIndex(
                "ix_memories_embedding_hnsw",
                "embedding",
                using="hnsw",
                ops={"embedding": "vector_cosine_ops"},
                pg_with={"m": "16", "ef_construction": "64"},
                purpose=(
                    "packages/agent/memory/src/threetears/agent/memory/collections.py"
                    "::MemoriesCollection.find_similar_for_dedup; "
                    "packages/agent/memory/src/threetears/agent/memory/collections.py"
                    "::MemoriesCollection.search_by_semantic; "
                    "packages/agent/memory/src/threetears/agent/memory/collections.py"
                    "::MemoriesCollection.hybrid_search._nearest"
                ),
                plan_test=(
                    "packages/agent/memory/tests/integration/test_vector_index_plans.py"
                    "::test_ix_memories_embedding_hnsw_serves_a_heavy_owners_nearest_reads"
                ),
                reason=(
                    "memories carries seven indexes: the k nearest memories of an owner too large to sort exactly, "
                    "for dedup, the search tool and ambient retrieval"
                ),
            ),
            # v025 (presence/aliveness): GIN over the ``tags`` JSONB label
            # set serves containment (``tags @> '["identity"]'``) and
            # existence (``tags ? 'identity'``) queries. KEEP (coordinator, 2026-11-15): a scoped search's
            # every-tag test (RetrievalScope.tags_all, ``tags @>``, retrieval_scope.build_scope_conditions) is read
            # through it on both engines when the tag is rarer than the owner's rows (test_vector_index_plans).
            # The any-tag test goes through gin_filter and never reads it (ybgin refuses a multi-key lookup).
            SchemaIndex(
                "idx_memories_tags",
                "tags",
                using="gin",
                purpose=(
                    "packages/agent/memory/src/threetears/agent/memory/collections.py"
                    "::MemoriesCollection.search_by_semantic; "
                    "packages/agent/memory/src/threetears/agent/memory/collections.py"
                    "::MemoriesCollection.search_by_fts; "
                    "packages/agent/memory/src/threetears/agent/memory/collections.py"
                    "::MemoriesCollection.hybrid_search; "
                    "packages/agent/memory/src/threetears/agent/memory/collections.py"
                    "::MemoriesCollection.hybrid_search._nearest"
                ),
                plan_test=(
                    "packages/agent/memory/tests/integration/test_vector_index_plans.py"
                    "::test_idx_memories_tags_serves_a_scoped_searchs_every_tag_test"
                ),
                reason=(
                    "memories carries seven indexes: a scoped search's rare tag among one owner's many memories, "
                    "found by containment"
                ),
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
        *,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
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
        :param embedding_dim: the ``embedding`` column's width, the deployment's (default 1024);
            a write carrying an embedding of another width is refused before any statement runs
        :ptype embedding_dim: int
        :raises ValueError: when ``embedding_dim`` is below 1 or above 16,000
        """
        self._authorizer = authorizer
        self.schema = schema_with_embedding_dim(type(self).schema, embedding_dim)
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
        rows = await read_by(
            self.l3_pool,
            "memories",
            by={"agent_id": agent_id, "customer_id": customer_id, "user_id": user_id},
            columns=_MEMORIES_SELECTED,
            where=filter_guards(self.read_filter, "t.tags"),
            order=_NEWEST_FIRST,
        )
        entities: list[MemoryEntity] = []
        for row in rows:
            data = dict(row)
            entity = self.entity_class(data, is_new=False, collection=self)
            entity.original_date_updated = data.get("date_updated")
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
        required for evaluator invocation. ``user_id`` must be given:
        the read is keyed by the owner's index, and ``agent_id`` alone
        or with only ``customer_id`` (every memory of the agent, or of a
        whole customer) is refused (0.95.0).

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
            ``customer_id``, or without ``user_id``
        """
        if user_id is None:
            raise ValueError(
                "find_by_scope reads one user's memories: give user_id (with customer_id or without);"
                " agent_id alone, or with only customer_id, would read a whole agent or customer",
            )
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
        # cache-bypass: one owner's rows by the owner's index are not primary-
        # key addressable; L1 row cache would not help. method on
        # Collection preserves single entry point + rbac gating. each owner
        # form's statement writes its owner out, so the read's key is in its
        # own text.
        owner: dict[str, Any] = (
            {"agent_id": agent_id, "user_id": user_id}
            if customer_id is None
            else {"agent_id": agent_id, "customer_id": customer_id, "user_id": user_id}
        )
        rows = await read_by(
            self.l3_pool,
            "memories",
            by=owner,
            columns=_MEMORIES_SELECTED,
            where=filter_guards(self.read_filter, "t.tags"),
            order=_NEWEST_FIRST,
        )

        entities: list[MemoryEntity] = []
        for row in rows:
            data = dict(row)
            entity = self.entity_class(data, is_new=False, collection=self)
            entity.original_date_updated = data.get("date_updated")
            entities.append(entity)
        return entities

    async def find_ids_by_tags(
        self,
        *,
        agent_id: UUID,
        customer_id: UUID,
        user_id: UUID,
        tags_any: tuple[str, ...],
    ) -> list[UUID]:
        """the ids of one owner's memories carrying at least one of ``tags_any``.

        for a host that files derived memories under a tag (a source document, a scene) and must
        drop them when the source goes. scoped to the full owner triple: the same tag on another
        agent's, customer's or user's rows is never found. an empty ``tags_any`` names nothing and
        finds nothing, never every row (:func:`scope_matches_nothing`). the caller authorizes.

        :param agent_id: partition column on memories
        :ptype agent_id: UUID
        :param customer_id: owning customer
        :ptype customer_id: UUID
        :param user_id: owning user
        :ptype user_id: UUID
        :param tags_any: a row qualifies when it carries any of these
        :ptype tags_any: tuple[str, ...]
        :return: the matching memory ids, oldest first
        :rtype: list[UUID]
        """
        return await self._ids_by_tags(
            agent_id=agent_id, customer_id=customer_id, user_id=user_id, tags_any=tags_any, filtered=True
        )

    async def _ids_by_tags(
        self,
        *,
        agent_id: UUID,
        customer_id: UUID,
        user_id: UUID,
        tags_any: tuple[str, ...],
        filtered: bool,
    ) -> list[UUID]:
        """the ids :meth:`find_ids_by_tags` names; ``filtered=False`` for a delete, which must reach every row."""
        scope = RetrievalScope(tags_any=tags_any)
        if self.l3_pool is None or scope_matches_nothing(scope):
            return []
        allowed = clause_guards(filter_clause, self.read_filter, "t.tags") if filtered else ()
        # cache-bypass: one owner's rows by tag, found through the owner's index (not the primary key).
        rows = await read_by(
            self.l3_pool,
            "memories",
            by={"agent_id": agent_id, "customer_id": customer_id, "user_id": user_id},
            columns=("memory_id",),
            where=[*_scope_guards(scope), *allowed],
            order=[Sort("date_created")],
        )
        return [row["memory_id"] for row in rows]

    async def delete_by_tags(
        self,
        *,
        agent_id: UUID,
        customer_id: UUID,
        user_id: UUID,
        tags_any: tuple[str, ...],
        caller_user_id: UUID | None = None,
        caller_agent_id: UUID | None = None,
    ) -> int:
        """delete one owner's memories carrying at least one of ``tags_any``, from every tier.

        each goes through :meth:`delete`, so it leaves the row cache and L2 and its removal is
        announced like any delete; a raw ``DELETE`` would leave a deleted memory answering from a
        cache. the same scoping and empty-means-nothing rule as :meth:`find_ids_by_tags`.

        given ``caller_user_id``, the rbac evaluator decides ``memory.write`` on the ``(agent_id,
        customer_id)`` namespace before anything is found, as :meth:`save_memory` does; a refusal deletes
        nothing. without one it is an agent-internal path (as :meth:`find_by_scope` without a caller), for
        a host that authorizes upstream. one delete per memory, not atomic: a failure part way leaves the
        ones before it deleted and raises.

        :param agent_id: partition column on memories
        :ptype agent_id: UUID
        :param customer_id: owning customer
        :ptype customer_id: UUID
        :param user_id: owning user
        :ptype user_id: UUID
        :param tags_any: a row qualifies when it carries any of these
        :ptype tags_any: tuple[str, ...]
        :param caller_user_id: invoking user, evaluated for ``memory.write`` when given
        :ptype caller_user_id: UUID | None
        :param caller_agent_id: invoking agent, passed to the evaluator
        :ptype caller_agent_id: UUID | None
        :return: how many memories were deleted
        :rtype: int
        :raises MemoryAccessDenied: when the evaluator refuses the caller
        """
        if caller_user_id is not None:
            await authorize_memory_access(
                action=ACTION_MEMORY_WRITE,
                agent_id=agent_id,
                customer_id=customer_id,
                caller_user_id=caller_user_id,
                caller_agent_id=caller_agent_id,
                deps=self._authorizer,
            )
        # unfiltered: a delete by tag removes the host's rows whether or not this task may read them
        ids = await self._ids_by_tags(
            agent_id=agent_id, customer_id=customer_id, user_id=user_id, tags_any=tags_any, filtered=False
        )
        for memory_id in ids:
            await self.delete((agent_id, memory_id))  # the key is (agent_id, memory_id)
        return len(ids)

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
            agent-internal writes -- owner short-circuit path)
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

        returns a boolean rather than an exact count -- every caller
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
        value = await holds_any(
            self.l3_pool,
            "memories",
            order=["agent_id", "user_id", "date_created"],
            by={"agent_id": agent_id, "user_id": user_id},
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
        # Self-healing supersession filter (matches hybrid_search +
        # find_active_for_consolidation): a source hidden behind a LIVE
        # gist is excluded from dedup, so a corrected fact never resolves
        # onto an ambient-hidden row (which would leave the correction
        # unreachable by proactive retrieval while the stale gist keeps
        # surfacing). A source orphaned by a dead gist self-heals back into
        # the candidate set. The gist itself (superseded_by NULL) stays a
        # candidate, so a correction can still land on the live gist.
        rows = await read_nearest(
            self.l3_pool,
            "memories",
            by={"agent_id": agent_id, "user_id": user_id},
            where=[
                Fragment("embedding IS NOT NULL", ()),
                Fragment(
                    "(superseded_by IS NULL OR NOT EXISTS ( SELECT 1 FROM memories g WHERE g.agent_id = t.agent_id AND g.memory_id = t.superseded_by))",
                    (),
                ),
                Fragment("NOT (COALESCE(tags, '[]'::jsonb) ? 'retracted')", ()),
                *clause_guards(filter_clause, self.read_filter, "t.tags"),
            ],
            nearest=Nearest("embedding", embedding_str, similarity="similarity"),
            limit=top_k,
            columns=("memory_id", "content", "type_memory"),
            records=True,
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

    async def find_same_event(
        self,
        *,
        user_id: UUID,
        agent_id: UUID,
        embedding: list[float],
        message_ids: list[UUID],
        threshold: float,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """the stored memories that came from these source messages and say nearly what ``embedding`` says.

        How a writer finds the first write of an event it is about to write
        again: the memories whose ``message_id_source`` is one of
        ``message_ids`` (this turn's message and the one before it), nearest
        first, at or above ``threshold``. Retracted memories and ones a live
        replacement stands in for are not candidates, as in
        :meth:`find_similar_for_dedup`. Permanent ones are returned and marked
        ``evergreen``: the caller decides what a permanent first write means.

        :param user_id: owning user UUID (row filter)
        :ptype user_id: UUID
        :param agent_id: partition column on memories; required
        :ptype agent_id: UUID
        :param embedding: the new memory's embedding
        :ptype embedding: list[float]
        :param message_ids: the source messages that count as the same turn or the one before
        :ptype message_ids: list[UUID]
        :param threshold: minimum cosine similarity
        :ptype threshold: float
        :param top_k: the most to return
        :ptype top_k: int
        :return: ``{memory_id, content, type_memory, evergreen, date_created, tags, similarity}``, nearest first
        :rtype: list[dict[str, Any]]
        """
        if self.l3_pool is None or not message_ids:
            return []
        # cache-bypass: vector-distance search, as find_similar_for_dedup.
        rows = await read_nearest(
            self.l3_pool,
            "memories",
            by={"agent_id": agent_id, "user_id": user_id, "message_id_source": AnyOf(message_ids, cast="uuid")},
            where=[
                Fragment("embedding IS NOT NULL", ()),
                Fragment(
                    "(superseded_by IS NULL OR NOT EXISTS ( SELECT 1 FROM memories g WHERE g.agent_id = t.agent_id AND g.memory_id = t.superseded_by))",
                    (),
                ),
                Fragment("NOT (COALESCE(tags, '[]'::jsonb) ? 'retracted')", ()),
                *clause_guards(filter_clause, self.read_filter, "t.tags"),
            ],
            nearest=Nearest("embedding", json.dumps(embedding), similarity="similarity"),
            limit=top_k,
            columns=("memory_id", "content", "type_memory", "evergreen", "date_created", "tags"),
            records=True,
        )
        return [
            {
                "memory_id": _as_pk_uuid(row["memory_id"]),
                "content": row["content"],
                "type_memory": row["type_memory"],
                "evergreen": bool(row["evergreen"]),
                "date_created": row["date_created"],
                "tags": row["tags"],
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
        row = await read_row(
            self.l3_pool,
            "memories",
            by={"agent_id": agent_id, "user_id": user_id, "alias": alias},
            where=[*clause_guards(filter_clause, self.read_filter, "t.tags")],
            columns=(
                "memory_id",
                "agent_id",
                "customer_id",
                "user_id",
                "conversation_id",
                "message_id_source",
                "type_memory",
                "content",
                "date_created",
                "date_updated",
                "alias",
                "evergreen",
                "tags",
                "superseded_by",
                Selected(
                    "(SELECT g.date_created FROM memories g WHERE g.agent_id = t.agent_id AND g.memory_id = t.superseded_by)",
                    "superseded_at",
                ),
                Selected(
                    "(SELECT c.consolidated_memory_id FROM memory_consolidations c WHERE c.agent_id = t.agent_id AND c.source_memory_id = t.memory_id AND t.superseded_by IS NULL ORDER BY c.date_created DESC LIMIT 1)",
                    "corrected_by",
                ),
            ),
            records=True,
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
        salience_ambient_floor: float = 0.0,
        fts_min_len: int = 3,
        fts_max_len: int = 500,
        date_after: datetime | None = None,
        date_before: datetime | None = None,
        scope: RetrievalScope | None = None,
        recent_cap: int | None = None,
        recent_window_hours: float = 168.0,
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
        :param similarity_threshold: floor on semantic similarity for admission
        :ptype similarity_threshold: float
        :param recency_half_life_hours: exponential decay half-life
        :ptype recency_half_life_hours: float
        :param signal_weights: mapping ``{"semantic", "keyword",
            "recency"}`` to weights
        :ptype signal_weights: dict[str, float]
        :param salience_ambient_floor: dormancy cutoff (v024). Ambient
            retrieval excludes rows below this salience AND any row with a
            non-null ``superseded_by`` (replaced by a consolidation gist).
            Default 0.0 admits everything; the direct-recall paths
            (``search_by_ids`` / ``find_by_alias``) always bypass this floor
            so a dormant memory stays reachable by id / alias.
        :ptype salience_ambient_floor: float
        :param fts_min_len: minimum query length for FTS activation
        :ptype fts_min_len: int
        :param fts_max_len: truncation length for FTS queries
        :ptype fts_max_len: int
        :param scope: optional eligibility predicate applied inside BOTH
            arms' candidate scans (:class:`RetrievalScope`); ``None`` is
            today's behaviour, an empty allow-list or tag tuple returns
            ``[]`` (fail-closed, never fail-open)
        :ptype scope: RetrievalScope | None
        :param recent_cap: at most this many of the returned memories were
            written inside ``recent_window_hours``; which ones is decided on
            semantic and keyword evidence, never on recency
            (:func:`_cap_recent`). ``None`` is no cap
        :ptype recent_cap: int | None
        :param recent_window_hours: how old a memory may be and still count
            against ``recent_cap``
        :ptype recent_window_hours: float
        :return: ranked candidate list, top_k entries
        :rtype: list[dict[str, Any]]
        """
        if self.l3_pool is None or scope_matches_nothing(scope):
            return []
        embedding_str = json.dumps(embedding)
        # v0.7.5 date filter: ``date_created >= / <=`` beside the owner on both the vector and the FTS read, each bound
        # on its own so either fires alone; then the retrieval scope (fail-closed above) and the host's read filter
        narrowing = (
            *_date_guards("t.date_created", date_after, date_before),
            *_scope_guards(scope),
            *((Fragment(_EVERY_TAG, (list(scope.tags_all),)),) if scope is not None and scope.tags_all else ()),
            *clause_guards(filter_clause, self.read_filter, "t.tags"),
        )
        salience = Fragment(f"{self._salience_read()} >= $1", (salience_ambient_floor,))

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
        # v024 ambient filter: exclude superseded rows (replaced by a
        # consolidation gist) and rows below the salience floor; default
        # 0.0 makes ``salience >= 0`` a no-op. Supersession is SELF-HEALING
        # (matches find_active_for_consolidation): a source is hidden only
        # while its gist still LIVES -- if the gist was hard-deleted, the
        # source's dangling superseded_by no longer suppresses it here, so its
        # knowledge returns to ambient retrieval instead of being lost
        # (superseded_by is a soft ref with no FK, so orphans do occur).
        def _nearest(older_than: datetime | None = None) -> Awaitable[list[Any]]:
            if older_than is None:
                return read_nearest(
                    self.l3_pool,
                    "memories",
                    by={"agent_id": agent_id, "customer_id": customer_id, "user_id": user_id},
                    where=[
                        Fragment("t.embedding IS NOT NULL", ()),
                        *narrowing,
                        Fragment(
                            "(t.superseded_by IS NULL OR NOT EXISTS (SELECT 1 FROM memories g WHERE g.agent_id = t.agent_id AND g.memory_id = t.superseded_by))",
                            (),
                        ),
                        Fragment("NOT (COALESCE(t.tags, '[]'::jsonb) ? 'retracted')", ()),
                        salience,
                    ],
                    nearest=Nearest("embedding", embedding_str, similarity="similarity"),
                    limit=candidate_limit,
                    columns=_HYBRID_MEMORY_COLUMNS,
                )
            # ``older_than`` narrows the read to the history (see the history pool below): the user's memories older
            # than the window, pinned by agent and user so the read is the one ix_memories_agent_user_date names, the
            # date bound read off it, the customer checked after. b2bc83998 fenced the customer out (an OFFSET 0
            # subquery): with it beside the pins, YugabyteDB read the owner's every memory by
            # idx_memories_agent_customer_user. Here it is ``(...) IS TRUE``, the same rows (a WHERE keeps neither a NULL)
            # as a predicate no index condition takes, as gin_filter keeps a predicate off an index
            # (test_index_plans::test_memories_user_date holds the plan on both engines).
            return read_nearest(
                self.l3_pool,
                "memories",
                by={"agent_id": agent_id, "user_id": user_id},
                where=[
                    Fragment("(t.customer_id = $1) IS TRUE AND t.date_created < $2", (customer_id, older_than)),
                    Fragment("t.embedding IS NOT NULL", ()),
                    *narrowing,
                    Fragment(
                        "(t.superseded_by IS NULL OR NOT EXISTS (SELECT 1 FROM memories g WHERE g.agent_id = t.agent_id AND g.memory_id = t.superseded_by))",
                        (),
                    ),
                    Fragment("NOT (COALESCE(t.tags, '[]'::jsonb) ? 'retracted')", ()),
                    salience,
                ],
                nearest=Nearest("embedding", embedding_str, similarity="similarity"),
                limit=candidate_limit,
                columns=_HYBRID_MEMORY_COLUMNS,
            )

        vec_coro = _nearest()
        # The history pool. The candidates are the nearest ``candidate_limit`` of
        # everything, and a burst of recent memories on one subject fills that pool
        # as surely as it fills the list: capped, the list then came back short,
        # with the places the cap freed left empty. So with a cap the nearest
        # ``candidate_limit`` memories older than the window are candidates too.
        history_coro = (
            _nearest(datetime.now(UTC) - timedelta(hours=recent_window_hours)) if recent_cap is not None else None
        )

        fts_text = _build_fts_text(user_text, fts_min_len, fts_max_len)
        if fts_text:
            # cache-bypass: FTS rank query is not primary-key-
            # addressable. See :meth:`hybrid_search` docstring.
            fts_coro = read_nearest(
                self.l3_pool,
                "memories",
                by={"agent_id": agent_id, "customer_id": customer_id, "user_id": user_id},
                where=[
                    Fragment("t.embedding IS NOT NULL", ()),
                    Fragment(_FTS_MATCH, (fts_text,)),
                    *narrowing,
                    Fragment(
                        "(t.superseded_by IS NULL OR NOT EXISTS (SELECT 1 FROM memories g WHERE g.agent_id = t.agent_id AND g.memory_id = t.superseded_by))",
                        (),
                    ),
                    Fragment("NOT (COALESCE(t.tags, '[]'::jsonb) ? 'retracted')", ()),
                    salience,
                ],
                nearest=TextRank("search_vector", fts_text),
                limit=candidate_limit,
                columns=_HYBRID_MEMORY_COLUMNS,
            )
            vec_rows, fts_rows = await asyncio.gather(vec_coro, fts_coro)
        else:
            vec_rows = await vec_coro
            fts_rows = []
        if history_coro is not None:
            vec_rows = [*vec_rows, *await history_coro]

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
            merged[mid][_FTS_MATCHED_KEY] = True

        candidates = list(merged.values())
        if not candidates:
            return []

        _normalize_scores(candidates, "fts_rank")

        for c in candidates:
            recency = _recency_weight(c["date_created"], recency_half_life_hours)
            c["recency"] = round(recency, 4)
            evidence = signal_weights["semantic"] * c["similarity"] + signal_weights["keyword"] * c["fts_rank"]
            c[_EVIDENCE_KEY] = evidence
            c["hybrid_score"] = round(evidence + signal_weights["recency"] * recency, 4)

        ranked = _admit_and_rank(candidates, similarity_threshold=similarity_threshold)
        return _cap_recent(ranked, top_k=top_k, recent_cap=recent_cap, recent_window_hours=recent_window_hours)

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
        rows = await read_by(
            self.l3_pool,
            "memories",
            by={"agent_id": agent_id, "memory_id": AnyOf(memory_ids, cast="uuid"), "user_id": user_id},
            where=[*clause_guards(filter_clause, self.read_filter, "t.tags")],
            columns=(
                "memory_id",
                "type_memory",
                "content",
                "date_created",
                "alias",
                "evergreen",
                "tags",
                "superseded_by",
                Selected(
                    "(SELECT g.date_created FROM memories g WHERE g.agent_id = t.agent_id AND g.memory_id = t.superseded_by)",
                    "superseded_at",
                ),
                Selected(
                    "(SELECT c.consolidated_memory_id FROM memory_consolidations c WHERE c.agent_id = t.agent_id AND c.source_memory_id = t.memory_id AND t.superseded_by IS NULL ORDER BY c.date_created DESC LIMIT 1)",
                    "corrected_by",
                ),
            ),
            records=True,
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
        scope: RetrievalScope | None = None,
        include_retracted: bool = False,
    ) -> list[dict[str, Any]]:
        """semantic (vector-only) search for the add/search memory tools.

        A retracted memory is not found unless ``include_retracted`` is set:
        it no longer holds, and only a read by its id or a listing of the
        retired ones opens it. Each row carries what a line is marked from
        (``alias``, ``evergreen``, ``tags``, ``superseded_by``,
        ``superseded_at``, ``corrected_by``).

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
        :param scope: optional eligibility predicate applied inside the
            candidate scan (:class:`RetrievalScope`); an empty allow-list
            or tag tuple returns ``[]``, fail-closed
        :ptype scope: RetrievalScope | None
        :return: list of row dicts with ``similarity`` field
        :rtype: list[dict[str, Any]]
        """
        if self.l3_pool is None or scope_matches_nothing(scope):
            return []
        # the narrowing after the read's key (agent_id, user_id): a type, a date window, the retrieval scope, embedded
        # rows only, unretracted unless asked, and the host's read filter
        narrowing = [
            *((Fragment("t.type_memory = $1", (type_filter,)),) if type_filter else ()),
            *_date_guards("t.date_created", date_after, date_before),
            *_scope_guards(scope),
            *((Fragment(_EVERY_TAG, (list(scope.tags_all),)),) if scope is not None and scope.tags_all else ()),
            Fragment("t.embedding IS NOT NULL", ()),
            *(() if include_retracted else (Fragment("NOT (COALESCE(t.tags, '[]'::jsonb) ? 'retracted')", ()),)),
            *clause_guards(filter_clause, self.read_filter, "t.tags"),
        ]
        # ``embedding IS NOT NULL`` guard: ``NULL <=> vector`` is NULL,
        # which trips ``float(similarity)`` downstream.
        # DELIBERATE-RECALL by design (§3): this is the agent-invoked
        # ``memory_search`` tool, NOT the proactive ambient path. It
        # therefore does NOT apply the salience floor OR the supersession
        # filter that ``hybrid_search`` (the during-a-turn grounding path)
        # applies -- an explicit search surfaces dormant + superseded rows
        # too (the source is "dormant, not gone"), and each row says what
        # replaced it so the searching agent can tell the two apart. Only
        # proactive grounding hides them for a clean turn; direct id/alias
        # recall likewise bypasses. (Contrast find_similar_for_dedup, which
        # DOES filter, because it drives write-path UPDATE targeting, not
        # display.)
        # cache-bypass: vector-distance ordered lookup; see
        # :meth:`hybrid_search` docstring.
        rows = await read_nearest(
            self.l3_pool,
            "memories",
            by={"agent_id": agent_id, "user_id": user_id},
            where=narrowing,
            nearest=Nearest("embedding", json.dumps(embedding), similarity="similarity"),
            limit=max_results,
            columns=(
                "memory_id",
                "type_memory",
                "content",
                "date_created",
                "alias",
                "evergreen",
                "tags",
                "superseded_by",
                Selected(
                    "(SELECT g.date_created FROM memories g WHERE g.agent_id = t.agent_id AND g.memory_id = t.superseded_by)",
                    "superseded_at",
                ),
                Selected(
                    "(SELECT c.consolidated_memory_id FROM memory_consolidations c WHERE c.agent_id = t.agent_id AND c.source_memory_id = t.memory_id AND t.superseded_by IS NULL ORDER BY c.date_created DESC LIMIT 1)",
                    "corrected_by",
                ),
            ),
        )
        return [
            {
                "memory_id": str(row["memory_id"]),
                "type": row["type_memory"],
                "content": row["content"],
                "date_created": row["date_created"],
                "similarity": float(row["similarity"]),
                **_marks(row),
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
        scope: RetrievalScope | None = None,
        include_retracted: bool = False,
    ) -> list[dict[str, Any]]:
        """FTS keyword search for the add/search memory tools.

        A retracted memory is not found unless ``include_retracted`` is set,
        and each row carries what a line is marked from, as
        :meth:`search_by_semantic` does.

        complements :meth:`search_by_semantic` -- run both in parallel
        and merge on ``memory_id``. ``agent_id`` is the partition
        column on memories and is required.

        v0.7.5: ``date_after`` / ``date_before`` (inclusive) narrow by
        ``date_created`` -- mirrors the equivalent filter on
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
        :param scope: optional eligibility predicate applied inside the
            candidate scan (:class:`RetrievalScope`); an empty allow-list
            or tag tuple returns ``[]``, fail-closed
        :ptype scope: RetrievalScope | None
        :return: list of row dicts
        :rtype: list[dict[str, Any]]
        """
        if self.l3_pool is None or scope_matches_nothing(scope):
            return []
        # the narrowing after the read's key (agent_id, user_id): the keyword match, a type, a date window, the
        # retrieval scope, unretracted unless asked, and the host's read filter
        narrowing = [
            Fragment(_FTS_MATCH, (fts_text,)),
            *((Fragment("t.type_memory = $1", (type_filter,)),) if type_filter else ()),
            *_date_guards("t.date_created", date_after, date_before),
            *_scope_guards(scope),
            *((Fragment(_EVERY_TAG, (list(scope.tags_all),)),) if scope is not None and scope.tags_all else ()),
            *(() if include_retracted else (Fragment("NOT (COALESCE(t.tags, '[]'::jsonb) ? 'retracted')", ()),)),
            *clause_guards(filter_clause, self.read_filter, "t.tags"),
        ]
        # DELIBERATE-RECALL by design (§3), mirroring search_by_semantic:
        # the agent-invoked memory_search keyword leg applies NO salience
        # floor and NO supersession filter (unlike ambient hybrid_search) --
        # an explicit search surfaces dormant + superseded rows.
        # cache-bypass: FTS rank query is not primary-key-addressable;
        # see :meth:`hybrid_search` docstring.
        rows = await read_nearest(
            self.l3_pool,
            "memories",
            by={"agent_id": agent_id, "user_id": user_id},
            where=narrowing,
            nearest=TextRank("search_vector", fts_text),
            limit=max_results,
            columns=(
                "memory_id",
                "type_memory",
                "content",
                "date_created",
                "alias",
                "evergreen",
                "tags",
                "superseded_by",
                Selected(
                    "(SELECT g.date_created FROM memories g WHERE g.agent_id = t.agent_id AND g.memory_id = t.superseded_by)",
                    "superseded_at",
                ),
                Selected(
                    "(SELECT c.consolidated_memory_id FROM memory_consolidations c WHERE c.agent_id = t.agent_id AND c.source_memory_id = t.memory_id AND t.superseded_by IS NULL ORDER BY c.date_created DESC LIMIT 1)",
                    "corrected_by",
                ),
            ),
        )
        return [
            {
                "memory_id": str(row["memory_id"]),
                "type": row["type_memory"],
                "content": row["content"],
                "date_created": row["date_created"],
                **_marks(row),
            }
            for row in rows
        ]

    async def find_for_recall(
        self,
        *,
        memory_id: UUID,
        user_id: UUID,
        agent_id: UUID,
    ) -> dict[str, Any] | None:
        """fetch one memory with what its header says, for a read by id.

        Scoped by ``agent_id`` (partition predicate) and ``user_id`` (row
        owner), like :meth:`fetch_content_for_recall`. A retired or replaced
        memory is returned: a read by id opens anything that is kept.

        :param memory_id: memory primary-key value
        :ptype memory_id: UUID
        :param user_id: owning user UUID (row filter)
        :ptype user_id: UUID
        :param agent_id: partition column on memories; required
        :ptype agent_id: UUID
        :return: ``memory_id``, ``type_memory``, ``content``, ``date_created``
            and the mark columns, or ``None`` when there is no such memory
        :rtype: dict[str, Any] | None
        """
        if self.l3_pool is None:
            return None
        # cache-bypass: the ``user_id`` guard is a security control the L1
        # row cache cannot enforce, and the marks are computed columns.
        row = await read_row(
            self.l3_pool,
            "memories",
            by={"agent_id": agent_id, "memory_id": memory_id, "user_id": user_id},
            where=[*clause_guards(filter_clause, self.read_filter, "t.tags")],
            columns=(
                "memory_id",
                "type_memory",
                "content",
                "date_created",
                "alias",
                "evergreen",
                "tags",
                "superseded_by",
                Selected(
                    "(SELECT g.date_created FROM memories g WHERE g.agent_id = t.agent_id AND g.memory_id = t.superseded_by)",
                    "superseded_at",
                ),
                Selected(
                    "(SELECT c.consolidated_memory_id FROM memory_consolidations c WHERE c.agent_id = t.agent_id AND c.source_memory_id = t.memory_id AND t.superseded_by IS NULL ORDER BY c.date_created DESC LIMIT 1)",
                    "corrected_by",
                ),
            ),
            records=True,
        )
        return dict(row) if row else None

    async def list_for_user(
        self,
        *,
        user_id: UUID,
        agent_id: UUID,
        date_after: datetime | None = None,
        date_before: datetime | None = None,
        type_filter: str | None = None,
        permanent: bool | None = None,
        has_alias: bool | None = None,
        state: str = "current",
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """list one person's memories newest first, with no query, and how many match in all.

        ``state`` picks which memories: ``current`` (neither retired nor
        replaced by a memory that exists), ``retired``, ``replaced`` or
        ``all``. Every other filter narrows that set; ``None`` leaves a filter
        off.

        :param user_id: owning user UUID (row filter)
        :ptype user_id: UUID
        :param agent_id: partition column on memories; required
        :ptype agent_id: UUID
        :param date_after: inclusive lower bound on ``date_created``
        :ptype date_after: datetime | None
        :param date_before: inclusive upper bound on ``date_created``
        :ptype date_before: datetime | None
        :param type_filter: ``type_memory`` equality filter
        :ptype type_filter: str | None
        :param permanent: only permanent memories, or only those that are not
        :ptype permanent: bool | None
        :param has_alias: only named memories, or only unnamed ones
        :ptype has_alias: bool | None
        :param state: ``current``, ``retired``, ``replaced`` or ``all``
        :ptype state: str
        :param limit: most rows to return
        :ptype limit: int
        :param offset: rows to skip, newest first
        :ptype offset: int
        :return: the page of rows (``memory_id``, ``type_memory``, ``content``,
            ``date_created`` and the mark columns) and the count of every match
        :rtype: tuple[list[dict[str, Any]], int]
        :raises ValueError: when ``state`` is not one of the four
        """
        if state not in {"current", "retired", "replaced", "all"}:
            raise ValueError(f"list_for_user: unknown state {state!r}")
        if self.l3_pool is None:
            return [], 0
        replaced = (
            "(t.superseded_by IS NOT NULL AND EXISTS (SELECT 1 FROM memories g "
            "WHERE g.agent_id = t.agent_id AND g.memory_id = t.superseded_by))"
        )
        retracted = "(COALESCE(t.tags, '[]'::jsonb) ? 'retracted')"
        states: dict[str, tuple[Fragment, ...]] = {
            "current": (Fragment(f"NOT {retracted}", ()), Fragment(f"NOT {replaced}", ())),
            "retired": (Fragment(retracted, ()),),
            "replaced": (Fragment(replaced, ()),),
            "all": (),
        }
        # the narrowing after the read's key (agent_id, user_id), the same for the count and the page
        narrowing = [
            *_date_guards("t.date_created", date_after, date_before),
            *((Fragment("t.type_memory = $1", (type_filter,)),) if type_filter else ()),
            *(() if permanent is None else (Fragment("t.evergreen" if permanent else "NOT t.evergreen", ()),)),
            *(() if has_alias is None else (Fragment(f"t.alias IS {'NOT ' if has_alias else ''}NULL", ()),)),
            *states[state],
            *clause_guards(filter_clause, self.read_filter, "t.tags"),
        ]
        owner = {"agent_id": agent_id, "user_id": user_id}
        # cache-bypass: a filtered scan of one person's rows is not primary-key
        # addressable; ix_memories_agent_user_date backs the order.
        (counted,) = await aggregate_keyed(
            self.l3_pool, "memories", aggregates={"n": Count()}, by=owner, where=narrowing
        )
        # the page ``offset`` rows in, newest first: the first ``offset + limit`` rows by the key's order, the first
        # ``offset`` dropped -- the rows a ``LIMIT n OFFSET k`` read and discarded, as it did
        page = await keyset_page(
            self.l3_pool,
            "memories",
            columns=(
                "memory_id",
                "type_memory",
                "content",
                "date_created",
                "alias",
                "evergreen",
                "tags",
                "superseded_by",
                Selected(
                    "(SELECT g.date_created FROM memories g WHERE g.agent_id = t.agent_id AND g.memory_id = t.superseded_by)",
                    "superseded_at",
                ),
                Selected(
                    "(SELECT c.consolidated_memory_id FROM memory_consolidations c WHERE c.agent_id = t.agent_id AND c.source_memory_id = t.memory_id AND t.superseded_by IS NULL ORDER BY c.date_created DESC LIMIT 1)",
                    "corrected_by",
                ),
            ),
            by=owner,
            order=[Sort("date_created", descending=True), Sort("memory_id", "uuid", descending=True)],
            limit=offset + limit,
            where=narrowing,
        )
        return page.rows[offset:], int(counted["n"] or 0)

    async def mark_retracted(
        self,
        agent_id: UUID,
        *,
        memory_id: UUID,
        reason: str,
        allow_permanent: bool = False,
    ) -> bool:
        """tag one memory retracted, with why, and drop its salience to the bottom, in one statement.

        The check and the write are one ``UPDATE``: a memory that is already
        retracted, or permanent without ``allow_permanent``, matches no row and
        nothing is written. What its salience was is kept as a tag, so
        :meth:`clear_retracted` puts it back. A permanent memory's salience is
        left alone: it never decays, and only the tag takes it out of recall.

        :param agent_id: partition column; required
        :ptype agent_id: UUID
        :param memory_id: the memory
        :ptype memory_id: UUID
        :param reason: why it no longer holds; the first 200 characters are kept
        :ptype reason: str
        :param allow_permanent: retract a permanent memory too
        :ptype allow_permanent: bool
        :return: ``True`` when the memory was retracted by this call
        :rtype: bool
        """
        if self.l3_pool is None:
            return False
        # tags and salience change together, so this raw pass is the writer and
        # the row is evicted for a later get() to read fresh.
        async with self.bypassing_write((agent_id, memory_id)) as write:
            written = await update_keyed(
                self.l3_pool,
                "memories",
                values={
                    "tags": Fragment(
                        "COALESCE(tags, '[]'::jsonb) || jsonb_build_array("
                        f"'retracted', $1::text, '{_RETRACTED_SALIENCE}' || salience::text)",
                        (f"{_RETRACTED_BECAUSE}{reason[:200]}",),
                    ),
                    "salience": Fragment("CASE WHEN evergreen THEN salience ELSE 0.0 END", ()),
                    "date_updated": Fragment("now()", ()),
                },
                by={"agent_id": agent_id, "memory_id": memory_id},
                where=[
                    Fragment(_NOT_RETRACTED, ()),
                    Fragment("($1::boolean OR NOT evergreen)", (allow_permanent,)),
                ],
                returning=("memory_id",),
            )
            changed = written.rows[0]["memory_id"] if written.rows else None
            if changed is None:
                write.unchanged()
        return changed is not None

    async def clear_retracted(
        self,
        agent_id: UUID,
        *,
        memory_id: UUID,
        default_salience: float = 0.5,
    ) -> bool:
        """take the retracted tags off one memory and put its salience back, in one statement.

        The undo of :meth:`mark_retracted`. Salience returns to what the
        retraction recorded, or to ``default_salience`` for a memory retracted
        before that was recorded. A memory that is not retracted matches no row.

        :param agent_id: partition column; required
        :ptype agent_id: UUID
        :param memory_id: the memory
        :ptype memory_id: UUID
        :param default_salience: the salience to restore when none was recorded
        :ptype default_salience: float
        :return: ``True`` when the memory was restored by this call
        :rtype: bool
        """
        if self.l3_pool is None:
            return False
        async with self.bypassing_write((agent_id, memory_id)) as write:
            written = await update_keyed(
                self.l3_pool,
                "memories",
                values={
                    "salience": Fragment(
                        "CASE WHEN evergreen THEN salience ELSE GREATEST(0.0, LEAST(1.0, COALESCE("
                        f"(SELECT substring(t FROM {len(_RETRACTED_SALIENCE) + 1})::numeric "
                        "FROM jsonb_array_elements_text(tags) AS t "
                        f"WHERE starts_with(t, '{_RETRACTED_SALIENCE}') LIMIT 1), "
                        "$1))) END",
                        (default_salience,),
                    ),
                    "tags": Fragment(
                        "(SELECT COALESCE(jsonb_agg(t), '[]'::jsonb) FROM jsonb_array_elements_text(tags) AS t "
                        f"WHERE t <> 'retracted' AND NOT starts_with(t, '{_RETRACTED_BECAUSE}') "
                        f"AND NOT starts_with(t, '{_RETRACTED_SALIENCE}'))",
                        (),
                    ),
                    "date_updated": Fragment("now()", ()),
                },
                by={"agent_id": agent_id, "memory_id": memory_id},
                where=[Fragment(f"NOT ({_NOT_RETRACTED})", ())],
                returning=("memory_id",),
            )
            changed = written.rows[0]["memory_id"] if written.rows else None
            if changed is None:
                write.unchanged()
        return changed is not None

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
        row = await read_row(
            self.l3_pool,
            "memories",
            by={"agent_id": agent_id, "memory_id": memory_id, "user_id": user_id},
            columns=["content"],
            where=filter_guards(self.read_filter, "t.tags"),
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
        owners: dict[str, Any] = {"agent_id": AnyOf(list(agent_ids), cast="uuid"), "user_id": user_id}
        if customer_id is not None:
            owners["customer_id"] = customer_id
        rows = await read_by(
            self.l3_pool,
            "memories",
            by=owners,
            columns=_MEMORIES_SELECTED,
            where=filter_guards(self.read_filter, "t.tags"),
            order=_NEWEST_FIRST,
        )
        entities: list[MemoryEntity] = []
        for row in rows:
            data = dict(row)
            entity = self.entity_class(data, is_new=False, collection=self)
            entity.original_date_updated = data.get("date_updated")
            entities.append(entity)
        return entities

    async def set_salience(
        self,
        agent_id: UUID,
        *,
        memory_ids: list[UUID],
        salience: float,
    ) -> None:
        """set salience outright on non-evergreen memories: a retraction sets it below the ambient floor.

        Like :meth:`bump_salience`, the raw pass is salience's only writer and
        each row is evicted so a later ``get()`` reads it fresh. ``evergreen``
        rows are never touched.

        :param agent_id: partition column; required
        :ptype agent_id: UUID
        :param memory_ids: the memories
        :ptype memory_ids: list[UUID]
        :param salience: the value, clamped to 0..1
        :ptype salience: float
        :return: nothing
        :rtype: None
        """
        if self.l3_pool is None or not memory_ids:
            return None
        async with self.bypassing_write(*[(agent_id, memory_id) for memory_id in memory_ids]):
            await update_keyed(
                self.l3_pool,
                "memories",
                values={"salience": Fragment("GREATEST(0.0, LEAST(1.0, $1))", (salience,))},
                by={"agent_id": agent_id, "memory_id": AnyOf(memory_ids, cast="uuid"), "evergreen": False},
            )
        return None

    def _salience_read(self) -> str:
        """what the ambient reads compare to their salience floor: the stored salience, or with decay on read the
        effective salience (an ``evergreen`` row's stored one).

        :return: the SQL expression
        :rtype: str
        """
        decay = type(self).salience_decay
        return "salience" if decay is None else decay.effective_sql()

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

    async def bump_salience(
        self,
        memory_ids: list[UUID],
        *,
        agent_id: UUID,
        access_bump: float,
    ) -> None:
        """reinforce salience for memories surfaced by ambient retrieval (v024).

        The other half of the decay cycle: on ambient retrieval, the
        surfaced memories get ``salience = LEAST(1.0, salience +
        access_bump)`` and a fresh ``last_accessed`` stamp, so used
        memories climb back up while only the neglected sink.
        ``evergreen`` rows are skipped (pinned, never bumped). Direct
        recall (by id / alias) does NOT bump -- reinforcement tracks
        proactive surfacing, not explicit lookup.

        Partition-scoped by ``agent_id`` (the surfaced ids all belong to
        the retrieving agent's partition), so no cross-partition sweep.

        :param memory_ids: primary-key ids surfaced this retrieval
        :ptype memory_ids: list[UUID]
        :param agent_id: partition column; required
        :ptype agent_id: UUID
        :param access_bump: increment added to salience (clamped to 1.0)
        :ptype access_bump: float
        :return: nothing
        :rtype: None
        """
        if self.l3_pool is None or not memory_ids:
            return None
        # bulk reinforcement UPDATE keyed by a memory_id set; salience /
        # last_accessed are immutable to the entity-UPDATE generator, so
        # this raw pass is the only writer. Each bumped row is evicted so a
        # subsequent get() re-reads the fresh salience from L3 (a bounded set --
        # the ids surfaced this retrieval).
        async with self.bypassing_write(*[(agent_id, memory_id) for memory_id in memory_ids]):
            await update_keyed(
                self.l3_pool,
                "memories",
                values={**self._reinforced(access_bump), "last_accessed": Fragment("now()", ())},
                by={"agent_id": agent_id, "memory_id": AnyOf(memory_ids, cast="uuid"), "evergreen": False},
            )
        return None

    async def find_active_for_consolidation(
        self,
        agent_id: UUID,
        *,
        customer_id: UUID | None = None,
        user_id: UUID | None = None,
    ) -> list[dict[str, Any]]:
        """load the consolidation candidate set for one scope grain (A5).

        Dream consolidation clusters memories WITHIN a pinned scope grain
        and the gist inherits that grain's identity, so the scope filter
        is EXACT -- a ``None`` grain matches ``IS NULL`` (not "any"). That
        is the isolation boundary: a user-scoped run
        (``customer_id``+``user_id`` set) sees only that user's rows; an
        agent-scoped run (both ``None``) sees only agent-scoped rows. This
        differs deliberately from :meth:`find_by_scope`, which drops the
        filter on a ``None`` argument.

        Excludes ``evergreen`` rows (pinned, never consolidated) and rows
        with no ``embedding`` (clustering needs a vector). Excludes
        already-superseded rows -- BUT only when their gist still exists:
        a source whose ``superseded_by`` points at a memory that no longer
        exists (e.g. the gist was bulk-deleted with its anchor
        conversation) becomes eligible again, so a fresh gist regenerates
        on the next run instead of the orphaned sources staying dormant
        forever. Supersession is thus defined as "pointed at a LIVE gist",
        which keeps the soft-ref self-healing without an FK.

        :param agent_id: partition column (required)
        :ptype agent_id: UUID
        :param customer_id: customer grain; ``None`` matches ``IS NULL``
        :ptype customer_id: UUID | None
        :param user_id: user grain; ``None`` matches ``IS NULL``
        :ptype user_id: UUID | None
        :return: candidate dicts with decoded ``embedding`` (``list[float]``)
            plus ``memory_id``, ``customer_id``, ``user_id``,
            ``conversation_id``, ``type_memory``, ``content``,
            ``date_created``
        :rtype: list[dict[str, Any]]
        """
        if self.l3_pool is None:
            return []
        # exact scope match, including the NULL grain (see docstring): read_by pins a None argument as IS NULL
        # rather than dropping the filter, one statement whatever the grain (it was four hand-written ones, one per
        # grain); the owner's index (agent_id, customer_id, user_id) serves IS NULL as an equality.
        # cache-bypass: a scope-scanned candidate load is not primary-key
        # addressable, so the L1 row cache cannot serve it.
        # embedding::text so asyncpg decodes it on the no-codec L3 pool
        # (same cast as _MEMORIES_SELECTED); decode_vector parses the
        # bracketed text back to list[float] below. The self-healing
        # supersession filter is a NOT EXISTS against a live gist row.
        rows = await read_by(
            self.l3_pool,
            "memories",
            by={"agent_id": agent_id, "customer_id": customer_id, "user_id": user_id},
            columns=_CANDIDATE_COLUMNS,
            where=_CANDIDATE_GUARDS + clause_guards(filter_clause, self.read_filter, "t.tags"),
            order=_OLDEST_FIRST,
        )
        candidates: list[dict[str, Any]] = []
        for row in rows:
            data = dict(row)
            data["memory_id"] = _as_pk_uuid(data["memory_id"])
            data["embedding"] = decode_vector(data["embedding"])
            candidates.append(data)
        return candidates

    async def mark_superseded(
        self,
        agent_id: UUID,
        *,
        source_memory_ids: list[UUID],
        gist_id: UUID,
    ) -> None:
        """point a set of source memories at the gist that consolidated them (A5).

        Sets ``superseded_by = gist_id`` on each source so ambient
        retrieval drops them (a gist now represents them) while direct
        recall by id / alias still finds them -- dormant, not gone, and
        non-destructive (the source's own ``salience`` / ``content`` are
        untouched). Partition-scoped by ``agent_id``.

        :param agent_id: partition column (required)
        :ptype agent_id: UUID
        :param source_memory_ids: the sources merged into the gist
        :ptype source_memory_ids: list[UUID]
        :param gist_id: the consolidation gist the sources are superseded by
        :ptype gist_id: UUID
        :return: nothing
        :rtype: None
        """
        if self.l3_pool is None or not source_memory_ids:
            return None
        # bulk supersession UPDATE keyed by a memory_id set; superseded_by
        # is immutable to the entity-UPDATE generator, so this raw pass is
        # the only writer (an entity save can't revert it). Each superseded
        # source is evicted so a subsequent get() re-reads the fresh
        # superseded_by + date_updated (the ambient-retrieval filter reads L3
        # directly, but a later entity save must see the advanced CAS fence).
        async with self.bypassing_write(*[(agent_id, source_memory_id) for source_memory_id in source_memory_ids]):
            await update_keyed(
                self.l3_pool,
                "memories",
                values={"superseded_by": gist_id, "date_updated": Fragment("now()")},
                by={"agent_id": agent_id, "memory_id": AnyOf(source_memory_ids, cast="uuid")},
            )
        return None


class MediaCollection(SchemaBackedCollection[MediaEntity]):
    """three-tier collection for :class:`MediaEntity` (table ``media``).

    the media parent record carries a category discriminator and a
    JSONB metadata blob; child rows live in
    :class:`MediaContentCollection` and :class:`MemoryChunkCollection`.
    adopted under namespace-task-01 phase 8.5b -- the v006 migration
    had no Collection before now. CRUD is generated from
    :attr:`schema` via :class:`SchemaBackedCollection`; no CAS path
    because the table has no ``date_updated`` fence column distinct
    from ``date_created``.
    """

    tenancy = TENANT_BY_SCHEMA

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
    # unified-model parent FK (3tears migration v017); v028 drops a
    # single-column ``memory_id`` key where one exists, so the composite
    # is the only key from media to memories.
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
            # the partition-aware parent key v017 builds: a media row's
            # memory lives in the same agent partition.
            SchemaForeignKey(
                ("agent_id", "memory_id"),
                "memories",
                ("agent_id", "memory_id"),
                on_delete="CASCADE",
            ),
        ),
        indexes=(
            # v036: led by the agent, so YugabyteDB looks up each chunk's file by key instead of hashing the agent's
            SchemaIndex(
                "ix_media_agent_memory",
                "agent_id",
                "memory_id",
                purpose=(
                    "packages/agent/memory/src/threetears/agent/memory/collections.py"
                    "::MemoryChunkCollection.search_by_ids; "
                    "packages/agent/memory/src/threetears/agent/memory/collections.py"
                    "::MemoryChunkCollection.find_for_recall; "
                    "packages/agent/memory/src/threetears/agent/memory/collections.py"
                    "::MemoryChunkCollection.hybrid_search; "
                    "packages/agent/memory/src/threetears/agent/memory/collections.py"
                    "::MemoryChunkCollection.search_by_semantic; "
                    "packages/agent/memory/src/threetears/agent/memory/collections.py"
                    "::MemoryChunkCollection.hybrid_search_within_memory; "
                    "packages/agent/memory/src/threetears/agent/memory/merge.py::_delete_alias_collisions"
                ),
                plan_test="packages/agent/memory/tests/integration/test_index_plans.py::test_media_memory_id",
                reason="media carries three indexes: the file a memory wraps, joined to that memory's chunks",
            ),
            SchemaIndex(
                "uq_media_cloud_connection_file",
                "cloud_connection_id",
                "cloud_file_id",
                unique=True,
                purpose="constraint",
                plan_test=(
                    "packages/agent/memory/tests/integration/test_index_plans.py::test_media_cloud_connection_file"
                ),
                reason="media carries three indexes: one media row per provider file of a cloud connection",
            ),
            # v022: one user's files under the schema's agent, for a user merge's repoint of them
            SchemaIndex(
                "idx_media_agent_user",
                "agent_id",
                "user_id",
                purpose="packages/core/src/threetears/core/collections/merge.py::repoint_user_rows",
                plan_test="packages/agent/memory/tests/integration/test_index_plans.py::test_repoint_agent_user",
                reason="media carries three indexes: one user's files under one agent, repointed by a user merge",
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


class MediaContentCollection(SchemaBackedCollection[MediaContentEntity], ReadFiltered):
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

    tenancy = TENANT_BY_SCHEMA

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
                purpose=(
                    "packages/agent/memory/src/threetears/agent/memory/collections.py"
                    "::MediaContentCollection.search_by_ids; "
                    "packages/agent/memory/src/threetears/agent/memory/merge.py::_delete_alias_collisions"
                ),
                plan_test="packages/agent/memory/tests/integration/test_index_plans.py::test_media_content_media_type",
                reason="media_content carries three indexes: the contents of the files a search names by file id",
            ),
            # v0.8.1: parity-gate enrichments relocated from upstream
            # alembic. tenancy-scope btree composite.
            SchemaIndex(
                "idx_media_content_agent_user",
                "agent_id",
                "user_id",
                purpose=(
                    "packages/agent/memory/src/threetears/agent/memory/collections.py"
                    "::MediaContentCollection.hybrid_search; "
                    "packages/agent/memory/src/threetears/agent/memory/collections.py"
                    "::MediaContentCollection.search_by_semantic; "
                    "packages/agent/memory/src/threetears/agent/memory/collections.py"
                    "::MediaContentCollection.search_by_fts; "
                    "packages/core/src/threetears/core/collections/merge.py::repoint_user_rows"
                ),
                plan_test="packages/agent/memory/tests/integration/test_index_plans.py::test_media_content_agent_user",
                reason="media_content carries three indexes: one user's file contents, every search of them",
            ),
            # no GIN over ``search_vector``: v027 dropped it. the keyword
            # predicate goes through ``gin_filter`` (ybgin refuses a
            # multi-entry scan), so the index had no reader.
            # v0.8.1: HNSW vector-similarity index with the
            # ``vector_cosine_ops`` opclass. Prod does NOT carry a
            # ``WITH`` clause here (the index was built before the
            # upstream migration started parametrising hnsw), so this
            # declaration has no ``pg_with=``.
            # KEEP (coordinator, 2026-11-15; v0.8.1, 8060d75d7): a heavy owner's file contents, nearest first, are
            # read through it on both engines; a small owner's are sorted off idx_media_content_agent_user
            # (test_vector_index_plans holds both). The YugabyteDB small-owner ybhnsw cost (~100 ms at k=10) is the
            # known follow-up at ix_memories_embedding_hnsw.
            SchemaIndex(
                "ix_media_content_embedding",
                "embedding",
                using="hnsw",
                ops={"embedding": "vector_cosine_ops"},
                purpose=(
                    "packages/agent/memory/src/threetears/agent/memory/collections.py"
                    "::MediaContentCollection.hybrid_search; "
                    "packages/agent/memory/src/threetears/agent/memory/collections.py"
                    "::MediaContentCollection.search_by_semantic"
                ),
                plan_test=(
                    "packages/agent/memory/tests/integration/test_vector_index_plans.py"
                    "::test_ix_media_content_embedding_serves_a_heavy_owners_nearest_reads"
                ),
                reason="media_content carries three indexes: the nearest file contents of an owner too large to sort",
            ),
        ),
        # v0.8.1: global uniqueness on ``content_id``. Modelled as a
        # UNIQUE CONSTRAINT (not a unique index) -- prod creates it via
        # ``ALTER TABLE ... ADD CONSTRAINT uq_media_content_content_id
        # UNIQUE`` in upstream alembic 064.
        unique_constraints=(SchemaUniqueConstraint("uq_media_content_content_id", "content_id"),),
    )

    def __init__(self, *args: Any, embedding_dim: int = DEFAULT_EMBEDDING_DIM, **kwargs: Any) -> None:
        """construct the collection with its ``embedding`` column ``embedding_dim`` wide.

        :param args: positional args forwarded to :class:`SchemaBackedCollection`
            (``registry``, ``config``, ``nats_client``, ``write_buffer``)
        :ptype args: Any
        :param embedding_dim: the ``embedding`` column's width, the deployment's (default 1024);
            a write carrying an embedding of another width is refused before any statement runs
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
        :param similarity_threshold: floor on semantic similarity for admission
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
        owner = {"agent_id": agent_id, "customer_id": customer_id, "user_id": user_id}
        allowed = _media_parent_guards(self.read_filter)

        # cache-bypass: vector-distance search joining media_content ->
        # media. not primary-key-addressable on either side; L1 row
        # cache cannot help. method on Collection preserves single
        # entry point + audit hooks. the content's file is joined by its
        # key (agent_id, media_id), the relationship inside one partition:
        # read for the ranked rows, after the limit. the foreign key
        # (agent_id, media_id) -> media ON DELETE CASCADE gives every
        # content row its file, so the LEFT join keeps exactly the rows the
        # inner JOIN kept.
        vec_coro = read_nearest(
            self.l3_pool,
            "media_content",
            by=owner,
            where=[Fragment("t.embedding IS NOT NULL", ()), *allowed],
            nearest=Nearest("embedding", json.dumps(embedding), similarity="similarity"),
            limit=candidate_limit,
            columns=_HYBRID_CONTENT_COLUMNS,
            joins=[
                Joined(
                    "media",
                    on={"media_id": "media_id"},
                    by={"agent_id": agent_id},
                    columns={"media_category": "media_category", "metadata_json": "metadata_json"},
                    left=True,
                )
            ],
        )

        fts_text = _build_fts_text(user_text, fts_min_len, fts_max_len)
        if fts_text:
            # cache-bypass: FTS rank query joining media_content ->
            # media. see vec_coro justification above.
            fts_coro = read_nearest(
                self.l3_pool,
                "media_content",
                by=owner,
                where=[Fragment("t.embedding IS NOT NULL", ()), Fragment(_FTS_MATCH, (fts_text,)), *allowed],
                nearest=TextRank("search_vector", fts_text),
                limit=candidate_limit,
                columns=_HYBRID_CONTENT_COLUMNS,
                joins=[
                    Joined(
                        "media",
                        on={"media_id": "media_id"},
                        by={"agent_id": agent_id},
                        columns={"media_category": "media_category", "metadata_json": "metadata_json"},
                        left=True,
                    )
                ],
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
            merged[cid][_FTS_MATCHED_KEY] = True

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

        return _admit_and_rank(candidates, similarity_threshold=similarity_threshold)[:top_k]

    async def search_by_ids(
        self,
        content_ids: list[UUID],
        user_id: UUID,
        *,
        agent_id: UUID,
    ) -> list[dict[str, Any]]:
        """batch lookup of media_content rows by their own id or their media's id.

        joins to the parent :class:`MediaEntity` for ``media_category``
        + ``metadata_json``, scoped to ``(agent_id, user_id)``. a search
        prints a file by its media id (the id the media tools take), so an id
        handed back is matched against both columns: a media id opens every
        content row of that file, newest first.

        :param content_ids: content ids or media ids to fetch
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
        # media. see :meth:`hybrid_search` for the detailed rationale. the
        # ids' rows are read by their keys alone, a keyed read each: by
        # content id (media_content_pkey) and by file id
        # (ix_media_content_media_type), then the user is checked on both the
        # content and its file: with the user beside the ids, YugabyteDB read
        # every content row of the user (b2bc83998). both are bound because no
        # constraint makes a content row's user its file's.
        columns = ("content_id", "content", "content_type", "media_id", Selected("t.date_created", "content_created"))
        guards = [Fragment("t.user_id = $1", (user_id,)), *_media_parent_guards(self.read_filter)]
        files = {"media_category": "media_category", "metadata_json": "metadata_json", "date_created": "date_created"}
        reads = (
            read_by_joined(
                self.l3_pool,
                "media_content",
                by={"agent_id": agent_id, "content_id": AnyOf(content_ids, cast="uuid")},
                columns=columns,
                where=guards,
                joins=[
                    Joined(
                        "media",
                        on={"media_id": "media_id"},
                        columns=files,
                        by={"agent_id": agent_id, "user_id": user_id},
                    )
                ],
            ),
            read_by_joined(
                self.l3_pool,
                "media_content",
                by={"agent_id": agent_id, "media_id": AnyOf(content_ids, cast="uuid")},
                columns=columns,
                where=guards,
                joins=[
                    Joined(
                        "media",
                        on={"media_id": "media_id"},
                        columns=files,
                        by={"agent_id": agent_id, "user_id": user_id},
                    )
                ],
            ),
        )
        # a row both reads name (its id and its file's both handed) is one row, as the ``OR`` read it once
        found = {row["content_id"]: row for answered in await asyncio.gather(*reads) for row in answered}
        # ORDER BY mc.media_id, mc.date_created DESC over the two keyed reads' rows: uuids and timestamps, compared in
        # Python as the database compares them (a uuid's bytes in order; no collation), newest first with a NULL
        # first as DESC puts it
        rows = sorted(found.values(), key=lambda row: (row["content_created"] is None, row["content_created"]))
        rows.reverse()
        rows.sort(key=lambda row: row["media_id"])
        for row in rows:
            del row["content_created"]
        return [dict(row) for row in rows]

    async def search_by_semantic(
        self,
        *,
        user_id: UUID,
        agent_id: UUID,
        embedding: list[float],
        max_results: int,
        similarity_threshold: float,
        date_after: datetime | None = None,
        date_before: datetime | None = None,
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
        :param date_after: inclusive lower bound on the file's ``date_created``
        :ptype date_after: datetime | None
        :param date_before: inclusive upper bound on the file's ``date_created``
        :ptype date_before: datetime | None
        :return: list of row dicts
        :rtype: list[dict[str, Any]]
        """
        if self.l3_pool is None:
            return []
        # cache-bypass: vector-distance search joining media_content ->
        # media. see :meth:`hybrid_search`. the file is joined by its key
        # (agent_id, media_id) after the limit, LEFT: the foreign key gives
        # every content row its file. the file's date window narrows the
        # ranked rows themselves, before the limit, as the hand-written
        # join's WHERE did: an EXISTS on the file by its key.
        rows = await read_nearest(
            self.l3_pool,
            "media_content",
            by={"agent_id": agent_id, "user_id": user_id},
            where=[
                Fragment("t.embedding IS NOT NULL", ()),
                *_file_window_guards(date_after, date_before),
                *_media_parent_guards(self.read_filter),
            ],
            nearest=Nearest("embedding", json.dumps(embedding), similarity="similarity"),
            limit=max_results,
            columns=("content_id", "content", "content_type", "media_id"),
            joins=[
                Joined(
                    "media",
                    on={"media_id": "media_id"},
                    by={"agent_id": agent_id},
                    columns={
                        "media_category": "media_category",
                        "metadata_json": "metadata_json",
                        "date_created": "date_created",
                    },
                    left=True,
                )
            ],
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
        date_after: datetime | None = None,
        date_before: datetime | None = None,
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
        :param date_after: inclusive lower bound on the file's ``date_created``
        :ptype date_after: datetime | None
        :param date_before: inclusive upper bound on the file's ``date_created``
        :ptype date_before: datetime | None
        :return: list of row dicts
        :rtype: list[dict[str, Any]]
        """
        if self.l3_pool is None:
            return []
        # cache-bypass: FTS rank query joining media_content -> media, as
        # :meth:`search_by_semantic` joins it.
        rows = await read_nearest(
            self.l3_pool,
            "media_content",
            by={"agent_id": agent_id, "user_id": user_id},
            where=[
                Fragment("t.embedding IS NOT NULL", ()),
                Fragment(_FTS_MATCH, (fts_text,)),
                *_file_window_guards(date_after, date_before),
                *_media_parent_guards(self.read_filter),
            ],
            nearest=TextRank("search_vector", fts_text),
            limit=max_results,
            columns=("content_id", "content", "content_type", "media_id"),
            joins=[
                Joined(
                    "media",
                    on={"media_id": "media_id"},
                    by={"agent_id": agent_id},
                    columns={
                        "media_category": "media_category",
                        "metadata_json": "metadata_json",
                        "date_created": "date_created",
                    },
                    left=True,
                )
            ],
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
        # cache-bypass: by-ID fetch scoped by (agent_id, user_id) --
        # both are security predicates the L1 cache cannot enforce.
        row = await read_row(
            self.l3_pool,
            "media_content",
            by={"agent_id": agent_id, "content_id": content_id, "user_id": user_id},
            columns=["content"],
            where=_media_parent_guards(self.read_filter),
        )
        if row is None:
            return None
        result: str = row["content"]
        return result


class MemoryChunkCollection(SchemaBackedCollection[MemoryChunkEntity], ReadFiltered):
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

    tenancy = TENANT_BY_SCHEMA

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
                purpose=(
                    "packages/agent/memory/src/threetears/agent/memory/collections.py"
                    "::MemoryChunkCollection.find_by_chunk_indexes; "
                    "packages/agent/memory/src/threetears/agent/memory/collections.py"
                    "::MemoryChunkCollection.count_by_memory_id; "
                    "packages/agent/memory/src/threetears/agent/memory/collections.py"
                    "::MemoryChunkCollection.find_for_recall; "
                    "packages/agent/memory/src/threetears/agent/memory/collections.py"
                    "::MemoryChunkCollection.hybrid_search_within_memory; "
                    "packages/agent/memory/src/threetears/agent/memory/collections.py"
                    "::MemoryChunkCollection.find_by_conversation_id; "
                    "packages/agent/memory/src/threetears/agent/memory/merge.py::_delete_alias_collisions"
                ),
                plan_test="packages/agent/memory/tests/integration/test_index_plans.py::test_memory_chunks_memory",
                reason="memory_chunks carries four indexes: one memory's chunks by their index, and their count",
            ),
            # v015: cursor paging within one memory (``find_by_memory_id``
            # orders by chunk_id).
            SchemaIndex(
                "idx_chunks_memory_id_chunk_id",
                "memory_id",
                "chunk_id",
                where="memory_id IS NOT NULL",
                purpose=(
                    "packages/agent/memory/src/threetears/agent/memory/collections.py"
                    "::MemoryChunkCollection.find_by_memory_id"
                ),
                plan_test=(
                    "packages/agent/memory/tests/integration/test_index_plans.py::test_chunks_memory_id_chunk_id"
                ),
                reason="memory_chunks carries four indexes: one memory's chunks paged in chunk order",
            ),
            # v0.8.1: tenancy-scope btree composite.
            SchemaIndex(
                "idx_memory_chunks_agent_user",
                "agent_id",
                "user_id",
                purpose=(
                    "packages/agent/memory/src/threetears/agent/memory/collections.py"
                    "::MemoryChunkCollection.hybrid_search; "
                    "packages/agent/memory/src/threetears/agent/memory/collections.py"
                    "::MemoryChunkCollection.search_by_semantic; "
                    "packages/core/src/threetears/core/collections/merge.py::repoint_user_rows"
                ),
                plan_test="packages/agent/memory/tests/integration/test_index_plans.py::test_memory_chunks_agent_user",
                reason="memory_chunks carries four indexes: one user's chunks, every search of them",
            ),
            # no GIN over ``search_vector``: v027 dropped it. the keyword
            # predicate goes through ``gin_filter`` (ybgin refuses a
            # multi-entry scan), so the index had no reader.
            # v0.8.1: HNSW vector-similarity index with
            # ``vector_cosine_ops``. Prod does NOT carry a ``WITH``
            # clause here (mirror of ``ix_media_content_embedding``).
            # KEEP (coordinator, 2026-11-15; v007 idx_chunks_embedding, renamed in 8060d75d7): a heavy owner's
            # chunks, nearest first, are read through it on both engines; a small owner's are sorted off
            # idx_memory_chunks_agent_user (test_vector_index_plans holds both). hybrid_search_within_memory is no
            # reader: one memory's chunks are read by ix_memory_chunks_memory. The YugabyteDB small-owner ybhnsw
            # cost (~100 ms at k=10) is the known follow-up at ix_memories_embedding_hnsw.
            SchemaIndex(
                "ix_memory_chunks_embedding",
                "embedding",
                using="hnsw",
                ops={"embedding": "vector_cosine_ops"},
                purpose=(
                    "packages/agent/memory/src/threetears/agent/memory/collections.py"
                    "::MemoryChunkCollection.hybrid_search; "
                    "packages/agent/memory/src/threetears/agent/memory/collections.py"
                    "::MemoryChunkCollection.search_by_semantic"
                ),
                plan_test=(
                    "packages/agent/memory/tests/integration/test_vector_index_plans.py"
                    "::test_ix_memory_chunks_embedding_serves_a_heavy_owners_nearest_reads"
                ),
                reason="memory_chunks carries four indexes: the nearest chunks of an owner too large to sort exactly",
            ),
        ),
        # v0.8.1: global uniqueness on ``chunk_id``. Modelled as a
        # UNIQUE CONSTRAINT (not a unique index) -- prod creates it via
        # ``ALTER TABLE ... ADD CONSTRAINT uq_memory_chunks_chunk_id
        # UNIQUE`` in upstream alembic 064.
        unique_constraints=(SchemaUniqueConstraint("uq_memory_chunks_chunk_id", "chunk_id"),),
    )

    def __init__(self, *args: Any, embedding_dim: int = DEFAULT_EMBEDDING_DIM, **kwargs: Any) -> None:
        """construct the collection with its ``embedding`` column ``embedding_dim`` wide.

        :param args: positional args forwarded to :class:`SchemaBackedCollection`
            (``registry``, ``config``, ``nats_client``, ``write_buffer``)
        :ptype args: Any
        :param embedding_dim: the ``embedding`` column's width, the deployment's (default 1024);
            a write carrying an embedding of another width is refused before any statement runs
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
        The cursor predicate is applied AFTER the auth filter -- auth
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
        :param similarity_threshold: floor on semantic similarity for admission
        :ptype similarity_threshold: float
        :param chunk_signal_weights: mapping ``{"semantic",
            "keyword"}`` to weights
        :ptype chunk_signal_weights: dict[str, float]
        :param fts_min_len: minimum query length for FTS activation
        :ptype fts_min_len: int
        :param fts_max_len: truncation length for FTS queries
        :ptype fts_max_len: int
        :param chunk_id_after: optional cursor -- restrict candidate
            pool to chunks with chunk_id strictly greater
        :ptype chunk_id_after: UUID | None
        :param chunk_id_before: optional cursor -- restrict to chunks
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
        owner = {"agent_id": agent_id, "customer_id": customer_id, "user_id": user_id}
        cursor: tuple[Fragment, ...] = (
            (Fragment("t.chunk_id > $1", (chunk_id_after,)),)
            if chunk_id_after is not None
            else (Fragment("t.chunk_id < $1", (chunk_id_before,)),)
            if chunk_id_before is not None
            else ()
        )
        allowed = _chunk_parent_guards(self.read_filter)

        # cache-bypass: vector-distance search joining memory_chunks ->
        # media (through the new memory parent FK). not primary-key-
        # addressable; L1 row cache cannot help. method on Collection
        # preserves single entry point. The LEFT JOIN matches media
        # rows whose ``memory_id`` equals the chunk's ``memory_id`` --
        # i.e. the chunk's parent memory IS the media's parent memory.
        # Transcript chunks naturally LEFT JOIN to NULL because their
        # parent memory has no media child. The media rows are read by
        # (agent_id, memory_id), ix_media_agent_memory, for the ranked
        # chunks, after the limit (a chunk matching two files is answered
        # once for each, and the first ``candidate_k`` kept, as before).
        # ``mc.embedding IS NOT NULL`` guard: ``NULL <=> vector`` is
        # NULL, which trips ``float(similarity)`` downstream.
        vec_coro = read_nearest(
            self.l3_pool,
            "memory_chunks",
            by=owner,
            where=[Fragment("t.embedding IS NOT NULL", ()), *cursor, *allowed],
            nearest=Nearest("embedding", json.dumps(embedding), similarity="similarity"),
            limit=candidate_k,
            columns=_HYBRID_CHUNK_COLUMNS,
            joins=[
                Joined(
                    "media",
                    on={"memory_id": "memory_id"},
                    by={"agent_id": agent_id},
                    columns={"metadata_json": "metadata_json", "media_id": "media_id"},
                    left=True,
                )
            ],
        )

        fts_text = _build_fts_text(user_text, fts_min_len, fts_max_len)
        if fts_text:
            # cache-bypass: FTS rank query joining memory_chunks ->
            # media through the new memory_id parent FK.
            fts_coro = read_nearest(
                self.l3_pool,
                "memory_chunks",
                by=owner,
                where=[
                    Fragment("t.embedding IS NOT NULL", ()),
                    Fragment(_FTS_MATCH, (fts_text,)),
                    *cursor,
                    *allowed,
                ],
                nearest=TextRank("search_vector", fts_text),
                limit=candidate_k,
                columns=_HYBRID_CHUNK_COLUMNS,
                joins=[
                    Joined(
                        "media",
                        on={"memory_id": "memory_id"},
                        by={"agent_id": agent_id},
                        columns={"metadata_json": "metadata_json", "media_id": "media_id"},
                        left=True,
                    )
                ],
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
        # naturally LEFT JOIN to NULL. the chunks by their keys, then their
        # files by (agent_id, memory_id), ix_media_agent_memory.
        rows = await read_by_joined(
            self.l3_pool,
            "memory_chunks",
            by={"agent_id": agent_id, "chunk_id": AnyOf(chunk_ids, cast="uuid"), "user_id": user_id},
            columns=(
                "chunk_id",
                "content",
                "heading_context",
                "page_number",
                "memory_id",
                "message_id_start",
                "message_id_end",
                "chunk_index",
                "date_created",
            ),
            where=[*_chunk_parent_guards(self.read_filter)],
            joins=[
                Joined(
                    "media",
                    on={"memory_id": "memory_id"},
                    by={"agent_id": agent_id},
                    columns={"metadata_json": "metadata_json", "media_id": "media_id"},
                    left=True,
                )
            ],
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
        date_after: datetime | None = None,
        date_before: datetime | None = None,
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
        :param date_after: inclusive lower bound on the passage's ``date_created``
        :ptype date_after: datetime | None
        :param date_before: inclusive upper bound on the passage's ``date_created``
        :ptype date_before: datetime | None
        :return: list of row dicts, each with the passage's ``date_created``
        :rtype: list[dict[str, Any]]
        """
        if self.l3_pool is None:
            return []
        # cache-bypass: vector-distance search joining memory_chunks
        # -> media via the memory_id parent FK. see :meth:`hybrid_search`.
        # ``mc.embedding IS NOT NULL`` guard: same NULL-cosine
        # rationale as :meth:`hybrid_search`. the file is read by key,
        # ix_media_agent_memory, for the ranked chunks (it was a lateral
        # lookup per chunk: as a plain join YugabyteDB hashed every file of
        # the agent); a chunk under two files is answered once for each, as
        # the lateral answered it.
        rows = await read_nearest(
            self.l3_pool,
            "memory_chunks",
            by={"agent_id": agent_id, "user_id": user_id},
            where=[
                Fragment("t.embedding IS NOT NULL", ()),
                *_date_guards("t.date_created", date_after, date_before),
                *_chunk_parent_guards(self.read_filter),
            ],
            nearest=Nearest("embedding", json.dumps(embedding), similarity="similarity"),
            limit=max_results,
            columns=(
                "chunk_id",
                "content",
                "heading_context",
                "page_number",
                "memory_id",
                "message_id_start",
                "message_id_end",
                "date_created",
            ),
            joins=[
                Joined(
                    "media",
                    on={"memory_id": "memory_id"},
                    by={"agent_id": agent_id},
                    columns={"metadata_json": "metadata_json"},
                    left=True,
                )
            ],
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
                    "date_created": row["date_created"],
                    "similarity": sim,
                },
            )
        return result

    async def find_for_recall(
        self,
        *,
        chunk_id: UUID,
        user_id: UUID,
        agent_id: UUID,
    ) -> dict[str, Any] | None:
        """fetch one passage whole, with where it sits in its memory and when it was written.

        ``total`` is how many passages its memory has, so a reader can say
        "part 12 of 37" and page to a neighbour by ``chunk_index``.

        :param chunk_id: chunk primary-key value
        :ptype chunk_id: UUID
        :param user_id: owning user UUID (row filter)
        :ptype user_id: UUID
        :param agent_id: partition column on memory_chunks; required
        :ptype agent_id: UUID
        :return: ``content``, ``memory_id``, ``chunk_index``, ``total``,
            ``date_created``, ``heading_context``, ``page_number`` and the
            source file's ``metadata_json``; ``None`` when there is no such passage
        :rtype: dict[str, Any] | None
        """
        if self.l3_pool is None:
            return None
        # cache-bypass: by-ID fetch scoped by (agent_id, user_id) -- both are
        # security predicates the L1 cache cannot enforce -- with a sibling count.
        # ``total`` needs no filter of its own: a passage's siblings share its parent memory
        row = await read_row(
            self.l3_pool,
            "memory_chunks",
            by={"agent_id": agent_id, "chunk_id": chunk_id, "user_id": user_id},
            where=[*clause_guards(_chunk_parent_clause, self.read_filter, "t")],
            columns=(
                "content",
                "memory_id",
                "chunk_index",
                "date_created",
                "heading_context",
                "page_number",
                Selected(
                    "(SELECT count(*) FROM memory_chunks s WHERE s.agent_id = t.agent_id AND s.memory_id = t.memory_id AND s.user_id = t.user_id)",
                    "total",
                ),
                Selected(
                    "(SELECT med.metadata_json FROM media med WHERE med.agent_id = t.agent_id AND med.memory_id = t.memory_id LIMIT 1)",
                    "metadata_json",
                ),
            ),
            records=True,
        )
        return dict(row) if row else None

    async def count_by_memory_id(
        self,
        memory_id: UUID,
        *,
        user_id: UUID,
        agent_id: UUID,
        customer_id: UUID,
    ) -> int:
        """how many passages ``memory_id`` has, under the same scope :meth:`find_by_memory_id` reads.

        :param memory_id: parent memory UUID
        :ptype memory_id: UUID
        :param user_id: owning user UUID
        :ptype user_id: UUID
        :param agent_id: partition column on memory_chunks; required
        :ptype agent_id: UUID
        :param customer_id: required sub-scope
        :ptype customer_id: UUID
        :return: the count
        :rtype: int
        """
        if self.l3_pool is None:
            return 0
        # cache-bypass: an aggregate over one memory's passages.
        (counted,) = await aggregate_keyed(
            self.l3_pool,
            "memory_chunks",
            aggregates={"n": Count()},
            by={"agent_id": agent_id, "memory_id": memory_id, "user_id": user_id, "customer_id": customer_id},
            where=_chunk_parent_guards(self.read_filter),
        )
        return int(counted["n"] or 0)

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
        # cache-bypass: by-ID fetch scoped by (agent_id, user_id) --
        # both are security predicates the L1 cache cannot enforce.
        row = await read_row(
            self.l3_pool,
            "memory_chunks",
            by={"agent_id": agent_id, "chunk_id": chunk_id, "user_id": user_id},
            columns=["content", "memory_id"],
            where=_chunk_parent_guards(self.read_filter),
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
        SQL site enforces -- skipping any one is a cross-tenant data
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
        :param chunk_id_after: cursor -- return chunks with chunk_id
            strictly greater
        :ptype chunk_id_after: UUID | None
        :param chunk_id_before: cursor -- return chunks with chunk_id
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
        # enforced on every row. a page in chunk_id order, past the cursor
        # (backward: DESC before ``chunk_id_before``).
        backward = chunk_id_before is not None
        cursor: tuple[Fragment, ...] = (
            (Fragment("t.chunk_id > $1", (chunk_id_after,)),)
            if chunk_id_after is not None
            else (Fragment("t.chunk_id < $1", (chunk_id_before,)),)
            if backward
            else ()
        )
        page = await keyset_page(
            self.l3_pool,
            "memory_chunks",
            columns=(
                "chunk_id",
                "memory_id",
                "chunk_index",
                "content",
                "summary",
                "heading_context",
                "page_number",
                "message_id_start",
                "message_id_end",
                "date_created",
            ),
            by={"agent_id": agent_id, "memory_id": memory_id, "user_id": user_id, "customer_id": customer_id},
            order=[Sort("chunk_id", "uuid", descending=backward)],
            limit=limit,
            where=[*cursor, *_chunk_parent_guards(self.read_filter)],
        )
        rows = page.rows
        order_dir = "DESC" if backward else "ASC"
        result = [dict(row) for row in rows]
        # client-facing order is always ASC; flip the DESC backward
        # page so callers get a stable left-to-right list.
        if order_dir == "DESC":
            result.reverse()
        return result

    async def find_by_chunk_indexes(
        self,
        memory_id: UUID,
        *,
        user_id: UUID,
        agent_id: UUID,
        customer_id: UUID,
        chunk_indexes: Sequence[int],
    ) -> list[dict[str, Any]]:
        """fetch the chunks of ``memory_id`` sitting at ``chunk_indexes``.

        ``chunk_index`` is the stable ordinal a caller reads off an
        earlier chunk listing, so a re-read of "chunks 3 and 7" is
        addressable without carrying chunk ids around. rows come back
        in ``chunk_index ASC`` order whatever order they were asked
        for, and indexes with no row are simply absent.

        Auth scoping: the SQL carries the full ``(user_id, agent_id,
        customer_id)`` triple, matching every other chunk lookup.
        ``chunk_indexes`` is the only caller-supplied filter and binds
        as one array parameter.

        :param memory_id: parent memory UUID
        :ptype memory_id: UUID
        :param user_id: owning user UUID
        :ptype user_id: UUID
        :param agent_id: partition column on memory_chunks; required
        :ptype agent_id: UUID
        :param customer_id: required sub-scope
        :ptype customer_id: UUID
        :param chunk_indexes: ordinals to fetch; empty returns empty
        :ptype chunk_indexes: Sequence[int]
        :return: chunk row dicts in chunk_index ASC order
        :rtype: list[dict[str, Any]]
        """
        wanted = list(chunk_indexes)
        if not wanted:
            return []
        if self.l3_pool is None:
            return []

        # cache-bypass: a multi-row scan selecting an arbitrary SET of
        # ordinals within one memory is not primary-key addressable
        # (the pk is ``(agent_id, chunk_id)``). index
        # ix_memory_chunks_memory backs (memory_id, chunk_index); the
        # auth triple is enforced on every row.
        rows = await read_by(
            self.l3_pool,
            "memory_chunks",
            by={
                "agent_id": agent_id,
                "memory_id": memory_id,
                "user_id": user_id,
                "customer_id": customer_id,
                "chunk_index": AnyOf(wanted, cast="integer"),
            },
            columns=[
                "chunk_id",
                "chunk_index",
                "content",
                "summary",
                "message_id_start",
                "message_id_end",
                "token_count",
                "date_created",
            ],
            where=_chunk_parent_guards(self.read_filter),
            order=[Sort("chunk_index")],
        )
        return [dict(row) for row in rows]

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
        on each chunk row.

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
        :param chunk_id_after: cursor -- chunks with chunk_id > cursor
        :ptype chunk_id_after: UUID | None
        :param chunk_id_before: cursor -- chunks with chunk_id < cursor
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

        # cache-bypass: the conversation's memories by idx_mem_conversation,
        # then their chunks by ix_memory_chunks_memory, the owner checked on
        # each chunk: two keyed reads, the form both planners took behind the
        # MATERIALIZED fence (b2bc83998: with the owner on the chunk in a plain
        # join YugabyteDB read every chunk of the user; with it on the memory,
        # PostgreSQL read every memory of the owner).
        memories = await read_by(
            self.l3_pool,
            "memories",
            by={"agent_id": agent_id, "conversation_id": conversation_id},
            columns=("memory_id",),
        )
        if not memories:
            return []
        backward = chunk_id_before is not None
        cursor: tuple[Fragment, ...] = (
            (Fragment("t.chunk_id > $1", (chunk_id_after,)),)
            if chunk_id_after is not None
            else (Fragment("t.chunk_id < $1", (chunk_id_before,)),)
            if backward
            else ()
        )
        page = await keyset_page(
            self.l3_pool,
            "memory_chunks",
            columns=(
                "chunk_id",
                "memory_id",
                "chunk_index",
                "content",
                "summary",
                "heading_context",
                "page_number",
                "message_id_start",
                "message_id_end",
                "date_created",
            ),
            by={
                "agent_id": agent_id,
                "memory_id": AnyOf([row["memory_id"] for row in memories], cast="uuid"),
                "user_id": user_id,
                "customer_id": customer_id,
            },
            order=[
                Sort("message_id_start", "uuid", descending=backward),
                Sort("chunk_id", "uuid", descending=backward),
            ],
            limit=limit,
            where=[Fragment("t.message_id_start IS NOT NULL", ()), *cursor, *_chunk_parent_guards(self.read_filter)],
        )
        rows = page.rows
        order_dir = "DESC" if backward else "ASC"
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
        the cross-memory path -- the memory_id filter is purely
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
        :param similarity_threshold: floor on semantic similarity for admission
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
        # the owner triple, then the one parent memory: the memory_id pin is
        # purely additive to the auth filter, never a replacement for it
        within = {"agent_id": agent_id, "customer_id": customer_id, "user_id": user_id, "memory_id": memory_id}
        allowed = _chunk_parent_guards(self.read_filter)

        # cache-bypass: vector-distance search joining memory_chunks
        # to media (via memory_id parent FK) scoped to one parent
        # memory, the files read by key for the ranked chunks, as
        # :meth:`hybrid_search` reads them. auth triple enforced.
        # ``mc.embedding IS NOT NULL`` guard: an un-embedded chunk has a
        # NULL distance, which ``float()`` in the merge cannot take; it is
        # left to the keyword arm, as in every other vector query here.
        vec_coro = read_nearest(
            self.l3_pool,
            "memory_chunks",
            by=within,
            where=[Fragment("t.embedding IS NOT NULL", ()), *allowed],
            nearest=Nearest("embedding", json.dumps(embedding), similarity="similarity"),
            limit=candidate_k,
            columns=_HYBRID_CHUNK_COLUMNS,
            joins=[
                Joined(
                    "media",
                    on={"memory_id": "memory_id"},
                    by={"agent_id": agent_id},
                    columns={"metadata_json": "metadata_json", "media_id": "media_id"},
                    left=True,
                )
            ],
        )

        fts_text = _build_fts_text(user_text, fts_min_len, fts_max_len)
        if fts_text:
            # cache-bypass: FTS rank query joining chunks -> media via
            # memory_id, scoped to one parent memory.
            fts_coro = read_nearest(
                self.l3_pool,
                "memory_chunks",
                by=within,
                where=[Fragment(_FTS_MATCH, (fts_text,)), *allowed],
                nearest=TextRank("search_vector", fts_text),
                limit=candidate_k,
                columns=_HYBRID_CHUNK_COLUMNS,
                joins=[
                    Joined(
                        "media",
                        on={"memory_id": "memory_id"},
                        by={"agent_id": agent_id},
                        columns={"metadata_json": "metadata_json", "media_id": "media_id"},
                        left=True,
                    )
                ],
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

    namespace-task-01 phase 8.5l-2: retires :class:`MemoryLedger` -- the
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

    tenancy = TENANT_BY_SCHEMA

    primary_key_column: str | tuple[str, ...] = ("conversation_id", "item_id")
    # rationale: ``save_to_store`` is a framework override that
    # truncates ``short_desc`` to the migration-v002 VARCHAR(150) bound;
    # the underlying ``data`` dict already carries ``conversation_id``
    # (the partition column) by construction since
    # :class:`SchemaBackedCollection` enforces required-column presence
    # before this override runs. exempting the framework override
    # keeps the partition contract on the read surface
    # (``find_by_conversation``) without weakening the static guard.
    partition_exempt_methods: ClassVar[frozenset[str]] = frozenset({"save_to_store"})
    # v0.8.0 enrichment: ``date_created`` carries ``server_default="now()"``
    # to match prod (prod ``information_schema`` confirms the default).
    # ``date_created`` is also immutable per the standard 3tears
    # convention. The FK on ``conversation_id`` matches the prod
    # constraint ``conversation_memory_refs_conversation_id_fkey``
    # (CASCADE on parent conversation delete) -- declared at table
    # level because the inline 2-tuple form does not carry
    # ``on_delete=``. The one secondary index is the (conversation_id,
    # date_created) one the migrations build for the ledger read.
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
            # the agent that surfaced the item (v032); NULL for a caller that names none
            Column("agent_id", UUID_TYPE, nullable=True),
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
            # the ledger read: the refs one agent surfaced in a conversation, newest first (v034; it
            # replaced the conversation-wide (conversation_id, date_created) index). date_created is
            # stored descending: YugabyteDB will not read an ascending one backwards for the newest first.
            # v037: item_id DESC after it, the key completing the page's order in its direction, so a page
            # walks the index rather than reading and sorting the agent's refs in the conversation (YugabyteDB
            # has no incremental sort); it covers v034's index, dropped in the same step
            SchemaIndex(
                "idx_conv_mem_refs_conversation_agent_key",
                "conversation_id",
                "agent_id",
                "date_created",
                "item_id",
                desc=("date_created", "item_id"),
                purpose=(
                    "packages/agent/memory/src/threetears/agent/memory/collections.py"
                    "::MemoryRefsCollection.find_by_conversation"
                ),
                plan_test=(
                    "packages/agent/memory/tests/integration/test_index_plans.py"
                    "::test_conv_mem_refs_conversation_agent_key"
                ),
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
        agent_id: UUID,
        *,
        limit: int = MEMORY_REFS_PAGE,
    ) -> list[MemoryRefEntity]:
        """fetch the newest ``limit`` refs one agent surfaced in a conversation, returned oldest first.

        in a conversation several agents share, what one recalled is its
        own: the read names the agent (0.95.0; a read of every agent's refs
        was a manager built without its agent, which read the others' refs).
        the surfaced-refs FIFO a caller keeps holds at most
        :data:`MEMORY_REFS_PAGE` per agent and deletes the oldest past it,
        so the default reads every ref it kept; each read is one page of
        the conversation's (and agent's) date index, newest first.

        absorbs the former ``MemoryLedger.load(pool, conversation_id)``
        SQL into a Collection method. multi-row scan is not primary-
        key-addressable: the query touches every row sharing the
        ``conversation_id`` prefix of the composite pk, so L1 row-level
        cache does not apply. hits are NOT promoted into L2 (not written to L2: a row read from L3 can be older than a write that landed after the query, and an unfenced put would serve that older row to every reader -- the next single-row read seeds L2 at the key's revision instead), so other
        pods starting cold can resolve the row without an L3 round-
        trip.

        :param conversation_id: conversation UUID to scan for
        :ptype conversation_id: UUID
        :param agent_id: the agent whose refs these are
        :ptype agent_id: UUID
        :param limit: the most refs to read, newest first
        :ptype limit: int
        :return: list of ref entities in chronological order
        :rtype: list[MemoryRefEntity]
        """
        if self.l3_pool is None:
            return []
        # cache-bypass: a page of one conversation's refs by its date index
        # is not primary-key-addressable. L1 row cache serves per-
        # ``(conversation_id, item_id)`` lookups only. keeping the query on
        # the Collection preserves the single entry point + lets callers
        # share L2 invalidation channels.
        rows = (
            await keyset_page(
                self.l3_pool,
                "conversation_memory_refs",
                columns=(
                    "conversation_id",
                    "item_id",
                    "item_type",
                    "short_desc",
                    "date_created",
                    "date_updated",
                    "agent_id",
                ),
                by={"conversation_id": conversation_id, "agent_id": agent_id},
                # item_id runs the way date_created does: idx_conv_mem_refs_conversation_agent_key (v037) walks
                # the page in this order; completed ascending it was read and sorted
                order=[Sort("date_created", descending=True), Sort("item_id", descending=True)],
                limit=limit,
                records=True,
            )
        ).rows
        entities: list[MemoryRefEntity] = []
        for row in reversed(rows):
            data = self._coerce_row(dict(row))
            entity = self.entity_class(data, is_new=False, collection=self)
            entities.append(entity)
        return entities


# Defensive upper bound on the consolidation-provenance walk. The visited
# set already terminates any pre-existing cycle among non-gist nodes; this
# caps a pathologically large provenance graph so the guard can never hang.
_MAX_CONSOLIDATION_WALK_NODES = 10_000


class ConsolidationCycleError(Exception):
    """raised when recording consolidation edges would form a cycle.

    A gist may not be consolidated from itself or from any of its
    descendants (memories reachable through existing consolidation
    edges). See :func:`assert_no_consolidation_cycle`.
    """


async def assert_no_consolidation_cycle(
    *,
    consolidated_memory_id: UUID,
    source_memory_ids: Iterable[UUID],
    sources_of: Callable[[UUID], Awaitable[Iterable[UUID]]],
    max_nodes: int = _MAX_CONSOLIDATION_WALK_NODES,
) -> None:
    """reject a consolidation whose gist is reachable from any source.

    A DAG reachability check (not a single-parent walk): starting from
    every source, walk the ``consolidated -> source`` provenance edges
    via ``sources_of`` and reject if the walk reaches
    ``consolidated_memory_id``. That catches both a **self-merge** (a
    source that IS the gist) and a **descendant-merge** (a source whose
    provenance chain leads back to the gist).

    A ``visited`` set makes the walk loop-safe and keeps a legitimate
    diamond DAG -- two sources that share a common ancestor -- from
    false-positiving: only the target gist is treated as a cycle, never a
    merely-revisited node. Read-only over the edge graph; the source
    rows are never mutated.

    :param consolidated_memory_id: the prospective gist id
    :ptype consolidated_memory_id: UUID
    :param source_memory_ids: the sources fanning into the gist
    :ptype source_memory_ids: Iterable[UUID]
    :param sources_of: async lookup returning the sources a given memory
        was itself consolidated from (empty for a non-gist memory)
    :ptype sources_of: Callable[[UUID], Awaitable[Iterable[UUID]]]
    :param max_nodes: defensive cap on distinct nodes walked
    :ptype max_nodes: int
    :return: nothing on success
    :rtype: None
    :raises ConsolidationCycleError: when the gist is reachable from a
        source, or the walk exceeds ``max_nodes``
    """
    visited: set[UUID] = set()
    stack: list[UUID] = list(source_memory_ids)
    while stack:
        node = stack.pop()
        if node == consolidated_memory_id:
            raise ConsolidationCycleError(
                f"consolidation cycle: gist {consolidated_memory_id} is itself a source "
                f"or a descendant of one of its sources",
            )
        if node in visited:
            continue
        visited.add(node)
        if len(visited) > max_nodes:
            raise ConsolidationCycleError(
                f"consolidation provenance walk for gist {consolidated_memory_id} "
                f"exceeded {max_nodes} nodes (likely a corrupt edge graph)",
            )
        stack.extend(await sources_of(node))


class MemoryConsolidationsCollection(SchemaBackedCollection[MemoryConsolidationEntity]):
    """three-tier collection for :class:`MemoryConsolidationEntity` (v026).

    The N:1 provenance edge table that Dream consolidation (A5) writes:
    one row per ``(consolidated_memory_id, source_memory_id)`` pair.
    Composite primary key ``(agent_id, consolidated_memory_id,
    source_memory_id)`` on top of the composite-pk
    :class:`BaseCollection` support; both memory refs are composite FKs
    to ``memories(agent_id, memory_id)`` ON DELETE CASCADE, so removing
    either endpoint memory drops its edges without touching the surviving
    endpoint's own row (fully non-destructive to the source's salience).

    CRUD is generated from :attr:`schema`; there is no CAS column (the
    edge is append-only). :meth:`find_sources` and
    :meth:`find_consolidated_into` carry ``# cache-bypass:`` because a
    back-lookup by one composite-pk prefix is not full-pk addressable.
    :meth:`assert_no_cycle` guards writes against a provenance cycle.
    """

    tenancy = TENANT_BY_SCHEMA

    primary_key_column: str | tuple[str, ...] = (
        "agent_id",
        "consolidated_memory_id",
        "source_memory_id",
    )
    schema = TableSchema(
        name="memory_consolidations",
        primary_key=("agent_id", "consolidated_memory_id", "source_memory_id"),
        columns=[
            Column("agent_id", UUID_TYPE, partition=True),
            Column("consolidated_memory_id", UUID_TYPE),
            Column("source_memory_id", UUID_TYPE),
            # the audit trail (why these merged); no crypto ceremony.
            Column("rationale", STRING_TYPE, nullable=True),
            Column(
                "date_created",
                DATETIMETZ_TYPE,
                immutable=True,
                server_default="now()",
            ),
            Column("date_updated", DATETIMETZ_TYPE, nullable=True),
        ],
        foreign_keys=(
            # both refs are composite FKs to memories with CASCADE, so a
            # memory delete cleans up its edges (either endpoint).
            SchemaForeignKey(
                ("agent_id", "consolidated_memory_id"),
                "memories",
                ("agent_id", "memory_id"),
                on_delete="CASCADE",
            ),
            SchemaForeignKey(
                ("agent_id", "source_memory_id"),
                "memories",
                ("agent_id", "memory_id"),
                on_delete="CASCADE",
            ),
        ),
        indexes=(
            # back-edge lookup "what was this source merged into?" + the
            # cycle-guard provenance walk.
            SchemaIndex(
                "idx_memory_consolidations_source",
                "agent_id",
                "source_memory_id",
                # the marks' corrected_by sub-select, matched on the outer memory's key, reads one source's
                # edges (a source is merged into few gists): those few rows sorted, not the order of an index
                purpose=(
                    "packages/agent/memory/src/threetears/agent/memory/collections.py"
                    "::MemoryConsolidationsCollection.find_consolidated_into; "
                    "packages/agent/memory/src/threetears/agent/memory/collections.py::MemoriesCollection.search_by_semantic; "
                    "packages/agent/memory/src/threetears/agent/memory/collections.py::MemoriesCollection.search_by_fts; "
                    "packages/agent/memory/src/threetears/agent/memory/collections.py::MemoriesCollection.list_for_user; "
                    "packages/agent/memory/src/threetears/agent/memory/collections.py::MemoriesCollection.find_by_alias; "
                    "packages/agent/memory/src/threetears/agent/memory/collections.py::MemoriesCollection.search_by_ids; "
                    "packages/agent/memory/src/threetears/agent/memory/collections.py::MemoriesCollection.find_for_recall; "
                    "packages/agent/memory/src/threetears/agent/memory/merge.py::_delete_alias_collisions"
                ),
                plan_test=(
                    "packages/agent/memory/tests/integration/test_index_plans.py::test_memory_consolidations_source"
                ),
            ),
        ),
    )

    @property
    def table_name(self) -> str:
        """return database table name.

        :return: table name
        :rtype: str
        """
        return "memory_consolidations"

    @property
    def entity_class(self) -> type[MemoryConsolidationEntity]:
        """return entity class for this collection.

        :return: entity class
        :rtype: type[MemoryConsolidationEntity]
        """
        return MemoryConsolidationEntity

    async def find_sources(
        self,
        agent_id: UUID,
        consolidated_memory_id: UUID,
    ) -> list[UUID]:
        """return the source ids a gist was consolidated from.

        The forward edge: ``consolidated_memory_id -> source_memory_id``.
        Drives the cycle-guard provenance walk.

        :param agent_id: partition column (required)
        :ptype agent_id: UUID
        :param consolidated_memory_id: the gist to look up sources for
        :ptype consolidated_memory_id: UUID
        :return: source memory ids (empty for a non-gist memory)
        :rtype: list[UUID]
        """
        if self.l3_pool is None:
            return []
        # cache-bypass: back-lookup by the (agent_id,
        # consolidated_memory_id) prefix of the composite pk is not
        # full-pk-addressable, so the L1 row cache does not serve it;
        # keeping it on the Collection preserves the single entry point.
        rows = await read_by(
            self.l3_pool,
            "memory_consolidations",
            by={"agent_id": agent_id, "consolidated_memory_id": consolidated_memory_id},
            columns=("source_memory_id",),
            records=True,
        )
        return [_as_pk_uuid(row["source_memory_id"]) for row in rows]

    async def find_consolidated_into(
        self,
        agent_id: UUID,
        source_memory_id: UUID,
    ) -> list[UUID]:
        """return the gist ids a source memory was merged into.

        The back edge (served by ``idx_memory_consolidations_source``):
        ``source_memory_id -> consolidated_memory_id``.

        :param agent_id: partition column (required)
        :ptype agent_id: UUID
        :param source_memory_id: the source to look up gists for
        :ptype source_memory_id: UUID
        :return: consolidated (gist) memory ids
        :rtype: list[UUID]
        """
        if self.l3_pool is None:
            return []
        # cache-bypass: back-lookup by (agent_id, source_memory_id) is
        # not full-pk-addressable; served by the back-edge GIN-free btree
        # index idx_memory_consolidations_source.
        rows = await read_by(
            self.l3_pool,
            "memory_consolidations",
            by={"agent_id": agent_id, "source_memory_id": source_memory_id},
            columns=("consolidated_memory_id",),
            records=True,
        )
        return [_as_pk_uuid(row["consolidated_memory_id"]) for row in rows]

    async def assert_no_cycle(
        self,
        agent_id: UUID,
        *,
        consolidated_memory_id: UUID,
        source_memory_ids: Iterable[UUID],
    ) -> None:
        """guard a prospective consolidation against a provenance cycle.

        Supplies the DB-backed ``sources_of`` lookup (scoped to
        ``agent_id``) to :func:`assert_no_consolidation_cycle`. Call
        before recording the edges (A5's ``run_consolidation``).

        :param agent_id: partition column (required)
        :ptype agent_id: UUID
        :param consolidated_memory_id: the prospective gist id
        :ptype consolidated_memory_id: UUID
        :param source_memory_ids: the sources fanning into the gist
        :ptype source_memory_ids: Iterable[UUID]
        :return: nothing on success
        :rtype: None
        :raises ConsolidationCycleError: when the gist is reachable from
            a source
        """

        async def _sources_of(memory_id: UUID) -> list[UUID]:
            # find_sources already normalises row ids via _as_pk_uuid.
            return await self.find_sources(agent_id, memory_id)

        # normalise the caller-supplied ids to the same canonical stdlib
        # UUID form find_sources returns, so the walk's set membership and
        # equality checks compare like with like.
        await assert_no_consolidation_cycle(
            consolidated_memory_id=_as_pk_uuid(consolidated_memory_id),
            source_memory_ids=[_as_pk_uuid(sid) for sid in source_memory_ids],
            sources_of=_sources_of,
        )
