"""
agent-memory v022: install HNSW / GIN / btree / unique indexes that
v0.8.1 added to the canonical TableSchemas.

v0.8.1 enriched the four memory-package schemas (memories, media,
media_content, memory_chunks) with 17 indexes / unique constraints
that match the per-table shape that upstream prod maintains via its
own Alembic migrations:

- ``memories`` (5): ``idx_memories_agent_user`` composite btree,
  ``idx_memories_agent_customer_user`` composite btree,
  ``idx_memories_search_vector`` GIN FTS,
  ``ix_memories_embedding_hnsw`` HNSW vector with
  ``vector_cosine_ops`` opclass and ``m=16, ef_construction=64``,
  ``uq_memories_memory_id`` UNIQUE constraint on the global id.
- ``media`` (3): ``idx_media_agent_user`` composite btree,
  ``ix_media_extraction_pending`` partial btree (rows in the
  extraction queue), ``uq_media_media_id`` UNIQUE constraint.
- ``media_content`` (4): ``idx_media_content_agent_user`` composite
  btree, ``idx_media_content_search_vector`` GIN FTS,
  ``ix_media_content_embedding`` HNSW vector (no WITH parameters --
  prod was created before upstream started parametrising HNSW),
  ``uq_media_content_content_id`` UNIQUE constraint.
- ``memory_chunks`` (5): ``idx_chunks_message_id_start`` partial
  composite btree on ``(agent_id, message_id_start)`` (fixed in
  v0.8.2; v0.8.1 mistakenly wrote ``(message_id_start, chunk_index)``
  -- see v023 for the drop-and-recreate path),
  ``idx_memory_chunks_agent_user`` composite btree,
  ``idx_memory_chunks_search_vector`` GIN FTS,
  ``ix_memory_chunks_embedding`` HNSW vector (no WITH parameters),
  ``uq_memory_chunks_chunk_id`` UNIQUE constraint.

Without this migration, 3tears agent pods (which run the 3tears
agent-memory migration runner against per-agent schema DBs) never
get these indexes -- hybrid search falls back to sequential scan and
the unique-on-id constraints are absent, so cross-agent leaks of the
same UUID do not surface at the storage layer.

Idempotency rules (all defensive):
- Indexes use ``CREATE INDEX IF NOT EXISTS``. Postgres has no
  ``CREATE UNIQUE INDEX IF NOT EXISTS`` variant of its own; the
  ``IF NOT EXISTS`` clause applies to UNIQUE indexes too.
- Unique constraints use a DO-block that checks
  ``pg_catalog.pg_constraint`` before ``ALTER TABLE ... ADD
  CONSTRAINT`` so replay is a no-op. Postgres has no
  ``ADD CONSTRAINT IF NOT EXISTS``.
- HNSW + GIN indexes require their respective extensions
  (``pgvector`` for HNSW, ``pg_trgm`` is NOT needed -- the GIN
  indexes here are on TSVECTOR columns which are core Postgres).
  ``pgvector`` is already installed by v001 (``CREATE EXTENSION IF
  NOT EXISTS vector``).
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "add_hnsw_gin_indexes",
]

log = get_logger(__name__)


# ----- memories indexes (5 adds) -------------------------------------- #

_IDX_MEMORIES_AGENT_USER_SQL = "CREATE INDEX IF NOT EXISTS idx_memories_agent_user ON memories (agent_id, user_id)"

_IDX_MEMORIES_AGENT_CUSTOMER_USER_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_memories_agent_customer_user ON memories (agent_id, customer_id, user_id)"
)

_IDX_MEMORIES_SEARCH_VECTOR_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_memories_search_vector ON memories USING gin (search_vector)"
)

_IDX_MEMORIES_EMBEDDING_HNSW_SQL = (
    "CREATE INDEX IF NOT EXISTS ix_memories_embedding_hnsw "
    "ON memories USING hnsw (embedding public.vector_cosine_ops) "
    "WITH (m = 16, ef_construction = 64)"
)

# ----- media indexes (3 adds) ----------------------------------------- #

_IDX_MEDIA_AGENT_USER_SQL = "CREATE INDEX IF NOT EXISTS idx_media_agent_user ON media (agent_id, user_id)"

_IDX_MEDIA_EXTRACTION_PENDING_SQL = (
    "CREATE INDEX IF NOT EXISTS ix_media_extraction_pending "
    "ON media (extraction_status) "
    "WHERE extraction_status = 'pending'"
)

# ----- media_content indexes (4 adds) --------------------------------- #

_IDX_MEDIA_CONTENT_AGENT_USER_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_media_content_agent_user ON media_content (agent_id, user_id)"
)

_IDX_MEDIA_CONTENT_SEARCH_VECTOR_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_media_content_search_vector ON media_content USING gin (search_vector)"
)

# Prod does NOT carry a WITH clause for this HNSW index -- mirror
# (see :class:`MediaContentCollection.schema` for the rationale).
_IX_MEDIA_CONTENT_EMBEDDING_SQL = "CREATE INDEX IF NOT EXISTS ix_media_content_embedding ON media_content USING hnsw (embedding public.vector_cosine_ops)"

# ----- memory_chunks indexes (5 adds) --------------------------------- #

# v0.8.2: column shape corrected to ``(agent_id, message_id_start)``
# to match the schema declaration in ``MemoryChunkCollection.schema``
# and the prod shape (verified against
# ``pg_indexes`` on the dev DB). v0.8.1 shipped this migration with
# ``(message_id_start, chunk_index)`` which mismatched both the
# schema and prod -- the parity test passed because it compares
# schema vs reference fixture (both correct) without consulting the
# migration SQL. v023 handles the drop-and-recreate path for any
# agent pods that already ran the broken v022.
_IDX_CHUNKS_MESSAGE_ID_START_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_chunks_message_id_start "
    "ON memory_chunks (agent_id, message_id_start) "
    "WHERE message_id_start IS NOT NULL"
)

_IDX_MEMORY_CHUNKS_AGENT_USER_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_memory_chunks_agent_user ON memory_chunks (agent_id, user_id)"
)

_IDX_MEMORY_CHUNKS_SEARCH_VECTOR_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_memory_chunks_search_vector ON memory_chunks USING gin (search_vector)"
)

_IX_MEMORY_CHUNKS_EMBEDDING_SQL = "CREATE INDEX IF NOT EXISTS ix_memory_chunks_embedding ON memory_chunks USING hnsw (embedding public.vector_cosine_ops)"

# ----- unique constraints (4 adds; DO-block guarded) ------------------ #

# Postgres has no ``ALTER TABLE ... ADD CONSTRAINT ... UNIQUE IF NOT
# EXISTS``. Each guard checks ``pg_catalog.pg_constraint`` for an
# existing constraint with the target name on the target table; only
# adds when absent. ``conrelid`` joins to ``pg_class`` to disambiguate
# multi-table constraints with the same name (rare but legal).
_ADD_UQ_MEMORIES_MEMORY_ID_SQL = """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_catalog.pg_constraint con
        JOIN pg_catalog.pg_class cls ON cls.oid = con.conrelid
        WHERE cls.relname = 'memories'
          AND con.conname = 'uq_memories_memory_id'
    ) THEN
        EXECUTE 'ALTER TABLE memories ADD CONSTRAINT uq_memories_memory_id UNIQUE (memory_id)';
    END IF;
END
$$
"""

_ADD_UQ_MEDIA_MEDIA_ID_SQL = """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_catalog.pg_constraint con
        JOIN pg_catalog.pg_class cls ON cls.oid = con.conrelid
        WHERE cls.relname = 'media'
          AND con.conname = 'uq_media_media_id'
    ) THEN
        EXECUTE 'ALTER TABLE media ADD CONSTRAINT uq_media_media_id UNIQUE (media_id)';
    END IF;
END
$$
"""

_ADD_UQ_MEDIA_CONTENT_CONTENT_ID_SQL = """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_catalog.pg_constraint con
        JOIN pg_catalog.pg_class cls ON cls.oid = con.conrelid
        WHERE cls.relname = 'media_content'
          AND con.conname = 'uq_media_content_content_id'
    ) THEN
        EXECUTE 'ALTER TABLE media_content ADD CONSTRAINT uq_media_content_content_id UNIQUE (content_id)';
    END IF;
END
$$
"""

_ADD_UQ_MEMORY_CHUNKS_CHUNK_ID_SQL = """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_catalog.pg_constraint con
        JOIN pg_catalog.pg_class cls ON cls.oid = con.conrelid
        WHERE cls.relname = 'memory_chunks'
          AND con.conname = 'uq_memory_chunks_chunk_id'
    ) THEN
        EXECUTE 'ALTER TABLE memory_chunks ADD CONSTRAINT uq_memory_chunks_chunk_id UNIQUE (chunk_id)';
    END IF;
END
$$
"""


async def add_hnsw_gin_indexes(store: DataStore) -> None:
    """install the v0.8.1 enrichment indexes on the per-agent schema.

    :param store: DataStore bound to per-agent schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("installing v0.8.1 HNSW/GIN/btree/unique indexes (v022)")
    # memories
    await store.execute(_IDX_MEMORIES_AGENT_USER_SQL)
    await store.execute(_IDX_MEMORIES_AGENT_CUSTOMER_USER_SQL)
    await store.execute(_IDX_MEMORIES_SEARCH_VECTOR_SQL)
    await store.execute(_IDX_MEMORIES_EMBEDDING_HNSW_SQL)
    # media
    await store.execute(_IDX_MEDIA_AGENT_USER_SQL)
    await store.execute(_IDX_MEDIA_EXTRACTION_PENDING_SQL)
    # media_content
    await store.execute(_IDX_MEDIA_CONTENT_AGENT_USER_SQL)
    await store.execute(_IDX_MEDIA_CONTENT_SEARCH_VECTOR_SQL)
    await store.execute(_IX_MEDIA_CONTENT_EMBEDDING_SQL)
    # memory_chunks
    await store.execute(_IDX_CHUNKS_MESSAGE_ID_START_SQL)
    await store.execute(_IDX_MEMORY_CHUNKS_AGENT_USER_SQL)
    await store.execute(_IDX_MEMORY_CHUNKS_SEARCH_VECTOR_SQL)
    await store.execute(_IX_MEMORY_CHUNKS_EMBEDDING_SQL)
    # unique constraints (DO-block guarded; the 4 ``uq_<table>_<id>``
    # constraints mirror the prod ``ALTER TABLE ... ADD CONSTRAINT
    # UNIQUE`` shape that upstream alembic 064 produced, so Alembic
    # auto-gen reads them out of ``information_schema.table_constraints``
    # rather than ``pg_indexes``).
    await store.execute(_ADD_UQ_MEMORIES_MEMORY_ID_SQL)
    await store.execute(_ADD_UQ_MEDIA_MEDIA_ID_SQL)
    await store.execute(_ADD_UQ_MEDIA_CONTENT_CONTENT_ID_SQL)
    await store.execute(_ADD_UQ_MEMORY_CHUNKS_CHUNK_ID_SQL)
