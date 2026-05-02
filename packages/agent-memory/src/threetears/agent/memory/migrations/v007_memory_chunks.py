"""
agent-memory v007: create memory_chunks table.

memory-task-01. ``retrieval.py._query_chunks`` and ``tools.py``'s batch
lookup / semantic search paths read from a ``memory_chunks`` table that
no earlier migration creates. Chunks carry document-like location
metadata (``heading_context``, ``page_number``) alongside the same
``content`` / ``summary`` / ``embedding`` / ``search_vector`` shape the
``memories`` and ``media_content`` tables carry, plus a nullable
``media_id`` that points back to the parent artifact so retrieval can
dedup chunks whose media has already been surfaced.

Idempotent: ``CREATE TABLE IF NOT EXISTS`` + ``CREATE INDEX IF NOT
EXISTS`` + trigger CREATE OR REPLACE pattern.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "create_memory_chunks",
]

log = get_logger(__name__)


_CREATE_MEMORY_CHUNKS_SQL = """
CREATE TABLE IF NOT EXISTS memory_chunks (
    chunk_id UUID PRIMARY KEY,
    media_id UUID NULL REFERENCES media(media_id) ON DELETE CASCADE,
    agent_id UUID NULL,
    customer_id UUID NULL,
    user_id UUID NOT NULL,
    content TEXT NOT NULL,
    summary TEXT NULL,
    heading_context TEXT NULL,
    page_number INTEGER NULL,
    embedding public.vector(1024) NULL,
    search_vector TSVECTOR NULL,
    date_created TIMESTAMP NOT NULL
)
"""

_CREATE_CHUNKS_USER_IDX_SQL = "CREATE INDEX IF NOT EXISTS idx_chunks_user ON memory_chunks (user_id)"

_CREATE_CHUNKS_MEDIA_IDX_SQL = "CREATE INDEX IF NOT EXISTS idx_chunks_media ON memory_chunks (media_id)"

_CREATE_CHUNKS_EMBEDDING_IDX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_chunks_embedding "
    "ON memory_chunks USING hnsw (embedding public.vector_cosine_ops) "
    "WHERE embedding IS NOT NULL"
)

_CREATE_CHUNKS_SEARCH_VECTOR_IDX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_chunks_search_vector ON memory_chunks USING GIN (search_vector)"
)

_CREATE_CHUNKS_TRIGGER_FUNC_SQL = """
CREATE OR REPLACE FUNCTION memory_chunks_search_vector_update()
RETURNS trigger AS $$
BEGIN
    NEW.search_vector :=
        setweight(to_tsvector('english', coalesce(NEW.content, '')), 'A')
        || setweight(to_tsvector('english', coalesce(NEW.summary, '')), 'B');
    RETURN NEW;
END
$$ LANGUAGE plpgsql
"""

_DROP_CHUNKS_TRIGGER_SQL = "DROP TRIGGER IF EXISTS memory_chunks_search_vector_trigger ON memory_chunks"

_CREATE_CHUNKS_TRIGGER_SQL = """
CREATE TRIGGER memory_chunks_search_vector_trigger
BEFORE INSERT OR UPDATE OF content, summary ON memory_chunks
FOR EACH ROW EXECUTE FUNCTION memory_chunks_search_vector_update()
"""


async def create_memory_chunks(store: DataStore) -> None:
    """
    create memory_chunks table with indexes and FTS trigger.

    :param store: DataStore bound to per-agent schema via search_path
    :ptype store: DataStore
    """
    log.info("creating memory_chunks table (v007)")
    await store.execute(_CREATE_MEMORY_CHUNKS_SQL)
    await store.execute(_CREATE_CHUNKS_USER_IDX_SQL)
    await store.execute(_CREATE_CHUNKS_MEDIA_IDX_SQL)
    await store.execute(_CREATE_CHUNKS_EMBEDDING_IDX_SQL)
    await store.execute(_CREATE_CHUNKS_SEARCH_VECTOR_IDX_SQL)
    await store.execute(_CREATE_CHUNKS_TRIGGER_FUNC_SQL)
    await store.execute(_DROP_CHUNKS_TRIGGER_SQL)
    await store.execute(_CREATE_CHUNKS_TRIGGER_SQL)
