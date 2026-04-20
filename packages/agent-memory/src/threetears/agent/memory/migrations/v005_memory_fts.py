"""
agent-memory v005: add full-text search vector to memories.

memory-task-01. ``retrieval.py`` and ``tools.py`` run FTS queries of
the shape::

    SELECT ..., ts_rank_cd(search_vector, websearch_to_tsquery('english', $1)) AS fts_rank
      FROM memories
     WHERE ... AND search_vector @@ websearch_to_tsquery('english', $1)

This migration creates the column, the GIN index, and a trigger that
keeps ``search_vector`` current with ``content`` + ``summary`` on
INSERT and UPDATE.

Idempotent: column / index creates use IF NOT EXISTS; trigger function
uses CREATE OR REPLACE; trigger itself is dropped + recreated so replay
is a no-op.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "add_memory_fts",
]

log = get_logger(__name__)


_ADD_SEARCH_VECTOR_SQL = (
    "ALTER TABLE memories ADD COLUMN IF NOT EXISTS search_vector TSVECTOR"
)

_CREATE_SEARCH_VECTOR_IDX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_mem_search_vector "
    "ON memories USING GIN (search_vector)"
)

_CREATE_TRIGGER_FUNC_SQL = """
CREATE OR REPLACE FUNCTION memories_search_vector_update()
RETURNS trigger AS $$
BEGIN
    NEW.search_vector :=
        setweight(to_tsvector('english', coalesce(NEW.content, '')), 'A')
        || setweight(to_tsvector('english', coalesce(NEW.summary, '')), 'B');
    RETURN NEW;
END
$$ LANGUAGE plpgsql
"""

_DROP_TRIGGER_SQL = (
    "DROP TRIGGER IF EXISTS memories_search_vector_trigger ON memories"
)

_CREATE_TRIGGER_SQL = """
CREATE TRIGGER memories_search_vector_trigger
BEFORE INSERT OR UPDATE OF content, summary ON memories
FOR EACH ROW EXECUTE FUNCTION memories_search_vector_update()
"""

_BACKFILL_SEARCH_VECTOR_SQL = """
UPDATE memories
   SET search_vector =
        setweight(to_tsvector('english', coalesce(content, '')), 'A')
        || setweight(to_tsvector('english', coalesce(summary, '')), 'B')
 WHERE search_vector IS NULL
"""


async def add_memory_fts(store: DataStore) -> None:
    """
    add search_vector column, GIN index, and maintenance trigger.

    :param store: DataStore bound to per-agent schema via search_path
    :ptype store: DataStore
    """
    log.info("adding memory FTS vector + trigger (v005)")
    await store.execute(_ADD_SEARCH_VECTOR_SQL)
    await store.execute(_CREATE_SEARCH_VECTOR_IDX_SQL)
    await store.execute(_CREATE_TRIGGER_FUNC_SQL)
    await store.execute(_DROP_TRIGGER_SQL)
    await store.execute(_CREATE_TRIGGER_SQL)
    await store.execute(_BACKFILL_SEARCH_VECTOR_SQL)
