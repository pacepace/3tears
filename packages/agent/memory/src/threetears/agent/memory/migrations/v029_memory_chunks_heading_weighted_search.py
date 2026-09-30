"""
agent-memory v029: a chunk's heading is its strongest search signal.

v007's trigger weighted ``content`` A and ``summary`` B and left
``heading_context`` out of ``search_vector`` altogether, so a query naming a
section only matched chunks whose body repeated the heading. The heading names
what the chunk is about, so it now carries weight A, the content B and the
summary C -- the weighting a consumer running these tables already uses. The
trigger also fires on a ``heading_context`` update, so an edited heading
re-indexes its chunk.

``CREATE OR REPLACE FUNCTION`` swaps the body in place; the trigger is dropped
and re-created under its v007 name so its ``UPDATE OF`` column list takes the
heading. Existing rows keep their old vector until v030 recomputes it -- that is
DML, so it runs in its own version.

Idempotent: replacing the function and re-creating the trigger give the same
result on every run.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "weight_memory_chunk_headings",
]

log = get_logger(__name__)


_CREATE_CHUNKS_TRIGGER_FUNC_SQL = """
CREATE OR REPLACE FUNCTION memory_chunks_search_vector_update()
RETURNS trigger AS $$
BEGIN
    NEW.search_vector :=
        setweight(to_tsvector('english', coalesce(NEW.heading_context, '')), 'A')
        || setweight(to_tsvector('english', coalesce(NEW.content, '')), 'B')
        || setweight(to_tsvector('english', coalesce(NEW.summary, '')), 'C');
    RETURN NEW;
END
$$ LANGUAGE plpgsql
"""

_DROP_CHUNKS_TRIGGER_SQL = "DROP TRIGGER IF EXISTS memory_chunks_search_vector_trigger ON memory_chunks"

_CREATE_CHUNKS_TRIGGER_SQL = """
CREATE TRIGGER memory_chunks_search_vector_trigger
BEFORE INSERT OR UPDATE OF content, summary, heading_context ON memory_chunks
FOR EACH ROW EXECUTE FUNCTION memory_chunks_search_vector_update()
"""


async def weight_memory_chunk_headings(store: DataStore) -> None:
    """index a chunk's heading at weight A, its content at B and its summary at C.

    :param store: DataStore bound to per-agent schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("weighting memory_chunks search by heading, content, summary (v029)")
    await store.execute(_CREATE_CHUNKS_TRIGGER_FUNC_SQL)
    await store.execute(_DROP_CHUNKS_TRIGGER_SQL)
    await store.execute(_CREATE_CHUNKS_TRIGGER_SQL)
