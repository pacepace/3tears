"""
agent-memory v030: recompute every chunk's ``search_vector`` under v029's weighting.

v029 changed how the trigger builds a chunk's vector; rows written before it
still carry v007's vector, which leaves the heading out. This sets each row's
vector to the same expression the trigger now computes. Setting
``search_vector`` directly does not fire the trigger (it fires on ``content``,
``summary`` and ``heading_context``), so the expression is written out here.

Replay guard: only a row whose vector differs from the recomputed one is
touched, so a second run updates nothing.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "recompute_memory_chunks_search_vector",
]

log = get_logger(__name__)


_RECOMPUTE_SEARCH_VECTOR_SQL = """
UPDATE memory_chunks
   SET search_vector =
        setweight(to_tsvector('english', coalesce(heading_context, '')), 'A')
        || setweight(to_tsvector('english', coalesce(content, '')), 'B')
        || setweight(to_tsvector('english', coalesce(summary, '')), 'C')
 WHERE search_vector IS DISTINCT FROM (
        setweight(to_tsvector('english', coalesce(heading_context, '')), 'A')
        || setweight(to_tsvector('english', coalesce(content, '')), 'B')
        || setweight(to_tsvector('english', coalesce(summary, '')), 'C')
       )
"""


async def recompute_memory_chunks_search_vector(store: DataStore) -> None:
    """rewrite each chunk's ``search_vector`` with the heading-weighted expression.

    :param store: DataStore bound to per-agent schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("recomputing memory_chunks.search_vector with heading weighting (v030)")
    await store.execute(_RECOMPUTE_SEARCH_VECTOR_SQL)
