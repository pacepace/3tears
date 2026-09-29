"""
agent-memory v027: drop the GIN indexes over ``search_vector``.

YugabyteDB implements ``USING gin`` as ``ybgin``, which refuses any scan that
needs more than one required entry, and the keyword predicates on these tables
(``search_vector @@ websearch_to_tsquery(...)``) need several whenever the text
carries an "or" or a leading "-". Every such predicate now goes through
:func:`threetears.core.data.gin_filter`, which renders ``(<predicate>) IS
TRUE`` so the planner never picks the GIN index: the rows are narrowed by the
btree scope columns (``agent_id``, ``user_id``) and the keyword test filters
that slice.

That leaves every GIN index over ``search_vector`` with no reader. Each still
costs a posting-list write on every INSERT and on every UPDATE that rewrites
the trigger-maintained tsvector, plus its storage, so this migration drops
them. Each table carries two copies under different names, and both go:

- ``memories``: ``idx_mem_search_vector`` (v005) and
  ``idx_memories_search_vector`` (v022).
- ``media_content``: ``idx_mc_search_vector`` (v006) and
  ``idx_media_content_search_vector`` (v022).
- ``memory_chunks``: ``idx_chunks_search_vector`` (v007) and
  ``idx_memory_chunks_search_vector`` (v022).

What stays:

- the ``search_vector`` columns and their maintenance triggers. The filters
  and the ``ts_rank_cd`` ranking still read them.
- ``idx_memories_tags`` (v025). ``tags @> $n::jsonb`` containment has one
  required entry, which ybgin serves, and ``RetrievalScope.tags_all`` reads
  it.

Idempotent: ``DROP INDEX IF EXISTS`` makes replay a no-op, and a schema that
never had one of the two copies (a hub-squashed schema, say) migrates the same
way.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "drop_search_vector_gin_indexes",
]

log = get_logger(__name__)


#: one statement per GIN index over a ``search_vector`` column this package ever
#: created, both copies per table.
_DROP_SEARCH_VECTOR_GIN_INDEXES_SQL: tuple[str, ...] = (
    "DROP INDEX IF EXISTS idx_mem_search_vector",
    "DROP INDEX IF EXISTS idx_memories_search_vector",
    "DROP INDEX IF EXISTS idx_mc_search_vector",
    "DROP INDEX IF EXISTS idx_media_content_search_vector",
    "DROP INDEX IF EXISTS idx_chunks_search_vector",
    "DROP INDEX IF EXISTS idx_memory_chunks_search_vector",
)


async def drop_search_vector_gin_indexes(store: DataStore) -> None:
    """drop every GIN index over ``search_vector`` on memories, media content and chunks.

    :param store: DataStore bound to per-agent schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("dropping search_vector GIN indexes on memories, media_content, memory_chunks (v027)")
    for statement in _DROP_SEARCH_VECTOR_GIN_INDEXES_SQL:
        await store.execute(statement)
