"""agent-skills v003: drop the GIN indexes on ``agent_skills``.

YugabyteDB implements ``USING gin`` as ``ybgin``, which refuses any scan that
needs more than one required entry. Both GIN-indexed predicates on this table
can need several: the typed query (``search_vector @@ websearch_to_tsquery``,
which turns an "or" or a leading "-" into OR / NOT) and the tag filter
(``tags && $n``, array overlap, once it holds more than one tag).
``list_for_user`` and ``count_for_user`` now render both through
:func:`threetears.core.data.gin_filter`, which emits ``(<predicate>) IS TRUE``
so the planner never picks the GIN index: the rows are narrowed by the btree
``(agent_id, user_id, enabled)`` index and the predicate filters that slice.

That leaves both v001 GIN indexes with no reader. Each still costs a
posting-list write on every INSERT and on every UPDATE that touches the
indexed column (the trigger rewrites ``search_vector`` whenever a weighted
text field changes), plus its storage, so this migration drops them:

- ``idx_skills_search_vector`` over ``search_vector``;
- ``idx_skills_tags`` over ``tags``.

The ``search_vector`` column and its maintenance trigger stay: the filter and
the ``ts_rank_cd`` ranking still read them.

Idempotent: ``DROP INDEX IF EXISTS`` makes replay a no-op.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["drop_gin_indexes"]

log = get_logger(__name__)


_DROP_SEARCH_VECTOR_INDEX_SQL = "DROP INDEX IF EXISTS idx_skills_search_vector"

_DROP_TAGS_INDEX_SQL = "DROP INDEX IF EXISTS idx_skills_tags"


async def drop_gin_indexes(store: DataStore) -> None:
    """drop the ``search_vector`` and ``tags`` GIN indexes on ``agent_skills``.

    :param store: DataStore bound to per-agent schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("dropping agent_skills GIN indexes idx_skills_search_vector, idx_skills_tags (v003)")
    await store.execute(_DROP_SEARCH_VECTOR_INDEX_SQL)
    await store.execute(_DROP_TAGS_INDEX_SQL)
