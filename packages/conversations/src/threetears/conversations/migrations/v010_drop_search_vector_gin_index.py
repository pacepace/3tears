"""
conversations v010: drop the GIN index over ``conversations.search_vector``.

YugabyteDB implements ``USING gin`` as ``ybgin``, which refuses any scan that
needs more than one required entry. :meth:`ConversationsCollection.search`
keeps the user's OR / NOT / phrase syntax (``websearch_to_tsquery``), which
needs several, so it now renders its keyword predicate through
:func:`threetears.core.data.gin_filter`: ``(<predicate>) IS TRUE``, which the
planner never matches to the GIN index. The rows are narrowed by
``agent_id`` / ``user_id`` (``idx_conv_user``) and the keyword test filters
that slice.

That leaves ``idx_conversations_search_vector`` (v005) with no reader. It
still costs a posting-list write on every INSERT and on every UPDATE of
``name`` or ``language`` (the trigger rewrites ``search_vector`` on both),
plus its storage, so this migration drops it.

The ``search_vector`` column and its maintenance trigger stay: the filter and
the ``ts_rank_cd`` ranking still read them.

idempotent: ``DROP INDEX IF EXISTS`` makes replay a no-op, and it is
search-path-relative like every other statement in this package.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "drop_search_vector_gin_index",
]

log = get_logger(__name__)


_DROP_INDEX_SQL = "DROP INDEX IF EXISTS idx_conversations_search_vector"


async def drop_search_vector_gin_index(store: DataStore) -> None:
    """
    drop the ``idx_conversations_search_vector`` GIN index.

    :param store: DataStore bound to per-agent schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("dropping conversations GIN index idx_conversations_search_vector (v010)")
    await store.execute(_DROP_INDEX_SQL)
