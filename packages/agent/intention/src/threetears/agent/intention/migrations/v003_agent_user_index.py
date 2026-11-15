"""agent-intention v003 (expand): ``ix_intentions_agent_user``, a person's every want by key.

:meth:`~threetears.agent.intention.collections.IntentionsCollection.find_by_user` (every want of the person, whatever
its status) and :meth:`~threetears.agent.intention.collections.IntentionsCollection.find_similar_for_dedup` pin
``(agent_id, user_id)``; the only index leading with those columns, ``idx_intentions_open_ranked``, is partial (open
wants), so ``find_by_user`` read the agent's every row by the primary key's lead, every person of it. This index holds
the person's rows. It covers no other index (the partial one stays: it serves the deliberation read's order), so it
drops none. ``IF NOT EXISTS``, so re-running it is a no-op.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["add_agent_user_index"]

log = get_logger(__name__)

_AGENT_USER_INDEX_SQL = "CREATE INDEX IF NOT EXISTS ix_intentions_agent_user ON intentions (agent_id, user_id)"


async def add_agent_user_index(store: DataStore) -> None:
    """Build ``ix_intentions_agent_user`` (v003, expand).

    :param store: ``DataStore`` bound to the target agent schema via ``search_path``
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    await store.execute(_AGENT_USER_INDEX_SQL)
    log.info("intentions v003: a person's wants are read by ix_intentions_agent_user")
