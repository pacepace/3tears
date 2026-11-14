"""agent-intention v002 (CONTRACT, held): v001's ``idx_intentions_last_surfaced`` is dropped.

``idx_intentions_last_surfaced`` (``(agent_id, last_surfaced_at)``) is read by nothing: the deliberation read
(:meth:`~threetears.agent.intention.collections.IntentionsCollection.find_open_for_deliberation`) pins the agent and
the user, which ``idx_intentions_open_ranked`` holds, and filters the cooldown over that one person's open wants. A pod
of an older release may still plan through it mid-rollout, so the step is held:
:func:`~threetears.agent.intention.migrations.register` attaches it only when asked (``held_contract=True``: the test
schemas now, every deployment from the release after this one).

``DROP INDEX IF EXISTS``, so re-running it is a no-op.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["drop_last_surfaced_index"]

log = get_logger(__name__)

_DROP_LAST_SURFACED_INDEX_SQL = "DROP INDEX IF EXISTS idx_intentions_last_surfaced"


async def drop_last_surfaced_index(store: DataStore) -> None:
    """Drop ``idx_intentions_last_surfaced`` (v002, contract).

    :param store: ``DataStore`` bound to the target agent schema via ``search_path``
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    await store.execute(_DROP_LAST_SURFACED_INDEX_SQL)
    log.info("intentions v002 (contract): idx_intentions_last_surfaced dropped")
