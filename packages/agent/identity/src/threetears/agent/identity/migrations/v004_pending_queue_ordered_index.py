"""
agent-identity v004: the pending queue's index holds its order.

:meth:`~threetears.agent.identity.collections.IdentityVersionsCollection.find_pending` reads a user's ``proposed``
versions oldest first: pinned ``(agent_id, user_id)`` under ``status = 'proposed'``, ordered ``date_created``.
v001's ``idx_identity_pending`` holds ``(agent_id, user_id)`` alone, so the read fetched every pending version of
the user and sorted them. ``idx_identity_pending_created`` holds ``(agent_id, user_id, date_created)`` over the same
``proposed`` rows, so the read walks it in order.

The new index has the old one's leading columns over the same rows (the same partial ``WHERE``), so it serves every
read the old one did: the old one is dropped in this same migration (covering-drop ruling, 2026-11-14), not held for
a contract step. Built before the drop, so no moment leaves the queue without an index. Idempotent throughout
(``IF NOT EXISTS`` / ``IF EXISTS``).
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "order_pending_queue_index",
]

log = get_logger(__name__)

_CREATE_PENDING_CREATED_IDX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_identity_pending_created "
    "ON identity_versions (agent_id, user_id, date_created) "
    "WHERE status = 'proposed'"
)

_DROP_PENDING_IDX_SQL = "DROP INDEX IF EXISTS idx_identity_pending"


async def order_pending_queue_index(store: DataStore) -> None:
    """build the ordered pending-queue index, then drop the one it covers.

    :param store: DataStore bound to per-agent schema via search_path
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("ordering the identity pending-queue index (v004)")
    await store.execute(_CREATE_PENDING_CREATED_IDX_SQL)
    await store.execute(_DROP_PENDING_IDX_SQL)
