"""agent-wake v012: the tick's due scan reads a range-sharded index, built beside the old one, which is then dropped.

``idx_wake_schedules_next_fire`` (v001) indexes ``next_fire_at`` with no order declared, so YugabyteDB hashes it:
the due scan (:meth:`~threetears.agent.wake.collections.WakeScheduleCollection.list_due_for_tick`,
``next_fire_at <= $1 ORDER BY next_fire_at LIMIT $2``, every tick) seeks no range there and reads and sorts every
active schedule. ``idx_wake_schedules_next_fire_asc`` declares the column ``ASC``, so YugabyteDB range-shards it and
a tick reads about its page; Postgres reads both the same way. The new index is built first, so the scan always has
one, then the old is dropped.

Every statement is guarded, so re-running it is a no-op.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["range_shard_next_fire_index"]

log = get_logger(__name__)

_NEXT_FIRE_ASC_INDEX_SQL = (
    "-- purpose: packages/agent/wake/src/threetears/agent/wake/collections.py::WakeScheduleCollection.list_due_for_tick\n"
    "-- plan_test: packages/agent/wake/tests/integration/test_index_plans.py::test_wake_schedules_next_fire\n"
    "-- reason: agent_wake_schedules carries five indexes: the tick's due scan, the agent's and the user's reads\n"
    "CREATE INDEX IF NOT EXISTS idx_wake_schedules_next_fire_asc "
    "ON agent_wake_schedules (next_fire_at ASC) "
    "WHERE status = 'active' AND next_fire_at IS NOT NULL"
)

_DROP_NEXT_FIRE_INDEX_SQL = "DROP INDEX IF EXISTS idx_wake_schedules_next_fire"


async def range_shard_next_fire_index(store: DataStore) -> None:
    """Build ``idx_wake_schedules_next_fire_asc``, then drop ``idx_wake_schedules_next_fire``.

    :param store: the migration's data store
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    await store.execute(_NEXT_FIRE_ASC_INDEX_SQL)
    await store.execute(_DROP_NEXT_FIRE_INDEX_SQL)
    log.info("agent_wake v012: the due scan's index range-sharded")
