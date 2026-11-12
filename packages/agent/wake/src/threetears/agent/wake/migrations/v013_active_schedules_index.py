"""agent-wake v013: a conversation's active schedules have an index of their own, partial on ``status = 'active'``.

``idx_wake_schedules_conv_status`` (v001) indexes ``(conversation_id, status)`` over every schedule a conversation ever
had. The cap check counts a conversation's active ones
(:meth:`~threetears.agent.wake.collections.WakeScheduleCollection.count_active_for_conversation`) and the admin view
lists them (:meth:`~threetears.agent.wake.collections.WakeScheduleCollection.list_active_for_conversation`): on
YugabyteDB the count reads this partial index alone (an index-only scan of the active rows); on PostgreSQL both read
it. YugabyteDB lists through the conversation's own primary key prefix (its cost model prices that range below the
index and a lookup per row; a two-step read of the keys first measured about twice as slow on both engines). Once
every pod runs this version, v001's index serves nothing; its drop waits for the held CONTRACT migration.

The statement is guarded, so re-running it is a no-op.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["add_active_schedules_index"]

log = get_logger(__name__)

_ACTIVE_SCHEDULES_INDEX_SQL = (
    "-- purpose: packages/agent/wake/src/threetears/agent/wake/collections.py::WakeScheduleCollection.list_active_for_conversation; packages/agent/wake/src/threetears/agent/wake/collections.py::WakeScheduleCollection.count_active_for_conversation\n"
    "-- plan_test: packages/agent/wake/tests/integration/test_index_plans.py::test_wake_schedules_conv_active\n"
    "-- reason: agent_wake_schedules carries five indexes: one conversation's active schedules, for its cap\n"
    "CREATE INDEX IF NOT EXISTS idx_wake_schedules_conv_active "
    "ON agent_wake_schedules (conversation_id) "
    "WHERE status = 'active'"
)


async def add_active_schedules_index(store: DataStore) -> None:
    """Build ``idx_wake_schedules_conv_active``.

    :param store: the migration's data store
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    await store.execute(_ACTIVE_SCHEDULES_INDEX_SQL)
    log.info("agent_wake v013: a conversation's active schedules indexed")
