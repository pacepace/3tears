"""agent-wake v011: the fires still in flight are found by an index of their own.

The tick's reaper fails every fire left ``'dispatching'`` past its age
(:meth:`~threetears.agent.wake.collections.WakeFireCollection.reap_stale_dispatching`).
``wake_fires`` keeps every fire ever run and had no index on ``status``, so each sweep read the
whole table to find the few in flight. ``idx_wake_fires_dispatching`` holds only the in-flight
rows, so the sweep reads those and nothing else.

Every statement is guarded, so re-running it is a no-op.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["add_dispatching_fires_index"]

log = get_logger(__name__)

_DISPATCHING_INDEX_SQL = (
    "-- purpose: packages/agent/wake/src/threetears/agent/wake/collections.py"
    "::WakeFireCollection.reap_stale_dispatching\n"
    "-- plan_test: packages/agent/wake/tests/integration/test_index_plans.py::test_wake_fires_dispatching\n"
    "-- reason: wake_fires carries five indexes: the reaper's in-flight sweep and the agent and user fire counts\n"
    "CREATE INDEX IF NOT EXISTS idx_wake_fires_dispatching "
    "ON wake_fires (status, actual_fired_at) WHERE status = 'dispatching'"
)


async def add_dispatching_fires_index(store: DataStore) -> None:
    """Create ``idx_wake_fires_dispatching``.

    :param store: ``DataStore`` bound to the target agent schema via ``search_path``
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("agent_wake v011: the fires in flight are indexed")
    await store.execute(_DISPATCHING_INDEX_SQL)
