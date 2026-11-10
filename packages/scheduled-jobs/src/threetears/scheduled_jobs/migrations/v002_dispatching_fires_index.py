"""scheduled-jobs v002: the fires still in flight are found by an index of their own.

The tick's reaper fails every fire left ``'dispatching'`` past its age
(:meth:`~threetears.scheduled_jobs.collections.JobFireCollection.reap_stale_dispatching`).
``job_fires`` keeps every fire ever run and had no index on ``status``, so each sweep read the whole
table to find the few in flight. ``idx_job_fires_dispatching`` holds only the in-flight rows.

Every statement is guarded, so re-running it is a no-op.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["add_dispatching_fires_index"]

log = get_logger(__name__)

_DISPATCHING_INDEX_SQL = (
    "-- purpose: packages/scheduled-jobs/src/threetears/scheduled_jobs/collections.py"
    "::JobFireCollection.reap_stale_dispatching\n"
    "-- plan_test: packages/scheduled-jobs/tests/integration/test_index_plans.py::test_job_fires_dispatching\n"
    "CREATE INDEX IF NOT EXISTS idx_job_fires_dispatching "
    "ON job_fires (status, actual_fired_at) WHERE status = 'dispatching'"
)


async def add_dispatching_fires_index(store: DataStore) -> None:
    """Create ``idx_job_fires_dispatching``.

    :param store: the migration's data store
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("scheduled_jobs v002: the fires in flight are indexed")
    await store.execute(_DISPATCHING_INDEX_SQL)
