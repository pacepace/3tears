"""scheduled-jobs v004: the tick's due scan reads its page off an index holding its order completed by the key.

The due scan (:meth:`~threetears.scheduled_jobs.collections.ScheduledJobCollection.list_due_for_tick`) reads its first
page through :func:`threetears.core.keyed.keyset_page`, ordered ``next_fire_at, job_id``: ``job_id`` (the table's
``UNIQUE`` key) makes the order total, so jobs due at the same instant come in one order every tick.
``idx_scheduled_jobs_next_fire_asc`` (v003) holds ``next_fire_at`` alone, so a page ordered by both reads and sorts
every due active job (YugabyteDB has no incremental sort). ``idx_scheduled_jobs_next_fire_key`` holds both, ranged
``ASC`` (range-sharded on YugabyteDB), over the same active rows.

Built beside v003's ``idx_scheduled_jobs_next_fire_asc``, which is then dropped, as v003 did with v001's: the new index
leads with the same ``next_fire_at ASC`` over the same rows, so a pod still running the old ``ORDER BY next_fire_at``
statement mid-rollout reads it the same way. Leaving the old one would not be harmless: YugabyteDB's planner picks it
(smaller) and sorts every due job, as the plan test caught. Every statement is guarded, so re-running it is a no-op.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["add_next_fire_key_index"]

log = get_logger(__name__)

_NEXT_FIRE_KEY_INDEX_SQL = (
    "-- purpose: packages/scheduled-jobs/src/threetears/scheduled_jobs/collections.py::ScheduledJobCollection.list_due_for_tick\n"
    "-- plan_test: packages/scheduled-jobs/tests/integration/test_index_plans.py::test_scheduled_jobs_next_fire\n"
    "CREATE INDEX IF NOT EXISTS idx_scheduled_jobs_next_fire_key "
    "ON scheduled_jobs (next_fire_at ASC, job_id ASC) "
    "WHERE status = 'active' AND next_fire_at IS NOT NULL"
)

_DROP_NEXT_FIRE_ASC_INDEX_SQL = "DROP INDEX IF EXISTS idx_scheduled_jobs_next_fire_asc"


async def add_next_fire_key_index(store: DataStore) -> None:
    """Build ``idx_scheduled_jobs_next_fire_key``, then drop ``idx_scheduled_jobs_next_fire_asc``.

    :param store: the migration's data store
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    await store.execute(_NEXT_FIRE_KEY_INDEX_SQL)
    await store.execute(_DROP_NEXT_FIRE_ASC_INDEX_SQL)
    log.info("scheduled_jobs v004: the due scan's index holds its key")
