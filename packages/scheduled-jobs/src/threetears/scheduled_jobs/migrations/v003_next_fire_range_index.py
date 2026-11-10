"""scheduled-jobs v003: the tick's due scan reads a range-sharded index, built beside the old one, which is then dropped.

``idx_scheduled_jobs_next_fire`` (v001) indexes ``next_fire_at`` with no order declared, so YugabyteDB hashes it:
the due scan (:meth:`~threetears.scheduled_jobs.collections.ScheduledJobCollection.list_due_for_tick`,
``next_fire_at <= $1 AND kind = ANY($2) ORDER BY next_fire_at LIMIT $3``, every tick) seeks no range there and
reads and sorts every active job. ``idx_scheduled_jobs_next_fire_asc`` declares the column ``ASC``, so YugabyteDB
range-shards it and a tick reads about its page; Postgres reads both the same way. The new index is built first, so the scan always has
one, then the old is dropped.

Every statement is guarded, so re-running it is a no-op.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["range_shard_next_fire_index"]

log = get_logger(__name__)

_NEXT_FIRE_ASC_INDEX_SQL = (
    "-- purpose: packages/scheduled-jobs/src/threetears/scheduled_jobs/collections.py::ScheduledJobCollection.list_due_for_tick\n"
    "-- plan_test: packages/scheduled-jobs/tests/integration/test_index_plans.py::test_scheduled_jobs_next_fire\n"
    "CREATE INDEX IF NOT EXISTS idx_scheduled_jobs_next_fire_asc "
    "ON scheduled_jobs (next_fire_at ASC) "
    "WHERE status = 'active' AND next_fire_at IS NOT NULL"
)

_DROP_NEXT_FIRE_INDEX_SQL = "DROP INDEX IF EXISTS idx_scheduled_jobs_next_fire"


async def range_shard_next_fire_index(store: DataStore) -> None:
    """Build ``idx_scheduled_jobs_next_fire_asc``, then drop ``idx_scheduled_jobs_next_fire``.

    :param store: the migration's data store
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    await store.execute(_NEXT_FIRE_ASC_INDEX_SQL)
    await store.execute(_DROP_NEXT_FIRE_INDEX_SQL)
    log.info("scheduled_jobs v003: the due scan's index range-sharded")
