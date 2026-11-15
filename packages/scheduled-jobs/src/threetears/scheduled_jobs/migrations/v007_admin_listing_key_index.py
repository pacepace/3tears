"""scheduled-jobs v007: the operator's listing of every schedule reads its pages off an index holding its order.

The admin listing (:meth:`~threetears.scheduled_jobs.collections.ScheduledJobCollection.list_jobs`) spans every
partition (each schedule carries its own ``partition_key``) and is ordered ``kind, name, job_id``: ``job_id`` (the
table's ``UNIQUE`` key) makes the order total. Until v007 no index held that order, so each page read and sorted every
schedule, and paging by ``OFFSET`` read every row before the page as well. ``idx_scheduled_jobs_kind_name_key`` holds
``(kind ASC, name ASC NULLS LAST, job_id ASC)``, ranged (range-sharded on YugabyteDB), so a page is a walk of the index
from its cursor, and a kind filter (``kind = ANY``) is a range of it. ``status`` rides last: a status filter is then
checked on the index entry, before the row is fetched; it never changes the order (``job_id`` is unique before it).

The bounded count beside the listing (:meth:`~threetears.scheduled_jobs.collections.ScheduledJobCollection.count_jobs`)
walks the same index in the same order, at most its cap plus one entry.

A new index, so nothing is dropped. Guarded, so re-running it is a no-op.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["add_admin_listing_key_index"]

log = get_logger(__name__)

_KIND_NAME_KEY_INDEX_SQL = (
    "-- purpose: packages/scheduled-jobs/src/threetears/scheduled_jobs/collections.py::ScheduledJobCollection.list_jobs; "
    "packages/scheduled-jobs/src/threetears/scheduled_jobs/collections.py::ScheduledJobCollection.count_jobs\n"
    "-- plan_test: packages/scheduled-jobs/tests/integration/test_index_plans.py::test_scheduled_jobs_kind_name\n"
    "CREATE INDEX IF NOT EXISTS idx_scheduled_jobs_kind_name_key "
    "ON scheduled_jobs (kind ASC, name ASC NULLS LAST, job_id ASC, status ASC)"
)


async def add_admin_listing_key_index(store: DataStore) -> None:
    """Build ``idx_scheduled_jobs_kind_name_key``.

    :param store: the migration's data store
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    await store.execute(_KIND_NAME_KEY_INDEX_SQL)
    log.info("scheduled_jobs v007: the admin listing's index holds its order")
