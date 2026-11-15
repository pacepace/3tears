"""scheduled-jobs v006: one job's fires are indexed in their order completed by the key.

One job's history (:meth:`~threetears.scheduled_jobs.collections.JobFireCollection.list_for_job`) reads its newest
fires first through :func:`threetears.core.keyed.keyset_page`, ordered ``actual_fired_at DESC, fire_id DESC``:
``fire_id`` (the table's ``UNIQUE`` key) makes the order total, so fires at the same instant come in one order on
every read. v001's ``idx_job_fires_job_time`` holds ``(job_id, actual_fired_at DESC)`` alone, so a page ordered by both
reads and sorts every fire of the job (YugabyteDB has no incremental sort). ``idx_job_fires_job_time_key`` holds
``(job_id, actual_fired_at DESC, fire_id DESC)``.

The new index leads with the old one's columns over the same rows, so it serves every read the old one did (the
latest fire of each listed job, ``latest_for_jobs``; a pod of the previous release's ``ORDER BY actual_fired_at DESC``
page mid-rollout): the old one is dropped in this same migration (covering-drop ruling, 2026-11-14), after the new one
is built, as v003 and v004 did. One index, not a second beside it (coordinator ruling 2026-11-14). Every statement is
guarded, so re-running it is a no-op.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["add_job_fires_time_key_index"]

log = get_logger(__name__)

_JOB_TIME_KEY_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_job_fires_job_time_key ON job_fires (job_id, actual_fired_at DESC, fire_id DESC)"
)

_DROP_JOB_TIME_INDEX_SQL = "DROP INDEX IF EXISTS idx_job_fires_job_time"


async def add_job_fires_time_key_index(store: DataStore) -> None:
    """Build ``idx_job_fires_job_time_key``, then drop ``idx_job_fires_job_time``.

    :param store: the migration's data store
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    await store.execute(_JOB_TIME_KEY_INDEX_SQL)
    await store.execute(_DROP_JOB_TIME_INDEX_SQL)
    log.info("scheduled_jobs v006: one job's fires are indexed in their order completed by the key")
