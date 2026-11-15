"""scheduled-jobs v005 (CONTRACT, held): the two indexes no read uses are dropped.

- ``idx_scheduled_jobs_partition_status`` (v001, ``(partition_key, status)``): no read pins a partition's jobs of a
  status. The due scan reads ``idx_scheduled_jobs_next_fire_key`` (v004), the admin listing pins no partition, and
  every other read of ``scheduled_jobs`` is a point read by its whole key.
- ``idx_job_fires_partition_time`` (v001, ``(partition_key, actual_fired_at DESC)``): no read orders a partition's
  fires by time. One job's history (``list_for_job``) pins the job and reads ``idx_job_fires_job_time`` (v006:
  ``idx_job_fires_job_time_key``); the latest fire of each listed job reads the same index.

Found unread by the index tool's run with the hub as consumer (o96, 2026-11-14); dropped under Pace's rule (b)
(2026-11-13: a non-unique index no read uses). A pod of an older release may still plan through a dropped index
mid-rollout, so the step is held: :func:`~threetears.scheduled_jobs.migrations.register` attaches it only when asked
(``held_contract=True``: the test schemas now, every deployment from the release after this one).

``DROP INDEX IF EXISTS``, so re-running it is a no-op.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = ["drop_unread_scheduled_job_indexes"]

log = get_logger(__name__)

_DROP_PARTITION_STATUS_INDEX_SQL = "DROP INDEX IF EXISTS idx_scheduled_jobs_partition_status"

_DROP_PARTITION_TIME_INDEX_SQL = "DROP INDEX IF EXISTS idx_job_fires_partition_time"


async def drop_unread_scheduled_job_indexes(store: DataStore) -> None:
    """Drop ``idx_scheduled_jobs_partition_status`` and ``idx_job_fires_partition_time`` (v005, contract).

    :param store: the migration's data store
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    await store.execute(_DROP_PARTITION_STATUS_INDEX_SQL)
    await store.execute(_DROP_PARTITION_TIME_INDEX_SQL)
    log.info(
        "scheduled_jobs v005 (contract): idx_scheduled_jobs_partition_status and idx_job_fires_partition_time dropped"
    )
