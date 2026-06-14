"""scheduled-jobs v001: create ``scheduled_jobs`` + ``job_fires`` + indexes.

The default store's two tables. Partition column is ``partition_key`` (a
plain denormalised UUID with no FK -- a consumer's partition referent is
its own concern). Composite primary keys ``(partition_key, job_id)`` and
``(partition_key, fire_id)``. Standalone ``UNIQUE (job_id)`` lets
``job_fires.job_id`` reference the bare column without partition
knowledge.

``scheduled_jobs`` columns:

- ``kind`` TEXT -- opaque routing discriminator (the consumer's job type).
- ``payload`` JSONB -- opaque per-job payload (never inspected by the
  platform).
- ``schedule_type`` TEXT -- enum-by-app (no DB CHECK; app-evolvable, the
  consumer validates).
- ``schedule_config`` JSONB -- per-schedule-type config.
- ``status`` TEXT CHECK ``('active', 'paused', 'expired')``.
- ``missed_fire_policy`` TEXT CHECK ``('coalesce', 'catch_up')``.

``job_fires`` columns:

- ``status`` TEXT CHECK ``('dispatching', 'succeeded', 'failed')``.
- ``output`` JSONB -- opaque captured output payload.
- FK ``job_id REFERENCES scheduled_jobs(job_id) ON DELETE CASCADE`` --
  deleting a job removes its fire history.

Every statement is idempotent (``CREATE TABLE IF NOT EXISTS`` /
``CREATE INDEX IF NOT EXISTS``) so re-running on a schema that already
has the tables is a no-op.

Anti-pattern reminders (mirrors agent-wake v001/v002):

- No ``gen_random_uuid()`` default on the id columns -- UUIDs are uuid7
  allocated app-side via ``uuid_utils.uuid7()``.
- No CHECK on ``payload`` / ``schedule_config`` shape -- the JSONB shape
  varies per consumer / ``schedule_type``.
"""

from __future__ import annotations

from threetears.core.data.store import DataStore
from threetears.observe import get_logger

__all__ = [
    "create_scheduled_jobs",
]

log = get_logger(__name__)


_CREATE_SCHEDULED_JOBS_SQL = """
CREATE TABLE IF NOT EXISTS scheduled_jobs (
    partition_key       UUID         NOT NULL,
    job_id              UUID         NOT NULL,
    kind                TEXT         NOT NULL,
    payload             JSONB        NOT NULL DEFAULT '{}'::jsonb,
    schedule_type       TEXT         NOT NULL,
    schedule_config     JSONB        NOT NULL DEFAULT '{}'::jsonb,
    status              TEXT         NOT NULL DEFAULT 'active',
    next_fire_at        TIMESTAMPTZ,
    last_fired_at       TIMESTAMPTZ,
    missed_fire_policy  TEXT         NOT NULL DEFAULT 'coalesce',
    name                TEXT,
    date_created        TIMESTAMPTZ  NOT NULL DEFAULT now(),
    date_updated        TIMESTAMPTZ  NOT NULL DEFAULT now(),
    PRIMARY KEY (partition_key, job_id),
    UNIQUE (job_id),
    CONSTRAINT scheduled_jobs_status_check
        CHECK (status IN ('active', 'paused', 'expired')),
    CONSTRAINT scheduled_jobs_missed_fire_policy_check
        CHECK (missed_fire_policy IN ('coalesce', 'catch_up'))
)
"""


# Partial index on the tick-engine's hot query path
# (``WHERE status = 'active' AND next_fire_at <= now``). Partial cuts the
# index footprint because expired one-shots accumulate.
_CREATE_NEXT_FIRE_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_scheduled_jobs_next_fire "
    "ON scheduled_jobs (next_fire_at) "
    "WHERE status = 'active' AND next_fire_at IS NOT NULL"
)


# Index for per-partition admin/list views.
_CREATE_PARTITION_STATUS_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_scheduled_jobs_partition_status ON scheduled_jobs (partition_key, status)"
)


_CREATE_JOB_FIRES_SQL = """
CREATE TABLE IF NOT EXISTS job_fires (
    partition_key     UUID         NOT NULL,
    fire_id           UUID         NOT NULL,
    job_id            UUID         NOT NULL,
    scheduled_fire_at TIMESTAMPTZ  NOT NULL,
    actual_fired_at   TIMESTAMPTZ  NOT NULL DEFAULT now(),
    status            TEXT         NOT NULL,
    output            JSONB,
    latency_ms        INTEGER,
    error             TEXT,
    date_created      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    PRIMARY KEY (partition_key, fire_id),
    UNIQUE (fire_id),
    CONSTRAINT job_fires_job_fk
        FOREIGN KEY (job_id) REFERENCES scheduled_jobs(job_id)
        ON DELETE CASCADE,
    CONSTRAINT job_fires_status_check
        CHECK (status IN ('dispatching', 'succeeded', 'failed'))
)
"""


# Per-job history index (descending time for "latest fire" hot path).
_CREATE_JOB_TIME_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_job_fires_job_time ON job_fires (job_id, actual_fired_at DESC)"
)


# Per-partition history index (descending time).
_CREATE_PARTITION_TIME_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_job_fires_partition_time ON job_fires (partition_key, actual_fired_at DESC)"
)


async def create_scheduled_jobs(store: DataStore) -> None:
    """Create the ``scheduled_jobs`` + ``job_fires`` tables + indexes.

    ``scheduled_jobs`` is created first so ``job_fires``'s FK on
    ``scheduled_jobs(job_id)`` resolves.

    :param store: ``DataStore`` bound to the target schema via
        ``search_path``
    :ptype store: DataStore
    :return: nothing
    :rtype: None
    """
    log.info("creating scheduled_jobs + job_fires tables (v001)")
    await store.execute(_CREATE_SCHEDULED_JOBS_SQL)
    await store.execute(_CREATE_NEXT_FIRE_INDEX_SQL)
    await store.execute(_CREATE_PARTITION_STATUS_INDEX_SQL)
    await store.execute(_CREATE_JOB_FIRES_SQL)
    await store.execute(_CREATE_JOB_TIME_INDEX_SQL)
    await store.execute(_CREATE_PARTITION_TIME_INDEX_SQL)
