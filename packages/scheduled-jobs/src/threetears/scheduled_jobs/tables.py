"""SQLAlchemy ``Table`` factories for the default-store L3 tables.

Cross-package consumers register the scheduled-jobs tables on their own
SQLAlchemy ``MetaData`` -- the L1 SQLite cache builds from that metadata,
and Alembic ``target_metadata`` reflection sees the same shape for
auto-generate. The factories here mirror agent-wake's
``agent_wake_schedules_table`` / ``wake_fires_table`` pattern so
consumers follow the established convention instead of hand-rolling the
``Table`` shape.

**Why standalone ``Table(...)`` and not a schema-backed delegate.** The
collections hand-roll their SQL in the v001 migration and are not
``SchemaBackedCollection`` subclasses, because the framework's
``TableSchema`` / ``to_sqlalchemy_table`` type system has no descriptor
for the CHECK constraints these tables carry (the ``status`` /
``schedule_type`` / ``missed_fire_policy`` enum-by-app checks) nor for
the JSONB columns. The factories construct the ``Table`` directly,
matching the canonical migration DDL.

**Registration order.** ``job_fires`` carries a composite-target FK on
``scheduled_jobs(job_id)`` (``job_fires_job_fk``). SQLAlchemy resolves
cross-``Table`` references lazily at DDL-emit / reflection time, but the
FK target table must be present on the same metadata before the
referencing table's FK can be emitted. Register the parent before the
child: :func:`scheduled_jobs_table` -> :func:`job_fires_table`.

**No ``partition_key`` FK.** ``partition_key`` is a denormalised UUID
with no FK constraint -- a consumer's partition referent is its own
concern (the table is payload-agnostic and cannot know what the
partition addresses).
"""

from __future__ import annotations

from sqlalchemy import (
    CheckConstraint,
    Column,
    ForeignKeyConstraint,
    Index,
    Integer,
    MetaData,
    PrimaryKeyConstraint,
    Table,
    Text,
    UniqueConstraint,
    text as sa_text,
)
from sqlalchemy import (
    DateTime as SADateTime,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PgUUID

from threetears.core.collections.schema_backed import register_sa_table_indexes
from threetears.core.eviction import Declared, Policy
from threetears.core.cache.l1_lru import L1_EVICTION_INFO

__all__ = [
    "job_fires_table",
    "scheduled_jobs_table",
]


def scheduled_jobs_table(metadata: MetaData) -> Table:
    """Register the ``scheduled_jobs`` table on the given SA metadata.

    Mirrors the canonical v001 migration DDL exactly: composite primary
    key ``(partition_key, job_id)``, standalone ``UNIQUE (job_id)`` so
    ``job_fires`` can reference the bare column, the JSONB ``payload`` /
    ``schedule_config`` columns, the two CHECK constraints (``status`` /
    ``missed_fire_policy`` enum-by-app), every NOT NULL DEFAULT, and the
    two indexes (one partial).

    There is no DB CHECK on ``schedule_type`` (app-evolvable; validated by
    the consumer) nor on the JSONB shapes -- the factory matches the
    migration in omitting them.

    Idempotent: returns the existing :class:`Table` if one with this name
    is already registered on ``metadata``.

    :param metadata: SQLAlchemy metadata to attach the table to
    :ptype metadata: MetaData
    :return: the ``scheduled_jobs`` :class:`Table`
    :rtype: Table
    """
    if "scheduled_jobs" in metadata.tables:
        return metadata.tables["scheduled_jobs"]
    return Table(
        "scheduled_jobs",
        metadata,
        Column("partition_key", PgUUID(as_uuid=True), nullable=False),
        Column("job_id", PgUUID(as_uuid=True), nullable=False),
        Column("kind", Text(), nullable=False),
        Column(
            "payload",
            JSONB(),
            nullable=False,
            server_default=sa_text("'{}'::jsonb"),
        ),
        Column("schedule_type", Text(), nullable=False),
        Column(
            "schedule_config",
            JSONB(),
            nullable=False,
            server_default=sa_text("'{}'::jsonb"),
        ),
        Column(
            "status",
            Text(),
            nullable=False,
            server_default=sa_text("'active'"),
        ),
        Column("next_fire_at", SADateTime(timezone=True), nullable=True),
        Column("last_fired_at", SADateTime(timezone=True), nullable=True),
        Column(
            "missed_fire_policy",
            Text(),
            nullable=False,
            server_default=sa_text("'coalesce'"),
        ),
        Column("name", Text(), nullable=True),
        Column(
            "date_created",
            SADateTime(timezone=True),
            nullable=False,
            server_default=sa_text("now()"),
        ),
        Column(
            "date_updated",
            SADateTime(timezone=True),
            nullable=False,
            server_default=sa_text("now()"),
        ),
        PrimaryKeyConstraint("partition_key", "job_id"),
        UniqueConstraint("job_id"),
        CheckConstraint(
            "status IN ('active', 'paused', 'expired')",
            name="scheduled_jobs_status_check",
        ),
        CheckConstraint(
            "missed_fire_policy IN ('coalesce', 'catch_up')",
            name="scheduled_jobs_missed_fire_policy_check",
        ),
        Index(
            "idx_scheduled_jobs_next_fire_key",
            sa_text("next_fire_at ASC"),
            sa_text("job_id ASC"),
            postgresql_where=sa_text("status = 'active' AND next_fire_at IS NOT NULL"),
            info={
                "purpose": "packages/scheduled-jobs/src/threetears/scheduled_jobs/collections.py::ScheduledJobCollection.list_due_for_tick",
                "plan_test": "packages/scheduled-jobs/tests/integration/test_index_plans.py::test_scheduled_jobs_next_fire",
            },
        ),
        # v007: the admin listing's order completed by the key, then the status a filter checks on the entry
        Index(
            "idx_scheduled_jobs_kind_name_key",
            sa_text("kind ASC"),
            sa_text("name ASC NULLS LAST"),
            sa_text("job_id ASC"),
            sa_text("status ASC"),
            info={
                "purpose": (
                    "packages/scheduled-jobs/src/threetears/scheduled_jobs/collections.py::ScheduledJobCollection.list_jobs; "
                    "packages/scheduled-jobs/src/threetears/scheduled_jobs/collections.py::ScheduledJobCollection.count_jobs"
                ),
                "plan_test": "packages/scheduled-jobs/tests/integration/test_index_plans.py::test_scheduled_jobs_kind_name",
            },
        ),
        info={
            L1_EVICTION_INFO: Declared(Policy.EVICTABLE, "rows cached on a read by key; a miss reads L2/L3 and refills")
        },
    )


def job_fires_table(metadata: MetaData) -> Table:
    """Register the ``job_fires`` table on the given SA metadata.

    Mirrors the canonical v001 migration DDL: composite primary key
    ``(partition_key, fire_id)``, standalone ``UNIQUE (fire_id)``, the FK
    ``job_id REFERENCES scheduled_jobs(job_id) ON DELETE CASCADE``
    (``job_fires_job_fk``), the ``status`` CHECK, the JSONB ``output``
    column, the ``actual_fired_at`` / ``date_created`` server defaults,
    and the two indexes.

    Register :func:`scheduled_jobs_table` on the same metadata first so
    the FK resolves at DDL-emit / reflection time.

    Idempotent: returns the existing :class:`Table` if one with this name
    is already registered on ``metadata``.

    :param metadata: SQLAlchemy metadata to attach the table to
    :ptype metadata: MetaData
    :return: the ``job_fires`` :class:`Table`
    :rtype: Table
    """
    if "job_fires" in metadata.tables:
        return metadata.tables["job_fires"]
    return Table(
        "job_fires",
        metadata,
        Column("partition_key", PgUUID(as_uuid=True), nullable=False),
        Column("fire_id", PgUUID(as_uuid=True), nullable=False),
        Column("job_id", PgUUID(as_uuid=True), nullable=False),
        Column("scheduled_fire_at", SADateTime(timezone=True), nullable=False),
        Column(
            "actual_fired_at",
            SADateTime(timezone=True),
            nullable=False,
            server_default=sa_text("now()"),
        ),
        Column("status", Text(), nullable=False),
        Column("output", JSONB(), nullable=True),
        Column("latency_ms", Integer(), nullable=True),
        Column("error", Text(), nullable=True),
        Column(
            "date_created",
            SADateTime(timezone=True),
            nullable=False,
            server_default=sa_text("now()"),
        ),
        PrimaryKeyConstraint("partition_key", "fire_id"),
        UniqueConstraint("fire_id"),
        ForeignKeyConstraint(
            ["job_id"],
            ["scheduled_jobs.job_id"],
            ondelete="CASCADE",
            name="job_fires_job_fk",
        ),
        CheckConstraint(
            "status IN ('dispatching', 'succeeded', 'failed')",
            name="job_fires_status_check",
        ),
        # v006: one job's fires newest first, completed by the key (covers v001's idx_job_fires_job_time, dropped)
        Index(
            "idx_job_fires_job_time_key",
            "job_id",
            sa_text("actual_fired_at DESC"),
            sa_text("fire_id DESC"),
            info={
                "purpose": (
                    "packages/scheduled-jobs/src/threetears/scheduled_jobs/collections.py::JobFireCollection.latest_for_jobs; "
                    "packages/scheduled-jobs/src/threetears/scheduled_jobs/collections.py::JobFireCollection.list_for_job; "
                    # a job's delete reads the fires its ON DELETE CASCADE takes, by the job (DeleteCascade)
                    "packages/core/src/threetears/core/collections/delete_cascade.py::_children"
                ),
                "plan_test": "packages/scheduled-jobs/tests/integration/test_index_plans.py::test_job_fires_job_time",
            },
        ),
        # v002: the fires in flight, the reaper's sweep
        Index(
            "idx_job_fires_dispatching",
            "status",
            "actual_fired_at",
            postgresql_where=sa_text("status = 'dispatching'"),
            info={
                "purpose": "packages/scheduled-jobs/src/threetears/scheduled_jobs/collections.py::JobFireCollection.reap_stale_dispatching",
                "plan_test": "packages/scheduled-jobs/tests/integration/test_index_plans.py::test_job_fires_dispatching",
            },
        ),
        info={
            L1_EVICTION_INFO: Declared(Policy.EVICTABLE, "rows cached on a read by key; a miss reads L2/L3 and refills")
        },
    )


# the indexes and partial predicates these tables declare, registered with the keyed helpers on import, as a
# TableSchema registers its own: a keyed read of them names its serving index to YugabyteDB and writes a predicate's
# text as its literal (register_sa_table_indexes; 3tears #100: unregistered, YugabyteDB counted a conversation's 10
# active wakes through the primary key, 400 rows read, on 6 of 6 fresh ANALYZEs)
_REGISTERED = MetaData()
for _declare in (scheduled_jobs_table, job_fires_table):
    register_sa_table_indexes(_declare(_REGISTERED))
