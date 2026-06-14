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
            "idx_scheduled_jobs_next_fire",
            "next_fire_at",
            postgresql_where=sa_text("status = 'active' AND next_fire_at IS NOT NULL"),
        ),
        Index(
            "idx_scheduled_jobs_partition_status",
            "partition_key",
            "status",
        ),
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
        Index(
            "idx_job_fires_job_time",
            "job_id",
            sa_text("actual_fired_at DESC"),
        ),
        Index(
            "idx_job_fires_partition_time",
            "partition_key",
            sa_text("actual_fired_at DESC"),
        ),
    )
