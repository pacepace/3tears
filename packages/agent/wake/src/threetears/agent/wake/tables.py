"""SQLAlchemy ``Table`` factories for the agent-wake L3 tables.

Cross-package consumers (metallm, future hosts) register the
agent-wake tables on their own SQLAlchemy ``MetaData`` -- the L1
SQLite cache builds from that metadata, and Alembic ``target_metadata``
reflection sees the same shape for auto-generate. The agent-memory
package exposes ``memories_table(metadata)`` etc. for exactly this and
the agent-skills package exposes ``agent_skills_table(metadata)`` /
``agent_skill_invocations_table(metadata)``; the factories here are the
agent-wake counterpart so consumers can follow the established pattern
instead of hand-rolling the ``Table`` shape.

**Why standalone ``Table(...)`` and not a schema-backed delegate.**
The agent-memory factories delegate to
``<Collection>.schema.to_sqlalchemy_table(metadata)`` because those
collections are :class:`SchemaBackedCollection` subclasses with a
``TableSchema`` single source of truth. The agent-wake collections
(like agent-skills) hand-roll their SQL in the v001/v002/v003
migrations and are not ``SchemaBackedCollection`` subclasses, because
the framework's ``TableSchema`` / ``to_sqlalchemy_table`` type system
has no descriptor for the CHECK constraints these tables carry (the
``status`` / ``execution_mode`` /
``missed_fire_policy`` enum-by-app checks, the ``wake_fires``
mutually-exclusive-source check, the ``verification_scheme``
slug-format regex check) nor for the JSONB / BYTEA columns. Adding
those to the core schema system is a framework change out of scope for
this package, so the factories below construct the ``Table`` directly.

**Drift protection.** A parallel hand-written DDL with no parity
guarantee would be the embedded-DDL-drift smell. The factory output is
pinned against the canonical migration DDL (the FINAL post-v005 shape:
v001-v003 create the tables, v004 extends the ``wake_fires.status``
CHECK to add ``'dispatching'``, v005 replaces the
``webhook_subscriptions.verification_scheme`` hardcoded-value CHECK
with a slug-format guard) by
``tests/integration/test_sqlalchemy_table_parity.py``: it applies the
migrations to one Postgres schema, emits each factory's ``CREATE
TABLE`` + indexes into a second schema, and asserts the two are
structurally identical (columns + types + nullability + server
defaults, primary key, UNIQUE constraints, foreign keys + ON DELETE
action, CHECK constraints, and indexes). The factory and the migration
cannot diverge without that test failing.

**Registration order.** ``wake_fires`` carries a composite-target FK
on ``agent_wake_schedules(schedule_id)`` (``wake_fires_schedule_fk``)
and a FK on ``webhook_subscriptions(subscription_id)``
(``wake_fires_webhook_subscription_fk``, retro-added in the v003
migration). ``agent_wake_schedules`` and ``webhook_subscriptions``
carry FKs on ``agent_skills(skill_id)`` (declared in the agent-skills
package). SQLAlchemy resolves cross-``Table`` references lazily at
DDL-emit / reflection time, so construction order is not strictly
required, but the FK target tables must be present on the same
metadata before the referencing table's FK can be emitted. Register
the parents before the children:
``agent_skills_table`` (from agent-skills) ->
:func:`agent_wake_schedules_table` ->
:func:`webhook_subscriptions_table` -> :func:`wake_fires_table`.

**No ``conversation_id`` FK.** Every table carries a denormalised
``conversation_id`` UUID with no FK constraint: the 3tears
``conversations`` table has composite PK ``(agent_id,
conversation_id)`` with no standalone ``UNIQUE (conversation_id)``, so
a single-column FK is not legal (same precedent as
``agent_skill_invocations.conversation_id``). The factory mirrors that
-- a plain UUID column, no FK.
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
    Boolean as SABoolean,
    DateTime as SADateTime,
    LargeBinary as SALargeBinary,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PgUUID

__all__ = [
    "agent_wake_schedules_table",
    "wake_fires_table",
    "webhook_subscriptions_table",
]


def agent_wake_schedules_table(metadata: MetaData) -> Table:
    """Register the ``agent_wake_schedules`` table on the given SA metadata.

    Mirrors the canonical v001 migration DDL
    (``migrations/v001_create_agent_wake_schedules.py``) exactly:
    composite primary key ``(conversation_id, schedule_id)``, standalone
    ``UNIQUE (schedule_id)`` so cross-package FKs (``wake_fires``) can
    reference the bare column, the nullable ``skill_id`` FK to
    ``agent_skills(skill_id) ON DELETE SET NULL``, the nullable self-FK
    ``context_from_schedule_id REFERENCES
    agent_wake_schedules(schedule_id) ON DELETE SET NULL``, the JSONB
    ``schedule_config`` column, the three CHECK constraints
    (``execution_mode`` / ``status`` / ``missed_fire_policy``
    enum-by-app), every NOT NULL DEFAULT, and the four indexes
    (including the two partial indexes).

    There is no DB CHECK on ``schedule_type`` (app-evolvable; validated
    in the agent-tools layer) nor on the JSONB ``schedule_config`` shape
    (varies per type) -- the factory matches the migration in omitting
    them.

    Register ``agent_skills_table`` (from the agent-skills package) on
    the same metadata first so the ``skill_id`` FK target resolves. See
    module docstring for the full registration order.

    Idempotent: returns the existing :class:`Table` if one with this
    name is already registered on ``metadata``. Call this before the
    consumer builds its L1 SQLite cache and before Alembic
    ``target_metadata`` reflection so both see the full shape.

    :param metadata: SQLAlchemy metadata to attach the table to
    :ptype metadata: MetaData
    :return: the ``agent_wake_schedules`` :class:`Table`
    :rtype: Table
    """
    if "agent_wake_schedules" in metadata.tables:
        return metadata.tables["agent_wake_schedules"]
    return Table(
        "agent_wake_schedules",
        metadata,
        Column("conversation_id", PgUUID(as_uuid=True), nullable=False),
        Column("schedule_id", PgUUID(as_uuid=True), nullable=False),
        Column("user_id", PgUUID(as_uuid=True), nullable=False),
        Column("agent_id", PgUUID(as_uuid=True), nullable=False),
        Column("skill_id", PgUUID(as_uuid=True), nullable=True),
        Column("schedule_type", Text(), nullable=False),
        Column(
            "schedule_config",
            JSONB(),
            nullable=False,
            server_default=sa_text("'{}'::jsonb"),
        ),
        Column("task_prompt", Text(), nullable=True),
        Column(
            "execution_mode",
            Text(),
            nullable=False,
            server_default=sa_text("'inline'"),
        ),
        Column(
            "status",
            Text(),
            nullable=False,
            server_default=sa_text("'active'"),
        ),
        Column("next_fire_at", SADateTime(timezone=True), nullable=True),
        Column("last_fired_at", SADateTime(timezone=True), nullable=True),
        Column("name", Text(), nullable=True),
        Column(
            "missed_fire_policy",
            Text(),
            nullable=False,
            server_default=sa_text("'coalesce'"),
        ),
        Column("context_from_schedule_id", PgUUID(as_uuid=True), nullable=True),
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
        PrimaryKeyConstraint("conversation_id", "schedule_id"),
        UniqueConstraint("schedule_id"),
        ForeignKeyConstraint(
            ["skill_id"],
            ["agent_skills.skill_id"],
            ondelete="SET NULL",
            name="agent_wake_schedules_skill_fk",
        ),
        ForeignKeyConstraint(
            ["context_from_schedule_id"],
            ["agent_wake_schedules.schedule_id"],
            ondelete="SET NULL",
            name="agent_wake_schedules_context_from_fk",
        ),
        CheckConstraint(
            "execution_mode IN ('inline', 'spawn')",
            name="agent_wake_schedules_execution_mode_check",
        ),
        CheckConstraint(
            "status IN ('active', 'paused', 'expired')",
            name="agent_wake_schedules_status_check",
        ),
        CheckConstraint(
            "missed_fire_policy IN ('coalesce', 'catch_up')",
            name="agent_wake_schedules_missed_fire_policy_check",
        ),
        Index(
            "idx_wake_schedules_next_fire",
            "next_fire_at",
            postgresql_where=sa_text(
                "status = 'active' AND next_fire_at IS NOT NULL",
            ),
        ),
        Index(
            "idx_wake_schedules_conv_status",
            "conversation_id",
            "status",
        ),
        Index(
            "idx_wake_schedules_user",
            "user_id",
        ),
        Index(
            "idx_wake_schedules_context_from",
            "context_from_schedule_id",
            postgresql_where=sa_text("context_from_schedule_id IS NOT NULL"),
        ),
    )


def wake_fires_table(metadata: MetaData) -> Table:
    """Register the ``wake_fires`` table on the given SA metadata.

    Mirrors the canonical migration DDL in its FINAL post-v004 shape
    (``migrations/v002_create_wake_fires.py`` for the table + the v003
    retro-added webhook FK + the v004 extended ``status`` CHECK):
    composite primary key ``(conversation_id, fire_id)``, standalone
    ``UNIQUE (fire_id)``, the FK ``schedule_id REFERENCES
    agent_wake_schedules(schedule_id) ON DELETE CASCADE``
    (``wake_fires_schedule_fk``), the FK ``webhook_subscription_id
    REFERENCES webhook_subscriptions(subscription_id) ON DELETE SET
    NULL`` (``wake_fires_webhook_subscription_fk``, retro-added in v003),
    the mutually-exclusive-source CHECK
    (``wake_fires_one_source_check``), the ``status`` CHECK INCLUDING the
    ``'dispatching'`` placeholder added in v004
    (``wake_fires_status_check``), the ``actual_fired_at`` /
    ``date_created`` server defaults, the ``display_suppressed`` boolean
    default, and the three indexes (two partial).

    Register both FK target tables on the same metadata first --
    :func:`agent_wake_schedules_table` and
    :func:`webhook_subscriptions_table` -- so both FKs resolve at
    DDL-emit / reflection time. See module docstring for the full
    registration order.

    Idempotent: returns the existing :class:`Table` if one with this
    name is already registered on ``metadata``.

    :param metadata: SQLAlchemy metadata to attach the table to
    :ptype metadata: MetaData
    :return: the ``wake_fires`` :class:`Table`
    :rtype: Table
    """
    if "wake_fires" in metadata.tables:
        return metadata.tables["wake_fires"]
    return Table(
        "wake_fires",
        metadata,
        Column("conversation_id", PgUUID(as_uuid=True), nullable=False),
        Column("fire_id", PgUUID(as_uuid=True), nullable=False),
        Column("schedule_id", PgUUID(as_uuid=True), nullable=True),
        Column("webhook_subscription_id", PgUUID(as_uuid=True), nullable=True),
        Column("scheduled_fire_at", SADateTime(timezone=True), nullable=True),
        Column(
            "actual_fired_at",
            SADateTime(timezone=True),
            nullable=False,
            server_default=sa_text("now()"),
        ),
        Column("status", Text(), nullable=False),
        Column(
            "display_suppressed",
            SABoolean(),
            nullable=False,
            server_default=sa_text("false"),
        ),
        Column("output_text", Text(), nullable=True),
        Column("latency_ms", Integer(), nullable=True),
        Column("error", Text(), nullable=True),
        Column(
            "date_created",
            SADateTime(timezone=True),
            nullable=False,
            server_default=sa_text("now()"),
        ),
        PrimaryKeyConstraint("conversation_id", "fire_id"),
        UniqueConstraint("fire_id"),
        ForeignKeyConstraint(
            ["schedule_id"],
            ["agent_wake_schedules.schedule_id"],
            ondelete="CASCADE",
            name="wake_fires_schedule_fk",
        ),
        ForeignKeyConstraint(
            ["webhook_subscription_id"],
            ["webhook_subscriptions.subscription_id"],
            ondelete="SET NULL",
            name="wake_fires_webhook_subscription_fk",
        ),
        CheckConstraint(
            "NOT (schedule_id IS NOT NULL AND webhook_subscription_id IS NOT NULL)",
            name="wake_fires_one_source_check",
        ),
        CheckConstraint(
            "status IN ("
            "'dispatching', "
            "'fired', "
            "'fired_silent', "
            "'yielded', "
            "'skipped_busy', "
            "'skipped_rate_limit', "
            "'skipped_cap', "
            "'skipped_no_handler', "
            "'failed'"
            ")",
            name="wake_fires_status_check",
        ),
        Index(
            "idx_wake_fires_schedule_time",
            "schedule_id",
            sa_text("actual_fired_at DESC"),
            postgresql_where=sa_text("schedule_id IS NOT NULL"),
        ),
        Index(
            "idx_wake_fires_webhook_time",
            "webhook_subscription_id",
            sa_text("actual_fired_at DESC"),
            postgresql_where=sa_text("webhook_subscription_id IS NOT NULL"),
        ),
        Index(
            "idx_wake_fires_conv_time",
            "conversation_id",
            sa_text("actual_fired_at DESC"),
        ),
    )


def webhook_subscriptions_table(metadata: MetaData) -> Table:
    """Register the ``webhook_subscriptions`` table on the given SA metadata.

    Mirrors the canonical migration DDL in its FINAL post-v005 shape
    (``migrations/v003_create_webhook_subscriptions.py`` for the table +
    the v005 opened ``verification_scheme`` CHECK): composite primary key
    ``(conversation_id, subscription_id)``, standalone ``UNIQUE
    (subscription_id)`` so the HTTP receiver can look up by bare id, the
    nullable ``default_skill_id`` FK to ``agent_skills(skill_id) ON
    DELETE SET NULL`` (``webhook_subscriptions_default_skill_fk``), the
    ``secret_ciphertext`` BYTEA column, the
    ``execution_mode`` / ``status``
    enum-by-app CHECKs, and the ``verification_scheme`` CHECK in its
    v005 slug-format form (``~ '^[a-z0-9_]+$' AND length(...) BETWEEN 1
    AND 64``, NOT the v003 hardcoded ``IN ('generic_hmac_sha256')``
    value), plus the two indexes.

    Register ``agent_skills_table`` (from the agent-skills package) on
    the same metadata first so the ``default_skill_id`` FK target
    resolves. See module docstring for the full registration order.

    Idempotent: returns the existing :class:`Table` if one with this
    name is already registered on ``metadata``.

    :param metadata: SQLAlchemy metadata to attach the table to
    :ptype metadata: MetaData
    :return: the ``webhook_subscriptions`` :class:`Table`
    :rtype: Table
    """
    if "webhook_subscriptions" in metadata.tables:
        return metadata.tables["webhook_subscriptions"]
    return Table(
        "webhook_subscriptions",
        metadata,
        Column("conversation_id", PgUUID(as_uuid=True), nullable=False),
        Column("subscription_id", PgUUID(as_uuid=True), nullable=False),
        Column("user_id", PgUUID(as_uuid=True), nullable=False),
        Column("agent_id", PgUUID(as_uuid=True), nullable=False),
        Column("default_skill_id", PgUUID(as_uuid=True), nullable=True),
        Column("name", Text(), nullable=True),
        Column("secret_ciphertext", SALargeBinary(), nullable=False),
        Column("allowed_source_pattern", Text(), nullable=True),
        Column(
            "execution_mode",
            Text(),
            nullable=False,
            server_default=sa_text("'inline'"),
        ),
        Column("task_prompt_template", Text(), nullable=True),
        Column(
            "verification_scheme",
            Text(),
            nullable=False,
            server_default=sa_text("'generic_hmac_sha256'"),
        ),
        Column(
            "status",
            Text(),
            nullable=False,
            server_default=sa_text("'active'"),
        ),
        Column("rate_limit_per_minute", Integer(), nullable=True),
        Column("last_fired_at", SADateTime(timezone=True), nullable=True),
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
        PrimaryKeyConstraint("conversation_id", "subscription_id"),
        UniqueConstraint("subscription_id"),
        ForeignKeyConstraint(
            ["default_skill_id"],
            ["agent_skills.skill_id"],
            ondelete="SET NULL",
            name="webhook_subscriptions_default_skill_fk",
        ),
        CheckConstraint(
            "execution_mode IN ('inline', 'spawn')",
            name="webhook_subscriptions_execution_mode_check",
        ),
        CheckConstraint(
            "verification_scheme ~ '^[a-z0-9_]+$' AND length(verification_scheme) BETWEEN 1 AND 64",
            name="webhook_subscriptions_verification_scheme_check",
        ),
        CheckConstraint(
            "status IN ('active', 'paused')",
            name="webhook_subscriptions_status_check",
        ),
        Index(
            "idx_webhook_subs_conv",
            "conversation_id",
            "status",
        ),
        Index(
            "idx_webhook_subs_user",
            "user_id",
        ),
    )
