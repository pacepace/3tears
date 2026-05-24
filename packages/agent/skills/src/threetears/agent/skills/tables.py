"""SQLAlchemy ``Table`` factories for the agent-skills L3 tables.

Cross-package consumers (metallm, future hosts) register the
agent-skills tables on their own SQLAlchemy ``MetaData`` -- the L1
SQLite cache builds from that metadata, and Alembic ``target_metadata``
reflection sees the same shape for auto-generate. The agent-memory
package exposes ``memories_table(metadata)`` etc. for exactly this; the
factories here are the agent-skills counterpart so consumers can follow
the established pattern instead of hand-rolling the ``Table`` shape.

**Why standalone ``Table(...)`` and not a schema-backed delegate.**
The agent-memory factories delegate to
``<Collection>.schema.to_sqlalchemy_table(metadata)`` because those
collections are :class:`SchemaBackedCollection` subclasses with a
``TableSchema`` single source of truth. The agent-skills collections
deliberately subclass :class:`BaseCollection` and hand-roll SQL
because three columns (``tool_additions`` / ``tool_restrictions`` /
``tags``) are Postgres ``TEXT[]`` arrays and two tables carry CHECK
constraints -- neither the array type tag nor a CHECK-constraint
descriptor exists in the framework's ``TableSchema`` /
``to_sqlalchemy_table`` type system, and adding them is a core
framework change out of scope for this package. So the factories below
construct the ``Table`` directly.

**Drift protection.** A parallel hand-written DDL with no parity
guarantee would be the embedded-DDL-drift smell. The factory output is
pinned against the canonical v001/v002 migration DDL by
``tests/integration/test_sqlalchemy_table_parity.py``: it applies the
migrations to one Postgres schema, emits each factory's ``CREATE
TABLE`` into a second schema, and asserts the two are structurally
identical (columns, types, nullability, primary key, UNIQUE
constraints, foreign key + ON DELETE action, CHECK constraints, and
indexes). The factory and the migration cannot diverge without that
test failing.

**Trigger-maintained ``search_vector``.** Mirroring the agent-memory
``memories`` / ``media_content`` factories, the ``search_vector``
TSVECTOR column is declared here but the FTS trigger + function DDL is
NOT -- triggers are not SQLAlchemy ``Table`` constructs. The consumer
installs the trigger in its own Alembic migration (the v001 migration
is the canonical trigger DDL); the factory only needs the column so the
metadata-driven INSERT/SELECT paths know it exists.
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
)
from sqlalchemy.dialects.postgresql import ARRAY, TSVECTOR
from sqlalchemy.dialects.postgresql import UUID as PgUUID

__all__ = [
    "agent_skill_invocations_table",
    "agent_skills_table",
]


def agent_skills_table(metadata: MetaData) -> Table:
    """Register the ``agent_skills`` table on the given SA metadata.

    Mirrors the canonical v001 migration DDL
    (``migrations/v001_create_agent_skills.py``) exactly: composite
    primary key ``(agent_id, skill_id)``, standalone ``UNIQUE
    (skill_id)`` so cross-package FKs can reference the bare column, the
    three ``TEXT[]`` array columns, the trigger-maintained
    ``search_vector`` TSVECTOR column, the two CHECK constraints
    (``prompt_mode`` enum-by-app + the at-least-one-payload invariant),
    every NOT NULL DEFAULT, and the four indexes.

    Idempotent: returns the existing :class:`Table` if one with this
    name is already registered on ``metadata``. Call this before the
    consumer builds its L1 SQLite cache and before Alembic
    ``target_metadata`` reflection so both see the full shape.

    The FTS trigger/function DDL is intentionally NOT emitted here (it
    is not a SQLAlchemy ``Table`` construct); the consumer installs it
    via its own Alembic migration. See module docstring.

    :param metadata: SQLAlchemy metadata to attach the table to
    :ptype metadata: MetaData
    :return: the ``agent_skills`` :class:`Table`
    :rtype: Table
    """
    if "agent_skills" in metadata.tables:
        return metadata.tables["agent_skills"]
    return Table(
        "agent_skills",
        metadata,
        Column("agent_id", PgUUID(as_uuid=True), nullable=False),
        Column("skill_id", PgUUID(as_uuid=True), nullable=False),
        Column("user_id", PgUUID(as_uuid=True), nullable=False),
        Column("name", Text(), nullable=False),
        Column("summary", Text(), nullable=False),
        Column("body", Text(), nullable=True),
        Column(
            "prompt_mode",
            Text(),
            nullable=False,
            server_default=sa_text("'additive'"),
        ),
        Column(
            "tool_additions",
            ARRAY(Text()),
            nullable=False,
            server_default=sa_text("'{}'"),
        ),
        Column(
            "tool_restrictions",
            ARRAY(Text()),
            nullable=False,
            server_default=sa_text("'{}'"),
        ),
        Column(
            "trigger_keywords",
            Text(),
            nullable=False,
            server_default=sa_text("''"),
        ),
        Column(
            "tags",
            ARRAY(Text()),
            nullable=False,
            server_default=sa_text("'{}'"),
        ),
        Column(
            "source",
            Text(),
            nullable=False,
            server_default=sa_text("'manual'"),
        ),
        Column(
            "enabled",
            SABoolean(),
            nullable=False,
            server_default=sa_text("true"),
        ),
        Column(
            "use_count",
            Integer(),
            nullable=False,
            server_default=sa_text("0"),
        ),
        Column("last_used_at", SADateTime(timezone=True), nullable=True),
        Column(
            "success_count",
            Integer(),
            nullable=False,
            server_default=sa_text("0"),
        ),
        Column(
            "failure_count",
            Integer(),
            nullable=False,
            server_default=sa_text("0"),
        ),
        Column("last_failure_at", SADateTime(timezone=True), nullable=True),
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
        Column("search_vector", TSVECTOR(), nullable=True),
        PrimaryKeyConstraint("agent_id", "skill_id"),
        UniqueConstraint("skill_id"),
        CheckConstraint(
            "prompt_mode IN ('additive', 'replace')",
            name="agent_skills_prompt_mode_check",
        ),
        CheckConstraint(
            "body IS NOT NULL "
            "OR array_length(tool_additions, 1) IS NOT NULL "
            "OR array_length(tool_restrictions, 1) IS NOT NULL",
            name="agent_skills_payload_check",
        ),
        Index(
            "uq_skills_agent_user_name",
            "agent_id",
            "user_id",
            "name",
            unique=True,
        ),
        Index(
            "idx_skills_agent_user_enabled",
            "agent_id",
            "user_id",
            "enabled",
        ),
        Index(
            "idx_skills_search_vector",
            "search_vector",
            postgresql_using="gin",
        ),
        Index(
            "idx_skills_tags",
            "tags",
            postgresql_using="gin",
        ),
    )


def agent_skill_invocations_table(metadata: MetaData) -> Table:
    """Register the ``agent_skill_invocations`` table on the SA metadata.

    Mirrors the canonical v002 migration DDL
    (``migrations/v002_create_agent_skill_invocations.py``) exactly:
    composite primary key ``(agent_id, invocation_id)``, standalone
    ``UNIQUE (invocation_id)``, the composite foreign key
    ``(agent_id, skill_id) REFERENCES agent_skills(agent_id, skill_id)
    ON DELETE CASCADE``, the two CHECK constraints
    (``invocation_source`` enum-by-app + the nullable ``outcome``
    enum), the ``invoked_at`` server default, and the two indexes.

    ``message_id`` is deliberately a plain UUID with no FK constraint:
    the ``messages`` table is consumer-owned and rows may be hard-
    deleted, but the invocation history must survive a message
    deletion (matches the v002 migration's design note).

    Idempotent: returns the existing :class:`Table` if one with this
    name is already registered on ``metadata``.

    Because the composite FK references ``agent_skills``, register that
    table first (call :func:`agent_skills_table` on the same metadata)
    so the FK target resolves; SQLAlchemy resolves the reference lazily
    at DDL-emit / reflection time, so registration order is not strictly
    required for construction, but the target table must exist on the
    metadata before the FK can be emitted.

    :param metadata: SQLAlchemy metadata to attach the table to
    :ptype metadata: MetaData
    :return: the ``agent_skill_invocations`` :class:`Table`
    :rtype: Table
    """
    if "agent_skill_invocations" in metadata.tables:
        return metadata.tables["agent_skill_invocations"]
    return Table(
        "agent_skill_invocations",
        metadata,
        Column("agent_id", PgUUID(as_uuid=True), nullable=False),
        Column("invocation_id", PgUUID(as_uuid=True), nullable=False),
        Column("skill_id", PgUUID(as_uuid=True), nullable=False),
        Column("user_id", PgUUID(as_uuid=True), nullable=False),
        Column("conversation_id", PgUUID(as_uuid=True), nullable=False),
        Column("message_id", PgUUID(as_uuid=True), nullable=True),
        Column("invocation_source", Text(), nullable=False),
        Column(
            "invoked_at",
            SADateTime(timezone=True),
            nullable=False,
            server_default=sa_text("now()"),
        ),
        Column("outcome", Text(), nullable=True),
        Column("outcome_source", Text(), nullable=True),
        Column("notes", Text(), nullable=True),
        PrimaryKeyConstraint("agent_id", "invocation_id"),
        UniqueConstraint("invocation_id"),
        ForeignKeyConstraint(
            ["agent_id", "skill_id"],
            ["agent_skills.agent_id", "agent_skills.skill_id"],
            ondelete="CASCADE",
            name="agent_skill_invocations_skill_fk",
        ),
        CheckConstraint(
            "invocation_source IN ('wake', 'invoke')",
            name="agent_skill_invocations_source_check",
        ),
        CheckConstraint(
            "outcome IS NULL OR outcome IN ('success', 'failure')",
            name="agent_skill_invocations_outcome_check",
        ),
        Index(
            "idx_skill_invocations_skill_time",
            "agent_id",
            "skill_id",
            sa_text("invoked_at DESC"),
        ),
        Index(
            "idx_skill_invocations_conv",
            "agent_id",
            "conversation_id",
            sa_text("invoked_at DESC"),
        ),
    )
