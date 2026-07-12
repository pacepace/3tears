"""Parity test: production ``IdentityVersionsCollection.schema`` must produce a
SQLAlchemy :class:`Table` structurally equal to a hand-written reference
fixture.

The reference fixture lives alongside the schema it guards so any future
column add has its reference in the same package. Structural comparison
helpers live in :mod:`threetears.core.testing.sqla_parity`. If the prod
schema legitimately changes, add a NEW reference function and run parity
against it in addition to the old.
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy import (
    Column as SAColumn,
    DateTime,
    Enum as SAEnum,
    Text,
)
from sqlalchemy import text as sa_text
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from threetears.agent.identity.collections import IdentityVersionsCollection
from threetears.core.testing.sqla_parity import assert_tables_equivalent


def _reference_identity_versions_table(metadata: sa.MetaData) -> sa.Table:
    """Hand-written reference Table for parity testing (v001).

    Frozen v0.15.0 canonical shape.
    """
    if "identity_versions" in metadata.tables:
        return metadata.tables["identity_versions"]
    return sa.Table(
        "identity_versions",
        metadata,
        SAColumn("version_id", PgUUID(as_uuid=True), primary_key=True, nullable=False),
        SAColumn("agent_id", PgUUID(as_uuid=True), primary_key=True, nullable=False),
        SAColumn("customer_id", PgUUID(as_uuid=True), nullable=True),
        SAColumn("user_id", PgUUID(as_uuid=True), nullable=True),
        SAColumn(
            "block_key",
            SAEnum(
                "personality",
                "reinforcement",
                "anti_sycophant",
                "self_improvement",
                "presence",
                name="identity_block_key",
                create_constraint=True,
            ),
            nullable=False,
        ),
        SAColumn("content", Text(), nullable=False),
        SAColumn("rationale", Text(), nullable=True),
        SAColumn("content_hash", Text(), nullable=False),
        SAColumn("parent_version_id", PgUUID(as_uuid=True), nullable=True),
        SAColumn(
            "status",
            SAEnum(
                "proposed",
                "active",
                "superseded",
                "rejected",
                name="identity_version_status",
                create_constraint=True,
            ),
            nullable=False,
            server_default="'proposed'",
        ),
        SAColumn("proposer_agent_id", PgUUID(as_uuid=True), nullable=True),
        SAColumn("consenter_user_id", PgUUID(as_uuid=True), nullable=True),
        SAColumn("date_created", DateTime(timezone=True), nullable=False),
        SAColumn("date_updated", DateTime(timezone=True), nullable=True),
        sa.Index(
            "uq_identity_active_per_block",
            "agent_id",
            "customer_id",
            "user_id",
            "block_key",
            unique=True,
            postgresql_where=sa_text("status = 'active'"),
        ),
        sa.Index(
            "idx_identity_block_history",
            "agent_id",
            "user_id",
            "block_key",
            "date_created",
        ),
        sa.Index(
            "idx_identity_pending",
            "agent_id",
            "user_id",
            postgresql_where=sa_text("status = 'proposed'"),
        ),
    )


def test_parity_identity_versions_collection_schema() -> None:
    """Production :attr:`IdentityVersionsCollection.schema` produces a Table
    structurally equal to ``_reference_identity_versions_table``.
    """
    via_reference = _reference_identity_versions_table(sa.MetaData())
    via_collection = IdentityVersionsCollection.schema.to_sqlalchemy_table(sa.MetaData())
    assert_tables_equivalent(via_reference, via_collection)
