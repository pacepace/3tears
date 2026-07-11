"""Parity test: production ``IntentionsCollection.schema`` must produce a
SQLAlchemy :class:`Table` structurally equal to a hand-written reference
fixture.

The reference fixture lives alongside the schema it guards so any future
column add to ``IntentionsCollection.schema`` has its reference fixture in
the same package -- no cross-package layering inversion. Structural
comparison helpers live in :mod:`threetears.core.testing.sqla_parity`.
"""

from __future__ import annotations

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Column as SAColumn,
    DateTime,
    Enum as SAEnum,
    Numeric,
    Text,
)
from sqlalchemy import text as sa_text
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from threetears.agent.intention.collections import IntentionsCollection
from threetears.core.testing.sqla_parity import assert_tables_equivalent

# Embedding dimension carried by the intentions reference fixture.
# Mirrors ``threetears.agent.intention.collections._INTENTION_VECTOR_DIM``;
# duplicated here so the reference fixture stays a self-contained, frozen
# regression net independent of the source-under-test module.
_REFERENCE_INTENTION_VECTOR_DIM = 1024


def _reference_intentions_table(metadata: sa.MetaData) -> sa.Table:
    """Hand-written reference Table for parity testing (v001).

    Frozen v0.15.0 canonical shape. If the prod schema legitimately
    changes, add a NEW reference function (e.g.
    ``_reference_intentions_table_v0160``) and run parity against the new
    one in addition to the old.
    """
    if "intentions" in metadata.tables:
        return metadata.tables["intentions"]
    return sa.Table(
        "intentions",
        metadata,
        SAColumn("intention_id", PgUUID(as_uuid=True), primary_key=True, nullable=False),
        SAColumn("agent_id", PgUUID(as_uuid=True), primary_key=True, nullable=False),
        SAColumn("customer_id", PgUUID(as_uuid=True), nullable=True),
        # user_id is a soft ref (no FK) -- isolation is the WHERE clause.
        SAColumn("user_id", PgUUID(as_uuid=True), nullable=True),
        SAColumn(
            "status",
            SAEnum(
                "open",
                "asked",
                "granted",
                "dropped",
                name="intention_status",
                create_constraint=True,
            ),
            nullable=False,
            server_default="'open'",
        ),
        SAColumn("content", Text(), nullable=False),
        SAColumn(
            "embedding",
            Vector(_REFERENCE_INTENTION_VECTOR_DIM),
            nullable=True,
        ),
        SAColumn(
            "salience",
            Numeric(5, 4),
            nullable=False,
            server_default="0.5",
        ),
        SAColumn("last_decayed_at", DateTime(timezone=True), nullable=True),
        SAColumn("last_surfaced_at", DateTime(timezone=True), nullable=True),
        SAColumn("source_memory_id", PgUUID(as_uuid=True), nullable=True),
        SAColumn("source_conversation_id", PgUUID(as_uuid=True), nullable=True),
        SAColumn("date_created", DateTime(timezone=True), nullable=False),
        SAColumn("date_updated", DateTime(timezone=True), nullable=True),
        sa.Index(
            "idx_intentions_open_ranked",
            "agent_id",
            "user_id",
            "salience",
            postgresql_where=sa_text("status = 'open'"),
        ),
        sa.Index("idx_intentions_last_surfaced", "agent_id", "last_surfaced_at"),
        sa.Index(
            "ix_intentions_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


def test_parity_intentions_collection_schema() -> None:
    """Production :attr:`IntentionsCollection.schema` produces a Table
    structurally equal to ``_reference_intentions_table``.
    """
    via_reference = _reference_intentions_table(sa.MetaData())
    via_collection = IntentionsCollection.schema.to_sqlalchemy_table(sa.MetaData())
    assert_tables_equivalent(via_reference, via_collection)
