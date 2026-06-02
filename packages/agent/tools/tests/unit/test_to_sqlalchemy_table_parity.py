"""Parity test: production :attr:`ContextItemCollection.schema`
must produce a SQLAlchemy :class:`Table` structurally equal to the
hand-written v0.8.0 reference fixture.

Reference fixture lives alongside the schema it guards so future
column adds keep the canonical-shape declaration in one place.
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy import (
    Column as SAColumn,
    DateTime,
    ForeignKeyConstraint as SAForeignKeyConstraint,
    Index as SAIndex,
    Text,
)
from sqlalchemy import text as sa_text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from threetears.agent.tools.collections import ContextItemCollection
from threetears.core.testing.sqla_parity import assert_tables_equivalent


def _reference_context_items_table(metadata: sa.MetaData) -> sa.Table:
    """Hand-written reference Table for parity testing.

    Frozen v0.8.0 canonical shape. If prod schema legitimately
    changes, add a NEW reference function (e.g.
    ``_reference_context_items_table_v090``) and run parity against
    the new one in addition to the old.
    """
    if "context_items" in metadata.tables:
        return metadata.tables["context_items"]
    return sa.Table(
        "context_items",
        metadata,
        SAColumn("conversation_id", PgUUID(as_uuid=True), primary_key=True, nullable=False),
        SAColumn("context_id", PgUUID(as_uuid=True), primary_key=True, nullable=False),
        SAColumn("context_type", Text(), nullable=False),
        SAColumn("key", Text(), nullable=False),
        SAColumn("short_desc", Text(), nullable=False),
        SAColumn("long_desc", Text(), nullable=False, server_default="''::text"),
        SAColumn("content", Text(), nullable=False),
        SAColumn("metadata", JSONB(), nullable=True),
        SAColumn("date_accessed", DateTime(timezone=True), nullable=False),
        SAColumn("date_created", DateTime(timezone=True), nullable=False),
        SAColumn("date_updated", DateTime(timezone=True), nullable=False),
        SAForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.conversation_id"],
            ondelete="CASCADE",
        ),
        SAIndex("ix_context_items_conv", "conversation_id"),
        SAIndex(
            "ix_context_items_type",
            "conversation_id",
            "context_type",
        ),
        SAIndex(
            "ix_context_items_lru",
            "conversation_id",
            "date_accessed",
        ),
        SAIndex(
            "ix_context_items_var_key",
            "conversation_id",
            "key",
            unique=True,
            postgresql_where=sa_text("context_type = 'variable'"),
        ),
        SAIndex(
            "ix_context_items_tool_result_key",
            "conversation_id",
            "key",
            unique=True,
            postgresql_where=sa_text("context_type = 'tool_result'"),
        ),
    )


def test_parity_context_items_collection_schema() -> None:
    """Production :attr:`ContextItemCollection.schema` produces a
    Table structurally equal to ``_reference_context_items_table``.
    """
    via_reference = _reference_context_items_table(sa.MetaData())
    via_collection = ContextItemCollection.schema.to_sqlalchemy_table(sa.MetaData())
    assert_tables_equivalent(via_reference, via_collection)
