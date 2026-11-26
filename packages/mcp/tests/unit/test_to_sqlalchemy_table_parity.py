"""Parity test: production :attr:`McpToolGrantCollection.schema`
must produce a SQLAlchemy :class:`Table` structurally equal to the
hand-written v0.8.0 reference fixture.

Reference fixture lives alongside the schema it guards so future
column adds keep the canonical-shape declaration in one place.
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy import (
    BigInteger,
    Boolean,
    Column as SAColumn,
    DateTime,
    Text,
)
from sqlalchemy import text as sa_text
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from threetears.core.testing.sqla_parity import assert_tables_equivalent
from threetears.mcp.rbac import McpToolGrantCollection
from threetears.core.testing.l1_tables import L1_EVICTION_INFO, TEST_L1_TABLE


def _reference_mcp_tool_grants_table(metadata: sa.MetaData) -> sa.Table:
    """Hand-written reference Table for parity testing.

    Frozen v0.8.0 canonical shape. If prod schema legitimately
    changes, add a NEW reference function (e.g.
    ``_reference_mcp_tool_grants_table_v090``) and run parity against
    the new one in addition to the old.
    """
    if "mcp_tool_grants" in metadata.tables:
        return metadata.tables["mcp_tool_grants"]
    return sa.Table(
        "mcp_tool_grants",
        metadata,
        SAColumn("grant_id", PgUUID(as_uuid=True), primary_key=True, nullable=False),
        SAColumn("principal_type", Text(), nullable=False),
        SAColumn("principal_id", PgUUID(as_uuid=True), nullable=False),
        SAColumn("tool_name", Text(), nullable=False),
        SAColumn("permission", Text(), nullable=False),
        SAColumn(
            "date_created",
            DateTime(timezone=True),
            nullable=False,
            server_default=sa_text("now()"),
        ),
        info={L1_EVICTION_INFO: TEST_L1_TABLE},
    )


def _reference_mcp_tool_grants_table_v095(metadata: sa.MetaData) -> sa.Table:
    """the v0.95.0 shape: the v0.8.0 table, stamped (``ROW_STAMPED``, mcp v003): its epoch, ``BIGINT NOT NULL`` with no
    default, and its gone flag, ``BOOLEAN NOT NULL DEFAULT false``. The restore bucket is L3's alone (a stored generated
    column), so a declared table carries none.

    :param metadata: the metadata to attach the table to
    :ptype metadata: sa.MetaData
    :return: the table
    :rtype: sa.Table
    """
    table = _reference_mcp_tool_grants_table(metadata)
    table.append_column(SAColumn("epoch", BigInteger(), nullable=False))
    table.append_column(SAColumn("gone", Boolean(), nullable=False, server_default=sa_text("false")))
    return table


def test_parity_mcp_tool_grants_collection_schema() -> None:
    """Production :attr:`McpToolGrantCollection.schema` produces a
    Table structurally equal to ``_reference_mcp_tool_grants_table_v095``: the frozen v0.8.0 shape, stamped.
    """
    via_reference = _reference_mcp_tool_grants_table_v095(sa.MetaData())
    via_collection = McpToolGrantCollection.schema.to_sqlalchemy_table(sa.MetaData())
    assert_tables_equivalent(via_reference, via_collection)
    # the v0.8.0 shape is the v0.95.0 one but for the two stamped columns
    frozen = {column.name for column in _reference_mcp_tool_grants_table(sa.MetaData()).columns}
    assert {column.name for column in via_collection.columns} - frozen == {"epoch", "gone"}
