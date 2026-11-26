"""the live tier-parity gate checks every migrated table a host has, whatever its column widths or schema.

Two gaps the hub found running it over its own tables:

- every string column was sent a 13-character sample, so a ``varchar(8)`` column refused it and the
  refusal aborted the whole run. The sample is cut to the column's length, and a value the table
  refuses is that column's finding, the rest still checked.
- a schema-qualified table (``platform_audit.audit_events``) was looked up as one name on the
  connection's own schema, and copied as one quoted identifier. Its schema and table are split, and
  each part quoted.

And one the hub found running it over its stamped tables: the scratch copy was made ``LIKE ... INCLUDING
DEFAULTS``, which copies a generated column as a plain one, still ``NOT NULL`` and with nothing to fill it,
so every write to the copy of a stamped table was refused on ``epoch_bucket``; and the table was held to its
declared columns alone, so the columns 3tears renders in L3 from the declaration (the bucket, a moveable
column's two move columns) were neither expected nor checked. The copy keeps its generated columns, and the
table is held to the shape 3tears stores it in.

Uses the session-scoped ``db_container`` fixture; a checkout without docker skips cleanly.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from threetears.core.collections.asyncpg_init import init_connection
from threetears.core.collections.schema_backed import (
    DATETIMETZ_TYPE,
    INT_TYPE,
    STRING_TYPE,
    UUID_TYPE,
    Column,
    SchemaBackedCollection,
    TableSchema,
)
from threetears.core.ddl.row_epoch import ROW_STAMPED
from threetears.core.entities.base import BaseEntity
from threetears.core.tenancy import tenant_column
from threetears.core.testing.l1_tables import TEST_L1_TABLE
from threetears.core.testing.plans import on_yugabyte
from threetears.core.testing.tier_parity import live_tier_parity_findings

pytestmark = pytest.mark.integration


class _Row(BaseEntity):
    primary_key_field = "id"


class _Narrow(SchemaBackedCollection[_Row]):
    """a table in another schema, with a column narrower than the gate's string sample."""

    schema = TableSchema(
        name="tp_other.narrow_rows",
        primary_key="id",
        columns=[
            Column("id", UUID_TYPE),
            Column("row_scope", STRING_TYPE),
            # "NULL" makes generated writes leave the column out; Postgres records no default for it
            Column("note", STRING_TYPE, nullable=True, server_default="NULL"),
            Column("small", INT_TYPE, nullable=True),
        ],
        l1_eviction=TEST_L1_TABLE,
    )
    entity_class = _Row

    @property
    def table_name(self) -> str:
        return self.schema.name


class _Stamped(SchemaBackedCollection[_Row]):
    """a stamped table keyed by two columns of two kinds, with a moveable tenant: in L3 it has a generated
    ``epoch_bucket`` and ``row_scope``'s two move columns, none of them declared."""

    schema = TableSchema(
        name="tp_stamped_rows",
        primary_key=("id", "at"),
        columns=[
            Column("id", UUID_TYPE),
            Column("at", DATETIMETZ_TYPE),
            Column("row_scope", STRING_TYPE),
            Column("small", INT_TYPE, nullable=True),
        ],
        tenancy=tenant_column("row_scope"),
        row_epoch=ROW_STAMPED,
        moveable=("row_scope",),
        l1_eviction=TEST_L1_TABLE,
    )
    entity_class = _Row

    @property
    def table_name(self) -> str:
        return self.schema.name


@pytest.fixture
async def conn(db_container: str) -> AsyncIterator[Any]:
    """a connection whose own schema is fresh, with ``tp_other.narrow_rows`` beside it.

    :param db_container: the Postgres container's connection URL
    :ptype db_container: str
    :return: the connection
    :rtype: AsyncIterator[Any]
    """
    own = f"tp_{uuid.uuid4().hex[:8]}"
    connection = await asyncpg.connect(db_container, server_settings={"search_path": own})
    await init_connection(connection)
    await connection.execute(f"CREATE SCHEMA {own}")
    await connection.execute("CREATE SCHEMA IF NOT EXISTS tp_other")
    await connection.execute("DROP TABLE IF EXISTS tp_other.narrow_rows")
    try:
        yield connection
    finally:
        await connection.execute(f"DROP SCHEMA {own} CASCADE")
        await connection.execute("DROP TABLE IF EXISTS tp_other.narrow_rows")
        await connection.close()


async def test_a_schema_qualified_table_with_a_narrow_varchar_is_checked_whole(conn: Any) -> None:
    await conn.execute(
        "CREATE TABLE tp_other.narrow_rows (id uuid PRIMARY KEY, row_scope varchar(8) NOT NULL, "
        "note text, small integer)"
    )
    assert await live_tier_parity_findings(conn, [_Narrow]) == []


async def test_a_value_the_table_refuses_is_that_columns_finding_and_the_rest_are_still_checked(conn: Any) -> None:
    # ``small`` is a smallint here, which refuses the gate's -2147483647; the other columns still pass
    await conn.execute(
        "CREATE TABLE tp_other.narrow_rows (id uuid PRIMARY KEY, row_scope varchar(8) NOT NULL, "
        "note text, small smallint)"
    )
    findings = await live_tier_parity_findings(conn, [_Narrow])
    assert [f.split(": ")[0:2] for f in findings] == [["_Narrow.small", "L3"], ["_Narrow.small", "L3"]], findings
    assert "migrated as smallint" in findings[0]
    assert "the table refused" in findings[1]


async def _migrate(conn: Any, statements: tuple[str, ...]) -> None:
    for statement in statements:
        await conn.execute(statement)


async def test_a_stamped_table_as_3tears_renders_it_passes(conn: Any) -> None:
    await _migrate(conn, _Stamped.schema.create_statements(yugabyte=await on_yugabyte(conn)))
    assert await live_tier_parity_findings(conn, [_Stamped]) == []


async def test_a_stamped_table_migrated_without_its_bucket_is_named(conn: Any) -> None:
    # the stamped columns all there but the generated bucket (and the index that leads with it)
    await conn.execute(
        "CREATE TABLE tp_stamped_rows (id uuid NOT NULL, at timestamptz NOT NULL, row_scope text NOT NULL, "
        "small integer, epoch bigint NOT NULL, gone boolean NOT NULL DEFAULT false, row_scope_moved_from text, "
        "row_scope_moved_epoch bigint, PRIMARY KEY (id, at))"
    )
    findings = await live_tier_parity_findings(conn, [_Stamped])
    assert len(findings) == 1 and findings[0].startswith("_Stamped.epoch_bucket: L3: 3tears renders the column"), (
        findings
    )


async def test_a_bucket_migrated_as_a_plain_column_and_a_missing_move_column_are_named(conn: Any) -> None:
    await conn.execute(
        "CREATE TABLE tp_stamped_rows (id uuid NOT NULL, at timestamptz NOT NULL, row_scope text NOT NULL, "
        "small integer, epoch bigint NOT NULL, gone boolean NOT NULL DEFAULT false, row_scope_moved_from text, "
        "epoch_bucket smallint NOT NULL DEFAULT 0, PRIMARY KEY (id, at))"
    )
    findings = await live_tier_parity_findings(conn, [_Stamped])
    assert [f.split(": ")[0] for f in findings] == ["_Stamped.epoch_bucket", "_Stamped.row_scope_moved_epoch"], findings
    assert "migrated as a plain column, rendered GENERATED ALWAYS" in findings[0]
    assert "3tears renders the column" in findings[1]
