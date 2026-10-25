"""the live tier-parity gate checks every migrated table a host has, whatever its column widths or schema.

Two gaps the hub found running it over its own tables:

- every string column was sent a 13-character sample, so a ``varchar(8)`` column refused it and the
  refusal aborted the whole run. The sample is cut to the column's length, and a value the table
  refuses is that column's finding, the rest still checked.
- a schema-qualified table (``platform_audit.audit_events``) was looked up as one name on the
  connection's own schema, and copied as one quoted identifier. Its schema and table are split, and
  each part quoted.

Uses the session-scoped ``db_container`` fixture; a checkout without docker skips cleanly.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any, ClassVar

import asyncpg
import pytest

from threetears.core.collections.asyncpg_init import init_connection
from threetears.core.collections.schema_backed import (
    INT_TYPE,
    STRING_TYPE,
    UUID_TYPE,
    Column,
    SchemaBackedCollection,
    TableSchema,
)
from threetears.core.entities.base import BaseEntity
from threetears.core.testing.tier_parity import live_tier_parity_findings

pytestmark = pytest.mark.integration


class _Row(BaseEntity):
    primary_key_field = "id"


class _Narrow(SchemaBackedCollection[_Row]):
    """a table in another schema, with a column narrower than the gate's string sample."""

    schema: ClassVar[TableSchema] = TableSchema(
        name="tp_other.narrow_rows",
        primary_key="id",
        columns=[
            Column("id", UUID_TYPE),
            Column("row_scope", STRING_TYPE),
            Column("note", STRING_TYPE, nullable=True),
            Column("small", INT_TYPE, nullable=True),
        ],
    )
    entity_class = _Row  # type: ignore[assignment]

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
