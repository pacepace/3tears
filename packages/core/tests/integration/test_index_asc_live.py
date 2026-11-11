"""Integration: an index declared ``asc`` on its first column serves a range read in order, a page at a time, on
YugabyteDB; the same index undeclared reads the table.

YugabyteDB hashes an index's first column unless it is declared ``ASC`` (or ``DESC``). A hashed column seeks no
range and answers no order, so a page of the rows past a bound (``expires_at > $1 ORDER BY expires_at LIMIT n``,
what :func:`~threetears.core.keyed.read_after` and ``delete_before`` send with nothing bound before the column) reads
every row past it and sorts them. :attr:`~threetears.core.collections.schema_backed.IndexDef.asc` declares the column
``ASC``: the DDL ``to_sqlalchemy_table`` renders range-shards the index, and the page reads about its own rows. Postgres
reads both the same way (``ASC`` is its default), so there the rows are checked only.

Against the session ``db_container`` (Postgres); set ``THREETEARS_TEST_POSTGRES_URL`` to a YugabyteDB URL to run it
there.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

import asyncpg
import pytest
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateIndex, CreateTable

from threetears.core.collections.schema_backed import BIGINT_TYPE, STRING_TYPE, Column, Index, TableSchema
from threetears.core.keyed import read_after
from threetears.core.testing.plans import PlanRecorder, StatementPlan, on_yugabyte

pytestmark = pytest.mark.integration

_ROWS = 50_000
_PAGE = 500


def _schema(name: str, *, ranged: bool) -> TableSchema:
    return TableSchema(
        name=name,
        primary_key="token_id",
        columns=[Column("token_id", STRING_TYPE), Column("expires_at", BIGINT_TYPE)],
        indexes=(Index(f"{name}_by_expiry", "expires_at", asc="expires_at" if ranged else ()),),
    )


@dataclass
class _Seeded:
    conn: asyncpg.Connection
    yugabyte: bool


@pytest.fixture
async def seeded(db_container: str) -> AsyncIterator[_Seeded]:
    """two tables of the same rows in a schema of their own: one whose expiry index is declared ``asc``, one not."""
    schema = f"index_asc_{uuid.uuid4().hex[:8]}"
    conn = await asyncpg.connect(db_container, command_timeout=600)
    try:
        await conn.execute(f"CREATE SCHEMA {schema}")
        await conn.execute(f"SET search_path = {schema}")
        for name, ranged in (("ranged", True), ("hashed", False)):
            table = _schema(name, ranged=ranged).to_sqlalchemy_table(sa.MetaData())
            await conn.execute(str(CreateTable(table).compile(dialect=postgresql.dialect())))
            for index in table.indexes:
                await conn.execute(str(CreateIndex(index).compile(dialect=postgresql.dialect())))
            await conn.execute(
                f"INSERT INTO {name} SELECT 't' || g, g FROM generate_series(1, $1) g",  # noqa: S608
                _ROWS,
            )
            await conn.execute(f"ANALYZE {name}")
        yield _Seeded(conn=conn, yugabyte=await on_yugabyte(conn))
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await conn.close()


async def _page(conn: asyncpg.Connection, table: str) -> tuple[list[int], StatementPlan]:
    recorder = PlanRecorder(conn)
    page = await read_after(
        recorder,
        table,
        columns=("token_id", "expires_at"),
        id_column="expires_at",
        after=_ROWS // 2,
        limit=_PAGE,
    )
    (plan,) = recorder.plans
    return [row["expires_at"] for row in page.rows], plan


class TestARangeReadOfAnIndexLedByItsColumn:
    async def test_an_index_declared_asc_pages_a_range_without_reading_the_table(self, seeded: _Seeded) -> None:
        """Flips if a page past a bound through an index declared ``asc`` reads past its own rows on YugabyteDB, or if
        the same index undeclared stops reading the table there (the declaration would then say nothing)."""
        expected = list(range(_ROWS // 2 + 1, _ROWS // 2 + _PAGE + 1))

        ranged, ranged_plan = await _page(seeded.conn, "ranged")
        hashed, hashed_plan = await _page(seeded.conn, "hashed")

        assert ranged == expected
        assert hashed == expected
        if seeded.yugabyte:
            assert ranged_plan.rows_scanned <= 2 * (_PAGE + 1), ranged_plan
            assert hashed_plan.rows_scanned >= _ROWS // 2, hashed_plan

    def test_asc_renders_the_column_ascending_in_so_many_words(self) -> None:
        """Flips if a column declared ``asc`` is rendered without ``ASC`` (YugabyteDB would hash it)."""
        table = _schema("ranged", ranged=True).to_sqlalchemy_table(sa.MetaData())
        (index,) = table.indexes
        assert "(expires_at ASC)" in str(CreateIndex(index).compile(dialect=postgresql.dialect()))
