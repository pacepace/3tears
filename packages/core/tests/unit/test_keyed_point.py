"""point reads (:mod:`threetears.core.keyed.point`): ``read_by``, ``read_row`` and ``read_value`` answer what ``fetch``,
``fetchrow`` and ``fetchval`` of the hand-written ``WHERE a = $1 AND b = $2`` answered, and send that statement.

Correctness on an in-process DuckDB (the helpers' SQL is one spelling for PostgreSQL, YugabyteDB and DuckDB); the
PostgreSQL / YugabyteDB proof of each rewrite is the repo's generated equivalence test
(:mod:`threetears.core.testing.rewrites`). A recording reader pins the statements.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import duckdb
import pytest

from threetears.core.keyed import AnyOf, ThreadedDuckDB, read_by, read_row, read_value


@pytest.fixture
def db() -> Iterator[duckdb.DuckDBPyConnection]:
    """``t``: keyed ``(a, b)``, three groups of four rows, ``v`` NULL on odd ``b``."""
    conn = duckdb.connect()
    conn.execute("CREATE TABLE t (a VARCHAR, b BIGINT, v INTEGER, flag BOOLEAN, PRIMARY KEY (a, b))")
    conn.execute(
        "INSERT INTO t SELECT 'k' || (i // 4), i, CASE WHEN i % 2 = 0 THEN i END, i % 3 = 0 FROM range(12) r(i)"
    )
    yield conn
    conn.close()


class _Recording:
    """a reader that answers from DuckDB and keeps every statement it was sent."""

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._inner = ThreadedDuckDB(conn)
        self.statements: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, query: str, *params: Any) -> list[dict[str, Any]]:
        self.statements.append((query, params))
        return await self._inner.fetch(query, *params)


def _plain(db: duckdb.DuckDBPyConnection, sql: str, *params: Any) -> list[tuple[Any, ...]]:
    return [tuple(row) for row in db.execute(sql, list(params)).fetchall()]


async def test_read_by_answers_every_row_the_pins_name(db: duckdb.DuckDBPyConnection) -> None:
    """Flips if read_by keeps a row the statement would not, or drops one it would."""
    rows = await read_by(ThreadedDuckDB(db), "t", by={"a": "k1", "flag": True}, columns=("a", "b"))
    assert sorted((row["a"], row["b"]) for row in rows) == _plain(
        db, "SELECT a, b FROM t WHERE a = $1 AND flag = TRUE ORDER BY b", "k1"
    )


async def test_read_by_reads_every_column_when_none_is_named(db: duckdb.DuckDBPyConnection) -> None:
    """Flips if ``columns=None`` stops reading the whole row, in the table's order."""
    rows = await read_by(ThreadedDuckDB(db), "t", by={"a": "k0", "b": 2})
    assert rows == [{"a": "k0", "b": 2, "v": 2, "flag": False}]


async def test_none_pins_is_null_and_anyof_a_set(db: duckdb.DuckDBPyConnection) -> None:
    """Flips if a None pin stops meaning IS NULL, or an AnyOf a set."""
    reader = _Recording(db)
    rows = await read_by(reader, "t", by={"a": AnyOf(["k0", "k2"], cast="text"), "v": None}, columns=("b",))
    assert sorted(row["b"] for row in rows) == [1, 3, 9, 11]
    assert reader.statements[0][0] == ('SELECT t."b" FROM "t" AS t WHERE t."a" = ANY($1::text[]) AND t."v" IS NULL')


async def test_each_pin_is_bound_as_the_hand_written_statement_bound_it(db: duckdb.DuckDBPyConnection) -> None:
    """Flips if a pin is sent other than ``= $n`` with its value as the parameter (the column types it, as before)."""
    reader = _Recording(db)
    await read_by(reader, "t", by={"a": "k1", "flag": False}, columns=("b",), records=True)
    assert reader.statements == [('SELECT t."b" FROM "t" AS t WHERE t."a" = $1 AND t."flag" = $2', ("k1", False))]


async def test_read_row_asks_for_one_row_and_answers_none_for_none(db: duckdb.DuckDBPyConnection) -> None:
    """Flips if read_row asks for more than one row, or answers anything but None where the key names none."""
    reader = _Recording(db)
    assert await read_row(reader, "t", by={"a": "k1", "b": 5}) == {"a": "k1", "b": 5, "v": None, "flag": False}
    assert await read_row(reader, "t", by={"a": "k9", "b": 5}) is None
    assert reader.statements[0][0].endswith(" LIMIT 1")


async def test_read_value_answers_the_column_of_the_first_row_or_none(db: duckdb.DuckDBPyConnection) -> None:
    """Flips if read_value answers another column, or anything but None for no row and for a NULL."""
    conn = ThreadedDuckDB(db)
    assert await read_value(conn, "t", by={"a": "k2", "b": 8}, column="v") == 8
    assert await read_value(conn, "t", by={"a": "k2", "b": 9}, column="v") is None
    assert await read_value(conn, "t", by={"a": "k7", "b": 9}, column="v") is None


async def test_a_read_pinning_nothing_is_refused(db: duckdb.DuckDBPyConnection) -> None:
    """Flips if a point read with no pin is sent: it reads the table whole."""
    with pytest.raises(ValueError, match="pins at least one column"):
        await read_by(ThreadedDuckDB(db), "t", by={})
    with pytest.raises(ValueError, match="names the columns"):
        await read_row(ThreadedDuckDB(db), "t", by={"a": "k1"}, columns=())
