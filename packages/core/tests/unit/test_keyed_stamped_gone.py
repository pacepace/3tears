"""a stamped table's delete is a gone row a restore hears, never a DELETE; every shared read skips it, and no read
answers its L3 restore bucket (append-only L3, decision 2: "deletes become markers everywhere, because a real delete
leaves nothing for a restore to find").

A stamped table (``TableSchema(row_epoch=ROW_STAMPED)``) keeps one row a key, updated in place. Its delete updates the
key's row in place: ``gone`` set, the epoch risen past the delete's stamp (``GREATEST(epoch + 1, <stamp>)``), so the
restore's bucket read (``epoch_bucket = b AND epoch >= X``) finds it and the key comes back. The seeded table:

- ``("CA", "r1")``: live, epoch 100, bucket 3;
- ``("CA", "r2")``: gone, epoch 100, bucket 5: never read;
- ``("NV", "r1")``: live, epoch 100, bucket 3.

Correctness runs on an in-process DuckDB behind a reader that names no dialect, so the helpers take it for L3 (the live
guard is L3's alone: a local copy is filled from reads that skip gone rows). Its ``epoch_bucket`` is a plain column here
(DuckDB stores no generated column): the L3 table's own, never one a read answers. The plans on PostgreSQL and
YugabyteDB are the integration pass's (``test_table_restore_plans_live``).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock

import duckdb
import pyarrow as pa
import pytest

from threetears.core.backends import schema_sql
from threetears.core.backends.sql import SqlL3Backend
from threetears.core.collections.schema_backed import BIGINT_TYPE, BOOL_TYPE, STRING_TYPE, Column, TableSchema
from threetears.core.collections.table_restore import StampedTable, _bucket_rows
from threetears.core.ddl.row_epoch import (
    BUCKET_COLUMN,
    EPOCH_COLUMN,
    GONE_COLUMN,
    RESTORE_BUCKETS,
    ROW_STAMPED,
    AppendOnlyWriteError,
)
from threetears.core.keyed import (
    AnyOf,
    Count,
    KeyBatch,
    Sort,
    Sum,
    ThreadedDuckDB,
    aggregate_keyed,
    any_rows,
    count_at_most,
    counts_by,
    delete_before,
    delete_keyed,
    empty_tables,
    holds_any,
    keyset_page,
    keyset_rows,
    newest_read,
    read_by,
    read_keyed,
    read_keyed_arrow,
    read_row,
    read_step,
    read_step_arrow,
    read_value,
    scan,
    semi,
    Match,
)
from threetears.core.collections.complete_copy import read_l3_rows
from threetears.core.tenancy import Tenancy
from threetears.core.testing.l1_tables import TEST_L1_TABLE

_TABLE = "sg_runs"
_GONE_ONLY = "sg_gone_only"

_RUNS = TableSchema(
    name=_TABLE,
    primary_key=("state", "run_id"),
    columns=[Column("state", STRING_TYPE), Column("run_id", STRING_TYPE), Column("rows", BIGINT_TYPE)],
    row_epoch=ROW_STAMPED,
    l1_eviction=TEST_L1_TABLE,
)

_GONE_ONLY_SCHEMA = TableSchema(
    name=_GONE_ONLY,
    primary_key=("state", "run_id"),
    columns=[Column("state", STRING_TYPE), Column("run_id", STRING_TYPE), Column("rows", BIGINT_TYPE)],
    row_epoch=ROW_STAMPED,
    l1_eviction=TEST_L1_TABLE,
)

#: the declared columns a read of every column answers: never the gone flag, never the L3 restore bucket
_DECLARED = ["state", "run_id", "rows", EPOCH_COLUMN]

#: the live rows, (state, run_id) -> rows
_LIVE = {("CA", "r1"): 10, ("NV", "r1"): 30}

_EVERY_KEY = [("CA", "r1"), ("CA", "r2"), ("NV", "r1")]

_DDL = (
    "(state VARCHAR NOT NULL, run_id VARCHAR NOT NULL, rows BIGINT NOT NULL, epoch BIGINT NOT NULL, "
    "gone BOOLEAN NOT NULL DEFAULT false, epoch_bucket SMALLINT NOT NULL, PRIMARY KEY (state, run_id))"
)


@pytest.fixture
def db() -> Iterator[duckdb.DuckDBPyConnection]:
    """the stamped table as the module's docstring seeds it, and a table of one gone row."""
    conn = duckdb.connect()
    conn.execute(f"CREATE TABLE {_TABLE} {_DDL}")
    conn.execute(f"CREATE TABLE {_GONE_ONLY} {_DDL}")
    conn.execute(
        f"INSERT INTO {_TABLE} (state, run_id, rows, epoch, gone, epoch_bucket) VALUES "
        "('CA', 'r1', 10, 100, false, 3), ('CA', 'r2', 20, 100, true, 5), ('NV', 'r1', 30, 100, false, 3)"
    )
    conn.execute(
        f"INSERT INTO {_GONE_ONLY} (state, run_id, rows, epoch, gone, epoch_bucket) VALUES ('CA', 'r1', 1, 9, true, 0)"
    )
    yield conn
    conn.close()


class _L3:
    """an L3 reader answering from DuckDB: it names no dialect, so every helper takes it for L3, as the rail's proxy
    or an asyncpg pool; it keeps every statement it was sent. ``copy_from_query`` answers a read as PostgreSQL's
    ``COPY ... CSV`` writes it (each forced column quoted, a NULL an empty field), so the restore's bucket read runs."""

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._conn = conn
        self._inner = ThreadedDuckDB(conn)
        self.statements: list[str] = []

    async def fetch(self, query: str, *params: Any) -> list[dict[str, Any]]:
        self.statements.append(query)
        return await self._inner.fetch(query, *params)

    async def execute(self, query: str, *params: Any) -> str:
        self.statements.append(query)
        return await self._inner.execute(query, *params)

    async def copy_from_query(self, query: str, *params: Any, output: Any, format: str, force_quote: Any = ()) -> str:  # noqa: A002 -- asyncpg's own name
        assert format == "csv"
        self.statements.append(query)
        rows = self._conn.execute(query, list(params)).fetchall()
        for row in rows:
            fields = ['"' + str(value).replace('"', '""') + '"' if value is not None else "" for value in row]
            await output((",".join(fields) + "\n").encode())
        return f"COPY {len(rows)}"


def _read(rows: Any) -> dict[tuple[str, str], int]:
    """the rows read, (state, run_id) -> rows, each key once."""
    answered: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["state"], row["run_id"])
        assert key not in answered, f"{key} read twice"
        answered[key] = int(row["rows"])
    return answered


def _every_key() -> KeyBatch:
    return KeyBatch(("state", "run_id"), [[k[0] for k in _EVERY_KEY], [k[1] for k in _EVERY_KEY]], ("text", "text"))


def _store() -> SqlL3Backend:
    store = SqlL3Backend(MagicMock())
    store.register_schema(_TABLE, _RUNS)
    return store


def _row(db: duckdb.DuckDBPyConnection, state: str, run_id: str) -> tuple[Any, ...] | None:
    return db.execute(
        f"SELECT rows, epoch, gone FROM {_TABLE} WHERE state = $1 AND run_id = $2", [state, run_id]
    ).fetchone()


# -- the declaration --------------------------------------------------------------------------------------------------


def test_a_stamped_table_declares_its_gone_flag_and_needs_no_live_check() -> None:
    """Flips if a stamped table's L3 DDL carries no gone flag (false unless a delete set it), or a versioned table's
    live CHECK (a gone stamped row keeps every column its last write left)."""
    gone = _RUNS.column(GONE_COLUMN)
    assert (gone.column_type, gone.nullable, gone.server_default, gone.immutable) == (BOOL_TYPE, False, "false", False)
    for yugabyte in (False, True):
        create = _RUNS.create_statements(yugabyte=yugabyte)[0]
        assert "gone BOOLEAN DEFAULT false NOT NULL" in create, create
        assert f"{_TABLE}_live" not in create and "rows BIGINT NOT NULL" in create, create


# -- a delete is a gone row the restore hears -------------------------------------------------------------------------


async def test_a_delete_marks_the_row_gone_in_place_and_the_restore_hears_it(db: duckdb.DuckDBPyConnection) -> None:
    """Flips if a stamped table's delete is a DELETE (the row is no longer there for a restore), or is not one
    UPDATE a batch setting the flag and raising the epoch past both the row's and the delete's stamp, or if the
    restore's bucket read of epoch >= X misses the gone row."""
    reader = _L3(db)
    store = _store()
    # a delete stamped below the row's own epoch: the row's epoch still rises (GREATEST(epoch + 1, stamp))
    assert await store.mark_gone(_TABLE, [("CA", "r1")], epoch=50, scope=None, max_rows=10, conn=reader) == 1
    # one stamped above it: the row takes the delete's stamp
    assert await store.mark_gone(_TABLE, [("NV", "r1")], epoch=500, scope=None, max_rows=10, conn=reader) == 1
    assert not [sql for sql in reader.statements if sql.lstrip().upper().startswith("DELETE")], reader.statements
    [first, second] = reader.statements
    assert first.startswith(
        f'UPDATE "{_TABLE}" SET "{GONE_COLUMN}" = $1, "{EPOCH_COLUMN}" = GREATEST("{EPOCH_COLUMN}" + 1, $2::bigint) '
    ), first
    assert f'NOT "{GONE_COLUMN}"' in first and first == second
    # one row a key, as before: kept, marked, its epoch risen
    assert _row(db, "CA", "r1") == (10, 101, True)
    assert _row(db, "NV", "r1") == (30, 500, True)
    # no read sees them now
    assert await read_by(reader, _TABLE, by={"state": "NV"}, columns=["run_id"]) == []

    # the restore reads bucket 3's rows stamped since X = 101: both deletes come back, each by its key
    table = StampedTable(
        schema="main",
        table=_TABLE,
        bucket=BUCKET_COLUMN,
        buckets=RESTORE_BUCKETS,
        keys=("state", "run_id"),
        instants=frozenset(),
        tenancy=Tenancy("undeclared"),
    )
    text = await _bucket_rows(reader, table, 3, 101)
    lines = sorted(text.decode().splitlines())
    assert lines == ['"""sg_runs""",,"[""CA"",""r1""]",', '"""sg_runs""",,"[""NV"",""r1""]",'], lines
    # and the read names no live guard: it is the one read that sees gone rows
    assert f'"{GONE_COLUMN}"' not in reader.statements[-1]


async def test_a_delete_of_a_key_already_gone_leaves_it_as_it_is(db: duckdb.DuckDBPyConnection) -> None:
    """Flips if a second delete of a gone key moves its epoch (the restore would hear a delete twice)."""
    await _store().mark_gone(_TABLE, [("CA", "r2")], epoch=900, scope=None, max_rows=10, conn=_L3(db))
    assert _row(db, "CA", "r2") == (20, 100, True)


def test_a_write_of_the_key_sets_it_live_again() -> None:
    """Flips if a stamped table's upsert leaves a gone row gone: its SET names the flag a write stamps false."""
    sql = schema_sql.build_insert_sql(
        _RUNS, {"state": "CA", "run_id": "r2", "rows": 1, EPOCH_COLUMN: 7, GONE_COLUMN: False}
    )
    assert f"{GONE_COLUMN} = EXCLUDED.{GONE_COLUMN}" in sql, sql
    assert f"{EPOCH_COLUMN} = GREATEST({_TABLE}.{EPOCH_COLUMN} + 1, EXCLUDED.{EPOCH_COLUMN})" in sql, sql


# -- a real DELETE is refused -----------------------------------------------------------------------------------------


async def test_a_real_delete_of_a_stamped_table_is_refused_before_anything_is_sent(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """Flips if any delete path of a stamped L3 table sends a DELETE: the store's key deletes, its Arrow door's delete,
    its prefix delete, a keyed delete or a sweep."""
    reader = _L3(db)
    store = _store()
    with pytest.raises(AppendOnlyWriteError, match="stamped"):
        await store.delete_many(_TABLE, [("CA", "r1")], max_rows=10, conn=reader)
    with pytest.raises(AppendOnlyWriteError, match="stamped"):
        await store.delete_many_returning(_TABLE, [("CA", "r1")], max_rows=10, conn=reader)
    with pytest.raises(AppendOnlyWriteError, match="stamped"):
        await store.delete(_TABLE, {"state": "CA", "run_id": "r1"}, conn=reader)
    with pytest.raises(AppendOnlyWriteError, match="stamped"):
        await store.delete_prefixes(_TABLE, [{"state": "CA"}], conn=reader)
    with pytest.raises(AppendOnlyWriteError, match="stamped"):
        store.arrow_write_plan(
            _TABLE, ["state", "run_id"], [pa.string(), pa.string()], kind="delete", conn=MagicMock(write_arrow=True)
        )
    with pytest.raises(AppendOnlyWriteError, match="stamped"):
        await delete_keyed(reader, _TABLE, by={"state": "CA", "run_id": "r1"})
    with pytest.raises(AppendOnlyWriteError, match="stamped"):
        await delete_before(
            _TABLE, collection=None, key=("state", "run_id"), column=EPOCH_COLUMN, before=1, page=10, conn=reader
        )
    assert reader.statements == []
    assert db.execute(f"SELECT count(*) FROM {_TABLE}").fetchone() == (3,)


async def test_a_local_copy_s_keyed_delete_is_its_own(db: duckdb.DuckDBPyConnection) -> None:
    """Flips if a delete that does not go to L3 (a local copy, filled with live rows alone) is refused."""
    await delete_keyed(ThreadedDuckDB(db), _TABLE, by={"state": "CA", "run_id": "r1"})
    assert db.execute(f"SELECT count(*) FROM {_TABLE}").fetchone() == (2,)


# -- every shared read skips a gone row -------------------------------------------------------------------------------


class TestReadsSkipGoneRows:
    async def test_keyed_reads_answer_live_rows_alone(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a keyed batch read (dicts or Arrow) answers a gone row."""
        reader = _L3(db)
        rows = await read_keyed(reader, _TABLE, keys=_every_key(), columns=["state", "run_id", "rows"])
        assert _read(rows) == _LIVE
        types = {"state": pa.string(), "run_id": pa.string(), "rows": pa.int64()}
        tables = await read_keyed_arrow(reader, _TABLE, keys=_every_key(), types=types)
        assert _read(pa.concat_tables(tables).to_pylist()) == _LIVE

    async def test_point_reads_answer_live_rows_alone(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a point read, a value read or a bounded look answers a gone row."""
        reader = _L3(db)
        assert _read(await read_by(reader, _TABLE, by={"state": "CA"}, columns=["state", "run_id", "rows"])) == {
            ("CA", "r1"): 10
        }
        assert await read_row(reader, _TABLE, by={"state": "CA", "run_id": "r2"}, columns=["rows"]) is None
        assert await read_value(reader, _TABLE, by={"state": "CA", "run_id": "r2"}, column="rows") is None
        assert await read_value(reader, _TABLE, by={"state": "CA", "run_id": "r1"}, column="rows") == 10
        looked = await any_rows(reader, _TABLE, columns=["run_id"], by={"state": "CA"}, limit=5)
        assert looked == [{"run_id": "r1"}]

    async def test_pages_answer_live_rows_alone(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a page, a drain or a whole read's pager (cold start, refill) answers a gone row."""
        reader = _L3(db)
        order = [Sort("state"), Sort("run_id")]
        read: list[dict[str, Any]] = []
        cursor = None
        while True:
            page = await keyset_page(
                reader, _TABLE, columns=["state", "run_id", "rows"], by={}, order=order, limit=1, cursor=cursor
            )
            read += page.rows
            if page.next_cursor is None:
                break
            cursor = page.next_cursor
        assert _read(read) == _LIVE
        drained = await keyset_rows(
            reader, _TABLE, columns=["state", "run_id", "rows"], by={"state": "CA"}, order=[Sort("run_id")], page=1
        )
        assert _read(drained) == {("CA", "r1"): 10}
        whole = await _whole_read(reader)
        assert _read(whole) == _LIVE

    async def test_counts_count_live_rows_alone(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a count, a probe or an aggregate takes a gone row in."""
        reader = _L3(db)
        assert await counts_by(reader, _TABLE, column="state", values=AnyOf(["CA", "NV"], "text")) == {"CA": 1, "NV": 1}
        assert (
            await count_at_most(reader, _TABLE, at_most=10, order=["state", "run_id"], by={"state": "CA"})
        ).count == 1
        assert not await holds_any(reader, _TABLE, order=["state", "run_id"], by={"state": "CA", "run_id": "r2"})
        assert await empty_tables(reader, [_TABLE, _GONE_ONLY]) == [_GONE_ONLY]
        rows = await aggregate_keyed(
            reader, _TABLE, aggregates={"n": Count(), "total": Sum("t.rows")}, by={"state": "CA"}
        )
        assert [(int(row["n"]), int(row["total"])) for row in rows] == [(1, 10)]

    def test_the_store_s_key_reads_carry_the_live_guard(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if the store's point read, whole-key read or key-led read of a stamped table answers a gone row."""
        point = schema_sql.build_fetch_sql(_RUNS)
        assert db.execute(point, ["CA", "r1"]).fetchall() == [("CA", "r1", 10, 100)]
        assert db.execute(point, ["CA", "r2"]).fetchall() == []
        led = schema_sql.build_led_by_select_sql(_RUNS, ["state", "run_id", "rows"])
        assert sorted(db.execute(led, [["CA", "NV"]]).fetchall()) == [("CA", "r1", 10), ("NV", "r1", 30)]
        assert schema_sql.build_whole_key_select_sql(_RUNS).endswith(f" AND NOT {GONE_COLUMN}")

    async def test_a_local_copy_is_read_as_it_holds(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a read that does not go to L3 (a local copy) is sent the live guard."""
        rows = await read_by(ThreadedDuckDB(db), _TABLE, by={"state": "CA"}, columns=["run_id"])
        assert sorted(row["run_id"] for row in rows) == ["r1", "r2"]


async def _whole_read(reader: _L3) -> list[dict[str, Any]]:
    """a whole read's pager, as complete_copy and the store's scan read a table a page at a time."""
    return await read_l3_rows(reader, _TABLE, ["state", "run_id", "rows"], ["state", "run_id"], page_size=1)


# -- no read answers the L3 restore bucket ----------------------------------------------------------------------------


class TestDeclaredColumnsOnly:
    async def test_a_read_of_every_column_answers_the_declared_columns_only(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        """Flips if a keyed read of every column of a stamped table (``t.*``) answers its L3 restore bucket or its gone
        flag: a page, a point read, a step, the store's projection."""
        reader = _L3(db)
        page = await keyset_page(reader, _TABLE, columns=None, by={"state": "CA"}, order=[Sort("run_id")], limit=5)
        assert [list(row) for row in page.rows] == [_DECLARED]
        [row] = await read_by(reader, _TABLE, by={"state": "CA", "run_id": "r1"})
        assert list(row) == _DECLARED
        held = await read_row(reader, _TABLE, by={"state": "NV", "run_id": "r1"})
        assert held is not None and list(held) == _DECLARED
        # a step reads a stamped L3 table through its live relation (a plain one is refused: TestSteps)
        stepped = await read_step(reader, scan(newest_read(reader, _TABLE), where=()))
        assert {tuple(row) for row in stepped} == {tuple(_DECLARED)}
        projected = schema_sql.build_select_column_list(_RUNS).split(", ")
        assert BUCKET_COLUMN not in projected and GONE_COLUMN not in projected
        assert projected == _DECLARED


# -- a generic step: refused on a stamped L3 table named plainly, its live rows through newest_read ---------------------


class TestSteps:
    async def test_a_step_over_a_stamped_table_on_l3_is_refused_or_reads_its_live_rows(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        """Flips if a step naming a stamped L3 table as a plain relation runs (it answers the gone rows), if a step
        over a local copy is refused, or if a step over the table's live relation (newest_read) answers a gone row."""
        reader = _L3(db)
        with pytest.raises(ValueError, match="newest_read"):
            await read_step(reader, scan(_TABLE, ["state", "run_id", "rows"]))
        with pytest.raises(ValueError, match="newest_read"):
            await read_step_arrow(reader, _TABLE, types={"state": pa.string()})
        keys = scan(_TABLE, ["state", "run_id"])
        with pytest.raises(ValueError, match="newest_read"):
            await read_step(reader, semi(_GONE_ONLY, keys, on=[Match("state"), Match("run_id")]))
        assert len(await read_step(ThreadedDuckDB(db), scan(_TABLE, ["rows"]))) == 3
        rows = await read_step(reader, scan(newest_read(reader, _TABLE), ["state", "run_id", "rows"]))
        assert _read(rows) == _LIVE
