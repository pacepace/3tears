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

import json
from collections.abc import Iterator
from typing import Any, cast
from unittest.mock import MagicMock

import duckdb
import pyarrow as pa
import pytest

from threetears.core.backends import schema_sql
from threetears.core.change_note_fold import RestoreFolder
from threetears.core.backends.protocol import L3Reader
from threetears.core.backends.sql import SqlL3Backend
from threetears.core.collections.schema_backed import BIGINT_TYPE, BOOL_TYPE, STRING_TYPE, Column, TableSchema
from threetears.core.collections.table_restore import StampedTable, TableChanges
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
    Derived,
    Fragment,
    Joined,
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
    Nearest,
    newest_read,
    read_by,
    read_by_joined,
    read_link_chain,
    read_live_rows,
    read_nearest,
    read_past_marks,
    read_reached,
    read_keyed,
    read_keyed_arrow,
    read_row,
    read_step,
    read_step_arrow,
    read_value,
    scan,
    semi,
    update_from,
    update_keyed,
    Match,
)
from threetears.interfaces.keyed.writes import update_keyed_statements
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

#: a stamped table whose rows link to one another (``origin_id``), as the hub's knowledge tables' origin chains do
_CHAIN = "sg_chain"

_CHAIN_SCHEMA = TableSchema(
    name=_CHAIN,
    primary_key=("id",),
    columns=[Column("id", STRING_TYPE), Column("origin_id", STRING_TYPE, nullable=True)],
    row_epoch=ROW_STAMPED,
    l1_eviction=TEST_L1_TABLE,
)

#: the chain's rows, (id, origin_id, gone): a -> b -> c (gone) -> d; x -> y -> z -> x, a cycle; n links nowhere
_CHAIN_ROWS = (
    ("a", "b", False),
    ("b", "c", False),
    ("c", "d", True),
    ("d", None, False),
    ("x", "y", False),
    ("y", "z", False),
    ("z", "x", False),
    ("n", None, False),
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

    # the restore reads every bucket's rows stamped since X = 101: both deletes come back, each by its key
    table = StampedTable(
        schema="main",
        table=_TABLE,
        bucket=BUCKET_COLUMN,
        buckets=RESTORE_BUCKETS,
        keys=("state", "run_id"),
        instants=frozenset(),
        tenancy=Tenancy("undeclared"),
    )
    sent = len(reader.statements)
    folder = RestoreFolder()
    folder.start()
    try:
        changes = [change async for change in TableChanges(reader, [table], 101, folder, None)]
    finally:
        await folder.close()
        await folder.closed()
    restored = sorted(
        (change.table, change.tenant, tuple(row["ids"]))
        for change in changes
        for body in change.bodies
        for row in json.loads(body)["rows"]
    )
    assert restored == [(_TABLE, None, ("CA", "r1")), (_TABLE, None, ("NV", "r1"))], changes
    # one read a bucket, and none names a live guard: they are the reads that see gone rows
    reads = reader.statements[sent:]
    assert len(reads) == RESTORE_BUCKETS, reads
    assert all(f'"{GONE_COLUMN}"' not in read for read in reads), reads


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

    async def test_a_drain_with_gone_reads_the_table_as_stored(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if ``keyset_rows(with_gone=True)`` skips a gone row (CA r2) or names the live guard: a walk that must
        reach deleted rows too (an erasure) drains them as ``keyset_page(with_gone=True)`` pages them."""
        reader = _L3(db)
        drained = await keyset_rows(
            reader,
            _TABLE,
            columns=["state", "run_id", "rows"],
            by={"state": "CA"},
            order=[Sort("run_id")],
            page=1,
            with_gone=True,
        )
        assert _read(drained) == {("CA", "r1"): 10, ("CA", "r2"): 20}
        assert reader.statements and not any('"gone"' in sql for sql in reader.statements), reader.statements

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
    return await read_l3_rows(
        cast(L3Reader, reader), _TABLE, ["state", "run_id", "rows"], ["state", "run_id"], page_size=1
    )


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


class _Sent:
    """an L3 reader (it names no dialect) that keeps each statement and answers no row: a vector read's statement is
    pgvector's, which DuckDB does not run."""

    def __init__(self) -> None:
        self.statements: list[str] = []

    async def fetch(self, query: str, *params: Any) -> list[dict[str, Any]]:
        self.statements.append(query)
        return []


async def test_a_vector_read_of_every_column_answers_the_declared_columns_only() -> None:
    """Flips if a nearest-vector read of a stamped table naming no columns carries its whole row through the wrapper
    (``(n."_nearest_row").*``), which answers its L3 restore bucket and its gone flag, instead of its declared
    columns; or if it reads its gone rows."""
    reader = _Sent()
    await read_nearest(reader, _TABLE, by={"state": "CA"}, nearest=Nearest("rows", "[1]"), limit=3)
    [sql] = reader.statements
    assert "_nearest_row" not in sql and BUCKET_COLUMN not in sql, sql
    outer = sql.split(" FROM (", 1)[0]
    assert outer == "SELECT " + ", ".join(f'n."{column}"' for column in _DECLARED), sql
    assert f'NOT t."{GONE_COLUMN}"' in sql, sql


class TestStatementReaders:
    """the readers that write their own statement (joins, walks, a link chain, a read past marks): each answers a
    stamped table's live rows alone, through the live guard (``keyed.live_read``/``live_only``), or refuses it as it
    refuses a versioned one where one statement reaches rows no live read guards."""

    async def test_a_joined_read_answers_live_rows_alone(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if read_by_joined answers a gone row of the table it pins or of the table it joins: both are keyed
        reads (read_by, read_keyed), each through the live guard."""
        reader = _L3(db)
        rows = await read_by_joined(
            reader,
            _TABLE,
            by={"state": "CA"},
            columns=["state", "run_id", "rows"],
            joins=[Joined(_TABLE, on={"state": "state", "run_id": "run_id"}, columns={"again": "rows"}, left=True)],
        )
        assert [(row["run_id"], row["again"]) for row in rows] == [("r1", 10)]
        joined = await read_by_joined(
            reader,
            _TABLE,
            by={"state": "NV"},
            columns=["state", "run_id"],
            joins=[Joined(_TABLE, on={"run_id": "run_id"}, columns={"other_state": "state"})],
        )
        assert sorted(row["other_state"] for row in joined) == ["CA", "NV"]

    async def test_a_walk_a_level_a_read_answers_live_rows_alone(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a walk read a level a read (read_by, then read_keyed) reaches a gone row."""
        reader = _L3(db)
        reached = await read_reached(
            reader, _TABLE, start={"state": "CA"}, on="run_id", to="run_id", columns=["state", "run_id"]
        )
        assert sorted((row["state"], row["run_id"]) for row in reached) == sorted(_LIVE)

    async def test_a_link_chain_of_a_stamped_table_walks_its_live_rows_once_in_one_statement(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        """Flips if a link chain of a stamped L3 table is refused, reaches a gone row or past it (c, and d beyond it),
        walks a cycle round (x y z answered twice under the bound), or is sent as more than one statement: the hub's
        origin chains of its stamped knowledge tables walk this way."""
        db.execute(f"CREATE TABLE {_CHAIN} (id VARCHAR PRIMARY KEY, origin_id VARCHAR, gone BOOLEAN NOT NULL)")
        db.executemany(f"INSERT INTO {_CHAIN} VALUES ($1, $2, $3)", [list(row) for row in _CHAIN_ROWS])
        reader = _L3(db)

        async def walked(start: str, most: int) -> list[str]:
            chained = await read_link_chain(
                reader, _CHAIN, key="id", link="origin_id", start=start, most=most, columns=["id", "origin_id"]
            )
            return [row["id"] for row in chained]

        assert await walked("a", 10) == ["a", "b"]  # stops at the gone row c, never reaching d
        assert await walked("c", 10) == []  # a gone start is no chain
        assert sorted(await walked("x", 10)) == ["x", "y", "z"]  # a cycle ends where it comes back
        assert sorted(await walked("y", 2)) == ["y", "z"]  # most rows
        assert await walked("n", 10) == ["n"]  # a NULL link ends it
        assert await walked("missing", 10) == []
        assert len(reader.statements) == 6
        assert all(sql.count('NOT t."gone"') == 2 and sql.startswith("WITH RECURSIVE") for sql in reader.statements)
        # a local copy holds live rows alone and no gone flag: its chain is read as it holds, a cycle walked round
        local = await read_link_chain(
            ThreadedDuckDB(db), _CHAIN, key="id", link="origin_id", start="x", most=4, columns=["id"]
        )
        assert [row["id"] for row in local] == ["x", "y", "z", "x"]

    async def test_a_walk_in_one_statement_reads_a_stamped_tables_live_rows_alone(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        """Flips if a walk in one statement over a stamped table reaches a gone row (CA r2 here, and so the r2 level
        beyond it): each level reads the table's live rows alone, ``NOT t."gone"`` on its first rows and on every
        level's join. 3tears' own access tables are stamped, and the visible-customer walk is one statement."""
        reader = _L3(db)
        reached = await read_reached(
            reader, _TABLE, start={"state": "CA"}, on="run_id", to="run_id", columns=["state", "run_id"], depth=2
        )
        assert sorted((row["state"], row["run_id"]) for row in reached) == sorted(_LIVE)
        [sql] = reader.statements
        assert sql.count('NOT t."gone"') == 2, sql
        # a local copy holds live rows alone and no gone flag: its walk names none
        local = await read_reached(
            ThreadedDuckDB(db), _TABLE, start={"state": "NV"}, on="run_id", to="run_id", columns=["run_id"], depth=1
        )
        assert [row["run_id"] for row in local] == ["r1"]

    async def test_a_ranked_read_joining_a_stamped_table_joins_its_live_rows_alone(self) -> None:
        """Flips if a ranked read joining a stamped table in one statement joins its gone rows: the lookup of the
        joined table carries its live guard (``NOT j."gone"``)."""
        reader = _Sent()
        await read_nearest(
            reader,
            "sg_other",
            by={"state": "CA"},
            nearest=Nearest("rows", "[1]", similarity="score"),
            limit=3,
            columns=["state"],
            joins=[Joined(_TABLE, on={"state": "state"}, columns={"j_rows": "rows"}, left=True)],
        )
        [sql] = reader.statements
        assert f'FROM "{_TABLE}" AS j WHERE' in sql and 'NOT j."gone"' in sql, sql

    async def test_a_read_past_marks_answers_live_rows_alone(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a read of a stamped table's change record past its marks answers a row its delete marked gone
        (its epoch past the mark like any live row's)."""
        reader = _L3(db)
        moved = await read_past_marks(
            reader, _TABLE, columns=["state", "run_id", "rows", EPOCH_COLUMN], marks={EPOCH_COLUMN: 0}
        )
        assert _read(moved.rows) == _LIVE
        types = {"state": pa.string(), "run_id": pa.string(), "rows": pa.int64(), EPOCH_COLUMN: pa.int64()}
        arrow = await read_past_marks(
            reader, _TABLE, columns=["state", "run_id", "rows", EPOCH_COLUMN], marks={EPOCH_COLUMN: 0}, types=types
        )
        assert arrow.table is not None
        assert _read(arrow.table.to_pylist()) == _LIVE


# -- a cold start's fill from a stamped table's live rows ------------------------------------------------------------


class TestReadLiveRows:
    async def test_a_cold_fill_reads_the_live_rows_alone_in_one_statement(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if the cold fill answers a row its delete marked gone, filters it in Python rather than in the
        statement, or pages."""
        reader = _L3(db)
        rows = await read_live_rows(reader, _TABLE, ("state", "run_id", "rows"))
        assert _read(rows) == _LIVE
        [statement] = reader.statements
        assert statement == f"SELECT state, run_id, rows FROM {_TABLE} WHERE NOT gone", statement

    async def test_a_table_no_stamped_declaration_names_is_refused(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if the helper filters on a gone flag a table never declared: its deleted rows are not marked by it."""
        reader = _L3(db)
        with pytest.raises(ValueError, match="not a stamped L3 table"):
            await read_live_rows(reader, "sg_undeclared", ("state",))
        assert reader.statements == []

    async def test_a_name_that_is_no_plain_identifier_is_refused(self, db: duckdb.DuckDBPyConnection) -> None:
        reader = _L3(db)
        with pytest.raises(ValueError, match="plain identifier"):
            await read_live_rows(reader, _TABLE, ("state; DROP TABLE x",))
        assert reader.statements == []


# -- a write by key writes live rows alone ----------------------------------------------------------------------------


class TestKeyedWritesSkipGoneRows:
    async def test_an_update_by_key_leaves_a_gone_row_as_it_is_and_counts_none(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        """Flips if a keyed update of a stamped L3 table changes a row a delete marked gone, or counts it: a write by
        its key, and a batch naming every key, write the live rows alone."""
        reader = _L3(db)
        written = await update_keyed(reader, _TABLE, values={"rows": 99}, by={"state": "CA", "run_id": "r2"})
        assert written.count == 0
        assert _row(db, "CA", "r2") == (20, 100, True)
        batch = await update_keyed(reader, _TABLE, values={"rows": 7}, keys=_every_key(), returning=("state", "run_id"))
        assert batch.count == 2
        assert sorted((row["state"], row["run_id"]) for row in batch.rows) == [("CA", "r1"), ("NV", "r1")]
        assert _row(db, "CA", "r2") == (20, 100, True)
        assert _row(db, "CA", "r1") == (7, 100, False)
        assert all(f'NOT "{GONE_COLUMN}"' in sql for sql in reader.statements), reader.statements

    async def test_a_revive_says_so_and_writes_the_gone_row(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if an update naming with_gone=True (a revive of a gone key) cannot reach the gone row."""
        reader = _L3(db)
        written = await update_keyed(
            reader,
            _TABLE,
            values={GONE_COLUMN: False, "rows": 21},
            by={"state": "CA", "run_id": "r2"},
            where=[Fragment(f'"{GONE_COLUMN}"')],
            with_gone=True,
        )
        assert written.count == 1
        assert _row(db, "CA", "r2") == (21, 100, False)

    async def test_a_revive_that_does_not_say_so_is_refused_before_anything_is_sent(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        """Flips if an update setting a stamped table's gone flag false without with_gone is sent: written to live
        rows alone it would find no gone row, a revive that silently does nothing."""
        reader = _L3(db)
        with pytest.raises(ValueError, match="with_gone=True"):
            await update_keyed(
                reader,
                _TABLE,
                values={GONE_COLUMN: False},
                by={"state": "CA", "run_id": "r2"},
                where=[Fragment('"gone"')],
            )
        assert reader.statements == []
        assert _row(db, "CA", "r2") == (20, 100, True)

    async def test_an_update_from_a_source_writes_live_rows_alone(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if an update of a stamped L3 table from a keyed source writes a gone row, or with_gone does not."""
        reader = _L3(db)
        source = Derived("SELECT 'CA' AS state, run_id, 5 AS rows FROM (VALUES ('r1'), ('r2')) AS v(run_id)")
        written = await update_from(reader, _TABLE, key=["state", "run_id"], source=source, columns=["rows"])
        assert written.count == 1
        assert (_row(db, "CA", "r1"), _row(db, "CA", "r2")) == ((5, 100, False), (20, 100, True))
        revived = await update_from(
            reader, _TABLE, key=["state", "run_id"], source=source, columns=["rows"], with_gone=True
        )
        assert revived.count == 2
        assert _row(db, "CA", "r2") == (5, 100, True)

    def test_the_statements_state_the_live_guard_once(self) -> None:
        """Flips if a keyed update's statements of a stamped table on L3 carry no live guard, carry it twice beside a
        caller's own (a gone mark's), or carry it for a write with_gone or one not to L3 (a local copy)."""
        guard = f'NOT "{GONE_COLUMN}"'
        for where in ((), (Fragment(guard),)):
            [(sql, _)] = update_keyed_statements(_TABLE, values={"rows": 1}, by={"state": "CA"}, where=where)
            assert sql.count(guard) == 1, sql
        for with_gone, l3 in ((True, True), (False, False)):
            [(sql, _)] = update_keyed_statements(
                _TABLE, values={"rows": 1}, by={"state": "CA"}, with_gone=with_gone, l3=l3
            )
            assert GONE_COLUMN not in sql, sql

    async def test_a_local_copy_s_update_writes_as_it_holds(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if an update that does not go to L3 (a local copy, filled with live rows alone) is guarded or
        refused."""
        written = await update_keyed(ThreadedDuckDB(db), _TABLE, values={"rows": 1}, by={"state": "CA", "run_id": "r2"})
        assert written.count == 1
