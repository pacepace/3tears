"""every shared read of a versioned L3 table hides each version older than the table's newest reset marker.

A versioned table reset with no ``TRUNCATE`` (``SchemaBackedCollection.reset_table``) is ONE row of
``threetears_run_markers``: ``scope`` chr(30), an empty ``prefix``, the reset's epoch. Every read must then answer no
version older than the newest such marker of the table, and every version at or after it (the rows the reset's own
transaction appended after the marker). The table here is keyed by a race, its scope (the state) no part of its key,
so no run marker can name it: the reset marker is the only way to empty it in place. The seeded table, reset at 30
(and once before, at 5):

- ``("r1", 1)``: a version at 10, then one at 40 (after the reset): only ``v = 2`` is read;
- ``("r1", 2)``: one version at 10: never read;
- ``("r2", 1)``: a version at 20, then a "gone" marker at 25: never read;
- ``("r2", 2)``: one version at 50: read;
- ``("r3", 1)``: one version at 20: never read, and race ``r3`` holds nothing a read sees.

A second table, never reset, shares the run-marker table: its rows at 10 stay read (a reset cuts its own table alone).

Correctness runs on an in-process DuckDB behind a reader that names no dialect, as ``test_keyed_newest_versions``
does; the marker is written by the writer's own statement (``table_reset_marker_sql``). The keyed plan on PostgreSQL
and YugabyteDB is the integration pass's (``test_reset_marker_reads_live``).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import duckdb
import pyarrow as pa
import pytest

from threetears.core.backends import schema_sql
from threetears.core.collections.schema_backed import INT_TYPE, STRING_TYPE, Column, TableSchema
from threetears.core.ddl.row_epoch import row_versions
from threetears.core.ddl.run_markers import table_reset_marker_sql
from threetears.core.keyed import (
    AnyOf,
    Count,
    KeyBatch,
    Sort,
    Sum,
    ThreadedDuckDB,
    aggregate_keyed,
    count_at_most,
    counts_by,
    empty_tables,
    holds_any,
    keyset_page,
    keyset_rows,
    newest_read,
    read_after,
    read_by,
    read_keyed,
    read_keyed_arrow,
    read_past_marks,
    read_row,
    read_step,
    read_value,
    scan,
)
from threetears.core.keyed import rows as keyed_rows
from threetears.core.keyset import keyset_arrow_pages, keyset_scope_pages, keyset_scope_values, read_keyset_pages
from threetears.core.testing.l1_tables import TEST_L1_TABLE
from threetears.interfaces.keyed.keys import versioned_table

_TABLE = "rs_results"
_OTHER = "rs_other"

_RESULTS = TableSchema(
    name=_TABLE,
    primary_key=("race", "id"),
    columns=[
        Column("race", STRING_TYPE),
        Column("id", INT_TYPE),
        Column("state", STRING_TYPE),
        Column("v", INT_TYPE, nullable=True),
    ],
    row_epoch=row_versions("state"),
    l1_eviction=TEST_L1_TABLE,
)

_OTHER_SCHEMA = TableSchema(
    name=_OTHER,
    primary_key=("race", "id"),
    columns=[
        Column("race", STRING_TYPE),
        Column("id", INT_TYPE),
        Column("state", STRING_TYPE),
        Column("v", INT_TYPE, nullable=True),
    ],
    row_epoch=row_versions("state"),
    l1_eviction=TEST_L1_TABLE,
)

#: each key's newest live version after the reset, (race, id) -> v
_NEWEST = {("r1", 1): 2, ("r2", 2): 5}

#: every key the table holds a version of
_EVERY_KEY = [("r1", 1), ("r1", 2), ("r2", 1), ("r2", 2), ("r3", 1)]

_DDL = (
    "(race VARCHAR, id INTEGER, state VARCHAR, v INTEGER, epoch BIGINT, gone BOOLEAN NOT NULL DEFAULT false, "
    "PRIMARY KEY (race, id, epoch))"
)


@pytest.fixture
def db() -> Iterator[duckdb.DuckDBPyConnection]:
    """the table as the module's docstring seeds it, its reset markers written by the writer's statement."""
    conn = duckdb.connect()
    conn.execute(f"CREATE TABLE {_TABLE} {_DDL}")
    conn.execute(f"CREATE TABLE {_OTHER} {_DDL}")
    conn.execute(
        "CREATE TABLE threetears_run_markers (table_name VARCHAR, scope VARCHAR, prefix VARCHAR, epoch BIGINT)"
    )
    conn.execute(
        f"INSERT INTO {_TABLE} (race, id, state, v, epoch, gone) VALUES "
        "('r1', 1, 'NC', 1, 10, false), ('r1', 1, 'NC', 2, 40, false), "
        "('r1', 2, 'NC', 3, 10, false), "
        "('r2', 1, 'VA', 4, 20, false), ('r2', 1, 'VA', NULL, 25, true), "
        "('r2', 2, 'VA', 5, 50, false), "
        "('r3', 1, 'VA', 6, 20, false)"
    )
    conn.execute(f"INSERT INTO {_OTHER} (race, id, state, v, epoch, gone) VALUES ('r1', 1, 'NC', 7, 10, false)")
    conn.execute(table_reset_marker_sql(), [_TABLE, 5])
    conn.execute(table_reset_marker_sql(), [_TABLE, 30])
    yield conn
    conn.close()


class _L3:
    """an L3 reader answering from DuckDB: it names no dialect, so every helper takes it for L3; it keeps every
    statement it was sent."""

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._inner = ThreadedDuckDB(conn)
        self.statements: list[str] = []

    async def fetch(self, query: str, *params: Any) -> list[dict[str, Any]]:
        self.statements.append(query)
        return await self._inner.fetch(query, *params)

    async def execute(self, query: str, *params: Any) -> str:
        self.statements.append(query)
        return await self._inner.execute(query, *params)


def _read(rows: Any) -> dict[tuple[str, int], int]:
    """the rows read, (race, id) -> v, each key once."""
    answered: dict[tuple[str, int], int] = {}
    for row in rows:
        key = (row["race"], int(row["id"]))
        assert key not in answered, f"{key} read twice"
        answered[key] = int(row["v"])
    return answered


def _every_key() -> KeyBatch:
    return KeyBatch(("race", "id"), [[k[0] for k in _EVERY_KEY], [k[1] for k in _EVERY_KEY]], ("text", "integer"))


_TYPES = {"race": pa.string(), "id": pa.int32(), "v": pa.int32()}


class TestTheDeclaration:
    def test_a_versioned_table_keyed_by_no_scope_is_read_with_its_reset_cut(self) -> None:
        """Flips if a versioned table whose key names no scope (no run marker can name it) is read with no reset
        cut, or a table's reset markers are looked for anywhere but the run-marker table."""
        declared = versioned_table(_TABLE, l3=True)
        assert declared is not None
        assert declared.markers is None
        assert declared.resets == "threetears_run_markers"


class TestKeyedReads:
    @pytest.mark.parametrize("yugabyte", [True, False])
    async def test_read_keyed_hides_every_version_older_than_the_reset(
        self, db: duckdb.DuckDBPyConnection, monkeypatch: pytest.MonkeyPatch, yugabyte: bool
    ) -> None:
        """Flips if a keyed batch, in either engine's form, answers a version older than the table's newest reset
        marker, or hides one written after it; or if the as-stored read cleanup makes (with_gone, the versions it
        removes) is cut by the reset."""

        async def engine(conn: Any) -> bool:
            return yugabyte

        monkeypatch.setattr(keyed_rows, "reads_yugabyte", engine)
        reader = _L3(db)
        assert _read(await read_keyed(reader, _TABLE, keys=_every_key(), columns=["race", "id", "v"])) == _NEWEST
        tables = await read_keyed_arrow(reader, _TABLE, keys=_every_key(), types=_TYPES)
        assert _read(pa.concat_tables(tables).to_pylist()) == _NEWEST
        stored = KeyBatch(("race", "id"), [["r1"], [2]], ("text", "integer"))
        types = {"race": pa.string(), "id": pa.int32(), "epoch": pa.int64()}
        held = await read_keyed_arrow(reader, _TABLE, keys=stored, types=types, with_gone=True)
        assert [row["epoch"] for row in pa.concat_tables(held).to_pylist()] == [10]

    async def test_a_prefix_batch_hides_what_the_reset_hides(
        self, db: duckdb.DuckDBPyConnection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Flips if a batch of races (a leading part of the key, PostgreSQL's LATERAL form) answers a version older
        than the reset."""

        async def postgres(conn: Any) -> bool:
            return False

        monkeypatch.setattr(keyed_rows, "reads_yugabyte", postgres)
        keys = KeyBatch(("race",), [["r1", "r2", "r3"]], ("text",))
        assert _read(await read_keyed(_L3(db), _TABLE, keys=keys, columns=["race", "id", "v"])) == _NEWEST


class TestPointReads:
    async def test_read_by_hides_versions_older_than_the_reset(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a race's read answers a version older than the reset, or a reset marker of one table hides another
        table's rows."""
        reader = _L3(db)
        assert _read(await read_by(reader, _TABLE, by={"race": "r1"}, columns=["race", "id", "v"])) == {("r1", 1): 2}
        assert await read_by(reader, _TABLE, by={"race": "r3"}, columns=["v"]) == []
        assert _read(await read_by(reader, _OTHER, by={"race": "r1"}, columns=["race", "id", "v"])) == {("r1", 1): 7}

    async def test_read_row_and_read_value_answer_nothing_below_the_reset(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a point read answers a key whose newest version is older than the reset."""
        reader = _L3(db)
        assert await read_value(reader, _TABLE, by={"race": "r1", "id": 1}, column="v") == 2
        assert await read_row(reader, _TABLE, by={"race": "r1", "id": 2}, columns=["v"]) is None


class TestPages:
    async def test_keyset_page_hides_versions_older_than_the_reset(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a keyset page answers a version older than the reset, or stops at a page all of whose keys it
        hides."""
        reader = _L3(db)
        order = [Sort("race"), Sort("id", cast="integer")]
        read: list[dict[str, Any]] = []
        cursor = None
        while True:
            page = await keyset_page(
                reader, _TABLE, columns=["race", "id", "v"], by={}, order=order, limit=1, cursor=cursor
            )
            read += page.rows
            if page.next_cursor is None:
                break
            cursor = page.next_cursor
        assert _read(read) == _NEWEST

    async def test_keyset_rows_reads_only_what_follows_the_reset(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a drain answers a version older than the reset."""
        rows = await keyset_rows(
            _L3(db), _TABLE, columns=["race", "id", "v"], by={"race": "r1"}, order=[Sort("id", cast="integer")], page=1
        )
        assert _read(rows) == {("r1", 1): 2}

    async def test_a_whole_read_of_the_table_hides_what_the_reset_hides(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if the pager of a whole read (a copy's cold start, the fold's rebuild) answers a version older than
        the reset, in dict rows or Arrow."""
        select = f"SELECT race, id, v FROM {_TABLE}"
        read = await read_keyset_pages(_L3(db), select, ["race", "id"], page_size=1, table=_TABLE)
        assert _read(read.rows) == _NEWEST
        pages = [
            page
            async for page in keyset_arrow_pages(
                _L3(db), select, ["race", "id"], types=_TYPES, page_size=1, table=_TABLE
            )
        ]
        assert _read(pa.concat_tables(pages).to_pylist()) == _NEWEST

    async def test_scope_pages_and_values_hide_what_the_reset_hides(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a read led by the key's first column (the fold's rebuild by range) answers a version older than
        the reset, or names a race holding nothing after it."""
        races = await keyset_scope_values(_L3(db), _TABLE, scope_column="race", scope_type=pa.string(), page_size=2)
        assert sorted(races) == ["r1", "r2"]
        pages = [
            page
            async for page in keyset_scope_pages(
                _L3(db),
                _TABLE,
                ["race", "id"],
                types=_TYPES,
                scope_column="race",
                scope_values=["r1", "r2", "r3"],
                scope_array="text[]",
                page_size=1,
            )
        ]
        assert _read(pa.concat_tables(pages).to_pylist()) == _NEWEST


class TestCounts:
    async def test_counts_count_only_what_follows_the_reset(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a count counts a version older than the reset."""
        reader = _L3(db)
        assert await counts_by(reader, _TABLE, column="race", values=AnyOf(["r1", "r2", "r3"], "text")) == {
            "r1": 1,
            "r2": 1,
        }
        assert (await count_at_most(reader, _TABLE, at_most=10, order=["race", "id"], by={"race": "r1"})).count == 1

    async def test_holds_any_is_false_for_a_race_the_reset_emptied(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a race holding only versions older than the reset holds a row."""
        reader = _L3(db)
        assert await holds_any(reader, _TABLE, order=["race", "id"], by={"race": "r1"})
        assert not await holds_any(reader, _TABLE, order=["race", "id"], by={"race": "r3"})

    async def test_empty_tables_names_a_table_the_reset_emptied(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a table whose every version is older than its reset reads as holding a row."""
        db.execute(table_reset_marker_sql(), [_OTHER, 60])
        assert await empty_tables(_L3(db), [_TABLE, _OTHER]) == [_OTHER]

    async def test_aggregate_keyed_sums_only_what_follows_the_reset(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if an aggregate takes a version older than the reset into its sum or count."""
        rows = await aggregate_keyed(
            _L3(db), _TABLE, aggregates={"n": Count(), "total": Sum("t.v")}, by={"race": AnyOf(["r1", "r2", "r3"])}
        )
        assert [(int(row["n"]), int(row["total"])) for row in rows] == [(2, 7)]


class TestStoreStatements:
    def test_the_store_s_key_reads_hide_versions_older_than_the_reset(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if the store's point read or key-led read answers a version older than the reset."""
        point = schema_sql.build_fetch_sql(_RESULTS)
        assert db.execute(point, ["r1", 1]).fetchall() == [("r1", 1, "NC", 2, 40)]
        assert db.execute(point, ["r1", 2]).fetchall() == []
        led = schema_sql.build_led_by_select_sql(_RESULTS, ["race", "id", "v"])
        assert sorted(db.execute(led, [["r1", "r2", "r3"]]).fetchall()) == [("r1", 1, 2), ("r2", 2, 5)]


class TestOtherReaders:
    async def test_read_past_marks_and_read_after_hide_what_the_reset_hides(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        """Flips if a read past a mark or after an id answers a version older than the reset."""
        read = await read_past_marks(_L3(db), _TABLE, columns=["race", "id", "v", "epoch"], marks={"epoch": 0})
        assert _read(read.rows) == _NEWEST
        page = await read_after(
            _L3(db), _TABLE, columns=["race", "id", "v"], id_column="id", after=0, limit=10, by={"race": "r1"}
        )
        assert _read(page.rows) == {("r1", 1): 2}

    async def test_a_step_over_the_newest_relation_hides_what_the_reset_hides(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        """Flips if a step over the table's newest relation answers a version older than the reset."""
        reader = _L3(db)
        assert _read(await read_step(reader, scan(newest_read(reader, _TABLE), ["race", "id", "v"]))) == _NEWEST


class TestTheCutIsOneKeyedRead:
    async def test_each_statement_reads_the_newest_reset_once_by_the_marker_table_s_key(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        """Flips if a read finds the table's reset by anything but one probe of the marker table's key, newest first
        (a scan of the markers, or a probe a row), or reads it more than once a statement."""
        reader = _L3(db)
        await read_by(reader, _TABLE, by={"race": "r1"}, columns=["v"])
        sql = reader.statements[-1]
        assert sql.count("AS reset_mark") == 1
        probe = sql[sql.index("(SELECT reset_mark.epoch") :]
        assert "reset_mark.table_name = 'rs_results'" in probe
        assert "reset_mark.scope = chr(30)" in probe
        assert "reset_mark.prefix = ''" in probe
        assert "ORDER BY reset_mark.epoch DESC LIMIT 1)" in probe
