"""every shared read of a versioned L3 table sees only each key's newest live version (append-only L3, chunk 3).

A versioned table (``TableSchema(row_epoch=row_versions(...))``) keeps a version of a key a write; a delete appends a
"gone" marker; a whole run deleted is one row of ``threetears_run_markers``. Each read here must answer only each key's
newest version, never an older one, never a key whose newest version is a marker, never a key under a newer run
marker. The seeded table:

- ``("CA", 1)``: three versions (epochs 10, 20, 30; ``v`` 1, 2, 3): only ``v = 3`` is read;
- ``("CA", 2)``: a version at 10, then a "gone" marker at 20: never read;
- ``("CA", 3)``: one version: read;
- ``("NV", 1)``: a version at 5 under the run marker ``NV`` at 15: never read;
- ``("NV", 2)``: a version at 25, newer than the marker: read;
- ``("TX", 1)``: a version, then a marker: the state holds no row a read sees;
- ``("OR", 1)``: a version at 10 under the two-column run marker ``(OR, 1)`` at 12: never read.

Correctness runs on an in-process DuckDB behind a reader that names no dialect, so the helpers take it for L3 (the
guard is L3's alone: a local copy keeps one row a key). The markers are written by the writer's own statement
(:func:`~threetears.core.ddl.run_markers.run_marker_insert_sql`), so the read's text of a prefix is proved to be the
write's. The plans on PostgreSQL and YugabyteDB (the probe a seek of ``(key..., epoch DESC)``) are the integration
pass's.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import duckdb
import pyarrow as pa
import pytest

from threetears.core.backends import schema_sql
from threetears.core.collections.schema_backed import INT_TYPE, STRING_TYPE, Column, TableSchema
from threetears.core.ddl.row_epoch import AppendOnlyWriteError, row_versions
from threetears.core.ddl.run_markers import run_marker_insert_sql
from threetears.core.keyed import (
    AnyOf,
    Count,
    KeyBatch,
    Lock,
    Sort,
    Sum,
    ThreadedDuckDB,
    aggregate_keyed,
    count_at_most,
    counts_by,
    delete_keyed,
    delete_matching,
    empty_tables,
    holds_any,
    keyset_page,
    keyset_rows,
    Match,
    latest_per_key,
    newest_read,
    read_after,
    read_by,
    read_keyed,
    read_keyed_arrow,
    read_link_chain,
    read_past_marks,
    read_reached,
    read_row,
    read_step,
    read_value,
    scan,
    semi,
    update_keyed,
)
from threetears.core.keyed import rows as keyed_rows
from threetears.core.keyset import keyset_arrow_pages, keyset_scope_pages, keyset_scope_values, read_keyset_pages
from threetears.core.testing.l1_tables import TEST_L1_TABLE
from threetears.interfaces.keyed.keys import next_newest_keys, versioned_table

_TABLE = "nv_results"
_GONE_ONLY = "nv_gone_only"

_RESULTS = TableSchema(
    name=_TABLE,
    primary_key=("state", "id"),
    columns=[Column("state", STRING_TYPE), Column("id", INT_TYPE), Column("v", INT_TYPE)],
    row_epoch=row_versions("state"),
    l1_eviction=TEST_L1_TABLE,
)

_GONE_ONLY_SCHEMA = TableSchema(
    name=_GONE_ONLY,
    primary_key=("state", "id"),
    columns=[Column("state", STRING_TYPE), Column("id", INT_TYPE), Column("v", INT_TYPE)],
    row_epoch=row_versions("state"),
    l1_eviction=TEST_L1_TABLE,
)

#: each key's newest live version, (state, id) -> v
_NEWEST = {("CA", 1): 3, ("CA", 3): 30, ("NV", 2): 60}

#: every key the table holds a version of
_EVERY_KEY = [("CA", 1), ("CA", 2), ("CA", 3), ("NV", 1), ("NV", 2), ("TX", 1), ("OR", 1)]

_DDL = "(state VARCHAR, id INTEGER, v INTEGER, epoch BIGINT, gone BOOLEAN NOT NULL DEFAULT false, PRIMARY KEY (state, id, epoch))"

#: the most versions one newest read's statement reads (``keys``' bound a statement's keys are sized by)
_NEWEST_ROWS_A_READ = 100_000


@pytest.fixture
def db() -> Iterator[duckdb.DuckDBPyConnection]:
    """the versioned table as the module's docstring seeds it, its run markers written by the writer's statement."""
    conn = duckdb.connect()
    conn.execute(f"CREATE TABLE {_TABLE} {_DDL}")
    conn.execute(f"CREATE TABLE {_GONE_ONLY} {_DDL}")
    conn.execute(
        "CREATE TABLE threetears_run_markers (table_name VARCHAR, scope VARCHAR, prefix VARCHAR, epoch BIGINT)"
    )
    conn.execute(
        f"INSERT INTO {_TABLE} (state, id, v, epoch, gone) VALUES "
        "('CA', 1, 1, 10, false), ('CA', 1, 2, 20, false), ('CA', 1, 3, 30, false), "
        "('CA', 2, 10, 10, false), ('CA', 2, NULL, 20, true), "
        "('CA', 3, 30, 10, false), "
        "('NV', 1, 50, 5, false), ('NV', 2, 60, 25, false), "
        "('TX', 1, 70, 10, false), ('TX', 1, NULL, 20, true), "
        "('OR', 1, 80, 10, false)"
    )
    conn.execute(f"INSERT INTO {_GONE_ONLY} (state, id, v, epoch, gone) VALUES ('CA', 1, NULL, 10, true)")
    conn.execute(run_marker_insert_sql(["text"], 0), [_TABLE, 15, "NV"])
    conn.execute(run_marker_insert_sql(["text", "integer"], 0), [_TABLE, 12, "OR", 1])
    yield conn
    conn.close()


class _L3:
    """an L3 reader answering from DuckDB: it names no dialect, so every helper takes it for L3, as the rail's proxy
    or an asyncpg pool; it keeps every statement it was sent."""

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
    """the rows read, (state, id) -> v, each key once."""
    answered: dict[tuple[str, int], int] = {}
    for row in rows:
        key = (row["state"], int(row["id"]))
        assert key not in answered, f"{key} read twice: a read answered more than its newest version"
        answered[key] = int(row["v"])
    return answered


def _every_key() -> KeyBatch:
    return KeyBatch(("state", "id"), [[k[0] for k in _EVERY_KEY], [k[1] for k in _EVERY_KEY]], ("text", "integer"))


class TestTheDeclaration:
    def test_a_versioned_schema_registers_its_key_and_run_markers(self) -> None:
        """Flips if a versioned TableSchema registers no newest read, or one whose markers cannot be found."""
        declared = versioned_table(_TABLE, l3=True)
        assert declared is not None
        assert declared.key == ("state", "id")
        assert declared.markers == "threetears_run_markers"
        assert declared.texts == ("text", "integer")
        assert versioned_table(_TABLE, l3=False) is None


class TestKeyedReads:
    async def test_read_keyed_answers_each_key_s_newest_live_version(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if an older version, a gone key or a key under a newer run marker is read."""
        rows = await read_keyed(_L3(db), _TABLE, keys=_every_key(), columns=["state", "id", "v"])
        assert _read(rows) == _NEWEST

    async def test_read_keyed_arrow_answers_each_key_s_newest_live_version(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips as read_keyed's does, for the Arrow read."""
        types = {"state": pa.string(), "id": pa.int32(), "v": pa.int32()}
        tables = await read_keyed_arrow(_L3(db), _TABLE, keys=_every_key(), types=types)
        assert _read(pa.concat_tables(tables).to_pylist()) == _NEWEST

    async def test_read_keyed_arrow_with_gone_reads_every_stored_version(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if the as-stored keyed read quiet-time cleanup makes (it reads the versions it removes) answers each
        key's newest live version alone, or anything but every version and gone marker as stored."""
        types = {"state": pa.string(), "id": pa.int32(), "epoch": pa.int64()}
        keys = KeyBatch(("state", "id"), [["CA", "CA"], [1, 2]], ("text", "integer"))
        tables = await read_keyed_arrow(_L3(db), _TABLE, keys=keys, types=types, with_gone=True)
        held = sorted((row["state"], row["id"], row["epoch"]) for row in pa.concat_tables(tables).to_pylist())
        assert held == [("CA", 1, 10), ("CA", 1, 20), ("CA", 1, 30), ("CA", 2, 10), ("CA", 2, 20)]

    async def test_a_local_copy_is_read_as_it_holds(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a read that does not go to L3 (a local copy, one row a key) is sent the newest guard."""
        rows = await read_by(ThreadedDuckDB(db), _TABLE, by={"state": "CA", "id": 1}, columns=["v"])
        assert sorted(int(row["v"]) for row in rows) == [1, 2, 3]


class TestPointReads:
    async def test_read_by_answers_a_state_s_newest_live_versions(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a state's read answers an older version or a gone key."""
        rows = await read_by(_L3(db), _TABLE, by={"state": "CA"}, columns=["state", "id", "v"])
        assert _read(rows) == {("CA", 1): 3, ("CA", 3): 30}

    async def test_read_by_skips_a_key_under_a_newer_run_marker(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a run marked gone after a key's version leaves the key readable, or hides a newer version."""
        assert _read(await read_by(_L3(db), _TABLE, by={"state": "NV"}, columns=["state", "id", "v"])) == {
            ("NV", 2): 60
        }
        assert await read_by(_L3(db), _TABLE, by={"state": "OR"}, columns=["v"]) == []

    async def test_read_row_and_read_value_answer_the_newest_or_nothing(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a point read answers an older version, or a key whose newest version is a marker."""
        reader = _L3(db)
        assert await read_value(reader, _TABLE, by={"state": "CA", "id": 1}, column="v") == 3
        assert await read_row(reader, _TABLE, by={"state": "CA", "id": 2}, columns=["v"]) is None
        assert await read_row(reader, _TABLE, by={"state": "OR", "id": 1}, columns=["v"]) is None


class TestPages:
    async def test_keyset_page_steps_past_a_cursor_by_the_key_alone(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a page answers an older version, or the cursor's key is refused as not unique (a versioned
        table's key is unique among the newest versions a read sees)."""
        reader = _L3(db)
        order = [Sort("state"), Sort("id", cast="integer")]
        read: list[dict[str, Any]] = []
        cursor = None
        while True:
            page = await keyset_page(
                reader, _TABLE, columns=["state", "id", "v"], by={}, order=order, limit=1, cursor=cursor
            )
            read += page.rows
            if page.next_cursor is None:
                break
            cursor = page.next_cursor
        assert _read(read) == _NEWEST

    async def test_keyset_rows_reads_every_newest_version(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a drain of a state answers an older version or a gone key."""
        rows = await keyset_rows(
            _L3(db),
            _TABLE,
            columns=["state", "id", "v"],
            by={"state": "CA"},
            order=[Sort("id", cast="integer")],
            page=1,
        )
        assert _read(rows) == {("CA", 1): 3, ("CA", 3): 30}

    async def test_read_keyset_pages_of_a_named_table_reads_its_newest_versions(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        """Flips if the keyset pager of a whole read (cold start, refill) answers older versions or gone keys."""
        read = await read_keyset_pages(
            _L3(db), f"SELECT state, id, v FROM {_TABLE}", ["state", "id"], page_size=1, table=_TABLE
        )
        assert _read(read.rows) == _NEWEST

    async def test_keyset_scope_values_skips_a_state_with_no_live_row(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a state holding only markers (TX) or rows under a newer run marker (OR) reads as a state."""
        states = await keyset_scope_values(_L3(db), _TABLE, scope_column="state", scope_type=pa.string(), page_size=2)
        assert sorted(states) == ["CA", "NV"]

    async def test_keyset_scope_pages_reads_the_states_newest_versions(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a state's pages answer an older version, a gone key or a key under a newer run marker."""
        types = {"state": pa.string(), "id": pa.int32(), "v": pa.int32()}
        pages = [
            page
            async for page in keyset_scope_pages(
                _L3(db),
                _TABLE,
                ["state", "id"],
                types=types,
                scope_column="state",
                scope_values=["CA", "NV", "OR", "TX"],
                scope_array="text[]",
                page_size=1,
            )
        ]
        assert _read(pa.concat_tables(pages).to_pylist()) == _NEWEST


class TestCounts:
    async def test_counts_count_newest_live_versions_alone(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a count counts an older version, a marker, or a row under a newer run marker."""
        reader = _L3(db)
        assert await counts_by(reader, _TABLE, column="id", by={"state": "CA"}, cast="integer") == {1: 1, 3: 1}
        assert await counts_by(reader, _TABLE, column="state", values=AnyOf(["CA", "NV", "OR", "TX"], "text")) == {
            "CA": 2,
            "NV": 1,
        }
        assert (await count_at_most(reader, _TABLE, at_most=10, order=["state", "id"], by={"state": "CA"})).count == 2

    async def test_holds_any_is_false_for_a_state_of_markers_alone(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a state holding only a gone marker, or rows under a newer run marker, holds a row."""
        reader = _L3(db)
        assert await holds_any(reader, _TABLE, order=["state", "id"], by={"state": "CA"})
        assert not await holds_any(reader, _TABLE, order=["state", "id"], by={"state": "TX"})
        assert not await holds_any(reader, _TABLE, order=["state", "id"], by={"state": "OR"})

    async def test_empty_tables_names_a_table_of_markers_alone(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a versioned table holding only gone markers reads as holding a row."""
        assert await empty_tables(_L3(db), [_TABLE, _GONE_ONLY]) == [_GONE_ONLY]

    async def test_aggregate_keyed_sums_newest_versions_alone(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if an aggregate takes an older version or a marker into its sum or count."""
        rows = await aggregate_keyed(
            _L3(db), _TABLE, aggregates={"n": Count(), "total": Sum("t.v")}, by={"state": "CA"}
        )
        assert [(int(row["n"]), int(row["total"])) for row in rows] == [(2, 33)]


class TestStoreStatements:
    def test_the_store_s_key_reads_read_the_newest_relation(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if the store's point read or key-led read of a versioned table answers an older version or a gone
        key, or a read answers the marker flag as a column."""
        point = schema_sql.build_fetch_sql(_RESULTS)
        assert "gone" not in schema_sql.build_select_column_list(_RESULTS).split(", ")
        assert db.execute(point, ["CA", 1]).fetchall() == [("CA", 1, 3, 30)]
        assert db.execute(point, ["CA", 2]).fetchall() == []
        assert db.execute(point, ["NV", 1]).fetchall() == []
        led = schema_sql.build_led_by_select_sql(_RESULTS, ["state", "id", "v"])
        assert sorted(db.execute(led, [["CA", "NV"]]).fetchall()) == [("CA", 1, 3), ("CA", 3, 30), ("NV", 2, 60)]
        # the whole-key batch names its keys inside the newest relation (no planner pushes a join's keys into it)
        whole = schema_sql.build_whole_key_select_sql(_RESULTS)
        assert "DISTINCT ON" in whole and whole.index("unnest") < whole.index(') AS "nv_results"')


class TestSteps:
    async def test_latest_per_key_with_gone_drops_a_key_whose_newest_is_a_marker(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        """Flips if a key whose first row in the order is a gone marker answers the version behind it."""
        step = latest_per_key(
            _TABLE,
            key=["state", "id"],
            order=[Sort("epoch", descending=True)],
            columns=["state", "id", "v"],
            gone="gone",
        )
        rows = await read_step(ThreadedDuckDB(db), step)
        # no run markers here: a step over a relation of versions judges its markers only; OR and NV 1 stay
        assert _read(rows) == {**_NEWEST, ("NV", 1): 50, ("OR", 1): 80}


class TestInPlaceWrites:
    async def test_a_keyed_update_or_delete_of_a_versioned_table_is_refused(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        """Flips if an operator's keyed update or delete changes a version in place (measure.py's edits), a write
        naming gone rows too (with_gone) among them."""
        reader = _L3(db)
        with pytest.raises(AppendOnlyWriteError):
            await update_keyed(reader, _TABLE, values={"v": 9}, by={"state": "CA", "id": 1})
        with pytest.raises(AppendOnlyWriteError):
            await update_keyed(reader, _TABLE, values={"v": 9}, by={"state": "CA", "id": 1}, with_gone=True)
        with pytest.raises(AppendOnlyWriteError):
            await delete_keyed(reader, _TABLE, by={"state": "CA", "id": 1})
        assert reader.statements == []


class TestEngineForms:
    """the keyed batch's two forms (YugabyteDB's batched DISTINCT ON, PostgreSQL's LATERAL seek a key) answer alike;
    the in-process DuckDB answers both."""

    @pytest.mark.parametrize("yugabyte", [True, False])
    async def test_read_keyed_answers_the_newest_in_either_engine_s_form(
        self, db: duckdb.DuckDBPyConnection, monkeypatch: pytest.MonkeyPatch, yugabyte: bool
    ) -> None:
        """Flips if a batch in either form answers an older version, a gone key or a key under a run marker, or
        leaves the versions count it sizes its next batch by among the row's columns."""

        async def engine(conn: Any) -> bool:
            return yugabyte

        monkeypatch.setattr(keyed_rows, "reads_yugabyte", engine)
        reader = _L3(db)
        rows = await read_keyed(reader, _TABLE, keys=_every_key(), columns=["state", "id", "v"])
        assert _read(rows) == _NEWEST
        assert all(set(row) == {"state", "id", "v"} for row in rows)
        assert ("DISTINCT ON" in reader.statements[-1]) is yugabyte
        assert ("LATERAL" in reader.statements[-1]) is not yugabyte
        types = {"state": pa.string(), "id": pa.int32(), "v": pa.int32()}
        tables = await read_keyed_arrow(reader, _TABLE, keys=_every_key(), types=types)
        assert _read(pa.concat_tables(tables).to_pylist()) == _NEWEST
        assert tables[0].column_names == ["state", "id", "v"]

    async def test_a_prefix_batch_answers_each_of_its_keys_newest(
        self, db: duckdb.DuckDBPyConnection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Flips if a batch of a prefix of the key (states) answers one row a prefix or an older version, or is not each
        prefix's own LATERAL DISTINCT ON over the rest of the key."""

        async def postgres(conn: Any) -> bool:
            return False

        monkeypatch.setattr(keyed_rows, "reads_yugabyte", postgres)
        keys = KeyBatch(("state",), [["CA", "NV", "OR", "TX"]], ("text",))
        reader = _L3(db)
        assert _read(await read_keyed(reader, _TABLE, keys=keys, columns=["state", "id", "v"])) == _NEWEST
        assert 'LATERAL (SELECT DISTINCT ON (t."id")' in reader.statements[-1]

    def test_a_batch_is_sized_by_the_versions_its_last_read(self) -> None:
        """Flips if a batch's keys are not sized by the versions the batch before read a key."""
        assert next_newest_keys(0, 0, 5_000) == _NEWEST_ROWS_A_READ // 50
        assert next_newest_keys(5_000, 250_000, 5_000) == _NEWEST_ROWS_A_READ // 50
        assert next_newest_keys(1_000, 1_000, 5_000) == 5_000
        assert next_newest_keys(10, 10 * _NEWEST_ROWS_A_READ, 5_000) == 1


class TestFillPages:
    async def test_a_filter_past_the_key_is_judged_on_the_newest_version(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a fill's filter on a column past the key picks an older version that matched it (v = 2 is CA 1's
        middle version), or loses the newest that does, or judges it inside the page's DISTINCT ON (before the newest
        version is chosen)."""
        select = f"SELECT state, id, v FROM {_TABLE}"
        reader = _L3(db)
        older = await read_keyset_pages(reader, select, ["state", "id"], where={"v": 2}, page_size=1, table=_TABLE)
        assert older.rows == []
        page = reader.statements[-1]
        chosen = page[page.index("SELECT DISTINCT ON") : page.index(") AS n")]
        assert '"v"' not in chosen and '"v" = $1' in page
        newest = await read_keyset_pages(_L3(db), select, ["state", "id"], where={"v": 3}, page_size=1, table=_TABLE)
        assert _read(newest.rows) == {("CA", 1): 3}

    async def test_arrow_pages_step_past_keys_they_keep_none_of(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if an Arrow fill stops at a page whose keys are all gone (TX, OR), answers an older version, hands
        over its own columns, or reads its pages by anything but each range's DISTINCT ON."""
        types = {"state": pa.string(), "id": pa.int32(), "v": pa.int32()}
        reader = _L3(db)
        pages = [
            page
            async for page in keyset_arrow_pages(
                reader, f"SELECT state, id, v FROM {_TABLE}", ["state", "id"], types=types, page_size=1, table=_TABLE
            )
        ]
        assert all("DISTINCT ON" in sql for sql in reader.statements if "_newest_kept" in sql)
        assert any("_newest_kept" in sql for sql in reader.statements)
        assert all(page.column_names == ["state", "id", "v"] for page in pages)
        assert _read(pa.concat_tables(pages).to_pylist()) == _NEWEST

    async def test_a_pager_naming_a_versioned_table_without_table_is_refused(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        """Flips if a keyset read of a versioned L3 table that does not name it (table=) pages every version."""
        with pytest.raises(ValueError, match="versioned"):
            await read_keyset_pages(_L3(db), f"SELECT state, id, v FROM {_TABLE}", ["state", "id"], page_size=1)


class TestNewlyCoveredReaders:
    async def test_read_past_marks_answers_the_newest_of_the_keys_past_the_mark(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        """Flips if a read past a mark answers an older version, a gone key, or a key under a newer run marker."""
        read = await read_past_marks(_L3(db), _TABLE, columns=["state", "id", "v", "epoch"], marks={"epoch": 0})
        assert _read(read.rows) == _NEWEST
        moved = await read_past_marks(_L3(db), _TABLE, columns=["state", "id", "v", "epoch"], marks={"epoch": 25})
        assert _read(moved.rows) == {("CA", 1): 3}

    async def test_read_after_answers_newest_live_versions(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a read after an id answers an older version or a gone key."""
        page = await read_after(
            _L3(db), _TABLE, columns=["state", "id", "v"], id_column="id", after=0, limit=10, by={"state": "CA"}
        )
        assert _read(page.rows) == {("CA", 1): 3, ("CA", 3): 30}

    async def test_a_step_over_a_versioned_table_on_l3_is_refused_or_reads_its_newest(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        """Flips if a step naming a versioned L3 table as a plain relation reads every version, if a step over a
        local copy is refused, or if a step over the table's newest relation answers an older version."""
        reader = _L3(db)
        with pytest.raises(ValueError, match="newest_read"):
            await read_step(reader, scan(_TABLE, ["state", "id", "v"]))
        keys = scan(_TABLE, ["state", "id"])
        with pytest.raises(ValueError, match="newest_read"):
            await read_step(reader, semi(_GONE_ONLY, keys, on=[Match("state"), Match("id")]))
        assert len(await read_step(ThreadedDuckDB(db), scan(_TABLE, ["v"]))) == 11
        rows = await read_step(reader, scan(newest_read(reader, _TABLE), ["state", "id", "v"]))
        assert _read(rows) == _NEWEST

    async def test_walks_and_chains_of_a_versioned_table_are_refused(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a recursive walk or link chain of a versioned L3 table runs (it would read every version)."""
        reader = _L3(db)
        with pytest.raises(ValueError, match="versioned"):
            await read_link_chain(reader, _TABLE, key="id", link="v", start=1, most=3, columns=["v"])
        with pytest.raises(ValueError, match="versioned"):
            await read_reached(reader, _TABLE, start={"state": "CA"}, on="id", to="v", columns=["v"], depth=2)
        assert reader.statements == []

    async def test_a_lock_or_a_delete_by_relation_of_a_versioned_table_is_refused(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        """Flips if a read locks a versioned table's versions or a delete by a change relation removes them."""
        reader = _L3(db)
        with pytest.raises(ValueError, match="versioned"):
            await read_by(reader, _TABLE, by={"state": "CA"}, columns=["v"], lock=Lock.UPDATE)
        with pytest.raises(AppendOnlyWriteError):
            await delete_matching(reader, _TABLE, "other", on=[Match("id")])
        assert reader.statements == []
