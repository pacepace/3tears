"""the keyed-read helpers (:mod:`threetears.core.keyed`): each answers exactly the rows a plain query names, and each
statement it sends names the key.

Correctness runs on an in-process DuckDB (the helpers' SQL is one spelling for PostgreSQL, YugabyteDB and DuckDB);
the PostgreSQL / YugabyteDB half, plans and real-size speed are
``packages/core/tests/integration/test_keyed_helpers_live.py``. A recording reader pins the statements themselves:
the batch bound, the placeholders, the cursor's step.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from datetime import datetime, timedelta
from typing import Any

import duckdb
import pytest

from threetears.core.keyed import (
    AnyOf,
    BoundedCount,
    Dialect,
    KeyBatch,
    KeyTooWide,
    Sort,
    ThreadedDuckDB,
    ThreadedSQLite,
    WholeReads,
    count_at_most,
    counts_by,
    delete_before,
    empty_tables,
    estimated_rows,
    exists_keyed,
    holds_any,
    keyset_page,
    read_after,
    read_keyed,
    read_past_marks,
    rows_affected,
    update_before,
)
from threetears.core.keyed._filters import Fragment, renumbered
from threetears.core.keyset import read_keyset_pages
from threetears.core.pagination import CursorError

_START = datetime(2026, 1, 1)


@pytest.fixture
def db() -> Iterator[duckdb.DuckDBPyConnection]:
    """``t``: 10 groups of 10 rows keyed ``(a, b)``, a nullable stamp every third row; ``grants``: who sees a group."""
    conn = duckdb.connect()
    conn.execute("CREATE TABLE t (a VARCHAR, b BIGINT, v INTEGER, stamp TIMESTAMP, PRIMARY KEY (a, b))")
    conn.execute(
        "INSERT INTO t SELECT 'k' || (i % 10), i, i, "
        "CASE WHEN i % 3 = 0 THEN NULL ELSE TIMESTAMP '2026-01-01' + to_seconds(i // 2) END FROM range(100) r(i)"
    )
    conn.execute("CREATE TABLE grants (who VARCHAR, a VARCHAR, PRIMARY KEY (who, a))")
    conn.execute("INSERT INTO grants VALUES ('ann', 'k1'), ('ann', 'k2'), ('bob', 'k3')")
    yield conn
    conn.close()


class _Recording:
    """a reader that answers from a DuckDB connection and keeps every statement it was sent."""

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._inner = ThreadedDuckDB(conn)
        self.statements: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, query: str, *params: Any) -> list[dict[str, Any]]:
        self.statements.append((query, params))
        return await self._inner.fetch(query, *params)

    async def execute(self, query: str, *params: Any) -> str:
        self.statements.append((query, params))
        return await self._inner.execute(query, *params)


def _plain(conn: duckdb.DuckDBPyConnection, sql: str, *params: Any) -> list[tuple[Any, ...]]:
    return [tuple(row) for row in conn.execute(sql, list(params)).fetchall()]


class TestReadKeyed:
    async def test_it_answers_exactly_the_named_keys_once_each(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a key is read twice, a key is dropped, or a row no key names is answered."""
        reader = _Recording(db)
        keys = KeyBatch(("a", "b"), [("k1", 1), ("k2", 12), ("k1", 1), ("k9", 1)], ("text", "bigint"))
        rows = await read_keyed(reader, "t", keys=keys, columns=["a", "b", "v"])
        assert sorted((row["a"], row["b"], row["v"]) for row in rows) == [("k1", 1, 1), ("k2", 12, 12)]
        assert len(reader.statements) == 1

    async def test_each_statement_names_at_most_a_batch_of_keys(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a read binds every key in one statement however many there are."""
        reader = _Recording(db)
        keys = KeyBatch(("a", "b"), [(f"k{i % 10}", i) for i in range(25)], ("text", "bigint"))
        rows = await read_keyed(reader, "t", keys=keys, columns=["a", "b"], batch=10)
        assert len(rows) == 25
        assert [len(params[0]) for _, params in reader.statements] == [10, 10, 5]
        for sql, _ in reader.statements:
            assert "JOIN (SELECT unnest($1::text[]) AS k0, unnest($2::text[]) AS k1) AS k" in sql

    async def test_by_pins_a_prefix_beside_the_keys(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if ``by`` is dropped (the read answers rows of every group) or bound to the wrong placeholder."""
        keys = KeyBatch(("b",), [(1,), (2,), (11,)], ("bigint",))
        rows = await read_keyed(ThreadedDuckDB(db), "t", keys=keys, columns=["a", "b"], by={"a": "k1"})
        assert sorted((row["a"], row["b"]) for row in rows) == [("k1", 1), ("k1", 11)]

    async def test_an_answer_at_the_cap_is_read_again_in_halves(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if an answer the transport may have cut is kept as if whole."""
        reader = _Recording(db)
        keys = KeyBatch(("a",), [("k1",), ("k2",), ("k3",), ("k4",)])
        rows = await read_keyed(reader, "t", keys=keys, columns=["a", "b"], row_cap=25)
        assert len(rows) == 40
        assert [len(params[0]) for _, params in reader.statements] == [4, 2, 2]

    async def test_one_key_at_the_cap_is_refused(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a key that alone fills a statement is read as if its answer were whole."""
        with pytest.raises(KeyTooWide):
            await read_keyed(ThreadedDuckDB(db), "t", keys=KeyBatch(("a",), [("k1",)]), columns=["a"], row_cap=10)

    def test_a_key_of_the_wrong_arity_is_refused(self) -> None:
        with pytest.raises(ValueError, match="has 2 values"):
            KeyBatch(("a", "b"), [("k1",)])


class TestKeysetPage:
    async def test_pages_answer_every_row_of_the_prefix_once_in_order(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a page skips or repeats a row at its boundary, or reads past the prefix."""
        order = [Sort("b", "bigint")]
        seen: list[int] = []
        cursor = None
        while True:
            page = await keyset_page(
                ThreadedDuckDB(db), "t", columns=["a", "b"], by={"a": "k3"}, order=order, limit=4, cursor=cursor
            )
            seen += [row["b"] for row in page.rows]
            cursor = page.next_cursor
            if cursor is None:
                break
        assert seen == [row[0] for row in _plain(db, "SELECT b FROM t WHERE a = 'k3' ORDER BY b")]

    @pytest.mark.parametrize("descending", [False, True])
    async def test_a_nullable_sort_pages_through_its_nulls(
        self, db: duckdb.DuckDBPyConnection, descending: bool
    ) -> None:
        """Flips if a page past a NULL lands in the wrong partition (rows dropped or repeated around the nulls)."""
        order = [Sort("stamp", "timestamp", descending=descending, nullable=True), Sort("b", "bigint", descending)]
        seen: list[int] = []
        cursor = None
        while True:
            page = await keyset_page(
                ThreadedDuckDB(db), "t", columns=["b", "stamp"], by={}, order=order, limit=7, cursor=cursor
            )
            seen += [row["b"] for row in page.rows]
            cursor = page.next_cursor
            if cursor is None:
                break
        direction = "DESC NULLS FIRST" if descending else "ASC NULLS LAST"
        tie = "DESC" if descending else "ASC"
        assert seen == [row[0] for row in _plain(db, f"SELECT b FROM t ORDER BY stamp {direction}, b {tie}")]

    async def test_visibility_is_a_keyed_exists_beside_the_cursor(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if the visibility clause's placeholders collide with the page's, or it is dropped."""
        reader = _Recording(db)
        visible = exists_keyed("grants", on={"a": "a"}, by={"who": "ann"})
        first = await keyset_page(
            reader, "t", columns=["a", "b"], by={}, order=[Sort("a"), Sort("b", "bigint")], limit=15, visible=[visible]
        )
        second = await keyset_page(
            reader,
            "t",
            columns=["a", "b"],
            by={},
            order=[Sort("a"), Sort("b", "bigint")],
            limit=15,
            cursor=first.next_cursor,
            visible=[visible],
        )
        assert second.next_cursor is None
        got = [(row["a"], row["b"]) for row in first.rows + second.rows]
        assert got == _plain(db, "SELECT a, b FROM t WHERE a IN ('k1', 'k2') ORDER BY a, b")
        sql, params = reader.statements[-1]
        assert 'EXISTS (SELECT 1 FROM "grants" AS v WHERE v."a" = t."a" AND v."who" = $3)' in sql
        assert params[-1] == "ann"

    async def test_a_cursor_of_another_sort_is_refused(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a cursor issued for one sort is read as a place in another, or garbage reads as the start."""
        page = await keyset_page(
            ThreadedDuckDB(db), "t", columns=["a", "b"], by={}, order=[Sort("a"), Sort("b", "bigint")], limit=3
        )
        with pytest.raises(CursorError):
            await keyset_page(
                ThreadedDuckDB(db),
                "t",
                columns=["b"],
                by={},
                order=[Sort("b", "bigint")],
                limit=3,
                cursor=page.next_cursor,
            )
        with pytest.raises(CursorError):
            await keyset_page(
                ThreadedDuckDB(db), "t", columns=["b"], by={}, order=[Sort("b", "bigint")], limit=3, cursor="!!"
            )

    def test_a_fragment_naming_a_parameter_it_does_not_bind_is_refused(self) -> None:
        with pytest.raises(ValueError, match="binds 1"):
            renumbered(Fragment("x = $1 AND y = $2", ("a",)), 3)
        assert renumbered(Fragment("x = $1 AND y = $2", ("a", "b")), 3) == "x = $4 AND y = $5"


class TestCounts:
    @pytest.mark.parametrize(
        ("status", "rows"), [("UPDATE 3", 3), ("DELETE 0", 0), ("INSERT 0 5", 5), ("SELECT 12", 12), ("COPY 7", 7)]
    )
    def test_rows_affected_reads_the_statement_s_own_count(self, status: str, rows: int) -> None:
        assert rows_affected(status) == rows

    def test_a_status_with_no_count_is_refused(self) -> None:
        with pytest.raises(ValueError):
            rows_affected("CREATE TABLE")

    async def test_count_at_most_counts_no_further_than_its_bound(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if the count reads every row the key names (no LIMIT under it) or misreports ``more``."""
        reader = _Recording(db)
        assert await count_at_most(reader, "t", by={"a": "k1"}, at_most=3) == BoundedCount(count=3, more=True)
        assert await count_at_most(reader, "t", by={"a": "k1"}, at_most=10) == BoundedCount(count=10, more=False)
        assert "LIMIT 4" in reader.statements[0][0]

    async def test_count_at_most_refuses_a_count_of_the_table(self, db: duckdb.DuckDBPyConnection) -> None:
        with pytest.raises(ValueError, match="names the rows"):
            await count_at_most(ThreadedDuckDB(db), "t", by={}, at_most=3)

    async def test_probes_read_one_row(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a probe reads more than one row to answer whether any is held."""
        db.execute("CREATE TABLE empty_one (a VARCHAR PRIMARY KEY)")
        reader = _Recording(db)
        assert await holds_any(reader, "t", by={"a": "k1"})
        assert not await holds_any(reader, "t", by={"a": "nope"})
        assert await empty_tables(reader, ["t", "empty_one"]) == ["empty_one"]
        assert all("LIMIT 1" in sql for sql, _ in reader.statements)

    async def test_counts_by_counts_each_key_in_bounded_batches(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a grouped count reads groups no key named, or binds every key in one statement."""
        reader = _Recording(db)
        counted = await counts_by(
            reader, "t", column="b", values=AnyOf([1, 11, 2, 500], "bigint"), by={"a": "k1"}, batch=2
        )
        assert counted == {1: 1, 11: 1}
        assert [len(params[-1]) for _, params in reader.statements] == [2, 2]

    async def test_estimated_rows_is_the_catalog_s_estimate(self, db: duckdb.DuckDBPyConnection) -> None:
        assert await estimated_rows(ThreadedDuckDB(db), "t", dialect=Dialect.DUCKDB) == 100
        assert await estimated_rows(ThreadedDuckDB(db), "absent", dialect=Dialect.DUCKDB) is None


class TestMarks:
    async def test_read_past_marks_reads_only_what_moved_and_answers_the_next_marks(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        """Flips if rows at or before the mark are read again, or the mark does not advance."""
        mark = _START + timedelta(seconds=45)
        read = await read_past_marks(ThreadedDuckDB(db), "t", columns=["b", "stamp"], marks={"stamp": mark})
        assert sorted(row["b"] for row in read.rows) == [
            row[0] for row in _plain(db, "SELECT b FROM t WHERE stamp > $1 ORDER BY b", mark)
        ]
        assert read.marks == {"stamp": _START + timedelta(seconds=49)}
        again = await read_past_marks(ThreadedDuckDB(db), "t", columns=["b", "stamp"], marks=read.marks)
        assert again.rows == [] and again.marks == read.marks

    async def test_a_read_with_no_mark_is_a_cold_start_and_refused(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a missing mark reads the whole record as "everything changed"."""
        with pytest.raises(ValueError, match="cold start"):
            await read_past_marks(ThreadedDuckDB(db), "t", columns=["b", "stamp"], marks={"stamp": None})

    async def test_read_after_pages_a_prefix_by_its_growing_id(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a page re-reads its ``after`` row or reads outside the prefix."""
        page = await read_after(
            ThreadedDuckDB(db), "t", columns=["a", "b"], id_column="b", after=5, limit=3, by={"a": "k5"}
        )
        assert [row["b"] for row in page.rows] == [15, 25, 35]
        assert page.last == 35 and page.more


class TestDeleteBefore:
    async def test_a_sweep_deletes_page_by_page_until_a_short_page(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if the sweep deletes everything in one statement, or stops before the rows past its bound are gone."""
        reader = _Recording(db)
        swept = await delete_before(reader, "t", key=["a", "b"], column="b", before=30, page=7)
        assert (swept.deleted, swept.statements) == (30, 5)
        assert _plain(db, "SELECT count(*) FROM t WHERE b < 30") == [(0,)]
        assert _plain(db, "SELECT count(*) FROM t") == [(70,)]
        assert all("LIMIT 7" in sql for sql, _ in reader.statements)


class TestKeysetPagesAfter:
    async def test_a_drain_starts_past_its_saved_place(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if ``after=`` is ignored and a drain re-reads from the first row."""
        read = await read_keyset_pages(
            ThreadedDuckDB(db), 'SELECT "a", "b" FROM "t"', ["a", "b"], where={"a": "k2"}, page_size=3, after=("k2", 42)
        )
        assert [row["b"] for row in read.rows] == [52, 62, 72, 82, 92]


class TestWholeReads:
    def test_a_why_it_does_not_name_is_refused(self) -> None:
        reads = WholeReads(
            "keyed_test_whole_reads_total", subject="t", whys={"cold_start": "first"}, expected={"cold_start"}
        )
        with pytest.raises(ValueError, match="not one of its whys"):
            reads.count("warm", table="t")

    def test_gated_is_every_why_but_the_expected(self, caplog: pytest.LogCaptureFixture) -> None:
        """Flips if an unexpected whole read is logged as routine, or gated misses one."""
        reads = WholeReads(
            "keyed_test_gated_total",
            subject="t",
            whys={"cold_start": "first", "reset": "operator", "schema_change": "dropped"},
            expected={"cold_start", "reset"},
        )
        assert reads.gated == frozenset({"schema_change"})
        with caplog.at_level(logging.INFO):
            reads.count("schema_change", table="x")
        assert any(record.levelno == logging.WARNING for record in caplog.records)

    def test_an_expected_why_it_does_not_name_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not among"):
            WholeReads("keyed_test_bad_total", subject="t", whys={"a": "b"}, expected={"c"})


class TestUpdateBefore:
    async def test_a_reap_updates_page_by_page_and_moves_its_rows_out(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a reap updates in one statement, or re-reads the rows it already moved."""
        db.execute("ALTER TABLE t ADD COLUMN status VARCHAR DEFAULT 'dispatching'")
        reader = _Recording(db)
        reaped = await update_before(
            reader,
            "t",
            key=["a", "b"],
            column="b",
            before=30,
            set_to={"status": "failed"},
            by={"status": "dispatching"},
            page=7,
        )
        assert (reaped.deleted, reaped.statements) == (30, 5)
        assert _plain(db, "SELECT count(*) FROM t WHERE status = 'failed'") == [(30,)]
        assert all("LIMIT 7" in sql for sql, _ in reader.statements)

    async def test_an_update_that_leaves_its_rows_in_its_pages_is_refused(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a reap that would read the same page forever is sent."""
        with pytest.raises(ValueError, match="move its rows out"):
            await update_before(ThreadedDuckDB(db), "t", key=["a", "b"], column="b", before=30, set_to={"v": 0}, page=7)


class TestWithoutArrays:
    """Redshift and SQLite have no arrays: sets and keys bind as lists of placeholders."""

    @pytest.fixture
    def lite(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.execute("CREATE TABLE t (a TEXT, b INTEGER, PRIMARY KEY (a, b))")
        conn.executemany("INSERT INTO t VALUES (?, ?)", [(f"k{i % 10}", i) for i in range(100)])
        yield conn
        conn.close()

    async def test_a_key_batch_binds_as_a_list(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a list-bound batch reads rows no key names, or binds an array."""
        reader = _Recording(db)
        keys = KeyBatch(("a", "b"), [("k1", 1), ("k2", 12), ("k9", 1)], as_list=True)
        rows = await read_keyed(reader, "t", keys=keys, columns=["a", "b"])
        assert sorted((row["a"], row["b"]) for row in rows) == [("k1", 1), ("k2", 12)]
        sql, params = reader.statements[0]
        assert '(t."a", t."b") IN (($1, $2), ($3, $4), ($5, $6))' in sql
        assert params == ("k1", 1, "k2", 12, "k9", 1)

    async def test_the_helpers_run_on_sqlite(self, lite: sqlite3.Connection) -> None:
        """Flips if a helper writes SQL SQLite cannot run (arrays, casts) where the caller asked for lists."""
        conn = ThreadedSQLite(lite)
        assert (
            len(await read_keyed(conn, "t", keys=KeyBatch(("a",), [("k1",), ("k2",)], as_list=True), columns=["b"]))
            == 20
        )
        assert await counts_by(conn, "t", column="b", values=AnyOf([1, 11, 2], as_list=True), by={"a": "k1"}) == {
            1: 1,
            11: 1,
        }
        assert await count_at_most(conn, "t", by={"a": "k1"}, at_most=3) == BoundedCount(count=3, more=True)
        seen: list[int] = []
        cursor = None
        while True:
            page = await keyset_page(
                conn, "t", columns=["a", "b"], by={"a": "k3"}, order=[Sort("b", None)], limit=4, cursor=cursor
            )
            seen += [row["b"] for row in page.rows]
            cursor = page.next_cursor
            if cursor is None:
                break
        assert seen == list(range(3, 100, 10))
        swept = await delete_before(conn, "t", key=["a", "b"], column="b", before=20, page=6)
        assert (swept.deleted, swept.statements) == (20, 4)
