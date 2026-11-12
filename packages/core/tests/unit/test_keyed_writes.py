"""the keyed writes and the drained listing (:func:`update_keyed`, :func:`delete_keyed`, :func:`keyset_rows`) and
``read_keyed``'s ``where=``: each touches exactly the rows a plain statement names, and each statement names its key.

Correctness on an in-process DuckDB (the helpers' SQL is one spelling for PostgreSQL, YugabyteDB and DuckDB); each
conversion's equivalence on PostgreSQL and YugabyteDB is its package's ``assert_same_results`` test.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterator
from typing import Any

import duckdb
import pytest

from threetears.core.backends.protocol import RowcountUnreadableError
from threetears.interfaces.keyed.sweeps import update_before_statement
from threetears.interfaces.keyed.writes import update_keyed_statements
from threetears.core.keyed import (
    AnyOf,
    Derived,
    Fragment,
    Increment,
    KeyBatch,
    Selected,
    Sort,
    StageMode,
    ThreadedDuckDB,
    ThreadedSQLite,
    apply_stage,
    counts_by,
    counts_by_relations,
    delete_keyed,
    keyset_page,
    keyset_rows,
    read_keyed,
    read_packed,
    update_from,
    update_keyed,
)


@pytest.fixture
def db() -> Iterator[duckdb.DuckDBPyConnection]:
    """``t``: 10 groups of 10 rows keyed ``(a, b)``, a nullable ``s`` every third row."""
    conn = duckdb.connect()
    conn.execute("CREATE TABLE t (a VARCHAR, b BIGINT, v INTEGER, s VARCHAR, PRIMARY KEY (a, b))")
    conn.execute(
        "INSERT INTO t SELECT 'k' || (i % 10), i, i, CASE WHEN i % 3 = 0 THEN NULL ELSE 'on' END FROM range(100) r(i)"
    )
    yield conn
    conn.close()


class _Recording:
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


class TestUpdateKeyed:
    async def test_it_sets_only_the_rows_its_key_and_guards_name(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a row outside the key or failing a guard is written, a value or an expression is set wrong, or
        the count is not the rows written."""
        reader = _Recording(db)
        written = await update_keyed(
            reader,
            "t",
            values={"s": "off", "v": Fragment('"v" + $1', (1000,))},
            by={"a": "k1"},
            where=[Fragment('"b" < $1', (50,))],
        )
        assert written.count == 5
        assert written.rows == []
        assert _plain(db, "SELECT b, v, s FROM t WHERE s = 'off' ORDER BY b") == [
            (b, b + 1000, "off") for b in (1, 11, 21, 31, 41)
        ]
        assert _plain(db, "SELECT count(*) FROM t WHERE v >= 1000") == [(5,)]
        sql, params = reader.statements[0]
        assert sql == 'UPDATE "t" SET "s" = $1, "v" = "v" + $2 WHERE "a" = $3 AND ("b" < $4)'
        assert params == ("off", 1000, "k1", 50)

    async def test_it_returns_the_rows_it_wrote(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if ``returning`` loses a written row, or answers one it did not write."""
        written = await update_keyed(
            ThreadedDuckDB(db), "t", values={"s": None}, by={"a": AnyOf(["k1", "k2"])}, returning=["a", "b"]
        )
        assert written.count == 20
        assert sorted((row["a"], row["b"]) for row in written.rows) == sorted(
            (f"k{b % 10}", b) for b in range(100) if b % 10 in (1, 2)
        )

    async def test_its_row_and_value_answer_as_fetchrow_and_fetchval_did(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if ``row`` / ``value`` answer other than ``fetchrow`` / ``fetchval`` of the same ``RETURNING``
        statement: the first row and its first column when a row was written, None when none was."""
        written = await update_keyed(
            ThreadedDuckDB(db), "t", values={"v": -7}, by={"a": "k3", "b": 13}, returning=["b", "v"]
        )
        assert written.row == {"b": 13, "v": -7}
        assert written.value == 13
        missed = await delete_keyed(ThreadedDuckDB(db), "t", by={"a": "k3", "b": 14}, returning=["b"])
        assert missed.rows == [] and missed.row is None and missed.value is None

    async def test_a_none_in_the_key_names_the_null_rows(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if ``by``'s None is bound as a value (matching nothing) rather than read as IS NULL."""
        written = await update_keyed(ThreadedDuckDB(db), "t", values={"v": -1}, by={"a": "k0", "s": None})
        assert written.count == len([b for b in range(0, 100, 10) if b % 3 == 0])

    async def test_a_write_naming_no_key_is_refused(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if an update or delete of every row is sent."""
        with pytest.raises(ValueError, match="names the key"):
            await update_keyed(ThreadedDuckDB(db), "t", values={"v": 0}, by={})
        with pytest.raises(ValueError, match="names the key"):
            await delete_keyed(ThreadedDuckDB(db), "t", by={}, where=[Fragment('"v" > 0')])
        assert _plain(db, "SELECT count(*) FROM t") == [(100,)]

    async def test_a_status_with_no_count_is_refused_not_read_as_no_rows(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a broker reply with no row_count (the bare verb) reads as 0: a fenced save that landed would say
        it lost its race. Every hand-written write read its status through parse_rowcount; the helpers keep that."""

        class _NoCount(_Recording):
            async def execute(self, query: str, *params: Any) -> str:
                return (await super().execute(query, *params)).split()[0]

        with pytest.raises(RowcountUnreadableError):
            await update_keyed(_NoCount(db), "t", values={"v": -1}, by={"a": "k0", "b": 0})
        with pytest.raises(RowcountUnreadableError):
            await delete_keyed(_NoCount(db), "t", by={"a": "k0", "b": 10})


class TestDeleteKeyed:
    async def test_it_deletes_only_the_rows_its_key_and_guards_name(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a row outside the key or failing a guard is deleted, or the count or the returned rows are
        wrong."""
        reader = _Recording(db)
        deleted = await delete_keyed(
            reader, "t", by={"a": "k3"}, where=[Fragment('"s" = $1', ("on",))], returning=["b"]
        )
        on = [b for b in range(3, 100, 10) if b % 3 != 0]
        assert deleted.count == len(on)
        assert sorted(row["b"] for row in deleted.rows) == on
        assert _plain(db, "SELECT count(*) FROM t") == [(100 - len(on),)]
        assert reader.statements[0][0] == 'DELETE FROM "t" WHERE "a" = $1 AND ("s" = $2) RETURNING "b"'


class TestKeysetRows:
    async def test_it_reads_every_row_page_by_page_in_order(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a row is lost or doubled across pages, the order is not the sort's, or the read is one
        statement rather than a page a statement."""
        reader = _Recording(db)
        rows = await keyset_rows(
            reader,
            "t",
            columns=["a", "b", "s"],
            by={"a": "k4"},
            order=[Sort("s", nullable=True), Sort("b", cast="bigint", descending=True)],
            page=3,
        )
        plain = _plain(db, "SELECT a, b, s FROM t WHERE a = 'k4' ORDER BY s ASC NULLS LAST, b DESC")
        assert [(row["a"], row["b"], row["s"]) for row in rows] == plain
        # 10 rows, 3 a page: each statement keeps the row that said more follow, so 4 + 4 + 2, never a row read twice
        assert len(reader.statements) == 3
        assert all("LIMIT 4" in sql for sql, _ in reader.statements)

    async def test_a_one_direction_sort_steps_by_the_row_comparison_off_yugabyte(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        """Flips if a drain of a one-direction sort past its first page asks the dialect more than once, steps by
        the expansion where the database is not YugabyteDB (PostgreSQL plans a bitmap scan and a sort for it), by
        the row comparison on YugabyteDB (it reads a hashed bucket from its start for it), or a one-page drain asks
        anything but its page."""
        order = [Sort("a"), Sort("b", cast="bigint")]
        plain = _plain(db, "SELECT a, b FROM t WHERE s = 'on' ORDER BY a, b")

        class _Yugabyte(_Recording):
            async def fetch(self, query: str, *params: Any) -> list[dict[str, Any]]:
                if query.startswith("SELECT version()"):
                    self.statements.append((query, params))
                    return [{"v": "PostgreSQL 15.12-YB-2026.1.0.0-b0"}]
                return await super().fetch(query, *params)

        for reader, by_row in ((_Recording(db), True), (_Yugabyte(db), False)):
            rows = await keyset_rows(reader, "t", columns=["a", "b"], by={"s": "on"}, order=order, page=7)
            assert [(row["a"], row["b"]) for row in rows] == plain
            asked = [sql for sql, _ in reader.statements]
            assert [sql.startswith("SELECT version()") for sql in asked].count(True) == 1
            assert asked[1].startswith("SELECT version()")
            pages = asked[2:]
            assert len(pages) == -(-len(plain) // 8) - 1
            # the pinned column is bound by equality beside the comparison, never inside it (there only an index it
            # leads takes the comparison as its condition), and leads the order
            assert all('t."s" = $1 AND ' in sql for sql in pages), pages[0]
            assert all(('(t."a", t."b") > ($2::text, $3::text::bigint)' in sql) is by_row for sql in pages), pages[0]
            assert not any('(t."s", ' in sql for sql in pages), pages[0]
            assert all('ORDER BY t."s" ASC, t."a" ASC, t."b" ASC' in sql for sql in pages), pages[0]
        single = _Recording(db)
        rows = await keyset_rows(single, "t", columns=["a", "b"], by={"a": "k1"}, order=order, page=50)
        assert len(rows) == 10 and len(single.statements) == 1

    async def test_an_untyped_set_and_sort_bind_as_the_column_types_them(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a set with no cast is cast (or refused), or a drain past its first page binds an uncast sort
        column's last value as text rather than the row's own value: a caller knowing a table only by its columns'
        names (a delete's cascade to a table only its key declares) cannot name the types, and text bound against
        a bigint key is refused by asyncpg."""
        reader = _Recording(db)
        rows = await keyset_rows(
            reader,
            "t",
            columns=["a", "b"],
            by={"a": AnyOf(["k2", "k5"], cast=None)},
            order=[Sort("b", cast=None)],
            page=4,
        )
        plain = _plain(db, "SELECT a, b FROM t WHERE a IN ('k2', 'k5') ORDER BY b")
        assert [(row["a"], row["b"]) for row in rows] == plain
        assert all('t."a" = ANY($1) ' in sql for sql, _ in reader.statements), reader.statements[0][0]
        past = [params[1:] for _, params in reader.statements[1:]]
        assert past and all(isinstance(value, int) for bound in past for value in bound), past


class TestSortExpression:
    async def test_a_page_sorted_by_an_expression_pages_by_its_value(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a sort expression is not what the pages order and step by (rows lost or doubled across cursor
        pages and drains), is not read under its own name, or may shadow a column read (its cursor would carry the
        column's value, not the expression's)."""
        order = [Sort("a_key", expression='upper(t."a")'), Sort("b", cast="bigint")]
        plain = _plain(db, "SELECT upper(a), b FROM t WHERE s = 'on' ORDER BY upper(a), b")
        paged: list[tuple[Any, ...]] = []
        cursor = None
        while True:
            page = await keyset_page(
                ThreadedDuckDB(db), "t", columns=["b"], by={"s": "on"}, order=order, limit=7, cursor=cursor
            )
            paged += [(row["a_key"], row["b"]) for row in page.rows]
            if page.next_cursor is None:
                break
            cursor = page.next_cursor
        assert paged == plain
        drained = await keyset_rows(ThreadedDuckDB(db), "t", columns=["b"], by={"s": "on"}, order=order, page=7)
        assert [(row["a_key"], row["b"]) for row in drained] == plain
        with pytest.raises(ValueError, match="name of its own"):
            await keyset_page(
                ThreadedDuckDB(db),
                "t",
                columns=["a", "b"],
                by={},
                order=[Sort("a", expression='upper(t."a")')],
                limit=3,
            )


class TestReadKeyedWhere:
    async def test_its_filters_keep_only_the_rows_they_name(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a ``where`` filter is dropped or misnumbered against the key and ``by``'s placeholders, in each
        of the read's statement shapes (one key column, several, null-safe, a list)."""
        guard = Fragment('t."v" > $1 AND t."s" IS NOT NULL', (20,))
        expected = sorted((f"k{b % 10}", b) for b in range(100) if b % 10 in (1, 2) and b > 20 and b % 3 != 0)
        shapes = [
            KeyBatch(("a",), [("k1",), ("k2",)]),
            KeyBatch(("a",), [("k1",), ("k2",)], as_list=True),
            KeyBatch(("a", "b"), [(f"k{b % 10}", b) for b in range(100) if b % 10 in (1, 2)], ("text", "bigint")),
            KeyBatch(("a",), [("k1",), ("k2",)], null_safe=True),
        ]
        for keys in shapes:
            rows = await read_keyed(
                ThreadedDuckDB(db), "t", keys=keys, columns=["a", "b"], by={"s": "on"}, where=[guard]
            )
            assert sorted((row["a"], row["b"]) for row in rows) == expected, keys


class TestReadKeyedSelected:
    async def test_a_selected_expression_is_read_under_its_name(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a Selected column is quoted as an identifier, loses its name, or is read as the typed column."""
        rows = await read_keyed(
            ThreadedDuckDB(db),
            "t",
            keys=KeyBatch(("a",), [("k1",)]),
            columns=["b", Selected("CAST(t.v AS VARCHAR)", "v")],
        )
        assert sorted((row["b"], row["v"]) for row in rows) == [(b, str(b)) for b in range(1, 100, 10)]

    def test_a_selected_expression_takes_no_parameters(self) -> None:
        """Flips if a placeholder in a trusted select item is let through (it would be misnumbered)."""
        with pytest.raises(ValueError, match="no parameters"):
            Selected("v + $1", "v")
        with pytest.raises(ValueError, match="names its expression"):
            Selected(" ", "v")


class TestWritesByAPrefix:
    @pytest.fixture
    def named(self) -> Iterator[duckdb.DuckDBPyConnection]:
        """``n``: text keys either side of a prefix's range, in code-point order."""
        conn = duckdb.connect()
        conn.execute("CREATE TABLE n (signal VARCHAR PRIMARY KEY, v INTEGER)")
        keys = ["job", "job/", "job/1", "job/1/x", "job/zzz", "job0", "jobs", "jo", "other/1", "job/\U0010ffff"]
        conn.executemany("INSERT INTO n VALUES (?, 0)", [[k] for k in keys])
        yield conn
        conn.close()

    async def test_only_the_rows_starting_with_the_prefix_are_written(self, named: duckdb.DuckDBPyConnection) -> None:
        """Flips if the range misses a row starting with the prefix (a trailing U+10FFFF, a deeper path) or reaches
        one that does not ("job0", "jobs", "job" itself)."""
        reader = _Recording(named)
        updated = await update_keyed(reader, "n", values={"v": 1}, prefix={"signal": "job/"})
        deleted = await delete_keyed(reader, "n", prefix={"signal": "job/"}, returning=["signal"])
        starting = ["job/", "job/1", "job/1/x", "job/zzz", "job/\U0010ffff"]
        assert updated.count == len(starting)
        assert sorted(row["signal"] for row in deleted.rows) == sorted(starting)
        assert sorted(row[0] for row in _plain(named, "SELECT signal FROM n")) == [
            "jo",
            "job",
            "job0",
            "jobs",
            "other/1",
        ]
        assert all(
            '"signal" COLLATE "C" >= $' in sql and '"signal" COLLATE "C" < $' in sql for sql, _ in reader.statements
        )
        assert reader.statements[-1][1] == ("job/", "job0")

    async def test_an_empty_or_untyped_prefix_is_refused(self, named: duckdb.DuckDBPyConnection) -> None:
        """Flips if an empty prefix (every row) or a non-text one is sent."""
        with pytest.raises(ValueError, match="empty prefix"):
            await delete_keyed(ThreadedDuckDB(named), "n", prefix={"signal": ""})
        with pytest.raises(ValueError, match="is text"):
            await update_keyed(ThreadedDuckDB(named), "n", values={"v": 1}, prefix={"signal": 7})  # type: ignore[dict-item]
        assert _plain(named, "SELECT count(*) FROM n") == [(10,)]


_BATCHES = [
    KeyBatch(("a",), [("k1",), ("k2",)]),
    KeyBatch(("a",), [("k1",), ("k2",)], as_list=True),
    KeyBatch(("a", "b"), [(f"k{b % 10}", b) for b in range(100) if b % 10 in (1, 2)], ("text", "bigint")),
    KeyBatch(("a", "b"), [(f"k{b % 10}", b) for b in range(100) if b % 10 in (1, 2)], as_list=True),
    KeyBatch(("a",), [("k1",), ("k2",)], null_safe=True),
]


class TestWritesByAKeyBatch:
    @pytest.mark.parametrize("keys", _BATCHES, ids=["one", "one-list", "two", "two-list", "null-safe"])
    async def test_an_update_and_a_delete_write_only_the_batch_s_rows(
        self, db: duckdb.DuckDBPyConnection, keys: KeyBatch
    ) -> None:
        """Flips if a batch's write touches a row no key names, misses one, mis-numbers ``by`` or ``where`` past
        the keys' placeholders, or a batch is not its own statement."""
        reader = _Recording(db)
        updated = await update_keyed(
            reader,
            "t",
            values={"v": Increment(1000)},
            keys=keys,
            by={"s": "on"},
            where=[Fragment('"b" > $1', (20,))],
            batch=7,
        )
        named = sorted(b for b in range(100) if b % 10 in (1, 2) and b > 20 and b % 3 != 0)
        assert updated.count == len(named)
        assert [row[0] for row in _plain(db, "SELECT b FROM t WHERE v >= 1000 ORDER BY b")] == named
        assert len(reader.statements) == -(-len(keys.values) // 7)
        deleted = await delete_keyed(reader, "t", keys=keys, where=[Fragment('"v" >= $1', (1000,))], returning=["b"])
        assert sorted(row["b"] for row in deleted.rows) == named
        assert _plain(db, "SELECT count(*) FROM t") == [(100 - len(named),)]


class TestUpdateFrom:
    async def test_it_sets_the_columns_from_the_source_by_the_key(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a row the source does not name is written, a source value lands on another key, or ``by`` is
        dropped."""
        db.execute("CREATE TABLE src (a VARCHAR, b BIGINT, v INTEGER)")
        db.execute("INSERT INTO src VALUES ('k1', 1, -1), ('k1', 11, -11), ('k2', 2, -2)")
        written = await update_from(
            ThreadedDuckDB(db),
            "t",
            key=["a", "b"],
            source=Derived("SELECT a, b, v FROM src"),
            columns=["v"],
            by={"a": "k1"},
        )
        assert written.count == 2
        assert _plain(db, "SELECT a, b, v FROM t WHERE v < 0 ORDER BY b") == [("k1", 1, -1), ("k1", 11, -11)]


def _staged(conn: Any) -> None:
    conn.execute("CREATE TABLE tgt (k INTEGER PRIMARY KEY, v VARCHAR)")
    conn.execute("INSERT INTO tgt VALUES (1, 'a'), (2, 'b'), (3, 'c')")
    conn.execute("CREATE TABLE stg (k INTEGER, v VARCHAR)")
    conn.execute("INSERT INTO stg VALUES (2, 'b'), (3, 'C'), (4, 'd')")


_AFTER = [(1, "a"), (2, "b"), (3, "C"), (4, "d")]


class TestApplyStage:
    @pytest.mark.parametrize(
        ("mode", "deleted", "written"),
        [(StageMode.REPLACE, 2, 3), (StageMode.UPSERT, 0, 3), (StageMode.DIFFERENCE, 1, 2)],
    )
    @pytest.mark.parametrize("engine", ["duckdb", "sqlite"])
    async def test_each_mode_leaves_the_stage_written_by_its_key(
        self, engine: str, mode: StageMode, deleted: int, written: int
    ) -> None:
        """Flips if a mode leaves the table other than the stage written over it by key, touches a row the stage
        does not name, or (difference) rewrites a row the stage holds as it is, on DuckDB or SQLite."""
        conn: Any = duckdb.connect() if engine == "duckdb" else sqlite3.connect(":memory:", check_same_thread=False)
        _staged(conn)
        reader = ThreadedDuckDB(conn) if engine == "duckdb" else ThreadedSQLite(conn)
        applied = await apply_stage(reader, "tgt", "stg", key=["k"], columns=["k", "v"], mode=mode)
        assert (applied.deleted, applied.written) == (deleted, written)
        assert sorted(tuple(row) for row in conn.execute("SELECT k, v FROM tgt").fetchall()) == _AFTER
        conn.close()

    async def test_a_key_outside_the_columns_is_refused(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a stage is applied by a key it does not write."""
        with pytest.raises(ValueError, match="names its key"):
            await apply_stage(ThreadedDuckDB(db), "t", "t", key=["z"], columns=["a"])


class TestApplyStageAppend:
    @pytest.mark.parametrize("engine", ["duckdb", "sqlite"])
    async def test_append_inserts_every_stage_row_deleting_nothing_and_keeping_duplicates(self, engine: str) -> None:
        """Flips if append deletes a held row, drops a stage row whose key is held or staged twice, or counts other
        than the rows it inserted, on DuckDB or SQLite; and if it takes no key."""
        conn: Any = duckdb.connect() if engine == "duckdb" else sqlite3.connect(":memory:", check_same_thread=False)
        conn.execute("CREATE TABLE seen (k INTEGER, v VARCHAR)")
        conn.execute("INSERT INTO seen VALUES (1, 'a'), (2, 'b')")
        conn.execute("CREATE TABLE stg (k INTEGER, v VARCHAR)")
        conn.execute("INSERT INTO stg VALUES (2, 'b'), (3, 'c'), (3, 'c')")
        reader = ThreadedDuckDB(conn) if engine == "duckdb" else ThreadedSQLite(conn)
        applied = await apply_stage(reader, "seen", "stg", key=[], columns=["k", "v"], mode=StageMode.APPEND)
        assert (applied.deleted, applied.written) == (0, 3)
        held = sorted(tuple(row) for row in conn.execute("SELECT k, v FROM seen").fetchall())
        assert held == [(1, "a"), (2, "b"), (2, "b"), (3, "c"), (3, "c")]
        conn.close()


class TestApplyStageDifferenceScope:
    async def test_a_scoped_difference_leaves_the_scope_holding_exactly_the_stage(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        """Flips if a held row of the scope whose key the stage lacks stays, a row outside the scope is touched, a
        row the stage holds as it is is rewritten, or the counts are not the rows deleted and written."""
        db.execute("CREATE TABLE stg AS SELECT * FROM t WHERE a = 'k1' AND b < 50")
        db.execute("UPDATE stg SET v = v + 1000 WHERE b = 21")
        db.execute("INSERT INTO stg VALUES ('k1', 101, 1, 'on')")
        before_other = _plain(db, "SELECT * FROM t WHERE a <> 'k1' ORDER BY a, b")
        applied = await apply_stage(
            ThreadedDuckDB(db),
            "t",
            "stg",
            key=["a", "b"],
            columns=["a", "b", "v", "s"],
            mode="difference",
            by={"a": "k1"},
        )
        # absent from the stage: 51..91 (5 rows); changed: 21; new: 101
        assert (applied.deleted, applied.written) == (6, 2)
        assert _plain(db, "SELECT * FROM t WHERE a = 'k1' ORDER BY b") == _plain(db, "SELECT * FROM stg ORDER BY b")
        assert _plain(db, "SELECT * FROM t WHERE a <> 'k1' ORDER BY a, b") == before_other

    async def test_a_scope_with_where_and_without_a_difference(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a where= scope is ignored, or a scope is taken by a mode that deletes no absent key."""
        db.execute("CREATE TABLE stg AS SELECT * FROM t WHERE a = 'k2' AND b < 30")
        applied = await apply_stage(
            ThreadedDuckDB(db),
            "t",
            "stg",
            key=["a", "b"],
            columns=["a", "b", "v", "s"],
            mode="difference",
            where=[Fragment('"a" = $1', ("k2",))],
        )
        # k2 holds b = 2, 12, ..., 92; the stage 2, 12, 22 as they are: the other 7 go, nothing is written
        assert (applied.deleted, applied.written) == (7, 0)
        assert _plain(db, "SELECT count(*) FROM t") == [(93,)]
        for mode in (StageMode.REPLACE, StageMode.UPSERT):
            with pytest.raises(ValueError, match="scope"):
                await apply_stage(
                    ThreadedDuckDB(db), "t", "stg", key=["a", "b"], columns=["a", "b"], mode=mode, by={"a": "k2"}
                )


class TestApplyStageDistinctDifference:
    async def test_a_distinct_difference_writes_a_row_staged_twice_once(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if distinct=True writes a doubled stage row twice, the default stops writing it twice, or a mode
        other than difference takes distinct."""
        db.execute("CREATE TABLE seen (k INTEGER, v VARCHAR)")
        db.execute("INSERT INTO seen VALUES (1, 'a')")
        db.execute("CREATE TABLE stg (k INTEGER, v VARCHAR)")
        db.execute("INSERT INTO stg VALUES (1, 'a'), (2, 'b'), (2, 'b')")
        reader = ThreadedDuckDB(db)
        applied = await apply_stage(
            reader, "seen", "stg", key=["k"], columns=["k", "v"], mode="difference", distinct=True
        )
        assert (applied.deleted, applied.written) == (0, 1)
        assert _plain(db, "SELECT k, v FROM seen ORDER BY k") == [(1, "a"), (2, "b")]
        db.execute("DELETE FROM seen WHERE k = 2")
        applied = await apply_stage(reader, "seen", "stg", key=["k"], columns=["k", "v"], mode="difference")
        assert applied.written == 2
        for mode in (StageMode.REPLACE, StageMode.UPSERT, StageMode.APPEND):
            with pytest.raises(ValueError, match="distinct"):
                await apply_stage(reader, "seen", "stg", key=["k"], columns=["k", "v"], mode=mode, distinct=True)


class TestCountsEveryValue:
    async def test_every_value_of_a_prefix_is_counted_a_page_at_a_time(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a value is lost or double-counted across pages, a value outside the prefix is counted, or the
        count is one statement rather than a page a statement; refused with no prefix (the whole table)."""
        reader = _Recording(db)
        counted = await counts_by(reader, "t", column="s", by={"a": "k3"}, batch=1)
        assert counted == dict(_plain(db, "SELECT s, count(*) FROM t WHERE a = 'k3' GROUP BY s"))
        # the 'on' page, the empty page past it, the NULL group's count
        assert len(reader.statements) == 3
        counted = await counts_by(reader, "t", column="b", by={"a": "k3"}, batch=4, cast="bigint")
        assert counted == {b: 1 for b in range(3, 100, 10)}
        paged = [sql for sql, _ in reader.statements[3:] if "GROUP BY" in sql]
        assert len(paged) == 3
        assert all("LIMIT 4" in sql for sql in paged)
        with pytest.raises(ValueError, match="the prefix of t it counts every value within"):
            await counts_by(reader, "t", column="b")


class TestCountsPacked:
    async def test_every_group_of_every_relation_comes_back_in_one_packed_statement(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        """Flips if the packed count answers other counts than the unpacked one, takes more than one statement, or
        answers more rows than its cap (several groups a row)."""
        db.execute("CREATE TABLE u AS SELECT * FROM t WHERE b % 2 = 0")
        tables = {"t": "t", "u": "u"}
        values = AnyOf([f"k{i}" for i in range(10)] + ["absent"], "text")
        unpacked = await counts_by_relations(ThreadedDuckDB(db), tables, column="a", values=values, by={"s": "on"})
        reader = _Recording(db)
        answered: list[int] = []

        class _Counting(_Recording):
            async def fetch(self, query: str, *params: Any) -> list[dict[str, Any]]:
                rows = await super().fetch(query, *params)
                answered.append(len(rows))
                return rows

        reader = _Counting(db)
        packed = await counts_by_relations(reader, tables, column="a", values=values, by={"s": "on"}, packed_rows=4)
        assert packed == unpacked and packed["t"] and packed["u"]
        assert len(reader.statements) == 1
        assert answered[0] <= 4 < sum(len(groups) for groups in packed.values())

    async def test_a_packed_count_names_how_many_groups_it_may_answer(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a packed count with neither values nor a bound on its groups is sent (its packing unknown)."""
        with pytest.raises(ValueError, match="groups="):
            await counts_by_relations(
                ThreadedDuckDB(db), {"t": "t"}, column="a", where=[Fragment('"v" > $1', (5,))], packed_rows=10
            )
        counted = await counts_by_relations(
            ThreadedDuckDB(db), {"t": "t"}, column="a", where=[Fragment('"v" > $1', (89,))], packed_rows=3, groups=10
        )
        assert counted == {"t": {f"k{i}": 1 for i in range(10)}}


class TestCountsPackedNullsAndName:
    async def test_a_null_group_survives_the_packing_and_the_text_takes_the_caller_s_name(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        """Flips if a NULL group is lost from a packed answer (a NULL in a ``||`` nulls the whole text), or the
        packed text is not named as the caller asks."""
        where = [Fragment('"v" > $1', (50,))]
        unpacked = await counts_by_relations(ThreadedDuckDB(db), {"t": "t"}, column="s", where=where)
        reader = _Recording(db)
        packed = await counts_by_relations(
            reader, {"t": "t"}, column="s", where=where, packed_rows=1, groups=2, packed_as="run_rows"
        )
        assert packed == unpacked and None in packed["t"]
        assert 'AS "run_rows"' in reader.statements[0][0] and "AS packed" not in reader.statements[0][0]


class TestReadPacked:
    async def test_the_texts_come_back_packed_in_one_statement_of_at_most_its_rows(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        """Flips if a packed read answers other texts than the plain read of the same rows (a pin or filter lost, a
        text split at its own separators), takes more than one statement, answers more rows than its cap, keeps a
        NULL as a text, or takes a cap under one row."""
        where = [Fragment('r."v" >= $1', (40,))]
        plain = sorted(
            str(row[0]) for row in db.execute("SELECT a || chr(31) || b FROM t WHERE s = 'on' AND v >= 40").fetchall()
        )
        answered: list[int] = []

        class _Counting(_Recording):
            async def fetch(self, query: str, *params: Any) -> list[dict[str, Any]]:
                rows = await super().fetch(query, *params)
                answered.append(len(rows))
                return rows

        reader = _Counting(db)
        texts = await read_packed(
            reader,
            "t",
            Selected("r.a || chr(31) || CAST(r.b AS VARCHAR)", "run_text"),
            by={"s": "on"},
            where=where,
            packed_rows=5,
            items=40,
        )
        assert sorted(texts) == plain and len(texts) == 40
        assert len(reader.statements) == 1 and answered == [5]
        nullable = await read_packed(
            ThreadedDuckDB(db), "t", "s", where=[Fragment('r."b" < $1', (6,))], packed_rows=2, items=6
        )
        assert sorted(nullable) == ["on"] * 4
        assert await read_packed(ThreadedDuckDB(db), "t", "s", packed_rows=1, items=0) == []
        with pytest.raises(ValueError, match="at least one row"):
            await read_packed(ThreadedDuckDB(db), "t", "s", packed_rows=0, items=3)

    async def test_a_long_text_is_never_cut_to_redshift_s_bare_varchar(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a packed read (or a packed count's value text) casts to a bare VARCHAR, which Redshift reads as
        VARCHAR(256) and cuts a longer text (a long model_version) to, or if a text past 256 characters does not come
        back whole."""
        long = "m" * 1000
        db.execute("CREATE TABLE longs AS SELECT 'on' AS s, $1 AS model_version", [long])
        reader = _Recording(db)
        texts = await read_packed(reader, "longs", "model_version", by={"s": "on"}, packed_rows=1, items=1)
        assert texts == [long]
        counted = await counts_by_relations(
            reader,
            {"longs": "longs"},
            column="model_version",
            where=[Fragment('"s" = $1', ("on",))],
            packed_rows=1,
            groups=1,
        )
        assert counted == {"longs": {long: 1}}
        for sql, _params in reader.statements:
            assert not re.search(r"AS VARCHAR\)", sql), sql
            assert "AS VARCHAR(65535))" in sql, sql


def _changes(db: duckdb.DuckDBPyConnection) -> None:
    """two report tables: ``up`` a text stamp (a warehouse's), ``called`` a timestamp, ``src`` the data time."""
    for name, upload_floor in (("ra", 0), ("rb", 30)):
        db.execute(
            f"CREATE TABLE {name} AS SELECT 'S' || (i % 3) AS state, CASE WHEN i % 7 = 0 THEN NULL ELSE 'g' || (i % 4) "
            "END AS geo, CAST(TIMESTAMP '2026-11-03 00:00:00' + to_minutes(i) AS VARCHAR) AS up, "
            "CASE WHEN i % 5 = 0 THEN TIMESTAMP '2026-11-04 00:00:00' + to_minutes(i) END AS called, "
            f"CAST(TIMESTAMP '2026-11-02 00:00:00' + to_minutes(i) AS VARCHAR) AS src FROM range({upload_floor}, 60) r(i)"
        )


_MARKS: dict[str, dict[str, str | None]] = {
    "ra": {"up": "2026-11-03 00:40:00", "called": None},
    "rb": {"up": "2026-11-03 00:50:00", "called": "2026-11-04 00:45:00"},
}


def _hand_written(db: duckdb.DuckDBPyConnection) -> set[tuple[Any, ...]]:
    """the change statement as reports wrote it by hand: rows past either mark, grouped by state and geography."""
    parts: list[str] = []
    params: list[Any] = []
    for index, (name, marks) in enumerate(_MARKS.items()):
        params += [marks["up"] or "0001-01-01 00:00:00", marks["called"] or "0001-01-01 00:00:00"]
        up, called = len(params) - 1, len(params)
        parts.append(
            f"SELECT {index} AS t, state, CAST(geo AS VARCHAR) AS grp, count(*) AS n, CAST(max(src) AS VARCHAR) AS src, "
            f"max(CASE WHEN CAST(up AS TIMESTAMP) > CAST(${up} AS TIMESTAMP) THEN 1 ELSE 0 END) AS fresh FROM {name} "
            f"WHERE CAST(up AS TIMESTAMP) > CAST(${up} AS TIMESTAMP) OR CAST(called AS TIMESTAMP) > CAST(${called} AS "
            "TIMESTAMP) GROUP BY state, geo"
        )
    rows = db.execute(" UNION ALL ".join(parts), params).fetchall()
    return {(list(_MARKS)[t], state, grp, n, src, bool(fresh)) for t, state, grp, n, src, fresh in rows}


class TestCountsSinceMarks:
    async def test_every_relation_s_groups_past_its_marks_come_in_one_statement(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        """Flips if the groups past each relation's marks (either stamp, a missing mark the floor, a text stamp read
        as a timestamp) differ from the hand-written change statement, packed or not, a NULL group or the fresh
        flag is lost, or either takes more than one statement."""
        _changes(db)
        expected = _hand_written(db)
        assert any(grp is None for _, _, grp, *_ in expected) and any(row[-1] for row in expected)
        for packed_rows in (None, 3):
            reader = _Recording(db)
            answer = await counts_by_relations(
                reader,
                {"ra": "ra", "rb": "rb"},
                column="state",
                since=_MARKS,
                group={"ra": "geo", "rb": "geo"},
                max_of=["src"],
                fresh="up",
                packed_rows=packed_rows,
                groups=40,
                packed_as="run_rows",
            )
            got = {
                (label, found.value, found.group, found.rows, found.newest["src"], found.fresh)
                for label, groups in answer.items()
                for found in groups
            }
            assert got == expected, packed_rows
            assert len(reader.statements) == 1

    async def test_a_count_past_marks_is_refused_what_it_cannot_answer(self, db: duckdb.DuckDBPyConnection) -> None:
        """Flips if a relation with no marks, a fresh stamp no relation marks, values= beside since=, or group= with
        no since= is sent rather than refused."""
        _changes(db)
        conn = ThreadedDuckDB(db)
        with pytest.raises(ValueError, match="none for"):
            await counts_by_relations(conn, {"ra": "ra"}, column="state", since={"ra": {}})
        with pytest.raises(ValueError, match="fresh="):
            await counts_by_relations(conn, {"ra": "ra"}, column="state", since={"ra": {"up": None}}, fresh="called")
        with pytest.raises(ValueError, match="values="):
            await counts_by_relations(
                conn, {"ra": "ra"}, column="state", since={"ra": {"up": None}}, values=AnyOf(["S1"], "text")
            )
        with pytest.raises(ValueError, match="need since="):
            await counts_by_relations(conn, {"ra": "ra"}, column="state", values=AnyOf(["S1"], "text"), max_of=["src"])


def test_a_none_assignment_is_the_literal_null_and_binds_nothing() -> None:
    """Flips if a keyed write's ``SET`` binds a None (a bind the driver must type by its column: asyncpg refuses
    YugabyteDB's built-in ``vector`` there), or a None pin in ``by`` stops reading as ``IS NULL``.

    Spec from hub #37 (2026-11-12): None renders as ``NULL`` in SET only; a filter keeps ``IS NULL``."""
    [(sql, params)] = update_keyed_statements(
        "t", values={"embedding": None, "name": "n", "note": None}, by={"id": 7, "gone": None}
    )
    assert '"embedding" = NULL' in sql, sql
    assert '"note" = NULL' in sql, sql
    assert '"name" = $1' in sql, sql
    assert '"gone" IS NULL' in sql, sql
    assert params == ["n", 7], params
    swept, bound = update_before_statement(
        "t", key=("id",), column="at", before=3, set_to={"status": None}, by={"status": "dispatching"}, page=5
    )
    assert '"status" = NULL' in swept, swept
    assert None not in bound, bound
