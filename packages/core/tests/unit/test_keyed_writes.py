"""the keyed writes and the drained listing (:func:`update_keyed`, :func:`delete_keyed`, :func:`keyset_rows`) and
``read_keyed``'s ``where=``: each touches exactly the rows a plain statement names, and each statement names its key.

Correctness on an in-process DuckDB (the helpers' SQL is one spelling for PostgreSQL, YugabyteDB and DuckDB); each
conversion's equivalence on PostgreSQL and YugabyteDB is its package's ``assert_same_results`` test.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from typing import Any

import duckdb
import pytest

from threetears.core.keyed import (
    AnyOf,
    Derived,
    Fragment,
    Increment,
    KeyBatch,
    Sort,
    StageMode,
    ThreadedDuckDB,
    ThreadedSQLite,
    apply_stage,
    counts_by,
    delete_keyed,
    keyset_rows,
    read_keyed,
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
        assert sql == 'UPDATE "t" SET "s" = $1, "v" = "v" + $2 WHERE "a" = $3 AND "b" < $4'
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
        assert reader.statements[0][0] == 'DELETE FROM "t" WHERE "a" = $1 AND "s" = $2 RETURNING "b"'


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
        assert len(reader.statements) == 4
        assert all("LIMIT 4" in sql for sql, _ in reader.statements)


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
