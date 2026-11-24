"""A null-safe key batch is sent split by its NULL parts, never ``IS NOT DISTINCT FROM`` a key.

``IS NOT DISTINCT FROM`` is no operator an index reads by: a null-safe batch joined to its table that way read the
table whole once a key on PostgreSQL and YugabyteDB (1,000 keys against 200,000 rows: 3.9 s on PostgreSQL, 5.3 s on YugabyteDB; split, 3-13 ms). Split
(``threetears.interfaces.keyed.rows.null_split``), each NULL pattern's NULL parts are pinned ``IS NULL`` beside
equality on the rest, and the index reads every pattern. These tests hold the split to the rows the old join named
(a property over composite keys with NULLs, reads and writes, plain and Arrow keys), and hold every keyed builder's
null-safe statement free of ``IS NOT DISTINCT FROM``; the plans are ``test_keyed_null_safe_live``'s.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import duckdb
import pyarrow as pa
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from threetears.core.keyed import KeyBatch, Lock, delete_keyed, read_keyed, update_keyed
from threetears.core.keyed.binding import binder_for
from threetears.core.keyed.threaded import ThreadedDuckDB
from threetears.interfaces.keyed.rows import PLAIN_KEYS, KeyBinder, KeyChunk, null_split, read_keyed_statements
from threetears.interfaces.keyed.writes import delete_keyed_statements, update_keyed_statements

_COLUMNS = ("a", "b", "c")
_CASTS = ("text", "bigint", "text")

_A = st.one_of(st.none(), st.sampled_from(["x", "y"]))
_B = st.one_of(st.none(), st.integers(0, 2))
_C = st.one_of(st.none(), st.sampled_from(["p", "q"]))
_KEY = st.tuples(_A, _B, _C)


def _table(rows: Sequence[tuple[Any, ...]]) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect()
    conn.execute("CREATE TABLE t (a TEXT, b BIGINT, c TEXT, v BIGINT)")
    if rows:
        conn.executemany("INSERT INTO t VALUES (?, ?, ?, ?)", [(*row, at) for at, row in enumerate(rows)])
    return conn


def _null_safe_rows(conn: duckdb.DuckDBPyConnection, keys: Sequence[tuple[Any, ...]], width: int) -> list[int]:
    """the rows ``IS NOT DISTINCT FROM`` each distinct key names: what a null-safe batch means."""
    distinct = list(dict.fromkeys(key[:width] for key in keys))
    if not distinct:
        return []
    columns = list(zip(*distinct, strict=True))
    unnested = ", ".join(f"unnest(${at + 1}::{_CASTS[at]}[]) AS k{at}" for at in range(width))
    joined = " AND ".join(f"t.{_COLUMNS[at]} IS NOT DISTINCT FROM k.k{at}" for at in range(width))
    found = conn.execute(f"SELECT t.v FROM t JOIN (SELECT {unnested}) AS k ON {joined}", [list(c) for c in columns])
    return sorted(row[0] for row in found.fetchall())


def _batch(keys: Sequence[tuple[Any, ...]], width: int, arrow: bool) -> KeyBatch:
    columns = [[key[at] for key in keys] for at in range(width)]
    held: list[Any] = [pa.array(column, pa.int64() if at == 1 else pa.string()) for at, column in enumerate(columns)]
    return KeyBatch(_COLUMNS[:width], held if arrow else columns, _CASTS[:width], null_safe=True)


@settings(max_examples=120, deadline=None)
@given(
    rows=st.lists(_KEY, max_size=40),
    named=st.lists(_KEY, max_size=12),
    width=st.integers(1, 3),
    arrow=st.booleans(),
    batch=st.integers(1, 6),
)
def test_the_split_names_the_rows_is_not_distinct_from_named(
    rows: list[tuple[Any, ...]], named: list[tuple[Any, ...]], width: int, arrow: bool, batch: int
) -> None:
    """Flips if the split names a row no key does, misses one (a NULL part matched by equality, a pattern dropped,
    a group's binds misnumbered), or answers a row twice, for reads, updates and deletes alike."""
    keys = [*rows[: len(rows) // 2], *named]
    if not keys:
        return
    conn = _table(rows)
    expected = _null_safe_rows(conn, keys, width)

    async def run() -> None:
        read = await read_keyed(ThreadedDuckDB(conn), "t", keys=_batch(keys, width, arrow), columns=["v"], batch=batch)
        assert sorted(row["v"] for row in read) == expected
        updated = await update_keyed(
            ThreadedDuckDB(conn), "t", values={"v": 1}, keys=_batch(keys, width, arrow), batch=batch, returning=["v"]
        )
        assert updated.count == len(expected)
        deleted = await delete_keyed(ThreadedDuckDB(conn), "t", keys=_batch(keys, width, arrow), batch=batch)
        assert deleted.count == len(expected)
        assert conn.execute("SELECT count(*) FROM t").fetchone() == (len(rows) - len(expected),)

    asyncio.run(run())


def test_the_split_puts_the_keys_with_no_null_first_then_each_pattern_in_code_order() -> None:
    """Flips if the patterns' order follows the keys' (two lockers of the same keys would lock in different orders),
    or the Arrow and plain binders group a chunk differently."""
    columns: list[list[Any]] = [["x", None, "y", None, None], [1, 2, None, None, 3]]
    keys = KeyBatch(("a", "b"), columns, ("text", "bigint"), null_safe=True)
    arrow: list[Any] = [pa.array(columns[0]), pa.array(columns[1], pa.int64())]
    chunks: list[tuple[KeyChunk, KeyBinder]] = [
        (tuple(columns), PLAIN_KEYS),
        (tuple(arrow), binder_for(object(), KeyBatch(("a", "b"), arrow, null_safe=True))),
    ]
    for chunk, binder in chunks:
        split = null_split(keys, chunk, binder)
        assert [group.nulled for group in split] == [(), ("a",), ("b",), ("a", "b")]
        as_lists = [
            None
            if group.keys is None
            else [column.to_pylist() if hasattr(column, "to_pylist") else list(column) for column in group.keys.keys]
            for group in split
        ]
        assert as_lists == [[["x"], [1]], [[2, 3]], [["y"]], None]


#: every shape a null-safe batch's statement takes: widths, a NULL in each part, every part NULL, a lock, a write
_SHAPES = [
    KeyBatch(("a",), [["k", None]], null_safe=True),
    KeyBatch(("a", "b"), [["k", None, "j", None], [1, 2, None, None]], ("text", "bigint"), null_safe=True),
    KeyBatch(("a", "b", "c"), [[None, "k"], ["g", None], [None, 3]], ("text", "text", "bigint"), null_safe=True),
]


@pytest.mark.parametrize("keys", _SHAPES, ids=["one", "two", "three"])
@pytest.mark.parametrize("arrow", [False, True], ids=["plain", "arrow"])
def test_no_null_safe_statement_matches_a_key_is_not_distinct_from(keys: KeyBatch, arrow: bool) -> None:
    """GUARD. Flips if any keyed builder sends a null-safe batch to PostgreSQL or YugabyteDB as ``IS NOT DISTINCT
    FROM`` a key (no index reads by it: the table read whole once a key), read, locked read, update or delete,
    through either binder."""
    if arrow:
        keys = KeyBatch(
            keys.columns,
            [
                pa.array(column, pa.int64() if cast == "bigint" else pa.string())
                for column, cast in zip(keys.keys, keys.casts or ("text",), strict=True)
            ],
            keys.casts,
            null_safe=True,
        )
    binder = binder_for(object(), keys)
    statements = [
        *read_keyed_statements("t", keys=keys, columns=["v"], binder=binder),
        *read_keyed_statements("t", keys=keys, columns=["v"], lock=Lock.UPDATE, binder=binder),
        *update_keyed_statements("t", values={"v": 1}, keys=keys, binder=binder),
        *delete_keyed_statements("t", keys=keys, binder=binder),
    ]
    assert statements
    for sql, _ in statements:
        assert "DISTINCT FROM" not in sql.upper(), sql
    assert all(any("IS NULL" in sql for sql, _ in sent) for sent in (statements[:1], statements[-1:])), statements
