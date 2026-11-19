"""Arrow rows matched by a key of several columns, in Arrow's compute kernels alone: never through Acero.

**Why not Acero** (``Table.join``, ``Table.group_by``, ``Table.filter`` given an expression): each runs an Acero plan
whose answer is ``DeclarationToTable``'s, and that answer is whatever the plan's table sink holds when the plan
reports itself finished (Arrow 25.0.1 ``acero/exec_plan.cc``, ``DeclarationToTableImpl``: ``*output_table``, never
checked). The sink sets it in a task it adds to the plan's scheduler as its input ends (``sink_node.cc``,
``ConsumingSinkNode::Finish``); a plan that reports itself finished before that task has run answers a null table with
an OK status. pyarrow wraps it as a ``Table`` whose C++ table is null, and the first use of it dereferences null:
reports' loop_timing run of 2026-11-18 04:40 died so (SIGSEGV, ``KERN_INVALID_ADDRESS at 0x8``, the table's schema
read at offset 8 of a null table: ``Table.column`` on ``follower_parts._named_rows``' left semi join, macOS crash
report ``python3.14-2026-11-18-044013.ips``). Arrow's own batch path checks for the same plan finishing early
("Plan finished but it did not emit the expected number of batches", apache/arrow GH-49948's intermittent parallel
``HashJoin`` failure); the table path does not. Rare and timing-bound: one run of eleven full loop_timing runs on the
build machine, none in some forty runs of the dying test alone.

So a key of several columns is coded instead: each column's values dictionary-encoded (a hash kernel) over every
table's rows together, the codes combined into one int64 per row, and the rows matched by ``is_in`` over the codes
(another hash kernel) or made distinct by ``np.unique``. Equal codes are exactly equal keys: each column's code is
its value's place in that column's dictionary, and the combination is a mixed-radix number, re-coded whenever the
next column would overflow it. A null is a value of its own (``null_encoding="encode"``), as a key column never holds
one (L3's primary key, a settle's keys) but a join on ``=`` would never match one.

Moved here from reports (``aibots_reports.pod.arrow_keys``, SG1 2026-11-18), where it replaced four Acero sites; 3tears'
own :meth:`threetears.core.cache.duckdb.DuckDBBackend.storage_arrow` keeps the last row of each key by
:func:`last_rows` instead of a ``group_by``. The enforcement rule ``tests/enforcement/test_no_acero_tables.py`` keeps
every package's source off Acero.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

__all__ = ["distinct_keys", "key_codes", "last_rows", "rows_named", "rows_not_named"]

#: the most distinct keys a combined code may count before the next column is folded in: past it the codes so far are
#: coded again (their own dictionary), so the mixed-radix number never passes an int64
_CODE_ROOM: Final = 1 << 62


def _column_codes(columns: Sequence[pa.ChunkedArray | pa.Array]) -> tuple[np.ndarray, int]:
    """one column's values over several tables, each value coded by its place in one dictionary of them all.

    :param columns: the column of each table; each after the first is cast to the first's type (a column of nulls
        alone is typed null)
    :ptype columns: Sequence[pa.ChunkedArray | pa.Array]
    :return: every row's code (the tables' rows one after another), and the dictionary's size
    :rtype: tuple[np.ndarray, int]
    """
    kind = columns[0].type
    chunks = [
        chunk
        for column in columns
        for chunk in (column.chunks if isinstance(column, pa.ChunkedArray) else [column])
        if len(chunk)
    ]
    values = pa.concat_arrays([chunk.cast(kind) for chunk in chunks]) if chunks else pa.array([], kind)
    encoded = pc.dictionary_encode(values, null_encoding="encode")
    codes = encoded.indices.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    return codes, max(len(encoded.dictionary), 1)


def key_codes(tables: Sequence[pa.Table], key: Sequence[str], *, room: int = _CODE_ROOM) -> list[np.ndarray]:
    """each table's rows' keys as int64 codes, one code a key: two rows of any of the tables hold the same code
    exactly when they hold the same key.

    :param tables: the tables, each holding every key column, of the same types
    :ptype tables: Sequence[pa.Table]
    :param key: the key's columns
    :ptype key: Sequence[str]
    :param room: the most distinct keys a combined code counts before it is coded again (:data:`_CODE_ROOM`, so it
        never passes an int64; smaller only to hold the re-coding to account)
    :ptype room: int
    :return: each table's codes, a row's at its row number
    :rtype: list[np.ndarray]
    :raises ValueError: for no key column
    """
    if not key:
        raise ValueError("a key names at least one column")
    combined: np.ndarray | None = None
    count = 1
    for column in key:
        codes, size = _column_codes([table[column] for table in tables])
        if combined is None:
            combined, count = codes, size
            continue
        if count * size >= room:
            recoded = pc.dictionary_encode(pa.array(combined))
            combined = recoded.indices.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            count = max(len(recoded.dictionary), 1)
        combined = combined * size + codes
        count *= size
    assert combined is not None  # noqa: S101 -- a key names at least one column
    bounds = np.cumsum([0, *(table.num_rows for table in tables)])
    return [combined[start:end] for start, end in zip(bounds[:-1], bounds[1:], strict=True)]


def rows_named(held: pa.Table, named: pa.Table, key: Sequence[str], *, room: int = _CODE_ROOM) -> np.ndarray:
    """the row numbers of ``held``'s rows whose key ``named`` holds: a left semi join's rows, by number.

    :param held: the rows looked in
    :ptype held: pa.Table
    :param named: the keys looked for (``key``'s columns, of ``held``'s types; other columns are not read)
    :ptype named: pa.Table
    :param key: the key's columns
    :ptype key: Sequence[str]
    :param room: the most distinct keys a combined code counts before it is coded again (:func:`key_codes`)
    :ptype room: int
    :return: the row numbers, ascending
    :rtype: np.ndarray

    Coded by ``named``'s values alone, a column at a time, the column naming the most distinct values first: its held
    values looked up (``index_in``) among them, and only the rows still holding a named value looked up in the next
    column, so the part is probed whole once and its few candidates after. Coded over both sides together
    (:func:`key_codes`), a 28k-row part matched against 382 named keys of six columns took 1.46 ms, 3x the Acero join
    it replaces (0.48 ms on Arrow's thread pool); every column probed whole, 1.16 ms (``index_in`` ~140 us a text
    column of 28k rows); so, 0.53 ms on the calling thread alone (SG1 bench, 2026-11-18).
    """
    found = np.empty(0, dtype=np.int64)
    if held.num_rows and named.num_rows:
        kinds = {column: held.schema.field(column).type for column in key}
        values = {column: pc.unique(named[column].cast(kinds[column])) for column in key}
        rows = np.arange(held.num_rows, dtype=np.int64)
        held_code = np.zeros(held.num_rows, dtype=np.int64)
        named_code = np.zeros(named.num_rows, dtype=np.int64)
        count = 1
        for column in sorted(key, key=lambda name: -len(values[name])):
            size = max(len(values[column]), 1)
            whole = len(rows) == held.num_rows
            at_held = _places(held[column] if whole else held[column].take(pa.array(rows)), values[column])
            if count * size >= room:
                distinct = pa.array(np.unique(named_code))
                held_code, named_code = _places(pa.array(held_code), distinct), _places(pa.array(named_code), distinct)
                count = len(distinct)
            # a held row with a value no named key holds, or whose columns so far hold no named key's together,
            # matches none: only the rest are looked up in the next column
            kept = (at_held >= 0) & (held_code >= 0)
            rows, held_code = rows[kept], held_code[kept] * size + at_held[kept]
            named_code = named_code * size + _places(named[column].cast(kinds[column]), values[column])
            count *= size
            if not len(rows):
                break
        if len(rows):
            hit = pc.is_in(pa.array(held_code), value_set=pa.array(named_code)).to_numpy(zero_copy_only=False)
            found = rows[hit]
    return found


def _places(values: pa.ChunkedArray | pa.Array, among: pa.Array) -> np.ndarray:
    """each value's place among ``among`` (a null's is a null's place there), -1 for a value not among them.

    :param values: the values
    :ptype values: pa.ChunkedArray | pa.Array
    :param among: distinct values, of ``values``' type
    :ptype among: pa.Array
    :return: the places
    :rtype: np.ndarray
    """
    placed = pc.fill_null(pc.index_in(values, value_set=among, skip_nulls=False), -1)
    places: np.ndarray = np.asarray(placed.to_numpy(), dtype=np.int64)
    return places


def rows_not_named(held: pa.Table, named: pa.Table, key: Sequence[str]) -> pa.Array:
    """the row numbers of ``held``'s rows whose key ``named`` does not hold: a left anti join's rows, by number.

    :param held: the rows looked in
    :ptype held: pa.Table
    :param named: the keys looked for (``key``'s columns, of ``held``'s types; other columns are not read)
    :ptype named: pa.Table
    :param key: the key's columns
    :ptype key: Sequence[str]
    :return: the row numbers, ascending, as an int64 array (what ``Table.take`` takes)
    :rtype: pa.Array
    """
    kept = np.ones(held.num_rows, dtype=bool)
    kept[rows_named(held, named, key)] = False
    return pa.array(np.flatnonzero(kept), pa.int64())


def distinct_keys(rows: pa.Table, key: Sequence[str]) -> pa.Table:
    """each key ``rows`` holds once, in the order each first appears: ``group_by(key).aggregate([])``'s keys.

    :param rows: the rows
    :ptype rows: pa.Table
    :param key: the key's columns
    :ptype key: Sequence[str]
    :return: the keys (``key``'s columns)
    :rtype: pa.Table
    """
    keys = rows.select(list(key))
    if rows.num_rows:
        (codes,) = key_codes([rows], key)
        _, first = np.unique(codes, return_index=True)
        keys = keys.take(pa.array(np.sort(first)))
    return keys


def last_rows(rows: pa.Table, key: Sequence[str]) -> np.ndarray:
    """the row number of the last row of each key ``rows`` holds, ascending: the rows a batch of dict rows keeps when
    a later row of a key replaces an earlier one (``group_by(key).aggregate([(row, "max")])``'s rows).

    :param rows: the rows
    :ptype rows: pa.Table
    :param key: the key's columns
    :ptype key: Sequence[str]
    :return: the row numbers, ascending
    :rtype: np.ndarray
    """
    if not rows.num_rows:
        return np.empty(0, dtype=np.int64)
    (codes,) = key_codes([rows], key)
    # the first of each code in the rows reversed is the last of it in the rows
    _, first_reversed = np.unique(codes[::-1], return_index=True)
    kept: np.ndarray = np.sort(rows.num_rows - 1 - first_reversed).astype(np.int64, copy=False)
    return kept
