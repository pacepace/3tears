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

Moved here from reports (``aibots_reports.pod.arrow_keys``, 2026-11-18), where it replaced four Acero sites; 3tears'
own :meth:`threetears.core.cache.duckdb.DuckDBBackend.storage_arrow` keeps the last row of each key by
:func:`last_rows` instead of a ``group_by``. The enforcement rule ``tests/enforcement/test_no_acero_tables.py`` keeps
every package's source off Acero.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from threetears.core.frozen_state import EMPTY

__all__ = [
    "CANONICAL_CASTS",
    "canonical_texts",
    "distinct_keys",
    "key_codes",
    "last_rows",
    "rows_named",
    "rows_found",
    "rows_not_named",
    "stored_instants",
]

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


def key_codes(tables: Sequence[pa.Table], key: Sequence[str]) -> list[np.ndarray]:
    """each table's rows' keys as int64 codes, one code a key: two rows of any of the tables hold the same code
    exactly when they hold the same key.

    :param tables: the tables, each holding every key column, of the same types
    :ptype tables: Sequence[pa.Table]
    :param key: the key's columns
    :ptype key: Sequence[str]
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
        if count * size >= _CODE_ROOM:
            recoded = pc.dictionary_encode(pa.array(combined))
            combined = recoded.indices.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            count = max(len(recoded.dictionary), 1)
        combined = combined * size + codes
        count *= size
    assert combined is not None  # noqa: S101 -- a key names at least one column
    bounds = np.cumsum([0, *(table.num_rows for table in tables)])
    return [combined[start:end] for start, end in zip(bounds[:-1], bounds[1:], strict=True)]


def rows_named(held: pa.Table, named: pa.Table, key: Sequence[str]) -> np.ndarray:
    """the row numbers of ``held``'s rows whose key ``named`` holds: a left semi join's rows, by number.

    :param held: the rows looked in
    :ptype held: pa.Table
    :param named: the keys looked for (``key``'s columns, of ``held``'s types; other columns are not read)
    :ptype named: pa.Table
    :param key: the key's columns
    :ptype key: Sequence[str]
    :return: the row numbers, ascending
    :rtype: np.ndarray

    Coded by ``named``'s values alone, a column at a time, the column naming the most distinct values first: its held
    values looked up (``index_in``) among them, and only the rows still holding a named value looked up in the next
    column, so the part is probed whole once and its few candidates after. Coded over both sides together
    (:func:`key_codes`), a 28k-row part matched against 382 named keys of six columns took 1.46 ms, 3x the Acero join
    it replaces (0.48 ms on Arrow's thread pool); every column probed whole, 1.16 ms (``index_in`` ~140 us a text
    column of 28k rows); so, 0.53 ms on the calling thread alone (2026-11-18).
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
            if count * size >= _CODE_ROOM:
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


# -- an instant in a key, in one spelling --------------------------------------------------------------------------

#: ISO 8601's basic forms and Python's separators, rewritten in its extended form before an instant is read: each a
#: (pattern, replacement) for Arrow's ``replace_substring_regex`` (RE2), in order
_ISO_EXTENDED: Final = (
    # a basic date, 20261103, then nothing or no digit
    (r"^(\d{4})(\d{2})(\d{2})(\D.*|)$", r"\1-\2-\3\4"),
    # any one character between the date and the hour (``datetime.fromisoformat`` takes any)
    (r"^(\d{4}-\d{2}-\d{2})[^\d](\d)", r"\1T\2"),
    # a basic time, T235900 or T2359
    (r"^(\d{4}-\d{2}-\d{2}T)(\d{2})(\d{2})(\d{2})(.*)$", r"\1\2:\3:\4\5"),
    (r"^(\d{4}-\d{2}-\d{2}T)(\d{2})(\d{2})(\D.*|)$", r"\1\2:\3\4"),
    # a comma before the fraction, and a fraction past the microsecond (cut, as ``fromisoformat`` cuts it)
    (r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}),(\d)", r"\1.\2"),
    (r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6})\d+", r"\1"),
)

#: an instant's text in the extended form: its date, then its hour, minute, second and fraction, then its offset
_ISO_INSTANT: Final = (
    r"^(?P<date>\d{4}-\d{2}-\d{2})"
    r"(?:T(?P<hour>\d{2})(?::(?P<minute>\d{2})(?::(?P<second>\d{2})(?:\.(?P<fraction>\d{1,6}))?)?)?"
    r"(?P<zone>Z|[+-]\d{2}(?::?\d{2})?)?)?$"
)
_OFFSET: Final = r"^(?P<sign>[+-])(?P<hours>\d{2}):?(?P<minutes>\d{2})?$"

#: an instant's stored text (``threetears.core.cache.base.stored_instant``): a zoned one in UTC with its offset, a
#: naive one its wall clock; ``%S`` carries the microseconds of a ``us`` timestamp
_STORED_ZONED: Final = "%Y-%m-%dT%H:%M:%S+00:00"
_STORED_NAIVE: Final = "%Y-%m-%dT%H:%M:%S"
_SECOND: Final = 1_000_000
#: a zoned instant's stored text, as RE2 reads it
_STORED_RE2: Final = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}\+00:00$"


def _group(found: pa.Array, at: int) -> pa.Array:
    """a regex's group ``at`` of each value (``extract_regex``): its text, empty where it took no part, null where
    the value did not match."""
    group: pa.Array = pc.struct_field(found, [at])
    taken: pa.Array = pc.if_else(pc.is_valid(found), pc.fill_null(group, ""), pa.nulls(len(found), pa.string()))
    return taken


def _or(values: pa.Array, default: str) -> pa.Array:
    """a regex group's text, ``default`` where the group took no part."""
    found: pa.Array = pc.if_else(pc.equal(values, ""), default, values)
    return found


def stored_instants(values: pa.ChunkedArray | pa.Array) -> pa.Array:
    """a column of instants as text in any ISO 8601 spelling, each as
    :func:`~threetears.core.cache.base.stored_instant_text` keeps it: a zoned instant in UTC to the microsecond
    (``2026-11-03T23:59:00.000000+00:00``), a naive one its wall clock to the microsecond, and text that is no instant
    as it is. In Arrow's kernels alone, never a Python object a value.

    **Why** (moved from reports' ``follower_parts.stored_instants``). The rail answers a ``timestamptz`` as the hub's
    JSON writes it (``2026-11-03T23:59:00Z``) and a change record names a key's instant as ``key_text`` does: matched
    as text, a deleted row keyed by its instant was never found and stayed in the copy. Reports read a column holding
    any value without an offset value by value in Python (``stored_instant_text``); here every spelling it meets is
    read in Arrow: the extended and basic forms, any separator between date and time, a comma or a fraction past the
    microsecond (cut), an offset of ``Z``, ``+HH``, ``+HHMM`` or ``+HH:MM``. A text the form does not take (a week
    or ordinal date, an offset in seconds, ``24:00``, a date that is not one: Feb 30) is kept as it is: those are
    never an instant L3 or the rail writes.

    :param values: the column, as text (or any type cast to text)
    :ptype values: pa.ChunkedArray | pa.Array
    :return: the column, each instant in its stored spelling; a null stays null
    :rtype: pa.Array
    """
    column = values.combine_chunks() if isinstance(values, pa.ChunkedArray) else values
    text: pa.Array = pc.cast(column, pa.string())
    if not len(text) or pc.all(pc.match_substring_regex(text, _STORED_RE2)).as_py() is not False:
        # each value spelled as stored already (a change record's key_text), or null: one pass, nothing rewritten
        return text
    try:
        # every value an instant with its offset (the rail's ``Z``): Arrow's own ISO parse, the cast the column takes
        # whole or not at all; 5x faster than the general reading below (200k values: 155 ms against 735 ms, Apple M4)
        zoned_only: pa.Array = pc.strftime(pc.cast(text, pa.timestamp("us", tz="UTC")), format=_STORED_ZONED)
    except pa.ArrowInvalid, pa.ArrowNotImplementedError:
        pass  # NOSILENT: a value with no offset, or no instant: the general reading below takes each
    else:
        return zoned_only
    extended = text
    for pattern, replacement in _ISO_EXTENDED:
        extended = pc.replace_substring_regex(extended, pattern=pattern, replacement=replacement)
    found = pc.extract_regex(extended, pattern=_ISO_INSTANT)
    date, hour, minute, second, fraction, zone = (_group(found, at) for at in range(6))
    hours = _or(hour, "00")
    base = pc.binary_join_element_wise(date, "T", hours, ":", _or(minute, "00"), ":", _or(second, "00"), "")
    wall = pc.strptime(base, format=_STORED_NAIVE, unit="us", error_is_null=True)
    # strptime carries an hour or a day past its range into the next (Feb 30 is Mar 2): such a text is no instant
    real = pc.equal(pc.utf8_slice_codeunits(pc.strftime(wall, format=_STORED_NAIVE), 0, 19), base)
    micros = pc.cast(pc.utf8_slice_codeunits(pc.utf8_rpad(fraction, 6, "0"), 0, 6), pa.int64())
    offset = pc.extract_regex(pc.if_else(pc.equal(zone, "Z"), "+00:00", zone), pattern=_OFFSET)
    sign, off_hours, off_minutes = (_group(offset, at) for at in range(3))
    off_h = pc.cast(_or(off_hours, "00"), pa.int64())
    off_m = pc.cast(_or(off_minutes, "00"), pa.int64())
    shift = pc.multiply(pc.add(pc.multiply(off_h, 3600), pc.multiply(off_m, 60)), _SECOND)
    shift = pc.if_else(pc.equal(sign, "-"), pc.negate(shift), shift)
    at = pc.add(pc.cast(wall, pa.int64()), micros)
    zoned = pc.not_equal(zone, "")
    real = pc.and_kleene(real, pc.or_kleene(pc.invert(zoned), pc.and_kleene(pc.less(off_h, 24), pc.less(off_m, 60))))
    # a date alone takes no offset (``fromisoformat`` refuses "2026-11-03+00:00")
    real = pc.and_kleene(real, pc.or_kleene(pc.invert(zoned), pc.not_equal(hour, "")))
    utc = pc.cast(pc.subtract(at, pc.fill_null(shift, 0)), pa.timestamp("us", tz="UTC"))
    spelled = pc.if_else(
        zoned,
        pc.strftime(utc, format=_STORED_ZONED),
        pc.strftime(pc.cast(at, pa.timestamp("us")), format=_STORED_NAIVE),
    )
    stored: pa.Array = pc.if_else(pc.fill_null(real, False), spelled, text)
    return stored


#: the rewrites that spell a NUMERIC's text one way, in order (RE2, Arrow's ``replace_substring_regex``): a leading
#: ``+`` dropped, a fraction's trailing zeros and then a bare point dropped, the whole part's leading zeros dropped, a
#: fraction with no whole part given its ``0``, and a negative zero made zero
_NUMERIC_REWRITES: Final = (
    (r"^\+", ""),
    (r"^(-?[0-9]*\.[0-9]*?)0+$", r"\1"),
    (r"^(-?[0-9]*)\.$", r"\1"),
    (r"^(-?)0+([0-9])", r"\1\2"),
    # RE2 names a group by one digit: ``\10.`` is group 1, then ``0.``
    (r"^(-?)\.", r"\10."),
    (r"^-0$", "0"),
)


def _numeric_texts(values: pa.ChunkedArray | pa.Array) -> pa.Array:
    """a column of NUMERIC values as text, each in one spelling of its value (``1.50``, ``01.5`` and ``+1.5`` are all
    ``1.5``; ``-0.0`` is ``0``), so two texts of one value are equal. In Arrow's kernels alone, never a Python object a
    value.

    **Why.** The rail answers a ``NUMERIC`` as its digits at the column's scale (``1.50``), and a change record names a
    key as ``key_text`` spells the value it was given (``1.5``): matched as text, the key never found its row. The
    copy's read by key compared them in SQL cast to ``numeric`` (``FollowedTable.held_key``); a keyed lookup in
    Arrow's hash kernels (``collections.copy_queries``) compares them so spelled. A text in exponent form (``1E+2``)
    or one that is no number is kept as it is (trimmed): neither is a ``NUMERIC``'s text the rail or L3 writes.

    :param values: the column, as text (or any type cast to text)
    :ptype values: pa.ChunkedArray | pa.Array
    :return: the column, each value in its one spelling; a null stays null
    :rtype: pa.Array
    """
    column = values.combine_chunks() if isinstance(values, pa.ChunkedArray) else values
    text: pa.Array = pc.utf8_trim_whitespace(pc.cast(column, pa.string()))
    for pattern, replacement in _NUMERIC_REWRITES:
        text = pc.replace_substring_regex(text, pattern=pattern, replacement=replacement)
    return text


#: the SQL types (``FollowedTable.key_cast``) of a key column held as text whose values have one spelling the copy keeps
#: (:func:`canonical_texts`): a ``NUMERIC``'s (and an integer's held as text) by :func:`_numeric_texts`, a UUID's lower
#: case. An instant's has its own (:func:`stored_instants`, the copy's ``instants``)
CANONICAL_CASTS: Final = frozenset({"numeric", "bigint", "uuid"})


def canonical_texts(values: pa.ChunkedArray | pa.Array, cast: str) -> pa.ChunkedArray | pa.Array:
    """a key column held as text in its type's one spelling: a ``NUMERIC``'s (or an integer's) by
    :func:`_numeric_texts` (``1.50`` is ``1.5``), a UUID's lower-cased; a column of any other type as it is. In Arrow's
    kernels alone, never a Python object a value.

    **Why** (3tears' copy, 2026-11-25). A lookup of the copy by key compared a NUMERIC or UUID held as text by value,
    respelling the held column across the whole part on every lookup. The copy now stores this spelling once, as each
    change is applied (``shared_copy.SharedCopyBackend``'s ``canonical``), and a lookup spells only its asked keys so.

    :param values: the column
    :ptype values: pa.ChunkedArray | pa.Array
    :param cast: the column's SQL type
    :ptype cast: str
    :return: the column in its one spelling; a null stays null
    :rtype: pa.ChunkedArray | pa.Array
    """
    if cast in ("numeric", "bigint"):
        return _numeric_texts(values)
    if cast == "uuid":
        lowered: pa.ChunkedArray | pa.Array = pc.utf8_lower(pc.cast(values, pa.string()))
        return lowered
    return values


def _compared(
    column: pa.ChunkedArray | pa.Array, cast: str | None, held: pa.DataType, *, asked: bool
) -> pa.ChunkedArray | pa.Array:
    """a key column as a lookup compares it: an asked key's or a held row's.

    A column held typed (an integer, a decimal, a boolean) is compared as it is held, the asked keys cast to its type
    (:func:`rows_named`, Arrow's cast). A column held as text whose type is not text is compared by value, as
    the copy's read by key compared it in SQL cast to the column's type (``FollowedTable.held_key``): the asked keys in
    the one spelling the copy stores (:func:`canonical_texts`: a NUMERIC's ``1.50`` is ``1.5``, a UUID lower-cased;
    an instant's :func:`stored_instants`), and the held column as it is, never respelled.

    **Why only the asked side** (2026-11-25). Respelled on both sides, every lookup rewrote the held column across the
    whole part it probed, a cost of the part's rows, not the keys'. The copy now stores that spelling once, as each
    change is applied (``collections.shared_copy.SharedCopyBackend``'s ``canonical`` and ``instants``), so a
    lookup's cost is its keys'. In Arrow's kernels alone, never a Python object a key.

    :param column: the column
    :ptype column: pa.ChunkedArray | pa.Array
    :param cast: the column's SQL type; None compares it as it is
    :ptype cast: str | None
    :param held: the copy's Arrow type of the column
    :ptype held: pa.DataType
    :param asked: whether the column is the asked keys' (else the copy's rows')
    :ptype asked: bool
    :return: the column as compared
    :rtype: pa.ChunkedArray | pa.Array
    """
    if not asked or cast is None or not (pa.types.is_string(held) or pa.types.is_large_string(held)):
        return column
    if cast in CANONICAL_CASTS:
        return canonical_texts(column, cast)
    if cast == "timestamptz":
        return stored_instants(column)
    return column


def rows_found(rows: pa.Table, keys: pa.Table, key: Sequence[str], casts: Mapping[str, str] = EMPTY) -> pa.Table:
    """the rows of ``rows`` whose key ``keys`` names, in their order in ``rows`` (Arrow's hash kernels, no Python
    object a key), each key column compared as :func:`_compared` compares it.

    :param rows: the rows looked in
    :ptype rows: pa.Table
    :param keys: the keys looked for
    :ptype keys: pa.Table
    :param key: the key's columns
    :ptype key: Sequence[str]
    :param casts: each key column's SQL type, for a column held as text compared by value; others as they are
    :ptype casts: Mapping[str, str]
    :return: the rows found, as the copy holds them
    :rtype: pa.Table
    """
    held = rows.select(list(key))
    asked = keys.select(list(key))
    for at, column in enumerate(key):
        kind = held.schema.field(column).type
        held = held.set_column(at, column, _compared(held.column(column), casts.get(column), kind, asked=False))
        asked = asked.set_column(at, column, _compared(asked.column(column), casts.get(column), kind, asked=True))
    return rows.take(pa.array(rows_named(held, asked, key)))
