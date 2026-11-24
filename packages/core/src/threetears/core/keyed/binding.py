"""a key batch's columns bound in Arrow: each key column ONE text, written in Arrow's kernels.

A :class:`~threetears.core.keyed.KeyBatch` holds one array a key column. Wherever pyarrow imports, the keyed helpers
bind it here (Arrow mode, Pace 2026-11-23) when a column arrives as an Arrow array (a plain column beside it is made
one by Arrow's own conversion), and every column's distinct keys, its chunks (zero-copy slices) and its bind are Arrow's work, never a Python object a
key. On PostgreSQL and YugabyteDB (asyncpg, the rail) a column is ONE text, its values' texts joined by a code point
none holds and split back in SQL (``string_to_array($1::text, chr(44))::uuid[]``, :class:`SplitText`); a reader naming
DuckDB gets the array's literal in DuckDB's list form; an in-process DuckDB gets the Arrow array of its texts itself,
which :func:`~threetears.core.cache.duckdb_binds.run_bound` sets as a DuckDB variable. A batch of plain columns (a small key
list program logic holds), or any batch where pyarrow does not import (the SDK's agents, the epoch follow, 3tears'
import-without-Arrow gate), binds through :data:`~threetears.interfaces.keyed.rows.PLAIN_KEYS`, a list a column
(:func:`binder_for`).

A batch bound as a list of placeholders (``as_list``: Redshift, SQLite) is the one exception: its driver binds a
placeholder a value, so its values are made Python objects at the bind, a statement's chunk at a time.

WHY one text (measured 2026-11-23, 50,000 keys on PostgreSQL 16 and YugabyteDB 2026.1; ``test_keyed_binding_live``):
asyncpg encodes a list a value at a time and the rail carries it as JSON a value at a time; one text is one bind, and
the serving process spends a third to a half of the CPU. WHY split, not an array literal: ``string_to_array`` is
immutable, so a custom plan holds the array as a constant and plans as it does a bound list; a literal's parse
(``array_in``) is stable, never folded, and is evaluated again at each estimate.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

from threetears.core.keyed.inline import InlineDuckDB
from threetears.core.keyed.threaded import ThreadedDuckDB
from threetears.interfaces.keyed.counts import SQL_DIALECT_ATTRIBUTE, Dialect
from threetears.interfaces.keyed.rows import PLAIN_KEYS, BindKind, KeyBatch, KeyBinder, KeyChunk, SplitText

try:
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc

    from threetears.core.arrow_keys import distinct_keys

    _ARROW = True
except ImportError:  # NOSILENT: the arrow extra is optional; with no pyarrow a batch binds its plain lists
    _ARROW = False

__all__ = ["binder_for"]

#: each byte's two lowercase hex digits as one 16-bit cell (the two ASCII bytes in memory order): a UUID's 16 bytes
#: written as its 32 digits in one NumPy take of 16 cells (2,000 UUIDs: 25 us, against 157 taking 32 single bytes)
_HEX: Final[Any] = (
    np.frombuffer(b"".join(f"{byte:02x}".encode() for byte in range(256)), dtype=np.uint16) if _ARROW else None
)


def _as_arrow(values: Sequence[Any]) -> Any:
    """``values`` as one Arrow array: an Arrow array as it is (a chunked one combined), a plain sequence converted by
    Arrow's own conversion (its UUIDs as Arrow's UUID type)."""
    if isinstance(values, pa.ChunkedArray):
        return values.combine_chunks()
    if isinstance(values, pa.Array):
        return values
    return pa.array(values)


def _unique(values: Any) -> Any:
    """an Arrow array's distinct values, in the order first named: an extension array (a UUID) by its storage, made
    its own type again."""
    if isinstance(values, pa.ExtensionArray):
        return pa.ExtensionArray.from_storage(values.type, values.storage.unique())
    return values.unique()


#: where a UUID's 32 hex digits sit in its 36-character canonical text (``8-4-4-4-12``), and where its hyphens do
_UUID_DIGITS: Final = ((0, 8, 0), (9, 13, 8), (14, 18, 12), (19, 23, 16), (24, 36, 20))
_UUID_HYPHENS: Final = (8, 13, 18, 23)


def _uuid_texts(storage: Any) -> Any:
    """a UUID array's storage (16 bytes a value) as its canonical texts, ``8-4-4-4-12`` lowercase hex, as PostgreSQL
    and Python write a UUID: its digits taken from :data:`_HEX` in one NumPy take and laid out with their hyphens, the
    texts one Arrow string array over that buffer; no value a Python object."""
    count = len(storage)
    raw = np.frombuffer(storage.buffers()[1], dtype=np.uint8)[storage.offset * 16 : (storage.offset + count) * 16]
    digits = _HEX[raw].view(np.uint8).reshape(count, 32)
    text = np.empty((count, 36), dtype=np.uint8)
    text[:, _UUID_HYPHENS] = ord("-")
    for start, stop, at in _UUID_DIGITS:
        text[:, start:stop] = digits[:, at : at + stop - start]
    offsets = np.arange(0, (count + 1) * 36, 36, dtype=np.int32)
    texts = pa.StringArray.from_buffers(count, pa.py_buffer(offsets), pa.py_buffer(text))
    return pc.if_else(storage.is_null(), pa.scalar(None, pa.string()), texts) if storage.null_count else texts


def _texts(values: Any) -> tuple[Any, bool]:
    """an Arrow array's values as the texts its column's type reads back (a UUID in its canonical form, a date in ISO
    form, a number as written), and whether a text may need quoting in an array literal (a text column's, a time's
    space): a UUID's, a number's or a date's never holds a quote, a backslash, a comma, a space or a brace."""
    kind = values.type
    if isinstance(kind, pa.BaseExtensionType) and kind.extension_name == "arrow.uuid":
        return _uuid_texts(values.storage), False
    if pa.types.is_string(kind) or pa.types.is_large_string(kind):
        return values, True
    bare = (
        pa.types.is_integer(kind) or pa.types.is_floating(kind) or pa.types.is_decimal(kind) or pa.types.is_date(kind)
    )
    return values.cast(pa.string()), not bare


def _arrow_literal(values: Any, brackets: bool = False) -> str:
    """an Arrow array as ONE array literal its database parses: a null as ``NULL``, a text quoted (``"`` and ``\\``
    escaped, where any holds one), a UUID, a number or a date bare, in ``{}`` (PostgreSQL, YugabyteDB) or ``[]``
    (DuckDB behind a reader: its list form). Written in Arrow's kernels: no value of it is a Python object.

    WHY bare where it can be (measured 2026-11-23, 2,000 UUIDs): quoting and escaping each text is two thirds of the
    literal's cost (367 us against 120), and a UUID's text holds nothing to quote.

    :param values: the values
    :ptype values: pyarrow.Array
    :param brackets: DuckDB's list form, ``[...]``
    :ptype brackets: bool
    :return: the literal
    :rtype: str
    """
    texts, quoting = _texts(values)
    if quoting:
        if pc.any(pc.match_substring_regex(texts, r'[\\"]')).as_py():
            texts = pc.replace_substring(pc.replace_substring(texts, "\\", "\\\\"), '"', '\\"')
        texts = pc.binary_join_element_wise('"', texts, '"', "")
    if texts.null_count:
        texts = pc.fill_null(texts, "NULL")
    joined = pc.binary_join(pa.ListArray.from_arrays(pa.array([0, len(texts)], pa.int32()), texts), ",")
    # enclosed in Arrow too: the literal becomes a Python str once, never copied again
    opened, closed = ("[", "]") if brackets else ("{", "}")
    return str(pc.binary_join_element_wise(opened, joined, closed, "")[0].as_py())


#: the code points a text column's values may be joined by, tried in order: control characters no key text holds in
#: practice (the unit, record, group and file separators). A bare column (a UUID, a number, a date, a time) holds no
#: comma, so it is joined by one
_DELIMITERS: Final = (31, 30, 29, 28, 27, 26, 25, 24)
_COMMA: Final = 44


def _join(texts: Any, delimiter: int) -> str:
    """the texts as ONE text, ``delimiter`` between two: joined in Arrow's kernels, made a Python str once."""
    joined = pc.binary_join(pa.ListArray.from_arrays(pa.array([0, len(texts)], pa.int32()), texts), chr(delimiter))
    return str(joined[0].as_py())


def _split_text(values: Any) -> tuple[str, SplitText] | None:
    """an Arrow array as ONE text its values' texts joined by a code point none holds (:class:`SplitText`), a null
    written as another; None when no code point fits (a text column holding every candidate), or the one value is
    the empty text (``string_to_array('', d)`` names no value): bound as the array's literal instead."""
    texts, quoting = _texts(values)
    needed = 2 if texts.null_count else 1
    if not quoting or pa.types.is_timestamp(values.type) or pa.types.is_time(values.type):
        # a bare text (a UUID, a number, a date; a time's holds a space, never a comma or a control character)
        free: list[int] = [_COMMA, *_DELIMITERS]
    else:
        # each candidate looked for in one pass of Arrow's kernel, only until enough are found (one; two with nulls)
        free = []
        for code in _DELIMITERS:
            if not pc.any(pc.match_substring(texts, chr(code))).as_py():
                free.append(code)
                if len(free) == needed:
                    break
    if len(free) < needed or (len(texts) == 1 and texts[0].as_py() == ""):
        return None
    null = free[1] if texts.null_count else None
    if null is not None:
        texts = pc.fill_null(texts, chr(null))
    return _join(texts, free[0]), SplitText(free[0], null)


class _ArrowKeys:
    """:class:`~threetears.interfaces.keyed.rows.KeyBinder` of Arrow columns: distinct keys by Arrow's hash
    kernels, chunks as zero-copy slices, each column bound as ONE text split back in SQL (:class:`SplitText`; the
    array's literal where none fits, and for DuckDB behind a reader), or, on an in-process DuckDB, its texts' Arrow
    array."""

    def __init__(self, duckdb: bool, brackets: bool) -> None:
        self._duckdb = duckdb
        self._brackets = brackets

    def distinct(self, keys: KeyBatch) -> KeyChunk:
        """the batch's distinct keys, in the order first named, one Arrow array a column."""
        columns = [_as_arrow(column) for column in keys.keys]
        if len(columns) == 1:
            return (_unique(columns[0]),)
        names = [f"k{index}" for index in range(len(columns))]
        # by Arrow's hash kernels, never Acero's group_by (:mod:`threetears.core.arrow_keys`): the keys come out in
        # the order first named. An extension column (a UUID) is coded by its storage and made its own type again
        stored = [column.storage if isinstance(column, pa.ExtensionArray) else column for column in columns]
        grouped = distinct_keys(pa.table(stored, names=names), names)
        return tuple(
            pa.ExtensionArray.from_storage(column.type, grouped.column(name).combine_chunks())
            if isinstance(column, pa.ExtensionArray)
            else grouped.column(name).combine_chunks()
            for name, column in zip(names, columns, strict=True)
        )

    def array(self, values: Sequence[Any], cast: str, natively: bool) -> tuple[Any, BindKind]:
        """one key column as ONE bind: its split text, its array's literal, or (in-process DuckDB) its texts."""
        del cast, natively
        arrow = _as_arrow(values)
        if self._duckdb:
            return _texts(arrow)[0], "text"
        if not self._brackets and (split := _split_text(arrow)) is not None:
            return split
        return _arrow_literal(arrow, self._brackets), "literal"

    def distinct_values(self, values: Sequence[Any]) -> Sequence[Any]:
        """one key column's distinct values, an Arrow array."""
        unique: Sequence[Any] = _unique(_as_arrow(values))
        return unique

    def listed(self, values: Sequence[Any]) -> list[Any]:
        """one key column's values as Python values, a placeholder each (``as_list`` only)."""
        # as_list only: its driver binds a placeholder a value (module docstring); a plain column's values as given
        if isinstance(values, pa.Array | pa.ChunkedArray):
            return list(values.to_pylist())
        return list(values)

    def null_groups(self, chunk: KeyChunk) -> list[tuple[int, KeyChunk]]:
        """the chunk's keys grouped by which parts are NULL, each group its pattern's code and its Arrow columns."""
        # each key's NULL pattern one int (null_pattern_code: bit i for part i) summed in Arrow's kernels, each
        # pattern's keys a filter of the columns: never a Python object a key
        columns = [_as_arrow(column) for column in chunk]
        if not any(column.null_count for column in columns):
            return [(0, tuple(columns))]
        codes: Any = None
        for index, column in enumerate(columns):
            if not column.null_count:
                continue
            bit = pc.multiply(pc.cast(pc.is_null(column), pa.int64()), pa.scalar(1 << index, pa.int64()))
            codes = bit if codes is None else pc.add(codes, bit)
        # the distinct codes: one a NULL pattern the chunk holds, a few small ints
        return [
            (int(code), tuple(column.filter(pc.equal(codes, pa.scalar(code, pa.int64()))) for column in columns))
            for code in codes.unique().to_pylist()
        ]


#: the Arrow binders: the literal's (PostgreSQL, YugabyteDB), an in-process DuckDB's (Arrow arrays), and a reader's
#: naming DuckDB (its JSON carries the literal in DuckDB's list form)
_LITERAL: Final[KeyBinder | None] = _ArrowKeys(duckdb=False, brackets=False) if _ARROW else None
_IN_PROCESS_DUCKDB: Final[KeyBinder | None] = _ArrowKeys(duckdb=True, brackets=False) if _ARROW else None
_DUCKDB_READER: Final[KeyBinder | None] = _ArrowKeys(duckdb=False, brackets=True) if _ARROW else None


def _arrives_as_arrow(keys: KeyBatch | None) -> bool:
    """whether any of a batch's key columns arrived as an Arrow array (a serving path's keys)."""
    return keys is not None and any(isinstance(column, pa.Array | pa.ChunkedArray) for column in keys.keys)


def binder_for(conn: Any, keys: KeyBatch | None) -> KeyBinder:
    """how a keyed helper on ``conn`` binds ``keys``' columns: in Arrow when a column arrived as an Arrow array (an
    in-process DuckDB's as Arrow arrays, run_bound's; a reader naming DuckDB's in its list form; else one text a
    column, split back in SQL), a plain column beside it made Arrow too; plain lists when every column is a plain
    sequence, or where pyarrow does not import.

    WHY plain columns keep their lists where pyarrow imports (Pace 2026-11-23, option A): a plain column is a small
    key list program logic already holds as Python values; made Arrow and sent as one text it read 2,000 UUIDs at
    1.27-1.34x the hand-written native bind on PostgreSQL 16 (1.09-1.12x on YugabyteDB), as a list at 1.05x, as the
    tuples did. Arrow mode is for keys that arrive as Arrow, where it spends a third to a half of the CPU.

    :param conn: the connection or reader the helper was handed
    :ptype conn: Any
    :param keys: the batch bound; None when the helper names no batch (a write by ``by`` alone)
    :ptype keys: KeyBatch | None
    :return: the binder
    :rtype: KeyBinder
    """
    if not _ARROW or _LITERAL is None or _IN_PROCESS_DUCKDB is None or _DUCKDB_READER is None:
        return PLAIN_KEYS
    if not _arrives_as_arrow(keys):
        return PLAIN_KEYS
    if isinstance(conn, InlineDuckDB | ThreadedDuckDB):
        return _IN_PROCESS_DUCKDB
    if getattr(conn, SQL_DIALECT_ATTRIBUTE, None) is Dialect.DUCKDB:
        return _DUCKDB_READER
    return _LITERAL
