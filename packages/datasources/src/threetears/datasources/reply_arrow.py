"""A datasource read's rows built straight into the Arrow table its reply carries, from the driver's own values.

**Why.** The hub answered a read asked as Arrow by making every row a dict, every dict JSON-safe and JSON, and then
parsing that JSON back into Arrow: a Python object and a Python call a row, on the serving process (77 loop stalls,
9.3 s in 14 minutes of replay 9, the hub's largest). Here the driver's values become Arrow a column at a time, in C:
the rows are turned into columns in one pass (``zip(*rows)``), and each column is one ``pyarrow.array`` call, or one
encode of the whole column with pydantic's JSON encoder and one parse by Arrow's JSON reader. No row is ever a dict,
and no value passes through Python code of ours.

**The same table the JSON reply gave.** The pod reads a column as the type its JSON value takes (text for text, a
``Decimal``, a date, a time or a timestamp; ``int64`` for an integer; ``float64`` for a float; ``bool`` for a boolean:
:mod:`threetears.datasources.arrow_rows`), and a value of another form was refused
(:class:`~threetears.core.backends.arrow_reply.ReplyNotArrowError`). So each column is built one of two ways, both
giving exactly what Arrow's JSON reader made of pydantic's JSON of the value:

- a column Arrow holds as it is (its values all ``str``, all ``int``, all ``float``, all ``bool`` or all ``None``,
  which ``pyarrow.array`` infers in C) whose type is the one asked for, or an integer column asked as ``float64``, is
  that array;
- any other column (a ``Decimal``, a date, a time, a timestamp, a UUID, or a type mismatch) is encoded whole with
  :func:`pydantic_core.to_json` (the encoder the JSON reply was made by) and read by Arrow's JSON reader as the type
  asked for (the reader the pod parsed the JSON reply with), so its text is pydantic's and its refusals the reader's;
- ``bytes`` are bare hex, as the JSON reply carried them (``json_safe_row``), made in one call over the whole column.

A column the read did not select is all null, as a JSON row without the key was read; a selected column not asked
for is left out, as the JSON reader ignored it.

**The byte cap.** A reply cut by its bytes keeps the rows whose Arrow buffers fit the cap (:class:`ArrowBudget`),
measured as the table's ``nbytes`` (the bytes its IPC stream's body takes, within its padding), a page at a time,
cut inside the page that crosses it by bisection on the page's own slices, never a row at a time.

Needs ``pyarrow`` (the ``export`` extra).
"""

from __future__ import annotations

import functools
import io
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import pydantic_core

from threetears.core.backends.arrow_reply import ReplyNotArrowError
from threetears.core.frozen_state import with_chunk

# pyarrow ships no py.typed marker; ignored here rather than in the root pyproject's per-module list (which this
# change leaves alone), with the same effect
try:
    import pyarrow as pa  # type: ignore[import-untyped]
    import pyarrow.compute as pc  # type: ignore[import-untyped]
    import pyarrow.json as pa_json  # type: ignore[import-untyped]

    _PYARROW = True
except ImportError:  # NOSILENT: the export extra is optional; an Arrow read says so without it
    _PYARROW = False

__all__ = [
    "ArrowBudget",
    "BoundedArrow",
    "ReplyNotArrowError",
    "records_table",
    "rows_table",
    "start_arrow_budget",
]

#: the share of the rows the bytes left are expected to hold that a page sized to fit whole asks for
#: (:class:`~threetears.datasources.bounded_rows.RowBudget`'s, for the same reason: rows widen as a result goes on)
_FIT_SHARE: Final[float] = 0.95


def _require_pyarrow() -> None:
    """refuse without pyarrow, saying how to get it.

    :return: nothing
    :rtype: None
    :raises ImportError: without pyarrow
    """
    if not _PYARROW:
        raise ImportError("a datasource read as Arrow needs pyarrow: install 3tears-datasources[export]")


def _is_text(kind: Any) -> bool:
    """whether an Arrow type is text.

    :param kind: the type
    :ptype kind: pyarrow.DataType
    :return: whether it is ``string`` or ``large_string``
    :rtype: bool
    """
    return bool(pa.types.is_string(kind) or pa.types.is_large_string(kind))


def _refusal(name: str, kind: Any) -> ReplyNotArrowError:
    """the refusal of a column whose values are not of the type asked for. It names the column and the type, never a
    value: the reply that carries it goes on a shared bus.

    :param name: the column
    :ptype name: str
    :param kind: the type asked for
    :ptype kind: pyarrow.DataType
    :return: the error
    :rtype: ReplyNotArrowError
    """
    return ReplyNotArrowError(f"column {name!r}: its values are not of the type asked for ({kind})")


def _hex(values: Any) -> Any:
    """a binary column as the bare hex text the JSON reply carried (``bytes.hex``), in one call over its bytes.

    :param values: the column, ``binary``
    :ptype values: pyarrow.BinaryArray
    :return: the column as ``large_string``, nulls kept
    :rtype: pyarrow.LargeStringArray
    """
    binary = values.cast(pa.large_binary())
    # the offsets as an array of their own, over the binary's buffer from where its slice starts
    offsets = pa.Array.from_buffers(pa.int64(), len(binary) + 1, [None, binary.buffers()[1]], offset=binary.offset)
    start = offsets[0].as_py()
    end = offsets[-1].as_py()
    digits = (
        pa.py_buffer(binary.buffers()[2].slice(start, end - start).to_pybytes().hex().encode("ascii"))
        if end > start
        else None
    )
    doubled = pc.multiply(pc.subtract(offsets, start), 2)
    validity = binary.buffers()[0] if binary.null_count else None
    if validity is not None and binary.offset:
        # the bitmap of a slice starts mid-byte: rebuilt from the slice, so the new array starts at 0
        validity = pc.is_valid(binary).buffers()[1]
    return pa.Array.from_buffers(
        pa.large_string(),
        len(binary),
        [validity, doubled.buffers()[1], digits if digits is not None else pa.py_buffer(b"")],
        null_count=binary.null_count,
    )


def _via_json(values: Sequence[Any], name: str, kind: Any) -> Any:
    """a column as Arrow's JSON reader reads pydantic's JSON of it: the JSON reply's own two steps, over the column.

    :param values: the column's values
    :ptype values: Sequence[Any]
    :param name: the column's name, for a refusal
    :ptype name: str
    :param kind: the type asked for
    :ptype kind: pyarrow.DataType
    :return: the column
    :rtype: pyarrow.Array
    :raises ReplyNotArrowError: when the reader refuses a value as ``kind``
    """
    data = b"".join((b'{"v":', pydantic_core.to_json(values), b"}"))
    try:
        parsed = pa_json.read_json(
            io.BytesIO(data),
            # one line, one block, the caller's thread
            read_options=pa_json.ReadOptions(use_threads=False, block_size=len(data) + 1),
            parse_options=pa_json.ParseOptions(explicit_schema=pa.schema([pa.field("v", pa.list_(kind))])),
        )
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
        raise _refusal(name, kind) from exc
    return parsed.column("v").combine_chunks().flatten()


#: the Python types of the values a column is built from directly, by Arrow's own inference (in C): what a driver
#: answers for text, an integer, a float and a boolean. A column of any other values (a ``Decimal``, a date, a time, a
#: timestamp, a UUID) is never handed to Arrow's inference, which calls back into Python a value at a time for them,
#: and goes the JSON reply's two steps instead (:func:`_via_json`), which encode and parse it without any
_INFERRED: Final = (str, int, float, bool)


def _first(values: Sequence[Any]) -> Any:
    """a column's first value that is not None (None when all are), found in C: a statement's column is of one
    warehouse type, which the driver answers as one Python type, so the first says which way the column is built.

    :param values: the column's values
    :ptype values: Sequence[Any]
    :return: the value
    :rtype: Any
    """
    return next(filter(functools.partial(operator.is_not, None), values), None)


def _column(values: Sequence[Any], name: str, kind: Any) -> Any:
    """one column of the reply's table, typed ``kind``, from the driver's values.

    :param values: the column's values, as the driver answered them
    :ptype values: Sequence[Any]
    :param name: the column's name
    :ptype name: str
    :param kind: the type asked for
    :ptype kind: pyarrow.DataType
    :return: the column
    :rtype: pyarrow.Array
    :raises ReplyNotArrowError: when its values are not of ``kind``
    """
    first = _first(values)
    column: Any = None
    if first is None:
        column = pa.nulls(len(values), kind)
    elif isinstance(first, bytes | bytearray):
        # bare hex, as json_safe_row wrote it; text in any other column is refused, as the reader refused it
        if not _is_text(kind):
            raise _refusal(name, kind)
        try:
            column = _hex(pa.array(values, pa.binary())).cast(kind)
        except (pa.ArrowException, TypeError, ValueError, OverflowError) as exc:
            # a column of bytes and something else, which no JSON reply of it read as text either
            raise _refusal(name, kind) from exc
    elif isinstance(first, _INFERRED):
        try:
            # Arrow infers the column's type from every value (a bool among ints, a float among them, says so)
            found = pa.array(values)
        except pa.ArrowException, TypeError, ValueError, OverflowError:
            # NOSILENT: values Arrow does not infer one type of: the JSON reply's two steps decide
            found = None
        if found is not None and (found.type.equals(kind) or (_is_text(found.type) and _is_text(kind))):
            column = found.cast(kind)
        elif found is not None and pa.types.is_floating(kind) and pa.types.is_integer(found.type):
            # an integer read as a float, as the reader reads a JSON integer for a float column
            column = pc.cast(found, kind, safe=False)
    if column is None:
        column = _via_json(values, name, kind)
    return column


def _columns_table(names: Sequence[str], columns: Sequence[Sequence[Any]], schema: Any) -> Any:
    """a read's columns as the table the JSON reply of the same rows read as, typed by ``schema``.

    Blocking, and CPU for as many values as the read has (in C, holding the interpreter lock where Arrow reads
    Python objects): on the thread the driver read the rows on, never the event loop.

    :param names: the columns' names, as the statement selects them (a name selected twice: the last, as a row's dict
        kept it)
    :ptype names: Sequence[str]
    :param columns: each column's values, in ``names``' order, every column as long as the others
    :ptype columns: Sequence[Sequence[Any]]
    :param schema: each column asked for and its Arrow type, in order
    :ptype schema: pyarrow.Schema
    :return: the rows
    :rtype: pyarrow.Table
    :raises ReplyNotArrowError: when a column's values are not of its type
    :raises ImportError: without pyarrow
    """
    _require_pyarrow()
    count = len(columns[0]) if columns else 0
    where = {name: index for index, name in enumerate(names)}
    arrays = []
    for field in schema:
        index = where.get(field.name)
        arrays.append(pa.nulls(count, field.type) if index is None else _column(columns[index], field.name, field.type))
    return pa.Table.from_arrays(arrays, schema=schema)


def rows_table(names: Sequence[str], rows: Sequence[Sequence[Any]], schema: Any) -> Any:
    """a read's rows (each a sequence of values, as a DB-API cursor answers them) as :func:`_columns_table`'s table.

    The rows become columns in one pass, in C (``zip(*rows)``), never a row at a time in Python.

    :param names: the columns' names, as the cursor describes them
    :ptype names: Sequence[str]
    :param rows: the rows, each its values in ``names``' order
    :ptype rows: Sequence[Sequence[Any]]
    :param schema: each column asked for and its Arrow type, in order
    :ptype schema: pyarrow.Schema
    :return: the rows
    :rtype: pyarrow.Table
    :raises ReplyNotArrowError: when a column's values are not of its type
    :raises ImportError: without pyarrow
    """
    columns: list[Sequence[Any]] = list(zip(*rows, strict=True)) if rows else [() for _ in names]
    return _columns_table(names, columns, schema)


def records_table(records: Sequence[Any], schema: Any) -> Any:
    """a read's records (asyncpg's ``Record``, or mappings) as :func:`_columns_table`'s table.

    :param records: the records, each with ``keys()`` and its values in that order
    :ptype records: Sequence[Any]
    :param schema: each column asked for and its Arrow type, in order
    :ptype schema: pyarrow.Schema
    :return: the rows
    :rtype: pyarrow.Table
    :raises ReplyNotArrowError: when a column's values are not of its type
    :raises ImportError: without pyarrow
    """
    names: list[str] = list(records[0].keys()) if records else list(schema.names)
    # a mapping iterates its keys, a Record its values: a mapping's values are taken whole, a row at a time (a
    # driver that answers mappings is not one the serving path reads through), a Record's in C
    values = records if not records or not isinstance(records[0], Mapping) else [record.values() for record in records]
    return rows_table(names, values, schema)


@dataclass(frozen=True, slots=True)
class BoundedArrow:
    """the rows a capped Arrow read kept.

    :ivar table: the rows kept, in the statement's order, typed as asked (in the chunks they were read in)
    :ivar truncated: whether the result had rows past the ones kept
    :ivar size: the bytes the kept rows take (the table's ``nbytes``), what the byte cap measured
    """

    table: Any
    truncated: bool
    size: int


class ArrowBudget:
    """the cut a byte-capped Arrow read makes: pages kept while their Arrow buffers fit ``max_bytes``.

    The Arrow counterpart of :class:`~threetears.datasources.bounded_rows.RowBudget`, sized the same way: the first
    page reads ``page_rows``; after it each page asks for the rows the bytes left are expected to hold, from the mean
    row of the last page kept. A page that fits is kept whole; the page that crosses is cut at the longest prefix that
    fits, found by bisecting its slices' ``nbytes`` (each a sum over the columns' buffers, in C). The first row is kept
    whatever its size, so a page always moves its reader on. ``max_rows`` bounds the rows kept: a row read past it
    ends the read, truncated.

    :param schema: each column asked for and its Arrow type
    :ptype schema: pyarrow.Schema
    :param max_bytes: the most bytes the kept rows may take
    :ptype max_bytes: int
    :param max_rows: the most rows kept; None for no row bound
    :ptype max_rows: int | None
    """

    __slots__ = ("_count", "_max_bytes", "_max_rows", "_page_mean", "_schema", "_size", "_tables", "_truncated")

    def __init__(self, schema: Any, max_bytes: int, max_rows: int | None) -> None:
        self._schema = schema
        self._max_bytes = max_bytes
        self._max_rows = max_rows
        # the pages kept: an append-only chunk log replaced whole (frozen_state.with_chunk), never changed in place,
        # so the awaiting coroutine never sees it change under it while a page is taken on a thread
        self._tables: tuple[tuple[Any, ...], ...] = ()
        self._count = 0
        self._size = 0
        self._truncated = False
        self._page_mean = 0.0

    @property
    def schema(self) -> Any:
        """the types the read is built in.

        :return: the schema
        :rtype: pyarrow.Schema
        """
        return self._schema

    def wanted(self, page_rows: int) -> int:
        """how many rows the next page should read (:meth:`RowBudget.wanted`'s sizing, by Arrow bytes).

        :param page_rows: the rows the first page reads, and about the rows the crossing page reads
        :ptype page_rows: int
        :return: the rows to ask for, at least 1
        :rtype: int
        """
        wanted = page_rows
        if self._count and self._page_mean > 0:
            expected = int(max(0, self._max_bytes - self._size) / self._page_mean)
            wanted = int(expected * _FIT_SHARE) - page_rows // 2 if expected > page_rows else expected + 1
        if self._max_rows is not None:
            wanted = min(wanted, self._max_rows + 1 - self._count)
        return max(1, wanted)

    def take(self, page: Any) -> bool:
        """keep the page's rows while they fit; say whether the read should go on.

        :param page: the next rows of the result, in order, as :func:`_columns_table` built them
        :ptype page: pyarrow.Table
        :return: True while every row so far was kept (a further page may be read), False once one was not
        :rtype: bool
        """
        rows = page
        if self._truncated:
            rows = page.slice(0, 0)
        elif self._max_rows is not None and self._count + page.num_rows > self._max_rows:
            rows = page.slice(0, self._max_rows - self._count)
            self._truncated = True
        if rows.num_rows:
            size = rows.nbytes
            if self._size + size > self._max_bytes and not (self._count == 0 and rows.num_rows == 1):
                kept = self._fitting(rows)
                self._truncated = True
                rows = rows.slice(0, kept)
                size = rows.nbytes
            if rows.num_rows:
                self._tables = with_chunk(self._tables, (rows,))
                self._count += rows.num_rows
                self._page_mean = size / rows.num_rows
                self._size += size
        return not self._truncated

    def _fitting(self, rows: Any) -> int:
        """the longest prefix of a page that fits what is left (at least one row when none is kept yet).

        :param rows: the page, which does not fit whole
        :ptype rows: pyarrow.Table
        :return: the rows of it to keep
        :rtype: int
        """
        left = self._max_bytes - self._size
        low, high = 0, rows.num_rows - 1
        while low < high:
            middle = (low + high + 1) // 2
            if rows.slice(0, middle).nbytes <= left:
                low = middle
            else:
                high = middle - 1
        return max(low, 1 if self._count == 0 else 0)

    def result(self) -> BoundedArrow:
        """the rows kept so far.

        :return: the kept rows (one table, a chunk a page), whether rows were left, and their bytes
        :rtype: BoundedArrow
        """
        pages = [table for chunk in self._tables for table in chunk]
        table = pa.concat_tables(pages) if pages else self._schema.empty_table()
        return BoundedArrow(table=table, truncated=self._truncated, size=self._size)


def start_arrow_budget(*, schema: Any, max_bytes: int, max_rows: int | None, page_rows: int) -> ArrowBudget:
    """check a byte-capped Arrow read's bounds and start its budget: the one entry every driver's
    ``fetch_arrow_within`` uses.

    :param schema: each column asked for and its Arrow type
    :ptype schema: pyarrow.Schema
    :param max_bytes: the most bytes the kept rows may take, at least 1
    :ptype max_bytes: int
    :param max_rows: the most rows kept; None for no row bound
    :ptype max_rows: int | None
    :param page_rows: the rows a page reads
    :ptype page_rows: int
    :return: an empty budget
    :rtype: ArrowBudget
    :raises ValueError: if ``max_bytes``, ``max_rows`` or ``page_rows`` is below 1
    :raises ImportError: without pyarrow
    """
    _require_pyarrow()
    if max_bytes < 1:
        raise ValueError(f"max_bytes must be at least 1, got {max_bytes}")
    if page_rows < 1:
        raise ValueError(f"page_rows must be at least 1, got {page_rows}")
    if max_rows is not None and max_rows < 1:
        raise ValueError(f"max_rows must be at least 1, got {max_rows}")
    return ArrowBudget(schema, max_bytes, max_rows)
