"""An ``l3.tx.copy`` request's rows as one Arrow IPC stream, behind the request's JSON envelope.

The rail's copy request carries its rows as JSON (:meth:`~threetears.core.backends.nats_proxy._ProxyConnection.copy_records_to_table`):
every value converted in Python (``transport_param``) and the whole request encoded by ``json``, then decoded
again value by value at the hub. A copy of 12 MB spent more on that than its ``COPY`` took. Sent as Arrow, a
column of text or numbers is laid out by Arrow in C, and the hub encodes its ``COPY`` straight from the columns.

**The wire.** the magic (``L3COPYA1``), the envelope's length (four bytes, big-endian), the envelope (the
request's JSON fields but its rows, its proof among them), then the rows' IPC stream. The envelope names the
stream's SHA-256 (:data:`ROWS_DIGEST_FIELD`), so a proof over the envelope covers the rows, and the columns sent as
JSON text (:data:`JSON_COLUMNS_FIELD`).

**The same values as JSON.** A column is sent as Arrow only where Arrow holds every value exactly as the JSON
request's parse gives it: all text (``string``), all integers that fit 64 bits (``int64``), all floats
(``float64``, NaN, infinities and the sign of a zero kept), all booleans (``bool``), each with ``None`` as null. A
UUID, datetime, Decimal or bytes column is converted as the JSON request converts it (``transport_param``) and is
then text. Any other column (an array, a JSON object, a mix of kinds, an integer past 64 bits) is sent as each
value's JSON text, encoded as the JSON request encodes it, and the hub parses it as it parses that request.

**Rolling.** A hub announces it reads this form in its ``tx.begin`` reply (:data:`COPY_ARROW_FIELD`); a client
sends it only to a hub that did, and JSON otherwise.

Needs ``pyarrow`` (``3tears[snapshot]``); without it every copy is sent as JSON.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import operator
import struct
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Final

from threetears.core.backends.schema_sql import json_default, transport_param
from threetears.core.loop_turns import LoopTurns as _Turns

try:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.ipc as pa_ipc

    _PYARROW = True
except ImportError:  # NOSILENT: the snapshot extra is optional; without it a copy is sent as JSON, as before
    _PYARROW = False

if _PYARROW:
    #: the pool every Arrow buffer of the build is allocated from: the system allocator, never pyarrow's default
    #: (mimalloc in our images). Under mimalloc a quarter of a 12 MB copy's builds held the lock one stretch of 5-15 ms
    #: of the build's CPU, inside a slice's conversion or a column's join, where every other stretch ran under 1 ms;
    #: under the system allocator 20 of 20 builds ran every stretch under 0.71 ms (first build in a fresh process,
    #: Linux container on Apple silicon, pyarrow 25, the pools alternated build by build). A serving process's Arrow
    #: work belongs on the system allocator in any case (:mod:`threetears.core.arrow_pool`)
    _POOL: Final[Any] = pa.system_memory_pool()

    # pyarrow looks for pandas at its first conversion of Python values: an import searched along sys.path, holding
    # the interpreter lock 1-4 ms with no turn. And its first conversion of text sets its converter up: 0.7-1.6 ms of
    # CPU against 0.01-0.04 ms for the next (same container). Both done here, at import, for every kind a column is
    # sent as, never in a process's first copy's build beside the caller's event loop
    pa.array((), pa.int64())
    for _value, _kind in (("a", pa.string()), (1, pa.int64()), (0.5, pa.float64()), (True, pa.bool_())):
        pa.array([_value, None], _kind, memory_pool=_POOL)
    del _value, _kind

__all__ = [
    "CONFLICT_COLUMNS_FIELD",
    "COPY_ARROW_FIELD",
    "APPEND_FIELD",
    "DELETE_FIELD",
    "JSON_COLUMNS_FIELD",
    "ON_CONFLICT_FIELD",
    "ROWS_DIGEST_FIELD",
    "UPDATE_COLUMNS_FIELD",
    "WRITE_ARROW_FIELD",
    "ArrowCopyRows",
    "arrow_copy_available",
    "arrow_python_columns",
    "arrow_python_rows",
    "copy_rows_arrow",
    "first_request_rows",
    "framed_copy_request",
    "planned_write_table",
    "rows_digest",
    "split_copy_request",
    "with_constant_columns",
    "with_typed_constants",
    "write_request_rows",
    "write_type",
]

#: the first bytes of an ``l3.tx.copy`` request whose rows are Arrow; a JSON request starts ``{``
_COPY_ARROW_MAGIC: Final = b"L3COPYA1"

#: the ``tx.begin`` reply's field by which a hub announces it reads Arrow copy requests
COPY_ARROW_FIELD: Final = "copy_arrow"

#: the ``tx.begin`` reply's field by which a hub announces it reads an Arrow ``l3.tx.copy`` that upserts or deletes its
#: rows (:data:`ON_CONFLICT_FIELD`, :data:`DELETE_FIELD`) and Arrow columns typed ``decimal128``, ``timestamp[us, UTC]``,
#: ``timestamp[us]`` and ``date32`` (:func:`write_type`). Never send either to a hub without it: one that ignored the
#: fields would insert the rows meant for an upsert or a delete
WRITE_ARROW_FIELD: Final = "write_arrow"

#: the envelope's field making its rows an upsert: ``"update"``, with :data:`CONFLICT_COLUMNS_FIELD` and
#: :data:`UPDATE_COLUMNS_FIELD`; absent for new rows
ON_CONFLICT_FIELD: Final = "on_conflict"

#: an upsert's conflict target, the table's primary key, each among the request's columns
CONFLICT_COLUMNS_FIELD: Final = "conflict_columns"

#: the columns an upsert's ``DO UPDATE`` sets from ``EXCLUDED``; none makes it ``DO NOTHING``
UPDATE_COLUMNS_FIELD: Final = "update_columns"

#: the envelope's field making its rows keys to delete: the request's columns are exactly the table's primary key
DELETE_FIELD: Final = "delete"

#: the envelope's field making its rows a versioned table's appended versions (or "gone" markers): inserted with no
#: conflict clause, as a copy is, but noted a row each as the upsert they replace was, never as a bulk copy's one
#: whole-table note. Never with :data:`ON_CONFLICT_FIELD` or :data:`DELETE_FIELD`
APPEND_FIELD: Final = "append"

#: the envelope's field naming the rows' IPC stream's SHA-256, base64url without padding (:func:`rows_digest`)
ROWS_DIGEST_FIELD: Final = "rows_sha256"

#: the envelope's field listing the columns (by position) whose values are each one's JSON text
JSON_COLUMNS_FIELD: Final = "json_columns"

#: the envelope's length, after the magic
_LENGTH: Final = struct.Struct(">I")

_NONE: Final = type(None)


def arrow_copy_available() -> bool:
    """whether this process can send a copy's rows as Arrow (``pyarrow`` is installed).

    :return: whether it can
    :rtype: bool
    """
    return _PYARROW


class ArrowCopyRows:
    """a copy's rows as an IPC stream, and what the request's envelope says of them.

    :ivar ipc: the stream
    :ivar digest: its :func:`rows_digest`
    :ivar json_columns: the columns sent as JSON text, by position
    """

    __slots__ = ("digest", "ipc", "json_columns")

    def __init__(self, ipc: bytes, digest: str, json_columns: list[int]) -> None:
        self.ipc = ipc
        self.digest = digest
        self.json_columns = json_columns


def rows_digest(ipc: Any) -> str:
    """the SHA-256 of a request's rows, as its envelope names it.

    :param ipc: the rows' IPC stream, any buffer
    :ptype ipc: bytes | memoryview | pyarrow.Buffer
    :return: the digest, base64url without padding
    :rtype: str
    """
    digest = hashlib.sha256(ipc).digest()
    return str(base64.urlsafe_b64encode(digest).rstrip(b"="), "ascii")


#: the values read or converted in Python between two of the build's turns: few enough that no stretch nears
#: :data:`~threetears.core.loop_turns._HOLD_SECONDS` (4096 held the caller's loop a whole switch interval, 5 ms, on a loaded host; 256 still ran
#: 0.6 ms a slice on an oversubscribed test host)
_SLICE_VALUES: Final = 128

#: the stream's bytes copied out of Arrow's buffer at a time, with the build's turns between: one copy of a 12 MB stream
#: into bytes held the interpreter lock for its length (3.6 ms on a Linux container). Under the C allocator's 128 KiB
#: threshold for mapping fresh pages: a 1 MiB piece was a fresh mapping each time, 256 page faults taken with the lock
#: held (up to 4 ms of the kernel's time on a loaded host)
_PIECE_BYTES: Final = 1 << 16


def _converted(convert: Any, values: Sequence[Any], turns: _Turns) -> list[Any]:
    """``convert`` of each value, a slice at a time with the build's turns between.

    :param convert: the conversion
    :ptype convert: Callable[[Any], Any]
    :param values: the values
    :ptype values: Sequence[Any]
    :param turns: the build's turns
    :ptype turns: _Turns
    :return: the converted values
    :rtype: list[Any]
    """
    converted: list[Any] = []
    for start in range(0, len(values), _SLICE_VALUES):
        converted.extend(map(convert, values[start : start + _SLICE_VALUES]))
        turns.take()
    return converted


def _json_value(value: Any) -> str | None:
    """one value's JSON text, as the JSON request encodes it; ``None`` as null.

    :param value: the value, already as the rail carries it
    :ptype value: Any
    :return: the text
    :rtype: str | None
    """
    return None if value is None else json.dumps(value, default=json_default)


def _json_text(values: Sequence[Any], turns: _Turns) -> Any:
    """a column as each value's JSON text, as the JSON request encodes it; ``None`` as null.

    :param values: the values, already as the rail carries them
    :ptype values: Sequence[Any]
    :param turns: the build's turns
    :ptype turns: _Turns
    :return: the column
    :rtype: pyarrow.Array
    """
    return _sliced_array(_converted(_json_value, values, turns), pa.string(), turns)


def _sliced_array(values: Sequence[Any], arrow_type: Any, turns: _Turns) -> Any:
    """``values`` as one Arrow array of ``arrow_type``, converted a slice at a time with the build's turns between: one
    call over a whole column of text held the caller's loop for its length. Allocated from :data:`_POOL`.

    :param values: the values
    :ptype values: Sequence[Any]
    :param arrow_type: the type
    :ptype arrow_type: pyarrow.DataType
    :param turns: the build's turns
    :ptype turns: _Turns
    :return: the array
    :rtype: pyarrow.Array
    :raises OverflowError: for an integer the type cannot hold
    :raises UnicodeEncodeError: for a string UTF-8 cannot hold
    :raises pyarrow.ArrowException: for a value the type cannot hold
    """
    slices = []
    for start in range(0, len(values), _SLICE_VALUES):
        slices.append(pa.array(values[start : start + _SLICE_VALUES], arrow_type, memory_pool=_POOL))
        turns.take()
    return pa.concat_arrays(slices, memory_pool=_POOL) if slices else pa.array([], arrow_type, memory_pool=_POOL)


#: the Arrow type a column of one kind of value is sent as
_ARROW_KINDS: Final[dict[type, Any]] = (
    {str: pa.string(), int: pa.int64(), float: pa.float64(), bool: pa.bool_()} if _PYARROW else {}
)


def _kinds(values: Sequence[Any], turns: _Turns) -> set[type]:
    """the types of a column's values, ``None``'s left out, read a slice at a time with the build's turns between.

    :param values: the values
    :ptype values: Sequence[Any]
    :param turns: the build's turns
    :ptype turns: _Turns
    :return: the types
    :rtype: set[type]
    """
    kinds: set[type] = set()
    for start in range(0, len(values), _SLICE_VALUES):
        kinds.update(map(type, values[start : start + _SLICE_VALUES]))
        turns.take()
    kinds.discard(_NONE)
    return kinds


def _column_of(records: Sequence[Sequence[Any]], position: int, turns: _Turns) -> list[Any]:
    """one column of the rows, read a slice of rows at a time with the build's turns between.

    :param records: the rows
    :ptype records: Sequence[Sequence[Any]]
    :param position: the column's position in each row
    :ptype position: int
    :param turns: the build's turns
    :ptype turns: _Turns
    :return: the column's values
    :rtype: list[Any]
    """
    taken = operator.itemgetter(position)
    values: list[Any] = []
    for start in range(0, len(records), _SLICE_VALUES):
        values.extend(map(taken, records[start : start + _SLICE_VALUES]))
        turns.take()
    return values


def _arrow_column(values: Sequence[Any], turns: _Turns) -> tuple[Any, bool]:
    """one column of a copy's rows as an Arrow array, and whether it is sent as JSON text.

    :param values: the column's values, as the caller gave them
    :ptype values: Sequence[Any]
    :param turns: the build's turns
    :ptype turns: _Turns
    :return: the array, and whether its values are their JSON text
    :rtype: tuple[pyarrow.Array, bool]
    """
    kinds = _kinds(values, turns)
    if not kinds <= _ARROW_KINDS.keys() or len(kinds) > 1:
        # converted as the JSON request converts them; a UUID, datetime, Decimal or bytes column is then text
        values = _converted(transport_param, values, turns)
        kinds = _kinds(values, turns)
    kind = next(iter(kinds)) if len(kinds) == 1 else str if not kinds else None
    array: Any = None
    if kind in _ARROW_KINDS:
        try:
            array = _sliced_array(values, _ARROW_KINDS[kind], turns)
        except OverflowError, UnicodeEncodeError, pa.ArrowException:
            # NOSILENT: an integer past 64 bits, a string UTF-8 cannot hold: sent as its JSON text instead
            array = None
    return (array, False) if array is not None else (_json_text(values, turns), True)


def copy_rows_arrow(
    columns: Sequence[str], records: Sequence[Sequence[Any]], *, loop: asyncio.AbstractEventLoop | None = None
) -> ArrowCopyRows:
    """a copy's rows as an Arrow IPC stream, one column per copied column, named as they are.

    Blocking, and CPU for as long as the rows are many: run it on a thread, as the JSON request is encoded. The rows
    are the caller's Python objects, so reading them holds the interpreter lock wherever it runs (handing them to a
    process would hold it as long, to pickle them); the build takes turns with the caller's event loop instead
    (:class:`~threetears.core.loop_turns.LoopTurns`), holding the lock at most a few milliseconds at a time.

    :param columns: the columns each row supplies, in order
    :ptype columns: Sequence[str]
    :param records: the rows' values, each in the order of ``columns``
    :ptype records: Sequence[Sequence[Any]]
    :param loop: the caller's event loop, handed each of the build's turns; None for a build off any loop
    :ptype loop: asyncio.AbstractEventLoop | None
    :return: the stream, its digest and its JSON columns
    :rtype: ArrowCopyRows
    :raises ImportError: without pyarrow
    :raises ValueError: when a row is not as wide as ``columns``
    """
    if not _PYARROW:
        raise ImportError("a copy's rows as Arrow need pyarrow: install 3tears[snapshot]")
    turns = _Turns(loop)
    width = len(columns)
    for start in range(0, len(records), _SLICE_VALUES):
        if any(len(record) != width for record in records[start : start + _SLICE_VALUES]):
            raise ValueError(f"a row has other than {width} values, one for each column")
        turns.take()
    arrays = []
    json_columns = []
    for position in range(width):
        # one column at a time, never the whole transposition in one call: each call holds the interpreter lock
        # for its length, and the caller's event loop runs between them
        values = _column_of(records, position, turns)
        array, as_json = _arrow_column(values, turns)
        turns.take()
        arrays.append(array)
        if as_json:
            json_columns.append(position)
    table = pa.Table.from_arrays(arrays, names=list(columns))
    turns.take()
    ipc = _stream_bytes(_ipc_stream(table, turns), turns)
    return ArrowCopyRows(ipc, rows_digest(ipc), json_columns)


def _ipc_stream(table: Any, turns: _Turns) -> Any:
    """the rows' IPC stream, written by Arrow with the interpreter lock released, then a turn: the build's hold counts
    from there.

    :param table: the rows
    :ptype table: pyarrow.Table
    :param turns: the build's turns
    :ptype turns: _Turns
    :return: the stream's buffer
    :rtype: pyarrow.Buffer
    """
    sink = pa.BufferOutputStream(memory_pool=_POOL)
    with pa_ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    turns.hand()
    return sink.getvalue()


def _stream_bytes(stream: Any, turns: _Turns) -> bytes:
    """an IPC stream's buffer as bytes, copied out :data:`_PIECE_BYTES` at a time with the build's turns between.

    CPython's ``bytes.join`` copies with the interpreter lock released once its result passes 1 MB and every part is
    ``bytes`` (never a ``memoryview``), so the pieces are joined, and the request framed from these bytes
    (:func:`framed_copy_request`), without holding the caller's loop.

    :param stream: the stream's buffer
    :ptype stream: pyarrow.Buffer
    :param turns: the build's turns
    :ptype turns: _Turns
    :return: the stream
    :rtype: bytes
    """
    pieces: list[bytes] = []
    for start in range(0, stream.size, _PIECE_BYTES):
        pieces.append(stream.slice(start, min(_PIECE_BYTES, stream.size - start)).to_pybytes())
        turns.take()
    return b"".join(pieces)


def framed_copy_request(envelope: bytes, ipc: bytes) -> bytes:
    """the wire bytes of an Arrow copy request: the magic, the envelope's length, the envelope, the rows.

    :param envelope: the envelope, as the request's JSON is encoded (its proof included)
    :ptype envelope: bytes
    :param ipc: the rows' IPC stream
    :ptype ipc: bytes
    :return: the request
    :rtype: bytes
    """
    return b"".join((_COPY_ARROW_MAGIC, _LENGTH.pack(len(envelope)), envelope, ipc))


def split_copy_request(data: bytes | memoryview) -> tuple[memoryview, int] | None:
    """an Arrow copy request's envelope and where its rows start; ``None`` for a request that is not one.

    :param data: the request's bytes
    :ptype data: bytes | memoryview
    :return: the envelope (a view of ``data``) and the offset of the rows' IPC stream
    :rtype: tuple[memoryview, int] | None
    :raises ValueError: when the request starts as one but its envelope runs past its end
    """
    view = memoryview(data)
    found: tuple[memoryview, int] | None = None
    if bytes(view[: len(_COPY_ARROW_MAGIC)]) == _COPY_ARROW_MAGIC:
        start = len(_COPY_ARROW_MAGIC) + _LENGTH.size
        if len(view) < start:
            raise ValueError("an Arrow copy request cut short before its envelope")
        (size,) = _LENGTH.unpack_from(view, len(_COPY_ARROW_MAGIC))
        if start + size > len(view):
            raise ValueError("an Arrow copy request's envelope runs past its end")
        found = view[start : start + size], start + size
    return found


# ── the write door (WRITE_ARROW_FIELD): a caller's Arrow table upserted, copied or deleted as it is ──────────────────

#: the values one :func:`arrow_python_rows` / :func:`arrow_python_columns` call turns into Python objects: one
#: ``to_pylist`` holds the interpreter lock for its whole length, so a table is read a slice at a time and the caller's
#: event loop runs between them (8,192 values, ~1 ms)
_PYTHON_VALUES: Final = 8192

#: the timestamp units a timestamp column is widened from to microseconds, losing nothing (``ns`` would: refused)
_WIDENED_UNITS: Final = frozenset({"s", "ms", "us"})


def write_type(arrow_type: Any, *, as_utc: bool = False) -> Any:
    """the Arrow type the write door carries a column of ``arrow_type`` as, the cast losing nothing; ``None`` when it
    carries no such column (the caller writes the rows as Python values instead).

    The door reads ``string``, ``int64``, ``float64``, ``bool``, ``decimal128``, ``timestamp[us, UTC]``,
    ``timestamp[us]`` and ``date32`` (``WIRE.md`` section 4), each landing what its values sent as JSON land. So text
    of any layout is ``string``; any signed integer and an unsigned one under 64 bits ``int64``; any float
    ``float64``; a decimal of 38 digits or fewer ``decimal128``; a timestamp in seconds to microseconds is in
    microseconds, at UTC when it carries a zone (the instant kept) and naive when not; ``date64`` is ``date32``; a
    dictionary is its values'; a column of nulls only is ``string``. ``uint64`` (past ``int64``), a timestamp in
    nanoseconds (finer than the door), binary, nested and every other type are carried by none.

    :param arrow_type: the column's type
    :ptype arrow_type: pyarrow.DataType
    :param as_utc: the table's column is ``timestamptz``: a naive timestamp is read as UTC, as
        :func:`~threetears.core.backends.schema_sql.coerce_datetime_for_write_tz` reads a naive datetime
    :ptype as_utc: bool
    :return: the type carried, or None
    :rtype: pyarrow.DataType | None
    """
    types = pa.types
    if types.is_dictionary(arrow_type):
        arrow_type = arrow_type.value_type
    carried: Any = None
    if types.is_string(arrow_type) or types.is_large_string(arrow_type) or types.is_string_view(arrow_type):
        carried = pa.string()
    elif types.is_null(arrow_type):
        carried = pa.string()
    elif types.is_boolean(arrow_type):
        carried = pa.bool_()
    elif types.is_signed_integer(arrow_type) or (types.is_unsigned_integer(arrow_type) and arrow_type.bit_width < 64):
        carried = pa.int64()
    elif types.is_floating(arrow_type):
        carried = pa.float64()
    elif types.is_decimal(arrow_type) and arrow_type.precision <= 38:
        carried = pa.decimal128(arrow_type.precision, arrow_type.scale)
    elif types.is_timestamp(arrow_type) and arrow_type.unit in _WIDENED_UNITS:
        carried = pa.timestamp("us", tz="UTC") if arrow_type.tz is not None or as_utc else pa.timestamp("us")
    elif types.is_date(arrow_type):
        carried = pa.date32()
    return carried


def with_typed_constants(table: Any, values: Mapping[str, tuple[Any, Any]]) -> Any:
    """``table`` with each of ``values`` as a column holding one value in every row, of the Arrow type named with it:
    replacing a column of that name, else added after the rest. Laid out by Arrow, never a Python value a row, and
    never a type inferred from the value. A versioned table's "gone" markers (the scope, the epoch, the flag).

    :param table: the rows
    :ptype table: pyarrow.Table
    :param values: each column's one value and its Arrow type
    :ptype values: Mapping[str, tuple[Any, pyarrow.DataType]]
    :return: the rows with them
    :rtype: pyarrow.Table
    """
    for name, (value, kind) in values.items():
        column = pa.repeat(pa.scalar(value, kind), table.num_rows, memory_pool=_POOL)
        position = table.schema.get_field_index(name)
        table = table.set_column(position, name, column) if position >= 0 else table.append_column(name, column)
    return table


def with_constant_columns(table: Any, values: Mapping[str, datetime | int]) -> Any:
    """``table`` with each of ``values`` as a column holding it in every row: replacing a column of that name, else
    added after the rest. Laid out by Arrow, never a Python value a row. A write's stamps: its times (``date_created``,
    ``date_updated``), aware datetimes, as ``timestamp[us, UTC]``; its epoch (``epoch``), an int, as ``int64``.

    :param table: the rows
    :ptype table: pyarrow.Table
    :param values: each column's one value, an aware datetime or an int
    :ptype values: Mapping[str, datetime | int]
    :return: the rows with them
    :rtype: pyarrow.Table
    """
    for name, value in values.items():
        kind = pa.int64() if isinstance(value, int) else pa.timestamp("us", tz="UTC")
        column = pa.repeat(pa.scalar(value, kind), table.num_rows, memory_pool=_POOL)
        position = table.schema.get_field_index(name)
        table = table.set_column(position, name, column) if position >= 0 else table.append_column(name, column)
    return table


def planned_write_table(
    table: Any, columns: Sequence[tuple[str, str | None, Any]], *, sort_by: Sequence[str] = ()
) -> Any:
    """the request's table: each of ``columns`` cast to the type the door carries it as, or all nulls where the rows
    supply none, in that order; ordered by ``sort_by`` (a copy's key, the order a bulk load into YugabyteDB wants).

    :param table: the rows
    :ptype table: pyarrow.Table
    :param columns: each column sent: its name, the rows' column it is read from (None: nulls) and its type
        (:func:`write_type`; None with no column read: text)
    :ptype columns: Sequence[tuple[str, str | None, pyarrow.DataType | None]]
    :param sort_by: the columns the rows are ordered by, ascending; none keeps the rows' order
    :ptype sort_by: Sequence[str]
    :return: the table
    :rtype: pyarrow.Table
    :raises pyarrow.ArrowInvalid: for a value its type cannot hold
    """
    arrays = []
    for _name, source, carried in columns:
        if source is None:
            # text nulls: a null lands NULL whatever the table's column, as a JSON null does
            arrays.append(pa.nulls(table.num_rows, carried or pa.string(), memory_pool=_POOL))
        else:
            column = table.column(source)
            if pa.types.is_dictionary(column.type):
                column = pc.cast(column, column.type.value_type, memory_pool=_POOL)
            arrays.append(column if column.type == carried else pc.cast(column, carried, memory_pool=_POOL))
    planned = pa.Table.from_arrays(arrays, names=[name for name, _source, _carried in columns])
    if sort_by and planned.num_rows > 1:
        order = pc.sort_indices(planned, sort_keys=[(name, "ascending") for name in sort_by], memory_pool=_POOL)
        planned = pc.take(planned, order, memory_pool=_POOL)
    return planned


def first_request_rows(table: Any, limit: int) -> int:
    """how many rows the first request of ``table`` tries: all of them when Arrow's layout of them fits ``limit``,
    else their share of it, a tenth spare for what the stream adds (its schema, each buffer's padding).

    :param table: the rows
    :ptype table: pyarrow.Table
    :param limit: the bytes one request's rows may come to
    :ptype limit: int
    :return: the rows, at least one
    :rtype: int
    """
    size = int(table.nbytes)
    rows = int(table.num_rows)
    return rows if size <= limit else max(1, rows * limit * 9 // (10 * size))


def write_request_rows(
    table: Any, start: int, rows: int, limit: int, loop: asyncio.AbstractEventLoop | None = None
) -> tuple[ArrowCopyRows, int]:
    """one request's rows of ``table`` from ``start``: ``rows`` of them, fewer while their IPC stream is past
    ``limit`` bytes. A statement cut by Arrow slices, never a Python value a row.

    Blocking: run it on the copy build thread. The stream is written by Arrow and copied out with the build's turns
    with the caller's loop (:func:`_ipc_stream`, :func:`_stream_bytes`).

    :param table: the rows, every column as the door carries it (:func:`planned_write_table`)
    :ptype table: pyarrow.Table
    :param start: the first row
    :ptype start: int
    :param rows: the rows to try, at least one
    :ptype rows: int
    :param limit: the bytes one request's rows may come to
    :ptype limit: int
    :param loop: the caller's event loop, handed the build's turns; None off any loop
    :ptype loop: asyncio.AbstractEventLoop | None
    :return: the request's rows, and how many rows of ``table`` they are
    :rtype: tuple[ArrowCopyRows, int]
    :raises ImportError: without pyarrow
    :raises ValueError: when one row alone is past ``limit``
    """
    if not _PYARROW:
        raise ImportError("an Arrow write needs pyarrow: install 3tears[snapshot]")
    turns = _Turns(loop)
    while True:
        piece = table.slice(start, rows)
        stream = _ipc_stream(piece, turns)
        if stream.size <= limit:
            break
        if piece.num_rows <= 1:
            raise ValueError(f"one row's Arrow stream is {stream.size} bytes, past the request's {limit}")
        rows = max(1, min(piece.num_rows - 1, piece.num_rows * limit * 9 // (10 * stream.size)))
    ipc = _stream_bytes(stream, turns)
    return ArrowCopyRows(ipc, rows_digest(ipc), []), int(piece.num_rows)


def arrow_python_columns(table: Any, names: Sequence[str], turns: _Turns | None = None) -> list[list[Any]]:
    """the values of ``names`` in ``table`` as Python objects, a column each, read a slice at a time
    (:data:`_PYTHON_VALUES`) so the caller's loop runs between them. Run it on a thread.

    :param table: the rows
    :ptype table: pyarrow.Table
    :param names: the columns
    :ptype names: Sequence[str]
    :param turns: the thread's turns with the caller's loop, handed one between slices; None to let the switch
        interval share the lock
    :ptype turns: LoopTurns | None
    :return: each column's values, in row order
    :rtype: list[list[Any]]
    """
    columns: list[list[Any]] = []
    for name in names:
        column = table.column(name)
        values: list[Any] = []
        for start in range(0, len(column), _PYTHON_VALUES):
            values.extend(column.slice(start, _PYTHON_VALUES).to_pylist())
            if turns is not None:
                turns.take()
        columns.append(values)
    return columns


def arrow_python_rows(table: Any) -> list[dict[str, Any]]:
    """``table``'s rows as Python dicts (``to_pylist``), read a slice of rows at a time so the caller's loop runs
    between them. Run it on a thread.

    :param table: the rows
    :ptype table: pyarrow.Table
    :return: the rows
    :rtype: list[dict[str, Any]]
    """
    step = max(1, _PYTHON_VALUES // max(1, table.num_columns))
    rows: list[dict[str, Any]] = []
    for start in range(0, table.num_rows, step):
        rows.extend(table.slice(start, step).to_pylist())
    return rows
