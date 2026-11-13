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
import threading
import time
from collections.abc import Sequence
from typing import Any, Final

from threetears.core.backends.schema_sql import json_default, transport_param

try:
    import pyarrow as pa
    import pyarrow.ipc as pa_ipc

    _PYARROW = True
except ImportError:  # NOSILENT: the snapshot extra is optional; without it a copy is sent as JSON, as before
    _PYARROW = False

if _PYARROW:
    # pyarrow looks for pandas at its first conversion of Python values: an import searched along sys.path, holding
    # the interpreter lock 1-4 ms with no turn. Done here, at import, never in a process's first copy's build beside
    # the caller's event loop
    pa.array((), pa.int64())

__all__ = [
    "COPY_ARROW_FIELD",
    "JSON_COLUMNS_FIELD",
    "ROWS_DIGEST_FIELD",
    "ArrowCopyRows",
    "arrow_copy_available",
    "copy_rows_arrow",
    "framed_copy_request",
    "rows_digest",
    "split_copy_request",
]

#: the first bytes of an ``l3.tx.copy`` request whose rows are Arrow; a JSON request starts ``{``
_COPY_ARROW_MAGIC: Final = b"L3COPYA1"

#: the ``tx.begin`` reply's field by which a hub announces it reads Arrow copy requests
COPY_ARROW_FIELD: Final = "copy_arrow"

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


#: the longest the build holds the interpreter lock before it lets the caller's event loop run: measured, a 12 MB copy's
#: build held the caller's loop 31 ms behind on a thread that never blocked (a waiting thread is not handed the lock
#: while the holder keeps releasing and taking it back in pyarrow's calls). With one slice past it (:data:`_SLICE_VALUES`)
#: a stretch stays under 1 ms: on a loaded host the OS can stop the build while it holds the lock, and the caller's
#: loop is late by that stop only when its tick falls inside a stretch, so the shorter the stretches the rarer that is
#: (1 ms stretches: 47.5 ms worst, p99 4.7 ms of added lateness on a Linux container with every CPU busy)
_HOLD_SECONDS: Final = 0.0005

#: how long the build blocks when it lets the loop run and has no loop to hand the turn to
_TURN_SECONDS: Final = 0.0002

#: the longest the build waits for the caller's loop to take a turn it was handed: a loop that has not run by then is
#: busy with work of its own, not waiting for the interpreter lock, and the build goes on
_TURN_WAIT_SECONDS: Final = 0.05

#: the values read or converted in Python between two of the build's turns: few enough that no stretch nears
#: :data:`_HOLD_SECONDS` (4096 held the caller's loop a whole switch interval, 5 ms, on a loaded host; 256 still ran
#: 0.6 ms a slice on an oversubscribed test host)
_SLICE_VALUES: Final = 128

#: the stream's bytes copied out of Arrow's buffer at a time, with the build's turns between: one copy of a 12 MB stream
#: into bytes held the interpreter lock for its length (3.6 ms on a Linux container). Under the C allocator's 128 KiB
#: threshold for mapping fresh pages: a 1 MiB piece was a fresh mapping each time, 256 page faults taken with the lock
#: held (up to 4 ms of the kernel's time on a loaded host)
_PIECE_BYTES: Final = 1 << 16


class _Turns:
    """the build's turns with the caller's event loop: past :data:`_HOLD_SECONDS` of holding the interpreter lock, the
    build hands the loop a turn and blocks until the loop has run.

    A turn is handed, not offered: the build schedules a callback on the loop and waits for it to run. A sleep offered
    the lock for :data:`_TURN_SECONDS` only, and on a loaded host the loop's thread was often not scheduled within it:
    the build took the lock back, and the loop waited for CPython's 5 ms switch interval, turn after turn (pool
    diagnostics: the loop 8-20 ms late in ``select`` while the build ran ``_sliced_array`` / ``_column_of``). With no
    loop (a build called off any loop), the turn is the sleep.
    """

    __slots__ = ("_handed", "_loop", "_since")

    def __init__(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self._since = time.monotonic()
        self._loop = loop
        self._handed = threading.Event()

    def take(self) -> None:
        """let the caller's loop run, when the build has held the lock its share.

        :return: nothing
        :rtype: None
        """
        if time.monotonic() - self._since >= _HOLD_SECONDS:
            self.hand()

    def hand(self) -> None:
        """let the caller's loop run now.

        :return: nothing
        :rtype: None
        """
        loop = self._loop
        if loop is None or loop.is_closed():
            time.sleep(_TURN_SECONDS)
        else:
            self._handed.clear()
            loop.call_soon_threadsafe(self._handed.set)
            self._handed.wait(_TURN_WAIT_SECONDS)
        self._since = time.monotonic()


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
    call over a whole column of text held the caller's loop for its length.

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
        slices.append(pa.array(values[start : start + _SLICE_VALUES], arrow_type))
        turns.take()
    return pa.concat_arrays(slices) if slices else pa.array([], arrow_type)


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
    (:class:`_Turns`), holding the lock at most a few milliseconds at a time.

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
    sink = pa.BufferOutputStream()
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
