"""An L3 rail reply's rows as one Arrow table, parsed from the reply's bytes by Arrow's JSON reader.

The broker answers a ``SELECT`` as one JSON object whose ``rows`` are a list of objects
(:meth:`~threetears.core.backends.nats_proxy.NatsProxyL3Backend.fetch`). Decoding it with ``json``
makes every value a Python object on the thread that decodes it, holding the interpreter lock for the
whole reply; a caller that hands the rows on (a worker process, a DuckDB table) then pays again to
turn them back into columns. :func:`reply_rows` hands the reply's bytes to Arrow's C++ JSON reader
instead, which parses without the interpreter lock, and answers the rows as one Arrow table: no row is
ever a Python object.

**The same rows ``fetch`` answers.** Each column is read as the type its JSON carries, the one
:func:`~threetears.core.collections.complete_copy.rail_arrow_types` names for an L3 column: text as
text (a ``NUMERIC`` and a non-``date_`` timestamp arrive as text, and ``fetch`` answers them as text),
an integer as ``int64``, a boolean as ``bool``, and a ``date_`` timestamp as an instant in UTC (``fetch``
answers those as aware datetimes). So ``reply_rows(reply, types).to_pylist()`` equals what ``fetch``
makes of the same reply. A value its column's type cannot hold (a number where text was named, a JSON
object), and text carrying the rail's bytes mark (``\\x``), which ``fetch`` would answer as bytes, are
refused (:class:`ReplyNotArrowError`) rather than answered otherwise than ``fetch`` would.

Needs ``pyarrow`` (``3tears[snapshot]``).
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Any, Final

from threetears.core.backends.protocol import ArrowL3Reader

try:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.ipc as pa_ipc
    import pyarrow.json as pa_json

    _PYARROW = True
except ImportError:  # NOSILENT: the snapshot extra is optional; reply_rows refuses without it, saying so
    _PYARROW = False

__all__ = [
    "ARROW_REPLY_MAGIC",
    "ReplyNotArrowError",
    "arrow_schema_text",
    "l3_fetch_arrow",
    "read_reply",
    "reply_rows",
]

#: the prefix the broker writes a bytes value with (``"\\x"`` and its hex), which ``fetch`` answers as bytes
_BYTES_MARK: Final = "\\x"

#: what an ``l3.query`` reply in Arrow begins with: then its envelope's length (4 bytes, big-endian), the envelope (the
#: JSON reply without its ``rows``), and the rows as one Arrow IPC stream. A JSON reply begins with ``{``.
ARROW_REPLY_MAGIC: Final = b"L3ARROW1"

#: the bytes before an Arrow reply's envelope: the magic and the envelope's length
_ARROW_HEAD: Final = len(ARROW_REPLY_MAGIC) + 4

#: what a JSON reply carries around its rows beyond the envelope's own bytes (``,"rows":``)
_ROWS_KEY_BYTES: Final = len(b',"rows":')


class ReplyNotArrowError(ValueError):
    """a rail reply whose rows cannot be the Arrow table of the types asked for, exactly as ``fetch`` answers them."""


class _RailReply:
    """a parsed rail reply: whether it succeeded, its rows, and whether the broker cut them at its row cap.

    :ivar success: the reply's ``success``
    :ivar rows: its rows, one column a type asked for, in that order (empty for a failed reply)
    :ivar truncated: the reply's ``truncated``: the broker answered fewer rows than the statement matched
    :ivar reply_cap: the reply's ``max_reply_bytes``: the bytes the broker cuts a read's reply at, when it
        cuts by bytes; None from a broker that cuts at its row cap
    """

    __slots__ = ("envelope", "json_bytes", "reply_cap", "rows", "success", "truncated")

    def __init__(
        self,
        success: bool,
        rows: Any,
        truncated: bool = False,
        reply_cap: int | None = None,
        *,
        envelope: dict[str, Any] | None = None,
        json_bytes: int | None = None,
    ) -> None:
        self.success = success
        self.rows = rows
        self.truncated = truncated
        self.reply_cap = reply_cap
        #: an Arrow reply's envelope (its JSON without ``rows``); None for a JSON reply, which the caller decodes
        self.envelope = envelope
        #: the bytes the reply's JSON would take, the measure the broker cuts by; None for a JSON reply (its own
        #: length), 0 for an Arrow reply that does not say its rows' JSON bytes
        self.json_bytes = json_bytes


def reply_rows(reply: bytes, types: Mapping[str, Any]) -> _RailReply:
    """a rail reply's rows as one Arrow table typed by ``types``, parsed without the interpreter lock.

    Blocking, and CPU for as long as the reply is long, but Arrow's reader lets go of the interpreter
    lock while it parses: run it on a thread (``asyncio.to_thread``) and the event loop runs beside it.

    :param reply: the broker's reply, as its bytes
    :ptype reply: bytes
    :param types: column name -> its Arrow type, every column the statement selects, in its order
    :ptype types: Mapping[str, pyarrow.DataType]
    :return: the reply's success, and its rows
    :rtype: _RailReply
    :raises ReplyNotArrowError: when a value is not of its column's type, or text carries the bytes mark
    :raises ImportError: without pyarrow
    """
    if not _PYARROW:
        raise ImportError("an L3 read as Arrow needs pyarrow: install 3tears[snapshot]")
    row_type = pa.struct([pa.field(name, kind) for name, kind in types.items()])
    schema = pa.schema(
        [
            pa.field("success", pa.bool_()),
            pa.field("rows", pa.list_(row_type)),
            pa.field("truncated", pa.bool_()),
            pa.field("max_reply_bytes", pa.int64()),
        ]
    )
    try:
        parsed = pa_json.read_json(
            io.BytesIO(reply),
            # the reply is one line, read in one block; Arrow's own threads stay idle (the caller's thread parses)
            read_options=pa_json.ReadOptions(use_threads=False, block_size=len(reply) + 1),
            parse_options=pa_json.ParseOptions(explicit_schema=schema, unexpected_field_behavior="ignore"),
        )
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
        raise ReplyNotArrowError(f"a rail reply's rows are not of the types asked for: {exc}") from exc
    success = parsed.num_rows == 1 and bool(parsed.column("success")[0].as_py())
    rows = pa.Table.from_struct_array(pa.array([], row_type))
    if success:
        rows = pa.Table.from_struct_array(parsed.column("rows").combine_chunks().flatten())
        for name, kind in types.items():
            if (pa.types.is_string(kind) or pa.types.is_large_string(kind)) and rows.num_rows:
                if pc.any(pc.starts_with(rows.column(name), _BYTES_MARK)).as_py():
                    raise ReplyNotArrowError(
                        f"text column {name!r} holds the rail's bytes mark, which fetch answers as bytes"
                    )
    truncated = parsed.num_rows == 1 and bool(parsed.column("truncated")[0].as_py())
    cap = parsed.column("max_reply_bytes")[0].as_py() if parsed.num_rows == 1 else None
    return _RailReply(success, rows, truncated, reply_cap=cap if isinstance(cap, int) and cap > 0 else None)


def arrow_schema_text(types: Mapping[str, Any]) -> str:
    """the ``arrow_schema`` an ``l3.query`` asking for an Arrow reply carries: the IPC serialization of the schema of
    ``types``, base64.

    :param types: column name -> its Arrow type, every column selected, in order
    :ptype types: Mapping[str, pyarrow.DataType]
    :return: the schema, as the request carries it
    :rtype: str
    :raises ImportError: without pyarrow
    """
    if not _PYARROW:
        raise ImportError("an L3 read as Arrow needs pyarrow: install 3tears[snapshot]")
    schema = pa.schema([pa.field(name, kind) for name, kind in types.items()])
    return base64.b64encode(schema.serialize().to_pybytes()).decode("ascii")


def _arrow_reply_rows(reply: bytes, types: Mapping[str, Any]) -> _RailReply:
    """an Arrow reply's rows (:data:`ARROW_REPLY_MAGIC`): its envelope, then its rows' IPC stream, read in place.

    The broker builds the table with :func:`reply_rows` over the rows it would have answered as JSON, so its columns
    are ``types``; a table of any other schema is refused, never answered otherwise than the JSON would be.

    :param reply: the reply's bytes, beginning with the magic
    :ptype reply: bytes
    :param types: column name -> its Arrow type, every column selected, in order
    :ptype types: Mapping[str, pyarrow.DataType]
    :return: the reply's success, rows and envelope
    :rtype: _RailReply
    :raises ReplyNotArrowError: when the reply is malformed or its table is not of ``types``
    """
    size = int.from_bytes(reply[len(ARROW_REPLY_MAGIC) : _ARROW_HEAD], "big")
    try:
        envelope = json.loads(reply[_ARROW_HEAD : _ARROW_HEAD + size])
    except ValueError as exc:
        raise ReplyNotArrowError(f"an Arrow rail reply whose envelope cannot be read: {exc}") from exc
    if not isinstance(envelope, dict):
        raise ReplyNotArrowError("an Arrow rail reply whose envelope is not an object")
    expected = pa.schema([pa.field(name, kind) for name, kind in types.items()])
    if not envelope.get("success"):
        # a failure carries no rows: the caller raises what its envelope names
        return _RailReply(False, expected.empty_table(), envelope=envelope, json_bytes=0)
    try:
        rows = pa_ipc.open_stream(pa.py_buffer(reply)[_ARROW_HEAD + size :]).read_all()
    except pa.ArrowException as exc:
        raise ReplyNotArrowError(f"an Arrow rail reply whose rows cannot be read: {exc}") from exc
    if not rows.schema.equals(expected, check_metadata=False):
        raise ReplyNotArrowError(f"an Arrow rail reply's rows are {rows.schema}, not the types asked for")
    cap = envelope.get("max_reply_bytes")
    rows_json = envelope.get("rows_json_bytes")
    # the JSON reply's bytes are its envelope's, its rows' and the key between; a hub that does not say its rows'
    # bytes gives 0, and the keyset pager keeps its page's rows rather than size them by the Arrow's smaller bytes
    json_bytes = 0
    if isinstance(rows_json, int) and not isinstance(rows_json, bool) and rows_json >= 0:
        json_bytes = size + _ROWS_KEY_BYTES + rows_json
    return _RailReply(
        True,
        rows,
        bool(envelope.get("truncated")),
        reply_cap=cap if isinstance(cap, int) and not isinstance(cap, bool) and cap > 0 else None,
        envelope=envelope,
        json_bytes=json_bytes,
    )


def read_reply(
    reply: bytes, types: Mapping[str, Any], json_rows: Callable[[bytes, Mapping[str, Any]], _RailReply] = reply_rows
) -> _RailReply:
    """an ``l3.query`` reply's rows as one Arrow table, whichever form the broker answered in: an Arrow reply
    (:data:`ARROW_REPLY_MAGIC`) read in place, anything else as JSON (``json_rows``, :func:`reply_rows`). Blocking:
    on a thread, as :func:`reply_rows`.

    :param reply: the reply's bytes
    :ptype reply: bytes
    :param types: column name -> its Arrow type, every column selected, in order
    :ptype types: Mapping[str, pyarrow.DataType]
    :param json_rows: the JSON reply's parse
    :ptype json_rows: Callable[[bytes, Mapping[str, Any]], _RailReply]
    :return: the reply's success and rows; an Arrow reply's envelope beside them
    :rtype: _RailReply
    :raises ReplyNotArrowError: when the rows are not of ``types``
    :raises ImportError: without pyarrow
    """
    if not _PYARROW:
        raise ImportError("an L3 read as Arrow needs pyarrow: install 3tears[snapshot]")
    if reply[: len(ARROW_REPLY_MAGIC)] == ARROW_REPLY_MAGIC:
        return _arrow_reply_rows(reply, types)
    return json_rows(reply, types)


def _rail_value(value: Any, kind: Any) -> Any:
    """one value a direct pool answered, as the rail's reply carries it in a column of Arrow type ``kind``.

    :param value: the value, as asyncpg answers it
    :ptype value: Any
    :param kind: its column's Arrow type
    :ptype kind: pyarrow.DataType
    :return: the value as the reply's JSON gives it: text for a text column (a datetime as the hub's JSON writes
        it, UTC as ``Z``; a ``Decimal`` or UUID as its ``str``), anything else as it is
    :rtype: Any
    """
    result = value
    if (
        value is not None
        and not isinstance(value, str)
        and (pa.types.is_string(kind) or pa.types.is_large_string(kind))
    ):
        result = value.isoformat().replace("+00:00", "Z") if isinstance(value, datetime) else str(value)
    return result


def _rail_arrow_table(rows: Sequence[Mapping[str, Any]], types: Mapping[str, Any]) -> Any:
    """rows an L3 answered as dicts, as the Arrow table the rail's reply of them reads as (:func:`reply_rows`).

    For an L3 that answers no Arrow (a direct pool, a test's): its rows are Python objects already, and each value
    is made the form the rail's JSON carries for its column's type, so a reader gets one table whichever L3 it holds.
    Value by value, holding the interpreter lock: a reader that must not (a serving process) holds the rail's
    proxy, which never makes a row a Python object (:func:`l3_fetch_arrow`).

    :param rows: the rows, by column name
    :ptype rows: Sequence[Mapping[str, Any]]
    :param types: column name -> its Arrow type, in order
    :ptype types: Mapping[str, pyarrow.DataType]
    :return: the rows
    :rtype: pyarrow.Table
    :raises ImportError: without pyarrow
    """
    if not _PYARROW:
        raise ImportError("an L3 read as Arrow needs pyarrow: install 3tears[snapshot]")
    schema = pa.schema([pa.field(name, kind) for name, kind in types.items()])
    columns = [pa.array([_rail_value(row.get(name), kind) for row in rows], kind) for name, kind in types.items()]
    return pa.Table.from_arrays(columns, schema=schema)


async def l3_fetch_arrow(l3: Any, query: str, *params: Any, types: Mapping[str, Any]) -> Any:
    """a SELECT's rows as the rail's Arrow, from any L3: the rail's proxy parses its reply into Arrow off the
    interpreter lock (:meth:`~threetears.core.backends.nats_proxy.NatsProxyL3Backend.fetch_arrow`); any other L3's
    dict rows are made the same table on a thread (:func:`_rail_arrow_table`).

    :param l3: the L3: an :class:`~threetears.core.backends.protocol.ArrowL3Reader`, or anything with ``fetch``
    :ptype l3: Any
    :param query: the SELECT
    :ptype query: str
    :param params: its parameters
    :ptype params: Any
    :param types: column name -> its Arrow type, every column selected, in order
    :ptype types: Mapping[str, pyarrow.DataType]
    :return: the rows
    :rtype: pyarrow.Table
    """
    if isinstance(l3, ArrowL3Reader):
        answered = await l3.fetch_arrow(query, *params, types=types)
    else:
        answered = await asyncio.to_thread(_rail_arrow_table, await l3.fetch(query, *params), types)
    return answered
