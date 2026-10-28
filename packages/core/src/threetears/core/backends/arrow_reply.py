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
import io
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Final

from threetears.core.backends.protocol import ArrowL3Reader

try:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.json as pa_json

    _PYARROW = True
except ImportError:  # NOSILENT: the snapshot extra is optional; reply_rows refuses without it, saying so
    _PYARROW = False

__all__ = ["ReplyNotArrowError", "l3_fetch_arrow", "reply_rows"]

#: the prefix the broker writes a bytes value with (``"\\x"`` and its hex), which ``fetch`` answers as bytes
_BYTES_MARK: Final = "\\x"


class ReplyNotArrowError(ValueError):
    """a rail reply whose rows cannot be the Arrow table of the types asked for, exactly as ``fetch`` answers them."""


class _RailReply:
    """a parsed rail reply: whether it succeeded, and its rows.

    :ivar success: the reply's ``success``
    :ivar rows: its rows, one column a type asked for, in that order (empty for a failed reply)
    """

    __slots__ = ("rows", "success")

    def __init__(self, success: bool, rows: Any) -> None:
        self.success = success
        self.rows = rows


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
    schema = pa.schema([pa.field("success", pa.bool_()), pa.field("rows", pa.list_(row_type))])
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
    return _RailReply(success, rows)


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
