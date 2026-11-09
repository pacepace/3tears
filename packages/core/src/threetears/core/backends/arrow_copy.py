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

import base64
import hashlib
import json
import operator
import struct
from collections.abc import Sequence
from typing import Any, Final

from threetears.core.backends.schema_sql import json_default, transport_param

try:
    import pyarrow as pa
    import pyarrow.ipc as pa_ipc

    _PYARROW = True
except ImportError:  # NOSILENT: the snapshot extra is optional; without it a copy is sent as JSON, as before
    _PYARROW = False

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


def _json_text(values: Sequence[Any]) -> Any:
    """a column as each value's JSON text, as the JSON request encodes it; ``None`` as null.

    :param values: the values, already as the rail carries them
    :ptype values: Sequence[Any]
    :return: the column
    :rtype: pyarrow.Array
    """
    return pa.array(
        [None if value is None else json.dumps(value, default=json_default) for value in values], pa.string()
    )


#: the Arrow type a column of one kind of value is sent as
_ARROW_KINDS: Final[dict[type, Any]] = (
    {str: pa.string(), int: pa.int64(), float: pa.float64(), bool: pa.bool_()} if _PYARROW else {}
)


def _arrow_column(values: Sequence[Any]) -> tuple[Any, bool]:
    """one column of a copy's rows as an Arrow array, and whether it is sent as JSON text.

    :param values: the column's values, as the caller gave them
    :ptype values: Sequence[Any]
    :return: the array, and whether its values are their JSON text
    :rtype: tuple[pyarrow.Array, bool]
    """
    kinds = set(map(type, values)) - {_NONE}
    if not kinds <= _ARROW_KINDS.keys() or len(kinds) > 1:
        # converted as the JSON request converts them; a UUID, datetime, Decimal or bytes column is then text
        values = [transport_param(value) for value in values]
        kinds = set(map(type, values)) - {_NONE}
    kind = next(iter(kinds)) if len(kinds) == 1 else str if not kinds else None
    array: Any = None
    if kind in _ARROW_KINDS:
        try:
            array = pa.array(values, _ARROW_KINDS[kind])
        except OverflowError, UnicodeEncodeError, pa.ArrowException:
            # NOSILENT: an integer past 64 bits, a string UTF-8 cannot hold: sent as its JSON text instead
            array = None
    return (array, False) if array is not None else (_json_text(values), True)


def copy_rows_arrow(columns: Sequence[str], records: Sequence[Sequence[Any]]) -> ArrowCopyRows:
    """a copy's rows as an Arrow IPC stream, one column per copied column, named as they are.

    Blocking, and CPU for as long as the rows are many: run it on a thread, as the JSON request is encoded.

    :param columns: the columns each row supplies, in order
    :ptype columns: Sequence[str]
    :param records: the rows' values, each in the order of ``columns``
    :ptype records: Sequence[Sequence[Any]]
    :return: the stream, its digest and its JSON columns
    :rtype: ArrowCopyRows
    :raises ImportError: without pyarrow
    :raises ValueError: when a row is not as wide as ``columns``
    """
    if not _PYARROW:
        raise ImportError("a copy's rows as Arrow need pyarrow: install 3tears[snapshot]")
    width = len(columns)
    for record in records:
        if len(record) != width:
            raise ValueError(f"a row has {len(record)} values for {width} columns")
    arrays = []
    json_columns = []
    for position in range(width):
        # one column at a time, never the whole transposition in one call: each call holds the interpreter lock
        # for its length, and the caller's event loop runs between them
        values = list(map(operator.itemgetter(position), records))
        array, as_json = _arrow_column(values)
        arrays.append(array)
        if as_json:
            json_columns.append(position)
    table = pa.Table.from_arrays(arrays, names=list(columns))
    sink = pa.BufferOutputStream()
    with pa_ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    ipc = sink.getvalue().to_pybytes()
    return ArrowCopyRows(ipc, rows_digest(ipc), json_columns)


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
