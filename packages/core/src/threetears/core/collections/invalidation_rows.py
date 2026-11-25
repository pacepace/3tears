"""the rows of a write of many rows as one invalidation broadcast carries them (``CacheInvalidationMessage.rows``).

Each key column goes on the wire as Arrow lays out a text column: every value's UTF-8 bytes end to end, and where each
ends (little-endian int32 offsets, the first 0). Why: a receiver with pyarrow (a copy) makes the column Arrow from those
two buffers, with no Python object a key between the wire and the batch it hands the copy (decision 35; keys are Arrow
end to end), and one without pyarrow (an agent pod) still reads it with the standard library, so the wire needs no
pyarrow. Before this, a write of many rows went as one message a row, each decoded as a pydantic message on the loop
and turned into Arrow one at a time by the copy: ~0.8 ms of the loop per 2,000 messages, 80 ms per 200,000
(2026-11-25).

A write's rows go as few messages as :data:`_BYTES_A_MESSAGE` allows (:func:`invalidation_batches`), usually one.
"""

from __future__ import annotations

import sys
from array import array
from collections.abc import Mapping, Sequence
from itertools import accumulate
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, model_validator

from threetears.core.collections.delta_files import entity_key_batch
from threetears.core.collections.epoch_change import key_text

try:
    import pyarrow as pa

    _PYARROW = True
except ImportError:  # NOSILENT: the snapshot extra is optional; the wire is read with the standard library without it
    _PYARROW = False

__all__ = [
    "InvalidatedRows",
    "invalidation_batches",
    "key_batch",
    "row_ids",
]

#: the most bytes of packed values (keys and their offsets, the invalidation columns' text) one message carries. Base64
#: makes it a third bigger on the wire: ~700 KB, under NATS' default 1 MB payload. A page of 5,000 keys of two text
#: columns of ~20 bytes each is ~240 KB, so a page goes as one message
_BYTES_A_MESSAGE: Final = 512 * 1024

#: the offsets' Python array type code: a 4-byte signed int, Arrow's ``string`` offsets
_OFFSET: Final = "i"

#: the wire is little-endian; the buffers are Arrow's own only on a little-endian machine
_LITTLE: Final = sys.byteorder == "little"


class _PackedTexts(BaseModel):
    """one column of text values, packed as Arrow lays out a ``string`` column.

    :ivar data: every value's UTF-8 bytes, end to end
    :ivar offsets: where each value starts and the last ends, in ``data``: one more little-endian int32 than values,
        the first 0
    """

    model_config = ConfigDict(frozen=True, ser_json_bytes="base64", val_json_bytes="base64")

    data: bytes
    offsets: bytes


class InvalidatedRows(BaseModel):
    """the rows a batched :class:`~threetears.core.collections.registry.CacheInvalidationMessage` names.

    :ivar count: how many rows
    :ivar keys: one packed column a key column, in declared key order, each value a key value as text
        (:func:`~threetears.core.collections.epoch_change.key_text`)
    :ivar columns: each row's declared invalidation columns, as a one-row message names them
        (``CacheInvalidationMessage.columns``: ``None`` for a row the write saw none of); ``None`` when the collection
        declares none. A dict a row, as the one-row message had: only a collection that declares the columns sends them,
        and a copy reads none of them
    """

    model_config = ConfigDict(frozen=True)

    count: int
    keys: list[_PackedTexts]
    columns: list[dict[str, str | None] | None] | None = None

    @model_validator(mode="after")
    def _every_column_names_every_row(self) -> InvalidatedRows:
        """refuse rows whose columns name another number of rows than :attr:`count`.

        :return: the rows
        :rtype: InvalidatedRows
        :raises ValueError: when a column is short or long
        """
        if self.count < 1:
            raise ValueError("a batched invalidation names at least one row")
        for packed in self.keys:
            if len(packed.offsets) != 4 * (self.count + 1):
                raise ValueError(f"a key column's offsets name {len(packed.offsets) // 4 - 1} rows, not {self.count}")
        if self.columns is not None and len(self.columns) != self.count:
            raise ValueError(f"the invalidation columns name {len(self.columns)} rows, not {self.count}")
        return self


def _pack(values: Sequence[str]) -> _PackedTexts:
    """values packed with the standard library.

    :param values: the values
    :ptype values: Sequence[str]
    :return: the packed column
    :rtype: _PackedTexts
    """
    parts = [value.encode() for value in values]
    offsets = array(_OFFSET, [0])
    offsets.extend(accumulate(map(len, parts)))
    if not _LITTLE:
        offsets.byteswap()
    return _PackedTexts(data=b"".join(parts), offsets=offsets.tobytes())


def _pack_arrow(column: Any) -> _PackedTexts:
    """an Arrow text column packed from its own buffers: no Python object a value.

    :param column: the values
    :ptype column: pyarrow.Array | pyarrow.ChunkedArray
    :return: the packed column
    :rtype: _PackedTexts
    """
    chunks = column.chunks if isinstance(column, pa.ChunkedArray) else [column]
    # one array at offset 0 (a slice keeps its parent's buffers): concat lays it out afresh
    whole = pa.concat_arrays(chunks) if chunks else pa.array([], pa.string())
    _, offsets, data = whole.buffers()
    count = len(whole)
    ends = memoryview(offsets)[: 4 * (count + 1)] if offsets is not None else b"\0\0\0\0"
    used = int.from_bytes(bytes(ends[-4:]), "little")
    return _PackedTexts(data=bytes(memoryview(data)[:used]) if data is not None else b"", offsets=bytes(ends))


def _texts(packed: _PackedTexts, count: int) -> list[str]:
    """a packed column's values, read with the standard library.

    :param packed: the column
    :ptype packed: _PackedTexts
    :param count: how many values
    :ptype count: int
    :return: the values
    :rtype: list[str]
    """
    offsets = array(_OFFSET)
    offsets.frombytes(packed.offsets)
    if not _LITTLE:
        offsets.byteswap()
    data = packed.data
    return [data[offsets[at] : offsets[at + 1]].decode() for at in range(count)]


def key_batch(rows: InvalidatedRows) -> Any:
    """the rows' keys as one Arrow batch, columns named by position (``k0``, ``k1``, ...) as
    :func:`~threetears.core.collections.delta_files.text_key_batch` names them: made from the wire's buffers, with no
    Python object a key. What the registry hands a cache that takes key batches (a copy, ``on_keys``).

    :param rows: the rows
    :ptype rows: InvalidatedRows
    :return: the batch
    :rtype: pyarrow.Table
    :raises ImportError: without pyarrow
    :raises pyarrow.ArrowInvalid: when a column's offsets or text are malformed
    """
    if not _PYARROW:
        raise ImportError("a key batch needs pyarrow: install 3tears[snapshot]")
    columns: dict[str, Any] = {}
    for at, packed in enumerate(rows.keys):
        if _LITTLE:
            column = pa.Array.from_buffers(
                pa.string(), rows.count, [None, pa.py_buffer(packed.offsets), pa.py_buffer(packed.data)]
            )
            column.validate(full=True)
        else:  # pragma: no cover - every machine this runs on is little-endian
            column = pa.array(_texts(packed, rows.count), pa.string())
        columns[f"k{at}"] = column
    return pa.table(columns)


def row_ids(rows: InvalidatedRows) -> list[list[str]]:
    """each row's key values as text, in key order: what a row message a row names as ``ids``. Built only for a
    receiver that applies the rows one at a time (a collection's L1 and L2, a cache that takes rows).

    :param rows: the rows
    :ptype rows: InvalidatedRows
    :return: one list a row
    :rtype: list[list[str]]
    """
    columns = [_texts(packed, rows.count) for packed in rows.keys]
    return [list(values) for values in zip(*columns, strict=True)] if columns else [[] for _ in range(rows.count)]


def invalidation_batches(
    entity_ids: Sequence[Any], columns: Sequence[Mapping[str, str | None] | None] | None = None
) -> list[InvalidatedRows]:
    """a write's rows as the batches its broadcast carries, each at most :data:`_BYTES_A_MESSAGE` of packed values
    (one row past it is still one message). Pure: a thread may make it.

    A key is spelled as a row message spells it (:func:`~threetears.core.collections.epoch_change.key_text`): by Arrow
    from the writer's values (:func:`~threetears.core.collections.delta_files.entity_key_batch`) where there is pyarrow,
    a value at a time where there is none (an agent pod, whose writes are few rows).

    :param entity_ids: the keys, at least one: pk values (single-pk) or tuples of pk values in declared order
    :ptype entity_ids: Sequence[Any]
    :param columns: each row's declared invalidation columns, in the order of ``entity_ids``; ``None`` when the
        collection declares none
    :ptype columns: Sequence[Mapping[str, str | None] | None] | None
    :return: the batches, in order
    :rtype: list[InvalidatedRows]
    """
    count = len(entity_ids)
    if not count:
        return []
    keys: list[Any]
    if _PYARROW:
        table = entity_key_batch(entity_ids)
        keys = [table.column(at) for at in range(table.num_columns)]
        size = sum(column.nbytes for column in keys)
    else:
        values = list(zip(*entity_ids, strict=True)) if isinstance(entity_ids[0], tuple) else [entity_ids]
        keys = [[key_text(value) for value in column] for column in values]
        size = sum(len(text) + 4 for column in keys for text in column)
    if columns is not None:
        size += sum(len(name) + len(text or "") + 8 for row in columns for name, text in (row or {}).items())
    pieces = max(1, -(-size // _BYTES_A_MESSAGE))
    step = -(-count // pieces)
    batches: list[InvalidatedRows] = []
    for start in range(0, count, step):
        stop = min(start + step, count)
        packed = [
            _pack_arrow(column.slice(start, stop - start)) if _PYARROW else _pack(column[start:stop]) for column in keys
        ]
        batches.append(
            InvalidatedRows(
                count=stop - start,
                keys=packed,
                columns=None
                if columns is None
                else [None if row is None else dict(row) for row in columns[start:stop]],
            )
        )
    return batches
