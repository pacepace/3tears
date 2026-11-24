"""A delta file read back as Python changes, for a test that reads a small file back and compares its rows.

No serving path reads a delta file so: the copy and the fold apply a file from its Arrow
(:func:`~threetears.core.collections.delta_files.read_delta_table`,
:class:`~threetears.core.collections.delta_files.HeldDeltaRows`), since a row as a Python dict costs about ten times its
Arrow (291 MB for a 200,000-row file of 15 text columns, 2026-11-23 audit). Here, where only tests import it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Mapping

from threetears.core.collections.delta_files import DeltaChange, read_delta_table

__all__ = ["apply_delta_row", "read_delta_file"]


@dataclass(frozen=True, slots=True)
class _DeltaFile:
    """a delta file read back as Python changes (:func:`read_delta_file`): the table it holds rows of, the write that
    saved it, and its changes in order.

    :ivar table: the table
    :ivar write: the write's unique id
    :ivar key: the table's key columns, in key order
    :ivar changes: every row and delete marker, in the order written
    :ivar prefixes: the key prefixes the write cleared before them, as text
    """

    table: str
    write: str
    key: tuple[str, ...]
    changes: list[DeltaChange]
    prefixes: tuple[dict[str, str], ...] = ()


def read_delta_file(path: Path, *, table: str | None = None) -> _DeltaFile:
    """read a delta file saved by :func:`~threetears.core.collections.delta_files.write_delta_file` as Python changes,
    every row a dict. Blocking.

    :param path: the file
    :ptype path: Path
    :param table: the table whose file it must be; None checks none
    :ptype table: str | None
    :return: the file's table, write, key and changes
    :rtype: _DeltaFile
    :raises FileNotFoundError: when it is not there
    :raises DeltaFileUnreadableError: when it is there but not a delta file of this format, of ``table`` and of the
        write its name says
    :raises OSError: when the volume cannot read it (an errno)
    :raises ImportError: without pyarrow
    """
    read = read_delta_table(path, table=table)
    changes = [
        DeltaChange(row=row, deleted=deleted, written=read.sets[written])
        for row, deleted, written in zip(
            read.rows.to_pylist(), read.deleted.to_pylist(), read.written.to_pylist(), strict=True
        )
    ]
    return _DeltaFile(table=read.table, write=read.write, key=read.key, changes=changes, prefixes=read.prefixes)


def apply_delta_row(held: Mapping[str, Any] | None, change: DeltaChange) -> dict[str, Any] | None:
    """the row L3 holds after a delta file's change, from the row it held before: merged as its upsert merged it.

    A delete leaves no row. A row L3 did not hold is the row as the write inserted it, every column as the file shows
    it (a column it left out is its fixed database default, or null). A row L3 held keeps every column the write did not set, its key and its ``date_created``
    (immutable, never set by an upsert) among them, and takes the file's value of each column the write set
    (:func:`~threetears.core.backends.schema_sql.upsert_set_columns`); a write that set none (a table that ignores
    conflicts) leaves it as it was. Pure: the rule the copy and the fold apply a file by, in Arrow
    (:class:`~threetears.core.collections.delta_files.HeldDeltaRows`), here a row at a time as the tests' reference.

    :param held: the row held before the change, keyed by column; None when there was none
    :ptype held: Mapping[str, Any] | None
    :param change: the file's change (a delta file's change, :class:`~threetears.core.collections.delta_files.DeltaChange`)
    :ptype change: DeltaChange
    :return: the row held after it, keyed by column; None when the change deleted it
    :rtype: dict[str, Any] | None
    """
    if change.deleted:
        return None
    if held is None:
        return dict(change.row)
    merged = dict(held)
    for name in change.written:
        merged[name] = change.row[name]
    return merged
