"""A delta file read back as Python changes, for a test that reads a small file back and compares its rows.

No serving path reads a delta file so: the copy and the fold apply a file from its Arrow
(:func:`~threetears.core.collections.delta_files.read_delta_table`,
:class:`~threetears.core.collections.delta_files.HeldDeltaRows`), since a row as a Python dict costs about ten times its
Arrow (291 MB for a 200,000-row file of 15 text columns, 2026-11-23 audit). Here, where only tests import it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from threetears.core.collections.delta_files import DeltaChange, read_delta_table

__all__ = ["read_delta_file"]


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
