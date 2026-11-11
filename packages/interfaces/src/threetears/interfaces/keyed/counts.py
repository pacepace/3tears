"""keyed counts' vocabulary: the dialects a statement is spelled for, what a bounded count answers, and a statement's
row count read off its own status. Pure: the counts themselves run in ``threetears.core.keyed``."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

__all__ = ["NO_MARK_FLOOR", "SQL_DIALECT_ATTRIBUTE", "BoundedCount", "Dialect", "MarkedCount", "rows_affected"]

#: the verbs whose status ends in the rows they touched (``INSERT`` carries an oid before it)
_COUNTED_VERBS: Final[frozenset[str]] = frozenset(
    {"INSERT", "UPDATE", "DELETE", "SELECT", "COPY", "MERGE", "MOVE", "FETCH"}
)


#: the attribute a reader names its database's dialect by (a datasource reader: its datasource's engine), a
#: :class:`Dialect`; a reader naming none is asked (:func:`threetears.core.keyed.dialect_of`)
SQL_DIALECT_ATTRIBUTE: Final = "sql_dialect"


class Dialect(StrEnum):
    """the database a statistics read is spelled for. PostgreSQL and YugabyteDB keep the same catalog."""

    POSTGRES = "postgres"
    YUGABYTE = "yugabyte"
    DUCKDB = "duckdb"
    #: a warehouse read through a datasource: no arrays, its own catalog (no estimate is read from it)
    REDSHIFT = "redshift"


def rows_affected(status: str) -> int:
    """the rows a statement touched, read off its own status (asyncpg's ``execute`` answer).

    :param status: ``UPDATE 3``, ``DELETE 0``, ``INSERT 0 5``, ...
    :ptype status: str
    :return: the rows
    :rtype: int
    :raises ValueError: when the status names no row count (``CREATE TABLE``)
    """
    words = status.split()
    if len(words) < 2 or words[0].upper() not in _COUNTED_VERBS or not words[-1].isdigit():
        raise ValueError(f"the status {status!r} names no row count")
    return int(words[-1])


@dataclass(frozen=True)
class BoundedCount:
    """a count of at most a bound's rows.

    :param count: the rows counted, never more than the bound
    :ptype count: int
    :param more: whether rows past the bound are held
    :ptype more: bool
    """

    count: int
    more: bool


#: the stamp a side with no mark yet is held past: earlier than any stamp, so every stamped row is past it
NO_MARK_FLOOR: Final = "0001-01-01 00:00:00"


@dataclass(frozen=True)
class MarkedCount:
    """one group of a relation's rows past its marks (:func:`counts_by_relations` with ``since``).

    :param value: the value of ``column`` (text when packed)
    :ptype value: Any
    :param group: the value of the relation's ``group`` column, as text; None for a NULL group or no group column
    :ptype group: str | None
    :param rows: the group's rows past the marks
    :ptype rows: int
    :param newest: each ``max_of`` column -> its newest value among those rows, as text (None when every one is NULL)
    :ptype newest: dict[str, str | None]
    :param fresh: whether a row of the group is past the ``fresh`` stamp's mark
    :ptype fresh: bool
    """

    value: Any
    group: str | None
    rows: int
    newest: dict[str, str | None]
    fresh: bool
