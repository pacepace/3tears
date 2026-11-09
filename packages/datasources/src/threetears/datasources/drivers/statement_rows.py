"""the rows one statement wrote, as the warehouse itself records it: never a count of the table it wrote.

A caller that needs the rows a ``CREATE TABLE AS`` or an ``INSERT`` wrote (the dataset executor's inventory and
completion marker) takes them from the statement: :meth:`Driver.execute` answers them. Each engine reports them its
own way:

- Postgres (asyncpg): the command tag, ``INSERT 0 5`` or ``SELECT 120`` for a ``CREATE TABLE AS``
  (:func:`rows_from_command_tag`);
- Redshift: the cursor's ``rowcount`` where its driver reports one, else the engine's own log of the session's last
  query, ``stl_insert`` (:data:`LAST_QUERY_INSERTED_SQL`), read on the same session before anything else runs on it.
"""

from __future__ import annotations

import re
from typing import Final

__all__ = ["LAST_QUERY_INSERTED_SQL", "rows_from_command_tag", "writes_rows"]

#: Redshift's own record of the rows the session's last query inserted, every slice's
LAST_QUERY_INSERTED_SQL: Final[str] = "SELECT sum(rows) FROM stl_insert WHERE query = pg_last_query_id()"

#: a statement that writes rows to a table: an insert, or a table created from a query
_WRITES_ROWS: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?:INSERT\b|CREATE\s+(?:(?:LOCAL\s+)?TEMP(?:ORARY)?\s+)?TABLE\b.*\bAS\b)", re.IGNORECASE | re.DOTALL
)

#: the command tags whose last word is the rows the statement wrote
_COUNTED_TAGS: Final[frozenset[str]] = frozenset({"INSERT", "UPDATE", "DELETE", "SELECT", "MERGE", "COPY"})


def rows_from_command_tag(status: object) -> int | None:
    """the rows a Postgres command tag reports.

    :param status: what ``asyncpg.Connection.execute`` answered (``INSERT 0 5``, ``SELECT 120``, ``CREATE TABLE``)
    :ptype status: object
    :return: the rows, or ``None`` for a tag that reports none
    :rtype: int | None
    """
    words = status.split() if isinstance(status, str) else []
    counted = bool(words) and words[0].upper() in _COUNTED_TAGS and words[-1].isdigit()
    return int(words[-1]) if counted else None


def writes_rows(sql: str) -> bool:
    """whether a statement writes rows to a table: an ``INSERT``, or a ``CREATE TABLE ... AS``.

    :param sql: the statement
    :ptype sql: str
    :return: whether it does
    :rtype: bool
    """
    return _WRITES_ROWS.match(sql) is not None
