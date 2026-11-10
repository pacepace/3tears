"""fill a test's DuckDB L1 table with exactly the rows it gives: a fixture, never a load path.

A pod's L1 changes by key (an upsert, a scope's replacement, a follower's staged change); nothing it runs empties a
table and fills it again. A test, though, often wants a table to hold exactly some rows, whatever it held before.
:func:`fill_table` does that in one transaction, so a row that fails to insert leaves the table as it was.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from threetears.core.cache.duckdb import DuckDBBackend
from threetears.core.sql_fragments import quote_identifier

__all__ = ["fill_table"]


def fill_table(
    backend: DuckDBBackend, table: str, rows: Sequence[Mapping[str, Any]], primary_key: str | tuple[str, ...]
) -> int:
    """make ``table`` hold exactly ``rows``, in one transaction; rows repeating a key keep the last.

    :param backend: the backend
    :ptype backend: DuckDBBackend
    :param table: a table the backend created
    :ptype table: str
    :param rows: every row the table is to hold, each keyed by column name
    :ptype rows: Sequence[Mapping[str, Any]]
    :param primary_key: pk column name or tuple of pk column names
    :ptype primary_key: str | tuple[str, ...]
    :return: how many rows were given
    :rtype: int
    :raises ValueError: when the table is not one the backend created
    """
    # the backend's own lock, connection and insert: one transaction, as no public write of it empties a table
    if table not in backend._schema_info:  # noqa: SLF001
        raise ValueError(f"unknown table {table!r}: fill_table fills tables the backend created")
    connection = backend._db  # noqa: SLF001
    with backend._db_lock:  # noqa: SLF001
        committed = False
        connection.execute("BEGIN TRANSACTION")
        try:
            connection.execute(f"DELETE FROM {quote_identifier(table)}")  # noqa: S608 - a test's own table
            backend._insert_rows(connection, table, rows, primary_key)  # noqa: SLF001
            connection.execute("COMMIT")
            committed = True
        finally:
            if not committed:
                connection.execute("ROLLBACK")
    backend.request_checkpoint()
    return len(rows)
