"""fill a test's DuckDB L1 table with exactly the rows it gives: a fixture, never a load path.

A pod's L1 changes by key (an upsert, a scope's replacement, a follower's staged change); nothing it runs empties a
table and fills it again. A test, though, often wants a table to hold exactly some rows, whatever it held before.
:func:`fill_table` does that through the backend's front door: the rows go in first (one write, so a row that fails
to insert leaves the table as it was), then every key it held and was not given goes, through the backend's own
delete (:meth:`~threetears.core.cache.duckdb.DuckDBBackend.delete_many_by_id`), so a backend that keeps something in
step with each write (a subclass renewing what it takes from a table's rows) keeps it in step for these too.
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
    """make ``table`` hold exactly ``rows``; rows repeating a key keep the last.

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
    types = backend.column_types(table)
    if not types:
        raise ValueError(f"unknown table {table!r}: fill_table fills tables the backend created")
    key = (primary_key,) if isinstance(primary_key, str) else tuple(primary_key)
    if rows:
        backend.upsert_many(table, rows, key)
    given = {tuple(backend.serialize_value(row[column], types[column]) for column in key) for row in rows}
    columns = ", ".join(quote_identifier(column) for column in key)
    held = backend.execute_query(f"SELECT {columns} FROM {quote_identifier(table)}")  # noqa: S608 - a test's own table
    gone = sorted({tuple(row[column] for column in key) for row in held} - given, key=repr)
    if gone:
        # through the backend's own delete, so a backend that keeps something in step with each delete stays in step
        backend.delete_many_by_id(table, gone, key)
    return len(rows)
