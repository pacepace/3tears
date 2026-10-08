"""SQL text every tier builds the same way: identifier quoting and equality filters.

The L1 backends (SQLite, DuckDB), the L3 readers (the whole-table copies, scope epochs, the
scoped snapshot) and the datasource drivers all interpolate identifiers into SQL. Each spelling
lives here once, so a table or column name means the same thing on every path that names it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

__all__ = [
    "as_written",
    "equality_conditions",
    "quote_identifier",
]


def quote_identifier(identifier: str) -> str:
    """an identifier quoted for SQL, so a name with spaces, capitals or a keyword survives.

    Every L1 backend quotes every table and column name it interpolates through this one
    function: SQLite and DuckDB quote alike, and a backend that quoted some statements and not
    others would create a table it could then not read. A converted extract's column names
    (``% of Exp. In``) are the case that needs it. Postgres, Yugabyte and Redshift accept the same
    spelling.

    :param identifier: a table or column name
    :ptype identifier: str
    :return: the quoted identifier, any double quote in it doubled
    :rtype: str
    """
    return '"' + identifier.replace('"', '""') + '"'


def as_written(identifier: str) -> str:
    """an identifier interpolated as the caller wrote it, unquoted.

    For a statement whose other identifiers are interpolated unquoted (a schema-qualified
    relation, which is not one identifier), so every name in it folds case alike. The caller
    vouches for it: a TRUSTED, validated plain identifier.

    :param identifier: a plain identifier
    :ptype identifier: str
    :return: the identifier unchanged
    :rtype: str
    """
    return identifier


def equality_conditions(
    where: Mapping[str, Any] | None,
    *,
    first: int = 1,
    quote: Callable[[str], str] = quote_identifier,
) -> tuple[str, list[Any]]:
    """the conditions keeping only ``where``'s rows, placeholders ``$first..``; the values bound.

    :param where: equality filters, column -> value; columns are TRUSTED identifiers
    :ptype where: Mapping[str, Any] | None
    :param first: the first placeholder's number
    :ptype first: int
    :param quote: how each column is spelled: quoted (the default), or :func:`as_written` in a
        statement whose other identifiers are unquoted
    :ptype quote: Callable[[str], str]
    :return: the conditions joined by ``AND`` (empty without filters), and their values in order
    :rtype: tuple[str, list[Any]]
    """
    filters = dict(where or {})
    conditions = " AND ".join(f"{quote(column)} = ${first + index}" for index, column in enumerate(filters))
    return conditions, list(filters.values())
