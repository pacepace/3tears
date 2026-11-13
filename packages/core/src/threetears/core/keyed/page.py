"""one page of a listing, by key: a bound prefix, a sort, an opaque cursor and a ``LIMIT``.

The one listing pager. A page reads::

    SELECT <columns> FROM <table> AS t
     WHERE <by: the bound prefix, the tenant>
       AND <past the cursor, in the sort's own direction and null order>
       AND <visible: a keyed EXISTS, the caller's grants>
     ORDER BY <sort columns> LIMIT <limit + 1>

and answers the page with the cursor of its last row when more follow. It reads about a page of rows only when an
index answers the sort after the bound prefix: ``(<by columns>, <sort columns>)``, the sort's last column unique
among the rows ``by`` names (the key, as the tiebreak). On YugabyteDB that index must be range (``ASC``) or bucketed
on the prefix: a hashed one reads every tablet for each page.

**The sort vocabulary is closed.** A caller maps its own closed set of fields to :class:`Sort` entries it writes; no
caller string reaches an ``ORDER BY``. Each sort column says its SQL type, so the cursor's text is cast back in SQL
(``$n::text::timestamptz``), and whether it is nullable, so a page past a ``NULL`` lands in the right partition
(``ASC NULLS LAST``, ``DESC NULLS FIRST``, PostgreSQL's own defaults, stated).

**How a page steps past its cursor.** A sort of one direction with no nullable column, on PostgreSQL (or an
in-process DuckDB or SQLite): the row comparison ``(c1, c2) > ($1, $2)``, which PostgreSQL seeks its index by.
On YugabyteDB (no row lock): a ``UNION ALL`` of legs, each an equality prefix of the cursor and one range
(``c1 = $1 AND c2 > $2``, then ``c1 > $1``), so each leg is one range of the index with nothing left to filter and
YugabyteDB pushes its ``LIMIT`` to storage: the one-condition forms (the expansion's ``OR``, the row comparison's
recheck) read a whole fetch, 1,024 rows, for a page of 9. Otherwise the lexicographic expansion over the sort
columns, each in its own direction (``c1 > $1 OR (c1 = $1 AND c2 > $2)``), led by a range on the first column when
it is not nullable (``c1 >= $1``); PostgreSQL plans the expansion as a bitmap scan and a sort. Which database it is comes
with no round trip (:func:`~threetears.core.backends.dialect.known_on_yugabyte`).

**The cursor is opaque**: base64 of the last row's sort values as text, each with a null flag. A cursor that does not
decode, or names another sort, is refused (:class:`~threetears.core.pagination.CursorError`), never read as "start".

IO only: one statement a page, on the caller's connection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from threetears.core.backends.dialect import known_on_yugabyte
from threetears.core.keyed._filters import Derived, Fragment, Lock, Selected, sent_by
from threetears.core.keyed.counts import SQL_DIALECT_ATTRIBUTE, Dialect
from threetears.core.keyed.inline import InlineDuckDB
from threetears.core.keyed.threaded import InlineSQLite, ThreadedDuckDB, ThreadedSQLite
from threetears.interfaces.keyed.keys import refuse_non_unique_order
from threetears.interfaces.keyed.page import (
    KeyPage,
    Sort,
    check_page,
    cursor_values,
    exists_keyed,
    keyset_cursor,
    next_past,
    page_statement,
    seeks_by_row,
)

__all__ = ["Sort", "KeyPage", "copy_keyed", "exists_keyed", "keyset_cursor", "keyset_page", "keyset_rows"]


async def keyset_page(
    conn: Any,
    table: str | Derived,
    *,
    columns: Sequence[str | Selected],
    by: Mapping[str, Any],
    order: Sequence[Sort],
    limit: int,
    cursor: str | None = None,
    visible: Sequence[Fragment] = (),
    where: Sequence[Fragment] = (),
    lock: Lock | None = None,
) -> KeyPage:
    """one page of ``table``'s rows ``by`` names, in ``order``, after ``cursor``.

    :param conn: anything with an asyncpg-shaped ``fetch(query, *params)``
    :ptype conn: Any
    :param table: the table (``schema.table``), or a derived relation
    :ptype table: str | Derived
    :param columns: the columns read, TRUSTED identifiers or :class:`~threetears.core.keyed.Selected` expressions
        under their own names (a row's hash); each sort column among them (a sort expression's name not: it is
        selected under it)
    :ptype columns: Sequence[str | Selected]
    :param by: the bound prefix (a tenant, a parent), each a value or an ``AnyOf``; empty pages across every row
        in the sort's index
    :ptype by: Mapping[str, Any]
    :param order: the sort, its last column unique among the rows ``by`` names (the key as the tiebreak)
    :ptype order: Sequence[Sort]
    :param limit: the rows a page keeps
    :ptype limit: int
    :param cursor: the cursor a previous page answered; the first page when None
    :ptype cursor: str | None
    :param visible: visibility clauses each row must pass (:func:`exists_keyed`, an ACL's caller clause)
    :ptype visible: Sequence[Fragment]
    :param where: filters beside the prefix, each a fragment written as ``t."col"`` from ``$1``: a range on the sort's
        lead (``t."due_at" <= $1``), or a partial index's own predicate (``t."due_at" IS NOT NULL``), so the page
        reads through that index
    :ptype where: Sequence[Fragment]
    :param lock: the row lock the page takes on its rows (the caller's transaction); none when None
    :ptype lock: Lock | None
    :return: the page
    :rtype: KeyPage
    :raises ValueError: when the sort is empty, a sort column is not read, a sort expression's name is a column read,
        ``limit`` is under one, or a page past a cursor is in an order that with ``by`` holds none of the table's
        registered keys (:func:`~threetears.interfaces.keyed.keys.refuse_non_unique_order`)
    :raises CursorError: when ``cursor`` is not one this sort issued
    """
    check_page(table, columns, order, limit)
    if cursor is not None:
        # a page past a cursor skips every row sharing its last row's order values unless the order holds a key
        refuse_non_unique_order(table, by, order)
    past = None if cursor is None else cursor_values(order, cursor)
    by_row = past is not None and await _steps_by_row(conn, order)
    legs = past is not None and lock is None and await _seeks_by_legs(conn, order)
    sql, params = page_statement(
        table,
        columns,
        by,
        order,
        limit + 1,
        past,
        by_row,
        visible,
        where,
        lock,
        seek_legs=legs,
        chain_legs=_chains_legs(conn),
    )
    return KeyPage.cut(order, [dict(row) for row in await sent_by(conn).fetch(sql, *params)], limit)


async def _steps_by_row(conn: Any, order: Sequence[Sort]) -> bool:
    """whether a page past a place in ``order`` steps by the row comparison: a sort :func:`~threetears.interfaces.keyed.page.seeks_by_row`, on a
    database other than YugabyteDB (an in-process DuckDB or SQLite is never it; an asyncpg connection says so with
    no round trip, :func:`~threetears.core.backends.dialect.known_on_yugabyte`)."""
    if not seeks_by_row(order):
        return False
    if isinstance(conn, ThreadedDuckDB | ThreadedSQLite | InlineSQLite | InlineDuckDB):
        return True
    # a reader naming its dialect (a datasource's engine) is not asked: a warehouse (Redshift) is read by the
    # expansion, plain SQL every engine takes
    named = getattr(conn, SQL_DIALECT_ATTRIBUTE, None)
    if isinstance(named, Dialect):
        return named in {Dialect.POSTGRES, Dialect.DUCKDB}
    return not await known_on_yugabyte(conn)


async def _seeks_by_legs(conn: Any, order: Sequence[Sort]) -> bool:
    """whether a page past a place in ``order`` steps by legs each one range of its index (:func:`page_statement`'s
    ``seek_legs``): on YugabyteDB, which pushes a page's ``LIMIT`` to storage only with nothing left to filter. An
    in-process DuckDB or SQLite, or a reader naming its dialect (a warehouse), never is; an asyncpg connection says
    so with no round trip (:func:`~threetears.core.backends.dialect.known_on_yugabyte`). Nothing is asked where the
    answer changes no statement: a sort of one column never NULL is one leg either way, and a sort led by a NULLable
    column steps by legs on every engine."""
    if order[0].nullable or len(order) == 1:
        return False
    if isinstance(conn, ThreadedDuckDB | ThreadedSQLite | InlineSQLite | InlineDuckDB):
        return False
    if isinstance(getattr(conn, SQL_DIALECT_ATTRIBUTE, None), Dialect):
        return False
    return await known_on_yugabyte(conn)


def _chains_legs(conn: Any) -> bool:
    """whether a page's legs each ask only what the legs before it left short (:func:`page_statement`'s
    ``chain_legs``, a ``LIMIT`` read from a subquery): every engine but a reader naming a warehouse's dialect, which
    may not take one."""
    named = getattr(conn, SQL_DIALECT_ATTRIBUTE, None)
    return not isinstance(named, Dialect) or named in {Dialect.POSTGRES, Dialect.DUCKDB, Dialect.YUGABYTE}


async def keyset_rows(
    conn: Any,
    table: str | Derived,
    *,
    columns: Sequence[str | Selected],
    by: Mapping[str, Any],
    order: Sequence[Sort],
    page: int = 500,
    visible: Sequence[Fragment] = (),
    where: Sequence[Fragment] = (),
) -> list[dict[str, Any]]:
    """every row of ``table`` that ``by`` names, in ``order``: :func:`keyset_page`'s statement read page after page
    to the last.

    For a listing whose caller takes every row (a conversation's schedules, a group's members): each statement reads
    one page through the sort's index, never the prefix's rows in one statement past the transport's cap. Each asks
    ``page + 1`` rows and keeps every one: a full answer says more follow, and the next statement starts past its
    last row, its sort values bound straight from that row (no cursor text between them; an uncast sort column,
    ``Sort(cast=None)``, binds the row's own value, so the database types it by the column). A listing of one page is
    one statement. Past a place, PostgreSQL steps by the row comparison, YugabyteDB by the leading bound and the
    expansion (:func:`~threetears.interfaces.keyed.page.page_statement`), as :func:`keyset_page` does.

    :param conn: anything with an asyncpg-shaped ``fetch(query, *params)``
    :ptype conn: Any
    :param table: the table (``schema.table``), or a derived relation
    :ptype table: str | Derived
    :param columns: the columns read, TRUSTED identifiers or :class:`~threetears.core.keyed.Selected` expressions
        under their own names (a row's hash); each sort column among them (a sort expression's name not: it is
        selected under it)
    :ptype columns: Sequence[str | Selected]
    :param by: the bound prefix, as :func:`keyset_page` takes it
    :ptype by: Mapping[str, Any]
    :param order: the sort, its last column unique among the rows ``by`` names
    :ptype order: Sequence[Sort]
    :param page: the rows a statement reads past the one that says more follow (it asks ``page + 1``)
    :ptype page: int
    :param visible: visibility clauses, as :func:`keyset_page` takes them
    :ptype visible: Sequence[Fragment]
    :param where: filters beside the prefix, as :func:`keyset_page` takes them
    :ptype where: Sequence[Fragment]
    :return: the rows, in the sort's order
    :rtype: list[dict[str, Any]]
    :raises ValueError: as :func:`keyset_page`
    """
    check_page(table, columns, order, page)
    # every statement after the first starts past the last row read: the order must hold a key
    refuse_non_unique_order(table, by, order)
    rows: list[dict[str, Any]] = []
    past: list[Any] | None = None
    by_row: bool | None = None
    legs = False
    while True:
        sql, params = page_statement(
            table,
            columns,
            by,
            order,
            page + 1,
            past,
            bool(by_row),
            visible,
            where,
            None,
            seek_legs=legs,
            chain_legs=_chains_legs(conn),
        )
        answered = await sent_by(conn).fetch(sql, *params)
        rows.extend(dict(row) for row in answered)
        if len(answered) <= page:
            return rows
        if by_row is None:
            by_row = await _steps_by_row(conn, order)
            legs = await _seeks_by_legs(conn, order)
        last = rows[-1]
        past = next_past(order, last)


async def copy_keyed(
    conn: Any,
    table: str,
    *,
    columns: Sequence[str | Selected],
    by: Mapping[str, Any],
    order: Sequence[Sort],
    where: Sequence[Fragment] = (),
    force_quote: Sequence[str] = (),
) -> bytes:
    """every row of ``table`` that ``by`` names, in ``order``, as PostgreSQL ``COPY`` CSV text (no header): the read
    :func:`keyset_rows` makes, streamed as the server writes it, never a row decoded here.

    For a pass that parses the text where it judges it (``pyarrow.csv`` in a worker process). A ``NULL`` is an
    unquoted empty field; ``force_quote`` columns are quoted always, so their empty strings stay apart from ``NULL``.
    asyncpg only (``copy_from_query``).

    :param conn: an asyncpg connection or pool
    :ptype conn: Any
    :param table: the table, a TRUSTED identifier
    :ptype table: str
    :param columns: the columns, TRUSTED identifiers, or :class:`~threetears.core.keyed.Selected` expressions under
        their names (a time as its UTC text)
    :ptype columns: Sequence[str | Selected]
    :param by: the bound prefix, as :func:`keyset_page` takes it; every row when empty (a security backstop's read)
    :ptype by: Mapping[str, Any]
    :param order: the order, as :func:`keyset_page` takes it
    :ptype order: Sequence[Sort]
    :param where: filters beside the prefix, as :func:`keyset_page` takes them
    :ptype where: Sequence[Fragment]
    :param force_quote: the columns (by their names) always quoted
    :ptype force_quote: Sequence[str]
    :return: the text, a line per row
    :rtype: bytes
    :raises ValueError: as :func:`keyset_page`
    """
    # no cursor is made from the rows, so a sort column need not be read (a time sorted by, read as its text)
    check_page(table, [*columns, *(sort.column for sort in order if sort.expression is None)], order, 1)
    sql, params = page_statement(table, columns, by, order, 1, None, False, (), where, None)
    sql = sql.removesuffix(" LIMIT 1")
    chunks: list[bytes] = []

    async def sink(chunk: bytes) -> None:
        chunks.append(chunk)

    options: dict[str, Any] = {"force_quote": list(force_quote)} if force_quote else {}
    await conn.copy_from_query(sql, *params, output=sink, format="csv", **options)
    return b"".join(chunks)
