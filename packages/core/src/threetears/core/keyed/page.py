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
Otherwise the lexicographic expansion over the sort columns, each in its own direction (``c1 > $1 OR (c1 = $1 AND
c2 > $2)``), led by a range on the first column when it is not nullable (``c1 >= $1``): YugabyteDB seeks a hashed
bucket by the lead and filters the rest in storage (:mod:`threetears.core.keyset`), where a row comparison would read
the bucket from its start; PostgreSQL plans the expansion as a bitmap scan and a sort. Which database it is comes
with no round trip (:func:`~threetears.core.backends.dialect.known_on_yugabyte`).

**The cursor is opaque**: base64 of the last row's sort values as text, each with a null flag. A cursor that does not
decode, or names another sort, is refused (:class:`~threetears.core.pagination.CursorError`), never read as "start".

IO only: one statement a page, on the caller's connection.
"""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Any

from threetears.core.backends.dialect import known_on_yugabyte
from threetears.core.keyed.inline import InlineDuckDB
from threetears.core.keyed.threaded import ThreadedDuckDB, ThreadedSQLite
from threetears.core.keyed._filters import (
    AnyOf,
    Derived,
    Fragment,
    Lock,
    by_conditions,
    quoted_table,
    relation,
    renumbered,
)
from threetears.core.pagination import CursorError
from threetears.core.sql_fragments import quote_identifier

__all__ = ["Sort", "KeyPage", "exists_keyed", "keyset_page", "keyset_rows"]


@dataclass(frozen=True)
class Sort:
    """one column of a listing's sort, written by the endpoint, never by its caller.

    :param column: the column, a TRUSTED identifier of the paged table
    :ptype column: str
    :param cast: its SQL type, which the cursor's text is cast back to; None binds the text as it is (SQLite, which
        has no ``::`` casts and compares by the column's own affinity)
    :ptype cast: str | None
    :param descending: whether it sorts descending
    :ptype descending: bool
    :param nullable: whether it can be ``NULL``
    :ptype nullable: bool
    :param expression: a TRUSTED SQL expression of the paged row's columns written as ``t."col"`` (``lower(t."name")``)
        the sort is by, read under the name ``column``: the page selects it ``AS column`` and its cursor carries its
        value; an index on the same expression answers the page (``CREATE INDEX ... (lower(name), id)``). None sorts
        by the column itself
    :ptype expression: str | None
    """

    column: str
    cast: str | None = "text"
    descending: bool = False
    nullable: bool = False
    expression: str | None = None

    @property
    def sorted_by(self) -> str:
        """what the statement orders and compares by: the expression, or the column of the paged row (``t.``)."""
        return self.expression if self.expression is not None else f"t.{quote_identifier(self.column)}"


@dataclass(frozen=True)
class KeyPage:
    """a listing's page.

    :param rows: the page's rows, in the sort's order
    :ptype rows: list[dict[str, Any]]
    :param next_cursor: the cursor of the page after it; None on the last page
    :ptype next_cursor: str | None
    """

    rows: list[dict[str, Any]]
    next_cursor: str | None


def _as_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, datetime | date | time):
        return value.isoformat()
    return str(value)


def _encode(order: Sequence[Sort], row: Mapping[str, Any]) -> str:
    values = [None if row[sort.column] is None else _as_text(row[sort.column]) for sort in order]
    raw = json.dumps([[sort.column for sort in order], values], separators=(",", ":"))
    return base64.urlsafe_b64encode(raw.encode()).decode()


def _decode(order: Sequence[Sort], cursor: str) -> list[str | None]:
    try:
        named, values = json.loads(base64.urlsafe_b64decode(cursor.encode()).decode())
    except (binascii.Error, ValueError, UnicodeError, TypeError) as exc:
        raise CursorError(f"invalid cursor token: {cursor!r}") from exc
    if named != [sort.column for sort in order] or not isinstance(values, list) or len(values) != len(order):
        raise CursorError("the cursor names another sort than this listing's")
    if not all(value is None or isinstance(value, str) for value in values):
        raise CursorError("the cursor's values are not text")
    return values


def _seeks_by_row(order: Sequence[Sort]) -> bool:
    """whether the rows past a place in ``order`` are one row comparison: every column one direction, none NULL."""
    return len(order) > 1 and not any(sort.nullable for sort in order) and len({sort.descending for sort in order}) == 1


def _past(
    order: Sequence[Sort],
    values: Sequence[str | None],
    first: int,
    *,
    by_row: bool = False,
    lead: Sequence[tuple[str, str]] = (),
) -> tuple[str, list[Any]]:
    """the condition naming the rows after the cursor, and its parameters numbered from ``first``.

    ``by_row`` (PostgreSQL, a sort :func:`_seeks_by_row`): the row comparison alone, ``(c1, c2) > ($1, $2)``, which
    PostgreSQL seeks its index by; the expansion makes it plan a bitmap scan and a sort of every row past the lead
    (a 1,001-row page of a 6,000-row prefix: 2.3 ms against 0.28 ms). Otherwise the leading bound and the expansion,
    which YugabyteDB seeks inside a hashed bucket by (a row comparison there reads the bucket from its start)."""
    params: list[Any] = []
    marks: list[str | None] = []
    for sort, value in zip(order, values, strict=True):
        if value is None:
            marks.append(None)
        else:
            params.append(value)
            place = f"${first + len(params) - 1}"
            if sort.cast is None:
                marks.append(place)
            else:
                marks.append(f"{place}::{'text' if sort.cast == 'text' else f'text::{sort.cast}'}")

    def after(sort: Sort, mark: str | None) -> str:
        column = sort.sorted_by
        if mark is None:
            # nulls last ascending: nothing follows within the column; nulls first descending: every value follows
            return f"{column} IS NOT NULL" if sort.descending else "FALSE"
        beyond = f"{column} {'<' if sort.descending else '>'} {mark}"
        return f"({beyond} OR {column} IS NULL)" if sort.nullable and not sort.descending else beyond

    def same(sort: Sort, mark: str | None) -> str:
        column = sort.sorted_by
        return f"{column} IS NULL" if mark is None else f"{column} = {mark}"

    if by_row and _seeks_by_row(order) and all(mark is not None for mark in marks):
        # the columns ``by`` pins lead the comparison (``lead``: each column and its own placeholder), so it is one
        # range of the index they lead, never a range of an index of the sort's columns alone, filtered
        ascending = not order[0].descending
        columns = ", ".join([column for column, _ in lead if ascending] + [sort.sorted_by for sort in order])
        bound = ", ".join([place for _, place in lead if ascending] + [str(mark) for mark in marks])
        return f"({columns}) {'>' if ascending else '<'} ({bound})", params
    expanded = after(order[-1], marks[-1])
    for sort, mark in zip(reversed(order[:-1]), reversed(marks[:-1]), strict=True):
        expanded = f"{after(sort, mark)} OR ({same(sort, mark)} AND ({expanded}))"
    lead, lead_mark = order[0], marks[0]
    if lead_mark is not None and not lead.nullable:
        bound = f"{lead.sorted_by} {'<=' if lead.descending else '>='} {lead_mark}"
        return f"{bound} AND ({expanded})", params
    return f"({expanded})", params


def _order_by(order: Sequence[Sort]) -> str:
    parts = []
    for sort in order:
        direction = "DESC" if sort.descending else "ASC"
        nulls = (" NULLS FIRST" if sort.descending else " NULLS LAST") if sort.nullable else ""
        parts.append(f"{sort.sorted_by} {direction}{nulls}")
    return ", ".join(parts)


def exists_keyed(table: str, *, on: Mapping[str, str], by: Mapping[str, Any] | None = None) -> Fragment:
    """a visibility clause: whether a row of ``table`` joined to the paged row by ``on`` (and ``by``) exists.

    ``EXISTS (SELECT 1 FROM <table> AS v WHERE v.<col> = t.<outer col> AND ... AND <by>)``, a correlated probe: with
    ``on`` and ``by`` covering a leading part of ``table``'s key, each probe is one keyed lookup.

    :param table: the table probed, a TRUSTED identifier
    :ptype table: str
    :param on: its column -> the paged row's column it equals, TRUSTED identifiers
    :ptype on: Mapping[str, str]
    :param by: its columns pinned to a value (the caller, the tenant), each a value or an ``AnyOf``
    :ptype by: Mapping[str, Any] | None
    :return: the clause, its placeholders numbered from ``$1``
    :rtype: Fragment
    :raises ValueError: when ``on`` names nothing
    """
    if not on:
        raise ValueError(f"a visibility probe of {table} names how it joins the paged row")
    joined = [f"v.{quote_identifier(inner)} = t.{quote_identifier(outer)}" for inner, outer in on.items()]
    pinned, params = by_conditions(by or {}, alias="v")
    sql = f"EXISTS (SELECT 1 FROM {quoted_table(table)} AS v WHERE {' AND '.join([*joined, *pinned])})"
    return Fragment(sql=sql, params=tuple(params))


async def keyset_page(
    conn: Any,
    table: str | Derived,
    *,
    columns: Sequence[str],
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
    :param columns: the columns read, TRUSTED identifiers; each sort column among them (a sort expression's name
        not: it is selected under it)
    :ptype columns: Sequence[str]
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
        or ``limit`` is under one
    :raises CursorError: when ``cursor`` is not one this sort issued
    """
    _check(table, columns, order, limit)
    past = None if cursor is None else _decode(order, cursor)
    by_row = past is not None and await _steps_by_row(conn, order)
    sql, params = _statement(table, columns, by, order, limit + 1, past, by_row, visible, where, lock)
    answered = [dict(row) for row in await conn.fetch(sql, *params)]
    more = len(answered) > limit
    if more:
        del answered[limit:]
    return KeyPage(rows=answered, next_cursor=_encode(order, answered[-1]) if more else None)


async def _steps_by_row(conn: Any, order: Sequence[Sort]) -> bool:
    """whether a page past a place in ``order`` steps by the row comparison: a sort :func:`_seeks_by_row`, on a
    database other than YugabyteDB (an in-process DuckDB or SQLite is never it; an asyncpg connection says so with
    no round trip, :func:`~threetears.core.backends.dialect.known_on_yugabyte`)."""
    if not _seeks_by_row(order):
        return False
    if isinstance(conn, ThreadedDuckDB | ThreadedSQLite | InlineDuckDB):
        return True
    return not await known_on_yugabyte(conn)


def _check(table: str | Derived, columns: Sequence[str], order: Sequence[Sort], limit: int) -> None:
    if not order:
        raise ValueError(f"a page of {table} names its sort")
    if limit < 1:
        raise ValueError(f"a page keeps at least one row, got {limit}")
    missing = [sort.column for sort in order if sort.expression is None and sort.column not in columns]
    if missing:
        raise ValueError(f"a page of {table} reads its sort columns, missing {missing}")
    shadowed = [sort.column for sort in order if sort.expression is not None and sort.column in columns]
    if shadowed:
        # the cursor would carry the column's value, not the expression's the page compares
        raise ValueError(f"a sort expression is read under a name of its own, not a column read: {shadowed}")


def _pinned(by: Mapping[str, Any], order: Sequence[Sort]) -> list[tuple[str, str]]:
    """each column ``by`` pins to one value (not a set, not NULL, not a sort column), as the page names it, and the
    placeholder :func:`~threetears.core.keyed._filters.by_conditions` binds it at."""
    pinned: list[tuple[str, str]] = []
    place = 1
    sorted_columns = {sort.column for sort in order}
    for column, value in by.items():
        if isinstance(value, AnyOf):
            place += len(list(value.values)) if value.as_list else 1
        elif value is not None:
            if column not in sorted_columns:
                pinned.append((f"t.{quote_identifier(column)}", f"${place}"))
            place += 1
    return pinned


def _statement(
    table: str | Derived,
    columns: Sequence[str],
    by: Mapping[str, Any],
    order: Sequence[Sort],
    rows: int,
    past: Sequence[str | None] | None,
    by_row: bool,
    visible: Sequence[Fragment],
    where: Sequence[Fragment],
    lock: Lock | None,
) -> tuple[str, list[Any]]:
    """one page's statement: ``rows`` rows past ``past`` (the last row's sort values as text; the first page when
    None), and its parameters."""
    conditions, params = by_conditions(by, alias="t")
    lead = _pinned(by, order)
    for fragment in where:
        conditions.append(renumbered(fragment, len(params)))
        params += list(fragment.params)
    if past is not None:
        condition, past_params = _past(order, past, len(params) + 1, by_row=by_row, lead=lead)
        conditions.append(condition)
        params += past_params
    for fragment in visible:
        conditions.append(renumbered(fragment, len(params)))
        params += list(fragment.params)
    clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""
    selected = ", ".join(
        [f"t.{quote_identifier(column)}" for column in columns]
        + [
            f"{sort.expression} AS {quote_identifier(sort.column)}"
            for sort in order
            if sort.expression is not None and sort.column not in columns
        ]
    )
    # the pinned columns, constant on every row, lead the ORDER BY too: the same order, read as one range of the
    # index they lead
    ordered = ", ".join([f"{column} ASC" for column, _ in lead] + [_order_by(order)])
    sql = f"SELECT {selected} FROM {relation(table, 't')}{clause} ORDER BY {ordered} LIMIT {rows}"
    if lock is not None:
        sql += lock.of("t")
    return sql, params


async def keyset_rows(
    conn: Any,
    table: str | Derived,
    *,
    columns: Sequence[str],
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
    last row, its sort values bound straight from that row (no cursor text between them). A listing of one page is
    one statement. Past a place, PostgreSQL steps by the row comparison, YugabyteDB by the leading bound and the
    expansion (:func:`_past`), as :func:`keyset_page` does.

    :param conn: anything with an asyncpg-shaped ``fetch(query, *params)``
    :ptype conn: Any
    :param table: the table (``schema.table``), or a derived relation
    :ptype table: str | Derived
    :param columns: the columns read, TRUSTED identifiers; each sort column among them (a sort expression's name
        not: it is selected under it)
    :ptype columns: Sequence[str]
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
    _check(table, columns, order, page)
    rows: list[dict[str, Any]] = []
    past: list[str | None] | None = None
    by_row: bool | None = None
    while True:
        sql, params = _statement(table, columns, by, order, page + 1, past, bool(by_row), visible, where, None)
        answered = await conn.fetch(sql, *params)
        rows.extend(dict(row) for row in answered)
        if len(answered) <= page:
            return rows
        if by_row is None:
            by_row = await _steps_by_row(conn, order)
        last = rows[-1]
        past = [None if last[sort.column] is None else _as_text(last[sort.column]) for sort in order]
