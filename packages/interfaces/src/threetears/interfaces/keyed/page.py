"""one page of a listing, by key, as a statement: a bound prefix, a sort, an opaque cursor and a ``LIMIT``.

A page reads::

    SELECT <columns> FROM <table> AS t
     WHERE <by: the bound prefix, the tenant>
       AND <past the cursor, in the sort's own direction and null order>
       AND <visible: a keyed EXISTS, the caller's grants>
     ORDER BY <sort columns> LIMIT <limit + 1>

**The sort vocabulary is closed** (:class:`Sort`): a caller maps its own closed set of fields to entries it writes; no
caller string reaches an ``ORDER BY``. **The cursor is opaque**: base64 of the last row's sort values as text, each
with a null flag; one that does not decode, or names another sort, is refused (:class:`CursorError`).

Pure: the statement and its binds. Whether a page past a place steps by the row comparison is a question of the
database (never on YugabyteDB, where the lexicographic expansion seeks a hashed bucket), answered by the runner
(``threetears.core.keyed.keyset_page`` asks its connection) and handed here as ``by_row``.
"""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Any

from threetears.interfaces.keyed.filters import (
    AnyOf,
    Derived,
    Fragment,
    Lock,
    Selected,
    by_conditions,
    narrowing,
    quote_identifier,
    quoted_table,
    relation,
    selected_column,
)

__all__ = [
    "CursorError",
    "KeyPage",
    "Sort",
    "check_page",
    "cursor_values",
    "exists_keyed",
    "keyset_cursor",
    "keyset_page_statement",
    "keyset_rows_statement",
    "next_past",
    "page_statement",
    "seeks_by_row",
]


class CursorError(ValueError):
    """A cursor token could not be decoded (malformed, tampered, or wrong arity)."""


@dataclass(frozen=True)
class Sort:
    """one column of a listing's sort, written by the endpoint, never by its caller.

    :param column: the column, a TRUSTED identifier of the paged table
    :ptype column: str
    :param cast: its SQL type, which the cursor's text is cast back to; None binds it uncast (SQLite, which has no
        ``::`` casts and compares by the column's own affinity; a column whose type the caller does not know): a page
        past a cursor binds its text, :func:`keyset_rows` the last row's own value, typed by the column
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

    @classmethod
    def cut(cls, order: Sequence[Sort], answered: list[dict[str, Any]], limit: int) -> KeyPage:
        """the page a statement asking ``limit + 1`` rows answered: its first ``limit`` rows, and the cursor of the
        last of them when the one past them says more follow.

        :param order: the page's sort
        :ptype order: Sequence[Sort]
        :param answered: the statement's rows, as dicts; cut in place
        :ptype answered: list[dict[str, Any]]
        :param limit: the rows a page keeps
        :ptype limit: int
        :return: the page
        :rtype: KeyPage
        """
        more = len(answered) > limit
        if more:
            del answered[limit:]
        return cls(rows=answered, next_cursor=keyset_cursor(order, answered[-1]) if more else None)


def _as_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, datetime | date | time):
        return value.isoformat()
    return str(value)


def keyset_cursor(order: Sequence[Sort], row: Mapping[str, Any]) -> str:
    """the cursor of a page that ends at ``row``: :func:`keyset_page` given it resumes right after ``row``.

    The cursor every page answers (:attr:`KeyPage.next_cursor`), made from a row's sort values: a caller moving a
    listing onto :func:`keyset_page` makes one from a place its own older cursor named, so those keep resuming.

    :param order: the sort the cursor resumes, as :func:`keyset_page` is given it
    :ptype order: Sequence[Sort]
    :param row: the row the page ended at: each sort column's value under its name (``None`` for ``NULL``)
    :ptype row: Mapping[str, Any]
    :return: the cursor
    :rtype: str
    :raises KeyError: when ``row`` lacks a sort column
    """
    values = [None if row[sort.column] is None else _as_text(row[sort.column]) for sort in order]
    raw = json.dumps([[sort.column for sort in order], values], separators=(",", ":"))
    return base64.urlsafe_b64encode(raw.encode()).decode()


def cursor_values(order: Sequence[Sort], cursor: str) -> list[str | None]:
    """the sort values a cursor names (each as text, or None for ``NULL``), checked against ``order``.

    :param order: the sort the cursor resumes
    :ptype order: Sequence[Sort]
    :param cursor: the cursor, as a page answered it
    :ptype cursor: str
    :return: the values, one a sort column
    :rtype: list[str | None]
    :raises CursorError: when the cursor does not decode, or names another sort
    """
    try:
        named, values = json.loads(base64.urlsafe_b64decode(cursor.encode()).decode())
    except (binascii.Error, ValueError, UnicodeError, TypeError) as exc:
        raise CursorError(f"invalid cursor token: {cursor!r}") from exc
    if named != [sort.column for sort in order] or not isinstance(values, list) or len(values) != len(order):
        raise CursorError("the cursor names another sort than this listing's")
    if not all(value is None or isinstance(value, str) for value in values):
        raise CursorError("the cursor's values are not text")
    return values


def seeks_by_row(order: Sequence[Sort]) -> bool:
    """whether the rows past a place in ``order`` are one row comparison: every column one direction, none NULL.

    :param order: the sort
    :ptype order: Sequence[Sort]
    :return: whether they are
    :rtype: bool
    """
    return len(order) > 1 and not any(sort.nullable for sort in order) and len({sort.descending for sort in order}) == 1


def _past(
    order: Sequence[Sort],
    values: Sequence[Any],
    first: int,
    *,
    by_row: bool = False,
) -> tuple[str, list[Any]]:
    """the condition naming the rows after the cursor, and its parameters numbered from ``first``.

    ``by_row`` (PostgreSQL, a sort :func:`seeks_by_row`): the row comparison alone, ``(c1, c2) > ($1, $2)``, which
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

    if by_row and seeks_by_row(order) and all(mark is not None for mark in marks):
        # the sort's columns alone: the columns ``by`` pins are bound by equality beside it (``kind = $1 AND (name,
        # id) > ($2, $3)``), never inside it. Inside it (``(kind, name, id) > ($1, $2, $3)``) only an index led by the
        # pinned columns can take it as its condition, and PostgreSQL may read another (a unique index of the next
        # column) from its start, filtering: 2,041 rows read for 21. Beside it, every index answering the order
        # takes the comparison as its condition, the pinned columns' index by its equality first
        columns = ", ".join(sort.sorted_by for sort in order)
        bound = ", ".join(str(mark) for mark in marks)
        return f"({columns}) {'<' if order[0].descending else '>'} ({bound})", params
    expanded = after(order[-1], marks[-1])
    for sort, mark in zip(reversed(order[:-1]), reversed(marks[:-1]), strict=True):
        expanded = f"{after(sort, mark)} OR ({same(sort, mark)} AND ({expanded}))"
    head, head_mark = order[0], marks[0]
    if head_mark is not None and not head.nullable:
        bound = f"{head.sorted_by} {'<=' if head.descending else '>='} {head_mark}"
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


def check_page(table: str | Derived, columns: Sequence[str | Selected], order: Sequence[Sort], limit: int) -> None:
    """refuse a page that cannot be read: no sort, no row kept, a sort column not read, or a sort expression read
    under a column's name.

    :param table: the table, for the message
    :ptype table: str | Derived
    :param columns: the columns read
    :ptype columns: Sequence[str | Selected]
    :param order: the sort
    :ptype order: Sequence[Sort]
    :param limit: the rows a page keeps
    :ptype limit: int
    :return: nothing
    :rtype: None
    :raises ValueError: for such a page
    """
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


def _pinned(by: Mapping[str, Any], order: Sequence[Sort]) -> list[str]:
    """each column ``by`` pins to one value (not a set, not NULL, not a sort column), as the page names it."""
    sorted_columns = {sort.column for sort in order}
    return [
        f"t.{quote_identifier(column)}"
        for column, value in by.items()
        if not isinstance(value, AnyOf) and value is not None and column not in sorted_columns
    ]


def page_statement(
    table: str | Derived,
    columns: Sequence[str | Selected],
    by: Mapping[str, Any],
    order: Sequence[Sort],
    rows: int,
    past: Sequence[Any] | None,
    by_row: bool,
    visible: Sequence[Fragment],
    where: Sequence[Fragment],
    lock: Lock | None,
) -> tuple[str, list[Any]]:
    """one page's statement: ``rows`` rows past ``past`` (the last row's sort values, as text where cast; the first
    page when None), and its parameters.

    :param table: the table, a TRUSTED identifier, or a derived relation
    :ptype table: str | Derived
    :param columns: the columns read
    :ptype columns: Sequence[str | Selected]
    :param by: the bound prefix
    :ptype by: Mapping[str, Any]
    :param order: the sort
    :ptype order: Sequence[Sort]
    :param rows: the rows the statement asks for (``LIMIT``)
    :ptype rows: int
    :param past: the place the page starts after; the first page when None
    :ptype past: Sequence[Any] | None
    :param by_row: step past ``past`` by the row comparison (PostgreSQL, DuckDB, SQLite; never YugabyteDB)
    :ptype by_row: bool
    :param visible: visibility clauses
    :ptype visible: Sequence[Fragment]
    :param where: filters beside the prefix
    :ptype where: Sequence[Fragment]
    :param lock: the row lock taken; none when None
    :ptype lock: Lock | None
    :return: the statement, and its parameters
    :rtype: tuple[str, list[Any]]
    """
    conditions, params = by_conditions(by, alias="t")
    lead = _pinned(by, order)
    for fragment in where:
        conditions.append(narrowing(fragment, len(params)))
        params += list(fragment.params)
    if past is not None:
        condition, past_params = _past(order, past, len(params) + 1, by_row=by_row)
        conditions.append(condition)
        params += past_params
    for fragment in visible:
        conditions.append(narrowing(fragment, len(params)))
        params += list(fragment.params)
    clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""
    selected = ", ".join(
        [selected_column(column) for column in columns]
        + [
            f"{sort.expression} AS {quote_identifier(sort.column)}"
            for sort in order
            if sort.expression is not None and sort.column not in columns
        ]
    )
    # the pinned columns, constant on every row, lead the ORDER BY too: the same order, read as one range of the
    # index they lead
    ordered = ", ".join([f"{column} ASC" for column in lead] + [_order_by(order)])
    sql = f"SELECT {selected} FROM {relation(table, 't')}{clause} ORDER BY {ordered} LIMIT {rows}"
    if lock is not None:
        sql += lock.of("t")
    return sql, params


def _bound(sort: Sort, value: Any) -> Any:
    """how :func:`keyset_rows` binds a sort column's value from the last row: its text, cast back by the sort; the
    value as read when the sort names no cast (the database types it by the column); None for ``NULL``."""
    if value is None or sort.cast is None:
        return value
    return _as_text(value)


def next_past(order: Sequence[Sort], row: Mapping[str, Any]) -> list[Any]:
    """the place the next statement of a listing read to its end starts after: ``row``'s sort values, each bound as
    :func:`page_statement` takes it (its text, cast back by the sort; the value itself for an uncast sort; None for
    ``NULL``), no cursor text between them.

    :param order: the sort
    :ptype order: Sequence[Sort]
    :param row: the last row read
    :ptype row: Mapping[str, Any]
    :return: the place
    :rtype: list[Any]
    """
    return [_bound(sort, row[sort.column]) for sort in order]


def keyset_page_statement(
    table: str | Derived,
    *,
    columns: Sequence[str | Selected],
    by: Mapping[str, Any],
    order: Sequence[Sort],
    limit: int,
    cursor: str | None = None,
    by_row: bool = False,
    visible: Sequence[Fragment] = (),
    where: Sequence[Fragment] = (),
    lock: Lock | None = None,
) -> tuple[str, list[Any]]:
    """the statement of one page of ``table``'s rows ``by`` names, in ``order``, after ``cursor``: what
    ``threetears.core.keyed.keyset_page`` sends. It asks ``limit + 1`` rows; :meth:`KeyPage.cut` makes the page.

    :param table: the table (``schema.table``), or a derived relation
    :ptype table: str | Derived
    :param columns: the columns read, each sort column among them
    :ptype columns: Sequence[str | Selected]
    :param by: the bound prefix (a tenant, a parent), each a value or an ``AnyOf``
    :ptype by: Mapping[str, Any]
    :param order: the sort, its last column unique among the rows ``by`` names
    :ptype order: Sequence[Sort]
    :param limit: the rows a page keeps
    :ptype limit: int
    :param cursor: the cursor a previous page answered; the first page when None
    :ptype cursor: str | None
    :param by_row: step past the cursor by the row comparison, where the database is not YugabyteDB (an
        in-process SQLite or DuckDB is not)
    :ptype by_row: bool
    :param visible: visibility clauses each row must pass (:func:`exists_keyed`)
    :ptype visible: Sequence[Fragment]
    :param where: filters beside the prefix, each a fragment written as ``t."col"`` from ``$1``
    :ptype where: Sequence[Fragment]
    :param lock: the row lock the page takes on its rows; none when None
    :ptype lock: Lock | None
    :return: the statement, and its parameters
    :rtype: tuple[str, list[Any]]
    :raises ValueError: as :func:`check_page`
    :raises CursorError: when ``cursor`` is not one this sort issued
    """
    check_page(table, columns, order, limit)
    past = None if cursor is None else cursor_values(order, cursor)
    return page_statement(table, columns, by, order, limit + 1, past, past is not None and by_row, visible, where, lock)


def keyset_rows_statement(
    table: str | Derived,
    *,
    columns: Sequence[str | Selected],
    by: Mapping[str, Any],
    order: Sequence[Sort],
    page: int = 500,
    past: Sequence[Any] | None = None,
    by_row: bool = False,
    visible: Sequence[Fragment] = (),
    where: Sequence[Fragment] = (),
) -> tuple[str, list[Any]]:
    """one statement of a listing read to its end, page after page: what ``threetears.core.keyed.keyset_rows``
    sends. It asks ``page + 1`` rows; every row answered is kept, and while the answer is full the next statement
    starts past its last row (:func:`next_past`).

    :param table: the table (``schema.table``), or a derived relation
    :ptype table: str | Derived
    :param columns: the columns read, each sort column among them
    :ptype columns: Sequence[str | Selected]
    :param by: the bound prefix, as :func:`keyset_page_statement` takes it
    :ptype by: Mapping[str, Any]
    :param order: the sort, its last column unique among the rows ``by`` names
    :ptype order: Sequence[Sort]
    :param page: the rows a statement reads past the one that says more follow
    :ptype page: int
    :param past: where the statement starts (:func:`next_past` of the last row read); the first when None
    :ptype past: Sequence[Any] | None
    :param by_row: step past ``past`` by the row comparison, as :func:`keyset_page_statement` takes it
    :ptype by_row: bool
    :param visible: visibility clauses
    :ptype visible: Sequence[Fragment]
    :param where: filters beside the prefix
    :ptype where: Sequence[Fragment]
    :return: the statement, and its parameters
    :rtype: tuple[str, list[Any]]
    :raises ValueError: as :func:`check_page`
    """
    check_page(table, columns, order, page)
    return page_statement(table, columns, by, order, page + 1, past, by_row, visible, where, None)
