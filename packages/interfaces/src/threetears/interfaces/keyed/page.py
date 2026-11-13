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
from dataclasses import dataclass, replace
from datetime import date, datetime, time
from typing import Any, Literal

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
    "order_by_terms",
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
    :param nulls: where its NULLs fall, ``"first"`` or ``"last"``, in either direction; None leaves it to
        ``nullable`` (a NULL sorts as the largest value: last ascending, first descending) or, without it, to the
        engine. Only an ordering takes another place (:func:`threetears.core.keyed.ordered`, ``read_step``'s
        ``order``): a keyset page steps past NULLs as the largest value, so :func:`page_statement` refuses a sort
        whose ``nulls`` says otherwise
    :ptype nulls: Literal["first", "last"] | None
    """

    column: str
    cast: str | None = "text"
    descending: bool = False
    nullable: bool = False
    expression: str | None = None
    nulls: Literal["first", "last"] | None = None

    def __post_init__(self) -> None:
        """refuse a null order that is neither ``"first"`` nor ``"last"``.

        :return: nothing
        :rtype: None
        :raises ValueError: for such a null order
        """
        if self.nulls not in (None, "first", "last"):
            raise ValueError(f"a sort's NULLs fall first or last, not {self.nulls!r}")

    @property
    def nulls_term(self) -> str:
        """where its NULLs fall as an ``ORDER BY`` term ends: ``nulls`` when set; with ``nullable``, as the largest
        value; nothing otherwise (the engine's own order).

        :return: `` NULLS FIRST``, `` NULLS LAST`` or the empty string
        :rtype: str
        """
        if self.nulls is not None:
            return f" NULLS {self.nulls.upper()}"
        if self.nullable:
            return " NULLS FIRST" if self.descending else " NULLS LAST"
        return ""

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


def _marks(order: Sequence[Sort], values: Sequence[Any], first: int) -> tuple[list[str | None], list[Any]]:
    """each sort value's placeholder (cast back by its sort; None for ``NULL``), numbered from ``first``, and the
    parameters they bind."""
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
    return marks, params


def _after(sort: Sort, mark: str | None) -> str:
    column = sort.sorted_by
    if mark is None:
        # nulls last ascending: nothing follows within the column; nulls first descending: every value follows
        return f"{column} IS NOT NULL" if sort.descending else "FALSE"
    beyond = f"{column} {'<' if sort.descending else '>'} {mark}"
    return f"({beyond} OR {column} IS NULL)" if sort.nullable and not sort.descending else beyond


def _same(sort: Sort, mark: str | None) -> str:
    column = sort.sorted_by
    return f"{column} IS NULL" if mark is None else f"{column} = {mark}"


def _past_marks(order: Sequence[Sort], marks: Sequence[str | None], *, by_row: bool) -> str:
    """the condition naming the rows after the place ``marks`` hold in ``order``; ``FALSE`` for no sort (nothing
    follows a tie of every column before it)."""
    if not order:
        return "FALSE"
    if by_row and seeks_by_row(order) and all(mark is not None for mark in marks):
        # the sort's columns alone: the columns ``by`` pins are bound by equality beside it (``kind = $1 AND (name,
        # id) > ($2, $3)``), never inside it. Inside it (``(kind, name, id) > ($1, $2, $3)``) only an index led by the
        # pinned columns can take it as its condition, and PostgreSQL may read another (a unique index of the next
        # column) from its start, filtering: 2,041 rows read for 21. Beside it, every index answering the order
        # takes the comparison as its condition, the pinned columns' index by its equality first
        columns = ", ".join(sort.sorted_by for sort in order)
        bound = ", ".join(str(mark) for mark in marks)
        return f"({columns}) {'<' if order[0].descending else '>'} ({bound})"
    expanded = _after(order[-1], marks[-1])
    for sort, mark in zip(reversed(order[:-1]), reversed(marks[:-1]), strict=True):
        expanded = f"{_after(sort, mark)} OR ({_same(sort, mark)} AND ({expanded}))"
    head, head_mark = order[0], marks[0]
    if head_mark is not None and not head.nullable:
        bound = f"{head.sorted_by} {'<=' if head.descending else '>='} {head_mark}"
        return f"{bound} AND ({expanded})"
    return f"({expanded})"


def _after_ranges(sort: Sort, mark: str | None) -> list[str]:
    """the rows after ``mark`` in one sort column, as conditions each an index takes as one range (no ``OR``): its
    values past the mark, and its NULLs where they follow (ascending NULLs last after a value); none where nothing
    follows (ascending, after a NULL)."""
    column = sort.sorted_by
    if mark is None:
        return [f"{column} IS NOT NULL"] if sort.descending else []
    beyond = f"{column} {'<' if sort.descending else '>'} {mark}"
    return [beyond, f"{column} IS NULL"] if sort.nullable and not sort.descending else [beyond]


def _seek_legs(
    order: Sequence[Sort], values: Sequence[Any], first: int
) -> tuple[list[tuple[str, Sequence[Sort]]], list[Any]]:
    """the rows after the cursor as legs each an index reads as one range with no filter beside it: for each sort
    column, the columns before it equal to the cursor's and the column past it (``c1 = $1 AND c2 < $2``, then
    ``c1 < $1``), its NULLs a leg of their own where they follow; each leg in the whole sort's order. And the
    parameters, numbered from ``first``, every leg shares.

    For YugabyteDB, which pushes a page's ``LIMIT`` down to storage only when no condition is left to filter after
    the index read: the one-condition expansion (``c1 <= $1 AND (c1 < $1 OR (c1 = $1 AND c2 < $2))``) leaves its
    ``OR`` as a filter, so each page read a whole fetch (1,024 rows) for a page of 9; and the row comparison is
    rechecked the same way. Each leg reads at most its ``LIMIT``, so a page of ``n`` reads at most
    ``n + 1`` rows a leg (two legs for a sort of two columns)."""
    marks, params = _marks(order, values, first)
    legs: list[tuple[str, Sequence[Sort]]] = []
    for depth in reversed(range(len(order))):
        equal = [_same(sort, mark) for sort, mark in zip(order[:depth], marks[:depth], strict=True)]
        legs.extend((" AND ".join([*equal, after]), order) for after in _after_ranges(order[depth], marks[depth]))
    return legs, params


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
    marks, params = _marks(order, values, first)
    return _past_marks(order, marks, by_row=by_row), params


def _chained(statements: Sequence[str], order: Sequence[Sort], rows: int) -> str:
    """the page from legs that follow one another in the sort (every row of a leg before every row of the next), each
    leg asking only the rows the legs before it left short of ``rows``: ``WITH l0 AS (<leg> LIMIT rows), l1 AS (<leg>
    LIMIT rows - (SELECT count(*) FROM l0)), ...``, then the legs in the sort's order.

    Each leg's ``LIMIT`` is known when its scan starts (YugabyteDB hands it to storage), so the page reads at most
    ``rows`` rows (and a leg's read-ahead) in all, however many legs it has; a leg the ones before it filled asks
    none and is never read. Every leg answering in full would read ``rows`` a leg."""
    names = [f"l{index}" for index in range(len(statements))]
    ctes = []
    for index, statement in enumerate(statements):
        head, _, _ = statement.rpartition(" LIMIT ")
        found = " - ".join(f"(SELECT count(*) FROM {name})" for name in names[:index])
        ctes.append(f"{names[index]} AS ({head} LIMIT {rows}{' - ' + found if found else ''})")
    union = " UNION ALL ".join(f"SELECT * FROM {name}" for name in names)
    outer = order_by_terms([replace(sort, expression=f"u.{quote_identifier(sort.column)}") for sort in order])
    return f"WITH {', '.join(ctes)} SELECT * FROM ({union}) AS u ORDER BY {outer} LIMIT {rows}"


def order_by_terms(order: Sequence[Sort]) -> str:
    """``order``'s terms as an ``ORDER BY`` lists them: each sort's column or expression, its direction, and where its
    NULLs fall when it can hold them.

    :param order: the sort
    :ptype order: Sequence[Sort]
    :return: the terms, comma separated
    :rtype: str
    """
    parts = []
    for sort in order:
        direction = "DESC" if sort.descending else "ASC"
        parts.append(f"{sort.sorted_by} {direction}{sort.nulls_term}")
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


def check_page(
    table: str | Derived, columns: Sequence[str | Selected] | None, order: Sequence[Sort], limit: int
) -> None:
    """refuse a page that cannot be read: no sort, no row kept, a sort column not read, or a sort expression read
    under a column's name (or beside every column, whose names it cannot be told apart from).

    :param table: the table, for the message
    :ptype table: str | Derived
    :param columns: the columns read; every column of the table when None
    :ptype columns: Sequence[str | Selected] | None
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
    if columns is None:
        if any(sort.expression is not None for sort in order):
            raise ValueError(f"a page of every column of {table} sorts by its columns, not an expression")
        return
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
    columns: Sequence[str | Selected] | None,
    by: Mapping[str, Any],
    order: Sequence[Sort],
    rows: int,
    past: Sequence[Any] | None,
    by_row: bool,
    visible: Sequence[Fragment],
    where: Sequence[Fragment],
    lock: Lock | None,
    *,
    seek_legs: bool = False,
    chain_legs: bool = True,
) -> tuple[str, list[Any]]:
    """one page's statement: ``rows`` rows past ``past`` (the last row's sort values, as text where cast; the first
    page when None), and its parameters.

    :param table: the table, a TRUSTED identifier, or a derived relation
    :ptype table: str | Derived
    :param columns: the columns read; every column of the table (``t.*``) when None
    :ptype columns: Sequence[str | Selected] | None
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
    :param seek_legs: step past ``past`` by legs each one range of the index with nothing left to filter
        (:func:`_seek_legs`, YugabyteDB), when no row lock is taken; a sort led by a NULLable column steps so on every
        engine
    :ptype seek_legs: bool
    :param chain_legs: each leg asks only the rows the legs before it left short (:func:`_chained`); a reader whose
        engine may not take a ``LIMIT`` read from a subquery (a warehouse) passes False, and each leg asks a page
    :ptype chain_legs: bool
    :return: the statement, and its parameters
    :rtype: tuple[str, list[Any]]
    :raises ValueError: for a sort whose ``nulls`` puts its NULLs anywhere but where a page steps past them (as the
        largest value: last ascending, first descending)
    """
    misplaced = [sort.column for sort in order if sort.nulls not in (None, "first" if sort.descending else "last")]
    if misplaced:
        raise ValueError(
            f"a keyset page of {table} steps past NULLs as the largest value; {misplaced} place them otherwise "
            "(an ordering of a step takes that: keyed.ordered)"
        )
    conditions, params = by_conditions(by, alias="t")
    lead = _pinned(by, order)
    for fragment in where:
        conditions.append(narrowing(fragment, len(params)))
        params += list(fragment.params)
    # the rows past the cursor: one condition, or legs (below)
    legs: list[tuple[str | None, Sequence[Sort]]] = [(None, order)]
    if past is not None and lock is None and (seek_legs or order[0].nullable):
        # one leg a sort column (a NULLable column's NULLs a leg of their own), each one range of the index with
        # nothing left to filter, in the sort's order, each asking only what the legs before it left short
        # (_seek_legs, _chained): on YugabyteDB, whose LIMIT reaches storage only with no filter left; and for a sort
        # led by a NULLable column on every engine, whose one-condition form gives the lead no bound an index can
        # start from (a UNION takes no row lock)
        seeks, past_params = _seek_legs(order, past, len(params) + 1)
        legs = [(condition, leg_order) for condition, leg_order in seeks] or [("FALSE", order)]
        params += past_params
    elif past is not None:
        past_condition, past_params = _past(order, past, len(params) + 1, by_row=by_row)
        conditions.append(past_condition)
        params += past_params
    after: list[str] = []
    for fragment in visible:
        after.append(narrowing(fragment, len(params)))
        params += list(fragment.params)
    selected = (
        "t.*"
        if columns is None
        else ", ".join(
            [selected_column(column) for column in columns]
            + [
                f"{sort.expression} AS {quote_identifier(sort.column)}"
                for sort in order
                if sort.expression is not None and sort.column not in columns
            ]
        )
    )

    statements: list[str] = []
    for condition, leg_order in legs:
        every = [*conditions, *([condition] if condition is not None else []), *after]
        clause = f" WHERE {' AND '.join(every)}" if every else ""
        # the pinned columns, constant on every row, lead the ORDER BY too: the same order, read as one range of
        # the index they lead
        ordered = ", ".join([f"{column} ASC" for column in lead] + ([order_by_terms(leg_order)] if leg_order else []))
        by_clause = f" ORDER BY {ordered}" if ordered else ""
        statements.append(f"SELECT {selected} FROM {relation(table, 't')}{clause}{by_clause} LIMIT {rows}")
    if len(statements) == 1:
        sql = statements[0]
    elif chain_legs:
        sql = _chained(statements, order, rows)
    else:
        # a reader whose engine may not take a LIMIT read from a subquery (a warehouse): each leg asks a whole page
        union = " UNION ALL ".join(f"({statement})" for statement in statements)
        outer = order_by_terms([replace(sort, expression=f"u.{quote_identifier(sort.column)}") for sort in order])
        sql = f"SELECT * FROM ({union}) AS u ORDER BY {outer} LIMIT {rows}"
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
