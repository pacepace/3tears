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

**How a page steps past its cursor.** The lexicographic expansion over the sort columns, each in its own direction
(``c1 > $1 OR (c1 = $1 AND c2 > $2)``), led by a range on the first column when it is not nullable (``c1 >= $1``),
so PostgreSQL and YugabyteDB seek the index by the lead and filter the rest in storage
(:mod:`threetears.core.keyset`).

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

from threetears.core.keyed._filters import Fragment, by_conditions, renumbered
from threetears.core.pagination import CursorError
from threetears.core.sql_fragments import quote_identifier

__all__ = ["Sort", "KeyPage", "exists_keyed", "keyset_page"]


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
    """

    column: str
    cast: str | None = "text"
    descending: bool = False
    nullable: bool = False


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


def _past(order: Sequence[Sort], values: Sequence[str | None], first: int) -> tuple[str, list[Any]]:
    """the condition naming the rows after the cursor, and its parameters numbered from ``first``."""
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
        column = f"t.{quote_identifier(sort.column)}"
        if mark is None:
            # nulls last ascending: nothing follows within the column; nulls first descending: every value follows
            return f"{column} IS NOT NULL" if sort.descending else "FALSE"
        beyond = f"{column} {'<' if sort.descending else '>'} {mark}"
        return f"({beyond} OR {column} IS NULL)" if sort.nullable and not sort.descending else beyond

    def same(sort: Sort, mark: str | None) -> str:
        column = f"t.{quote_identifier(sort.column)}"
        return f"{column} IS NULL" if mark is None else f"{column} = {mark}"

    expanded = after(order[-1], marks[-1])
    for sort, mark in zip(reversed(order[:-1]), reversed(marks[:-1]), strict=True):
        expanded = f"{after(sort, mark)} OR ({same(sort, mark)} AND ({expanded}))"
    lead, lead_mark = order[0], marks[0]
    if lead_mark is not None and not lead.nullable:
        bound = f"t.{quote_identifier(lead.column)} {'<=' if lead.descending else '>='} {lead_mark}"
        return f"{bound} AND ({expanded})", params
    return f"({expanded})", params


def _order_by(order: Sequence[Sort]) -> str:
    parts = []
    for sort in order:
        direction = "DESC" if sort.descending else "ASC"
        nulls = (" NULLS FIRST" if sort.descending else " NULLS LAST") if sort.nullable else ""
        parts.append(f"t.{quote_identifier(sort.column)} {direction}{nulls}")
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
    sql = f"EXISTS (SELECT 1 FROM {quote_identifier(table)} AS v WHERE {' AND '.join([*joined, *pinned])})"
    return Fragment(sql=sql, params=tuple(params))


async def keyset_page(
    conn: Any,
    table: str,
    *,
    columns: Sequence[str],
    by: Mapping[str, Any],
    order: Sequence[Sort],
    limit: int,
    cursor: str | None = None,
    visible: Sequence[Fragment] = (),
) -> KeyPage:
    """one page of ``table``'s rows ``by`` names, in ``order``, after ``cursor``.

    :param conn: anything with an asyncpg-shaped ``fetch(query, *params)``
    :ptype conn: Any
    :param table: the table, a TRUSTED identifier
    :ptype table: str
    :param columns: the columns read, TRUSTED identifiers; each sort column among them
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
    :return: the page
    :rtype: KeyPage
    :raises ValueError: when the sort is empty, a sort column is not read, or ``limit`` is under one
    :raises CursorError: when ``cursor`` is not one this sort issued
    """
    if not order:
        raise ValueError(f"a page of {table} names its sort")
    if limit < 1:
        raise ValueError(f"a page keeps at least one row, got {limit}")
    missing = [sort.column for sort in order if sort.column not in columns]
    if missing:
        raise ValueError(f"a page of {table} reads its sort columns, missing {missing}")
    conditions, params = by_conditions(by, alias="t")
    if cursor is not None:
        past, past_params = _past(order, _decode(order, cursor), len(params) + 1)
        conditions.append(past)
        params += past_params
    for fragment in visible:
        conditions.append(renumbered(fragment, len(params)))
        params += list(fragment.params)
    where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
    selected = ", ".join(f"t.{quote_identifier(column)}" for column in columns)
    sql = f"SELECT {selected} FROM {quote_identifier(table)} AS t{where} ORDER BY {_order_by(order)} LIMIT {limit + 1}"
    answered = [dict(row) for row in await conn.fetch(sql, *params)]
    rows = answered[:limit]
    more = len(answered) > limit
    return KeyPage(rows=rows, next_cursor=_encode(order, rows[-1]) if more else None)
