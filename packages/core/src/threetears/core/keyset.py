"""One keyset pager for L3: every row a statement shape names, a page at a time, past the last key read.

The L3 rail answers at most :data:`~threetears.core.backends.protocol.L3_RAIL_ROW_CAP` rows a
statement and does not say when it cut, so a read that might pass it asks for one row more than a
page and pages on a unique key until a page comes back without that row. The whole-table copies
(:func:`~threetears.core.collections.complete_copy.read_l3_rows`) and the key-led read's paging of
one leading value (``SqlL3Backend.fetch_led_by``) both page through :func:`read_keyset_pages`, so the
two cannot drift apart in how they step, stop, or spell a name.

**What bounds a page is the order it is read in.** Each page is ``ORDER BY`` the key named here. A page
touches about its own rows only when an index answers the filters in that order: its leading columns
pinned by the equality filters, then the key's columns in the key's order. Otherwise every page reads and
sorts every row the filters leave (YugabyteDB, a state of 33,222 rows: 34 pages, about 1.13M rows read,
10.1 s). A filtered reader whose schema declares such an index pages on
:meth:`~threetears.core.collections.schema_backed.TableSchema.keyset_order`, not the bare primary key.

**How a page steps past its cursor.** After the first page the statement names the cursor three ways, all
the same rows: ``k1 >= $1``, the row comparison ``(k1, k2, ...) > ($1, $2, ...)``, and its expansion
``k1 > $1 OR (k1 = $1 AND (k2 > $2 OR ...))``. Postgres seeks the index by the row comparison. YugabyteDB
seeks no row comparison inside a hashed index's bucket (measured on 2026.1: it reads the bucket from its
start), so it seeks by the leading bound and applies the expansion in storage, as an index filter; a page
then reads at most the rows of the cursor's leading value that the page before already read. With only
the row comparison YugabyteDB rechecks it above storage and cannot push the page's ``LIMIT`` down, reading
twice a page; with only the expansion Postgres plans a bitmap scan and sorts. A one-column key needs none
of this: ``(k1) > ($1)`` is a plain comparison both seek by.

**How many rows a page asks for.** A transport that answers pages (``fetch_page`` /
``fetch_arrow_page``, :class:`~threetears.core.backends.protocol.L3Page`) says in each reply the bytes its
broker cuts a reply at, when it cuts by bytes. The first page asks for ``page_size`` rows; each after, for
as many as the bytes a row took fit in :data:`_BUDGET_SHARE` of the announced cap, so a state's rows take a
few statements, not one a thousand rows. A broker that announces no cap (one that cuts at
:data:`~threetears.core.backends.protocol.L3_RAIL_ROW_CAP` rows, or a direct pool) is paged ``page_size``
rows a statement, as before. A reply the broker cut is the statement's first rows in key order: it is
taken as a short page, and the read steps past its last row.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from threetears.core.backends.arrow_reply import l3_fetch_arrow
from threetears.core.backends.protocol import L3Page
from threetears.core.sql_fragments import equality_conditions, quote_identifier

__all__ = ["KeysetRead", "keyset_arrow_pages", "keyset_pages", "read_keyset_pages"]

#: the share of a byte-capped reply a page is sized to fill: rows vary in width, so a page sized to the whole
#: cap from the rows before would be cut about half the time
_BUDGET_SHARE: Final = 0.8

#: the most rows a byte-sized page asks for, whatever its rows' width
_MOST_ROWS_A_PAGE: Final = 1_000_000


def _next_rows(page: L3Page, rows: int, asked: int) -> int:
    """how many rows the next page asks for: as many as fit in the announced cap at this page's bytes a row;
    as many as this one when no cap was announced.

    :param page: the page just read
    :ptype page: L3Page
    :param rows: how many rows it answered
    :ptype rows: int
    :param asked: how many it asked for
    :ptype asked: int
    :return: the next page's rows
    :rtype: int
    """
    if page.reply_cap is None or rows == 0 or page.reply_bytes <= 0:
        return asked
    fitting = int(page.reply_cap * _BUDGET_SHARE * rows / page.reply_bytes)
    return max(1, min(fitting, _MOST_ROWS_A_PAGE))


@dataclass(frozen=True)
class KeysetRead:
    """what a keyset read answered, and how many statements it took.

    :param rows: the rows, in key order
    :ptype rows: list[dict[str, Any]]
    :param statements: the SELECTs issued, one a page
    :ptype statements: int
    """

    rows: list[dict[str, Any]]
    statements: int


async def read_keyset_pages(
    l3: Any,
    select_from: str,
    key: Sequence[str],
    *,
    where: Mapping[str, Any] | None = None,
    where_any: Mapping[str, tuple[str, Sequence[Any]]] | None = None,
    page_size: int,
    quote: Callable[[str], str] = quote_identifier,
    mark: Callable[[str, int], str] | None = None,
    after: Sequence[Any] | None = None,
) -> KeysetRead:
    """every row ``select_from`` names (within ``where``), in ``key`` order, ``page_size`` rows a statement.

    :func:`keyset_pages`, its pages gathered into one list.

    :param l3: anything with an asyncpg-shaped ``fetch(query, *params)``: an L3 backend, or a
        caller's connection
    :ptype l3: Any
    :param select_from: ``SELECT <projection> FROM <table>``, spelled by the caller in its own policy;
        the projection must carry every ``key`` column under its own name
    :ptype select_from: str
    :param key: the columns to page on, TRUSTED identifiers; unique within ``where``, so paging
        steps past no row
    :ptype key: Sequence[str]
    :param where: equality filters naming the rows; every row when None
    :ptype where: Mapping[str, Any] | None
    :param where_any: set filters beside them, column -> (its array type, the values it may hold): each one
        ``"column" = ANY($n::<array type>)``, every value bound at once as one array, so a page is one statement
        however many values there are
    :ptype where_any: Mapping[str, tuple[str, Sequence[Any]]] | None
    :param page_size: rows a page keeps; the statement asks for one more, so ``page_size + 1`` must
        not pass the transport's cap
    :ptype page_size: int
    :param quote: how ``key`` and ``where``'s columns are spelled: quoted (the default), or
        :func:`~threetears.core.sql_fragments.as_written` beside a schema's unquoted names
    :ptype quote: Callable[[str], str]
    :param mark: the cursor's placeholder for a key column and its number, with any write cast it
        needs (a schema's ``render_param``, so a jsonb key binds ``$n::jsonb``); a plain ``$n``
        when None
    :ptype mark: Callable[[str, int], str] | None
    :param after: the key the read starts past; the first row when None
    :ptype after: Sequence[Any] | None
    :return: the rows and the statement count
    :rtype: KeysetRead
    :raises ValueError: when ``page_size`` is under one
    """
    rows: list[dict[str, Any]] = []
    statements = 0
    async for page in keyset_pages(
        l3, select_from, key, where=where, where_any=where_any, page_size=page_size, quote=quote, mark=mark, after=after
    ):
        statements += 1
        rows.extend(page)
    return KeysetRead(rows=rows, statements=statements)


async def keyset_pages(
    l3: Any,
    select_from: str,
    key: Sequence[str],
    *,
    where: Mapping[str, Any] | None = None,
    where_any: Mapping[str, tuple[str, Sequence[Any]]] | None = None,
    page_size: int,
    quote: Callable[[str], str] = quote_identifier,
    mark: Callable[[str, int], str] | None = None,
    after: Sequence[Any] | None = None,
) -> AsyncGenerator[list[dict[str, Any]], None]:
    """the rows ``select_from`` names (within ``where``), in ``key`` order, one page a statement as each is read.

    For a reader that must not hold every row at once: each page is handed over before the next is
    asked for, so what it keeps of a page is up to it. One page per statement, in order; the last
    may be empty (the rows ended on a page boundary), and a read of no rows is one empty page.

    :param l3: anything with an asyncpg-shaped ``fetch(query, *params)``
    :ptype l3: Any
    :param select_from: ``SELECT <projection> FROM <table>``, as :func:`read_keyset_pages` takes it
    :ptype select_from: str
    :param key: the columns to page on, TRUSTED identifiers, unique within ``where``
    :ptype key: Sequence[str]
    :param where: equality filters naming the rows; every row when None
    :ptype where: Mapping[str, Any] | None
    :param where_any: set filters beside them, as :func:`read_keyset_pages` takes them
    :ptype where_any: Mapping[str, tuple[str, Sequence[Any]]] | None
    :param page_size: rows a page keeps; ``page_size + 1`` must not pass the transport's cap
    :ptype page_size: int
    :param quote: how ``key`` and ``where``'s columns are spelled
    :ptype quote: Callable[[str], str]
    :param mark: the cursor's placeholder for a key column and its number; a plain ``$n`` when None
    :ptype mark: Callable[[str, int], str] | None
    :param after: the key the read starts past (a drain's saved place: the last key a previous read took); the
        first row when None
    :ptype after: Sequence[Any] | None
    :return: the pages, as they are read
    :rtype: AsyncGenerator[list[dict[str, Any]], None]
    :raises ValueError: when ``page_size`` is under one
    """
    statement = _KeysetStatement(
        select_from, key, where=where, where_any=where_any, page_size=page_size, quote=quote, mark=mark
    )
    # looked up on the class: a mock's instance answers any name it is asked
    paged = callable(getattr(type(l3), "fetch_page", None))
    rows = page_size
    cursor: Sequence[Any] | None = None if after is None else tuple(after)
    more = True
    while more:
        if not paged:
            page = await l3.fetch(*statement.page(cursor))
            kept = [dict(row) for row in page[:page_size]]
            more = len(page) > page_size and bool(kept)
        else:
            answered = await l3.fetch_page(*statement.page(cursor, rows=rows))
            page = answered.rows
            kept = [dict(row) for row in page[:rows]]
            # a cut page is the statement's first rows in key order: more may follow its last
            more = (answered.truncated or len(page) > rows) and bool(kept)
            rows = _next_rows(answered, len(page), rows)
        if more:
            cursor = tuple(kept[-1][column] for column in key)
        del page
        yield kept


async def keyset_arrow_pages(
    l3: Any,
    select_from: str,
    key: Sequence[str],
    *,
    types: Mapping[str, Any],
    where: Mapping[str, Any] | None = None,
    where_any: Mapping[str, tuple[str, Sequence[Any]]] | None = None,
    page_size: int,
    quote: Callable[[str], str] = quote_identifier,
    mark: Callable[[str, int], str] | None = None,
    after: Sequence[Any] | None = None,
) -> AsyncGenerator[Any, None]:
    """:func:`keyset_pages`, each page one Arrow table: the same statements, in the same order, its rows never
    Python objects.

    Each statement goes to :func:`~threetears.core.backends.arrow_reply.l3_fetch_arrow`: the rail's proxy answers the
    rows its ``fetch`` would as Arrow typed by ``types``, parsed off the interpreter lock; any other L3's dict rows are
    made the same table. The cursor is the last kept row's key, read off the table, as :func:`keyset_pages` reads it
    off a dict. Needs ``pyarrow``.

    :param l3: the rail's proxy (``fetch_arrow``), or anything with an asyncpg-shaped ``fetch``
    :ptype l3: Any
    :param select_from: ``SELECT <projection> FROM <table>``, as :func:`read_keyset_pages` takes it; the projection is
        the columns of ``types``, in order
    :ptype select_from: str
    :param key: the columns to page on, TRUSTED identifiers, unique within ``where``; each one of ``types``
    :ptype key: Sequence[str]
    :param types: column name -> its Arrow type, every column selected, in order
    :ptype types: Mapping[str, pyarrow.DataType]
    :param where: equality filters naming the rows; every row when None
    :ptype where: Mapping[str, Any] | None
    :param where_any: set filters beside them, as :func:`read_keyset_pages` takes them
    :ptype where_any: Mapping[str, tuple[str, Sequence[Any]]] | None
    :param page_size: rows a page keeps; ``page_size + 1`` must not pass the transport's cap
    :ptype page_size: int
    :param quote: how ``key`` and ``where``'s columns are spelled
    :ptype quote: Callable[[str], str]
    :param mark: the cursor's placeholder for a key column and its number; a plain ``$n`` when None
    :ptype mark: Callable[[str, int], str] | None
    :param after: the key the read starts past (a drain's saved place: the last key a previous read took); the
        first row when None
    :ptype after: Sequence[Any] | None
    :return: the pages, as they are read, each a ``pyarrow.Table``
    :rtype: AsyncGenerator[pyarrow.Table, None]
    :raises ValueError: when ``page_size`` is under one
    """
    statement = _KeysetStatement(
        select_from, key, where=where, where_any=where_any, page_size=page_size, quote=quote, mark=mark
    )
    paged = callable(getattr(type(l3), "fetch_arrow_page", None))
    rows = page_size
    cursor: Sequence[Any] | None = None if after is None else tuple(after)
    more = True
    while more:
        if not paged:
            page = await l3_fetch_arrow(l3, *statement.page(cursor), types=types)
            kept = page.slice(0, page_size)
            more = page.num_rows > page_size and kept.num_rows > 0
        else:
            answered = await l3.fetch_arrow_page(*statement.page(cursor, rows=rows), types=types)
            page = answered.rows
            kept = page.slice(0, rows)
            # a cut page is the statement's first rows in key order: more may follow its last
            more = (answered.truncated or page.num_rows > rows) and kept.num_rows > 0
            rows = _next_rows(answered, page.num_rows, rows)
        if more:
            last = kept.slice(kept.num_rows - 1, 1).select(list(key)).to_pylist()[0]
            cursor = tuple(last[column] for column in key)
        del page
        yield kept


class _KeysetStatement:
    """the statements one keyset read sends: the first page's, and each next page's past a cursor.

    One builder for :func:`keyset_pages` and :func:`keyset_arrow_pages`, so the two step, stop and spell a
    statement alike.
    """

    def __init__(
        self,
        select_from: str,
        key: Sequence[str],
        *,
        where: Mapping[str, Any] | None,
        where_any: Mapping[str, tuple[str, Sequence[Any]]] | None,
        page_size: int,
        quote: Callable[[str], str],
        mark: Callable[[str, int], str] | None,
    ) -> None:
        if page_size < 1:
            raise ValueError(f"a keyset page keeps at least one row, got {page_size}")
        self._select_from = select_from
        self._key = list(key)
        self._order = ", ".join(quote(column) for column in key)
        self._page_size = page_size
        self._filters, self._values = equality_conditions(where, first=1, quote=quote)
        for column, (array, values) in (where_any or {}).items():
            condition = f"{quote(column)} = ANY(${len(self._values) + 1}::{array})"
            self._filters = f"{self._filters} AND {condition}" if self._filters else condition
            self._values.append(list(values))
        self._mark = mark
        self._quoted = [quote(column) for column in key]

    def _past(self, before: int) -> str:
        """the condition naming the rows past the cursor, its values the parameters after the first ``before``.

        The leading bound, the row comparison and its expansion (the module's "How a page steps past its
        cursor"): the same rows, said so that Postgres and YugabyteDB each seek by one of them.

        :param before: how many parameters the filters bind
        :ptype before: int
        :return: the condition
        :rtype: str
        """
        marks = [
            self._mark(column, before + index) if self._mark is not None else f"${before + index}"
            for index, column in enumerate(self._key, start=1)
        ]
        row = f"({self._order}) > ({', '.join(marks)})"
        if len(self._key) == 1:
            return row
        expanded = f"{self._quoted[-1]} > {marks[-1]}"
        for column, mark in zip(reversed(self._quoted[:-1]), reversed(marks[:-1]), strict=True):
            expanded = f"{column} > {mark} OR ({column} = {mark} AND ({expanded}))"
        return f"{self._quoted[0]} >= {marks[0]} AND {row} AND ({expanded})"

    def page(self, cursor: Sequence[Any] | None, *, rows: int | None = None) -> tuple[Any, ...]:
        """the statement and parameters of the page after ``cursor`` (the first page for None).

        :param cursor: the last kept row's key, or None
        :ptype cursor: Sequence[Any] | None
        :param rows: the rows the page keeps; its ``page_size`` when None. It asks for one more
        :ptype rows: int | None
        :return: the statement, then its parameters
        :rtype: tuple[Any, ...]
        """
        conditions = [self._filters] if self._filters else []
        params = list(self._values)
        if cursor is not None:
            conditions.append(self._past(len(self._values)))
            params += list(cursor)
        where_sql = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        limit = (self._page_size if rows is None else rows) + 1
        return (f"{self._select_from}{where_sql} ORDER BY {self._order} LIMIT {limit}", *params)
