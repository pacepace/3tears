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

**A hashed lead (YugabyteDB).** YugabyteDB hashes a key's first column unless its declaration says ``ASC``: the
rows of one lead value sit in one bucket, in the order of the rest of the key, and the buckets in no order at all.
The leading bound cannot seek a hashed column, so a page past a cursor reads every row of the table and sorts it
(2026.1: a 200,000-row table, 50 values, every page scanned 200,000 rows; a cold start of 1,000,000 rows read the
table a thousand times). A reader whose declaration leaves the lead hashed names its array type (``lead_array``),
and on YugabyteDB the read walks the lead: its values from ``SELECT DISTINCT``, which YugabyteDB answers from the
bucket heads (one read, a row a value), in the database's order; then each value's rows, ``lead = $1`` and the rest
of the key past the cursor, which seeks inside the bucket (1,000 rows scanned for a page of 1,000). Values whose
rows are fewer than a page go together, ``lead = ANY($1)``, as many as about fill one (:func:`_next_group`), so a
lead of many small values costs a statement a page, not one a value. The rows come in the key's order, as the
cursor's form reads them. Postgres pages by the cursor as above: there ``DISTINCT`` reads the whole table.

**A hashed lead a set filter pins** (``where_any`` naming the key's first column, ``lead_array`` named, on
YugabyteDB): the filter's values are walked the same way, put in the database's order by one statement over the
values alone (``unnest``, no table read). Paged by the cursor instead, ``lead = ANY(values)`` read every row of
every value on every page and sorted them across values (YugabyteDB, 33,528 rows in three pages: 33,528 scanned a
page, two sorts on disk). An ordered lead -- Postgres, or a lead declared ``ASC`` -- pages by the cursor: its index
answers ``lead = ANY(values)`` in key order and stops at the page, and a set that fits one page is one statement
(the walk measured about twice the rows it answered on Postgres 16, and costs a set of small values a statement
for their order and more for their rows). On Postgres such a read names its cursor by the row comparison alone,
the one Postgres seeks by: beside ``lead = ANY(values)`` the leading bound and the expansion only shrink its row
estimate until, a few pages from the end, it plans a bitmap scan of every row left in the set and sorts them
(Postgres 16, ten values of 3,000 rows: 85,019 rows scanned for 30,029; by the row comparison alone, 31,028).

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
from dataclasses import dataclass, replace
from typing import Any, Final

from threetears.core.backends.arrow_reply import l3_fetch_arrow
from threetears.core.backends.dialect import known_on_yugabyte, l3_on_yugabyte
from threetears.core.backends.protocol import L3Page
from threetears.core.keyed._filters import Fragment, narrowing
from threetears.core.sql_fragments import equality_conditions, quote_identifier
from threetears.observe import get_logger

__all__ = ["KeysetRead", "keyset_arrow_pages", "keyset_pages", "read_keyset_pages"]

#: the share of a byte-capped reply a page is sized to fill: rows vary in width, so a page sized to the whole
#: cap from the rows before would be cut about half the time
_BUDGET_SHARE: Final = 0.8

#: the most rows a byte-sized page asks for, whatever its rows' width
_MOST_ROWS_A_PAGE: Final = 1_000_000

#: the most lead values one statement of a hashed lead's walk names (``= ANY`` the values)
_MOST_LEADS_A_STATEMENT: Final = 1_000

#: the most lead values a hashed lead's walk takes; each page of them reads every value again from the bucket
#: heads, so past this the walk's cost grows with the square of the values, and the read goes on by the cursor
_MOST_LEADS: Final = 20_000

_logger = get_logger(__name__)


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
    conditions: Sequence[Fragment] = (),
    lead_array: str | None = None,
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
    :param conditions: filters beside ``where``, each a fragment from ``$1`` (``"due_at" IS NOT NULL``, a partial
        index's own predicate; a range on a key column), so the read goes through that index
    :ptype conditions: Sequence[Fragment]
    :param lead_array: the SQL array type of the key's leading column, named when its declaration leaves it hashed
        on YugabyteDB (:func:`keyset_pages`)
    :ptype lead_array: str | None
    :return: the rows and the statement count (a hashed lead's walk counts the pages it read, not its reads of the
        lead's values)
    :rtype: KeysetRead
    :raises ValueError: when ``page_size`` is under one
    """
    rows: list[dict[str, Any]] = []
    statements = 0
    async for page in keyset_pages(
        l3,
        select_from,
        key,
        where=where,
        where_any=where_any,
        page_size=page_size,
        quote=quote,
        mark=mark,
        after=after,
        conditions=conditions,
        lead_array=lead_array,
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
    conditions: Sequence[Fragment] = (),
    lead_array: str | None = None,
) -> AsyncGenerator[list[dict[str, Any]], None]:
    """the rows ``select_from`` names (within ``where``), in ``key`` order, one page a statement as each is read.

    For a reader that must not hold every row at once: each page is handed over before the next is
    asked for, so what it keeps of a page is up to it. One page per statement, in order; the last
    may be empty (the rows ended on a page boundary), and a read of no rows is one empty page. A read
    that walks a hashed lead (the module's "A hashed lead") hands over no empty page but that one.

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
    :param conditions: filters beside ``where``, each a fragment from ``$1`` (``"due_at" IS NOT NULL``, a partial
        index's own predicate; a range on a key column), so the read goes through that index
    :ptype conditions: Sequence[Fragment]
    :param lead_array: the SQL array type of the key's leading column (``text[]``), named when the table's
        declaration leaves that column hashed on YugabyteDB (its default for a key's first column): on YugabyteDB
        the read then walks each lead value in turn (the module's "A hashed lead"); None pages by the cursor alone
    :ptype lead_array: str | None
    :return: the pages, as they are read
    :rtype: AsyncGenerator[list[dict[str, Any]], None]
    :raises ValueError: when ``page_size`` is under one
    """
    shape = _ReadShape(select_from, key, where, where_any, page_size, quote, mark, conditions)
    reads = _DictReads(l3, page_size)
    pinned = await _pinned_lead(l3, shape, lead_array)
    if pinned.walk is not None:
        async for page in _lead_walk(reads, shape, pinned.walk[0], after, given=pinned.walk[1]):
            yield page
        return
    walked = await _hashed_lead(l3, shape, lead_array)
    if walked is not None:
        async for page in _lead_walk(reads, shape, walked, after):
            yield page
        return
    statement = shape.statement(row_seek=pinned.row_seek)
    async for page in _cursor_pages(reads, statement, None if after is None else tuple(after)):
        yield page


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
    conditions: Sequence[Fragment] = (),
    lead_array: str | None = None,
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
    :param conditions: filters beside ``where``, each a fragment from ``$1`` (``"due_at" IS NOT NULL``, a partial
        index's own predicate; a range on a key column), so the read goes through that index
    :ptype conditions: Sequence[Fragment]
    :param lead_array: the SQL array type of the key's leading column, as :func:`keyset_pages` takes it
    :ptype lead_array: str | None
    :return: the pages, as they are read, each a ``pyarrow.Table``
    :rtype: AsyncGenerator[pyarrow.Table, None]
    :raises ValueError: when ``page_size`` is under one
    """
    shape = _ReadShape(select_from, key, where, where_any, page_size, quote, mark, conditions)
    reads = _ArrowReads(l3, page_size, types)
    pinned = await _pinned_lead(l3, shape, lead_array)
    if pinned.walk is not None:
        async for page in _lead_walk(reads, shape, pinned.walk[0], after, given=pinned.walk[1]):
            yield page
        return
    walked = await _hashed_lead(l3, shape, lead_array)
    if walked is not None:
        async for page in _lead_walk(reads, shape, walked, after):
            yield page
        return
    statement = shape.statement(row_seek=pinned.row_seek)
    async for page in _cursor_pages(reads, statement, None if after is None else tuple(after)):
        yield page


@dataclass(frozen=True)
class _ReadShape:
    """what one keyset read was asked for, from which each of its statements is built."""

    select_from: str
    key: Sequence[str]
    where: Mapping[str, Any] | None
    where_any: Mapping[str, tuple[str, Sequence[Any]]] | None
    page_size: int
    quote: Callable[[str], str]
    mark: Callable[[str, int], str] | None
    conditions: Sequence[Fragment]

    def statement(
        self, *, key: Sequence[str] | None = None, conditions: Sequence[Fragment] = (), row_seek: bool = False
    ) -> _KeysetStatement:
        """the statement paging ``key`` (the read's own when None), ``conditions`` beside the read's own.

        :param key: the columns this statement pages on
        :ptype key: Sequence[str] | None
        :param conditions: filters beside the read's
        :ptype conditions: Sequence[Fragment]
        :param row_seek: name the cursor by the row comparison alone (:meth:`_KeysetStatement._past`)
        :ptype row_seek: bool
        :return: the statement
        :rtype: _KeysetStatement
        """
        return _KeysetStatement(
            self.select_from,
            self.key if key is None else key,
            where=self.where,
            where_any=self.where_any,
            page_size=self.page_size,
            quote=self.quote,
            mark=self.mark,
            conditions=[*self.conditions, *conditions],
            row_seek=row_seek,
        )

    def placeholder(self, column: str, number: int) -> str:
        """the placeholder binding a value of ``column`` as ``$number``, with the write cast it needs.

        :param column: the column
        :ptype column: str
        :param number: the parameter's number
        :ptype number: int
        :return: the placeholder
        :rtype: str
        """
        return self.mark(column, number) if self.mark is not None else f"${number}"


class _DictReads:
    """one statement's page as dict rows: ``fetch_page`` on a transport that answers pages, else ``fetch``."""

    def __init__(self, l3: Any, page_size: int) -> None:
        self._l3 = l3
        self._page_size = page_size
        # looked up on the class: a mock's instance answers any name it is asked
        self._paged = callable(getattr(type(l3), "fetch_page", None))

    async def page(self, statement: _KeysetStatement, cursor: Sequence[Any] | None, rows: int) -> tuple[Any, bool, int]:
        """the rows a page keeps, whether more may follow, and how many rows the next page asks for.

        :param statement: the statement
        :ptype statement: _KeysetStatement
        :param cursor: the key the page starts past; the first page when None
        :ptype cursor: Sequence[Any] | None
        :param rows: the rows it asks for (a paged transport's)
        :ptype rows: int
        :return: the kept rows, whether more may follow, the next page's rows
        :rtype: tuple[list[dict[str, Any]], bool, int]
        """
        if not self._paged:
            page = await self._l3.fetch(*statement.page(cursor))
            kept = [dict(row) for row in page[: self._page_size]]
            return kept, len(page) > self._page_size and bool(kept), rows
        answered = await self._l3.fetch_page(*statement.page(cursor, rows=rows))
        kept = [dict(row) for row in answered.rows[:rows]]
        # a cut page is the statement's first rows in key order: more may follow its last
        more = (answered.truncated or len(answered.rows) > rows) and bool(kept)
        return kept, more, _next_rows(answered, len(answered.rows), rows)

    @staticmethod
    def count(kept: Any) -> int:
        """how many rows ``kept`` holds."""
        return len(kept)

    @staticmethod
    def last(kept: Any, key: Sequence[str]) -> tuple[Any, ...]:
        """the last row's ``key``."""
        return tuple(kept[-1][column] for column in key)

    async def values(self, query: str, params: Sequence[Any], column: str) -> list[Any]:
        """the one column ``query`` selects, row by row.

        :param query: the SELECT
        :ptype query: str
        :param params: its parameters
        :ptype params: Sequence[Any]
        :param column: the column it selects
        :ptype column: str
        :return: the values
        :rtype: list[Any]
        """
        return [row[column] for row in await self._l3.fetch(query, *params)]


class _ArrowReads:
    """one statement's page as an Arrow table: ``fetch_arrow_page`` on a transport that answers pages, else
    :func:`~threetears.core.backends.arrow_reply.l3_fetch_arrow`."""

    def __init__(self, l3: Any, page_size: int, types: Mapping[str, Any]) -> None:
        self._l3 = l3
        self._page_size = page_size
        self._types = types
        self._paged = callable(getattr(type(l3), "fetch_arrow_page", None))

    async def page(self, statement: _KeysetStatement, cursor: Sequence[Any] | None, rows: int) -> tuple[Any, bool, int]:
        """the rows a page keeps, whether more may follow, and how many rows the next page asks for.

        :param statement: the statement
        :ptype statement: _KeysetStatement
        :param cursor: the key the page starts past; the first page when None
        :ptype cursor: Sequence[Any] | None
        :param rows: the rows it asks for (a paged transport's)
        :ptype rows: int
        :return: the kept rows, whether more may follow, the next page's rows
        :rtype: tuple[pyarrow.Table, bool, int]
        """
        if not self._paged:
            page = await l3_fetch_arrow(self._l3, *statement.page(cursor), types=self._types)
            kept = page.slice(0, self._page_size)
            return kept, page.num_rows > self._page_size and kept.num_rows > 0, rows
        answered = await self._l3.fetch_arrow_page(*statement.page(cursor, rows=rows), types=self._types)
        kept = answered.rows.slice(0, rows)
        # a cut page is the statement's first rows in key order: more may follow its last
        more = (answered.truncated or answered.rows.num_rows > rows) and kept.num_rows > 0
        return kept, more, _next_rows(answered, answered.rows.num_rows, rows)

    @staticmethod
    def count(kept: Any) -> int:
        """how many rows ``kept`` holds."""
        return int(kept.num_rows)

    @staticmethod
    def last(kept: Any, key: Sequence[str]) -> tuple[Any, ...]:
        """the last row's ``key``, read off the table."""
        last = kept.slice(kept.num_rows - 1, 1).select(list(key)).to_pylist()[0]
        return tuple(last[column] for column in key)

    async def values(self, query: str, params: Sequence[Any], column: str) -> list[Any]:
        """the one column ``query`` selects, row by row (typed as the read's own column).

        :param query: the SELECT
        :ptype query: str
        :param params: its parameters
        :ptype params: Sequence[Any]
        :param column: the column it selects
        :ptype column: str
        :return: the values
        :rtype: list[Any]
        """
        answered = await l3_fetch_arrow(self._l3, query, *params, types={column: self._types[column]})
        return list(answered.column(column).to_pylist())


async def _cursor_pages(
    reads: _DictReads | _ArrowReads, statement: _KeysetStatement, cursor: Sequence[Any] | None
) -> AsyncGenerator[Any, None]:
    """every page of one statement past ``cursor``, each handed over as it is read; the last may be empty.

    :param reads: how a page is read
    :ptype reads: _DictReads | _ArrowReads
    :param statement: the statement
    :ptype statement: _KeysetStatement
    :param cursor: the key the read starts past; the first row when None
    :ptype cursor: Sequence[Any] | None
    :return: the pages
    :rtype: AsyncGenerator[Any, None]
    """
    rows = statement.page_size
    more = True
    while more:
        kept, more, rows = await reads.page(statement, cursor, rows)
        if more:
            cursor = reads.last(kept, statement.key)
        yield kept


@dataclass(frozen=True)
class _Pinned:
    """how a read whose set filter may pin its key's lead goes: the walk's ``(array type, values)`` when it walks the
    values one by one, and whether its cursor statement seeks by the row comparison alone (Postgres)."""

    walk: tuple[str, Sequence[Any]] | None = None
    row_seek: bool = False


#: a read no set filter pins the lead of: it pages as it always has
_UNPINNED: Final = _Pinned()


async def _pinned_lead(l3: Any, shape: _ReadShape, lead_array: str | None) -> _Pinned:
    """how a read goes whose ``where_any`` names the lead of a key of more than one column (any other read:
    :data:`_UNPINNED`).

    It walks when ``where_any`` names the lead of a key of more than one column, the lead is declared hashed
    (``lead_array``) and the L3 is YugabyteDB. Paged by the cursor there, ``lead = ANY(values)`` reads every row of
    every value on every page and sorts them across values (a hashed lead's buckets come in no order): three pages of
    33,528 rows scanned 33,528 rows each and sorted two on disk. Walked, each value's rows are one bucket read past the
    cursor on the rest of the key: every row scanned once, nothing sorted but a small group's rows. An ordered lead
    pages by the cursor; on Postgres that cursor is named by the row comparison alone, which Postgres seeks by
    (the module's "A hashed lead a set filter pins").

    :param l3: the L3, asked whether it is YugabyteDB only when a set filter pins the lead (once a reader:
        :func:`~threetears.core.backends.dialect.known_on_yugabyte`)
    :ptype l3: Any
    :param shape: the read
    :ptype shape: _ReadShape
    :param lead_array: the lead's array type when its declaration leaves it hashed
    :ptype lead_array: str | None
    :return: the walk, or the cursor's form
    :rtype: _Pinned
    """
    if len(shape.key) < 2 or not shape.where_any:
        return _UNPINNED
    pinned = shape.where_any.get(shape.key[0])
    if pinned is None:
        return _UNPINNED
    if not await known_on_yugabyte(l3):
        return _Pinned(row_seek=True)
    return _Pinned(walk=pinned) if lead_array is not None else _UNPINNED


async def _hashed_lead(l3: Any, shape: _ReadShape, lead_array: str | None) -> str | None:
    """the lead's array type when a read walks its lead value by value: a lead named hashed, of a key of more than one
    column, that no filter pins, read from YugabyteDB; None when it pages by the cursor alone.

    :param l3: the L3
    :ptype l3: Any
    :param shape: the read
    :ptype shape: _ReadShape
    :param lead_array: the lead's array type when its declaration leaves it hashed
    :ptype lead_array: str | None
    :return: the lead's array type, or None
    :rtype: str | None
    """
    if lead_array is None or len(shape.key) < 2:
        return None
    lead = shape.key[0]
    if lead in (shape.where or {}) or lead in (shape.where_any or {}):
        return None
    return lead_array if await l3_on_yugabyte(l3) else None


async def _lead_walk(
    reads: _DictReads | _ArrowReads,
    shape: _ReadShape,
    lead_array: str,
    after: Sequence[Any] | None,
    *,
    given: Sequence[Any] | None = None,
) -> AsyncGenerator[Any, None]:
    """the read's rows in key order, a lead value's rows at a time (the module's "A hashed lead").

    The lead values come from ``SELECT DISTINCT`` in pages of the read's own size, in the database's order -- or,
    when a set filter pins the lead (``given``), are that filter's values put in the database's order by one
    statement over the values alone, never the table (:class:`_GivenLeads`); a value whose rows fill a page is read
    on its own, seeking the rest of the key inside its bucket; values whose rows are fewer go together, as many a
    statement as about fill one (``= ANY`` the values, ordered by the whole key). Past :data:`_MOST_LEADS` values the
    read goes on by the cursor alone, from the last key it took.

    :param reads: how a page is read
    :ptype reads: _DictReads | _ArrowReads
    :param shape: the read
    :ptype shape: _ReadShape
    :param lead_array: the lead's SQL array type
    :ptype lead_array: str
    :param after: the key the read starts past; the first row when None
    :ptype after: Sequence[Any] | None
    :param given: the values a set filter pins the lead to; None to read them from the table
    :ptype given: Sequence[Any] | None
    :return: the pages, none empty but a read of no rows
    :rtype: AsyncGenerator[Any, None]
    """
    lead, rest = shape.key[0], list(shape.key[1:])
    quoted = shape.quote(lead)
    leads: _LeadValues | _GivenLeads = (
        _LeadValues(reads, shape, after) if given is None else _GivenLeads(reads, shape, lead_array, given, after)
    )
    # each value's statement pins the lead itself; the set filter naming it stays on the read's own statements
    walk = (
        shape
        if given is None
        else replace(
            shape,
            where_any={column: held for column, held in (shape.where_any or {}).items() if column != lead} or None,
        )
    )
    group = 1
    walked = 0
    last_key: tuple[Any, ...] | None = None if after is None else tuple(after)
    handed = False
    while True:
        values = await leads.take(group)
        if not values:
            break
        if walked >= _MOST_LEADS:
            _logger.warning(
                "keyset: a hashed lead's walk took %d values of %s; the rest is read by the cursor", walked, lead
            )
            async for page in _cursor_pages(reads, shape.statement(), last_key):
                yield page
            return
        starts_past = after is not None and walked == 0 and values[0] == after[0]
        if len(values) == 1:
            statement = walk.statement(
                key=rest, conditions=[Fragment(sql=f"{quoted} = {shape.placeholder(lead, 1)}", params=(values[0],))]
            )
            cursor: Sequence[Any] | None = tuple(after[1:]) if after is not None and starts_past else None
        else:
            statement = walk.statement(
                conditions=[Fragment(sql=f"{quoted} = ANY($1::{lead_array})", params=(list(values),))]
            )
            cursor = tuple(after) if after is not None and starts_past else None
        took = 0
        async for page in _cursor_pages(reads, statement, cursor):
            counted = reads.count(page)
            if counted == 0:
                continue
            took += counted
            last_key = (values[0], *reads.last(page, rest)) if len(values) == 1 else reads.last(page, shape.key)
            handed = True
            yield page
        walked += len(values)
        group = _next_group(len(values), took, shape.page_size)
    if not handed:
        # no rows: the one empty page a read of none hands over
        async for page in _cursor_pages(reads, shape.statement(), None if after is None else tuple(after)):
            yield page


def _next_group(values: int, rows: int, page_size: int) -> int:
    """how many lead values the next statement names: as many as fill :data:`_BUDGET_SHARE` of a page at the rows a
    value held.

    :param values: the values the last statement named
    :ptype values: int
    :param rows: the rows they held
    :ptype rows: int
    :param page_size: the rows a page keeps
    :ptype page_size: int
    :return: the next group's values, from one to :data:`_MOST_LEADS_A_STATEMENT`, at most four times the last
    :rtype: int
    """
    if rows >= page_size:
        return 1
    # sized to a share of the page, as a byte-sized page is: values vary in their rows
    fitting = int(values * page_size * _BUDGET_SHARE / max(rows, 1))
    return max(1, min(fitting, values * 4, _MOST_LEADS_A_STATEMENT))


class _LeadValues:
    """a read's lead values in the database's order, read ``SELECT DISTINCT`` a page at a time as they are taken."""

    def __init__(self, reads: _DictReads | _ArrowReads, shape: _ReadShape, after: Sequence[Any] | None) -> None:
        self._reads = reads
        self._shape = shape
        self._held: list[Any] = []
        self._ended = False
        self._past: Any = None if after is None else after[0]
        self._first = after is not None

    async def take(self, count: int) -> list[Any]:
        """the next ``count`` values (fewer at the end; none once every value is taken).

        :param count: how many
        :ptype count: int
        :return: the values
        :rtype: list[Any]
        """
        while len(self._held) < count and not self._ended:
            await self._read()
        taken, self._held = self._held[:count], self._held[count:]
        return taken

    async def _read(self) -> None:
        shape = self._shape
        lead = shape.key[0]
        quoted = shape.quote(lead)
        statement = shape.statement()
        inner, params = statement.filtered()
        clause = ""
        if self._past is not None:
            params.append(self._past)
            clause = f" WHERE {quoted} {'>=' if self._first else '>'} {shape.placeholder(lead, len(params))}"
        query = (
            f"SELECT DISTINCT {quoted} FROM ({inner}) AS {shape.quote('_leads')}{clause} "  # noqa: S608 - trusted
            f"ORDER BY {quoted} LIMIT {shape.page_size}"
        )
        values = await self._reads.values(query, params, lead)
        self._held.extend(values)
        self._ended = len(values) < shape.page_size
        if values:
            self._past = values[-1]
            self._first = False


class _GivenLeads:
    """the values a set filter pins a read's lead to, in the database's order, put so by one statement over the values
    alone (``unnest``, no table read); from the cursor's lead value on when the read resumes."""

    def __init__(
        self,
        reads: _DictReads | _ArrowReads,
        shape: _ReadShape,
        array: str,
        values: Sequence[Any],
        after: Sequence[Any] | None,
    ) -> None:
        self._reads = reads
        self._shape = shape
        self._array = array
        self._values = list(values)
        self._after = after
        self._held: list[Any] | None = None

    async def take(self, count: int) -> list[Any]:
        """the next ``count`` values (fewer at the end; none once every value is taken).

        :param count: how many
        :ptype count: int
        :return: the values
        :rtype: list[Any]
        """
        if self._held is None:
            self._held = await self._ordered()
        taken, self._held = self._held[:count], self._held[count:]
        return taken

    async def _ordered(self) -> list[Any]:
        """the distinct non-NULL values (``= ANY`` never matches a NULL), in the database's order of the lead's type.

        :return: the values
        :rtype: list[Any]
        """
        if not self._values:
            return []
        shape = self._shape
        lead = shape.key[0]
        quoted = shape.quote(lead)
        params: list[Any] = [self._values]
        clause = ""
        if self._after is not None:
            params.append(self._after[0])
            clause = f" AND {quoted} >= {shape.placeholder(lead, 2)}"
        query = (
            f"SELECT DISTINCT {quoted} FROM unnest($1::{self._array}) AS {shape.quote('_leads')}({quoted}) "
            f"WHERE {quoted} IS NOT NULL{clause} ORDER BY {quoted}"
        )
        return await self._reads.values(query, params, lead)


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
        conditions: Sequence[Fragment] = (),
        row_seek: bool = False,
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
        for fragment in conditions:
            condition = narrowing(fragment, len(self._values))
            self._filters = f"{self._filters} AND {condition}" if self._filters else condition
            self._values += list(fragment.params)
        self._mark = mark
        self._quoted = [quote(column) for column in key]
        self._row_seek = row_seek

    def _past(self, before: int) -> str:
        """the condition naming the rows past the cursor, its values the parameters after the first ``before``.

        The leading bound, the row comparison and its expansion (the module's "How a page steps past its
        cursor"): the same rows, said so that Postgres and YugabyteDB each seek by one of them. A statement built
        ``row_seek`` (a set-pinned lead read on Postgres) names only the row comparison.

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
        if len(self._key) == 1 or self._row_seek:
            return row
        expanded = f"{self._quoted[-1]} > {marks[-1]}"
        for column, mark in zip(reversed(self._quoted[:-1]), reversed(marks[:-1]), strict=True):
            expanded = f"{column} > {mark} OR ({column} = {mark} AND ({expanded}))"
        return f"{self._quoted[0]} >= {marks[0]} AND {row} AND ({expanded})"

    @property
    def key(self) -> list[str]:
        """the columns this statement pages on."""
        return self._key

    @property
    def page_size(self) -> int:
        """the rows a page keeps."""
        return self._page_size

    def filtered(self) -> tuple[str, list[Any]]:
        """``select_from`` within the read's filters, with no order, cursor or limit, and its parameters.

        :return: the statement, then its parameters
        :rtype: tuple[str, list[Any]]
        """
        where_sql = f" WHERE {self._filters}" if self._filters else ""
        return f"{self._select_from}{where_sql}", list(self._values)

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
