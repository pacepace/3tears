"""One keyset pager for L3: every row a statement shape names, a page at a time, past the last key read.

The L3 rail answers at most :data:`~threetears.core.backends.protocol.L3_RAIL_ROW_CAP` rows a
statement and does not say when it cut, so a read that might pass it asks for one row more than a
page and pages on a unique key until a page comes back without that row. The whole-table copies
(:func:`~threetears.core.collections.complete_copy.read_l3_rows`) and the key-led read's paging of
one leading value (``SqlL3Backend.fetch_led_by``) both page through :func:`read_keyset_pages`, so the
two cannot drift apart in how they step, stop, or spell a name.

**What bounds a page is the caller's filters, not this loop.** Each page is ``ORDER BY`` the key:
answered in order by a btree key, but on a YugabyteDB hash-sharded key only when an equality filter
pins the hashed column and the key named here is the range part that follows it.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from threetears.core.backends.arrow_reply import l3_fetch_arrow
from threetears.core.sql_fragments import equality_conditions, quote_identifier

__all__ = ["KeysetRead", "keyset_arrow_pages", "keyset_pages", "read_keyset_pages"]


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
    :return: the rows and the statement count
    :rtype: KeysetRead
    :raises ValueError: when ``page_size`` is under one
    """
    rows: list[dict[str, Any]] = []
    statements = 0
    async for page in keyset_pages(
        l3, select_from, key, where=where, where_any=where_any, page_size=page_size, quote=quote, mark=mark
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
    :return: the pages, as they are read
    :rtype: AsyncGenerator[list[dict[str, Any]], None]
    :raises ValueError: when ``page_size`` is under one
    """
    statement = _KeysetStatement(
        select_from, key, where=where, where_any=where_any, page_size=page_size, quote=quote, mark=mark
    )
    cursor: Sequence[Any] | None = None
    more = True
    while more:
        page = await l3.fetch(*statement.page(cursor))
        kept = [dict(row) for row in page[:page_size]]
        more = len(page) > page_size and bool(kept)
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
    :return: the pages, as they are read, each a ``pyarrow.Table``
    :rtype: AsyncGenerator[pyarrow.Table, None]
    :raises ValueError: when ``page_size`` is under one
    """
    statement = _KeysetStatement(
        select_from, key, where=where, where_any=where_any, page_size=page_size, quote=quote, mark=mark
    )
    cursor: Sequence[Any] | None = None
    more = True
    while more:
        page = await l3_fetch_arrow(l3, *statement.page(cursor), types=types)
        kept = page.slice(0, page_size)
        more = page.num_rows > page_size and kept.num_rows > 0
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
        self._tail = f" ORDER BY {self._order} LIMIT {page_size + 1}"
        self._filters, self._values = equality_conditions(where, first=1, quote=quote)
        for column, (array, values) in (where_any or {}).items():
            condition = f"{quote(column)} = ANY(${len(self._values) + 1}::{array})"
            self._filters = f"{self._filters} AND {condition}" if self._filters else condition
            self._values.append(list(values))
        self._mark = mark

    def page(self, cursor: Sequence[Any] | None) -> tuple[Any, ...]:
        """the statement and parameters of the page after ``cursor`` (the first page for None).

        :param cursor: the last kept row's key, or None
        :ptype cursor: Sequence[Any] | None
        :return: the statement, then its parameters
        :rtype: tuple[Any, ...]
        """
        conditions = [self._filters] if self._filters else []
        params = list(self._values)
        if cursor is not None:
            marks = ", ".join(
                self._mark(column, len(self._values) + index)
                if self._mark is not None
                else f"${len(self._values) + index}"
                for index, column in enumerate(self._key, start=1)
            )
            conditions.append(f"({self._order}) > ({marks})")
            params += list(cursor)
        where_sql = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        return (f"{self._select_from}{where_sql}{self._tail}", *params)
