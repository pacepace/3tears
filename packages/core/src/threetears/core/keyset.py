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

from threetears.core.sql_fragments import equality_conditions, quote_identifier

__all__ = ["KeysetRead", "keyset_pages", "read_keyset_pages"]


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
    async for page in keyset_pages(l3, select_from, key, where=where, page_size=page_size, quote=quote, mark=mark):
        statements += 1
        rows.extend(page)
    return KeysetRead(rows=rows, statements=statements)


async def keyset_pages(
    l3: Any,
    select_from: str,
    key: Sequence[str],
    *,
    where: Mapping[str, Any] | None = None,
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
    if page_size < 1:
        raise ValueError(f"a keyset page keeps at least one row, got {page_size}")
    order = ", ".join(quote(column) for column in key)
    tail = f" ORDER BY {order} LIMIT {page_size + 1}"
    filters, values = equality_conditions(where, first=1, quote=quote)
    cursor: Sequence[Any] | None = None
    more = True
    while more:
        conditions = [filters] if filters else []
        params = list(values)
        if cursor is not None:
            marks = ", ".join(
                mark(column, len(values) + index) if mark is not None else f"${len(values) + index}"
                for index, column in enumerate(key, start=1)
            )
            conditions.append(f"({order}) > ({marks})")
            params += list(cursor)
        where_sql = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        page = await l3.fetch(f"{select_from}{where_sql}{tail}", *params)
        kept = [dict(row) for row in page[:page_size]]
        more = len(page) > page_size and bool(kept)
        if more:
            cursor = tuple(kept[-1][column] for column in key)
        del page
        yield kept
