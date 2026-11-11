"""a source's own change record, read past the marks a load saved: only what moved since.

An external source says what changed by its own record: stamps that grow on each write (``upload_timestamp``,
``winnerdatetime``), or an id that grows (a run id, a write mark's id). A load saves the newest it read, and the next
load reads only the rows past it::

    SELECT <columns> FROM <relation> WHERE "upload_timestamp" > $1 OR "winnerdatetime" > $2

:func:`read_past_marks` reads it and answers the marks to save next: each stamp's newest value among the rows read,
or the mark it was given when none moved. A load with no mark yet has no change record to read past: that is a
cold start, ledgered on its own (the no-whole-reads rule's exemptions ledger), never this read with no bound. A
missing mark is refused here, not read as "everything".

:func:`read_after` is the id-ordered form, a page at a time: ``WHERE <by> AND "id" > $n ORDER BY "id" LIMIT n``
(write marks, a runs table's new runs), its answer the rows and the last id read.

The function that calls either on a relation no declaration names (a warehouse's table) carries
``@source_change_read(stamps=...)`` / ``ids=`` (:mod:`threetears.core.operator`); its proof test shows the mark
advances and an unchanged load reads no row.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from threetears.core.keyed._filters import Derived, Fragment, by_conditions, relation, narrowing
from threetears.core.sql_fragments import quote_identifier

__all__ = ["MarkedRead", "PageAfter", "read_after", "read_past_marks"]


@dataclass(frozen=True)
class MarkedRead:
    """what a read past the marks answered.

    :param rows: the rows past any mark
    :ptype rows: list[dict[str, Any]]
    :param marks: each stamp's mark to save next: the newest read, or the one given when none moved
    :ptype marks: dict[str, Any]
    """

    rows: list[dict[str, Any]]
    marks: dict[str, Any]


async def read_past_marks(
    conn: Any,
    relation: str,
    *,
    columns: Sequence[str],
    marks: Mapping[str, Any],
    casts: Mapping[str, str] | None = None,
    quote: Callable[[str], str] = quote_identifier,
    by: Mapping[str, Any] | None = None,
    where: Sequence[Fragment] = (),
    floor: Mapping[str, Any] | None = None,
) -> MarkedRead:
    """the rows of ``relation`` whose stamps moved past ``marks``, and the marks to save next.

    :param conn: anything with an asyncpg-shaped ``fetch`` (an L3, a warehouse's reader)
    :ptype conn: Any
    :param relation: the relation, a TRUSTED name (``quote`` spells it; :func:`~threetears.core.sql_fragments.as_written`
        for a schema-qualified one)
    :ptype relation: str
    :param columns: the columns read, each stamp among them
    :ptype columns: Sequence[str]
    :param marks: stamp column -> the newest value a load read; a row moved when any stamp passed its mark
    :ptype marks: Mapping[str, Any]
    :param casts: stamp column -> a SQL type its column is cast to before comparing (``timestamp`` for a
        warehouse's text stamp); compared as stored when absent
    :ptype casts: Mapping[str, str] | None
    :param quote: how the relation and its columns are spelled
    :ptype quote: Callable[[str], str]
    :param by: leading key columns pinned (a state), each a value or an :class:`~threetears.core.keyed.AnyOf`; none
        when None
    :ptype by: Mapping[str, Any] | None
    :param where: filters beside the marks, each a fragment written as the relation's columns from ``$1``
        (``"geo" IS NULL``); none when empty
    :ptype where: Sequence[Fragment]
    :param floor: stamp -> the bound a stamp with no saved mark is read past, bound deliberately by a caller whose
        relation's every row is past it (``0001-01-01``): never a cold start's whole read; none when None
    :ptype floor: Mapping[str, Any] | None
    :return: the rows and the marks to save
    :rtype: MarkedRead
    :raises ValueError: when no mark is named, a mark is None with no floor (a cold start, read on its own), or a
        stamp is not read
    """
    if not marks:
        raise ValueError(f"a read of {relation}'s change record names the marks it reads past")
    floors = dict(floor or {})
    marks = {stamp: floors.get(stamp) if mark is None else mark for stamp, mark in marks.items()}
    unmarked = [stamp for stamp, mark in marks.items() if mark is None]
    if unmarked:
        raise ValueError(
            f"{relation}: no mark for {unmarked}; a load with no mark is a cold start, read on its own (ledgered)"
        )
    missing = [stamp for stamp in marks if stamp not in columns]
    if missing:
        raise ValueError(f"{relation}: the stamps {missing} are not among the columns read")
    cast_of = dict(casts or {})
    pinned, params = by_conditions(by or {}, quote=quote)
    for fragment in where:
        pinned.append(narrowing(fragment, len(params)))
        params += list(fragment.params)
    conditions = []
    for stamp, mark in marks.items():
        params.append(mark)
        column, bound = quote(stamp), f"${len(params)}"
        if stamp in cast_of:
            # the bind cast too: a warehouse reading its parameters as text compares a timestamp to a varchar else
            column, bound = f"CAST({column} AS {cast_of[stamp]})", f"CAST({bound} AS {cast_of[stamp]})"
        conditions.append(f"{column} > {bound}")
    moved = " OR ".join(conditions)
    clause = " AND ".join([*pinned, f"({moved})"]) if pinned else moved
    selected = ", ".join(quote(column) for column in columns)
    sql = f"SELECT {selected} FROM {quote(relation)} WHERE {clause}"
    rows = [dict(row) for row in await conn.fetch(sql, *params)]
    newest = dict(marks)
    for row in rows:
        for stamp in marks:
            value = row[stamp]
            if value is not None and value > newest[stamp]:
                newest[stamp] = value
    return MarkedRead(rows=rows, marks=newest)


@dataclass(frozen=True)
class PageAfter:
    """a page of rows after an id.

    :param rows: the rows, in id order
    :ptype rows: list[dict[str, Any]]
    :param last: the last id read: the next page's ``after``; the one given when no row was read
    :ptype last: Any
    :param more: whether the page was full, so more may follow
    :ptype more: bool
    """

    rows: list[dict[str, Any]]
    last: Any
    more: bool


async def read_after(
    conn: Any,
    table: str | Derived,
    *,
    columns: Sequence[str],
    id_column: str,
    after: Any,
    limit: int,
    by: Mapping[str, Any] | None = None,
) -> PageAfter:
    """the rows of ``table`` (within ``by``) whose ``id_column`` is past ``after``, ``limit`` of them in id order.

    ``by`` and ``id_column`` together are a leading part of the key (or of an index declared for the caller's read),
    the id its next column: each page is one index range.

    :param conn: anything with an asyncpg-shaped ``fetch``
    :ptype conn: Any
    :param table: the table, a TRUSTED identifier
    :ptype table: str
    :param columns: the columns read, ``id_column`` among them
    :ptype columns: Sequence[str]
    :param id_column: the growing id, a TRUSTED identifier
    :ptype id_column: str
    :param after: the last id read; a load with none starts from the type's floor it names (``0``)
    :ptype after: Any
    :param limit: the rows a page reads
    :ptype limit: int
    :param by: the prefix pinned ahead of the id
    :ptype by: Mapping[str, Any] | None
    :return: the page
    :rtype: PageAfter
    :raises ValueError: when ``after`` is None, ``limit`` is under one, or the id is not read
    """
    if after is None:
        raise ValueError(f"a read of {table} after an id names the id (the floor on a first read)")
    if limit < 1:
        raise ValueError(f"a page reads at least one row, got {limit}")
    if id_column not in columns:
        raise ValueError(f"a read of {table} after {id_column} reads {id_column}")
    conditions, params = by_conditions(by or {})
    conditions.append(f"{quote_identifier(id_column)} > ${len(params) + 1}")
    params.append(after)
    selected = ", ".join(quote_identifier(column) for column in columns)
    sql = (
        f"SELECT {selected} FROM {relation(table, 'r')} WHERE {' AND '.join(conditions)} "
        f"ORDER BY {quote_identifier(id_column)} LIMIT {limit}"
    )
    rows = [dict(row) for row in await conn.fetch(sql, *params)]
    return PageAfter(rows=rows, last=rows[-1][id_column] if rows else after, more=len(rows) == limit)
