"""counts without a count of a table: a statement's own rows, the planner's estimate, or a bounded probe.

A ``count(*)`` of a table reads the table. Each question a count answered has a cheaper answer, and this module is the
one place each is spelled:

- **how many rows a write touched**: the statement says so itself (``UPDATE 3``, ``DELETE 0``, ``INSERT 0 5``):
  :func:`rows_affected` reads it off the status;
- **about how big a table is** (a progress bar, a size class, a choice of plan): the planner's estimate,
  ``pg_class.reltuples`` on PostgreSQL and YugabyteDB, ``duckdb_tables().estimated_size`` on DuckDB:
  :func:`estimated_rows`. None for a table never analysed: an estimate that does not exist is not zero;
- **whether any row is held** (an empty copy, a scope still holding rows): :func:`holds_any`, one ``EXISTS`` of one
  row, and :func:`empty_tables` for several tables in one statement;
- **how many rows each key holds** (a run's rows, to tell a complete load from a cut one): :func:`counts_by`, a
  ``GROUP BY`` of the key's own rows, a bounded batch of keys a statement;
- **how many, up to a bound** (a page's "more than 100", a cap): :func:`count_at_most`, a count of at most
  ``at_most + 1`` rows the key names, never more;
- **how many and how much, over a window** (a tenant set's requests and tokens in a period, by hour or by agent):
  :func:`aggregate_keyed`, counts and sums of the rows a pinned prefix, a set of key values and a range of the next
  column name, grouped or not, a bounded batch of values a statement and the batches added.

Each read that takes a ``by`` reads the rows it names (:func:`~threetears.core.keyed.read_keyed`'s ``by``): a leading
part of the key, or an index declared for the caller's read.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final, overload

from threetears.core.keyed import _filters
from threetears.core.keyed._filters import (
    AnyOf,
    Derived,
    Fragment,
    Selected,
    by_conditions,
    narrowing,
    relation,
    selected_column,
    sent_by,
)
from threetears.core.sql_fragments import quote_identifier
from threetears.interfaces.keyed.counts import (
    NO_MARK_FLOOR,
    SQL_DIALECT_ATTRIBUTE,
    BoundedCount,
    Count,
    Dialect,
    MarkedCount,
    Range,
    Sum,
    rows_affected,
)
from threetears.interfaces.keyed.filters import SQL_TYPE

__all__ = [
    "SQL_DIALECT_ATTRIBUTE",
    "BoundedCount",
    "Dialect",
    "MarkedCount",
    "aggregate_keyed",
    "count_at_most",
    "counts_by",
    "counts_by_relations",
    "empty_tables",
    "estimated_rows",
    "holds_any",
    "read_packed",
    "rows_affected",
]

#: the planner's row estimate of a table, by dialect: ``$1`` the table, ``$2`` its schema (the current one when null)
_ESTIMATE_SQL: Final[Mapping[Dialect, str]] = {
    Dialect.POSTGRES: (
        "SELECT c.reltuples::bigint AS estimate FROM pg_catalog.pg_class c "
        "JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace "
        "WHERE c.relname = $1 AND n.nspname = COALESCE($2, current_schema())"
    ),
    Dialect.DUCKDB: (
        "SELECT estimated_size AS estimate FROM duckdb_tables() "
        "WHERE table_name = $1 AND schema_name = COALESCE($2, current_schema())"
    ),
}


async def estimated_rows(
    conn: Any, table: str, *, schema: str | None = None, dialect: Dialect = Dialect.POSTGRES
) -> int | None:
    """the planner's estimate of ``table``'s rows: no row of the table is read.

    :param conn: anything with an asyncpg-shaped ``fetch``
    :ptype conn: Any
    :param table: the table's name, unquoted, as the catalog holds it
    :ptype table: str
    :param schema: its schema; the connection's current one when None
    :ptype schema: str | None
    :param dialect: the database's catalog to ask
    :ptype dialect: Dialect
    :return: the estimate; None when the table is unknown or was never analysed
    :rtype: int | None
    """
    spelled = Dialect.POSTGRES if dialect is Dialect.YUGABYTE else dialect
    if spelled not in _ESTIMATE_SQL:
        raise ValueError(f"no row estimate is read from {dialect}")
    sql = _ESTIMATE_SQL[spelled]
    rows = await sent_by(conn).fetch(sql, table, schema)
    if not rows:
        return None
    estimate = rows[0]["estimate"]
    # -1 is PostgreSQL 14+'s "never analysed"; YugabyteDB says 0 until its first ANALYZE
    if estimate is None or int(estimate) < 0:
        return None
    return int(estimate)


def _where(by: Mapping[str, Any]) -> tuple[str, list[Any]]:
    conditions, values = by_conditions(by)
    return (f" WHERE {' AND '.join(conditions)}" if conditions else ""), values


def _ordered(table: str | Derived, order: Sequence[str]) -> str:
    """the ``ORDER BY`` a probe walks its rows in: the table's key (or a declared index's columns) after ``by``, so the
    bound is a walk of that index, never a scan stopped early. A scan the planner stops at a LIMIT reads every row
    before the ones it keeps: a key whose rows lie at the end of the heap is a whole read.

    :raises ValueError: when ``order`` names no column
    """
    if not order:
        raise ValueError(f"a probe of {table} names the key it walks (order=)")
    return " ORDER BY " + ", ".join(f"r.{quote_identifier(column)}" for column in order)


async def count_at_most(
    conn: Any,
    table: str | Derived,
    *,
    at_most: int,
    order: Sequence[str],
    by: Mapping[str, Any] | None = None,
) -> BoundedCount:
    """how many of the rows ``by`` names ``table`` holds (any of its rows when None), counting at most
    ``at_most + 1`` of them: ``SELECT count(*) FROM (SELECT 1 FROM t WHERE <by> ORDER BY <order> LIMIT $n)``.

    Unkeyed, it is a bounded probe of the table's first rows, never more than the bound: the size class of a table no
    estimate is kept for (an L1 cache, a table never analysed).

    :param conn: anything with an asyncpg-shaped ``fetch``
    :ptype conn: Any
    :param table: the table (``schema.table`` quoted a segment at a time), or a derived relation
    :ptype table: str | Derived
    :param at_most: the bound
    :ptype at_most: int
    :param order: the key the probe walks (the table's key, or a declared index's columns): ``by``'s columns and those
        after them
    :ptype order: Sequence[str]
    :param by: the rows to count: a leading part of the key, each a value or an :class:`~threetears.core.keyed.AnyOf`;
        any rows when None
    :ptype by: Mapping[str, Any] | None
    :return: the count, and whether rows past the bound are held
    :rtype: BoundedCount
    :raises ValueError: when ``at_most`` is under zero
    """
    if at_most < 0:
        raise ValueError(f"a bounded count counts at least none, got {at_most}")
    where, values = _where(by or {})
    ordered = _ordered(table, order)
    sql = (
        f"SELECT count(*) AS held FROM (SELECT 1 FROM {relation(table, 'r')}{where}{ordered} "
        f"LIMIT ${len(values) + 1}) AS held"
    )
    rows = await sent_by(conn).fetch(sql, *values, at_most + 1)
    held = int(rows[0]["held"])
    return BoundedCount(count=min(held, at_most), more=held > at_most)


async def holds_any(
    conn: Any, table: str | Derived, *, order: Sequence[str], by: Mapping[str, Any] | None = None
) -> bool:
    """whether ``table`` holds any row ``by`` names (any row at all when None): one row read, at most.

    :param conn: anything with an asyncpg-shaped ``fetch``
    :ptype conn: Any
    :param table: the table, a TRUSTED identifier
    :ptype table: str
    :param order: the key the probe walks (the table's key, or a declared index's columns)
    :ptype order: Sequence[str]
    :param by: the rows asked about, a leading part of the key; None asks whether the table is empty
    :ptype by: Mapping[str, Any] | None
    :return: whether a row is held
    :rtype: bool
    """
    where, values = _where(by or {})
    rows = await sent_by(conn).fetch(
        f"SELECT 1 AS held FROM {relation(table, 'r')}{where}{_ordered(table, order)} LIMIT 1", *values
    )
    return bool(rows)


async def empty_tables(conn: Any, tables: Sequence[str]) -> list[str]:
    """which of ``tables`` hold no row, in one statement: a probe of one row each.

    :param conn: anything with an asyncpg-shaped ``fetch``
    :ptype conn: Any
    :param tables: the tables, TRUSTED identifiers
    :ptype tables: Sequence[str]
    :return: the empty ones, in the order given
    :rtype: list[str]
    """
    if not tables:
        return []
    probes = ", ".join(
        f"NOT EXISTS (SELECT 1 FROM {quote_identifier(table)} LIMIT 1) AS e{index}"
        for index, table in enumerate(tables)
    )
    rows = await sent_by(conn).fetch(f"SELECT {probes}")
    answer = rows[0]
    return [table for index, table in enumerate(tables) if answer[f"e{index}"]]


async def aggregate_keyed(
    conn: Any,
    table: str | Derived,
    *,
    aggregates: Mapping[str, Count | Sum],
    by: Mapping[str, Any] | None = None,
    ranges: Sequence[Range] = (),
    where: Sequence[Fragment] = (),
    group: Sequence[str | Selected] = (),
    batch: int = 999,
) -> list[dict[str, Any]]:
    """counts and sums of the rows of ``table`` that ``by`` pins and ``ranges`` bound, grouped by ``group`` or not:
    :func:`counts_by`'s keyed ``GROUP BY`` with any counts and sums, a window and any group::

        SELECT <group>, count(*) AS "requests", SUM(<expr>) AS "tokens" FROM <table> AS t
         WHERE <by> AND (t."created" >= $lo AND t."created" < $hi) AND <where> GROUP BY <group>

    ``by`` and the range's column together must be a leading part of the key (or of an index declared for the
    caller's read), the range's column last: each statement reads only the rows they name. An :class:`AnyOf` in
    ``by`` (a tenant set) is named ``batch`` values a statement (the largest when several), and the statements'
    answers merged: a group's counts and sums added. With ``group`` and no aggregate it is the keyed distinct values
    of the group, each once however many batches hold it.

    :param conn: anything with an asyncpg-shaped ``fetch``
    :ptype conn: Any
    :param table: the table (``schema.table``), or a derived relation
    :ptype table: str | Derived
    :param aggregates: each answer's name, a TRUSTED identifier -> its :class:`Count` or :class:`Sum`
    :ptype aggregates: Mapping[str, Count | Sum]
    :param by: the prefix pinned (a value, None, or an :class:`AnyOf` of values)
    :ptype by: Mapping[str, Any] | None
    :param ranges: ranges of ONE column, the column after the prefix, OR-ed
    :ptype ranges: Sequence[Range]
    :param where: filters beside the key, each a fragment written as ``t."col"`` from ``$1``
    :ptype where: Sequence[Fragment]
    :param group: the columns or :class:`Selected` expressions (a time bucket) grouped by, each answered under its
        name; none: one row
    :ptype group: Sequence[str | Selected]
    :param batch: the most values of the batched :class:`AnyOf` a statement names
    :ptype batch: int
    :return: one row a group (its group values and aggregates by name); ungrouped, exactly one row (a count of no
        row is 0, a sum of none None)
    :rtype: list[dict[str, Any]]
    :raises ValueError: when neither ``by`` nor ``ranges`` names the rows (the whole table), the ranges name more
        than one column, nothing is answered (no aggregate and no group), or ``batch`` is under one
    """
    pinned = dict(by or {})
    if not pinned and not ranges:
        raise ValueError(f"a keyed aggregate of {table} names its rows (by=, ranges=): the whole table is not read")
    if len({one.column for one in ranges}) > 1:
        raise ValueError(f"a keyed aggregate's ranges are of one column, the one after its prefix, got {ranges}")
    if not aggregates and not group:
        raise ValueError("a keyed aggregate answers an aggregate or a group")
    if batch < 1:
        raise ValueError(f"a keyed aggregate names at least one value a statement, got batch={batch}")
    names = [part.alias if isinstance(part, Selected) else part for part in group]
    empty = {name: 0 if isinstance(aggregate, Count) else None for name, aggregate in aggregates.items()}
    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    for chunk in _value_chunks(pinned, batch):
        for row in await sent_by(conn).fetch(*_aggregate_statement(table, aggregates, chunk, ranges, where, group)):
            held = merged.setdefault(tuple(row[name] for name in names), {name: row[name] for name in names} | empty)
            for name, aggregate in aggregates.items():
                value = row[name]
                if isinstance(aggregate, Count):
                    held[name] += int(value)
                elif value is not None:
                    held[name] = value if held[name] is None else held[name] + value
    if not group and not merged:
        return [dict(empty)]
    return list(merged.values())


def _value_chunks(pinned: Mapping[str, Any], batch: int) -> list[dict[str, Any]]:
    """``pinned`` a statement at a time: its largest :class:`AnyOf` cut into ``batch`` values each (none for no
    value); unchanged when it holds none."""
    sets = [column for column, value in pinned.items() if isinstance(value, AnyOf)]
    if not sets:
        return [dict(pinned)]
    column = max(sets, key=lambda name: len(pinned[name].values))
    named: AnyOf = pinned[column]
    unique = list(dict.fromkeys(named.values))
    return [
        {**pinned, column: AnyOf(unique[start : start + batch], named.cast, as_list=named.as_list)}
        for start in range(0, len(unique), batch)
    ]


def _aggregate_statement(
    table: str | Derived,
    aggregates: Mapping[str, Count | Sum],
    pinned: Mapping[str, Any],
    ranges: Sequence[Range],
    where: Sequence[Fragment],
    group: Sequence[str | Selected],
) -> tuple[str, ...]:
    """one batch's statement and its parameters."""
    conditions, params = by_conditions(pinned, alias="t")
    edges: list[str] = []
    for one in ranges:
        column = f"t.{quote_identifier(one.column)}"
        bounds: list[str] = []
        for value, operator in ((one.low, ">="), (one.high, "<=" if one.high_inclusive else "<")):
            if value is None:
                continue
            params.append(value)
            bounds.append(
                f"{column} {operator} {f'${len(params)}' if one.cast is None else f'CAST(${len(params)} AS {one.cast})'}"
            )
        edges.append(f"({' AND '.join(bounds)})")
    if edges:
        conditions.append(f"({' OR '.join(edges)})")
    for fragment in where:
        conditions.append(narrowing(fragment, len(params)))
        params += list(fragment.params)
    grouped = [part.expression if isinstance(part, Selected) else f"t.{quote_identifier(part)}" for part in group]
    items = [selected_column(part) for part in group]
    for name, aggregate in aggregates.items():
        if isinstance(aggregate, Count):
            counted = "*" if aggregate.expression is None else aggregate.expression
            items.append(f"count({counted}) AS {quote_identifier(name)}")
        else:
            items.append(f"SUM({aggregate.expression}) AS {quote_identifier(name)}")
    clause = f" GROUP BY {', '.join(grouped)}" if grouped else ""
    sql = f"SELECT {', '.join(items)} FROM {relation(table, 't')} WHERE {' AND '.join(conditions)}{clause}"
    return (sql, *params)


async def counts_by(
    conn: Any,
    table: str | Derived,
    *,
    column: str,
    values: AnyOf | None = None,
    where: Sequence[Fragment] = (),
    by: Mapping[str, Any] | None = None,
    batch: int = 999,
    cast: str | None = "text",
    nullable: bool = True,
) -> dict[Any, int]:
    """how many rows of ``table`` each value of ``column`` holds (within ``by``), ``batch`` values a statement.

    With neither ``values`` nor ``where``, every value of ``column`` within the prefix ``by`` is counted, ``batch``
    values a statement in the column's order (``... AND "column" > $last GROUP BY "column" ORDER BY "column" LIMIT
    batch``): each statement reads the rows of its own groups through the ``(by, column)`` index, never the prefix's
    rows at once. ``cast`` is the column's SQL type the page's last value binds as (None binds it as it is: SQLite).

    ``SELECT "column", count(*) FROM t WHERE <by> AND "column" = ANY($n::<cast>[]) GROUP BY "column"``: each statement
    counts the rows its own keys name, so ``by`` and ``column`` together must be a leading part of the key (or of an
    index declared for the caller's read). The values are named by ``values``, or by ``where`` (a bounded subquery on
    the source's own record, ``"run_id" IN (SELECT run_id FROM runs WHERE status = 'ok' AND ...)``) in one statement.

    :param conn: anything with an asyncpg-shaped ``fetch``
    :ptype conn: Any
    :param table: the table (``schema.table``), or a derived relation
    :ptype table: str | Derived
    :param column: the column counted by, a TRUSTED identifier
    :ptype column: str
    :param values: its values, and how they bind (an array, or a list for a warehouse); None when ``where`` names them
    :ptype values: AnyOf | None
    :param where: filters naming the rows counted beside ``by``, each a fragment from ``$1``
    :ptype where: Sequence[Fragment]
    :param by: the prefix pinned ahead of it (a state, a tenant); ``column`` pinned here to an :class:`AnyOf` (with no
        ``values``) names the values counted, as ``values`` does
    :ptype by: Mapping[str, Any] | None
    :param batch: the most values a statement names (or, paging every value, the most groups a statement answers)
    :ptype batch: int
    :param cast: the column's SQL type, binding the last value of a page when every value is counted; None binds it
        as it is (SQLite)
    :ptype cast: str | None
    :param nullable: False when ``column`` is declared NOT NULL (a key column, a ``Column(nullable=False)``):
        counting every value, no statement counts the NULL group (it holds no row); True counts it apart, one keyed
        statement after the pages
    :ptype nullable: bool
    :return: value -> its rows; a value holding none is absent
    :rtype: dict[Any, int]
    :raises ValueError: when ``batch`` is under one, ``by`` pins ``column`` to other than an :class:`AnyOf` (an
        :class:`AnyOf` there, with no ``values``, names the values counted), or neither ``values``, ``where`` nor a
        prefix ``by`` names the rows
    """
    values, by = _counted_pin(column, values, by)
    if values is None and not where:
        return await _counts_paged(
            conn, table, column=column, by=dict(by or {}), batch=batch, cast=cast, nullable=nullable
        )
    counted = await counts_by_relations(
        conn, {"": table}, column=column, values=values, where=where, by=by, batch=batch
    )
    return counted[""]


def _counted_pin(
    column: str, values: AnyOf | None, by: Mapping[str, Any] | None
) -> tuple[AnyOf | None, Mapping[str, Any] | None]:
    """a count by ``column`` whose ``by=`` pins that column to an :class:`AnyOf` (and names no ``values``): those are
    the values counted, so ``by={"run_id": AnyOf(runs), **others}`` counts each run's rows as
    ``values=AnyOf(runs), by=others`` does; anything else as given.

    :param column: the column counted by
    :ptype column: str
    :param values: the values named apart, or None
    :ptype values: AnyOf | None
    :param by: the pins
    :ptype by: Mapping[str, Any] | None
    :return: the values and the pins left
    :rtype: tuple[AnyOf | None, Mapping[str, Any] | None]
    """
    if values is not None or not by or not isinstance(by.get(column), AnyOf):
        return values, by
    rest = {name: value for name, value in by.items() if name != column}
    return by[column], rest or None


async def _counts_paged(
    conn: Any, table: str | Derived, *, column: str, by: dict[str, Any], batch: int, cast: str | None, nullable: bool
) -> dict[Any, int]:
    """every value of ``column`` within the prefix ``by`` counted, ``batch`` groups a statement in its order.

    :raises ValueError: when ``by`` is empty (that is the whole table), names ``column``, or ``batch`` is under one
    """
    if not by:
        raise ValueError(
            f"a keyed count names its values of {column}, a filter naming them, or the prefix of {table} it counts "
            "every value within (by=)"
        )
    if column in by:
        raise ValueError(f"{column} is counted by, so it is not also pinned")
    if batch < 1:
        raise ValueError(f"a keyed count answers at least one group a statement, got {batch}")
    if cast is not None and not SQL_TYPE.match(cast):
        raise ValueError(f"counts_by: {cast!r} is not a SQL type name")
    quoted = quote_identifier(column)
    counted: dict[Any, int] = {}
    last: Any = None
    while True:
        conditions, params = by_conditions(by)
        conditions.append(f"{quoted} IS NOT NULL")
        if last is not None:
            params.append(last)
            conditions.append(f"{quoted} > ${len(params)}{'' if cast is None else f'::{cast}'}")
        sql = (
            f"SELECT {quoted} AS counted, count(*) AS held FROM {relation(table, 'r')} WHERE {' AND '.join(conditions)}"
            f" GROUP BY {quoted} ORDER BY {quoted} LIMIT {batch}"
        )
        rows = await sent_by(conn).fetch(sql, *params)
        for row in rows:
            counted[row["counted"]] = int(row["held"])
        if len(rows) < batch:
            break
        last = rows[-1]["counted"]
    if not nullable:
        # a column declared NOT NULL has no NULL group to count
        return counted
    # the NULL group, which no "> last" reaches: one keyed count of its own
    conditions, params = by_conditions({**by, column: None})
    rows = await sent_by(conn).fetch(
        f"SELECT count(*) AS held FROM {relation(table, 'r')} WHERE {' AND '.join(conditions)}", *params
    )
    if rows and int(rows[0]["held"]):
        counted[None] = int(rows[0]["held"])
    return counted


@overload
async def counts_by_relations(
    conn: Any,
    tables: Mapping[str, str | Derived],
    *,
    column: str,
    values: AnyOf | None = None,
    where: Sequence[Fragment] = (),
    by: Mapping[str, Any] | None = None,
    batch: int = 999,
    packed_rows: int | None = None,
    groups: int | None = None,
    packed_as: str = "packed",
    since: None = None,
    group: Mapping[str, str] | None = None,
    max_of: Sequence[str] = (),
    fresh: str | None = None,
) -> dict[str, dict[Any, int]]: ...


@overload
async def counts_by_relations(
    conn: Any,
    tables: Mapping[str, str | Derived],
    *,
    column: str,
    values: AnyOf | None = None,
    where: Sequence[Fragment] = (),
    by: Mapping[str, Any] | None = None,
    batch: int = 999,
    packed_rows: int | None = None,
    groups: int | None = None,
    packed_as: str = "packed",
    since: Mapping[str, Mapping[str, str | None]],
    group: Mapping[str, str] | None = None,
    max_of: Sequence[str] = (),
    fresh: str | None = None,
) -> dict[str, list[MarkedCount]]: ...


async def counts_by_relations(
    conn: Any,
    tables: Mapping[str, str | Derived],
    *,
    column: str,
    values: AnyOf | None = None,
    where: Sequence[Fragment] = (),
    by: Mapping[str, Any] | None = None,
    batch: int = 999,
    packed_rows: int | None = None,
    groups: int | None = None,
    packed_as: str = "packed",
    since: Mapping[str, Mapping[str, str | None]] | None = None,
    group: Mapping[str, str] | None = None,
    max_of: Sequence[str] = (),
    fresh: str | None = None,
) -> dict[str, dict[Any, int]] | dict[str, list[MarkedCount]]:
    """:func:`counts_by` over several relations at once: one statement a batch, a ``UNION ALL`` of each relation's
    keyed count, so a warehouse answers every relation in the statements one relation took.

    A statement answers at most ``batch`` groups over every relation together: each names ``batch // len(tables)``
    values (at least one).

    With ``packed_rows`` (a transport that cuts an answer at a row cap: the datasource rail's 1,000), every group of
    every relation is counted in ONE statement, its answer packed: each group a text (``relation``, the value and its
    count, joined by ``CHR(31)``), the texts numbered and set side by side, as many to a row as keep the answer at or
    under ``packed_rows`` rows, by SQL every engine the platform admits spells alike (``ROW_NUMBER``, ``CASE``,
    ``MAX``). A value comes back as the one ``values`` named whose text it is; counted by ``where`` alone, as its
    text. ``packed_as`` names the packed text's column (a caller's own name for it).

    With ``since`` (a source's change record: each relation's rows past its saved marks), each relation's rows past
    any of its marks are grouped by ``column`` and, where ``group`` names one, the relation's own second column, in
    ONE statement, a ``UNION ALL`` tagged by relation (packed with ``packed_rows`` as above)::

        SELECT <relation>, column, CAST(<group> AS VARCHAR), count(*), CAST(max(<max_of>) AS VARCHAR) ...,
               max(CASE WHEN CAST(<fresh> AS TIMESTAMP) > CAST($m AS TIMESTAMP) THEN 1 ELSE 0 END)
          FROM <relation> WHERE <by> AND <where>
           AND (CAST(<stamp> AS TIMESTAMP) > CAST($n AS TIMESTAMP) OR ...)
         GROUP BY column, <group>

    Each stamp is compared as a timestamp whatever its column's type (a warehouse's text stamp too); a stamp with no
    mark yet is past :data:`NO_MARK_FLOOR`, which every stamp is, and a NULL stamp is past nothing. Each relation
    reads only its rows past the marks through its stamp's index: a relation with nothing past them answers none.
    The answer is :class:`MarkedCount` groups, per label.

    :param conn: anything with an asyncpg-shaped ``fetch``
    :ptype conn: Any
    :param tables: a label -> each relation (``schema.table``, or a derived relation)
    :ptype tables: Mapping[str, str | Derived]
    :param column: the column counted by, in every relation
    :ptype column: str
    :param values: its values; None when ``where`` names them
    :ptype values: AnyOf | None
    :param where: filters naming the rows counted beside ``by``, each a fragment from ``$1``, the same in every
        relation
    :ptype where: Sequence[Fragment]
    :param by: the prefix pinned in every relation
    :ptype by: Mapping[str, Any] | None
    :param batch: the most groups a statement answers (unpacked)
    :ptype batch: int
    :param packed_rows: the most rows the one packed statement may answer; unpacked when None
    :ptype packed_rows: int | None
    :param groups: packed with no ``values``, the most groups ``where`` (or ``since``) can name over every relation
        together (its packing is taken from it)
    :ptype groups: int | None
    :param packed_as: the packed text's column name, a TRUSTED identifier
    :ptype packed_as: str
    :param since: label -> its marks: each stamp column -> the mark its rows are past (None: no mark yet, the floor);
        every label named, each with at least one stamp
    :ptype since: Mapping[str, Mapping[str, str | None]] | None
    :param group: with ``since``, label -> the relation's second group column, TRUSTED; a label absent groups by
        ``column`` alone
    :ptype group: Mapping[str, str] | None
    :param max_of: with ``since``, the columns (in every relation) whose newest value each group answers, as text
    :ptype max_of: Sequence[str]
    :param fresh: with ``since``, the stamp (among each relation's marks) past whose mark a group is :attr:`fresh
        <MarkedCount.fresh>`
    :ptype fresh: str | None
    :return: label -> value -> its rows; every label present. With ``since``: label -> its groups past the marks
    :rtype: dict[str, dict[Any, int]] | dict[str, list[MarkedCount]]
    :raises ValueError: as :func:`counts_by`, when no relation is named, or packed with neither ``values`` nor
        ``groups``, or ``packed_rows`` under one; with ``since``, when a label has no marks, ``values`` is given, or
        ``fresh`` is not a stamp of every relation's marks; ``group``, ``max_of`` or ``fresh`` without ``since``
    """
    quote_identifier(packed_as)
    if since is None:
        values, by = _counted_pin(column, values, by)
    if since is not None:
        return await _counts_since(
            conn,
            tables,
            column,
            since,
            group or {},
            max_of,
            fresh,
            where,
            dict(by or {}),
            values,
            packed_rows,
            groups,
            packed_as,
        )
    if group or max_of or fresh is not None:
        raise ValueError("group=, max_of= and fresh= group the rows past a relation's marks: they need since=")
    if batch < 1:
        raise ValueError(f"a keyed count answers at least one group a statement, got {batch}")
    if not tables:
        raise ValueError("a keyed count names at least one relation")
    pinned = dict(by or {})
    if column in pinned:
        raise ValueError(f"{column} is counted by, so it is not also pinned")
    if values is None and not where:
        raise ValueError(f"a keyed count names its values of {column}, or a filter naming them")
    labels = list(tables)
    counted: dict[str, dict[Any, int]] = {label: {} for label in labels}
    quoted = quote_identifier(column)
    if packed_rows is not None:
        await _counts_packed(conn, tables, counted, column, values, where, pinned, packed_rows, groups, packed_as)
        return counted
    if values is None:
        chunks: list[dict[str, Any]] = [{}]
    else:
        unique = list(dict.fromkeys(values.values))
        size = max(1, batch // len(labels))
        chunks = [
            {column: AnyOf(unique[start : start + size], values.cast, as_list=values.as_list)}
            for start in range(0, len(unique), size)
        ]
    for chunk in chunks:
        conditions, params = by_conditions({**pinned, **chunk})
        for fragment in where:
            conditions.append(narrowing(fragment, len(params)))
            params += list(fragment.params)
        clause = " AND ".join(conditions)
        parts = [
            f"SELECT {index} AS relation, {quoted} AS counted, count(*) AS held FROM {relation(tables[label], 'r')} "
            f"WHERE {clause} GROUP BY {quoted}"
            for index, label in enumerate(labels)
        ]
        for row in await sent_by(conn).fetch(" UNION ALL ".join(parts), *params):
            counted[labels[int(row["relation"])]][row["counted"]] = int(row["held"])
    return counted


#: what joins a packed count's fields: the relation's place, the value's text, its count
_PACKED_SEPARATOR: Final = chr(31)

#: the type a packed text is cast to: sized, since Redshift reads a bare ``VARCHAR`` as ``VARCHAR(256)`` and cuts a
#: longer text to it (a long ``model_version``); 65535 is Redshift's widest (``VARCHAR(MAX)``), and PostgreSQL,
#: YugabyteDB and DuckDB take the same spelling
_PACKED_TEXT: Final = "VARCHAR(65535)"


async def _counts_packed(
    conn: Any,
    tables: Mapping[str, str | Derived],
    counted: dict[str, dict[Any, int]],
    column: str,
    values: AnyOf | None,
    where: Sequence[Fragment],
    pinned: Mapping[str, Any],
    packed_rows: int,
    groups: int | None,
    packed_as: str = "packed",
) -> None:
    """every group of every relation counted in one statement, its texts packed several to a row (into ``counted``).

    :raises ValueError: when ``packed_rows`` is under one, or neither ``values`` nor ``groups`` bounds the groups
    """
    if packed_rows < 1:
        raise ValueError(f"a packed count answers at least one row, got {packed_rows}")
    labels = list(tables)
    named: dict[str, Any] = {}
    by = dict(pinned)
    if values is not None:
        unique = list(dict.fromkeys(values.values))
        named = {str(value): value for value in unique}
        by[column] = AnyOf(unique, values.cast, as_list=values.as_list)
        items = len(unique) * len(labels)
    elif groups is not None:
        items = groups
    else:
        raise ValueError(f"a packed count of {column} names its values, or the most groups its filter names (groups=)")
    if items == 0:
        return
    conditions, params = by_conditions(by)
    for fragment in where:
        conditions.append(narrowing(fragment, len(params)))
        params += list(fragment.params)
    clause = " AND ".join(conditions)
    quoted = quote_identifier(column)
    alias = quote_identifier(packed_as)
    texts = " UNION ALL ".join(
        f"SELECT '{index}' || CHR(31) || {_nullable_text(quoted)} || CHR(31) || CAST(count(*) AS {_PACKED_TEXT}) "
        f"AS {alias} "
        f"FROM {relation(tables[label], 'r')} WHERE {clause} GROUP BY {quoted}"
        for index, label in enumerate(labels)
    )
    for fields in await _packed_rows(conn, texts, params, items, packed_rows, alias):
        place, value, held = fields
        text = _unpacked_text(value)
        counted[labels[int(place)]][named.get(text, text) if named and text is not None else text] = int(held)


def _nullable_text(expression: str) -> str:
    """a packed field's text: ``v`` and the value's text, or empty for NULL, so a NULL never nulls the whole text."""
    return f"COALESCE('v' || CAST({expression} AS {_PACKED_TEXT}), '')"


def _unpacked_text(field: str) -> str | None:
    return field[1:] if field else None


async def _packed_rows(
    conn: Any, texts: str, params: Sequence[Any], items: int, packed_rows: int, alias: str
) -> list[list[str]]:
    """the fields of each packed text ``texts`` answers, packed several to a row so at most ``packed_rows`` come."""
    return [
        text.split(_PACKED_SEPARATOR) for text in await _packed_cells(conn, texts, params, items, packed_rows, alias)
    ]


async def _packed_cells(
    conn: Any, texts: str, params: Sequence[Any], items: int, packed_rows: int, alias: str
) -> list[str]:
    """each text ``texts`` answers (its column ``alias``), packed several to a row so at most ``packed_rows`` come:
    the texts numbered from 0, ``width`` a row (a 1-based number would start a row more at the last)."""
    width = max(1, -(-items // packed_rows))
    cells = ", ".join(f"MAX(CASE WHEN n % {width} = {cell} THEN {alias} END) AS c{cell}" for cell in range(width))
    sql = (
        f"SELECT {cells} FROM (SELECT ROW_NUMBER() OVER (ORDER BY {alias}) - 1 AS n, {alias} FROM ({texts}) AS texts) "
        f"AS numbered GROUP BY n - n % {width}"
    )
    return [
        str(text) for row in await sent_by(conn).fetch(sql, *params) for text in dict(row).values() if text is not None
    ]


async def read_packed(
    conn: Any,
    relation: str | Derived,
    column: str | Selected,
    *,
    by: Mapping[str, Any] | None = None,
    where: Sequence[Fragment] = (),
    packed_rows: int,
    items: int,
    packed_as: str = "packed",
) -> list[str]:
    """each text of ``column`` the rows ``by`` and ``where`` name hold, read packed several to a row so at most
    ``packed_rows`` rows come back: for a reader answering at most so many rows a statement (the hub's rail, its row
    cap), a read of up to ``items`` short texts in one statement. In the texts' order; a NULL is no text (left out).

    :param conn: anything with an asyncpg-shaped ``fetch``
    :ptype conn: Any
    :param relation: the relation read, a TRUSTED name or a :class:`Derived` select
    :ptype relation: str | Derived
    :param column: the column read (TRUSTED), or a :class:`Selected` text expression of the row's columns (``r.``)
    :ptype column: str | Selected
    :param by: columns pinned, each a value, ``None`` (``IS NULL``) or an :class:`AnyOf`; none when None
    :ptype by: Mapping[str, Any] | None
    :param where: conditions beside the pins, each a fragment written as ``r."col"`` from ``$1``
    :ptype where: Sequence[Fragment]
    :param packed_rows: the most rows the statement answers, at least one
    :ptype packed_rows: int
    :param items: the most texts the rows hold (what sizes each row's width)
    :ptype items: int
    :param packed_as: the packed column's alias
    :ptype packed_as: str
    :return: the texts
    :rtype: list[str]
    :raises ValueError: when ``packed_rows`` is under one or ``items`` is negative
    """
    if packed_rows < 1:
        raise ValueError(f"a packed read answers at least one row, got {packed_rows}")
    if items < 0:
        raise ValueError(f"a packed read holds no fewer than no texts, got {items}")
    if items == 0:
        return []
    conditions, params = by_conditions(dict(by or {}))
    for fragment in where:
        conditions.append(narrowing(fragment, len(params)))
        params += list(fragment.params)
    alias = quote_identifier(packed_as)
    text = column.expression if isinstance(column, Selected) else f"r.{quote_identifier(column)}"
    clause = f" WHERE {' AND '.join(conditions)}" if conditions else ""
    texts = f"SELECT CAST({text} AS {_PACKED_TEXT}) AS {alias} FROM {_filters.relation(relation, 'r')}{clause}"
    return await _packed_cells(conn, texts, params, items, packed_rows, alias)


def _past_marks(marks: Mapping[str, str | None], first: int) -> tuple[str, list[Any], dict[str, str]]:
    """rows past any of ``marks``, each stamp a timestamp; the condition, its parameters from ``first``, and each
    stamp's comparison (for the fresh flag)."""
    compared: dict[str, str] = {}
    params: list[Any] = []
    for stamp, mark in marks.items():
        params.append(NO_MARK_FLOOR if mark is None else mark)
        compared[stamp] = (
            f"CAST({quote_identifier(stamp)} AS TIMESTAMP) > CAST(${first + len(params) - 1} AS TIMESTAMP)"
        )
    return f"({' OR '.join(compared.values())})", params, compared


async def _counts_since(
    conn: Any,
    tables: Mapping[str, str | Derived],
    column: str,
    since: Mapping[str, Mapping[str, str | None]],
    group: Mapping[str, str],
    max_of: Sequence[str],
    fresh: str | None,
    where: Sequence[Fragment],
    pinned: dict[str, Any],
    values: AnyOf | None,
    packed_rows: int | None,
    groups: int | None,
    packed_as: str,
) -> dict[str, list[MarkedCount]]:
    """every relation's groups past its marks, one statement (:func:`counts_by_relations` with ``since``).

    :raises ValueError: as :func:`counts_by_relations` says for ``since``
    """
    if not tables:
        raise ValueError("a keyed count names at least one relation")
    if values is not None:
        raise ValueError("a count past marks counts the groups its marks name: values= is not given with since=")
    if column in pinned:
        raise ValueError(f"{column} is counted by, so it is not also pinned")
    labels = list(tables)
    unmarked = [label for label in labels if not since.get(label)]
    if unmarked:
        raise ValueError(f"a count past marks names each relation's marks; none for {unmarked}")
    if fresh is not None and any(fresh not in since[label] for label in labels):
        raise ValueError(f"fresh={fresh!r} is a stamp each relation's marks name")
    unknown = set(group) - set(labels)
    if unknown:
        raise ValueError(f"group= names relations not counted: {sorted(unknown)}")
    answer: dict[str, list[MarkedCount]] = {label: [] for label in labels}
    conditions, params = by_conditions(pinned)
    for fragment in where:
        conditions.append(narrowing(fragment, len(params)))
        params += list(fragment.params)
    quoted = quote_identifier(column)
    newest = [f"CAST(max({quote_identifier(name)}) AS VARCHAR)" for name in max_of]
    parts: list[tuple[int, str, str, str, str]] = []
    for index, label in enumerate(labels):
        past, marks, compared = _past_marks(since[label], len(params) + 1)
        params += marks
        grouped = group.get(label)
        group_text = "CAST(NULL AS VARCHAR)" if grouped is None else f"CAST({quote_identifier(grouped)} AS VARCHAR)"
        flag = "0" if fresh is None else f"max(CASE WHEN {compared[fresh]} THEN 1 ELSE 0 END)"
        clause = " AND ".join([*conditions, past])
        tail = f"FROM {relation(tables[label], 'r')} WHERE {clause} GROUP BY {quoted}"
        parts.append((index, group_text, flag, tail, "" if grouped is None else f", {quote_identifier(grouped)}"))
    if packed_rows is not None:
        if packed_rows < 1:
            raise ValueError(f"a packed count answers at least one row, got {packed_rows}")
        if groups is None:
            raise ValueError("a packed count past marks names the most groups its marks can name (groups=)")
        if groups == 0:
            return answer
        alias = quote_identifier(packed_as)
        texts = " UNION ALL ".join(
            f"SELECT '{index}' || CHR(31) || {_nullable_text(quoted)} || CHR(31) || "
            f"COALESCE('v' || {group_text}, '') || CHR(31) || CAST(count(*) AS VARCHAR) || CHR(31) || "
            + "".join(f"COALESCE('v' || {stamp}, '') || CHR(31) || " for stamp in newest)
            + f"CAST({flag} AS VARCHAR) AS {alias} {tail}{extra}"
            for index, group_text, flag, tail, extra in parts
        )
        for fields in await _packed_rows(conn, texts, params, groups, packed_rows, alias):
            place, value, grouped_text, held, *stamps, flagged = fields
            answer[labels[int(place)]].append(
                MarkedCount(
                    value=_unpacked_text(value),
                    group=_unpacked_text(grouped_text),
                    rows=int(held),
                    newest={name: _unpacked_text(stamp) for name, stamp in zip(max_of, stamps, strict=True)},
                    fresh=bool(int(flagged)),
                )
            )
        return answer
    newest_columns = "".join(f", {stamp} AS m{at}" for at, stamp in enumerate(newest))
    sql = " UNION ALL ".join(
        f"SELECT {index} AS relation, {quoted} AS counted, {group_text} AS grp, count(*) AS held{newest_columns}, "
        f"{flag} AS fresh {tail}{extra}"
        for index, group_text, flag, tail, extra in parts
    )
    for row in await sent_by(conn).fetch(sql, *params):
        answer[labels[int(row["relation"])]].append(
            MarkedCount(
                value=row["counted"],
                group=None if row["grp"] is None else str(row["grp"]),
                rows=int(row["held"]),
                newest={name: None if row[f"m{at}"] is None else str(row[f"m{at}"]) for at, name in enumerate(max_of)},
                fresh=bool(int(row["fresh"])),
            )
        )
    return answer
