"""point reads: the rows one key names, bound as a hand-written ``WHERE a = $1 AND b = $2`` binds it.

The one spelling of the read a module used to write as text and send through ``fetch``, ``fetchrow`` or ``fetchval``::

    SELECT t."id", t."name" FROM "agents" AS t WHERE t."id" = $1 AND t."is_deleted" = $2

Each pinned column is compared with one parameter, ``= $n``, whose type the database infers from the column, exactly as
it inferred it for the statement written by hand, so a ``uuid`` key takes a ``UUID`` (or its text) and its index is read
by it; a ``None`` pins ``IS NULL``; an :class:`~threetears.core.keyed.AnyOf` pins a set. Where :func:`read_keyed` reads a
batch of keys, this reads one: the rows of one key, the first of them, or the first column of the first.

The read is keyed when its pinned columns cover a leading part of the table's key, or the leading column of an index
declared for the caller's read; the no-whole-reads rule judges each call by the statement it sends
(:mod:`threetears.enforcement.no_whole_reads.helpers`), under the caller's name. A read pinning nothing reads the table
whole, and is refused.

IO only: one statement, awaited on the caller's connection (anything with an asyncpg-shaped ``fetch``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal, overload

from threetears.core.keyed._filters import Fragment, Lock, Selected, newest_read, reads_l3, refuse_versioned, sent_by
from threetears.core.keyed._joining import Joined, check_joins, join_rows, joined_columns, lateral_joins
from threetears.core.keyed.inline import InlineDuckDB
from threetears.core.keyed.page import yugabyte_index_hint
from threetears.core.keyed.threaded import ThreadedDuckDB
from threetears.interfaces.keyed.filters import (
    Derived,
    by_conditions,
    narrowing,
    quote_identifier,
    quoted_table,
    relation,
    selected_column,
)
from threetears.interfaces.keyed.keys import stated_literals
from threetears.interfaces.keyed.page import Sort
from threetears.interfaces.keyed.point import (
    Nearest,
    PatternCount,
    TextRank,
    read_by_statement,
    read_nearest_statement,
    read_row_beside_statement,
    read_row_statement,
    then_terms,
)
from threetears.interfaces.keyed.rows import DeclaredScope
from threetears.interfaces.keyed.steps import Step

__all__ = [
    "read_by",
    "read_nearest",
    "read_row",
    "read_row_beside",
    "read_state_arrow",
    "read_value",
]


@overload
async def read_by(
    conn: Any,
    table: str | Derived,
    *,
    by: Mapping[str, Any],
    columns: Sequence[str | Selected] | None = ...,
    records: Literal[False] = ...,
    lock: Lock | None = ...,
    where: Sequence[Fragment] = ...,
    order: Sequence[Sort] = ...,
) -> list[dict[str, Any]]: ...


@overload
async def read_by(
    conn: Any,
    table: str | Derived,
    *,
    by: Mapping[str, Any],
    columns: Sequence[str | Selected] | None = ...,
    records: Literal[True],
    lock: Lock | None = ...,
    where: Sequence[Fragment] = ...,
    order: Sequence[Sort] = ...,
) -> list[Any]: ...


async def read_by(
    conn: Any,
    table: str | Derived,
    *,
    by: Mapping[str, Any],
    columns: Sequence[str | Selected] | None = None,
    records: bool = False,
    lock: Lock | None = None,
    where: Sequence[Fragment] = (),
    order: Sequence[Sort] = (),
) -> list[dict[str, Any]] | list[Any]:
    """every row of ``table`` the pinned columns name: what ``fetch`` of the hand-written statement answered.

    :param conn: anything with an asyncpg-shaped ``fetch(query, *params)``
    :ptype conn: Any
    :param table: the table, a TRUSTED identifier, or a :class:`~threetears.core.keyed.Derived` relation read as one
    :ptype table: str | Derived
    :param by: column -> its value (``= $n``, typed by the column), ``None`` (``IS NULL``) or an
        :class:`~threetears.core.keyed.AnyOf`; columns are TRUSTED identifiers; at least one
    :ptype by: Mapping[str, Any]
    :param columns: the columns read, TRUSTED identifiers or :class:`~threetears.core.keyed.Selected` expressions;
        every column of the table, in its order, when None
    :ptype columns: Sequence[str | Selected] | None
    :param records: answer the rows as the connection answers them (asyncpg ``Record``s on a direct connection), with
        no copy into a dict; dicts when False
    :ptype records: bool
    :param lock: the row lock taken on the rows read (``FOR UPDATE OF t NOWAIT``, say), in the caller's transaction;
        none when None
    :ptype lock: Lock | None
    :param where: guards beside the pins, each a fragment written as ``t."col"`` from ``$1``; none when empty
    :ptype where: Sequence[Fragment]
    :param order: the sort the rows come in, the columns after the pins in the key or index serving the read; none
        when empty
    :ptype order: Sequence[Sort]
    :return: the rows, in ``order``, else in no promised order
    :rtype: list[dict[str, Any]] | list[Any]
    :raises ValueError: when nothing is pinned, or ``columns`` is empty
    """
    sql, params = read_by_statement(
        newest_read(conn, table, lock=lock), by=by, columns=columns, lock=lock, where=where, order=order
    )
    # on YugabyteDB, a read names the declared index that serves its pins and order (unordered: binds the most pins)
    sql = (await yugabyte_index_hint(conn, table, by, order, where=where)) + sql
    rows = list(await sent_by(conn).fetch(sql, *params))
    return rows if records else [dict(row) for row in rows]


@overload
async def read_row(
    conn: Any,
    table: str,
    *,
    by: Mapping[str, Any],
    columns: Sequence[str | Selected] | None = ...,
    records: Literal[False] = ...,
    lock: Lock | None = ...,
    where: Sequence[Fragment] = ...,
    beside: Derived | None = ...,
) -> dict[str, Any] | None: ...


@overload
async def read_row(
    conn: Any,
    table: str,
    *,
    by: Mapping[str, Any],
    columns: Sequence[str | Selected] | None = ...,
    records: Literal[True],
    lock: Lock | None = ...,
    where: Sequence[Fragment] = ...,
    beside: Derived | None = ...,
) -> Any: ...


async def read_row(
    conn: Any,
    table: str,
    *,
    by: Mapping[str, Any],
    columns: Sequence[str | Selected] | None = None,
    records: bool = False,
    lock: Lock | None = None,
    where: Sequence[Fragment] = (),
    beside: Derived | None = None,
) -> dict[str, Any] | Any | None:
    """the first row of ``table`` the pinned columns name, or None: what ``fetchrow`` of the hand-written statement
    answered (one row asked for, ``LIMIT 1``; which one, where the key names several, is the database's, as it was).

    :param conn: anything with an asyncpg-shaped ``fetch(query, *params)``
    :ptype conn: Any
    :param table: the table, a TRUSTED identifier
    :ptype table: str
    :param by: as :func:`read_by`'s
    :ptype by: Mapping[str, Any]
    :param columns: as :func:`read_by`'s
    :ptype columns: Sequence[str | Selected] | None
    :param records: as :func:`read_by`'s; the row is then typed as the connection's (``Any``), as ``fetchrow``'s was
    :ptype records: bool
    :param lock: the row lock taken on the rows read (``FOR UPDATE OF t NOWAIT``, say), in the caller's transaction;
        none when None
    :ptype lock: Lock | None
    :param where: guards beside the pins, each a fragment written as ``t."col"`` from ``$1``; none when empty
    :ptype where: Sequence[Fragment]
    :param beside: a TRUSTED table-less derived row read beside the table's (``b``), its values taken before the
        select list runs (:func:`~threetears.interfaces.keyed.read_row_statement`); none when None
    :ptype beside: Derived | None
    :return: the row, or None when the key names none
    :rtype: dict[str, Any] | Any | None
    :raises ValueError: when nothing is pinned, or ``columns`` is empty
    """
    sql, params = read_row_statement(
        newest_read(conn, table, lock=lock), by=by, columns=columns, lock=lock, where=where, beside=beside
    )
    rows = list(await sent_by(conn).fetch(sql, *params))
    if not rows:
        return None
    return rows[0] if records else dict(rows[0])


async def read_row_beside(
    conn: Any,
    table: str,
    *,
    by: Mapping[str, Any],
    columns: Sequence[str | Selected],
    beside: Step,
    limit: int,
    present: str,
    first: str | None = None,
) -> list[Any]:
    """the first row of ``table`` the pins name and, beside it, up to ``limit`` rows of a step reading that row's
    columns as ``r."col"``, in one statement: one round trip, and one snapshot for the row and the rows it decides
    (:func:`~threetears.interfaces.keyed.read_row_beside_statement`). Each answered row is the table's row's columns,
    then ``present`` (TRUE; NULL on the row alone when the step holds none), then the step's columns, as records.

    :param conn: anything with an asyncpg-shaped ``fetch(query, *params)``
    :ptype conn: Any
    :param table: the table, a TRUSTED identifier
    :ptype table: str
    :param by: as :func:`read_by`'s; at least one
    :ptype by: Mapping[str, Any]
    :param columns: the row's columns, as :func:`read_by`'s; each one the step reads among them
    :ptype columns: Sequence[str | Selected]
    :param beside: the step (:mod:`threetears.core.keyed.steps`), its conditions on the row written ``r."col"``
    :ptype beside: Step
    :param limit: the most step rows read
    :ptype limit: int
    :param present: the name of the column TRUE on a step row, a TRUSTED identifier
    :ptype present: str
    :param first: given, the name of a column TRUE on one answered row alone: the table's row's columns come back on
        that row (or on the row alone) and NULL on the rest, rather than repeated on every step row
        (:func:`~threetears.interfaces.keyed.read_row_beside_statement`); the rows come in no particular order
    :ptype first: str | None
    :return: the rows, empty when the pins name no row
    :rtype: list[Any]
    :raises ValueError: when nothing is pinned, or for a limit under one row
    """
    sql, params = read_row_beside_statement(
        newest_read(conn, table),
        by=by,
        columns=columns,
        beside=beside,
        limit=limit,
        present=present,
        first=first,
    )
    return list(await sent_by(conn).fetch(sql, *params))


async def read_value(
    conn: Any, table: str, *, by: Mapping[str, Any], column: str, where: Sequence[Fragment] = ()
) -> Any:
    """one column of the first row of ``table`` the pinned columns name, or None: what ``fetchval`` of the
    hand-written statement answered (None both when the key names no row and when the column holds none).

    :param conn: anything with an asyncpg-shaped ``fetch(query, *params)``
    :ptype conn: Any
    :param table: the table, a TRUSTED identifier
    :ptype table: str
    :param by: as :func:`read_by`'s
    :ptype by: Mapping[str, Any]
    :param column: the column, a TRUSTED identifier
    :ptype column: str
    :param where: guards beside the pins, each a fragment written as ``t."col"`` from ``$1``; none when empty
    :ptype where: Sequence[Fragment]
    :return: its value, or None
    :rtype: Any
    :raises ValueError: when nothing is pinned
    """
    sql, params = read_row_statement(newest_read(conn, table), by=by, columns=(column,), where=where)
    rows = list(await sent_by(conn).fetch(sql, *params))
    return rows[0][column] if rows else None


async def read_nearest(
    conn: Any,
    table: str,
    *,
    by: Mapping[str, Any],
    nearest: Nearest | TextRank | PatternCount,
    limit: int,
    columns: Sequence[str | Selected] | None = None,
    records: bool = False,
    where: Sequence[Fragment] = (),
    joins: Sequence[Joined] = (),
    batch: int | None = None,
) -> list[Any]:
    """the ``limit`` rows of ``table`` the pinned columns name nearest to a vector, nearest first: what ``fetch`` of
    the hand-written ``... ORDER BY embedding <=> $n LIMIT $k`` answered
    (:func:`~threetears.interfaces.keyed.read_nearest_statement`); or, given a :class:`TextRank`, best matched by a
    text query, best first (``... ORDER BY ts_rank_cd(search_vector, websearch_to_tsquery(...)) DESC LIMIT $k``); or,
    given a :class:`PatternCount`, by how many of its ``ILIKE`` patterns each row matches, best first, then its
    ``then`` columns (:func:`_pattern_count_statement`: one statement, its ``LEFT`` joins inside it).

    The order is the distance (or rank) alone, so an approximate-nearest-neighbour index on the vector column (its
    purpose naming the caller) answers it; the no-whole-reads rule judges the call by the statement it sends.

    With ``joins``, each row is then joined to its parent rows (``LEFT JOIN media med ON ...``), each joined table read
    by the values the ranked rows hold (one keyed batch, :func:`~threetears.core.keyed.read_joined`), never a join the
    planner may drive from the joined side. A row matching several is answered once for each, in rank order, and the
    first ``limit`` joined rows are kept: what ``ORDER BY ... LIMIT`` over the hand-written ``LEFT JOIN`` kept, since a
    row's joined copies share its rank. Each join is a ``LEFT`` (or counted) one: an inner join would drop rows after
    the limit that the statement dropped before it, so a parent every row has (a foreign key's) is joined ``LEFT``.

    :param conn: anything with an asyncpg-shaped ``fetch(query, *params)``
    :ptype conn: Any
    :param table: the table, a TRUSTED identifier
    :ptype table: str
    :param by: as :func:`read_by`'s; at least one column
    :ptype by: Mapping[str, Any]
    :param nearest: the vector column, the vector (its text), the distance, and the similarity's name if selected;
        or the ``tsvector`` column, the query's text and the rank's name; or the patterns counted over fields
    :ptype nearest: Nearest | TextRank | PatternCount
    :param limit: how many rows, at most
    :ptype limit: int
    :param columns: as :func:`read_by`'s; the similarity (or rank), when named, after them
    :ptype columns: Sequence[str | Selected] | None
    :param records: as :func:`read_by`'s; with ``joins``, the rows are dicts
    :ptype records: bool
    :param where: as :func:`read_by`'s
    :ptype where: Sequence[Fragment]
    :param joins: the tables joined after the ranked read, in order, each ``LEFT`` or counted
        (:class:`~threetears.core.keyed.Joined`); none when empty
    :ptype joins: Sequence[Joined]
    :param batch: the most values a joined table's statement names; None: the connection's
    :ptype batch: int | None
    :return: the rows, nearest (or best) first
    :rtype: list[Any]
    :raises ValueError: when nothing is pinned, ``columns`` is empty, ``limit`` is negative, or a join is an inner
        one or names nothing to join on
    """
    # a versioned table's newest live versions, read in its place; a joined table is joined inside the statement,
    # where no newest relation reads it, so a versioned one is refused
    read_from = newest_read(conn, table)
    refuse_versioned(conn, [join.table for join in joins], f"a ranked read of {table} joining")
    if isinstance(nearest, PatternCount):
        sql, params = _pattern_count_statement(read_from, by, nearest, limit, columns, where, joins)
        rows = list(await sent_by(conn).fetch(sql, *params))
        return rows if records and not joins else [dict(row) for row in rows]
    if not joins:
        sql, params = read_nearest_statement(
            read_from, by=by, nearest=nearest, limit=limit, columns=columns, where=where
        )
        rows = list(await sent_by(conn).fetch(sql, *params))
        return rows if records else [dict(row) for row in rows]
    check_joins(table, joins)
    if any(not (join.left or join.counts) for join in joins):
        raise ValueError(f"a ranked read of {table} joins LEFT only (an inner join drops rows past the limit)")
    read, hidden = joined_columns(columns, joins)
    sql, params = read_nearest_statement(read_from, by=by, nearest=nearest, limit=limit, columns=read, where=where)
    if all(not (join.counts or join.matched) for join in joins) and _ranked_by(nearest) is not None:
        # One statement: the ranked rows, each joined to its parents by key (a LATERAL lookup per ranked row). The
        # ranked read then a keyed batch of the parents (join_rows) cost a second round trip: at real size (20k
        # media_content rows of one user, PostgreSQL) the FTS search took 5.96 ms against the hand-written join's
        # 4.92 ms (A/A gap 5%), over the bar of "not slower than the old statement" (S/c2/lat, coordinator
        # 2026-11-14). LATERAL keeps the lookup driven from the ranked rows: as a plain join YugabyteDB hashed every
        # file of the agent. The joined rows are ordered by the rank the ranked read selects (the similarity, or the
        # text rank): ranked_sql is itself wrapped to re-sort by distance (read_nearest_statement), so the distance
        # is not in its outer select list. Counted and matched joins, and a vector read naming no similarity, keep
        # the two-step read. Measured on this one statement (2 users x 10k rows, 1024-dim, quiet, S/c2/lat/run2.log):
        # PostgreSQL content FTS 2.44 ms against the bare ranked read's 2.33 and the old join's 2.60, chunk
        # within_memory 5.81 / 5.73 / 5.69; semantic 39.9 ms against the old 1567; YugabyteDB at most 1.14x the bare
        # read. Guarded by packages/core/tests/integration/test_nearest_join_leg_speed_live.py (2x the bare read, one
        # statement a read).
        joined_sql, joined_params = _joined_nearest_statement(sql, list(params), nearest, joins)
        rows = [dict(row) for row in await sent_by(conn).fetch(joined_sql, *joined_params)]
        for row in rows:
            for column in hidden:
                del row[column]
        return rows
    ranked = [dict(row) for row in await sent_by(conn).fetch(sql, *params)]
    return (await join_rows(conn, ranked, joins, batch=batch, hidden=hidden))[:limit]


def _pattern_count_statement(
    table: str | Derived,
    by: Mapping[str, Any],
    rank: PatternCount,
    limit: int,
    columns: Sequence[str | Selected] | None,
    where: Sequence[Fragment],
    joins: Sequence[Joined],
) -> tuple[str, list[Any]]:
    """the rows ``by`` (and ``where``) name of ``table`` that match any of ``rank``'s patterns, each ``LEFT`` joined to
    the rows of each joined table its columns name, best first by how many patterns a row matches, then by ``then``,
    up to ``limit``::

        SELECT t.*, j2."description" AS "template_description",
               (CASE WHEN t."column_name" ILIKE ($2::text[])[1] OR j2."description" ILIKE ($2::text[])[1]
                     THEN 1 ELSE 0 END + ...) AS "match_score"
          FROM "datasource_columns" AS t
          LEFT JOIN "datasource_tables" AS j1 ON j1."datasource_id" = t."datasource_id" AND ...
          LEFT JOIN "table_template_columns" AS j2 ON j2."template_id" = j1."template_id" AND ...
         WHERE t."datasource_id" = $1 AND (t."column_name" ILIKE ANY($2::text[]) OR j2."description" ILIKE ANY(...))
         ORDER BY "match_score" DESC, t."schema_name", t."table_name", t."column_name" LIMIT $3

    The joins are plain ``LEFT JOIN``s, inside the ranked read: a field a join adds ranks the row before the limit
    cuts, so the joined rows are read beside the pinned ones (the planner's hash join of each joined table), never
    after them as :func:`_joined_nearest_statement`'s ``LATERAL`` lookups of the ranked few are. A join's ``on``
    names a column of the row or a name an earlier join adds; its ``by`` pins its own columns.

    :return: the statement and its binds
    :rtype: tuple[str, list[Any]]
    :raises ValueError: when nothing is pinned, ``limit`` is negative, or a join is not a plain ``LEFT`` one or names
        nothing to join on
    """
    if not by:
        raise ValueError(f"a ranked read of {table} pins at least one column")
    if limit < 0:
        raise ValueError(f"a ranked read of {table} asks for no fewer than no rows, got {limit}")
    if joins:
        check_joins(table if isinstance(table, str) else "a derived relation", joins)
    if any(not join.left or join.counts or join.matched for join in joins):
        raise ValueError(f"a pattern-ranked read of {table} joins plain LEFT joins only")
    conditions, params = by_conditions(by, alias="t", stated=stated_literals(table))
    for fragment in where:
        conditions.append(narrowing(fragment, len(params)))
        params += list(fragment.params)
    qualifier: dict[str, str] = {}
    joined: list[str] = []
    selected = ["t.*"] if columns is None else [selected_column(column) for column in columns]
    for number, join in enumerate(joins, start=1):
        alias = f"j{number}"
        on = [
            f"{alias}.{quote_identifier(column)} = {qualifier.get(row, 't.' + quote_identifier(row))}"
            for column, row in join.on.items()
        ]
        pinned, values = by_conditions(join.by or {}, first=len(params) + 1, alias=alias)
        params += values
        joined.append(f" LEFT JOIN {quoted_table(join.table)} AS {alias} ON {' AND '.join([*on, *pinned])}")
        for name, column in join.columns.items():
            selected.append(f"{alias}.{quote_identifier(column)} AS {quote_identifier(name)}")
            qualifier[name] = f"{alias}.{quote_identifier(column)}"
    fields = [qualifier.get(field, f"t.{quote_identifier(field)}") for field in rank.fields]
    params.append(list(rank.patterns))
    patterns = f"${len(params)}::text[]"
    # a row passes when a field matches any pattern (each field's ILIKE ANY, the engine's own loop over the array);
    # only the rows passing count their patterns, one array scan a row
    conditions.append(f"({' OR '.join(f'{field} ILIKE ANY({patterns})' for field in fields)})")
    # WHY each pattern its own CASE over the one array's element, and not ``(SELECT count(*) FROM unnest($n) ...)``
    # (2026-11-15, measured at the hub search's real size, 117,540 columns, PostgreSQL, scratchpad h3/search): the
    # count sub-select is run once a passing row, and its set-up cost 1.75x / 1.78x / 1.81x / 1.11x the hand-written
    # statement for 1 / 3 / 8 / 1 broad tokens (pg1.log); a CASE an element is what the hand-written statement
    # summed, at its cost: 1.03x / 1.00x / 0.98x / 0.60x on PostgreSQL (pg2.log), 1.00x / 1.01x / 0.98x / 1.09x on
    # YugabyteDB (yb1.log). The patterns stay ONE array parameter whatever their number
    score = " + ".join(
        f"CASE WHEN {' OR '.join(f'{field} ILIKE ({patterns})[{place}]' for field in fields)} THEN 1 ELSE 0 END"
        for place in range(1, len(rank.patterns) + 1)
    )
    selected.append(f"({score or '0'}) AS {quote_identifier(rank.rank)}")
    params.append(limit)
    order = f"{quote_identifier(rank.rank)} DESC{then_terms(rank.then)}"
    sql = (
        f"SELECT {', '.join(selected)} FROM {relation(table, 't')}{''.join(joined)} "
        f"WHERE {' AND '.join(conditions)} ORDER BY {order} LIMIT ${len(params)}"
    )
    return sql, params


def _ranked_by(nearest: Nearest | TextRank) -> str | None:
    """the column a ranked read's rows say their rank under, best first when sorted descending: a text read's rank,
    a vector read's similarity (``1 - distance``); None for a vector read naming no similarity."""
    return nearest.rank if isinstance(nearest, TextRank) else nearest.similarity


def _joined_nearest_statement(
    ranked_sql: str, params: list[Any], nearest: Nearest | TextRank, joins: Sequence[Joined]
) -> tuple[str, list[Any]]:
    """the ranked statement (:func:`~threetears.interfaces.keyed.read_nearest_statement`'s) as a derived table, each
    of its rows ``LEFT JOIN LATERAL``-ed to the rows of each joined table its columns name (``j1."media_id" =
    r."media_id"`` and the join's own pins), best first by the rank it selects (:func:`_ranked_by`), up to the same
    ``LIMIT``: what :func:`join_rows` over the ranked rows answered, a row matching several parents once for each.

    :param ranked_sql: the ranked statement
    :ptype ranked_sql: str
    :param params: its binds, the limit last
    :ptype params: list[Any]
    :param nearest: the rank it orders by, naming the column it selects the rank under
    :ptype nearest: Nearest | TextRank
    :param joins: the joins, each a plain LEFT one
    :ptype joins: Sequence[Joined]
    :return: the statement and its binds
    :rtype: tuple[str, list[Any]]
    """
    ranked = _ranked_by(nearest)
    assert ranked is not None
    binds = list(params)
    joined, laterals = lateral_joins(joins, binds)
    selected = ["r.*", *joined]
    sql = (
        f"SELECT {', '.join(selected)} FROM ({ranked_sql}) AS r{laterals}"
        f" ORDER BY r.{quote_identifier(ranked)} DESC LIMIT ${len(params)}"
    )
    return sql, binds


async def read_state_arrow(
    conn: Any,
    table: str,
    *,
    scope_column: Annotated[str, DeclaredScope],
    state: Any,
    columns: Sequence[str] | None = None,
) -> Any:
    """every row of a LOCAL copy's table that one state holds, pinned by the table's declared state column
    (``scope_column = $1``), as DuckDB's own Arrow answer: one state's rows of the pod's own copy, never L3.

    For a table whose key does not name its state (AP results: the state is a row column, ``TableSchema.delta_scope``),
    the only bound read of one state's rows. The no_whole_reads rule judges the call bounded only when
    ``scope_column`` flows from a :class:`~threetears.interfaces.keyed.DeclaredScope` source (the table's own
    declaration); any other column is a predicate it cannot see.

    :param conn: the copy's DuckDB, an :class:`~threetears.core.keyed.InlineDuckDB` or
        :class:`~threetears.core.keyed.ThreadedDuckDB`
    :ptype conn: Any
    :param table: the table, a TRUSTED identifier
    :ptype table: str
    :param scope_column: the table's declared state column, a TRUSTED identifier
    :ptype scope_column: str
    :param state: the state's value of the column (``= $1``, typed by the column)
    :ptype state: Any
    :param columns: the columns read, TRUSTED identifiers; every column of the table when None
    :ptype columns: Sequence[str] | None
    :return: the rows
    :rtype: pyarrow.Table
    :raises ValueError: when ``conn`` reads L3: a state's rows there are read by key, never by this
    :raises TypeError: when ``conn`` is no DuckDB copy
    """
    if reads_l3(conn):
        raise ValueError(f"read_state_arrow reads a local copy, never L3: {table} by {scope_column} on {conn!r}")
    if not isinstance(conn, InlineDuckDB | ThreadedDuckDB):
        raise TypeError(f"read_state_arrow reads a DuckDB copy, not {type(conn).__name__}")
    sql, params = read_by_statement(table, by={scope_column: state}, columns=columns)
    return await sent_by(conn).fetch_arrow(sql, *params, types=None)
