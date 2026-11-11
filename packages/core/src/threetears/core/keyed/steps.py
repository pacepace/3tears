"""relational steps over change relations: one helper for each shape a load's computation takes, written once.

A load computes over the relations its change registers (the rows a change produced, the held rows it touches, a
partition's chunk): which held rows a change names, which it leaves, the newest row of each key, a group's marks.
Each of those is one of a few shapes, and each shape is written here once, as a :class:`Step` (a ``SELECT`` and its
parameters), so a load names the step rather than writing its SQL again:

- the rows of one relation another names, or does not: :func:`semi`, :func:`anti` (``EXISTS`` / ``NOT EXISTS`` on
  :class:`Match` pairs, each null-safe when it says so, plus conditions of its own ``also=``);
- a projection, its distinct rows, a grouped aggregate, the newest row of each key (or every row tied newest): :func:`scan`,
  :func:`distinct`, :func:`grouped` (:func:`largest`, :func:`smallest`, :func:`row_count`, :func:`xor_hash`),
  :func:`latest_per_key`; rows kept by a window's value (a tier's rank): :func:`windowed`;
- set operations: :func:`except_rows`, :func:`union`;
- run: :func:`read_step` (rows), :func:`read_step_arrow` (one Arrow table), :func:`count_step`; a delete of a table's
  rows a change relation's keys name: :func:`delete_matching`; the first row two relations hold differently:
  :func:`first_difference`.

A step reads a relation by its name (a change relation's, declared by :func:`change_relation`), a :class:`Derived`
select, or another step, which it nests with its parameters renumbered, so steps compose: ``distinct(semi(held,
changed, on=...), ["state", "run_id"])`` is one statement. A step's relation is read as ``t``; the other relation of a
semi or anti step as ``k``. The SQL is one spelling for DuckDB (an :class:`~threetears.core.keyed.InlineDuckDB` in a
worker process, :class:`~threetears.core.keyed.ThreadedDuckDB`) and PostgreSQL.

The no-whole-reads rule judges a step over a change relation as bounded by that change; the same SQL written by hand
over one is a use-the-helper finding. A relation registered from a whole read of a table is judged at that read: the
marker never makes it keyed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from threetears.core.backends.arrow_reply import l3_fetch_arrow
from threetears.core.backends.protocol import parse_rowcount
from threetears.core.keyed._filters import Derived, Fragment, Selected, quoted_table, renumbered, selected_column
from threetears.core.keyed.page import Sort
from threetears.core.sql_fragments import quote_identifier

__all__ = [
    "Match",
    "Step",
    "anti",
    "change_relation",
    "count_step",
    "delete_matching",
    "distinct",
    "except_rows",
    "first_difference",
    "grouped",
    "largest",
    "latest_per_key",
    "read_step",
    "read_step_arrow",
    "row_count",
    "scan",
    "semi",
    "smallest",
    "union",
    "windowed",
    "xor_hash",
]


@dataclass(frozen=True)
class Step:
    """a relational step: one ``SELECT`` and its parameters, its placeholders numbered from ``$1``. Built by the
    helpers here, never written by hand; read by another step, or run by :func:`read_step` and its siblings.

    :param sql: the ``SELECT``, TRUSTED
    :ptype sql: str
    :param params: its parameters, ``$1`` first
    :ptype params: tuple[Any, ...]
    """

    sql: str
    params: tuple[Any, ...] = ()


#: what a step reads: a change relation's TRUSTED name, a derived select, or another step
Rel = str | Derived | Step


@dataclass(frozen=True)
class Match:
    """one column pair two relations are matched on.

    :param column: the column of the step's relation (``t``)
    :ptype column: str
    :param other: the other relation's column (``k``); the same name when None
    :ptype other: str | None
    :param null_safe: whether two NULLs match (``IS NOT DISTINCT FROM``); ``=`` otherwise
    :ptype null_safe: bool
    """

    column: str
    other: str | None = None
    null_safe: bool = False

    def condition(self, left: str, right: str) -> str:
        """the pair's condition between ``left`` and ``right`` (the aliases).

        :param left: the step relation's alias
        :ptype left: str
        :param right: the other relation's alias
        :ptype right: str
        :return: the condition, TRUSTED
        :rtype: str
        """
        mine = f"{left}.{quote_identifier(self.column)}"
        theirs = f"{right}.{quote_identifier(self.other or self.column)}"
        return f"{mine} IS NOT DISTINCT FROM {theirs}" if self.null_safe else f"{mine} = {theirs}"


def change_relation(name: str) -> str:
    """declare ``name`` a change relation: one a change registers (its produced rows, the held rows it touches, a
    partition's chunk), bounded by that change. The no-whole-reads rule reads the steps over it as change-bounded.

    :param name: the relation's TRUSTED name, as registered
    :ptype name: str
    :return: the name
    :rtype: str
    :raises ValueError: when the name is empty
    """
    if not name.strip():
        raise ValueError("a change relation is named")
    return name


class _Statement:
    """a statement being written: its text in parts and the parameters bound so far."""

    def __init__(self) -> None:
        self.params: list[Any] = []

    def source(self, rel: Rel, alias: str) -> str:
        """``rel`` as a FROM item under ``alias``, its parameters bound after those already bound."""
        if isinstance(rel, Step):
            text = renumbered(Fragment(rel.sql, rel.params), len(self.params))
            self.params += list(rel.params)
            return f"({text}) AS {alias}"
        if isinstance(rel, Derived):
            return f"({rel.sql}) AS {alias}"
        return f"{quoted_table(rel)} AS {alias}"

    def fragments(self, fragments: Sequence[Fragment]) -> list[str]:
        """each fragment's text, its parameters bound after those already bound."""
        texts = []
        for fragment in fragments:
            texts.append(renumbered(fragment, len(self.params)))
            self.params += list(fragment.params)
        return texts

    def step(self, sql: str) -> Step:
        """the statement ``sql`` as a step, with the parameters bound so far.

        :param sql: the statement's text
        :ptype sql: str
        :return: the step
        :rtype: Step
        """
        return Step(sql, tuple(self.params))


def _columns(columns: Sequence[str | Selected] | None, alias: str = "t") -> str:
    if columns is None:
        return f"{alias}.*"
    if not columns:
        raise ValueError("a step selects at least one column (None selects every one)")
    return ", ".join(selected_column(column, alias) for column in columns)


def _where(conditions: Sequence[str]) -> str:
    return f" WHERE {' AND '.join(conditions)}" if conditions else ""


def scan(rel: Rel, columns: Sequence[str | Selected] | None = None, *, where: Sequence[Fragment] = ()) -> Step:
    """``rel``'s rows (those ``where`` names), ``columns`` of them.

    :param rel: the relation
    :ptype rel: str | Derived | Step
    :param columns: the columns as written (``t."col"``) or expressions under their names; every column when None
    :ptype columns: Sequence[str | Selected] | None
    :param where: conditions on ``t``, each written from ``$1``
    :ptype where: Sequence[Fragment]
    :return: the step
    :rtype: Step
    """
    statement = _Statement()
    source = statement.source(rel, "t")
    conditions = statement.fragments(where)
    return statement.step(f"SELECT {_columns(columns)} FROM {source}{_where(conditions)}")


def _matched(
    rel: Rel,
    keys: Rel,
    on: Sequence[Match],
    *,
    present: bool,
    columns: Sequence[str | Selected] | None,
    where: Sequence[Fragment],
    also: Sequence[Fragment],
) -> Step:
    if not on and not also:
        raise ValueError("a semi or anti step matches on at least one pair or condition")
    statement = _Statement()
    source = statement.source(rel, "t")
    other = statement.source(keys, "k")
    inner = [match.condition("t", "k") for match in on] + statement.fragments(also)
    exists = f"{'' if present else 'NOT '}EXISTS (SELECT 1 FROM {other} WHERE {' AND '.join(inner)})"
    conditions = [exists, *statement.fragments(where)]
    return statement.step(f"SELECT {_columns(columns)} FROM {source}{_where(conditions)}")


def semi(
    rel: Rel,
    keys: Rel,
    *,
    on: Sequence[Match],
    columns: Sequence[str | Selected] | None = None,
    where: Sequence[Fragment] = (),
    also: Sequence[Fragment] = (),
) -> Step:
    """``rel``'s rows (``t``) that some row of ``keys`` (``k``) matches: ``EXISTS``, each row once however many match.

    :param rel: the relation read
    :ptype rel: str | Derived | Step
    :param keys: the relation naming the rows kept
    :ptype keys: str | Derived | Step
    :param on: the pairs matched
    :ptype on: Sequence[Match]
    :param columns: the columns of ``t`` kept; every one when None
    :ptype columns: Sequence[str | Selected] | None
    :param where: conditions on ``t``, outside the match, each written from ``$1``
    :ptype where: Sequence[Fragment]
    :param also: conditions inside the match, on ``t`` and ``k``, each written from ``$1``
    :ptype also: Sequence[Fragment]
    :return: the step
    :rtype: Step
    :raises ValueError: when nothing is matched on
    """
    return _matched(rel, keys, on, present=True, columns=columns, where=where, also=also)


def anti(
    rel: Rel,
    keys: Rel,
    *,
    on: Sequence[Match],
    columns: Sequence[str | Selected] | None = None,
    where: Sequence[Fragment] = (),
    also: Sequence[Fragment] = (),
) -> Step:
    """``rel``'s rows (``t``) that no row of ``keys`` (``k``) matches: ``NOT EXISTS`` (a NULL never drops a row, as
    ``NOT IN`` would). As :func:`semi` takes its arguments.

    :return: the step
    :rtype: Step
    :raises ValueError: when nothing is matched on
    """
    return _matched(rel, keys, on, present=False, columns=columns, where=where, also=also)


def distinct(rel: Rel, columns: Sequence[str | Selected], *, where: Sequence[Fragment] = ()) -> Step:
    """the distinct rows of ``columns`` of ``rel``.

    :param rel: the relation
    :ptype rel: str | Derived | Step
    :param columns: the columns, or expressions under their names
    :ptype columns: Sequence[str | Selected]
    :param where: conditions on ``t``, each written from ``$1``
    :ptype where: Sequence[Fragment]
    :return: the step
    :rtype: Step
    """
    statement = _Statement()
    source = statement.source(rel, "t")
    conditions = statement.fragments(where)
    return statement.step(f"SELECT DISTINCT {_columns(columns)} FROM {source}{_where(conditions)}")


def largest(column: str, *, cast: str | None = None, default: str | None = None, alias: str | None = None) -> Selected:
    """``max`` of a column as a grouped aggregate, cast first when ``cast`` says, ``default`` for a group of NULLs.

    :param column: the column of ``t``
    :ptype column: str
    :param cast: the SQL type it is compared as (``TIMESTAMPTZ``); as it is when None
    :ptype cast: str | None
    :param default: a TRUSTED SQL literal for a group with no value (``'-infinity'::TIMESTAMPTZ``); NULL when None
    :ptype default: str | None
    :param alias: the name the group carries it under; the column's when None
    :ptype alias: str | None
    :return: the aggregate
    :rtype: Selected
    """
    return _extreme("max", column, cast, default, alias)


def smallest(column: str, *, cast: str | None = None, default: str | None = None, alias: str | None = None) -> Selected:
    """``min`` of a column, as :func:`largest` takes it.

    :return: the aggregate
    :rtype: Selected
    """
    return _extreme("min", column, cast, default, alias)


def _extreme(function: str, column: str, cast: str | None, default: str | None, alias: str | None) -> Selected:
    value = f"t.{quote_identifier(column)}"
    if cast is not None:
        value = f"CAST({value} AS {cast})"
    expression = f"{function}({value})"
    if default is not None:
        expression = f"coalesce({expression}, {default})"
    return Selected(expression, alias or column)


def row_count(alias: str = "n") -> Selected:
    """the group's rows, ``count(*)``.

    :param alias: the name the group carries it under
    :ptype alias: str
    :return: the aggregate
    :rtype: Selected
    """
    return Selected("count(*)", alias)


def xor_hash(columns: Sequence[str], alias: str = "content") -> Selected:
    """an order-free digest of the group's rows over ``columns``: ``bit_xor(hash(...))`` (DuckDB).

    :param columns: the columns hashed, of ``t``
    :ptype columns: Sequence[str]
    :param alias: the name the group carries it under
    :ptype alias: str
    :return: the aggregate
    :rtype: Selected
    :raises ValueError: when no column is named
    """
    if not columns:
        raise ValueError("a group's digest covers at least one column")
    hashed = ", ".join(f"t.{quote_identifier(column)}" for column in columns)
    return Selected(f"bit_xor(hash({hashed}))", alias)


def grouped(
    rel: Rel,
    *,
    by: Sequence[str | Selected],
    aggregates: Sequence[Selected],
    where: Sequence[Fragment] = (),
) -> Step:
    """``rel``'s rows grouped by ``by``, each group one row of ``by`` and ``aggregates``.

    :param rel: the relation
    :ptype rel: str | Derived | Step
    :param by: the grouping columns, or expressions under their names (grouped by the expression)
    :ptype by: Sequence[str | Selected]
    :param aggregates: the aggregates (:func:`largest`, :func:`smallest`, :func:`row_count`, :func:`xor_hash`, or a
        TRUSTED :class:`Selected` of ``t``'s columns)
    :ptype aggregates: Sequence[Selected]
    :param where: conditions on ``t``, each written from ``$1``
    :ptype where: Sequence[Fragment]
    :return: the step
    :rtype: Step
    :raises ValueError: when there is nothing to group by or to aggregate
    """
    if not aggregates:
        raise ValueError("a grouped step aggregates at least one value (distinct() takes groups alone)")
    statement = _Statement()
    source = statement.source(rel, "t")
    conditions = statement.fragments(where)
    selected = ", ".join([selected_column(column) for column in by] + [selected_column(item) for item in aggregates])
    grouping = ", ".join(
        f"t.{quote_identifier(column)}" if isinstance(column, str) else column.expression for column in by
    )
    tail = f" GROUP BY {grouping}" if by else ""
    return statement.step(f"SELECT {selected} FROM {source}{_where(conditions)}{tail}")


def latest_per_key(
    rel: Rel,
    *,
    key: Sequence[str],
    order: Sequence[Sort],
    columns: Sequence[str | Selected] | None = None,
    where: Sequence[Fragment] = (),
    ties: bool = False,
) -> Step:
    """the first row of each ``key`` in ``order`` (the newest, ordered descending): ``DISTINCT ON``. With ``ties``,
    every row of a key whose ``order[0]`` value is the key's first (``max`` descending, ``min`` ascending), its NULLs
    never first: a window filter (:func:`windowed`).

    :param rel: the relation
    :ptype rel: str | Derived | Step
    :param key: the columns a row is kept once for
    :ptype key: Sequence[str]
    :param order: which row of a key comes first; the last column breaks ties, as a page's sort does
    :ptype order: Sequence[Sort]
    :param columns: the columns kept; every one when None
    :ptype columns: Sequence[str | Selected] | None
    :param where: conditions on ``t``, each written from ``$1``
    :ptype where: Sequence[Fragment]
    :param ties: keep every row tied first by ``order[0]`` (one sort column only), not one row a key
    :ptype ties: bool
    :return: the step
    :rtype: Step
    :raises ValueError: when ``key`` or ``order`` is empty, or ``ties`` is given more than one sort column
    """
    if not key or not order:
        raise ValueError("the newest row of each key names the key and the order")
    if ties:
        if len(order) != 1:
            raise ValueError("rows tied first are tied on one sort column")
        (sort,) = order
        partition = ", ".join(f"t.{quote_identifier(column)}" for column in key)
        first = "max" if sort.descending else "min"
        top = Selected(f"{first}({sort.sorted_by}) OVER (PARTITION BY {partition})", "__first")
        kept = Fragment(f"w.{quote_identifier(sort.column)} = w.__first")
        return windowed(rel, windows=[top], keep=[kept], columns=columns, where=where)
    statement = _Statement()
    source = statement.source(rel, "t")
    conditions = statement.fragments(where)
    keyed = ", ".join(f"t.{quote_identifier(column)}" for column in key)
    ordered = ", ".join(
        [f"t.{quote_identifier(column)}" for column in key]
        + [
            f"{sort.sorted_by} {'DESC' if sort.descending else 'ASC'}"
            + ((" NULLS LAST" if sort.descending else " NULLS FIRST") if sort.nullable else "")
            for sort in order
        ]
    )
    return statement.step(
        f"SELECT DISTINCT ON ({keyed}) {_columns(columns)} FROM {source}{_where(conditions)} ORDER BY {ordered}"
    )


def except_rows(rel: Rel, other: Rel) -> Step:
    """the distinct rows of ``rel`` that ``other`` does not hold, compared whole (two NULLs alike): ``EXCEPT``. Both
    hold the same columns in the same order.

    :param rel: the relation
    :ptype rel: str | Derived | Step
    :param other: the rows taken away
    :ptype other: str | Derived | Step
    :return: the step
    :rtype: Step
    """
    statement = _Statement()
    left = statement.source(rel, "a")
    right = statement.source(other, "b")
    return statement.step(f"SELECT a.* FROM {left} EXCEPT SELECT b.* FROM {right}")


def union(*rels: Rel, keep_all: bool = False) -> Step:
    """the rows of every relation: distinct (``UNION``), or every one as held (``UNION ALL``, ``keep_all``). Each
    holds the same columns in the same order.

    :param rels: the relations, at least two
    :ptype rels: str | Derived | Step
    :param keep_all: keep each relation's rows as they are, duplicates too
    :ptype keep_all: bool
    :return: the step
    :rtype: Step
    :raises ValueError: when fewer than two relations are named
    """
    if len(rels) < 2:
        raise ValueError("a union names at least two relations")
    statement = _Statement()
    parts = [f"SELECT u{index}.* FROM {statement.source(rel, f'u{index}')}" for index, rel in enumerate(rels)]
    return statement.step(f" UNION {'ALL ' if keep_all else ''}".join(parts))


def _as_step(rel: Rel) -> Step:
    return rel if isinstance(rel, Step) else scan(rel)


async def read_step(conn: Any, step: Rel, *, order: Sequence[Sort] = ()) -> list[dict[str, Any]]:
    """the step's rows, as dicts.

    :param conn: anything with an asyncpg-shaped ``fetch`` (an :class:`~threetears.core.keyed.InlineDuckDB`)
    :ptype conn: Any
    :param step: the step, or a relation read whole (a change relation's rows)
    :ptype step: str | Derived | Step
    :param order: the rows' order; none promised when empty
    :ptype order: Sequence[Sort]
    :return: the rows
    :rtype: list[dict[str, Any]]
    """
    sql, params = _ordered(_as_step(step), order)
    return [dict(row) for row in await conn.fetch(sql, *params)]


async def read_step_arrow(conn: Any, step: Rel, *, types: Mapping[str, Any], order: Sequence[Sort] = ()) -> Any:
    """the step's rows as one Arrow table typed by ``types`` (DuckDB's own Arrow answer through
    :meth:`~threetears.core.keyed.InlineDuckDB.fetch_arrow`; any other reader's rows made the table on a thread).

    :param conn: an :class:`~threetears.core.keyed.InlineDuckDB` / :class:`~threetears.core.keyed.ThreadedDuckDB`,
        or anything with ``fetch``
    :ptype conn: Any
    :param step: the step, or a relation read whole
    :ptype step: str | Derived | Step
    :param types: column name -> its Arrow type, every column the step selects, in order
    :ptype types: Mapping[str, pyarrow.DataType]
    :param order: the rows' order; none promised when empty
    :ptype order: Sequence[Sort]
    :return: the rows
    :rtype: pyarrow.Table
    """
    sql, params = _ordered(_as_step(step), order)
    return await l3_fetch_arrow(conn, sql, *params, types=types)


def _ordered(step: Step, order: Sequence[Sort]) -> tuple[str, list[Any]]:
    if not order:
        return step.sql, list(step.params)
    ordered = ", ".join(
        f"o.{quote_identifier(sort.column)} {'DESC' if sort.descending else 'ASC'}"
        + ((" NULLS FIRST" if sort.descending else " NULLS LAST") if sort.nullable else "")
        for sort in order
    )
    return f"SELECT o.* FROM ({step.sql}) AS o ORDER BY {ordered}", list(step.params)


async def count_step(conn: Any, step: Rel) -> int:
    """how many rows the step answers, counted where it runs.

    :param conn: anything with an asyncpg-shaped ``fetch``
    :ptype conn: Any
    :param step: the step, or a relation
    :ptype step: str | Derived | Step
    :return: the count
    :rtype: int
    """
    statement = _Statement()
    source = statement.source(_as_step(step), "c")
    rows = await conn.fetch(f"SELECT count(*) AS n FROM {source}", *statement.params)
    return int(rows[0]["n"])


async def delete_matching(
    conn: Any, table: str, keys: Rel, *, on: Sequence[Match], also: Sequence[Fragment] = ()
) -> int:
    """delete ``table``'s rows (``t``) that some row of ``keys`` (``k``, a change relation's) matches.

    :param conn: anything with an asyncpg-shaped ``execute``
    :ptype conn: Any
    :param table: the table, a TRUSTED identifier (``main.results``)
    :ptype table: str
    :param keys: the relation naming the rows deleted
    :ptype keys: str | Derived | Step
    :param on: the pairs matched
    :ptype on: Sequence[Match]
    :param also: conditions inside the match, on ``t`` and ``k``, each written from ``$1``
    :ptype also: Sequence[Fragment]
    :return: the rows deleted
    :rtype: int
    :raises ValueError: when nothing is matched on
    """
    if not on:
        raise ValueError(f"a delete from {table} by a change relation matches on its key")
    statement = _Statement()
    other = statement.source(keys, "k")
    inner = [match.condition("t", "k") for match in on] + statement.fragments(also)
    sql = f"DELETE FROM {quoted_table(table)} AS t WHERE EXISTS (SELECT 1 FROM {other} WHERE {' AND '.join(inner)})"
    return parse_rowcount(await conn.execute(sql, *statement.params))


async def first_difference(
    conn: Any,
    rel: Rel,
    other: Rel,
    *,
    on: Sequence[Match],
    compare: Sequence[tuple[str | Selected, str]],
    order: Sequence[str],
) -> dict[str, Any] | None:
    """the first row (in ``order``) of ``rel`` (``t``) whose matched row of ``other`` (``k``) holds a compared value
    otherwise (two NULLs alike), or None when every matched row agrees.

    :param conn: anything with an asyncpg-shaped ``fetch``
    :ptype conn: Any
    :param rel: the relation
    :ptype rel: str | Derived | Step
    :param other: the relation compared with
    :ptype other: str | Derived | Step
    :param on: the pairs a row is matched by
    :ptype on: Sequence[Match]
    :param compare: each value of ``t`` (a column, or a TRUSTED expression of ``t``'s columns) and the column of
        ``k`` it must equal
    :ptype compare: Sequence[tuple[str | Selected, str]]
    :param order: ``t``'s columns that order the rows, the first one differing answered
    :ptype order: Sequence[str]
    :return: the ``on`` and ``order`` columns of that row of ``t``, or None
    :rtype: dict[str, Any] | None
    :raises ValueError: when nothing is matched on, compared or ordered by
    """
    if not on or not compare or not order:
        raise ValueError("a first difference matches, compares and orders by at least one column each")
    statement = _Statement()
    left = statement.source(rel, "t")
    right = statement.source(other, "k")
    differs = " OR ".join(f"{_value(value)} IS DISTINCT FROM k.{quote_identifier(theirs)}" for value, theirs in compare)
    named = list(dict.fromkeys([match.column for match in on] + list(order)))
    selected = ", ".join(f"t.{quote_identifier(column)}" for column in named)
    joined = " AND ".join(match.condition("t", "k") for match in on)
    ordered = ", ".join(f"t.{quote_identifier(column)}" for column in order)
    rows = await conn.fetch(
        f"SELECT {selected} FROM {left} JOIN {right} ON {joined} WHERE {differs} ORDER BY {ordered} LIMIT 1",
        *statement.params,
    )
    return dict(rows[0]) if rows else None


def _value(value: str | Selected) -> str:
    return f"t.{quote_identifier(value)}" if isinstance(value, str) else value.expression


def windowed(
    rel: Rel,
    *,
    windows: Sequence[Selected],
    keep: Sequence[Fragment],
    columns: Sequence[str | Selected] | None = None,
    where: Sequence[Fragment] = (),
) -> Step:
    """``rel``'s rows kept by window values: each window (a TRUSTED expression of ``t``'s columns with an ``OVER``)
    is computed beside the row under its name, and the rows ``keep`` names (written over ``w.``: the row's columns and
    the windows' names) are kept, ``columns`` of them (``QUALIFY``, as a subquery: with ``columns`` named,
    PostgreSQL takes it too; every column, with the windows left out, is DuckDB's ``EXCLUDE``).

    :param rel: the relation
    :ptype rel: str | Derived | Step
    :param windows: the window values (``Selected("row_number() OVER (PARTITION BY t.run_id ORDER BY t.instant)",
        "rank")``)
    :ptype windows: Sequence[Selected]
    :param keep: the conditions a kept row meets, over ``w.``, each written from ``$1``
    :ptype keep: Sequence[Fragment]
    :param columns: the row's columns kept (never a window's name); every column ``rel`` holds when None (DuckDB)
    :ptype columns: Sequence[str | Selected] | None
    :param where: conditions on ``t`` before the windows are computed, each written from ``$1``
    :ptype where: Sequence[Fragment]
    :return: the step
    :rtype: Step
    :raises ValueError: when no window or no condition is named
    """
    if not windows or not keep:
        raise ValueError("a windowed step computes at least one window and keeps rows by at least one condition")
    statement = _Statement()
    source = statement.source(rel, "t")
    conditions = statement.fragments(where)
    inner = ", ".join(["t.*", *(selected_column(window) for window in windows)])
    kept = statement.fragments(keep)
    if columns is None:
        hidden = ", ".join(quote_identifier(window.alias) for window in windows)
        selected = f"w.* EXCLUDE ({hidden})"
    else:
        selected = _columns(columns, "w")
    return statement.step(
        f"SELECT {selected} FROM (SELECT {inner} FROM {source}{_where(conditions)}) AS w{_where(kept)}"
    )
