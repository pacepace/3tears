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
- set operations: :func:`except_rows`, :func:`union`; each row of one beside the rows of another it matches:
  :func:`joined` (a ``JOIN``, or a ``LEFT JOIN``);
- run: :func:`read_step` (rows), :func:`read_step_arrow` (one Arrow table), each up to a ``limit`` when asked,
  :func:`count_step`, :func:`any_step` (whether it answers any row); a table of a step's rows: :func:`materialize`; a
  delete of a table's rows a change relation's keys name: :func:`delete_matching`; the first row two relations hold
  differently: :func:`first_difference`.

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
from typing import Any

from threetears.core.backends.arrow_reply import l3_fetch_arrow
from threetears.core.backends.protocol import parse_rowcount
from threetears.core.keyed._filters import (
    Derived,
    Fragment,
    Selected,
    quoted_table,
    narrowing,
    renumbered,
    selected_column,
)
from threetears.core.keyed.inline import InlineDuckDB
from threetears.core.keyed.page import Sort
from threetears.core.keyed.threaded import ThreadedDuckDB
from threetears.core.sql_fragments import quote_identifier
from threetears.interfaces.keyed.steps import Match, Quoted, Rel, Step, change_relation

__all__ = [
    "Match",
    "Quoted",
    "Step",
    "anti",
    "any_step",
    "change_relation",
    "count_step",
    "delete_matching",
    "distinct",
    "except_rows",
    "first_difference",
    "grouped",
    "joined",
    "largest",
    "latest_per_key",
    "materialize",
    "ordered",
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
        if isinstance(rel, Quoted):
            return f"{rel.name} AS {alias}"
        return f"{quoted_table(rel)} AS {alias}"

    def fragments(self, fragments: Sequence[Fragment]) -> list[str]:
        """each fragment's text, its parameters bound after those already bound."""
        texts = []
        for fragment in fragments:
            texts.append(narrowing(fragment, len(self.params)))
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


def joined(
    rel: Rel,
    other: Rel,
    *,
    on: Sequence[Match],
    columns: Sequence[str | Selected],
    where: Sequence[Fragment] = (),
    also: Sequence[Fragment] = (),
    left: bool = False,
) -> Step:
    """each row of ``rel`` (``t``) beside each row of ``other`` (``k``) the pairs match: a ``JOIN``, a ``t`` row once
    for every ``k`` row matching it; with ``left``, a ``LEFT JOIN``: a ``t`` row no ``k`` row matches kept once, every
    ``k`` value NULL.

    :param rel: the relation read
    :ptype rel: str | Derived | Step
    :param other: the relation joined to it
    :ptype other: str | Derived | Step
    :param on: the pairs matched
    :ptype on: Sequence[Match]
    :param columns: the columns kept: a ``str`` is a column of ``t``; a :class:`Selected` any TRUSTED expression of
        ``t``'s and ``k``'s columns under its name (``Selected('k."place"', "place")``)
    :ptype columns: Sequence[str | Selected]
    :param where: conditions on ``t`` and ``k``, outside the match, each written from ``$1``
    :ptype where: Sequence[Fragment]
    :param also: conditions inside the match (``ON``), on ``t`` and ``k``, each written from ``$1``
    :ptype also: Sequence[Fragment]
    :param left: keep a ``t`` row no ``k`` row matches
    :ptype left: bool
    :return: the step
    :rtype: Step
    :raises ValueError: when nothing is matched on, or no column is kept
    """
    if not on and not also:
        raise ValueError("a join matches on at least one pair or condition")
    if not columns:
        raise ValueError("a join names the columns it keeps")
    statement = _Statement()
    source = statement.source(rel, "t")
    joined_to = statement.source(other, "k")
    matched = [match.condition("t", "k") for match in on] + statement.fragments(also)
    conditions = statement.fragments(where)
    kind = "LEFT JOIN" if left else "JOIN"
    return statement.step(
        f"SELECT {_columns(columns)} FROM {source} {kind} {joined_to} ON {' AND '.join(matched)}{_where(conditions)}"
    )


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


def ordered(step: Rel, order: Sequence[Sort], *, limit: int | None = None) -> Step:
    """the step's rows in ``order`` (and the first ``limit`` of them), as a step of its own: what a read of it, or a
    table made from it whole (:func:`materialize`, an ordered renewal's ``INSERT ... SELECT``), answers in that order.
    A step reading it as one of its relations promises no order: a relation's rows have none.

    :param step: the step, or a relation read whole
    :ptype step: str | Derived | Step
    :param order: the order, at least one sort
    :ptype order: Sequence[Sort]
    :param limit: the most rows (the first in ``order``); every row when None
    :ptype limit: int | None
    :return: the step
    :rtype: Step
    :raises ValueError: for an order naming nothing, or a limit that is no count of rows
    """
    if not order:
        raise ValueError("an ordered step names its order: at least one sort")
    sql, params = _ordered(_as_step(step), order, limit)
    return Step(sql, tuple(params))


async def read_step(
    conn: Any, step: Rel, *, order: Sequence[Sort] = (), limit: int | None = None
) -> list[dict[str, Any]]:
    """the step's rows, as dicts.

    :param conn: anything with an asyncpg-shaped ``fetch`` (an :class:`~threetears.core.keyed.InlineDuckDB`)
    :ptype conn: Any
    :param step: the step, or a relation read whole (a change relation's rows)
    :ptype step: str | Derived | Step
    :param order: the rows' order; none promised when empty
    :ptype order: Sequence[Sort]
    :param limit: the most rows answered (the first in ``order``); every row when None
    :ptype limit: int | None
    :return: the rows
    :rtype: list[dict[str, Any]]
    :raises ValueError: for a limit that is no count of rows
    """
    sql, params = _ordered(_as_step(step), order, limit)
    return [dict(row) for row in await conn.fetch(sql, *params)]


async def read_step_arrow(
    conn: Any,
    step: Rel,
    *,
    types: Mapping[str, Any] | None = None,
    order: Sequence[Sort] = (),
    limit: int | None = None,
    size: int | None = None,
) -> Any:
    """the step's rows as one Arrow table typed by ``types`` (DuckDB's own Arrow answer through
    :meth:`~threetears.core.keyed.InlineDuckDB.fetch_arrow`; any other reader's rows made the table on a thread).
    On DuckDB (:class:`~threetears.core.keyed.InlineDuckDB`, :class:`~threetears.core.keyed.ThreadedDuckDB`) ``types``
    may be left out: the table is DuckDB's own answer, its columns and types the statement's.

    :param conn: an :class:`~threetears.core.keyed.InlineDuckDB` / :class:`~threetears.core.keyed.ThreadedDuckDB`,
        or anything with ``fetch``
    :ptype conn: Any
    :param step: the step, or a relation read whole
    :ptype step: str | Derived | Step
    :param types: column name -> its Arrow type, every column the step selects, in order; None on DuckDB for its own
    :ptype types: Mapping[str, pyarrow.DataType] | None
    :param order: the rows' order; none promised when empty
    :ptype order: Sequence[Sort]
    :param limit: the most rows answered (the first in ``order``); every row when None
    :ptype limit: int | None
    :param size: on DuckDB, the most rows one Arrow batch is sized for (the rows the step holds at most, as
        :func:`threetears.core.cache.duckdb.fetch_arrow` takes them): DuckDB sizes a batch's buffers for a million
        rows otherwise, and the table holds them while it is held. ``limit`` when None and given; DuckDB's own
        default when both are None. Unused off DuckDB (the rows arrive as they are and make the table)
    :ptype size: int | None
    :return: the rows
    :rtype: pyarrow.Table
    :raises TypeError: without ``types`` on a reader that is no DuckDB (its rows carry no Arrow types)
    :raises ValueError: for a limit that is no count of rows, or a size under one row
    """
    if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size < 1):
        raise ValueError(f"an Arrow batch is sized for at least one row, not {size!r}")
    sql, params = _ordered(_as_step(step), order, limit)
    if isinstance(conn, (InlineDuckDB, ThreadedDuckDB)):
        return await conn.fetch_arrow(sql, *params, types=types, size=size if size is not None else limit)
    if types is None:
        raise TypeError("read_step_arrow needs types= off DuckDB: its rows carry no Arrow types")
    return await l3_fetch_arrow(conn, sql, *params, types=types)


def _ordered(step: Step, order: Sequence[Sort], limit: int | None = None) -> tuple[str, list[Any]]:
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 0):
        raise ValueError(f"a step's limit is a count of rows, not {limit!r}")
    bounded = "" if limit is None else f" LIMIT {limit}"
    if not order:
        return (f"SELECT o.* FROM ({step.sql}) AS o{bounded}" if bounded else step.sql), list(step.params)
    ordered = ", ".join(
        f"o.{quote_identifier(sort.column)} {'DESC' if sort.descending else 'ASC'}"
        + ((" NULLS FIRST" if sort.descending else " NULLS LAST") if sort.nullable else "")
        for sort in order
    )
    return f"SELECT o.* FROM ({step.sql}) AS o ORDER BY {ordered}{bounded}", list(step.params)


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


async def any_step(conn: Any, step: Rel) -> bool:
    """whether the step answers any row: ``SELECT EXISTS``, its first row the end of the read.

    :param conn: anything with an asyncpg-shaped ``fetch``
    :ptype conn: Any
    :param step: the step, or a relation
    :ptype step: str | Derived | Step
    :return: whether it answers a row
    :rtype: bool
    """
    statement = _Statement()
    source = statement.source(_as_step(step), "e")
    rows = await conn.fetch(f"SELECT EXISTS (SELECT 1 FROM {source}) AS held", *statement.params)
    return bool(rows[0]["held"])


async def materialize(conn: Any, table: str, step: Rel, *, temp: bool = False, replace: bool = True) -> int:
    """write the step's rows into ``table``, a table of their own (``CREATE [TEMP] TABLE ... AS``): a change's stage
    of rows read again by later steps. With ``replace``, a table of that name is dropped first.

    :param conn: anything with an asyncpg-shaped ``execute``
    :ptype conn: Any
    :param table: the table made, a TRUSTED identifier (a stage, a change relation's)
    :ptype table: str
    :param step: the rows, a step or a relation
    :ptype step: str | Derived | Step
    :param temp: make it a temporary table (the session's own)
    :ptype temp: bool
    :param replace: drop a table of the name first; when False one there already is an error
    :ptype replace: bool
    :return: how many rows it holds
    :rtype: int
    """
    built = _as_step(step)
    target = quoted_table(table)
    if replace:
        await conn.execute(f"DROP TABLE IF EXISTS {target}")
    status = await conn.execute(f"CREATE {'TEMP ' if temp else ''}TABLE {target} AS {built.sql}", *built.params)
    return parse_rowcount(status)


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
    where: Sequence[Fragment] = (),
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
    :param where: conditions on ``t`` and ``k`` a row compared meets, beside the difference, each written from ``$1``
    :ptype where: Sequence[Fragment]
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
    conditions = [*statement.fragments(where), f"({differs})"]
    rows = await conn.fetch(
        f"SELECT {selected} FROM {left} JOIN {right} ON {joined} WHERE {' AND '.join(conditions)} "
        f"ORDER BY {ordered} LIMIT 1",
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
