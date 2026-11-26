"""a point read with joins, as keyed reads: the rows a ``WHERE`` names, then each joined table by the values they hold.

The one spelling of what a module used to write as one joined statement::

    SELECT dt.*, d.name AS datasource_name FROM datasource_tables dt
      JOIN datasources d ON d.id = dt.datasource_id
     WHERE dt.id = $1

is :func:`read_by_joined`: the pinned table's rows by its pins (:func:`~threetears.core.keyed.read_by`), then each
joined table by the values those rows hold (:func:`~threetears.core.keyed.read_joined`, one keyed batch), never a join
the planner may drive from the joined side. Each read row is matched to the joined rows holding its values, in Python,
and answered once for each, as the join answers it: a row no joined row matches is dropped (``JOIN``) or kept once
with every joined column None (``LEFT JOIN``); a NULL value matches nothing, as ``=`` matches nothing. The rows come in
the read's ``order``; the rows one read row matches, in no promised order, as the join left them.

A correlated count in the select list (``(SELECT COUNT(*) FROM datasource_columns dc WHERE dc.table_id = dt.id AND
dc.description IS NOT NULL) AS described``) is a counted join (:class:`Joined` with ``counts``): the joined table read
by the same keyed batch, and each read row kept once beside how many of its rows match it (0 when none does).

The no-whole-reads rule judges a call by the statements it sends (the read by its pins, each joined table by the
columns it is joined on), each keyed only as the hand-written join's tables were.

IO only: one statement for the read, one keyed batch a joined table.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Any, Final

from threetears.core.keyed._filters import Const, Fragment, Selected
from threetears.core.keyed._filters import refuse_append_only, sent_by
from threetears.core.keyed._joining import Joined, cast_of, check_joins, join_rows, joined_columns, lateral_joins
from threetears.interfaces.keyed.filters import AnyOf, by_conditions, quote_identifier, quoted_table
from threetears.core.keyed.page import keyset_page
from threetears.core.keyed.point import read_by
from threetears.core.keyed.rows import read_keyed
from threetears.interfaces.keyed.page import Sort
from threetears.interfaces.keyed.rows import KeyBatch
from threetears.observe import BuildOnce

__all__ = [
    "Joined",
    "read_by_joined",
    "read_reached",
]


async def read_by_joined(
    conn: Any,
    table: str,
    *,
    by: Mapping[str, Any],
    columns: Sequence[str | Selected] | None,
    joins: Sequence[Joined],
    where: Sequence[Fragment] = (),
    order: Sequence[Sort] = (),
    batch: int | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """the rows of ``table`` the pins name, each beside the rows of each joined table it matches: what ``fetch`` of
    the hand-written join answered.

    With ``limit``, the read is the first page of its rows in ``order`` (:func:`~threetears.core.keyed.keyset_page`,
    a seek past the page before in ``where``): ``... LEFT JOIN ... ORDER BY ... LIMIT n``, each join a ``LEFT`` (or
    counted) one, so the page holds the rows the statement's did.

    :param conn: anything with an asyncpg-shaped ``fetch``
    :ptype conn: Any
    :param table: the table pinned, a TRUSTED identifier
    :ptype table: str
    :param by: its pins, as :func:`~threetears.core.keyed.read_by` takes them; at least one
    :ptype by: Mapping[str, Any]
    :param columns: its columns in the row, as :func:`~threetears.core.keyed.read_by` takes them; every one when None
    :ptype columns: Sequence[str | Selected] | None
    :param joins: the tables joined, in order; each joins the row as the joins before it left it
    :ptype joins: Sequence[Joined]
    :param where: guards beside the pins, as :func:`~threetears.core.keyed.read_by` takes them
    :ptype where: Sequence[Fragment]
    :param order: the read's order, its own columns; none promised when empty
    :ptype order: Sequence[Sort]
    :param batch: the most values a joined table's statement names; None: the connection's
    :ptype batch: int | None
    :param limit: the most rows read, the first in ``order``; every row the pins name when None
    :ptype limit: int | None
    :return: the rows, as dicts: the read's columns, then each join's
    :rtype: list[dict[str, Any]]
    :raises ValueError: when nothing is joined, a join names nothing to join on, a counted join widens the row, or a
        page has no order or an inner join
    """
    check_joins(table, joins)
    if limit is not None and not order:
        raise ValueError(f"a page of a joined read of {table} names its order")
    if limit is not None and any(not (join.left or join.counts) for join in joins):
        raise ValueError(f"a page of a joined read of {table} joins LEFT only (an inner join drops rows past the page)")
    read, hidden = joined_columns(columns, joins)
    rows: list[dict[str, Any]] = (
        await read_by(conn, table, by=by, columns=read, where=where, order=order)
        if limit is None
        else (await keyset_page(conn, table, columns=read, by=by, order=order, limit=limit, where=where)).rows
    )
    return await join_rows(conn, rows, joins, batch=batch, hidden=hidden)


async def read_reached(
    conn: Any,
    table: str,
    *,
    start: Mapping[str, Any],
    on: str,
    to: str,
    columns: Sequence[str],
    by: Mapping[str, Any] | None = None,
    follow: Mapping[str, Any] | None = None,
    batch: int | None = None,
    depth: int | None = None,
    joins: Sequence[Joined] = (),
) -> list[dict[str, Any]]:
    """every row a walk of ``table``'s links reaches, each once: what a recursive statement walking them answered
    (``WITH RECURSIVE reached(node, path) AS (... UNION ALL SELECT s.<to>, ... JOIN reached r ON s.<on> = r.node
    WHERE ... AND NOT (s.<to> = ANY(r.path))) SELECT DISTINCT ...``), as a keyed read a level.

    The rows ``start`` pins are read first (:func:`~threetears.core.keyed.read_by`); each row names a node in ``to``
    (one whose ``follow`` columns hold their values; every row when None), and the rows whose ``on`` holds a node not
    walked yet (within ``by``) are read next, one keyed batch a level, until a level names no new node. A cycle ends
    where it comes back. Each read is keyed as ``on`` leads an index of the table (``start``'s own pins likewise).

    With ``depth``, the walk stops after that many levels (the ``start`` rows are the first) and is ONE statement
    (:func:`_reached_statement`): the levels as a recursive term capped at the depth, each level's rows found by key
    from the level before, and each reached row then joined by ``joins`` (a ``LATERAL`` lookup a row,
    :func:`~threetears.core.keyed._joining.lateral_joins`) -- the walk and what it reaches in one round trip, where a
    level a read cost one each (why: the comment at :func:`_reached_statement`).

    :param conn: anything with an asyncpg-shaped ``fetch``
    :ptype conn: Any
    :param table: the table, a TRUSTED identifier
    :ptype table: str
    :param start: the first rows' pins, as :func:`~threetears.core.keyed.read_by` takes them; when it pins ``on``,
        that value is the walk's first node
    :ptype start: Mapping[str, Any]
    :param on: the column a row is reached by (it holds a node), a TRUSTED identifier
    :ptype on: str
    :param to: the column naming the node a row leads to, a TRUSTED identifier
    :ptype to: str
    :param columns: the columns answered, TRUSTED identifiers; the rows answered distinct over them
    :ptype columns: Sequence[str]
    :param by: columns pinned on every row read after the first (``source_kind = 'dataset'``)
    :ptype by: Mapping[str, Any] | None
    :param follow: the values a row's columns hold for its ``to`` to be walked; every row's when None
    :ptype follow: Mapping[str, Any] | None
    :param batch: the most nodes a statement names; None: the connection's
    :ptype batch: int | None
    :param depth: the most levels walked, the ``start`` rows the first, in one statement; every level, a read each,
        when None
    :ptype depth: int | None
    :param joins: the tables each reached row is joined to, by its ``columns`` (a ``left`` join keeps a row none
        matches); only with ``depth``
    :ptype joins: Sequence[Joined]
    :return: the rows reached, each its ``columns`` (and each join's), distinct, level by level
    :rtype: list[dict[str, Any]]
    :raises ValueError: when ``start`` pins nothing, no column is answered, ``depth`` is under one, or ``joins`` come
        without a ``depth``
    """
    if not start:
        raise ValueError(f"a walk of {table} starts at the rows its pins name")
    if not columns:
        raise ValueError(f"a walk of {table} answers at least one column")
    if depth is not None:
        # one recursive statement: no live read reaches its levels, so an append-only table is refused there (a
        # versioned one's every version, a stamped one's gone rows); a walk a level a read is keyed reads, each live
        refuse_append_only(conn, [table, *(join.table for join in joins or ())], f"a walk of {table} in one statement")
        sql, params = _held_statement(
            table, start=start, on=on, to=to, columns=columns, by=by, follow=follow, depth=depth, joins=joins
        )
        return [dict(row) for row in await sent_by(conn).fetch(sql, *params)]
    if joins:
        raise ValueError(f"a walk of {table} joins what it reaches only in one statement: give it a depth")
    followed = dict(follow or {})
    read = list(dict.fromkeys([on, to, *followed, *columns]))
    walked: set[Any] = {start[on]} if on in start and start[on] is not None else set()
    answered: dict[tuple[Any, ...], dict[str, Any]] = {}
    level: list[dict[str, Any]] = await read_by(conn, table, by=start, columns=read)
    cast: str | None = None
    while level:
        nodes: list[Any] = []
        for row in level:
            answered.setdefault(tuple(row[column] for column in columns), {column: row[column] for column in columns})
            node = row[to]
            if node is None or node in walked or any(row[column] != value for column, value in followed.items()):
                continue
            walked.add(node)
            nodes.append(node)
        if not nodes:
            break
        cast = cast or cast_of(nodes)
        level = await read_keyed(
            conn, table, keys=KeyBatch((on,), [list(nodes)], (cast,)), columns=read, by=by, batch=batch
        )
    return list(answered.values())


#: a walk's statement by the shape of its call (:func:`_statement_key`), and the binds after its first rows' pins: the
#: text a shape writes is the same every call, so it is written once. WHY (2026-11-21): writing the visible-customer
#: walk's statement (threetears.agent.acl.visible_customers: four joins, 14 columns) cost 16.4 us of a 149 us call on
#: the exclusive runner, against the 72 us of the reached-CTE it replaced, and took the set over its bar of twice that
#: statement (test_visible_customers_speed_live, 2.0-2.4x; held, with the set's Python trimmed: 1.60-1.67x). Bounded at
#: 256 shapes (a process walks a few); a shape past the bound is written for its call
_HELD_STATEMENTS: BuildOnce[Hashable, tuple[str, tuple[Any, ...]]] = BuildOnce(cost=lambda _: 1, max_cost=256)

#: a first-row pin bound as a placeholder (``= $n``), whatever its value: the statement's text does not hold it
_BOUND: Final = object()


def _pin_shape(value: Any) -> Any:
    """how :func:`~threetears.interfaces.keyed.filters.by_conditions` writes a pin of ``value`` into the text: a value
    it binds as one placeholder (:data:`_BOUND`), or, for one it writes otherwise (a literal, ``IS NULL``, a set),
    the value itself beside its type (``1`` binds; ``True`` is written as ``TRUE``).

    :param value: the pin's value
    :ptype value: Any
    :return: the shape, hashable when the value is
    :rtype: Any
    """
    if value is None or value is True or value is False or isinstance(value, (AnyOf, Const)):
        return (type(value), value)
    return _BOUND


def _mapping_key(pins: Mapping[str, Any] | None) -> Hashable:
    """``pins`` exactly, each column and value beside its type (a :class:`~threetears.core.keyed.Pin` is rendered
    as written, a column of the same text quoted).

    :param pins: the pins
    :ptype pins: Mapping[str, Any] | None
    :return: a tuple of them; None for None
    :rtype: Hashable
    """
    return None if pins is None else tuple((type(c), c, type(v), v) for c, v in pins.items())


def _statement_key(
    table: str,
    *,
    start: Mapping[str, Any],
    on: str,
    to: str,
    columns: Sequence[str],
    by: Mapping[str, Any] | None,
    follow: Mapping[str, Any] | None,
    depth: int,
    joins: Sequence[Joined],
) -> Hashable | None:
    """what a walk's statement text is written from: every argument as given, except the first rows' bound pins,
    held by their shape (:func:`_pin_shape`); None when a value is not hashable (an :class:`AnyOf` of a list).

    :return: the key, or None
    :rtype: Hashable | None
    """
    key = (
        table,
        on,
        to,
        tuple(columns),
        type(depth),
        depth,
        tuple((type(c), c, _pin_shape(v)) for c, v in start.items()),
        _mapping_key(by),
        _mapping_key(follow),
        tuple(
            (
                j.table,
                tuple(j.on.items()),
                tuple(j.columns.items()),
                j.left,
                _mapping_key(j.by),
                j.matched,
                tuple(j.casts.items()),
                tuple(j.counts.items()),
            )
            for j in joins
        ),
    )
    try:
        hash(key)
    except TypeError:
        # NOSILENT: an unhashable pin (an AnyOf of a list) only means the shape is not held: its statement is
        # written for the call, as every statement was before shapes were held
        return None
    return key


def _held_statement(
    table: str,
    *,
    start: Mapping[str, Any],
    on: str,
    to: str,
    columns: Sequence[str],
    by: Mapping[str, Any] | None,
    follow: Mapping[str, Any] | None,
    depth: int,
    joins: Sequence[Joined],
) -> tuple[str, list[Any]]:
    """:func:`_reached_statement`'s statement and binds, its text written once a call shape
    (:data:`_HELD_STATEMENTS`): the first rows' pins are bound first, so a held shape's binds are this call's
    pins' then the binds held after them.

    :return: the statement and its binds
    :rtype: tuple[str, list[Any]]
    """
    key = _statement_key(
        table, start=start, on=on, to=to, columns=columns, by=by, follow=follow, depth=depth, joins=joins
    )
    if key is None:
        return _reached_statement(
            table, start=start, on=on, to=to, columns=columns, by=by, follow=follow, depth=depth, joins=joins
        )
    pinned = by_conditions(start, alias="t")[1]

    def write() -> tuple[str, tuple[Any, ...]]:
        sql, params = _reached_statement(
            table, start=start, on=on, to=to, columns=columns, by=by, follow=follow, depth=depth, joins=joins
        )
        return sql, tuple(params[len(pinned) :])

    sql, after = _HELD_STATEMENTS.get(key, write)
    return sql, [*pinned, *after]


def _reached_statement(
    table: str,
    *,
    start: Mapping[str, Any],
    on: str,
    to: str,
    columns: Sequence[str],
    by: Mapping[str, Any] | None,
    follow: Mapping[str, Any] | None,
    depth: int,
    joins: Sequence[Joined],
) -> tuple[str, list[Any]]:
    """the walk of :func:`read_reached` capped at ``depth`` levels, and what each reached row joins, as ONE statement::

        WITH RECURSIVE reached("member_id", "group_id", "level") AS (
            SELECT t."member_id", t."group_id", 1 FROM "group_members" AS t
             WHERE t."member_type" = 'user' AND t."member_id" = $1
            UNION
            SELECT t."member_id", t."group_id", r."level" + 1 FROM reached AS r
              JOIN "group_members" AS t ON t."member_id" = r."group_id" AND t."member_type" = 'group'
             WHERE r."level" < $2
        )
        SELECT DISTINCT r."group_id", j1."scope_type" AS "scope_type", ...
          FROM (SELECT DISTINCT "group_id" FROM reached) AS r
          LEFT JOIN LATERAL (SELECT ... FROM "role_assignments" AS j WHERE j."group_id" = r."group_id") AS j1 ON TRUE

    Each level's rows are found by key (``on`` and ``by``) from the level before; a row is reached at most once a
    level (``UNION``), so a cycle walks round only until the cap. The rows reached within the cap are the rows a walk
    of every level reaches within it, each at its shortest level, so the answer is :func:`read_reached`'s cut to the
    cap.

    WHY one statement (2026-11-15, D5): the visible-customer set of a caller (``threetears.agent.acl.visible_customers``)
    is the closure of the caller's groups to the configured nesting and those groups' grants; read a level a read and
    the grants a read more, it cost 3 round trips where the hand-written reached-CTE cost one (PostgreSQL, 5,000
    groups, a caller in two: 1.18-1.26 ms against 0.38-0.42 ms; scratchpad d5/lat/pg.log). As this one statement:
    0.55-0.69 ms on PostgreSQL (1.23-1.36x the hand-written CTE), 1.64-2.09 ms on YugabyteDB (1.14-1.15x); the
    guard is the caller's speed test (``test_visible_customers_speed_live``).

    A pin the caller fixes in code (:class:`~threetears.core.keyed.Const`, ``'user'`` / ``'group'`` above) is written as
    its literal, in ``start``, ``by`` and ``follow`` alike: bound, PostgreSQL never cached the walk's generic plan and
    re-planned it every call (Pace's ruling 2026-11-16; the why and numbers at :class:`Const`).

    :return: the statement and its binds
    :rtype: tuple[str, list[Any]]
    :raises ValueError: when ``depth`` is under one
    """
    if isinstance(depth, bool) or depth < 1:
        raise ValueError(f"a walk of {table} walks at least one level, not {depth!r}")
    followed = dict(follow or {})
    read = list(dict.fromkeys([on, to, *followed, *columns]))
    level = quote_identifier("level")
    listed = ", ".join(quote_identifier(column) for column in read)
    first, params = by_conditions(start, alias="t")
    pinned, values = by_conditions(by or {}, first=len(params) + 1, alias="t")
    params += values
    walked: list[str] = []
    for column, value in followed.items():
        # a code constant is written as its literal, as by_conditions writes one (Const: the why)
        if isinstance(value, Const):
            walked.append(f"r.{quote_identifier(column)} = {value.sql}")
            continue
        params.append(value)
        walked.append(f"r.{quote_identifier(column)} = ${len(params)}")
    params.append(depth)
    walked.append(f"r.{level} < ${len(params)}")
    on_link = [f"t.{quote_identifier(on)} = r.{quote_identifier(to)}", *pinned]
    recursive = (
        f"WITH RECURSIVE reached({listed}, {level}) AS ("
        f" SELECT {', '.join(f't.{quote_identifier(column)}' for column in read)}, 1"
        f" FROM {quoted_table(table)} AS t WHERE {' AND '.join(first)}"
        f" UNION SELECT {', '.join(f't.{quote_identifier(column)}' for column in read)}, r.{level} + 1"
        f" FROM reached AS r JOIN {quoted_table(table)} AS t ON {' AND '.join(on_link)}"
        f" WHERE {' AND '.join(walked)} )"
    )
    answered = ", ".join(f"r.{quote_identifier(column)}" for column in columns)
    distinct = ", ".join(quote_identifier(column) for column in columns)
    # fenced (``OFFSET 0``): PostgreSQL cannot estimate the walk's rows, so an unfenced lookup is pulled up into a join
    # it hashed from the joined side (every grant of 500 groups scanned for the caller's one, the plan test
    # test_the_typed_set_reads_the_callers_memberships_then_their_groups_grants, 2026-11-15); the reached-CTE this
    # replaced fenced its grants the same way
    joined, laterals = lateral_joins(joins, params, fenced=True)
    selected = ", ".join([answered, *joined])
    return (
        f"{recursive} SELECT DISTINCT {selected} FROM (SELECT DISTINCT {distinct} FROM reached) AS r{laterals}",
        params,
    )
