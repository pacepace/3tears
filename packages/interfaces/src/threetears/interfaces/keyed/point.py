"""point reads' statements: the rows one key names, bound as a hand-written ``WHERE a = $1 AND b = $2`` binds it.

The one spelling of the read a module used to write as text and send through ``fetch``, ``fetchrow`` or ``fetchval``::

    SELECT t."id", t."name" FROM "agents" AS t WHERE t."id" = $1 AND t."is_deleted" = $2

Each pinned column is compared with one parameter, ``= $n``, whose type the database infers from the column, exactly as
it inferred it for the statement written by hand; a ``None`` pins ``IS NULL``; an :class:`AnyOf` pins a set. A read
pinning nothing reads the table whole, and is refused.

Pure: :func:`read_by_statement` and :func:`read_row_statement` write the statement and its binds;
``threetears.core.keyed.read_by`` / ``read_row`` / ``read_value`` run them, and a reader with no 3tears core runs them
on its own connection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from dataclasses import dataclass
from typing import Any, Final

from threetears.interfaces.keyed.filters import (
    Derived,
    Fragment,
    Lock,
    Selected,
    by_conditions,
    narrowing,
    quote_identifier,
    relation,
    renumbered,
    selected_column,
)
from threetears.interfaces.keyed.keys import primary_key, stated_literals
from threetears.interfaces.keyed.page import Sort, order_by_terms
from threetears.interfaces.keyed.steps import Step

__all__ = [
    "Nearest",
    "TextRank",
    "read_by_statement",
    "read_nearest_statement",
    "read_row_beside_statement",
    "read_row_statement",
    "then_terms",
]


def _clauses(
    table: str | Derived,
    by: Mapping[str, Any],
    columns: Sequence[str | Selected] | None,
    limit: bool,
    lock: Lock | None = None,
    where: Sequence[Fragment] = (),
    order: Sequence[Sort] = (),
    also: str = "",
) -> tuple[str, str, str, list[Any]]:
    """a point read's select list, conditions and tail, and its parameters: the statement but its ``FROM``.

    :param table: the table, a TRUSTED identifier (``schema.table`` quoted a segment at a time)
    :ptype table: str
    :param by: the pinned columns
    :ptype by: Mapping[str, Any]
    :param columns: the columns read; every column (``t.*``) when None
    :ptype columns: Sequence[str | Selected] | None
    :param limit: whether one row is asked for (``LIMIT 1``)
    :ptype limit: bool
    :param lock: the row lock taken on the rows read, in the caller's transaction; none when None
    :ptype lock: Lock | None
    :param where: guards beside the pins, each a fragment written as ``t."col"`` from ``$1``, ANDed in its own
        parentheses after them; none when empty
    :ptype where: Sequence[Fragment]
    :param order: the sort the rows come in, the columns after the pins in the key or index that serves them; no
        order when empty
    :ptype order: Sequence[Sort]
    :param also: a TRUSTED select-list item after the columns (a nearest read's similarity), naming placeholders past
        the pins' and guards'; none when empty
    :ptype also: str
    :return: the select list, the conditions ANDed, the tail (order, limit, lock), and the parameters
    :rtype: tuple[str, str, str, list[Any]]
    :raises ValueError: when nothing is pinned, or ``columns`` is empty
    """
    if not by:
        raise ValueError(f"a point read of {table} pins at least one column; with none it reads the table whole")
    if columns is not None and not columns:
        raise ValueError(f"a point read of {table} names the columns it reads, or None for every column")
    selected = "t.*" if columns is None else ", ".join(selected_column(column) for column in columns)
    if also:
        selected = f"{selected}, {also}"
    # a text a declared partial index's predicate names is written as its literal (keys.stated_literals)
    conditions, params = by_conditions(by, alias="t", stated=stated_literals(table))
    for fragment in where:
        conditions.append(narrowing(fragment, len(params)))
        params += list(fragment.params)
    tail = (
        (f" ORDER BY {order_by_terms(order)}" if order else "")
        + (" LIMIT 1" if limit else "")
        + ("" if lock is None else lock.of("t"))
    )
    return selected, " AND ".join(conditions), tail, params


def _sql(
    table: str | Derived,
    by: Mapping[str, Any],
    columns: Sequence[str | Selected] | None,
    limit: bool,
    lock: Lock | None = None,
    where: Sequence[Fragment] = (),
    order: Sequence[Sort] = (),
    also: str = "",
) -> tuple[str, list[Any]]:
    """the statement and its parameters, as :func:`_clauses` takes them.

    :return: the statement, and its parameters
    :rtype: tuple[str, list[Any]]
    :raises ValueError: when nothing is pinned, or ``columns`` is empty
    """
    selected, conditions, tail, params = _clauses(table, by, columns, limit, lock, where, order, also)
    return f"SELECT {selected} FROM {relation(table, 't', params)} WHERE {conditions}{tail}", params


def _beside_sql(
    table: str | Derived,
    beside: Derived,
    by: Mapping[str, Any],
    columns: Sequence[str | Selected] | None,
    lock: Lock | None,
    where: Sequence[Fragment],
) -> tuple[str, list[Any]]:
    """the first row's read (:func:`_sql`, ``LIMIT 1``) with a table-less derived row read beside it (``b``), its
    values taken before the select list runs.

    :return: the statement, and its parameters
    :rtype: tuple[str, list[Any]]
    :raises ValueError: when nothing is pinned, or ``columns`` is empty
    """
    selected, conditions, tail, params = _clauses(table, by, columns, True, lock, where)
    return (
        f"SELECT {selected} FROM {relation(beside, 'b', params)} CROSS JOIN {relation(table, 't', params)} WHERE {conditions}{tail}",
        params,
    )


def read_by_statement(
    table: str | Derived,
    *,
    by: Mapping[str, Any],
    columns: Sequence[str | Selected] | None = None,
    lock: Lock | None = None,
    where: Sequence[Fragment] = (),
    order: Sequence[Sort] = (),
) -> tuple[str, list[Any]]:
    """the statement reading every row of ``table`` the pinned columns name, and its binds: what
    ``threetears.core.keyed.read_by`` sends.

    :param table: the table, a TRUSTED identifier (``schema.table`` quoted a segment at a time)
    :ptype table: str | Derived
    :param by: column -> its value (``= $n``, typed by the column), ``None`` (``IS NULL``) or an :class:`AnyOf`;
        columns are TRUSTED identifiers; at least one
    :ptype by: Mapping[str, Any]
    :param columns: the columns read, TRUSTED identifiers or :class:`Selected` expressions; every column (``t.*``)
        when None
    :ptype columns: Sequence[str | Selected] | None
    :param lock: the row lock taken on the rows read, in the caller's transaction; none when None
    :ptype lock: Lock | None
    :param where: guards beside the pins (an expiry the row must not be past), each a fragment written as
        ``t."col"`` from ``$1``; none when empty
    :ptype where: Sequence[Fragment]
    :param order: the sort the rows come in (insertion order by a rowid an index carries after the pins, say): the
        columns after the pins in the key or index that serves the read; in no promised order when empty
    :ptype order: Sequence[Sort]
    :return: the statement, and its binds
    :rtype: tuple[str, list[Any]]
    :raises ValueError: when nothing is pinned, or ``columns`` is empty
    """
    return _sql(table, by, columns, limit=False, lock=lock, where=where, order=order)


def read_row_statement(
    table: str | Derived,
    *,
    by: Mapping[str, Any],
    columns: Sequence[str | Selected] | None = None,
    lock: Lock | None = None,
    where: Sequence[Fragment] = (),
    beside: Derived | None = None,
) -> tuple[str, list[Any]]:
    """the statement reading the first row of ``table`` the pinned columns name (``LIMIT 1``), and its binds: what
    ``threetears.core.keyed.read_row`` (and ``read_value``, of one column) sends.

    :param table: as :func:`read_by_statement` takes it
    :ptype table: str | Derived
    :param by: as :func:`read_by_statement` takes it
    :ptype by: Mapping[str, Any]
    :param columns: as :func:`read_by_statement` takes them
    :ptype columns: Sequence[str | Selected] | None
    :param lock: as :func:`read_by_statement` takes it
    :ptype lock: Lock | None
    :param where: as :func:`read_by_statement` takes them
    :ptype where: Sequence[Fragment]
    :param beside: a TRUSTED table-less derived row (``SELECT current_setting(...) AS was ... OFFSET 0``) read beside
        the table's row, as ``FROM (<beside>) AS b CROSS JOIN <table> AS t``, its columns selectable as ``b.<column>``
        (:class:`Selected`): its values are taken before the select list runs, so a value read there is the one from
        before a ``set_config`` the select list makes. None reads the table alone
    :ptype beside: Derived | None
    :return: the statement, and its binds
    :rtype: tuple[str, list[Any]]
    :raises ValueError: when nothing is pinned, or ``columns`` is empty
    """
    if beside is not None:
        return _beside_sql(table, beside, by, columns, lock, where)
    return _sql(table, by, columns, limit=True, lock=lock, where=where)


def read_row_beside_statement(
    table: str,
    *,
    by: Mapping[str, Any],
    columns: Sequence[str | Selected],
    beside: Step,
    limit: int,
    present: str,
    first: str | None = None,
) -> tuple[str, list[Any]]:
    """the statement reading the first row of ``table`` the pins name (``r``, as :func:`read_row_statement` reads it)
    and, beside it, up to ``limit`` rows of a step that reads that row's columns (``r."col"``): one statement, so one
    round trip and one snapshot for the row and the rows it decides. What ``threetears.core.keyed.read_row_beside``
    sends::

        SELECT r.*, s.* FROM (<the row's read>) AS r
          LEFT JOIN LATERAL (SELECT TRUE AS "<present>", b.* FROM (<beside>) AS b LIMIT $n) AS s ON TRUE

    A row with no step row comes back once, ``present`` and every step column NULL; no row comes back when the pins
    name none.

    :param table: the table, a TRUSTED identifier (``schema.table`` quoted a segment at a time)
    :ptype table: str
    :param by: its pins, as :func:`read_by_statement` takes them; at least one
    :ptype by: Mapping[str, Any]
    :param columns: the row's columns, as :func:`read_by_statement` takes them; each one the step reads among them
    :ptype columns: Sequence[str | Selected]
    :param beside: the step, its conditions on the row written ``r."col"`` (a correlated read, each leg a keyed read
        of its own relation: :mod:`threetears.core.keyed.steps`)
    :ptype beside: Step
    :param limit: the most step rows read
    :ptype limit: int
    :param present: the name of the column that is TRUE on a step row (NULL on the row alone), a TRUSTED identifier
    :ptype present: str
    :param first: given, the name of a column TRUE on one step row and FALSE on the rest, a TRUSTED identifier: the
        row's columns then come back on that row alone (and on the row alone), NULL on every other, in no particular
        order of rows. Each answered row otherwise repeats them, and at hundreds of step rows that is hundreds of
        values more to send and decode for one row read (the change notes' ring, 2026-11-16, YugabyteDB, 500 notes
        with the ring's five columns: the statement's best 1.58 -> 1.33 ms, its median 2.08 -> 1.64 ms, 2026-11-16).
        ``None`` repeats them on every row
    :ptype first: str | None
    :return: the statement, and its binds: the pins', the step's, then ``limit``
    :rtype: tuple[str, list[Any]]
    :raises ValueError: when nothing is pinned, or for a limit under one row
    """
    if isinstance(limit, bool) or limit < 1:
        raise ValueError(f"a read beside a row reads at least one row of its step, not {limit!r}")
    row, params = _sql(table, by, columns, limit=True)
    step = renumbered(Fragment(beside.sql, beside.params), len(params))
    params += list(beside.params)
    params.append(limit)
    marked = quote_identifier(present)
    if first is None:
        return (
            f"SELECT r.*, s.* FROM ({row}) AS r LEFT JOIN LATERAL (SELECT TRUE AS {marked}, b.* "
            f"FROM ({step}) AS b LIMIT ${len(params)}) AS s ON TRUE",
            params,
        )
    # numbered after the limit, so the one numbered first is among the rows kept whatever order an engine limits in
    once = quote_identifier(first)
    names = [quote_identifier(column.alias if isinstance(column, Selected) else column) for column in columns]
    head = ", ".join(f"CASE WHEN s.{once} IS NOT FALSE THEN r.{name} END AS {name}" for name in names)
    return (
        f"SELECT {head}, s.* FROM ({row}) AS r LEFT JOIN LATERAL (SELECT TRUE AS {marked}, "
        f"row_number() OVER () = 1 AS {once}, k.* FROM (SELECT b.* FROM ({step}) AS b LIMIT ${len(params)}) AS k) "
        "AS s ON TRUE",
        params,
    )


#: the names a nearest read's wrapper uses: the distance it re-sorts by, and the whole row (every column)
_NEAREST_DISTANCE: Final = "_nearest_distance"
_NEAREST_ROW: Final = "_nearest_row"


#: pgvector's distance operators: L2, cosine, inner product (negated), L1
_DISTANCES: Final[frozenset[str]] = frozenset({"<->", "<=>", "<#>", "<+>"})
#: a text search configuration's name, plain or schema-qualified (``english``, ``pg_catalog.simple``): what
#: :class:`TextRank` writes into its statement as a literal, so nothing else may reach it
_CONFIGURATION: Final[re.Pattern[str]] = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?")


@dataclass(frozen=True)
class Nearest:
    """the distance a nearest-rows read orders by: ``t."<column>" OPERATOR(<schema>.<operator>) $n::text::<schema>.vector``,
    the vector bound as its text (asyncpg has no vector codec), as an approximate-nearest-neighbour index on the column
    reads it.

    :param column: the vector column, a TRUSTED identifier
    :ptype column: str
    :param vector: the vector the rows are nearest to, as pgvector's text (``'[1,0,0]'``)
    :ptype vector: Any
    :param operator: the distance, one of pgvector's ``<->`` (L2), ``<=>`` (cosine), ``<#>``, ``<+>``
    :ptype operator: str
    :param schema: the schema pgvector's type and operators live in, a TRUSTED identifier
    :ptype schema: str
    :param similarity: the name each row carries ``1 - <distance>`` under (cosine similarity, for ``<=>``); not
        selected when None
    :ptype similarity: str | None
    """

    column: str
    vector: Any
    operator: str = "<=>"
    schema: str = "public"
    similarity: str | None = None

    def __post_init__(self) -> None:
        if self.operator not in _DISTANCES:
            raise ValueError(f"a nearest read's distance is one of {sorted(_DISTANCES)}, got {self.operator!r}")

    def distance(self, place: int) -> str:
        """the distance of the row ``t`` from the vector bound at ``$place``.

        :param place: the vector's placeholder number
        :ptype place: int
        :return: the expression
        :rtype: str
        """
        return (
            f"t.{quote_identifier(self.column)} OPERATOR({self.schema}.{self.operator}) "
            f"${place}::text::{self.schema}.vector"
        )


@dataclass(frozen=True)
class TextRank:
    """the full-text rank a ranked read orders by, best first: ``ts_rank_cd(t."<column>", websearch_to_tsquery('<config>',
    $n))``, the query's text bound once for the rank. The match itself (``@@``) is the caller's guard, written through
    :func:`~threetears.core.data.gin.gin_filter` where YugabyteDB's GIN index refuses the tsquery: the rank orders the
    rows the pins and guards leave, and names no index.

    :param column: the ``tsvector`` column, a TRUSTED identifier
    :ptype column: str
    :param text: the query's text, as ``websearch_to_tsquery`` reads it
    :ptype text: Any
    :param rank: the name each row carries its rank under
    :ptype rank: str
    :param config: the text search configuration's name, plain or schema-qualified; written into the statement as a
        literal, so anything else (a quote, a space, a third part) is refused here, before any statement is written
        (a caller's language, conversations' ``search(query_language=...)``, passes through it)
    :ptype config: str
    :param then: the order rows tied on the rank come in, after it (:func:`then_terms`: a column named as text,
        ascending, or a :class:`Sort`, ``Sort("last_used_at", descending=True, nulls="last")``), ending in the key so
        the order is total: a hand-written ``ORDER BY fts_rank DESC, <then>``; the engine's order among ties when
        empty. A ranked read with ``joins=`` orders its joined rows by the rank alone
    :ptype then: tuple[str | Sort, ...]
    """

    column: str
    text: Any
    rank: str = "fts_rank"
    config: str = "english"
    then: tuple[str | Sort, ...] = ()

    def __post_init__(self) -> None:
        if _CONFIGURATION.fullmatch(self.config) is None:
            raise ValueError(f"a text search configuration is named by an identifier, got {self.config!r}")

    def distance(self, place: int) -> str:
        """the row ``t``'s rank against the query bound at ``$place``, the expression a ranked read orders by
        (descending: the best match first).

        :param place: the query text's placeholder number
        :ptype place: int
        :return: the expression
        :rtype: str
        """
        return f"ts_rank_cd(t.{quote_identifier(self.column)}, websearch_to_tsquery('{self.config}', ${place}))"


@dataclass(frozen=True)
class PatternCount:
    """the rank a ranked read orders by, best first: how many of ``patterns`` the row matches, a pattern matched when
    any of ``fields`` is ``ILIKE`` it; then ``then``, ascending. A row matching none is not read. The patterns are bound
    as ONE array whatever their number (``$n::text[]``), counted a pattern a ``CASE`` over the array's element
    (``CASE WHEN t."f1" ILIKE ($n::text[])[1] OR ... THEN 1 ELSE 0 END + ...``); the rows pass by
    ``t."f1" ILIKE ANY($n::text[]) OR ...``. A field names a column of the row or a column a ``LEFT`` join of the
    read adds (``threetears.core.keyed.read_nearest``'s ``joins=``, by the name it lands under), so a field the read's
    table does not hold (a template's description) ranks the row too.

    WHY (hub use-the-helper, 2026-11-15): the hub's schema search wrote one ``CASE WHEN <field> ILIKE $k ...`` a token
    summed into ``match_score`` (``schema_endpoints._fetch_search_results``), its text growing with the tokens. Read a
    token at a time through the keyed helpers it cost 1.15-1.87x the one statement on PostgreSQL and up to 3.27x on
    YugabyteDB at real size (117,540 columns of 5,615 tables; scratchpad h2/search pg2.log, yb1.log), so the rank stays
    one statement, written here, its pins judged as every keyed read's: 0.60-1.03x the hand-written statement on
    PostgreSQL, 0.98-1.09x on YugabyteDB (scratchpad h3/search pg2.log, yb1.log; the cost of each shape at
    ``threetears.core.keyed.point._pattern_count_statement``).

    :param fields: the matched fields, TRUSTED identifiers: a column of the row, or a joined column's name
    :ptype fields: tuple[str, ...]
    :param patterns: the ``ILIKE`` patterns, as the caller wrote them (``%age%``); none reads no row
    :ptype patterns: Sequence[str]
    :param rank: the name each row carries its count under
    :ptype rank: str
    :param then: the order rows of the same count come in (:func:`then_terms`: a column of the row named as text,
        ascending, or a :class:`Sort`), the key after the pins last, so the order is total; TRUSTED identifiers
    :ptype then: tuple[str | Sort, ...]
    """

    fields: tuple[str, ...]
    patterns: Sequence[str]
    rank: str = "match_score"
    then: tuple[str | Sort, ...] = ()

    def __post_init__(self) -> None:
        if not self.fields:
            raise ValueError("a pattern count names at least one field it matches")


def then_terms(then: Sequence[str | Sort]) -> str:
    """the ``ORDER BY`` terms a ranked read (a :class:`TextRank`'s, a :class:`PatternCount`'s) orders the rows tied on
    its rank by, after the rank: ``, t."a", t."b" DESC NULLS LAST``; empty for none. One spelling for both ranks: a
    column of the row named as text sorts ascending, a :class:`Sort` as a listing's ``order=`` term sorts.

    :param then: the order, each a column of the row (TRUSTED) or a :class:`Sort`
    :ptype then: Sequence[str | Sort]
    :return: the terms, each after a comma
    :rtype: str
    """
    terms = [f"t.{quote_identifier(term)}" if isinstance(term, str) else order_by_terms([term]) for term in then]
    return "".join(f", {term}" for term in terms)


def read_nearest_statement(
    table: str | Derived,
    *,
    by: Mapping[str, Any],
    nearest: Nearest | TextRank,
    limit: int,
    columns: Sequence[str | Selected] | None = None,
    where: Sequence[Fragment] = (),
) -> tuple[str, list[Any]]:
    """the statement reading the ``limit`` rows of ``table`` the pinned columns name nearest to a vector, nearest
    first (or, given a :class:`TextRank`, best matched by a text query, best first), and its binds: what
    ``threetears.core.keyed.read_nearest`` sends::

        SELECT n."id", n."similarity" FROM (
            SELECT t."id", 1 - (t."embedding" OPERATOR(public.<=>) $3::text::public.vector) AS "similarity",
                t."embedding" OPERATOR(public.<=>) $3::text::public.vector AS "_nearest_distance"
            FROM "memories" AS t WHERE t."agent_id" = $1 AND t."user_id" = $2
            ORDER BY t."embedding" OPERATOR(public.<=>) $3::text::public.vector LIMIT $4
        ) AS n ORDER BY n."_nearest_distance"

    The inner order is the distance alone (no tie-break column), so an approximate-nearest-neighbour index on the column
    answers it; rows at the same distance come in the engine's order, as they did from the statement written by hand.
    The wrapper re-sorts the ``limit`` rows by distance (why: the comment at it).

    :param table: the table, a TRUSTED identifier (``schema.table`` quoted a segment at a time)
    :ptype table: str | Derived
    :param by: as :func:`read_by_statement` takes it; at least one column
    :ptype by: Mapping[str, Any]
    :param nearest: the vector column, the vector, and the distance; or the ``tsvector`` column and the query's text
    :ptype nearest: Nearest | TextRank
    :param limit: how many rows, at most
    :ptype limit: int
    :param columns: as :func:`read_by_statement` takes them; the similarity, when named, after them
    :ptype columns: Sequence[str | Selected] | None
    :param where: as :func:`read_by_statement` takes them
    :ptype where: Sequence[Fragment]
    :return: the statement, and its binds
    :rtype: tuple[str, list[Any]]
    :raises ValueError: when nothing is pinned, ``columns`` is empty, or ``limit`` is negative
    """
    if limit < 0:
        raise ValueError(f"a nearest read of {table} asks for no fewer than no rows, got {limit}")
    place = (
        1
        + len(by_conditions(by, alias="t", stated=stated_literals(table))[1])
        + sum(len(fragment.params) for fragment in where)
        # a derived relation's parameters, bound after the pins' and the filters' (:func:`relation`)
        + (len(table.params) if isinstance(table, Derived) else 0)
    )
    distance = nearest.distance(place)
    if isinstance(nearest, TextRank):
        # the best match first, as the hand-written ``ORDER BY fts_rank DESC LIMIT $k`` read it; ties on the rank in
        # the ``then`` order (a listing's recency, its key last), as ``ORDER BY fts_rank DESC, <then>`` read them
        also = f"{distance} AS {quote_identifier(nearest.rank)}"
        ties = then_terms(nearest.then)
        if columns is None:
            sql, params = _sql(table, by, columns, limit=False, where=where, also=also)
            return f"{sql} ORDER BY {distance} DESC{ties} LIMIT ${place + 1}", [*params, nearest.text, limit]
        # WHY the columns are read for the kept rows alone (MS1, 2026-11-15): no index serves a text rank, so the read
        # ranks every row its pins and match leave, and a select list written beside the rank is read and evaluated
        # for each of those rows, not for the ``limit`` kept. Memory's ambient keyword arm selects
        # ``t.embedding::text``. For an owner of 20,000 memories (1024-dim, among 100,050) all matching the query:
        # PostgreSQL 357 ms, 330 of them casting every embedding to text; YugabyteDB 662-699 ms, reading every
        # embedding from storage. Ranking the rows whole in a derived table (``t.*``, the select list over its kept
        # rows) fixed PostgreSQL (35 ms) but not YugabyteDB (687 ms: the outer list naming ``t.embedding`` still
        # fetched it for every ranked row; 154 ms with no embedding named at all). So the ranked read selects only
        # the table's primary key and the rank, under its order and ``LIMIT``, and each kept row is read again by
        # that key (``CROSS JOIN LATERAL``, driven from the kept rows: as a plain join YugabyteDB hashed every row of
        # the agent, see the joined read below) for the select list, Selected expressions included (each names the
        # row ``t``, as the lateral read is named), re-sorted by the rank, ties in the same ``then`` order. Same rows
        # (one statement, one snapshot: each kept row is there to read again), same order. A relation with no
        # declared primary key (a derived one) keeps the derived-table form. By key, the same owner: PostgreSQL 27 ms,
        # YugabyteDB 74-90 ms. The cost is a key lookup per kept row: a 50-memory owner's 20 kept rows took
        # 1.8 ms on PostgreSQL as before, and 2.95 ms on YugabyteDB against 1.8-2.2 (its selective queries 1.0 ms
        # against 0.9). Measured: scratchpad ms1/ms1-m2-*.raw (before), ms1-m4-*.raw (derived table), ms1-m5-*.raw
        # (by key); held by packages/agent/memory/tests/integration/test_keyword_arm_plans.py.
        rank = quote_identifier(nearest.rank)
        selected = [selected_column(column) for column in columns]
        key = primary_key(table)
        if key is None:
            sql, params = _sql(table, by, None, limit=False, where=where, also=also)
            return (
                f"SELECT {', '.join([*selected, f't.{rank}'])} FROM ({sql} ORDER BY {distance} DESC{ties}"
                f" LIMIT ${place + 1}) AS t ORDER BY t.{rank} DESC{ties}",
                [*params, nearest.text, limit],
            )
        listed = ", ".join([*selected, f"r.{rank}"])
        sql, params = _sql(table, by, key, limit=False, where=where, also=also)
        again = " AND ".join(f"t.{quote_identifier(column)} = r.{quote_identifier(column)}" for column in key)
        reread = relation(table, "t", [])
        return (
            f"SELECT {listed} FROM ({sql} ORDER BY {distance} DESC{ties} LIMIT ${place + 1}) AS r"
            f" CROSS JOIN LATERAL (SELECT * FROM {reread} WHERE {again}) AS t ORDER BY r.{rank} DESC{ties}",
            [*params, nearest.text, limit],
        )
    # WHY the read is wrapped (``SELECT ... FROM (<the read> LIMIT k) AS n ORDER BY distance``), measured 2026-11-14:
    # every vector reader pins an owner (agent_id + user_id). At a heavy owner the planner walks the HNSW index and then
    # filters it to the owner; pgvector's HNSW scan stops at ``hnsw.ef_search`` (40) candidates, so a small owner among a
    # heavy table got FEWER than ``limit`` rows (PG 16 + pgvector 0.8.7, a 20,000-row owner beside a 50-row one, k=10:
    # 0 of 10 by default). Every platform pool therefore starts with ``hnsw.iterative_scan = relaxed_order``
    # (``threetears.core.utils.pg_pool_kwargs._PLATFORM_SERVER_SETTINGS``; 10 of 10 in 13 ms with it), which keeps
    # scanning until ``limit`` rows pass the filter but may hand them back slightly out of distance order; pgvector's
    # advice for relaxed order is to materialize the ``limit`` rows, then sort them by distance. pgvector writes that as
    # ``WITH ... AS MATERIALIZED``; a subquery under its own LIMIT is the same fence (the planner never pulls a LIMIT
    # subquery up), and names no relation of its own: the hub broker's gates read every relation a statement names, and
    # its agent-data gate would refuse a CTE's name as an undeclared table. YugabyteDB's
    # ybhnsw applies the owner pin inside its index scan (a Storage Filter) and returned 10 of 10 without the setting; it
    # refuses ``SET hnsw.*`` (a reserved prefix), so no statement here ever sets it. The re-sort costs a sort of ``limit``
    # rows. Probes: scratchpad a100/hnsw/probe.py, probe_startup.py; tests:
    # packages/core/tests/integration/test_nearest_heavy_owner_live.py.
    similarity = [] if nearest.similarity is None else [f"1 - ({distance}) AS {quote_identifier(nearest.similarity)}"]
    hidden = quote_identifier(_NEAREST_DISTANCE)
    if columns is None:
        # every column: the row travels whole through the wrapper and is spread again outside it
        row = quote_identifier(_NEAREST_ROW)
        inner: Sequence[str | Selected] | None = (Selected("t", _NEAREST_ROW),)
        outer = [f"(n.{row}).*"]
    else:
        inner = columns
        outer = [f"n.{quote_identifier(c.alias if isinstance(c, Selected) else c)}" for c in columns]
    if nearest.similarity is not None:
        outer.append(f"n.{quote_identifier(nearest.similarity)}")
    sql, params = _sql(
        table, by, inner, limit=False, where=where, also=", ".join([*similarity, f"{distance} AS {hidden}"])
    )
    return (
        f"SELECT {', '.join(outer)} FROM ({sql} ORDER BY {distance} LIMIT ${place + 1}) AS n ORDER BY n.{hidden}",
        [*params, nearest.vector, limit],
    )
