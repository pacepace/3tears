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
    selected_column,
)
from threetears.interfaces.keyed.page import Sort, order_by_terms

__all__ = ["Nearest", "read_by_statement", "read_nearest_statement", "read_row_statement"]


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
    """the statement and its parameters.

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
    :return: the statement, and its parameters
    :rtype: tuple[str, list[Any]]
    :raises ValueError: when nothing is pinned, or ``columns`` is empty
    """
    if not by:
        raise ValueError(f"a point read of {table} pins at least one column; with none it reads the table whole")
    if columns is not None and not columns:
        raise ValueError(f"a point read of {table} names the columns it reads, or None for every column")
    selected = "t.*" if columns is None else ", ".join(selected_column(column) for column in columns)
    if also:
        selected = f"{selected}, {also}"
    conditions, params = by_conditions(by, alias="t")
    for fragment in where:
        conditions.append(narrowing(fragment, len(params)))
        params += list(fragment.params)
    tail = (
        (f" ORDER BY {order_by_terms(order)}" if order else "")
        + (" LIMIT 1" if limit else "")
        + ("" if lock is None else lock.of("t"))
    )
    return f"SELECT {selected} FROM {relation(table, 't')} WHERE {' AND '.join(conditions)}{tail}", params


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
    :return: the statement, and its binds
    :rtype: tuple[str, list[Any]]
    :raises ValueError: when nothing is pinned, or ``columns`` is empty
    """
    return _sql(table, by, columns, limit=True, lock=lock, where=where)


#: pgvector's distance operators: L2, cosine, inner product (negated), L1
_DISTANCES: Final[frozenset[str]] = frozenset({"<->", "<=>", "<#>", "<+>"})


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


def read_nearest_statement(
    table: str | Derived,
    *,
    by: Mapping[str, Any],
    nearest: Nearest,
    limit: int,
    columns: Sequence[str | Selected] | None = None,
    where: Sequence[Fragment] = (),
) -> tuple[str, list[Any]]:
    """the statement reading the ``limit`` rows of ``table`` the pinned columns name nearest to a vector, nearest
    first, and its binds: what ``threetears.core.keyed.read_nearest`` sends::

        SELECT t."id", 1 - (t."embedding" OPERATOR(public.<=>) $3::text::public.vector) AS "similarity"
        FROM "memories" AS t WHERE t."agent_id" = $1 AND t."user_id" = $2
        ORDER BY t."embedding" OPERATOR(public.<=>) $3::text::public.vector LIMIT $4

    The order is the distance alone (no tie-break column), so an approximate-nearest-neighbour index on the column
    answers it; rows at the same distance come in the engine's order, as they did from the statement written by hand.

    :param table: the table, a TRUSTED identifier (``schema.table`` quoted a segment at a time)
    :ptype table: str | Derived
    :param by: as :func:`read_by_statement` takes it; at least one column
    :ptype by: Mapping[str, Any]
    :param nearest: the vector column, the vector, and the distance
    :ptype nearest: Nearest
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
    place = 1 + len(by_conditions(by, alias="t")[1]) + sum(len(fragment.params) for fragment in where)
    distance = nearest.distance(place)
    also = "" if nearest.similarity is None else f"1 - ({distance}) AS {quote_identifier(nearest.similarity)}"
    sql, params = _sql(table, by, columns, limit=False, where=where, also=also)
    return f"{sql} ORDER BY {distance} LIMIT ${place + 1}", [*params, nearest.vector, limit]
