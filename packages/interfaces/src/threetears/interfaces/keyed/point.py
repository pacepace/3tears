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
from typing import Any

from threetears.interfaces.keyed.filters import (
    Derived,
    Fragment,
    Lock,
    Selected,
    by_conditions,
    narrowing,
    relation,
    selected_column,
)
from threetears.interfaces.keyed.page import Sort, order_by_terms

__all__ = ["read_by_statement", "read_row_statement"]


def _sql(
    table: str | Derived,
    by: Mapping[str, Any],
    columns: Sequence[str | Selected] | None,
    limit: bool,
    lock: Lock | None = None,
    where: Sequence[Fragment] = (),
    order: Sequence[Sort] = (),
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
    :return: the statement, and its parameters
    :rtype: tuple[str, list[Any]]
    :raises ValueError: when nothing is pinned, or ``columns`` is empty
    """
    if not by:
        raise ValueError(f"a point read of {table} pins at least one column; with none it reads the table whole")
    if columns is not None and not columns:
        raise ValueError(f"a point read of {table} names the columns it reads, or None for every column")
    selected = "t.*" if columns is None else ", ".join(selected_column(column) for column in columns)
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
