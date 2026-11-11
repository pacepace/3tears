"""paged sweeps' statements: the rows past a bound deleted (or updated) a page at a time, through the bound's index.

A sweep of expired rows (leases, tokens, dedup entries) deletes the rows whose ``expires_at`` passed. Written as one
``DELETE ... WHERE expires_at < $1`` it reads every row of the table; written as a page it reads only the page::

    DELETE FROM <table> WHERE ("k1", "k2") IN (
        SELECT "k1", "k2" FROM <table> WHERE <by> AND "expires_at" < $n ORDER BY "expires_at" LIMIT <page>)

and the sweep repeats it until a page deletes fewer rows than it asked for (the rows its ``RETURNING`` answers). The
index it reads is ``(<by columns>, <column>)``.

Pure: :func:`delete_before_statement` and :func:`update_before_statement` write one page's statement and its binds,
the same every page; ``threetears.core.keyed.delete_before`` / ``update_before`` run them (each page settled in its
collection's caches), and a store with no 3tears core runs them on its own connection, page after page.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from threetears.interfaces.keyed.filters import (
    IN_SUBQUERY,
    batched_join_hint,
    by_conditions,
    quote_identifier,
    quoted_table,
)

__all__ = ["SweptRows", "delete_before_statement", "update_before_statement"]


@dataclass(frozen=True)
class SweptRows:
    """what a paged sweep deleted (or a paged update touched).

    :param deleted: the rows deleted, or updated
    :ptype deleted: int
    :param statements: the DELETEs sent, one a page
    :ptype statements: int
    """

    deleted: int
    statements: int


def _paged(
    table: str, key: Sequence[str], column: str, conditions: Sequence[str], page: int, assignments: Sequence[str] = ()
) -> str:
    """one page of a paged write: the rows of ``table`` matching ``conditions``, ``page`` of them in ``column``'s
    order, deleted, or set to ``assignments`` when there are any; each returning its key."""
    named = ", ".join(quote_identifier(part) for part in key)
    written = ", ".join(f"t.{quote_identifier(part)}" for part in key)
    quoted = quoted_table(table)
    # the page's keys joined back to the table by its key, batched on YugabyteDB (batched_join_hint): one storage
    # request a page, not one a row
    hint = batched_join_hint(IN_SUBQUERY, "t")
    rows_of = (
        f"({written}) IN (SELECT {named} FROM {quoted} WHERE {' AND '.join(conditions)} "
        f"ORDER BY {quote_identifier(column)} LIMIT {page}) RETURNING {written}"
    )
    if assignments:
        return f"{hint}UPDATE {quoted} AS t SET {', '.join(assignments)} WHERE {rows_of}"
    return f"{hint}DELETE FROM {quoted} AS t WHERE {rows_of}"


def delete_before_statement(
    table: str,
    *,
    key: Sequence[str],
    column: str,
    before: Any,
    page: int,
    by: Mapping[str, Any] | None = None,
) -> tuple[str, list[Any]]:
    """one page's statement deleting the rows of ``table`` (within ``by``) whose ``column`` is before ``before``, and
    its binds: sent page after page until one returns fewer than ``page`` keys.

    :param table: the table, a TRUSTED identifier (``schema.table`` quoted a segment at a time)
    :ptype table: str
    :param key: its primary key's columns, TRUSTED identifiers: the rows a page deletes are named, and returned, by them
    :ptype key: Sequence[str]
    :param column: the column compared with ``before`` (``expires_at``), led by ``by`` in its index
    :ptype column: str
    :param before: the bound: rows whose ``column`` is less are deleted
    :ptype before: Any
    :param page: the most rows a statement deletes
    :ptype page: int
    :param by: columns pinned ahead of ``column`` (a purpose, a tenant)
    :ptype by: Mapping[str, Any] | None
    :return: the statement, and its binds
    :rtype: tuple[str, list[Any]]
    :raises ValueError: when ``key`` is empty or ``page`` is under one
    """
    if not key:
        raise ValueError(f"a sweep of {table} names its key")
    if page < 1:
        raise ValueError(f"a sweep of {table} deletes at least one row in at least one page")
    conditions, params = by_conditions(by or {})
    conditions.append(f"{quote_identifier(column)} < ${len(params) + 1}")
    params.append(before)
    return _paged(table, key, column, conditions, page), params


def update_before_statement(
    table: str,
    *,
    key: Sequence[str],
    column: str,
    before: Any,
    set_to: Mapping[str, Any],
    page: int,
    by: Mapping[str, Any] | None = None,
) -> tuple[str, list[Any]]:
    """one page's statement setting ``set_to`` on the rows of ``table`` (within ``by``) whose ``column`` is before
    ``before``, and its binds. Each update must move its rows out of the pages it reads (``set_to`` names a column of
    ``by``, or ``column`` itself), or the next page would read them again: that is refused.

    :param table: the table, a TRUSTED identifier
    :ptype table: str
    :param key: its primary key's columns, TRUSTED identifiers
    :ptype key: Sequence[str]
    :param column: the column compared with ``before``, led by ``by`` in its index
    :ptype column: str
    :param before: the bound
    :ptype before: Any
    :param set_to: column -> its new value, each bound
    :ptype set_to: Mapping[str, Any]
    :param page: the most rows a statement updates
    :ptype page: int
    :param by: columns pinned ahead of ``column`` (a status, a tenant)
    :ptype by: Mapping[str, Any] | None
    :return: the statement, and its binds
    :rtype: tuple[str, list[Any]]
    :raises ValueError: when ``key`` or ``set_to`` is empty, ``page`` is under one, or the update leaves its rows in
        the pages it reads
    """
    pinned = dict(by or {})
    if not key or not set_to:
        raise ValueError(f"a paged update of {table} names its key and what it sets")
    if page < 1:
        raise ValueError(f"a paged update of {table} touches at least one row in at least one page")
    if not set(set_to) & (set(pinned) | {column}):
        raise ValueError(
            f"a paged update of {table} must move its rows out of the pages it reads: set a column of by, or {column}"
        )
    assignments = []
    params: list[Any] = []
    for name, value in set_to.items():
        params.append(value)
        assignments.append(f"{quote_identifier(name)} = ${len(params)}")
    conditions, values = by_conditions(pinned, first=len(params) + 1)
    params += values
    conditions.append(f"{quote_identifier(column)} < ${len(params) + 1}")
    params.append(before)
    return _paged(table, key, column, conditions, page, tuple(assignments)), params
