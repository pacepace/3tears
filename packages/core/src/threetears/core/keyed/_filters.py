"""the one spelling of a keyed read's filters, shared by every helper in :mod:`threetears.core.keyed`.

The filters, the names and the vocabulary they are written in are pure, and live in the dependency-free
:mod:`threetears.interfaces.keyed.filters` (re-exported here under the names every helper imports); what stays here
is the one question a filter asks of the connection it is sent on: whether it binds typed arrays natively.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

import asyncpg
from asyncpg.pool import PoolConnectionProxy
from threetears.core.backends.dialect import known_on_yugabyte
from threetears.core.keyed.inline import InlineDuckDB
from threetears.core.keyed.threaded import InlineSQLite, ThreadedDuckDB, ThreadedSQLite
from threetears.interfaces.keyed.counts import SQL_DIALECT_ATTRIBUTE, Dialect
from threetears.interfaces.keyed.keys import index_hint, not_null_guards, serving_index
from threetears.interfaces.keyed.page import Sort
from threetears.interfaces.keyed.filters import (
    IN_SUBQUERY,
    AnyOf,
    Const,
    Derived,
    Fragment,
    Lock,
    Pin,
    Selected,
    batched_join_hint,
    by_conditions,
    clause_guards,
    each_placeholder,
    narrowing,
    one_use_each,
    prefix_conditions,
    quoted_table,
    relation,
    renumbered,
    selected_column,
)

__all__ = [
    "IN_SUBQUERY",
    "AnyOf",
    "Const",
    "Derived",
    "Fragment",
    "Lock",
    "Pin",
    "Selected",
    "batched_join_hint",
    "binds_native_arrays",
    "by_conditions",
    "clause_guards",
    "each_placeholder",
    "prefix_conditions",
    "quoted_table",
    "reads_l3",
    "selected_column",
    "relation",
    "narrowing",
    "one_use_each",
    "renumbered",
    "sent_by",
    "yugabyte_index_hint",
]


def binds_native_arrays(conn: Any) -> bool:
    """whether ``conn`` binds a typed array natively (``$1::bigint[]`` from a list of ints): asyncpg's own
    connection, pool or pooled connection, or anything saying so (``binds_native_arrays = True``: a wrapper of one).
    The L3 rail carries its parameters as JSON, so a keyed read through it binds text arrays cast to the column's
    type (``$1::text[]::bigint[]``); identity, not truthiness, so a mock's auto-attribute never reads as yes.

    :param conn: the connection a keyed helper reads through
    :ptype conn: Any
    :return: whether its arrays bind natively
    :rtype: bool
    """
    if isinstance(conn, asyncpg.Connection | asyncpg.Pool | PoolConnectionProxy):
        return True
    return getattr(conn, "binds_native_arrays", None) is True


#: the dialects whose drivers take a placeholder named more than once (a reader naming one is sent as written)
_TAKES_REUSE: Final[frozenset[Dialect]] = frozenset({Dialect.POSTGRES, Dialect.YUGABYTE, Dialect.DUCKDB})


class _OneUseEach:
    """a reader naming its engine's dialect, sent every statement with one placeholder a use (:func:`sent_by`)."""

    def __init__(self, reader: Any) -> None:
        self._reader = reader

    def __getattr__(self, name: str) -> Any:
        return getattr(self._reader, name)

    async def fetch(self, sql: str, *params: Any, **options: Any) -> Any:
        """the reader's ``fetch`` of ``sql`` sent one placeholder a use.

        :param sql: the statement
        :ptype sql: str
        :param params: its parameters
        :ptype params: Any
        :param options: the reader's own options, passed on
        :ptype options: Any
        :return: what the reader answers
        :rtype: Any
        """
        statement, sent = one_use_each(sql, params)
        return await self._reader.fetch(statement, *sent, **options)

    async def fetchrow(self, sql: str, *params: Any, **options: Any) -> Any:
        """the reader's ``fetchrow`` of ``sql`` sent one placeholder a use.

        :param sql: the statement
        :ptype sql: str
        :param params: its parameters
        :ptype params: Any
        :param options: the reader's own options, passed on
        :ptype options: Any
        :return: what the reader answers
        :rtype: Any
        """
        statement, sent = one_use_each(sql, params)
        return await self._reader.fetchrow(statement, *sent, **options)

    async def fetchval(self, sql: str, *params: Any, **options: Any) -> Any:
        """the reader's ``fetchval`` of ``sql`` sent one placeholder a use.

        :param sql: the statement
        :ptype sql: str
        :param params: its parameters
        :ptype params: Any
        :param options: the reader's own options, passed on
        :ptype options: Any
        :return: what the reader answers
        :rtype: Any
        """
        statement, sent = one_use_each(sql, params)
        return await self._reader.fetchval(statement, *sent, **options)

    async def fetch_arrow(self, sql: str, *params: Any, **options: Any) -> Any:
        """the reader's ``fetch_arrow`` of ``sql`` sent one placeholder a use.

        :param sql: the statement
        :ptype sql: str
        :param params: its parameters
        :ptype params: Any
        :param options: the reader's own options, passed on
        :ptype options: Any
        :return: what the reader answers
        :rtype: Any
        """
        statement, sent = one_use_each(sql, params)
        return await self._reader.fetch_arrow(statement, *sent, **options)


def sent_by(conn: Any) -> Any:
    """what a keyed read sends its statement through: ``conn`` itself, or, for a reader naming its engine's dialect
    (a warehouse's datasource reader, any but PostgreSQL, YugabyteDB and DuckDB: Redshift binds each placeholder once, and refuses a statement naming ``$1`` twice where
    PostgreSQL takes it), ``conn`` sending each statement :func:`one_use_each`, whatever builder wrote it.

    :param conn: the connection or reader the read was handed
    :ptype conn: Any
    :return: what to send the statement through
    :rtype: Any
    """
    named = getattr(conn, SQL_DIALECT_ATTRIBUTE, None)
    return _OneUseEach(conn) if isinstance(named, Dialect) and named not in _TAKES_REUSE else conn


def reads_l3(conn: Any) -> bool:
    """whether a read on ``conn`` goes to L3, where the declared tables (their keys, their indexes) are: an in-process
    DuckDB or SQLite does not, nor does a reader naming its dialect (a datasource's, a warehouse's, even one spelled as
    PostgreSQL); an asyncpg connection, a pool, the rail's proxy do.

    :param conn: the connection or reader the read was handed
    :ptype conn: Any
    :return: whether it reads L3
    :rtype: bool
    """
    if isinstance(conn, ThreadedDuckDB | ThreadedSQLite | InlineSQLite | InlineDuckDB):
        return False
    # looked up on the class (a reader's property, its class attribute) and the instance's own attributes: a mock
    # answers any name it is asked, and is no datasource
    named = getattr(type(conn), SQL_DIALECT_ATTRIBUTE, None)
    return named is None and getattr(conn, "__dict__", {}).get(SQL_DIALECT_ATTRIBUTE) is None


async def yugabyte_index_hint(
    conn: Any,
    table: Any,
    by: Mapping[str, Any],
    order: Sequence[Sort],
    *,
    alias: str = "t",
    where: Sequence[Fragment] = (),
) -> str:
    """the hint a keyed read of ``table`` pinning ``by`` in ``order`` starts with on YugabyteDB: the declared index
    that serves it (:func:`~threetears.interfaces.keyed.keys.serving_index`), named to pg_hint_plan
    (:func:`~threetears.interfaces.keyed.keys.index_hint`). Nothing on any other database, for a read that does not go
    to L3, or where no declared index but the primary key serves it.

    The one place a keyed helper asks it: YugabyteDB read one agent's conversations of one status through the primary
    key filtering the status, on every one of 15 fresh worlds, rather than the declared ``(agent_id, status,
    conversation_id)`` (3tears #87, #88); statistics did not move it.

    :param conn: the reader
    :ptype conn: Any
    :param table: the relation read
    :ptype table: Any
    :param by: the read's pins
    :ptype by: Mapping[str, Any]
    :param order: the read's sort
    :ptype order: Sequence[Sort]
    :param alias: the alias the statement reads the table by (a keyed count reads it as ``r``)
    :ptype alias: str
    :param where: the read's guards beside its pins: each ``<col> IS NOT NULL`` of them states that guard to a
        ``col IS NOT NULL`` partial index (:func:`~threetears.interfaces.keyed.keys.not_null_guards`; WHY at
        :func:`~threetears.interfaces.keyed.keys.partial_predicate`)
    :ptype where: Sequence[Fragment]
    :return: the hint and a space, or the empty string
    :rtype: str
    """
    index = serving_index(table, by, order, guarded=not_null_guards(fragment.sql for fragment in where))
    if index is None or index.primary or not reads_l3(conn):
        return ""
    return index_hint(index, alias) if await known_on_yugabyte(conn) else ""
