"""a read's bounds on L3: the settings a newest-per-key read of a versioned table runs under (``work_mem``, and on
YugabyteDB ``yb_fetch_row_limit``: :func:`~threetears.interfaces.keyed.keys.newest_settings`), held for the read's
own transaction alone (``SET LOCAL``), on every path the read is sent by.

**One context, every transport.** A newest read enters :func:`bounded_read` around its statement; whatever sends it
reads :func:`read_bounds`:

- a direct asyncpg connection or pool (``SqlL3Backend`` over one, its bound request connection, or the connection
  itself) runs the statement in a transaction that first sets them (:func:`bounded_call`);
- the L3 rail (``NatsProxyL3Backend``: ``l3.query``, and each ``l3.tx`` statement) names them on the request,
  :data:`READ_SETTINGS_FIELD` (:func:`read_setting_fields`), and the hub's broker sets them as it sets the read's
  ``statement_timeout``: a pod cannot, as the broker refuses ``SET`` and ``set_config`` at the door.

WHY one context and not a wrapper per transport: a read is sent through several layers (a keyed helper, the store,
the rail's proxy), any of which may be the one that talks to the database; each reads the bounds where it sends, the
way every request reads its stamp (``broker_generation.stamp_fields``).

**The values are integers in each setting's base unit** (``work_mem`` in kB, ``yb_fetch_row_limit`` in rows): the
broker bounds them by comparing numbers, never by parsing a unit, and a statement setting them names a number
alone (``SET LOCAL work_mem = 65536``).

**Pods have no scratch volume** (no disk spill, by design): the work a newest read asks is bounded by its rows read
(:data:`~threetears.interfaces.keyed.keys._NEWEST_ROWS_A_READ`), and ``work_mem`` is the memory that bound is sized
against, so its sort never spills.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Final, TypeVar

import asyncpg
from asyncpg.pool import PoolConnectionProxy

__all__ = [
    "READ_SETTINGS_FIELD",
    "answers_directly",
    "bounded_call",
    "bounded_read",
    "read_bounds",
    "read_setting_fields",
]

#: a read's settings: each name and its value in the setting's base unit, in the order set
_ReadSettings = tuple[tuple[str, int], ...]

#: the request field naming the settings a read runs under: ``{"work_mem": 65536, "yb_fetch_row_limit": 10000}``,
#: each value an integer in the setting's base unit. Sent on an ``l3.query`` or ``l3.tx`` statement request only
#: while a :func:`bounded_read` block is open; a request without it runs as every request did before it. The hub
#: imports this name, applies the settings with ``SET LOCAL`` in the read's own transaction, and refuses a request
#: naming any other setting or a value past the bound it holds for each.
READ_SETTINGS_FIELD: Final = "read_settings"

#: the settings a read may name: the bounds of a newest read, and nothing else
_READ_SETTING_NAMES: Final = frozenset({"work_mem", "yb_fetch_row_limit"})

#: the settings the read being sent in this context runs under (:func:`bounded_read`); none outside one
_BOUNDS: ContextVar[_ReadSettings] = ContextVar("threetears_read_bounds", default=())

_T = TypeVar("_T")


@contextmanager
def bounded_read(settings: Sequence[tuple[str, int]]) -> Iterator[None]:
    """send every L3 read made inside this block under ``settings`` (:data:`READ_SETTINGS_FIELD` on the rail,
    ``SET LOCAL`` on a direct connection). Replaces what an enclosing block named; an empty ``settings`` sends the
    reads with none.

    :param settings: each setting's name and its value in the setting's base unit
    :ptype settings: Sequence[tuple[str, int]]
    :return: a context in which the reads run under them
    :rtype: Iterator[None]
    :raises ValueError: for a setting no read names (:data:`_READ_SETTING_NAMES`), or a value no positive integer
    """
    held: _ReadSettings = tuple((name, value) for name, value in settings)
    for name, value in held:
        if name not in _READ_SETTING_NAMES:
            raise ValueError(f"a read names only {sorted(_READ_SETTING_NAMES)}, not {name!r}")
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"a read's {name} is a positive integer in its base unit, got {value!r}")
    token = _BOUNDS.set(held)
    try:
        yield
    finally:
        _BOUNDS.reset(token)


def read_bounds() -> _ReadSettings:
    """the settings the read being sent runs under, as the enclosing :func:`bounded_read` named them; none outside one.

    :return: each setting's name and value
    :rtype: _ReadSettings
    """
    return _BOUNDS.get()


def read_setting_fields() -> dict[str, Any]:
    """the fields an L3 request carries to name the settings its read runs under: ``{read_settings: {name: value}}``
    inside a :func:`bounded_read` block, nothing outside one (the request as it was before the field).

    :return: the fields, a fresh dict the caller merges into its payload
    :rtype: dict[str, Any]
    """
    held = _BOUNDS.get()
    if not held:
        return {}
    return {READ_SETTINGS_FIELD: {name: value for name, value in held}}


def _set_local_statement(settings: Sequence[tuple[str, int]]) -> str:
    """the one simple-protocol statement setting ``settings`` for the transaction it runs in (one round trip).

    :param settings: each setting's name and its integer value (names are :data:`_READ_SETTING_NAMES`')
    :ptype settings: Sequence[tuple[str, int]]
    :return: the statement
    :rtype: str
    :raises ValueError: for a name no read sets
    """
    for name, _ in settings:
        if name not in _READ_SETTING_NAMES:
            raise ValueError(f"a read sets only {sorted(_READ_SETTING_NAMES)}, not {name!r}")
    return "; ".join(f"SET LOCAL {name} = {int(value)}" for name, value in settings)


def answers_directly(conn: Any) -> bool:
    """whether ``conn`` is asyncpg's own connection, pooled connection or pool: one a read sets its bounds on itself.

    :param conn: the handle a read runs on
    :ptype conn: Any
    :return: whether it is
    :rtype: bool
    """
    return isinstance(conn, asyncpg.Connection | PoolConnectionProxy | asyncpg.Pool)


async def bounded_call(conn: Any, settings: Sequence[tuple[str, int]], call: Callable[[Any], Awaitable[_T]]) -> _T:
    """``call`` run on ``conn`` (asyncpg's connection, pooled connection or pool) in a transaction that first sets
    ``settings`` (``SET LOCAL``), so they end with the read: a pool's connection acquired for it, a connection already
    in a transaction (the caller's) set inside that one, to its end.

    :param conn: the connection or pool
    :ptype conn: Any
    :param settings: each setting's name and value
    :ptype settings: Sequence[tuple[str, int]]
    :param call: the read, given the connection it runs on
    :ptype call: Callable[[Any], Awaitable[_T]]
    :return: what the read answers
    :rtype: _T
    """
    statement = _set_local_statement(settings)

    async def on(connection: Any) -> _T:
        if connection.is_in_transaction():
            # the caller's transaction: the bounds hold to its end, the read's own statement among its others
            await connection.execute(statement)
            return await call(connection)
        async with connection.transaction():
            await connection.execute(statement)
            return await call(connection)

    if isinstance(conn, asyncpg.Pool):
        async with conn.acquire() as acquired:
            return await on(acquired)
    return await on(conn)
