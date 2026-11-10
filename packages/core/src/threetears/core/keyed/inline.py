"""a DuckDB connection as the asyncpg-shaped reader the keyed helpers take, each statement run inline, on it.

:class:`~threetears.core.keyed.ThreadedDuckDB` runs each statement on a cursor of its own in a thread, for an event
loop that must not wait. A cursor is a connection of its own: it sees no transaction the caller opened and no
relation the caller registered (``conn.register("chunk", arrow)``). Synchronous code that owns its connection (a
worker process applying a load inside its transaction) needs the helpers' statements on THAT connection, in that
transaction, seeing those relations. :class:`InlineDuckDB` runs each on the connection itself, at once, and
:func:`run_inline` drives the helper's coroutine to its answer without an event loop: the helper never suspends on
it, so one ``send`` runs it whole.

.. code-block:: python

    db = InlineDuckDB(conn)  # conn is the caller's, inside its own BEGIN ... COMMIT
    conn.register("chunk", arrow)
    run_inline(keyed.apply_stage(db, "results", "chunk", key=KEY, columns=COLUMNS, mode="difference", by=scope))
"""

from __future__ import annotations

import inspect
from collections.abc import Coroutine, Iterable, Sequence
from types import TracebackType
from typing import Any, TypeVar

__all__ = ["InlineDuckDB", "InlineTransaction", "run_inline"]

_T = TypeVar("_T")


def _rows(result: Any) -> list[dict[str, Any]]:
    names = [column[0] for column in result.description or ()]
    return [dict(zip(names, row, strict=True)) for row in result.fetchall()]


def _status(result: Any, query: str) -> str:
    verb = query.lstrip().split(None, 1)[0].upper()
    answered = result.fetchall() if result.description else []
    touched = int(answered[0][0]) if answered and verb in {"INSERT", "UPDATE", "DELETE"} else 0
    return f"INSERT 0 {touched}" if verb == "INSERT" else f"{verb} {touched}"


class InlineTransaction:
    """a transaction on an :class:`InlineDuckDB`'s connection, shaped as asyncpg's (``start`` / ``commit`` /
    ``rollback``, or ``async with``: committed on a clean exit, rolled back on an error).

    :param conn: the ``duckdb.DuckDBPyConnection``
    :ptype conn: duckdb.DuckDBPyConnection
    """

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    async def start(self) -> None:
        """``BEGIN TRANSACTION``."""
        self._conn.execute("BEGIN TRANSACTION")

    async def commit(self) -> None:
        """``COMMIT``."""
        self._conn.execute("COMMIT")

    async def rollback(self) -> None:
        """``ROLLBACK``."""
        self._conn.execute("ROLLBACK")

    async def __aenter__(self) -> InlineTransaction:
        await self.start()
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        if exc_type is None:
            await self.commit()
        else:
            await self.rollback()


class InlineDuckDB:
    """a DuckDB connection read and written through ``fetch`` / ``execute`` / ``executemany`` / ``transaction``,
    each statement on the connection itself, at once: in the caller's transaction, seeing its registered relations.

    For synchronous code (a worker process) through :func:`run_inline`; never awaited on an event loop that must not
    wait (that is :class:`~threetears.core.keyed.ThreadedDuckDB`).

    :param conn: a ``duckdb.DuckDBPyConnection``, the caller's own
    :ptype conn: duckdb.DuckDBPyConnection
    """

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    async def fetch(self, query: str, /, *params: Any) -> list[dict[str, Any]]:
        """the statement's rows, as dicts.

        :param query: the statement, ``$n`` placeholders
        :ptype query: str
        :param params: its parameters
        :ptype params: Any
        :return: the rows
        :rtype: list[dict[str, Any]]
        """
        return _rows(self._conn.execute(query, list(params)))

    async def execute(self, query: str, /, *params: Any) -> str:
        """run a write and answer its status as asyncpg spells it (``DELETE 3``).

        :param query: the statement, ``$n`` placeholders
        :ptype query: str
        :param params: its parameters
        :ptype params: Any
        :return: the status
        :rtype: str
        """
        return _status(self._conn.execute(query, list(params)), query)

    async def executemany(self, query: str, args: Iterable[Sequence[Any]], /) -> None:
        """run one statement once for each parameter set, as asyncpg's ``executemany``.

        :param query: the statement, ``$n`` placeholders
        :ptype query: str
        :param args: a parameter set a run
        :ptype args: Iterable[Sequence[Any]]
        :return: nothing
        :rtype: None
        """
        sets = [list(one) for one in args]
        if sets:
            self._conn.executemany(query, sets)

    def transaction(self) -> InlineTransaction:
        """a transaction on the connection, asyncpg-shaped.

        :return: the transaction, not yet started
        :rtype: InlineTransaction
        """
        return InlineTransaction(self._conn)


def run_inline(awaitable: Coroutine[Any, Any, _T]) -> _T:
    """a coroutine's answer, run to its end here, with no event loop.

    It is sent ``None`` until it returns; a bare yield (``asyncio.sleep(0)``) is stepped past. A coroutine that
    awaits real IO (a future, a task, a socket) cannot finish here: it is closed and refused, never left half run.

    :param awaitable: the coroutine, a helper's call on an :class:`InlineDuckDB` (or anything that never suspends)
    :ptype awaitable: Coroutine[Any, Any, _T]
    :return: its answer
    :rtype: _T
    :raises TypeError: when ``awaitable`` is not a coroutine
    :raises RuntimeError: when it awaits something that would suspend it
    """
    if not inspect.iscoroutine(awaitable):
        raise TypeError(f"run_inline drives a coroutine, got {type(awaitable).__name__}")
    try:
        while True:
            yielded = awaitable.send(None)
            if yielded is not None:
                raise RuntimeError(f"run_inline drives only a coroutine that never waits on IO; it awaited {yielded!r}")
    except StopIteration as done:
        answer: _T = done.value
        return answer
    finally:
        awaitable.close()
