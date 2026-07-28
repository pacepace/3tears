"""backend shims shared by the dsd-task-01 driver transaction / timeout tests.

two mocked backends, one per concrete driver, built here rather than
inline in each test module so the transaction tests and the
statement-timeout isolation tests observe the SAME recorded surface:

- ``sql_log`` -- every statement the driver issued, setup statements
  included. the statement-timeout tests read this to prove which
  ``SET`` variants reached the session and in what order
- ``statement_log`` -- only the caller's statements (open-time
  ``SET`` / ``SELECT pg_backend_pid()`` filtered out), each recorded
  with the connection object it ran on. the transaction tests read
  this to prove a multi-statement transaction pinned one session

deliberately NOT in the public API: only the test suite imports it.
"""

from __future__ import annotations

from types import TracebackType
from typing import Any
from unittest.mock import AsyncMock, MagicMock

__all__ = [
    "build_mock_redshift_connection",
    "build_transaction_capable_pool",
    "is_open_setup_stmt",
    "log_after_first_caller_statement",
]


def log_after_first_caller_statement(sql_log: list[str]) -> list[str]:
    """return the tail of ``sql_log`` starting at the first caller statement.

    the driver commits at connection open (so its session settings
    survive the release-path rollback), which means a raw
    ``conn.commit.call_count`` mixes setup with the caller's
    disposition. slicing from the first caller statement is what makes
    "the transaction committed exactly once" a statement about the
    transaction rather than about connection setup.

    :param sql_log: full statement log from the mocked backend
    :ptype sql_log: list[str]
    :return: log entries from the first caller statement onwards
    :rtype: list[str]
    """
    tail: list[str] = []
    for index, entry in enumerate(sql_log):
        if entry not in {"COMMIT", "ROLLBACK"} and not is_open_setup_stmt(entry):
            tail = sql_log[index:]
            break
    return tail


def is_open_setup_stmt(sql: str) -> bool:
    """classify a statement as driver session setup rather than a caller statement.

    the driver issues ``SET statement_timeout``, ``SET LOCAL
    statement_timeout``, ``SET search_path`` and ``SELECT
    pg_backend_pid()`` on its own behalf. tests that count caller
    statements filter these out.

    :param sql: SQL text handed to ``cursor.execute``
    :ptype sql: str
    :return: True iff the statement is driver session setup
    :rtype: bool
    """
    return sql.startswith("SET ") or sql.startswith("SELECT pg_backend_pid")


def build_mock_redshift_connection(
    *,
    fetchall_rows: list[tuple[Any, ...]] | None = None,
    description: list[tuple[str, Any]] | None = None,
    backend_pid: int = 4242,
) -> MagicMock:
    """build a mock behaving like a ``redshift_connector.Connection``.

    the cursor records every statement into ``conn.sql_log`` and every
    caller statement into ``conn.statement_log`` as
    ``{"conn": conn, "sql": sql}`` dicts. ``conn.fail_caller_statement_at``
    arms an exception on the Nth caller statement.

    :param fetchall_rows: rows ``cursor.fetchall()`` returns
    :ptype fetchall_rows: list[tuple[Any, ...]] | None
    :param description: ``cursor.description`` value
    :ptype description: list[tuple[str, Any]] | None
    :param backend_pid: value ``SELECT pg_backend_pid()`` resolves to
    :ptype backend_pid: int
    :return: connection mock with cursor / commit / rollback / close wired
    :rtype: MagicMock
    """
    conn = MagicMock(name="MockRedshiftConn")
    cursor = MagicMock(name="MockRedshiftCursor")
    cursor.description = description or []
    cursor.fetchall = MagicMock(return_value=fetchall_rows or [])
    cursor.fetchone = MagicMock(return_value=(backend_pid,))
    cursor.fetchmany = MagicMock(return_value=[])
    cursor.close = MagicMock(return_value=None)

    sql_log: list[str] = []
    statement_log: list[dict[str, Any]] = []
    armed: dict[str, Any] = {"index": None, "error": None, "seen": 0}

    def _execute(*args: Any, **kwargs: Any) -> None:
        """record the statement, then raise when armed for this position."""
        sql = args[0] if args else ""
        sql_log.append(sql)
        if is_open_setup_stmt(sql):
            return
        armed["seen"] = int(armed["seen"]) + 1
        statement_log.append({"conn": conn, "sql": sql})
        if armed["index"] is not None and armed["seen"] == armed["index"]:
            raise armed["error"]

    cursor.execute = MagicMock(side_effect=_execute)

    def _fail_caller_statement_at(index: int, error: Exception) -> None:
        """arm ``error`` on the ``index``-th caller statement (1-based)."""
        armed["index"] = index
        armed["error"] = error

    # commit / rollback land in the SAME log as the statements so tests
    # can reason about ORDER -- "did the transaction close before the
    # connection went back in the cache" is an ordering question.
    conn.cursor = MagicMock(return_value=cursor)
    conn.commit = MagicMock(side_effect=lambda: sql_log.append("COMMIT"))
    conn.rollback = MagicMock(side_effect=lambda: sql_log.append("ROLLBACK"))
    conn.close = MagicMock(return_value=None)
    conn.sql_log = sql_log
    conn.statement_log = statement_log
    conn.fail_caller_statement_at = _fail_caller_statement_at
    # surface the cursor so tests can assert against it directly
    conn.recorded_cursor = cursor
    return conn


class _TransactionHandle:
    """stand-in for :class:`asyncpg.transaction.Transaction`.

    supports BOTH the explicit ``start`` / ``commit`` / ``rollback``
    surface the driver's ``begin`` uses and the async-context-manager
    surface the timeout-override wrapper uses, because the real
    asyncpg type supports both and a shim that supports only one hides
    a divergence.
    """

    def __init__(self) -> None:
        """wire the three await-counting hooks.

        :return: nothing
        :rtype: None
        """
        self.start = AsyncMock(name="tx.start")
        self.commit = AsyncMock(name="tx.commit")
        self.rollback = AsyncMock(name="tx.rollback")

    async def __aenter__(self) -> "_TransactionHandle":
        """start the transaction on context entry.

        :return: self
        :rtype: _TransactionHandle
        """
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """commit on clean exit, roll back on exception.

        :param exc_type: exception class raised in the body, or None
        :ptype exc_type: type[BaseException] | None
        :param exc: exception instance raised in the body, or None
        :ptype exc: BaseException | None
        :param traceback: traceback of the raised exception, or None
        :ptype traceback: TracebackType | None
        :return: False -- never suppress the caller's exception
        :rtype: bool
        """
        if exc_type is None:
            await self.commit()
        else:
            await self.rollback()
        return False


class _PoolAcquireHandle:
    """stand-in for :class:`asyncpg.pool.PoolAcquireContext`.

    awaitable (``conn = await pool.acquire()``) AND an async context
    manager (``async with pool.acquire() as conn``), matching asyncpg.
    the driver uses the awaitable form to pin a session for a
    transaction and the context-manager form for single statements.
    """

    def __init__(self, pool: MagicMock, conn: MagicMock) -> None:
        """capture the pool and the connection it hands out.

        :param pool: owning pool mock
        :ptype pool: MagicMock
        :param conn: connection mock to yield
        :ptype conn: MagicMock
        :return: nothing
        :rtype: None
        """
        self._pool = pool
        self._conn = conn

    def __await__(self) -> Any:
        """yield the connection for the awaitable acquire form.

        :return: generator resolving to the connection mock
        :rtype: Any
        """

        async def _resolve() -> MagicMock:
            return self._conn

        return _resolve().__await__()

    async def __aenter__(self) -> MagicMock:
        """yield the connection for the context-manager acquire form.

        :return: connection mock
        :rtype: MagicMock
        """
        return self._conn

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """release the connection back to the pool.

        :param exc_type: exception class raised in the body, or None
        :ptype exc_type: type[BaseException] | None
        :param exc: exception instance raised in the body, or None
        :ptype exc: BaseException | None
        :param traceback: traceback of the raised exception, or None
        :ptype traceback: TracebackType | None
        :return: False -- never suppress the caller's exception
        :rtype: bool
        """
        await self._pool.release(self._conn)
        return False


def build_transaction_capable_pool(
    *,
    fetch_records: list[dict[str, Any]] | None = None,
) -> MagicMock:
    """build a mock behaving like an ``asyncpg.Pool`` with transaction support.

    the connection records every statement into ``pool.sql_log`` and
    every caller statement into ``pool.statement_log`` as
    ``{"conn": conn, "sql": sql}`` dicts. ``pool.fail_caller_statement_at``
    arms an exception on the Nth caller statement.
    ``pool.transaction_handle`` is the single shared
    :class:`_TransactionHandle` whose await counts the tests assert on.

    :param fetch_records: rows ``conn.fetch`` resolves to
    :ptype fetch_records: list[dict[str, Any]] | None
    :return: pool mock with acquire / release / close wired
    :rtype: MagicMock
    """
    records = fetch_records or []
    pool = MagicMock(name="MockPool")
    conn = MagicMock(name="MockConn")

    sql_log: list[str] = []
    statement_log: list[dict[str, Any]] = []
    armed: dict[str, Any] = {"index": None, "error": None, "seen": 0}

    def _record(sql: str) -> None:
        """record a statement and raise when armed for this position."""
        sql_log.append(sql)
        if sql.startswith("SET ") or sql.startswith("RESET "):
            return
        armed["seen"] = int(armed["seen"]) + 1
        statement_log.append({"conn": conn, "sql": sql})
        if armed["index"] is not None and armed["seen"] == armed["index"]:
            raise armed["error"]

    async def _execute(sql: str, *params: Any, **kwargs: Any) -> None:
        """mocked ``Connection.execute``."""
        _record(sql)

    async def _fetch(sql: str, *params: Any, **kwargs: Any) -> list[dict[str, Any]]:
        """mocked ``Connection.fetch``."""
        _record(sql)
        return records

    async def _fetchval(sql: str, *params: Any, **kwargs: Any) -> Any:
        """mocked ``Connection.fetchval``."""
        _record(sql)
        return 1

    transaction_handle = _TransactionHandle()

    conn.execute = AsyncMock(side_effect=_execute)
    conn.fetch = AsyncMock(side_effect=_fetch)
    conn.fetchval = AsyncMock(side_effect=_fetchval)
    conn.cancel = MagicMock(return_value=None)
    conn.terminate = MagicMock(return_value=None)
    conn.transaction = MagicMock(return_value=transaction_handle)

    def _fail_caller_statement_at(index: int, error: Exception) -> None:
        """arm ``error`` on the ``index``-th caller statement (1-based)."""
        armed["index"] = index
        armed["error"] = error

    pool.acquire = MagicMock(side_effect=lambda *a, **k: _PoolAcquireHandle(pool, conn))
    pool.release = AsyncMock(return_value=None)
    pool.close = AsyncMock(return_value=None)
    pool.is_closing = MagicMock(return_value=False)
    pool.sql_log = sql_log
    pool.statement_log = statement_log
    pool.fail_caller_statement_at = _fail_caller_statement_at
    pool.transaction_handle = transaction_handle
    pool.recorded_conn = conn
    return pool
