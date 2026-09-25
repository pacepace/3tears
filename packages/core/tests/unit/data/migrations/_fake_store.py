"""
fake migration session for migration runner unit tests.

mirrors the execute/query surface used by MigrationRunner without
requiring a real YugabyteDB connection. tracks applied rows in a
simple in-memory list so idempotent-apply semantics can be asserted,
and answers the database-wide DDL lock's statements the way one
PostgreSQL session holding the lock alone would.
"""

from __future__ import annotations

from typing import Any


# parity-with: threetears.core.data.migrations.session.MigrationSession
class FakeDataStore:
    """
    in-memory migration session capturing executed SQL and emulating the
    ``_schema_migrations`` bookkeeping contract used by MigrationRunner.

    :ivar executed: list of (sql, params) tuples for every execute call
    :ivar queried: list of (sql, params) tuples for every query call
    :ivar statements: list of (sql, params) tuples for every call of either
        kind, in the order they were issued
    :ivar migrations_rows: list of applied migration row dicts
    :ivar lock_holds: how many holds of the DDL lock this session has now
    :ivar fail_on: sql substring that, when present, triggers RuntimeError
    """

    def __init__(
        self,
        fail_on: str | None = None,
        schema: str = "public",
        database: str = "appdb",
        lock_held_elsewhere: bool = False,
    ) -> None:
        """
        initialize empty execution log and migration tracker.

        :param fail_on: SQL substring that triggers RuntimeError on match
        :ptype fail_on: str | None
        :param schema: schema name returned by ``current_schema()``
        :ptype schema: str
        :param database: database name returned by ``current_database()``,
            which is what the DDL lock is keyed on
        :ptype database: str
        :param lock_held_elsewhere: answer every ``pg_try_advisory_lock`` with
            false, as while another session holds the DDL lock
        :ptype lock_held_elsewhere: bool
        """
        self.executed: list[tuple[str, tuple[Any, ...]]] = []
        self.queried: list[tuple[str, tuple[Any, ...]]] = []
        self.statements: list[tuple[str, tuple[Any, ...]]] = []
        self.migrations_rows: list[dict[str, Any]] = []
        self.migrations_table_created = False
        self.lock_holds = 0
        self._fail_on = fail_on
        self._schema = schema
        self._database = database
        self._lock_held_elsewhere = lock_held_elsewhere
        self._tables: set[str] = set()
        # monotonically increasing counter stamped as date_applied so
        # history ordering is deterministic in tests.
        self._apply_counter = 0

    async def execute(self, sql: str, *params: Any) -> str:
        """
        record SQL execution and emulate ``_schema_migrations`` side effects.

        :param sql: SQL statement text
        :ptype sql: str
        :param params: positional statement parameters
        :ptype params: Any
        :return: synthetic status string
        :rtype: str
        :raises RuntimeError: if ``sql`` contains the fail_on substring
        """
        self.executed.append((sql, params))
        self.statements.append((sql, params))
        if self._fail_on is not None and self._fail_on in sql:
            msg = f"fake store forced failure on sql matching '{self._fail_on}'"
            raise RuntimeError(msg)
        normalized = " ".join(sql.split()).upper()
        result: str
        if "CREATE TABLE IF NOT EXISTS _SCHEMA_MIGRATIONS" in normalized:
            self.migrations_table_created = True
            result = "CREATE TABLE"
            return result
        if normalized.startswith("INSERT INTO _SCHEMA_MIGRATIONS"):
            self._apply_counter += 1
            self.migrations_rows.append(
                {
                    "version": params[0],
                    "package": params[1],
                    "description": params[2],
                    "date_applied": self._apply_counter,
                }
            )
            result = "INSERT 0 1"
            return result
        if normalized.startswith("DELETE FROM _SCHEMA_MIGRATIONS"):
            target_version = params[0]
            target_package = params[1]
            self.migrations_rows = [
                row
                for row in self.migrations_rows
                if not (row["version"] == target_version and row["package"] == target_package)
            ]
            result = "DELETE 1"
            return result
        result = "EXECUTE"
        return result

    async def query(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        """
        emulate DataStore.query for the statements MigrationRunner issues.

        :param sql: SQL query text
        :ptype sql: str
        :param params: positional query parameters
        :ptype params: Any
        :return: list of row dictionaries
        :rtype: list[dict[str, Any]]
        """
        self.queried.append((sql, params))
        self.statements.append((sql, params))
        normalized = " ".join(sql.split()).upper()
        result: list[dict[str, Any]]
        if "CURRENT_DATABASE()" in normalized:
            result = [{"database_name": self._database}]
            return result
        if "PG_TRY_ADVISORY_LOCK" in normalized:
            granted = not self._lock_held_elsewhere
            self.lock_holds += int(granted)
            result = [{"acquired": granted}]
            return result
        if "PG_ADVISORY_UNLOCK" in normalized:
            released = self.lock_holds > 0
            self.lock_holds -= int(released)
            result = [{"released": released}]
            return result
        if "TO_REGCLASS(" in normalized:
            result = [{"present": self.migrations_table_created}]
            return result
        if "CURRENT_SCHEMA()" in normalized:
            result = [{"schema_name": self._schema}]
            return result
        if "SELECT VERSION, PACKAGE, DESCRIPTION FROM _SCHEMA_MIGRATIONS" in normalized:
            result = [
                {"version": row["version"], "package": row["package"], "description": row.get("description")}
                for row in self.migrations_rows
            ]
            return result
        if "SELECT VERSION, PACKAGE, DESCRIPTION, DATE_APPLIED FROM _SCHEMA_MIGRATIONS" in normalized:
            result = [
                {
                    "version": row["version"],
                    "package": row["package"],
                    "description": row["description"],
                    "date_applied": row["date_applied"],
                }
                for row in sorted(
                    self.migrations_rows,
                    key=lambda r: (r["date_applied"], r["version"], r["package"]),
                )
            ]
            return result
        if "COALESCE(MAX(VERSION)" in normalized:
            max_version = max((row["version"] for row in self.migrations_rows), default=0)
            result = [{"max_version": max_version}]
            return result
        result = []
        return result


# parity-exempt: a pool of recording one-session connections; its whole point is the acquire() boundary and per-connection lock ownership, which no single production protocol names
class FakeLockingPool:
    """
    an asyncpg-shaped pool of recording connections that models session lock ownership.

    ``acquire()`` hands out one :class:`FakeLockingConnection` per holder; the
    advisory lock belongs to the connection that took it, so a lock taken on
    one connection and released on another reports not-held, as PostgreSQL
    does. statements sent to the pool itself (one borrowed connection per
    statement) are recorded separately in ``direct``.

    :ivar connections: every connection the pool owns
    :ivar direct: SQL sent to the pool itself rather than an acquired connection
    :ivar lock_holder: the connection holding the DDL lock, or None
    :ivar ledger_present: whether ``_schema_migrations`` exists
    :ivar rows: the ledger's rows
    """

    def __init__(self, *, size: int = 2, ledger_present: bool = False) -> None:
        """
        build the pool.

        :param size: how many connections the pool owns
        :ptype size: int
        :param ledger_present: whether ``_schema_migrations`` already exists
        :ptype ledger_present: bool
        """
        self.connections = [FakeLockingConnection(index, self) for index in range(size)]
        self.direct: list[str] = []
        self.lock_holder: FakeLockingConnection | None = None
        self.lock_holds = 0
        self.ledger_present = ledger_present
        self.rows: list[dict[str, Any]] = []
        self._in_use: set[int] = set()
        self._applied = 0

    def acquire(self) -> _FakeAcquire:
        """
        hand out a free connection for the length of an ``async with``.

        :return: async context manager yielding the connection
        :rtype: _FakeAcquire
        """
        return _FakeAcquire(self)

    def take(self) -> FakeLockingConnection:
        """
        mark the first free connection in use and return it.

        :return: a free connection
        :rtype: FakeLockingConnection
        """
        connection = next(c for c in self.connections if c.index not in self._in_use)
        self._in_use.add(connection.index)
        return connection

    def give_back(self, connection: FakeLockingConnection) -> None:
        """
        return a connection to the pool.

        :param connection: the connection being returned
        :ptype connection: FakeLockingConnection
        """
        self._in_use.discard(connection.index)

    async def execute(self, sql: str, *params: Any) -> str:
        """
        record a statement sent to the pool itself.

        :param sql: SQL text
        :ptype sql: str
        :param params: positional parameters
        :ptype params: Any
        :return: synthetic status tag
        :rtype: str
        """
        self.direct.append(sql)
        return "EXECUTE"

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        """
        record a query sent to the pool itself.

        :param sql: SQL text
        :ptype sql: str
        :param params: positional parameters
        :ptype params: Any
        :return: no rows
        :rtype: list[dict[str, Any]]
        """
        self.direct.append(sql)
        return []

    def next_applied(self) -> int:
        """
        return the next ``date_applied`` stamp.

        :return: a strictly increasing counter
        :rtype: int
        """
        self._applied += 1
        return self._applied


# parity-exempt: the async context manager asyncpg.Pool.acquire() returns, reduced to taking and giving back one fake connection
class _FakeAcquire:
    """``async with pool.acquire() as conn`` for :class:`FakeLockingPool`."""

    def __init__(self, pool: FakeLockingPool) -> None:
        """
        remember the pool.

        :param pool: the pool to take a connection from
        :ptype pool: FakeLockingPool
        """
        self._pool = pool
        self._connection: FakeLockingConnection | None = None

    async def __aenter__(self) -> FakeLockingConnection:
        """
        take a connection.

        :return: the connection
        :rtype: FakeLockingConnection
        """
        self._connection = self._pool.take()
        return self._connection

    async def __aexit__(self, *exc_info: object) -> None:
        """
        give the connection back.

        :param exc_info: the exception triple, ignored
        :ptype exc_info: object
        """
        if self._connection is not None:
            self._pool.give_back(self._connection)


# parity-with: threetears.core.data.migrations.session.SqlConnection
class FakeLockingConnection:
    """
    one recording session of :class:`FakeLockingPool`.

    :ivar index: the connection's position in its pool
    :ivar statements: every SQL text this connection ran, in order
    """

    def __init__(self, index: int, pool: FakeLockingPool) -> None:
        """
        bind the connection to its pool.

        :param index: the connection's position in its pool
        :ptype index: int
        :param pool: the owning pool
        :ptype pool: FakeLockingPool
        """
        self.index = index
        self.statements: list[str] = []
        self._pool = pool

    async def execute(self, sql: str, *params: Any) -> str:
        """
        record a statement and emulate the ledger's writes.

        :param sql: SQL text
        :ptype sql: str
        :param params: positional parameters
        :ptype params: Any
        :return: synthetic status tag
        :rtype: str
        """
        self.statements.append(sql)
        normalized = " ".join(sql.split()).upper()
        if "CREATE TABLE IF NOT EXISTS _SCHEMA_MIGRATIONS" in normalized:
            self._pool.ledger_present = True
        elif normalized.startswith("INSERT INTO _SCHEMA_MIGRATIONS"):
            self._pool.rows.append(
                {
                    "version": params[0],
                    "package": params[1],
                    "description": params[2],
                    "date_applied": self._pool.next_applied(),
                }
            )
        elif normalized.startswith("DELETE FROM _SCHEMA_MIGRATIONS"):
            self._pool.rows = [
                row for row in self._pool.rows if not (row["version"] == params[0] and row["package"] == params[1])
            ]
        return "EXECUTE"

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        """
        record a query and answer the lock, the ledger probe and the ledger reads.

        :param sql: SQL text
        :ptype sql: str
        :param params: positional parameters
        :ptype params: Any
        :return: rows as dicts
        :rtype: list[dict[str, Any]]
        """
        self.statements.append(sql)
        normalized = " ".join(sql.split()).upper()
        pool = self._pool
        result: list[dict[str, Any]] = []
        if "CURRENT_DATABASE()" in normalized:
            result = [{"database_name": "appdb"}]
        elif "PG_TRY_ADVISORY_LOCK" in normalized:
            granted = pool.lock_holder in (None, self)
            if granted:
                pool.lock_holder = self
                pool.lock_holds += 1
            result = [{"acquired": granted}]
        elif "PG_ADVISORY_UNLOCK" in normalized:
            released = pool.lock_holder is self and pool.lock_holds > 0
            if released:
                pool.lock_holds -= 1
                if pool.lock_holds == 0:
                    pool.lock_holder = None
            result = [{"released": released}]
        elif "TO_REGCLASS(" in normalized:
            result = [{"present": pool.ledger_present}]
        elif "FROM _SCHEMA_MIGRATIONS" in normalized and "MAX(" not in normalized:
            result = [dict(row) for row in sorted(pool.rows, key=lambda r: (r["date_applied"], r["version"]))]
        return result
