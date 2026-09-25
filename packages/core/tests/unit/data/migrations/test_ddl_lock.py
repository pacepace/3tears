"""
unit tests for the database-wide DDL lock.

the lock's contracts, each held here against a scripted fake session:

- the key comes from ``current_database()``, never the schema
- the waiter POLLS with ``pg_try_advisory_lock`` and never issues the blocking
  ``pg_advisory_lock`` -- a blocked waiter deadlocks an online index build
- ``max_wait`` raises a typed error and the body never runs
- a waiter reports itself at INFO, naming the database and the time waited
- the lock is released on the session after a body that returns, raises, or is
  cancelled, and a release that fails or finds nothing held raises loudly
- a session that is not PostgreSQL fails loudly rather than spinning

the behaviour against a real engine -- two sessions serialising, no deadlock
with ``CREATE INDEX CONCURRENTLY`` -- is in
``tests/integration/migrations/test_ddl_lock_against_postgres.py``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from threetears.core.data.migrations import (
    DDL_LOCK_NAMESPACE,
    ConnectionSession,
    DdlLockError,
    DdlLockPolicy,
    DdlLockReleaseError,
    DdlLockTimeoutError,
    SessionRequiredError,
    database_ddl_lock,
    ddl_lock_key,
)

_FAST = DdlLockPolicy(poll_interval=0.001, log_interval=60.0)


# parity-with: threetears.core.data.migrations.session.MigrationSession
class _FakeLockSession:
    """
    a session whose lock answers are scripted, recording every statement.

    :ivar statements: every (sql, params) the lock issued, in order
    """

    def __init__(
        self,
        *,
        database: str = "appdb",
        grants: list[bool] | None = None,
        release_answer: bool = True,
        release_error: BaseException | None = None,
        empty_answers: bool = False,
    ) -> None:
        """
        script the session's answers.

        :param database: what ``current_database()`` answers
        :ptype database: str
        :param grants: successive answers to ``pg_try_advisory_lock``; the
            last answer repeats once the list is used up
        :ptype grants: list[bool] | None
        :param release_answer: what ``pg_advisory_unlock`` answers
        :ptype release_answer: bool
        :param release_error: raised by ``pg_advisory_unlock`` instead of answering
        :ptype release_error: BaseException | None
        :param empty_answers: answer every query with no rows, as a store that
            is not PostgreSQL would
        :ptype empty_answers: bool
        """
        self.statements: list[tuple[str, tuple[Any, ...]]] = []
        self._database = database
        self._grants = list(grants) if grants is not None else [True]
        self._release_answer = release_answer
        self._release_error = release_error
        self._empty_answers = empty_answers
        self.held = 0

    async def execute(self, sql: str, *params: Any) -> str:
        """
        record a statement; the lock never executes, so any call is a finding.

        :param sql: SQL text
        :ptype sql: str
        :param params: positional parameters
        :ptype params: Any
        :return: synthetic status tag
        :rtype: str
        """
        self.statements.append((sql, params))
        result = "EXECUTE"
        return result

    async def query(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        """
        answer the lock's three statements from the script.

        :param sql: SQL text
        :ptype sql: str
        :param params: positional parameters
        :ptype params: Any
        :return: scripted rows
        :rtype: list[dict[str, Any]]
        """
        self.statements.append((sql, params))
        result: list[dict[str, Any]] = []
        if self._empty_answers:
            return result
        if "current_database()" in sql:
            result = [{"database_name": self._database}]
        elif "pg_try_advisory_lock" in sql:
            granted = self._grants.pop(0) if len(self._grants) > 1 else self._grants[0]
            self.held += int(granted)
            result = [{"acquired": granted}]
        elif "pg_advisory_unlock" in sql:
            if self._release_error is not None:
                raise self._release_error
            self.held -= int(self._release_answer)
            result = [{"released": self._release_answer}]
        return result

    def lock_statements(self) -> list[str]:
        """
        return the advisory-lock statements issued, in order.

        :return: SQL text of every statement naming an advisory lock function
        :rtype: list[str]
        """
        result = [sql for sql, _ in self.statements if "advisory" in sql]
        return result


class TestKey:
    """the lock is keyed on the database, so every schema in it shares one lock."""

    def test_key_is_stable_for_one_database(self) -> None:
        """every pod computing the key for one database agrees."""
        assert ddl_lock_key("appdb") == ddl_lock_key("appdb")

    def test_key_differs_between_databases(self) -> None:
        """different databases do not serialise against each other."""
        assert ddl_lock_key("appdb") != ddl_lock_key("otherdb")

    def test_key_and_namespace_fit_a_signed_int4(self) -> None:
        """both keys bind to the (int4, int4) overload."""
        for value in (DDL_LOCK_NAMESPACE, ddl_lock_key("appdb"), ddl_lock_key("")):
            assert 0 <= value <= 0x7FFFFFFF

    async def test_lock_is_taken_on_the_database_key(self) -> None:
        """the try-lock binds the namespace and the key of current_database()."""
        session = _FakeLockSession(database="appdb")
        async with database_ddl_lock(session, _FAST) as lease:
            assert lease.database == "appdb"
            assert lease.key == ddl_lock_key("appdb")
        try_params = next(params for sql, params in session.statements if "pg_try_advisory_lock" in sql)
        assert try_params == (DDL_LOCK_NAMESPACE, ddl_lock_key("appdb"))


class TestPolling:
    """a waiter polls and never blocks inside a statement."""

    async def test_waiter_polls_until_granted(self) -> None:
        """three refusals then a grant: four attempts, then the body runs."""
        session = _FakeLockSession(grants=[False, False, False, True])
        ran = False
        async with database_ddl_lock(session, _FAST):
            ran = True
        attempts = [sql for sql in session.lock_statements() if "pg_try_advisory_lock" in sql]
        assert ran
        assert len(attempts) == 4

    async def test_blocking_lock_is_never_issued(self) -> None:
        """no statement ever waits on the lock server-side."""
        session = _FakeLockSession(grants=[False, False, True])
        async with database_ddl_lock(session, _FAST):
            pass
        blocking = [sql for sql in session.lock_statements() if "pg_advisory_lock(" in sql]
        assert blocking == []

    async def test_waiting_is_reported_with_database_and_elapsed(self, caplog: pytest.LogCaptureFixture) -> None:
        """a caller still waiting says so at INFO, naming the database and the wait."""
        session = _FakeLockSession(database="appdb", grants=[False] * 30 + [True])
        policy = DdlLockPolicy(poll_interval=0.002, log_interval=0.01)
        with caplog.at_level(logging.INFO, logger="threetears.core.data.migrations.ddl_lock"):
            async with database_ddl_lock(session, policy):
                pass
        waiting = [r for r in caplog.records if "waiting for the DDL lock" in r.getMessage()]
        assert len(waiting) >= 2
        assert all(r.levelno == logging.INFO for r in waiting)
        assert all("database=appdb" in r.getMessage() and "waited=" in r.getMessage() for r in waiting)

    async def test_a_granted_first_attempt_logs_no_wait(self, caplog: pytest.LogCaptureFixture) -> None:
        """an uncontended lock reports no waiting."""
        session = _FakeLockSession()
        with caplog.at_level(logging.INFO, logger="threetears.core.data.migrations.ddl_lock"):
            async with database_ddl_lock(session, _FAST):
                pass
        assert not [r for r in caplog.records if "waiting for the DDL lock" in r.getMessage()]


class TestMaxWait:
    """``max_wait`` bounds the wait with a typed error."""

    async def test_max_wait_raises_typed_error_and_body_never_runs(self) -> None:
        """a lock that never frees raises DdlLockTimeoutError naming the database."""
        session = _FakeLockSession(database="appdb", grants=[False])
        ran = False
        with pytest.raises(DdlLockTimeoutError) as info:
            async with database_ddl_lock(session, DdlLockPolicy(poll_interval=0.005, max_wait=0.05)):
                ran = True
        assert not ran
        assert info.value.database == "appdb"
        assert info.value.waited_seconds >= 0.05
        assert "appdb" in str(info.value)
        assert [sql for sql in session.lock_statements() if "pg_advisory_unlock" in sql] == []

    async def test_max_wait_zero_tries_exactly_once(self) -> None:
        """max_wait=0 is a single attempt."""
        session = _FakeLockSession(grants=[False])
        with pytest.raises(DdlLockTimeoutError):
            async with database_ddl_lock(session, DdlLockPolicy(max_wait=0.0)):
                pass
        attempts = [sql for sql in session.lock_statements() if "pg_try_advisory_lock" in sql]
        assert len(attempts) == 1

    async def test_max_wait_does_not_oversleep_a_long_poll(self) -> None:
        """a poll interval longer than max_wait is cut short at the deadline."""
        session = _FakeLockSession(grants=[False])
        loop = asyncio.get_running_loop()
        started = loop.time()
        with pytest.raises(DdlLockTimeoutError):
            async with database_ddl_lock(session, DdlLockPolicy(poll_interval=30.0, max_wait=0.05)):
                pass
        assert loop.time() - started < 5.0

    @pytest.mark.parametrize(
        ("kwargs", "field"),
        [
            ({"poll_interval": 0.0}, "poll_interval"),
            ({"log_interval": 0.0}, "log_interval"),
            ({"max_wait": -1.0}, "max_wait"),
        ],
    )
    def test_policy_refuses_values_that_spin_or_wait_negatively(self, kwargs: dict[str, float], field: str) -> None:
        """a zero poll spins, a zero report interval floods, a negative wait means nothing."""
        with pytest.raises(ValueError, match=field):
            DdlLockPolicy(**kwargs)


class TestRelease:
    """the lock goes back on the same session however the body ends."""

    async def test_released_after_body_returns(self) -> None:
        """a body that returns leaves the session holding nothing."""
        session = _FakeLockSession()
        async with database_ddl_lock(session, _FAST):
            assert session.held == 1
        assert session.held == 0

    async def test_released_after_body_raises(self) -> None:
        """a raising body still releases, and its own error propagates."""
        session = _FakeLockSession()
        with pytest.raises(RuntimeError, match="body failed"):
            async with database_ddl_lock(session, _FAST):
                msg = "body failed"
                raise RuntimeError(msg)
        assert session.held == 0

    async def test_released_after_body_is_cancelled(self) -> None:
        """a cancelled body still releases before the cancellation propagates."""
        session = _FakeLockSession()
        entered = asyncio.Event()

        async def _hold() -> None:
            """hold the lock until cancelled."""
            async with database_ddl_lock(session, _FAST):
                entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(_hold())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert session.held == 0

    async def test_release_that_finds_nothing_held_raises(self) -> None:
        """an unlock answering false means the run was not on one session."""
        session = _FakeLockSession(database="appdb", release_answer=False)
        with pytest.raises(DdlLockReleaseError) as info:
            async with database_ddl_lock(session, _FAST):
                pass
        assert info.value.database == "appdb"
        assert "one session" in str(info.value)

    async def test_release_that_fails_raises_chained(self) -> None:
        """an unlock that raises is reported as a release failure, with its cause."""
        cause = OSError("connection reset")
        session = _FakeLockSession(release_error=cause)
        with pytest.raises(DdlLockReleaseError) as info:
            async with database_ddl_lock(session, _FAST):
                pass
        assert info.value.__cause__ is cause

    @pytest.mark.parametrize("unlock", ["raises", "finds nothing held"])
    async def test_a_failed_body_keeps_its_own_error_when_the_unlock_also_fails(
        self, unlock: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        """the body's exception propagates, noted and logged, never replaced by a lock error."""
        session = (
            _FakeLockSession(database="appdb", release_error=OSError("connection reset"))
            if unlock == "raises"
            else _FakeLockSession(database="appdb", release_answer=False)
        )
        with caplog.at_level(logging.ERROR, logger="threetears.core.data.migrations.ddl_lock"):
            with pytest.raises(RuntimeError, match="migration body failed") as info:
                async with database_ddl_lock(session, _FAST):
                    msg = "migration body failed"
                    raise RuntimeError(msg)
        assert any("appdb" in note and "not released" in note for note in info.value.__notes__)
        errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
        assert any("failed with RuntimeError" in m and "database=appdb" in m and "terminate" in m for m in errors)

    async def test_a_cancelled_body_stays_cancelled_when_the_unlock_fails(self) -> None:
        """a timeout wrapper still sees its cancellation, not a lock error."""
        session = _FakeLockSession(release_error=OSError("connection reset"))
        entered = asyncio.Event()

        async def _hold() -> None:
            """hold the lock until cancelled."""
            async with database_ddl_lock(session, _FAST):
                entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(_hold())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_cancelled_attempt_gives_back_what_it_may_have_taken(self) -> None:
        """a cancellation landing on the try-lock unlocks before propagating."""

        class _SlowGrantSession(_FakeLockSession):
            """a session whose try-lock is granted server-side but answers slowly."""

            async def query(self, sql: str, *params: Any) -> list[dict[str, Any]]:
                """
                grant the lock, then stall before answering the try-lock.

                :param sql: SQL text
                :ptype sql: str
                :param params: positional parameters
                :ptype params: Any
                :return: scripted rows
                :rtype: list[dict[str, Any]]
                """
                rows = await super().query(sql, *params)
                if "pg_try_advisory_lock" in sql:
                    await asyncio.Event().wait()
                return rows

        session = _SlowGrantSession()

        async def _take() -> None:
            """try to take the lock."""
            async with database_ddl_lock(session, _FAST):
                pass

        task = asyncio.create_task(_take())
        while session.held == 0:
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert session.held == 0


class TestNotPostgres:
    """a session that answers nothing fails loudly instead of polling forever."""

    async def test_no_row_from_the_lock_statements_raises(self) -> None:
        """no current_database() row: DdlLockError, before any lock statement."""
        session = _FakeLockSession(empty_answers=True)
        with pytest.raises(DdlLockError, match="PostgreSQL"):
            async with database_ddl_lock(session, _FAST):
                pass
        assert session.lock_statements() == []


class TestConnectionSession:
    """``ConnectionSession`` wraps one connection and refuses a pool."""

    def test_refuses_anything_with_acquire(self) -> None:
        """a pool hands statements to any connection, so it is refused by name."""

        class _PoolShaped:
            """has acquire(), as asyncpg.Pool and every L3Backend do."""

            def acquire(self) -> None:
                """pretend to hand out a connection."""

            async def execute(self, query: str, *args: Any) -> str:
                """
                unused.

                :param query: SQL text
                :ptype query: str
                :param args: positional parameters
                :ptype args: Any
                :return: nothing useful
                :rtype: str
                """
                return ""

            async def fetch(self, query: str, *args: Any) -> list[Any]:
                """
                unused.

                :param query: SQL text
                :ptype query: str
                :param args: positional parameters
                :ptype args: Any
                :return: nothing
                :rtype: list[Any]
                """
                return []

        with pytest.raises(SessionRequiredError, match="_PoolShaped"):
            ConnectionSession(_PoolShaped())

    async def test_routes_to_the_connection_and_returns_dict_rows(self) -> None:
        """execute and query go to the one connection; rows come back as dicts."""
        calls: list[tuple[str, str, tuple[Any, ...]]] = []

        class _Conn:
            """records execute/fetch."""

            async def execute(self, query: str, *args: Any) -> str:
                """
                record an execute.

                :param query: SQL text
                :ptype query: str
                :param args: positional parameters
                :ptype args: Any
                :return: status tag
                :rtype: str
                """
                calls.append(("execute", query, args))
                return "SELECT 1"

            async def fetch(self, query: str, *args: Any) -> list[Any]:
                """
                record a fetch.

                :param query: SQL text
                :ptype query: str
                :param args: positional parameters
                :ptype args: Any
                :return: one row as key/value pairs
                :rtype: list[Any]
                """
                calls.append(("fetch", query, args))
                return [[("n", 1)]]

        session = ConnectionSession(_Conn())
        assert await session.execute("SET x = $1", 1) == "SELECT 1"
        assert await session.query("SELECT $1 AS n", 1) == [{"n": 1}]
        assert calls == [("execute", "SET x = $1", (1,)), ("fetch", "SELECT $1 AS n", (1,))]
