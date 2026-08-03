"""unit tests for the driver transaction API (dsd-task-01, DSD-01-01/02).

scope:

- the :class:`Driver` ABC exposes ``begin`` and a concrete
  :meth:`Driver.transaction` async context manager
- :class:`Transaction` pins one session: two statements in one
  transaction provably route to the SAME backend connection object
- clean exit commits; an exception rolls back and re-raises
- a second-statement failure rolls the first statement back
  (atomicity, verified here against mocked backends; the live proof
  is ``tests/integration/test_transaction_live.py``)
- driver parity: :class:`RedshiftDriver` and :class:`AsyncpgDriver`
  run the SAME contract body, so a transaction feature added to one
  and not the other fails here rather than in the local stand-in

the mocked backends make the session-pinning assertion deterministic;
with a real cluster "same session" needs ``pg_backend_pid()``, which
belongs in the live suite.
"""

from __future__ import annotations

import asyncio
from abc import abstractmethod
from typing import Any
from unittest.mock import patch

import pytest

from threetears.datasources.config import (
    PostgresConnectionConfig,
    RedshiftConnectionConfig,
)
from threetears.datasources.drivers.asyncpg_driver import AsyncpgDriver
from threetears.datasources.drivers.base import (
    CallbackTransaction,
    ColumnRow,
    Driver,
    TableRow,
    Transaction,
    TransactionContext,
)
from threetears.datasources.drivers.redshift_driver import RedshiftDriver
from threetears.datasources.entities import DataSourceType

from ._helpers.driver_shims import (
    build_mock_redshift_connection,
    build_transaction_capable_pool,
    log_after_first_caller_statement,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_redshift_config() -> RedshiftConnectionConfig:
    """build the redshift config the mocked transaction tests share.

    :return: config with a small cache so pinning is observable
    :rtype: RedshiftConnectionConfig
    """
    return RedshiftConnectionConfig(
        datasource_type=DataSourceType.REDSHIFT,
        host="rs.example.com",
        port=5439,
        database="analytics",
        username="rs_user",
        password_ref=None,
        executor_max_workers=2,
        connection_cache_size=2,
        query_timeout_seconds=300,
    )


def _make_postgres_config() -> PostgresConnectionConfig:
    """build the postgres config the mocked transaction tests share.

    :return: config for the mocked asyncpg driver
    :rtype: PostgresConnectionConfig
    """
    return PostgresConnectionConfig(
        datasource_type=DataSourceType.POSTGRES,
        host="localhost",
        database="x",
    )


@pytest.fixture
def redshift_config() -> RedshiftConnectionConfig:
    """redshift config fixture.

    :return: config with a small cache so pinning is observable
    :rtype: RedshiftConnectionConfig
    """
    return _make_redshift_config()


@pytest.fixture
def postgres_config() -> PostgresConnectionConfig:
    """postgres config fixture.

    :return: config for the mocked asyncpg driver
    :rtype: PostgresConnectionConfig
    """
    return _make_postgres_config()


# ---------------------------------------------------------------------------
# ABC surface
# ---------------------------------------------------------------------------


class TestTransactionAbcSurface:
    """the ABC exposes begin / commit / rollback and one context manager."""

    def test_begin_is_abstract_on_driver(self) -> None:
        """``begin`` joins the abstract set so every driver must supply it."""
        assert "begin" in Driver.__abstractmethods__

    def test_transaction_is_concrete_on_driver(self) -> None:
        """``transaction`` is the shared context manager, NOT per-driver."""
        assert "transaction" not in Driver.__abstractmethods__

    def test_transaction_abstract_methods_match_documented_set(self) -> None:
        """:class:`Transaction` pins exactly fetch / execute / commit / rollback."""
        assert set(Transaction.__abstractmethods__) == {
            "fetch",
            "execute",
            "commit",
            "rollback",
        }

    def test_cannot_instantiate_raw_transaction(self) -> None:
        """the ABC itself is not constructible."""
        with pytest.raises(TypeError):
            Transaction()  # type: ignore[abstract]


class TestCallbackTransaction:
    """the shared concrete :class:`Transaction` both drivers construct."""

    @pytest.mark.asyncio
    async def test_fetch_forwards_sql_params_and_timeout(self) -> None:
        """``fetch`` hands sql / params / timeout to the driver callback."""
        seen: list[tuple[str, tuple[Any, ...], int | None]] = []

        async def _on_fetch(sql: str, params: tuple[Any, ...], timeout: int | None) -> list[dict[str, Any]]:
            seen.append((sql, params, timeout))
            return [{"a": 1}]

        transaction = CallbackTransaction(
            on_fetch=_on_fetch,
            on_execute=_unused_execute,
            on_finish=_unused_finish,
        )
        rows = await transaction.fetch("SELECT $1", 7, timeout_seconds=120)
        assert rows == [{"a": 1}]
        assert seen == [("SELECT $1", (7,), 120)]

    @pytest.mark.asyncio
    async def test_execute_forwards_sql_params_and_timeout(self) -> None:
        """``execute`` hands sql / params / timeout to the driver callback."""
        seen: list[tuple[str, tuple[Any, ...], int | None]] = []

        async def _on_execute(sql: str, params: tuple[Any, ...], timeout: int | None) -> None:
            seen.append((sql, params, timeout))

        transaction = CallbackTransaction(
            on_fetch=_unused_fetch,
            on_execute=_on_execute,
            on_finish=_unused_finish,
        )
        await transaction.execute("DROP TABLE $1", "t", timeout_seconds=300)
        assert seen == [("DROP TABLE $1", ("t",), 300)]

    @pytest.mark.asyncio
    async def test_commit_then_use_raises(self) -> None:
        """a committed transaction rejects further statements."""
        transaction = CallbackTransaction(
            on_fetch=_unused_fetch,
            on_execute=_unused_execute,
            on_finish=_noop_finish,
        )
        await transaction.commit()
        with pytest.raises(RuntimeError, match="already"):
            await transaction.fetch("SELECT 1")

    @pytest.mark.asyncio
    async def test_double_commit_raises(self) -> None:
        """the pinned connection is released once, so a second finish is an error."""
        finishes: list[bool] = []

        async def _on_finish(commit: bool) -> None:
            finishes.append(commit)

        transaction = CallbackTransaction(
            on_fetch=_unused_fetch,
            on_execute=_unused_execute,
            on_finish=_on_finish,
        )
        await transaction.commit()
        with pytest.raises(RuntimeError, match="already"):
            await transaction.commit()
        assert finishes == [True]

    @pytest.mark.asyncio
    async def test_finished_flag_set_before_callback_runs(self) -> None:
        """a raising finish callback still marks the transaction finished.

        otherwise a failed commit would let a caller retry and
        double-release the pinned connection.
        """

        async def _on_finish(commit: bool) -> None:
            raise RuntimeError("commit exploded")

        transaction = CallbackTransaction(
            on_fetch=_unused_fetch,
            on_execute=_unused_execute,
            on_finish=_on_finish,
        )
        with pytest.raises(RuntimeError, match="exploded"):
            await transaction.commit()
        assert transaction.finished is True


class TestTransactionContext:
    """the shared async context manager commits / rolls back exactly once."""

    @pytest.mark.asyncio
    async def test_clean_exit_commits(self) -> None:
        """no exception -> commit."""
        driver = _RecordingTransactionDriver()
        async with driver.transaction():
            pass
        assert driver.finishes == [True]

    @pytest.mark.asyncio
    async def test_exception_rolls_back_and_propagates(self) -> None:
        """an exception in the body -> rollback, original exception re-raised."""
        driver = _RecordingTransactionDriver()
        with pytest.raises(ValueError, match="boom"):
            async with driver.transaction():
                raise ValueError("boom")
        assert driver.finishes == [False]

    @pytest.mark.asyncio
    async def test_cancellation_rolls_back(self) -> None:
        """cancellation mid-transaction rolls back rather than leaking the session."""
        driver = _RecordingTransactionDriver()

        async def _run() -> None:
            async with driver.transaction():
                await asyncio.sleep(10)

        task = asyncio.create_task(_run())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert driver.finishes == [False]

    @pytest.mark.asyncio
    async def test_rollback_failure_does_not_mask_original_error(self) -> None:
        """a failing rollback is logged, never substituted for the caller's error."""
        driver = _RecordingTransactionDriver(finish_error=RuntimeError("rollback failed"))
        with pytest.raises(ValueError, match="boom"):
            async with driver.transaction():
                raise ValueError("boom")

    @pytest.mark.asyncio
    async def test_transaction_returns_context_object(self) -> None:
        """``transaction()`` is sync and returns the shared context type."""
        driver = _RecordingTransactionDriver()
        context = driver.transaction()
        assert isinstance(context, TransactionContext)
        async with context as transaction:
            assert isinstance(transaction, Transaction)


# ---------------------------------------------------------------------------
# Shared driver contract (parity between redshift + asyncpg)
# ---------------------------------------------------------------------------


class DriverTransactionContractTest:
    """assertions every concrete driver's transaction API must satisfy.

    concrete test classes supply :meth:`make_driver` plus the four
    backend accessors. running one body against both drivers is what
    makes a feature added to one and not the other fail at build time
    rather than in the local stand-in.
    """

    @abstractmethod
    async def make_driver(self) -> tuple[Driver, Any]:
        """build a driver over a mocked backend.

        :return: ``(driver, backend)`` where backend exposes the mocked
            connection surface the assertions inspect
        :rtype: tuple[Driver, Any]
        """

    @abstractmethod
    async def teardown_driver(self, driver: Driver, backend: Any) -> None:
        """close the driver and undo any backend patching.

        :param driver: driver from :meth:`make_driver`
        :ptype driver: Driver
        :param backend: the mock handle from :meth:`make_driver`
        :ptype backend: Any
        :return: nothing
        :rtype: None
        """

    @abstractmethod
    def sessions_used(self, backend: Any) -> list[Any]:
        """return the connection object each caller statement ran on.

        :param backend: the mock handle from :meth:`make_driver`
        :ptype backend: Any
        :return: connection objects in statement order
        :rtype: list[Any]
        """

    @abstractmethod
    def commit_count(self, backend: Any) -> int:
        """return how many times the backend committed.

        :param backend: the mock handle from :meth:`make_driver`
        :ptype backend: Any
        :return: commit call count
        :rtype: int
        """

    @abstractmethod
    def rollback_count(self, backend: Any) -> int:
        """return how many times the backend rolled back.

        :param backend: the mock handle from :meth:`make_driver`
        :ptype backend: Any
        :return: rollback call count
        :rtype: int
        """

    @abstractmethod
    def arm_failure(self, backend: Any, index: int) -> type[Exception]:
        """arm the backend so the ``index``-th caller statement raises.

        :param backend: the mock handle from :meth:`make_driver`
        :ptype backend: Any
        :param index: 1-based caller-statement position to fail at
        :ptype index: int
        :return: the exception type the statement raises
        :rtype: type[Exception]
        """

    @pytest.mark.asyncio
    async def test_begin_returns_transaction(self) -> None:
        """``begin`` yields a :class:`Transaction` handle."""
        driver, _backend = await self.make_driver()
        try:
            transaction = await driver.begin()
            assert isinstance(transaction, Transaction)
            await transaction.rollback()
        finally:
            await self.teardown_driver(driver, _backend)

    @pytest.mark.asyncio
    async def test_two_statements_share_one_session(self) -> None:
        """DSD-01-02: a multi-statement transaction is pinned to ONE session."""
        driver, backend = await self.make_driver()
        try:
            async with driver.transaction() as transaction:
                await transaction.execute("CREATE TABLE t (a int)")
                await transaction.execute("INSERT INTO t VALUES (1)")
            sessions = self.sessions_used(backend)
            assert len(sessions) == 2, f"expected 2 caller statements, saw {len(sessions)}"
            assert len({id(session) for session in sessions}) == 1, (
                f"statements landed on {len({id(s) for s in sessions})} distinct sessions"
            )
        finally:
            await self.teardown_driver(driver, backend)

    @pytest.mark.asyncio
    async def test_clean_exit_commits_once(self) -> None:
        """the context manager commits exactly once on a clean exit."""
        driver, backend = await self.make_driver()
        try:
            async with driver.transaction() as transaction:
                await transaction.execute("INSERT INTO t VALUES (1)")
            assert self.commit_count(backend) == 1
        finally:
            await self.teardown_driver(driver, backend)

    @pytest.mark.asyncio
    async def test_second_statement_failure_rolls_back_atomically(self) -> None:
        """DSD-01-08: a two-statement transaction rolls back on the second failure."""
        driver, backend = await self.make_driver()
        try:
            error_type = self.arm_failure(backend, 2)
            with pytest.raises(error_type):
                async with driver.transaction() as transaction:
                    await transaction.execute("CREATE TABLE t (a int)")
                    await transaction.execute("INSERT INTO t VALUES (bad)")
            assert self.commit_count(backend) == 0, "a failed transaction must never commit"
            assert self.rollback_count(backend) >= 1, "a failed transaction must roll back"
        finally:
            await self.teardown_driver(driver, backend)

    @pytest.mark.asyncio
    async def test_explicit_commit_marks_transaction_finished(self) -> None:
        """explicit commit works without the context manager."""
        driver, backend = await self.make_driver()
        try:
            transaction = await driver.begin()
            await transaction.execute("INSERT INTO t VALUES (1)")
            await transaction.commit()
            assert self.commit_count(backend) == 1
            with pytest.raises(RuntimeError, match="already"):
                await transaction.execute("INSERT INTO t VALUES (2)")
        finally:
            await self.teardown_driver(driver, backend)


class TestRedshiftDriverTransaction(DriverTransactionContractTest):
    """redshift half of the driver-parity contract."""

    async def make_driver(self) -> tuple[Driver, Any]:
        """build a :class:`RedshiftDriver` over a mocked connector.

        the patch is stopped by the driver fixture's ``close`` in each
        test body; ``patch.start`` here keeps the contract methods free
        of ``with`` nesting so the shared body reads the same for both
        drivers.

        :return: driver plus the mocked connection
        :rtype: tuple[Driver, Any]
        """
        conn = build_mock_redshift_connection()
        patcher = patch(
            "threetears.datasources.drivers.redshift_driver.redshift_connector.connect",
            return_value=conn,
        )
        patcher.start()
        conn.stop_patch = patcher.stop
        return RedshiftDriver(_make_redshift_config()), conn

    async def teardown_driver(self, driver: Driver, backend: Any) -> None:
        """close the driver, then stop the connector patch.

        :param driver: driver from :meth:`make_driver`
        :ptype driver: Driver
        :param backend: mocked redshift connection
        :ptype backend: Any
        :return: nothing
        :rtype: None
        """
        try:
            await driver.close()
        finally:
            backend.stop_patch()

    def sessions_used(self, backend: Any) -> list[Any]:
        """return the connection each caller statement ran on.

        :param backend: mocked redshift connection
        :ptype backend: Any
        :return: connections in statement order
        :rtype: list[Any]
        """
        return [entry["conn"] for entry in backend.statement_log]

    def commit_count(self, backend: Any) -> int:
        """count ``conn.commit`` calls made after the first caller statement.

        the driver commits at connection open so its session settings
        survive the release-path rollback; counting from the first
        caller statement keeps this an assertion about the transaction.

        :param backend: mocked redshift connection
        :ptype backend: Any
        :return: commit call count
        :rtype: int
        """
        return log_after_first_caller_statement(backend.sql_log).count("COMMIT")

    def rollback_count(self, backend: Any) -> int:
        """count ``conn.rollback`` calls made after the first caller statement.

        :param backend: mocked redshift connection
        :ptype backend: Any
        :return: rollback call count
        :rtype: int
        """
        return log_after_first_caller_statement(backend.sql_log).count("ROLLBACK")

    def arm_failure(self, backend: Any, index: int) -> type[Exception]:
        """arm the mocked cursor so the ``index``-th caller statement raises.

        :param backend: mocked redshift connection
        :ptype backend: Any
        :param index: 1-based caller-statement position
        :ptype index: int
        :return: the exception type raised
        :rtype: type[Exception]
        """

        class _ProgrammingError(Exception):
            pass

        backend.fail_caller_statement_at(index, _ProgrammingError("syntax error at or near bad"))
        return _ProgrammingError


class TestAsyncpgDriverTransaction(DriverTransactionContractTest):
    """asyncpg half of the driver-parity contract."""

    async def make_driver(self) -> tuple[Driver, Any]:
        """build an :class:`AsyncpgDriver` over a mocked pool.

        :return: driver plus the mocked pool
        :rtype: tuple[Driver, Any]
        """
        pool = build_transaction_capable_pool()
        driver = AsyncpgDriver(_make_postgres_config(), external_pool=pool)
        return driver, pool

    async def teardown_driver(self, driver: Driver, backend: Any) -> None:
        """close the driver; the mocked pool needs no unpatching.

        :param driver: driver from :meth:`make_driver`
        :ptype driver: Driver
        :param backend: mocked pool
        :ptype backend: Any
        :return: nothing
        :rtype: None
        """
        await driver.close()

    def sessions_used(self, backend: Any) -> list[Any]:
        """return the connection each caller statement ran on.

        :param backend: mocked pool
        :ptype backend: Any
        :return: connections in statement order
        :rtype: list[Any]
        """
        return [entry["conn"] for entry in backend.statement_log]

    def commit_count(self, backend: Any) -> int:
        """count transaction commits on the mocked pool.

        :param backend: mocked pool
        :ptype backend: Any
        :return: commit await count
        :rtype: int
        """
        return int(backend.transaction_handle.commit.await_count)

    def rollback_count(self, backend: Any) -> int:
        """count transaction rollbacks on the mocked pool.

        :param backend: mocked pool
        :ptype backend: Any
        :return: rollback await count
        :rtype: int
        """
        return int(backend.transaction_handle.rollback.await_count)

    def arm_failure(self, backend: Any, index: int) -> type[Exception]:
        """arm the mocked connection so the ``index``-th caller statement raises.

        :param backend: mocked pool
        :ptype backend: Any
        :param index: 1-based caller-statement position
        :ptype index: int
        :return: the exception type raised
        :rtype: type[Exception]
        """

        class _PostgresError(Exception):
            pass

        backend.fail_caller_statement_at(index, _PostgresError("syntax error at or near bad"))
        return _PostgresError


# ---------------------------------------------------------------------------
# Session pinning consumes a pooled connection for the duration
# ---------------------------------------------------------------------------


class TestRedshiftSessionPinning:
    """the pinned connection is out of the cache until the transaction finishes."""

    @pytest.mark.asyncio
    async def test_connection_not_in_cache_while_transaction_open(
        self,
        redshift_config: RedshiftConnectionConfig,
    ) -> None:
        """DSD-01-02: the cache must not hand the pinned session to a second caller."""
        conn = build_mock_redshift_connection()
        with patch(
            "threetears.datasources.drivers.redshift_driver.redshift_connector.connect",
            return_value=conn,
        ):
            driver = RedshiftDriver(redshift_config)
            transaction = await driver.begin()
            await transaction.execute("INSERT INTO t VALUES (1)")
            assert conn not in driver._cache  # noqa: SLF001
            await transaction.commit()
            assert conn in driver._cache  # noqa: SLF001
            await driver.close()

    @pytest.mark.asyncio
    async def test_rollback_returns_connection_to_cache(
        self,
        redshift_config: RedshiftConnectionConfig,
    ) -> None:
        """an explicit rollback releases the pinned connection back to the cache."""
        conn = build_mock_redshift_connection()
        with patch(
            "threetears.datasources.drivers.redshift_driver.redshift_connector.connect",
            return_value=conn,
        ):
            driver = RedshiftDriver(redshift_config)
            transaction = await driver.begin()
            await transaction.execute("INSERT INTO t VALUES (1)")
            await transaction.rollback()
            assert conn in driver._cache  # noqa: SLF001
            await driver.close()


class TestAsyncpgSessionPinning:
    """the pinned connection is released back to the pool exactly once."""

    @pytest.mark.asyncio
    async def test_connection_released_after_commit(
        self,
        postgres_config: PostgresConnectionConfig,
    ) -> None:
        """DSD-01-02: pool release happens on finish, not between statements."""
        pool = build_transaction_capable_pool()
        driver = AsyncpgDriver(postgres_config, external_pool=pool)
        transaction = await driver.begin()
        await transaction.execute("INSERT INTO t VALUES (1)")
        assert pool.release.await_count == 0
        await transaction.commit()
        assert pool.release.await_count == 1
        await driver.close()

    @pytest.mark.asyncio
    async def test_connection_released_after_failed_transaction(
        self,
        postgres_config: PostgresConnectionConfig,
    ) -> None:
        """a failed transaction still returns its connection to the pool."""

        class _PostgresError(Exception):
            pass

        pool = build_transaction_capable_pool()
        pool.fail_caller_statement_at(1, _PostgresError("boom"))
        driver = AsyncpgDriver(postgres_config, external_pool=pool)
        with pytest.raises(_PostgresError):
            async with driver.transaction() as transaction:
                await transaction.execute("INSERT INTO t VALUES (1)")
        assert pool.release.await_count == 1
        await driver.close()


# ---------------------------------------------------------------------------
# Closed-driver guards
# ---------------------------------------------------------------------------


class TestBeginRejectsClosedDriver:
    """``begin`` honours the close-concurrency contract on both drivers."""

    @pytest.mark.asyncio
    async def test_redshift_begin_after_close(self, redshift_config: RedshiftConnectionConfig) -> None:
        """closed redshift driver rejects ``begin``."""
        driver = RedshiftDriver(redshift_config)
        await driver.close()
        with pytest.raises(RuntimeError, match="closed"):
            await driver.begin()

    @pytest.mark.asyncio
    async def test_asyncpg_begin_after_close(self, postgres_config: PostgresConnectionConfig) -> None:
        """closed asyncpg driver rejects ``begin``."""
        pool = build_transaction_capable_pool()
        driver = AsyncpgDriver(postgres_config, external_pool=pool)
        await driver.close()
        with pytest.raises(RuntimeError, match="closed"):
            await driver.begin()


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------


async def _unused_fetch(sql: str, params: tuple[Any, ...], timeout: int | None) -> list[dict[str, Any]]:
    """fetch callback that must never be reached.

    :param sql: unused
    :ptype sql: str
    :param params: unused
    :ptype params: tuple[Any, ...]
    :param timeout: unused
    :ptype timeout: int | None
    :return: never returns
    :rtype: list[dict[str, Any]]
    :raises AssertionError: always
    """
    raise AssertionError("fetch callback should not have been invoked")


async def _unused_execute(sql: str, params: tuple[Any, ...], timeout: int | None) -> None:
    """execute callback that must never be reached.

    :param sql: unused
    :ptype sql: str
    :param params: unused
    :ptype params: tuple[Any, ...]
    :param timeout: unused
    :ptype timeout: int | None
    :return: never returns
    :rtype: None
    :raises AssertionError: always
    """
    raise AssertionError("execute callback should not have been invoked")


async def _unused_finish(commit: bool) -> None:
    """finish callback that must never be reached.

    :param commit: unused
    :ptype commit: bool
    :return: never returns
    :rtype: None
    :raises AssertionError: always
    """
    raise AssertionError("finish callback should not have been invoked")


async def _noop_finish(commit: bool) -> None:
    """finish callback that records nothing and succeeds.

    :param commit: True on commit, False on rollback
    :ptype commit: bool
    :return: nothing
    :rtype: None
    """
    return None


class _RecordingTransactionDriver(Driver):
    """minimal :class:`Driver` recording how its transactions finished.

    exists so :class:`TransactionContext` can be exercised without
    either backend library in the loop.
    """

    def __init__(self, *, finish_error: Exception | None = None) -> None:
        """capture the optional finish failure and reset the record.

        :param finish_error: exception the finish callback raises, or None
        :ptype finish_error: Exception | None
        :return: nothing
        :rtype: None
        """
        self.finishes: list[bool] = []
        self._finish_error = finish_error

    async def begin(self) -> Transaction:
        """return a transaction recording its finish disposition.

        :return: recording transaction handle
        :rtype: Transaction
        """
        return CallbackTransaction(
            on_fetch=self._on_fetch,
            on_execute=self._on_execute,
            on_finish=self._on_finish,
        )

    async def _on_fetch(self, sql: str, params: tuple[Any, ...], timeout: int | None) -> list[dict[str, Any]]:
        """record nothing; return no rows.

        :param sql: statement text
        :ptype sql: str
        :param params: bind values
        :ptype params: tuple[Any, ...]
        :param timeout: per-statement timeout override
        :ptype timeout: int | None
        :return: empty row list
        :rtype: list[dict[str, Any]]
        """
        return []

    async def _on_execute(self, sql: str, params: tuple[Any, ...], timeout: int | None) -> None:
        """record nothing.

        :param sql: statement text
        :ptype sql: str
        :param params: bind values
        :ptype params: tuple[Any, ...]
        :param timeout: per-statement timeout override
        :ptype timeout: int | None
        :return: nothing
        :rtype: None
        """
        return None

    async def _on_finish(self, commit: bool) -> None:
        """record the disposition and optionally fail.

        :param commit: True on commit, False on rollback
        :ptype commit: bool
        :return: nothing
        :rtype: None
        :raises Exception: the seeded ``finish_error`` when configured
        """
        self.finishes.append(commit)
        if self._finish_error is not None:
            raise self._finish_error

    async def fetch(self, sql: str, *params: Any, timeout_seconds: int | None = None) -> list[dict[str, Any]]:
        """unused on this recorder.

        :param sql: statement text
        :ptype sql: str
        :param params: bind values
        :ptype params: Any
        :param timeout_seconds: per-statement timeout override
        :ptype timeout_seconds: int | None
        :return: empty row list
        :rtype: list[dict[str, Any]]
        """
        return []

    async def execute(self, sql: str, *params: Any, timeout_seconds: int | None = None) -> None:
        """unused on this recorder.

        :param sql: statement text
        :ptype sql: str
        :param params: bind values
        :ptype params: Any
        :param timeout_seconds: per-statement timeout override
        :ptype timeout_seconds: int | None
        :return: nothing
        :rtype: None
        """
        return None

    async def list_tables(self, schemas: list[str]) -> list[TableRow]:
        """unused on this recorder.

        :param schemas: schema allow-list
        :ptype schemas: list[str]
        :return: empty list
        :rtype: list[TableRow]
        """
        return []

    async def list_columns(self, schemas: list[str]) -> list[ColumnRow]:
        """unused on this recorder.

        :param schemas: schema allow-list
        :ptype schemas: list[str]
        :return: empty list
        :rtype: list[ColumnRow]
        """
        return []

    async def table_hashes(self, schemas: list[str]) -> dict[tuple[str, str], str]:
        """unused on this recorder.

        :param schemas: schema allow-list
        :ptype schemas: list[str]
        :return: empty mapping
        :rtype: dict[tuple[str, str], str]
        """
        return {}

    async def test_connection(self) -> None:
        """unused on this recorder.

        :return: nothing
        :rtype: None
        """
        return None

    async def close(self) -> None:
        """unused on this recorder.

        :return: nothing
        :rtype: None
        """
        return None
