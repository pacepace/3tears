"""a credential the warehouse refused is not sent again, by any replica.

the contract (hub issue #523, the lockout it closes):

- one refusal pauses the credential for EVERY replica sharing the coordination tier,
  and a paused connect raises :class:`DriverCredentialPausedError` without a login;
- the pause belongs to the refused CREDENTIAL: a holder of a superseded one cannot pause
  the credential that replaced it, and replacing a credential needs no step to resume;
- logins with a credential go one at a time, always, so a burst against a refusing
  warehouse costs one login -- a cold start's, and the first after a warehouse-side
  password change on a credential that worked a moment ago;
- several background passes against a refusing warehouse cost exactly one login;
- a missing credential, refused before any login, pauses nothing;
- clearing the pause (a new credential, a successful probe) lets the next login through;
- a probe -- a driver built with no guard -- logs in even while the credential is paused;
- a coordination store that cannot be read lets the connect go ahead, and one that
  cannot be written never replaces the refusal the caller must see (fail-open);
- both drivers that classify a refusal consult the guard: Redshift on every fresh
  connection, the cancel path's terminate login included, and asyncpg on every login
  its pool makes -- which concurrent first callers share rather than each building.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import asyncpg
import pytest
import redshift_connector

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.coordination.tables import CoordinationCountersCollection
from threetears.core.exceptions import DataLayerUnavailableError
from threetears.core.testing.kv import FakeNatsClient
from threetears.datasources.config import PostgresConnectionConfig, RedshiftConnectionConfig
from threetears.datasources.drivers import (
    ConnectGuard,
    CredentialRefusalGuards,
    DriverAuthError,
    DriverConnectError,
    DriverCredentialPausedError,
    DriverMissingCredentialError,
    create_driver,
    guarded_connect,
)
from threetears.datasources.drivers.asyncpg_driver import AsyncpgDriver
from threetears.datasources.drivers.redshift_driver import RedshiftDriver
from threetears.datasources.entities import DataSourceType
from threetears.nats import KvError

_PASSWORD_ENV = "TEST_CONNECT_GUARD_PW"
_REVISION = "rev-1"
_PASSWORD = "connect-guard-test-password"
_SCOPE = "connect-guard-test"


@pytest.fixture(autouse=True)
def _password(monkeypatch: pytest.MonkeyPatch) -> None:
    """make every config's password reference resolvable.

    :param monkeypatch: pytest's environment patcher
    :ptype monkeypatch: pytest.MonkeyPatch
    :return: None
    :rtype: None
    """
    monkeypatch.setenv(_PASSWORD_ENV, _PASSWORD)


class _Nats(FakeNatsClient):
    """the shared collections bucket every replica's registry reaches."""

    async def publish(self, *, subject: Any, message: Any, reply_to: Any = None) -> None:
        return None


class _FailingNats(FakeNatsClient):
    """a bucket that refuses every operation, the way an unreachable broker does."""

    async def kv_bucket(self, **kwargs: Any) -> Any:
        del kwargs
        raise KvError("kv down")

    async def publish(self, *, subject: Any, message: Any, reply_to: Any = None) -> None:
        return None


def _replica(nats: FakeNatsClient) -> CredentialRefusalGuards:
    """one replica's guards: its own L1, the fleet's shared L2.

    :param nats: the shared bucket
    :ptype nats: FakeNatsClient
    :return: the replica's guards
    :rtype: CredentialRefusalGuards
    """
    registry = CollectionRegistry()
    registry.configure(
        l1_backend=SQLiteBackend(db_name=f"guard_{uuid.uuid4().hex[:8]}"),
        l2_client=nats,
        kv_key_scope=_SCOPE,
    )
    return CredentialRefusalGuards(registry)


def _redshift_config() -> RedshiftConnectionConfig:
    """a Redshift config whose password resolves.

    :return: the config
    :rtype: RedshiftConnectionConfig
    """
    return RedshiftConnectionConfig(
        datasource_type=DataSourceType.REDSHIFT,
        host="rs.example.com",
        port=5439,
        database="analytics",
        username="ripple_build",
        password_ref=f"env://{_PASSWORD_ENV}",
        executor_max_workers=2,
        connection_cache_size=2,
        query_timeout_seconds=60,
    )


def _refused() -> redshift_connector.InterfaceError:
    """the shape ``redshift_connector`` raises for a server refusal with SQLSTATE 28000.

    :return: the backend exception
    :rtype: redshift_connector.InterfaceError
    """
    return redshift_connector.InterfaceError(
        {"S": "FATAL", "C": "28000", "M": 'password authentication failed for user "ripple_build"'}
    )


class TestOneRefusalPausesTheFleet:
    """the pause lives in the shared tier, so every replica honours it."""

    async def test_a_refusal_on_one_replica_pauses_the_other(self) -> None:
        nats = _Nats()
        datasource_id = uuid.uuid4()
        here = _replica(nats).for_credential(
            datasource_id, credential_revision=_REVISION, datasource_name="influencers-build"
        )
        there = _replica(nats).for_credential(
            datasource_id, credential_revision=_REVISION, datasource_name="influencers-build"
        )

        await here.record_refusal(DriverAuthError("refused", sqlstate="28000"))

        with pytest.raises(DriverCredentialPausedError) as exc_info:
            await there.admit()
        assert "influencers-build" in str(exc_info.value)
        assert exc_info.value.refused_at is not None

    async def test_another_datasource_is_not_paused(self) -> None:
        guards = _replica(_Nats())
        await guards.for_credential(uuid.uuid4(), credential_revision=_REVISION, datasource_name="a").record_refusal(
            DriverAuthError("refused")
        )

        await guards.for_credential(uuid.uuid4(), credential_revision=_REVISION, datasource_name="b").admit()

    async def test_clearing_lets_the_next_login_through(self) -> None:
        """a clear drops the shared row; another replica's cached copy goes with the
        registry's invalidation listener, as for every coordination primitive."""
        datasource_id = uuid.uuid4()
        guards = _replica(_Nats())
        guard = guards.for_credential(datasource_id, credential_revision=_REVISION, datasource_name="ds")
        await guard.record_refusal(DriverAuthError("refused"))

        await guards.clear(datasource_id, _REVISION)

        await guard.admit()
        assert await guards.refused_at(datasource_id, _REVISION) is None

    async def test_an_unreadable_store_lets_the_connect_go_ahead(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """fail-open: a storage outage must not become an outage of every datasource.

        the read raises a storage failure the counter itself sees -- an L2 miss would not do,
        because the collection already reads a KvError as a miss, and the test would then pass
        with the guard failing closed.
        """

        async def _unreadable(self: CoordinationCountersCollection, *args: Any, **kwargs: Any) -> Any:
            raise DataLayerUnavailableError("l3 down")

        monkeypatch.setattr(CoordinationCountersCollection, "get", _unreadable)
        guard = _replica(_Nats()).for_credential(uuid.uuid4(), credential_revision=_REVISION, datasource_name="ds")

        await guard.admit()

    async def test_an_unwritable_store_never_replaces_the_refusal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """the refusal is what tells the caller to stop; a storage error in its place is retried."""

        async def _unwritable(self: CoordinationCountersCollection, *args: Any, **kwargs: Any) -> Any:
            raise KvError("kv down")

        monkeypatch.setattr(CoordinationCountersCollection, "l2_cas_mutate", _unwritable)
        guard = _replica(_Nats()).for_credential(uuid.uuid4(), credential_revision=_REVISION, datasource_name="ds")
        refusal = DriverAuthError("refused", sqlstate="28000")

        with pytest.raises(DriverAuthError) as exc_info:
            await guarded_connect(guard, AsyncMock(side_effect=refusal))

        assert exc_info.value is refusal


class _ScriptedGuard(ConnectGuard):
    """a guard whose admission is scripted and whose calls are recorded."""

    def __init__(self, *, admit_error: Exception | None = None) -> None:
        """script the admission.

        :param admit_error: what :meth:`admit` raises, or ``None`` to admit
        :ptype admit_error: Exception | None
        """
        self.admit_error = admit_error
        self.refusals: list[DriverAuthError] = []
        self.slots_entered = 0

    def serialized(self) -> AbstractAsyncContextManager[None]:
        """count the slot and hold nothing.

        :return: the slot
        :rtype: AbstractAsyncContextManager[None]
        """

        @asynccontextmanager
        async def _slot() -> AsyncIterator[None]:
            self.slots_entered += 1
            yield

        return _slot()

    async def admit(self) -> None:
        """admit, or raise the scripted error.

        :return: nothing
        :rtype: None
        """
        if self.admit_error is not None:
            raise self.admit_error

    async def record_refusal(self, error: DriverAuthError) -> None:
        """record the refusal.

        :param error: the refusal
        :ptype error: DriverAuthError
        :return: nothing
        :rtype: None
        """
        self.refusals.append(error)


class TestGuardedConnect:
    """the one place a login meets its guard."""

    async def test_a_paused_credential_never_reaches_the_login(self) -> None:
        guard = _ScriptedGuard(admit_error=DriverCredentialPausedError("paused", refused_at=datetime.now(UTC)))
        login = AsyncMock()

        with pytest.raises(DriverCredentialPausedError):
            await guarded_connect(guard, login)

        login.assert_not_awaited()

    async def test_a_refusal_is_recorded_then_raised(self) -> None:
        guard = _ScriptedGuard()
        refusal = DriverAuthError("refused", sqlstate="28000")

        with pytest.raises(DriverAuthError):
            await guarded_connect(guard, AsyncMock(side_effect=refusal))

        assert guard.refusals == [refusal]

    async def test_the_login_runs_in_its_slot(self) -> None:
        guard = _ScriptedGuard()

        assert await guarded_connect(guard, AsyncMock(return_value="connection")) == "connection"

        assert guard.slots_entered == 1

    @pytest.mark.parametrize(
        "error",
        [
            pytest.param(DriverMissingCredentialError("no password"), id="missing-credential-no-login-happened"),
            pytest.param(
                DriverCredentialPausedError("paused", refused_at=datetime.now(UTC)),
                id="already-paused-no-login-happened",
            ),
            pytest.param(DriverConnectError("host unreachable"), id="not-a-refusal"),
        ],
    )
    async def test_only_a_server_refusal_pauses(self, error: DriverConnectError) -> None:
        guard = _ScriptedGuard()

        with pytest.raises(type(error)):
            await guarded_connect(guard, AsyncMock(side_effect=error))

        assert guard.refusals == []

    async def test_no_guard_is_a_plain_login(self) -> None:
        login = AsyncMock(return_value="connection")

        assert await guarded_connect(None, login) == "connection"


class TestThePauseBelongsToACredential:
    """a password change must not be undone by a holder of the password it replaced."""

    async def test_a_stale_holder_cannot_pause_the_replacement(self) -> None:
        """a pool still holding the old password is refused after the new one was stored."""
        nats = _Nats()
        datasource_id = uuid.uuid4()
        stale = _replica(nats).for_credential(datasource_id, credential_revision="old", datasource_name="ds")
        current = _replica(nats).for_credential(datasource_id, credential_revision="new", datasource_name="ds")

        await stale.record_refusal(DriverAuthError("refused", sqlstate="28000"))

        await current.admit()
        with pytest.raises(DriverCredentialPausedError):
            await stale.admit()

    async def test_replacing_a_refused_credential_needs_no_clear(self) -> None:
        guards = _replica(_Nats())
        datasource_id = uuid.uuid4()
        await guards.record(datasource_id, "wrong")

        await guards.for_credential(datasource_id, credential_revision="fixed", datasource_name="ds").admit()

        assert await guards.refused_at(datasource_id, "wrong") is not None
        assert await guards.refused_at(datasource_id, "fixed") is None

    def test_a_guard_without_a_revision_is_refused(self) -> None:
        """an empty revision would put every credential the datasource ever has under one pause."""
        with pytest.raises(ValueError, match="revision"):
            _replica(_Nats()).for_credential(uuid.uuid4(), credential_revision="", datasource_name="ds")


class TestLoginsGoOneAtATime:
    """every login with a credential waits its turn, so a burst against a refusing warehouse costs one."""

    async def test_a_burst_of_cold_logins_against_a_refusing_warehouse_costs_one(self) -> None:
        """five at once is a Redshift lock on its own, before the first refusal could be recorded."""
        datasource_id = uuid.uuid4()
        guards = _replica(_Nats())
        driver = RedshiftDriver(
            _redshift_config(),
            datasource_name="ds",
            connect_guard=guards.for_credential(datasource_id, credential_revision=_REVISION, datasource_name="ds"),
        )
        try:
            with patch(
                "threetears.datasources.drivers.redshift_driver.redshift_connector.connect",
                side_effect=_refused(),
            ) as connect:
                outcomes = await asyncio.gather(
                    *(driver.test_connection() for _ in range(5)),
                    return_exceptions=True,
                )
        finally:
            await driver.close()

        assert connect.call_count == 1
        assert sorted(type(o).__name__ for o in outcomes) == ["DriverAuthError"] + ["DriverCredentialPausedError"] * 4

    async def test_logins_wait_their_turn_even_after_a_success(self) -> None:
        """a credential that worked is not trusted to keep working: its logins still queue."""
        guard = _replica(_Nats()).for_credential(uuid.uuid4(), credential_revision=_REVISION, datasource_name="ds")
        await guarded_connect(guard, AsyncMock(return_value="first"))
        inside = 0
        most_at_once = 0

        async def _login() -> str:
            nonlocal inside, most_at_once
            inside += 1
            most_at_once = max(most_at_once, inside)
            await asyncio.sleep(0.01)
            inside -= 1
            return "connection"

        results = await asyncio.gather(*(guarded_connect(guard, _login) for _ in range(5)))

        assert results == ["connection"] * 5
        assert most_at_once == 1

    async def test_a_password_changed_on_the_warehouse_costs_one_login(self) -> None:
        """the gap this closes: a login succeeded, then someone changed the password on the warehouse.

        nothing here learns of that change until a login is refused. were logins trusted after
        the earlier success, the next burst -- a pool refilling, five queries at once -- would send
        the old password five times, and five failures is a Redshift lock.
        """
        guard = _replica(_Nats()).for_credential(uuid.uuid4(), credential_revision=_REVISION, datasource_name="ds")
        await guarded_connect(guard, AsyncMock(return_value="first"))

        async def _refused_login() -> str:
            await asyncio.sleep(0.01)
            raise DriverAuthError("refused", sqlstate="28000")

        login = AsyncMock(side_effect=_refused_login)

        results = await asyncio.gather(*(guarded_connect(guard, login) for _ in range(5)), return_exceptions=True)

        assert login.await_count == 1
        assert sum(isinstance(r, DriverCredentialPausedError) for r in results) == 4


class TestARefusingWarehouseCostsOneLogin:
    """the #523 acceptance: passes on two replicas against a refusing warehouse, one login."""

    async def test_every_pass_after_the_first_is_refused_without_a_login(self) -> None:
        nats = _Nats()
        datasource_id = uuid.uuid4()
        replicas = [_replica(nats), _replica(nats)]
        outcomes: list[type[DriverAuthError]] = []
        with patch(
            "threetears.datasources.drivers.redshift_driver.redshift_connector.connect",
            side_effect=_refused(),
        ) as connect:
            for guards in replicas * 3:
                driver = RedshiftDriver(
                    _redshift_config(),
                    datasource_name="influencers-build",
                    connect_guard=guards.for_credential(
                        datasource_id, credential_revision=_REVISION, datasource_name="influencers-build"
                    ),
                )
                try:
                    with pytest.raises(DriverAuthError) as exc_info:
                        await driver.test_connection()
                    outcomes.append(type(exc_info.value))
                finally:
                    await driver.close()

        assert connect.call_count == 1
        assert outcomes == [DriverAuthError] + [DriverCredentialPausedError] * 5

    async def test_a_probe_logs_in_while_paused_and_a_clear_resumes(self) -> None:
        nats = _Nats()
        datasource_id = uuid.uuid4()
        guards = _replica(nats)
        await guards.record(datasource_id, _REVISION)
        connection = MagicMock()
        cursor = MagicMock()
        cursor.fetchone.return_value = (1,)
        connection.cursor.return_value = cursor

        with patch(
            "threetears.datasources.drivers.redshift_driver.redshift_connector.connect",
            return_value=connection,
        ) as connect:
            probe = RedshiftDriver(_redshift_config(), datasource_name="ds")
            try:
                await probe.test_connection()
            finally:
                await probe.close()
            assert connect.call_count == 1

            guarded = RedshiftDriver(
                _redshift_config(),
                datasource_name="ds",
                connect_guard=guards.for_credential(datasource_id, credential_revision=_REVISION, datasource_name="ds"),
            )
            try:
                with pytest.raises(DriverCredentialPausedError):
                    await guarded.test_connection()
                await guards.clear(datasource_id, _REVISION)
                await guarded.test_connection()
            finally:
                await guarded.close()
            assert connect.call_count == 2


class TestAsyncpgGuardsEveryPooledLogin:
    """the pool's own logins -- its first and every replacement -- go through the guard."""

    def _driver(self, guards: CredentialRefusalGuards, datasource_id: UUID) -> AsyncpgDriver:
        """a Postgres driver guarded for one datasource.

        :param guards: the replica's guards
        :ptype guards: CredentialRefusalGuards
        :param datasource_id: the datasource
        :ptype datasource_id: UUID
        :return: the driver
        :rtype: AsyncpgDriver
        """
        config = PostgresConnectionConfig(
            datasource_type=DataSourceType.POSTGRES,
            host="pg.example.com",
            database="warehouse",
            username="reader",
            password_ref=f"env://{_PASSWORD_ENV}",
        )
        return AsyncpgDriver(
            config,
            datasource_name="pg",
            connect_guard=guards.for_credential(datasource_id, credential_revision=_REVISION, datasource_name="pg"),
        )

    async def test_the_pool_logs_in_through_the_guard(self, monkeypatch: pytest.MonkeyPatch) -> None:
        create_pool = AsyncMock(return_value=MagicMock())
        monkeypatch.setattr("threetears.datasources.drivers.asyncpg_driver.asyncpg.create_pool", create_pool)
        driver = self._driver(_replica(_Nats()), uuid.uuid4())

        await driver._ensure_pool()  # noqa: SLF001 -- the pool's construction arguments are the contract

        assert create_pool.await_args.kwargs["connect"] == driver._connect_one  # noqa: SLF001

    async def test_concurrent_first_callers_share_one_pool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """each first caller building its own pool was one login apiece, and a leaked pool apiece."""
        created = asyncio.Event()

        async def _slow_create_pool(**kwargs: Any) -> MagicMock:
            del kwargs
            await created.wait()
            return MagicMock()

        create_pool = AsyncMock(side_effect=_slow_create_pool)
        monkeypatch.setattr("threetears.datasources.drivers.asyncpg_driver.asyncpg.create_pool", create_pool)
        driver = self._driver(_replica(_Nats()), uuid.uuid4())

        callers = [asyncio.create_task(driver._ensure_pool()) for _ in range(5)]  # noqa: SLF001 -- pool creation is the contract
        await asyncio.sleep(0)
        created.set()
        pools = await asyncio.gather(*callers)

        assert create_pool.await_count == 1
        assert all(pool is pools[0] for pool in pools)

    async def test_a_refused_pooled_login_pauses_the_next(self, monkeypatch: pytest.MonkeyPatch) -> None:
        connect = AsyncMock(side_effect=asyncpg.exceptions.InvalidPasswordError("password authentication failed"))
        monkeypatch.setattr("threetears.datasources.drivers.asyncpg_driver.asyncpg.connect", connect)
        driver = self._driver(_replica(_Nats()), uuid.uuid4())
        login = {"host": "pg.example.com", "port": 5432, "database": "warehouse", "password": _PASSWORD}

        with pytest.raises(DriverAuthError) as exc_info:
            await driver._connect_one(**login)  # noqa: SLF001 -- the pool's per-login hook
        with pytest.raises(DriverCredentialPausedError):
            await driver._connect_one(**login)  # noqa: SLF001

        assert connect.await_count == 1
        assert _PASSWORD not in str(exc_info.value)


class TestTheCancelPathLogsInUnderTheGuard:
    """cancelling a Redshift query opens a fresh login to terminate its backend; that login is guarded too."""

    async def test_a_paused_credential_skips_the_terminate_login(self) -> None:
        datasource_id = uuid.uuid4()
        guards = _replica(_Nats())
        await guards.record(datasource_id, _REVISION)
        driver = RedshiftDriver(
            _redshift_config(),
            datasource_name="ds",
            connect_guard=guards.for_credential(datasource_id, credential_revision=_REVISION, datasource_name="ds"),
        )
        try:
            with patch("threetears.datasources.drivers.redshift_driver.redshift_connector.connect") as connect:
                await driver._terminate_backend(4242)  # noqa: SLF001 -- the cancel path's login is the contract
        finally:
            await driver.close()

        connect.assert_not_called()

    async def test_a_refused_terminate_login_pauses_the_credential(self) -> None:
        datasource_id = uuid.uuid4()
        guards = _replica(_Nats())
        driver = RedshiftDriver(
            _redshift_config(),
            datasource_name="ds",
            connect_guard=guards.for_credential(datasource_id, credential_revision=_REVISION, datasource_name="ds"),
        )
        try:
            with patch(
                "threetears.datasources.drivers.redshift_driver.redshift_connector.connect",
                side_effect=_refused(),
            ):
                await driver._terminate_backend(4242)  # noqa: SLF001 -- never raises; the pause is the effect
        finally:
            await driver.close()

        assert await guards.refused_at(datasource_id, _REVISION) is not None


class TestNoLoginIsLeftOpen:
    """a connection opened by a login nobody is waiting for any more is closed, not dropped.

    a login cannot be interrupted once its worker thread starts, so a caller that
    gives up mid-login -- a cancelled query, a cancel path that timed out -- leaves
    the thread to finish and open a connection with nothing holding it. one of
    those held a production pool slot for hours.
    """

    async def test_the_cancel_login_uses_the_same_connect_settings_as_every_login(self) -> None:
        """the cancel path's own copy of the login once dropped sslmode, and failed on verify-full."""
        driver = RedshiftDriver(_redshift_config(), datasource_name="ds")
        connection = MagicMock()
        try:
            with patch(
                "threetears.datasources.drivers.redshift_driver.redshift_connector.connect",
                return_value=connection,
            ) as connect:
                await driver._terminate_backend(4242)  # noqa: SLF001 -- the cancel path's login is the contract
        finally:
            await driver.close()

        assert connect.call_args.kwargs["sslmode"] == _redshift_config().sslmode
        connection.close.assert_called_once()

    async def test_a_terminate_that_outlasts_its_timeout_still_closes_its_connection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("threetears.datasources.drivers.redshift_driver._CANCEL_TIMEOUT_SECONDS", 0.05)
        driver = RedshiftDriver(_redshift_config(), datasource_name="ds")
        connection = MagicMock()
        closed = asyncio.Event()
        loop = asyncio.get_running_loop()
        connection.close.side_effect = lambda: loop.call_soon_threadsafe(closed.set)

        def _slow_login(**kwargs: Any) -> MagicMock:
            del kwargs
            import time

            time.sleep(0.2)
            return connection

        try:
            with patch(
                "threetears.datasources.drivers.redshift_driver.redshift_connector.connect",
                side_effect=_slow_login,
            ):
                await driver._terminate_backend(4242)  # noqa: SLF001 -- times out; the close is the effect
                async with asyncio.timeout(2.0):
                    await closed.wait()
        finally:
            await driver.close()

        connection.close.assert_called_once()

    async def test_a_caller_cancelled_mid_login_does_not_strand_the_connection(self) -> None:
        driver = RedshiftDriver(_redshift_config(), datasource_name="ds")
        connection = MagicMock()
        cursor = MagicMock()
        cursor.fetchone.return_value = (1,)
        connection.cursor.return_value = cursor
        login_started = asyncio.Event()
        closed = asyncio.Event()
        loop = asyncio.get_running_loop()
        connection.close.side_effect = lambda: loop.call_soon_threadsafe(closed.set)

        def _slow_login(**kwargs: Any) -> MagicMock:
            del kwargs
            import time

            loop.call_soon_threadsafe(login_started.set)
            time.sleep(0.2)
            return connection

        try:
            with patch(
                "threetears.datasources.drivers.redshift_driver.redshift_connector.connect",
                side_effect=_slow_login,
            ):
                acquiring = asyncio.create_task(driver._acquire_connection())  # noqa: SLF001 -- the login path under test
                await login_started.wait()
                acquiring.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await acquiring
                async with asyncio.timeout(2.0):
                    await closed.wait()
        finally:
            await driver.close()

        connection.close.assert_called_once()


class TestTheFactoryHandsTheGuardOn:
    """``create_driver`` passes a guard to the drivers that honour one, and says when it cannot."""

    def test_a_redshift_driver_carries_it(self) -> None:
        guard = MagicMock()

        driver = create_driver(_redshift_config(), datasource_name="ds", connect_guard=guard)

        assert driver._connect_guard is guard  # noqa: SLF001 -- what the factory wired is the contract
