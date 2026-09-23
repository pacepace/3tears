"""a credential the warehouse refused is not sent again, by any replica.

the contract (hub issue #523, the lockout it closes):

- one refusal pauses the credential for EVERY replica sharing the coordination tier,
  and a paused connect raises :class:`DriverCredentialPausedError` without a login;
- several background passes against a refusing warehouse cost exactly one login;
- a missing credential, refused before any login, pauses nothing;
- clearing the pause (a new credential, a successful probe) lets the next login through;
- a probe -- a driver built with no guard -- logs in even while the credential is paused;
- a coordination store that cannot be read lets the connect go ahead (fail-open);
- both drivers that classify a refusal consult the guard: Redshift on every fresh
  connection, asyncpg on every login its pool makes.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import asyncpg
import pytest
import redshift_connector

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.testing.kv import FakeNatsClient
from threetears.datasources.config import PostgresConnectionConfig, RedshiftConnectionConfig
from threetears.datasources.drivers import (
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
        here = _replica(nats).for_datasource(datasource_id, datasource_name="influencers-build")
        there = _replica(nats).for_datasource(datasource_id, datasource_name="influencers-build")

        await here.record_refusal(DriverAuthError("refused", sqlstate="28000"))

        with pytest.raises(DriverCredentialPausedError) as exc_info:
            await there.admit()
        assert "influencers-build" in str(exc_info.value)
        assert exc_info.value.refused_at is not None

    async def test_another_datasource_is_not_paused(self) -> None:
        guards = _replica(_Nats())
        await guards.for_datasource(uuid.uuid4(), datasource_name="a").record_refusal(DriverAuthError("refused"))

        await guards.for_datasource(uuid.uuid4(), datasource_name="b").admit()

    async def test_clearing_lets_the_next_login_through(self) -> None:
        """a clear drops the shared row; another replica's cached copy goes with the
        registry's invalidation listener, as for every coordination primitive."""
        datasource_id = uuid.uuid4()
        guards = _replica(_Nats())
        guard = guards.for_datasource(datasource_id, datasource_name="ds")
        await guard.record_refusal(DriverAuthError("refused"))

        await guards.clear(datasource_id)

        await guard.admit()
        assert await guards.refused_at(datasource_id) is None

    async def test_an_unreadable_store_lets_the_connect_go_ahead(self) -> None:
        """fail-open: a storage outage must not become an outage of every datasource."""
        registry = CollectionRegistry()
        registry.configure(
            l1_backend=SQLiteBackend(db_name=f"guard_{uuid.uuid4().hex[:8]}"),
            l2_client=_FailingNats(),
            kv_key_scope=_SCOPE,
        )
        guard = CredentialRefusalGuards(registry).for_datasource(uuid.uuid4(), datasource_name="ds")

        await guard.admit()


class TestGuardedConnect:
    """the one place a login meets its guard."""

    async def test_a_paused_credential_never_reaches_the_login(self) -> None:
        guard = MagicMock()
        guard.admit = AsyncMock(side_effect=DriverCredentialPausedError("paused", refused_at=MagicMock()))
        login = AsyncMock()

        with pytest.raises(DriverCredentialPausedError):
            await guarded_connect(guard, login)

        login.assert_not_awaited()

    async def test_a_refusal_is_recorded_then_raised(self) -> None:
        guard = MagicMock()
        guard.admit = AsyncMock()
        guard.record_refusal = AsyncMock()
        refusal = DriverAuthError("refused", sqlstate="28000")

        with pytest.raises(DriverAuthError):
            await guarded_connect(guard, AsyncMock(side_effect=refusal))

        guard.record_refusal.assert_awaited_once_with(refusal)

    @pytest.mark.parametrize(
        "error",
        [
            pytest.param(DriverMissingCredentialError("no password"), id="missing-credential-no-login-happened"),
            pytest.param(
                DriverCredentialPausedError("paused", refused_at=MagicMock()), id="already-paused-no-login-happened"
            ),
            pytest.param(DriverConnectError("host unreachable"), id="not-a-refusal"),
        ],
    )
    async def test_only_a_server_refusal_pauses(self, error: DriverConnectError) -> None:
        guard = MagicMock()
        guard.admit = AsyncMock()
        guard.record_refusal = AsyncMock()

        with pytest.raises(type(error)):
            await guarded_connect(guard, AsyncMock(side_effect=error))

        guard.record_refusal.assert_not_awaited()

    async def test_no_guard_is_a_plain_login(self) -> None:
        login = AsyncMock(return_value="connection")

        assert await guarded_connect(None, login) == "connection"


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
                    connect_guard=guards.for_datasource(datasource_id, datasource_name="influencers-build"),
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
        await guards.record(datasource_id)
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
                connect_guard=guards.for_datasource(datasource_id, datasource_name="ds"),
            )
            try:
                with pytest.raises(DriverCredentialPausedError):
                    await guarded.test_connection()
                await guards.clear(datasource_id)
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
            connect_guard=guards.for_datasource(datasource_id, datasource_name="pg"),
        )

    async def test_the_pool_logs_in_through_the_guard(self, monkeypatch: pytest.MonkeyPatch) -> None:
        create_pool = AsyncMock(return_value=MagicMock())
        monkeypatch.setattr("threetears.datasources.drivers.asyncpg_driver.asyncpg.create_pool", create_pool)
        driver = self._driver(_replica(_Nats()), uuid.uuid4())

        await driver._ensure_pool()  # noqa: SLF001 -- the pool's construction arguments are the contract

        assert create_pool.await_args.kwargs["connect"] == driver._connect_one  # noqa: SLF001

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


class TestTheFactoryHandsTheGuardOn:
    """``create_driver`` passes a guard to the drivers that honour one, and says when it cannot."""

    def test_a_redshift_driver_carries_it(self) -> None:
        guard = MagicMock()

        driver = create_driver(_redshift_config(), datasource_name="ds", connect_guard=guard)

        assert driver._connect_guard is guard  # noqa: SLF001 -- what the factory wired is the contract
