"""unit tests for :mod:`threetears.mcp.auth` -- identity + authorizer."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from threetears.mcp.auth import (
    EnvVarIdentityProvider,
    Identity,
    LocalGrantAuthorizer,
)


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def _make_listener_capturing_subscribe() -> tuple[Any, Any, list[Any]]:
    """build a fake EpochClient + EpochListener; capture subscribe callback."""
    fake_client = MagicMock()
    fake_client.current = AsyncMock(return_value=0)

    fake_listener = MagicMock()
    captured: list[Any] = []

    async def _subscribe(subject: Any, on_bump: Any, primed_epoch: Any = None) -> None:  # noqa: ARG001
        captured.append(on_bump)

    fake_listener.subscribe = AsyncMock(side_effect=_subscribe)
    fake_listener.catch_up = AsyncMock(return_value=0)
    return fake_client, fake_listener, captured


async def _build_started_authorizer(
    *,
    loader: AsyncMock,
    admin_principal_ids: set[UUID] | None = None,
) -> tuple[LocalGrantAuthorizer, list[Any]]:
    """construct + start a LocalGrantAuthorizer; return (authz, captured_callbacks).

    every test that uses this helper MUST ``await authz.stop()`` at
    teardown so the spawned catch-up task is cancelled. catchup
    interval is set to a very large value so the task never fires
    during the test (deterministic).
    """
    client, listener, captured = _make_listener_capturing_subscribe()
    authz = LocalGrantAuthorizer(
        grant_loader=loader,
        epoch_client=client,
        epoch_listener=listener,
        admin_principal_ids=admin_principal_ids,
        catchup_interval_seconds=3600.0,
    )
    await authz.start()
    return authz, captured


# ---------------------------------------------------------------------
# EnvVarIdentityProvider
# ---------------------------------------------------------------------


class TestEnvVarIdentityProvider:
    """env-var-derived identity for v1 stdio mode."""

    @pytest.mark.asyncio
    async def test_explicit_principal_id_bypasses_env(self) -> None:
        """when ``principal_id`` is set the env var is never read."""
        target = uuid4()
        provider = EnvVarIdentityProvider(principal_id=target)
        identity = await provider.identify()
        assert identity.principal_id == target
        assert identity.principal_type == "user"

    @pytest.mark.asyncio
    async def test_is_admin_defaults_deny(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """is_admin defaults False: a wirer that omits the flag gets a
        non-admin identity, not a total RBAC bypass. admin authority is
        granted only when the wirer passes is_admin=True explicitly."""
        target = uuid4()
        monkeypatch.setenv("MCP_ADMIN_USER_ID", str(target))
        assert (await EnvVarIdentityProvider().identify()).is_admin is False
        assert (await EnvVarIdentityProvider(principal_id=target).identify()).is_admin is False

    @pytest.mark.asyncio
    async def test_is_admin_granted_when_explicit(self) -> None:
        """admin authority is honoured when the wirer opts in explicitly."""
        provider = EnvVarIdentityProvider(principal_id=uuid4(), is_admin=True)
        identity = await provider.identify()
        assert identity.is_admin is True

    @pytest.mark.asyncio
    async def test_env_var_provides_principal_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """env var carrying a UUID populates principal_id."""
        target = uuid4()
        monkeypatch.setenv("MCP_ADMIN_USER_ID", str(target))
        provider = EnvVarIdentityProvider()
        identity = await provider.identify()
        assert identity.principal_id == target

    @pytest.mark.asyncio
    async def test_missing_env_var_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """no env var + no explicit principal_id raises a clear error."""
        monkeypatch.delenv("MCP_ADMIN_USER_ID", raising=False)
        provider = EnvVarIdentityProvider()
        with pytest.raises(RuntimeError, match="MCP_ADMIN_USER_ID"):
            await provider.identify()

    @pytest.mark.asyncio
    async def test_invalid_env_var_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """malformed env var value raises a UUID-parse error."""
        monkeypatch.setenv("MCP_ADMIN_USER_ID", "not-a-uuid")
        provider = EnvVarIdentityProvider()
        with pytest.raises(RuntimeError, match="valid UUID"):
            await provider.identify()

    @pytest.mark.asyncio
    async def test_is_admin_flag_explicitly_disable(self) -> None:
        """non-admin identity construction works for future bearer-token paths."""
        provider = EnvVarIdentityProvider(principal_id=uuid4(), is_admin=False)
        identity = await provider.identify()
        assert identity.is_admin is False


# ---------------------------------------------------------------------
# LocalGrantAuthorizer
# ---------------------------------------------------------------------


class TestLocalGrantAuthorizer:
    """default-deny + admin auto-grant + cache reload on epoch bump."""

    @pytest.mark.asyncio
    async def test_admin_short_circuit_allows_without_grant(self) -> None:
        """is_admin=True identity allows every permission without DB lookup."""
        loader = AsyncMock(return_value=[])
        authz, _ = await _build_started_authorizer(loader=loader)
        try:
            identity = Identity(principal_type="user", principal_id=uuid4(), is_admin=True)
            assert await authz.allows(identity, "anything.at.all") is True
            loader.assert_awaited_once()  # called by start() reload
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_default_deny_when_no_grant(self) -> None:
        """non-admin identity with no matching grant is denied."""
        loader = AsyncMock(return_value=[])
        authz, _ = await _build_started_authorizer(loader=loader)
        try:
            identity = Identity(principal_type="user", principal_id=uuid4())
            assert await authz.allows(identity, "product.conv.read") is False
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_principal_grant_allows(self) -> None:
        """direct principal_id+permission grant authorizes the call."""
        principal_id = uuid4()
        loader = AsyncMock(
            return_value=[
                {"principal_id": principal_id, "permission": "product.conv.read"},
            ]
        )
        authz, _ = await _build_started_authorizer(loader=loader)
        try:
            identity = Identity(principal_type="user", principal_id=principal_id)
            assert await authz.allows(identity, "product.conv.read") is True
            assert await authz.allows(identity, "product.conv.write") is False
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_group_grant_allows_via_membership(self) -> None:
        """grant on a group_id allows when identity carries that group."""
        group_id = uuid4()
        loader = AsyncMock(
            return_value=[
                {"principal_id": group_id, "permission": "audit.records.read"},
            ]
        )
        authz, _ = await _build_started_authorizer(loader=loader)
        try:
            identity = Identity(
                principal_type="user",
                principal_id=uuid4(),
                groups=frozenset({group_id}),
            )
            assert await authz.allows(identity, "audit.records.read") is True
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_role_grant_allows_via_assignment(self) -> None:
        """grant on a role_id allows when identity carries that role."""
        role_id = uuid4()
        loader = AsyncMock(
            return_value=[
                {"principal_id": role_id, "permission": "audit.records.read"},
            ]
        )
        authz, _ = await _build_started_authorizer(loader=loader)
        try:
            identity = Identity(
                principal_type="user",
                principal_id=uuid4(),
                roles=frozenset({role_id}),
            )
            assert await authz.allows(identity, "audit.records.read") is True
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_start_subscribes_to_rbac_epoch(self) -> None:
        """start() subscribes the listener to Subjects.mcp_rbac_epoch."""
        loader = AsyncMock(return_value=[])
        client, listener, _ = _make_listener_capturing_subscribe()
        authz = LocalGrantAuthorizer(
            grant_loader=loader,
            epoch_client=client,
            epoch_listener=listener,
            catchup_interval_seconds=3600.0,
        )
        await authz.start()
        try:
            listener.subscribe.assert_awaited_once()
            from threetears.nats import Subjects

            called_subject = listener.subscribe.await_args.args[0]
            assert called_subject == Subjects.mcp_rbac_epoch()
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_start_primes_listener_to_epoch_read_before_cache_load(self) -> None:
        """start() reads current() BEFORE reloading, primes to that epoch.

        guards the permanent-staleness race: if the listener primed
        last-seen via current()-at-subscribe (AFTER the cache load), a
        bump landing in the load->subscribe window would pin last-seen
        PAST the loaded grants and the catch-up tick (current ==
        last_seen) would never recover. priming to the epoch read
        BEFORE the load keeps last-seen <= the cache's epoch.
        """
        order: list[str] = []

        async def _current(subject: Any) -> int:  # noqa: ARG001
            order.append("current")
            return 5

        async def _loader() -> list[dict[str, Any]]:
            order.append("reload")
            return []

        client, listener, _ = _make_listener_capturing_subscribe()
        client.current = AsyncMock(side_effect=_current)
        authz = LocalGrantAuthorizer(
            grant_loader=AsyncMock(side_effect=_loader),
            epoch_client=client,
            epoch_listener=listener,
            catchup_interval_seconds=3600.0,
        )
        await authz.start()
        try:
            # epoch read happened strictly before the cache load
            assert order == ["current", "reload"]
            # listener primed to the epoch the loaded cache reflects
            assert listener.subscribe.await_args.kwargs["primed_epoch"] == 5
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_epoch_bump_reloads_cache(self) -> None:
        """on rbac bump, reload pulls current grants and replaces cache."""
        principal_id = uuid4()
        loader = AsyncMock(
            side_effect=[
                [],
                [{"principal_id": principal_id, "permission": "product.conv.read"}],
            ]
        )
        authz, captured = await _build_started_authorizer(loader=loader)
        try:
            identity = Identity(principal_type="user", principal_id=principal_id)
            assert await authz.allows(identity, "product.conv.read") is False
            await captured[0](7, {"hint": "added grant"})
            assert await authz.allows(identity, "product.conv.read") is True
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_loader_failure_keeps_prior_cache(self) -> None:
        """transient L3 hiccup logs + leaves prior cache in place."""
        principal_id = uuid4()
        loader = AsyncMock(
            side_effect=[
                [{"principal_id": principal_id, "permission": "audit.records.read"}],
                RuntimeError("L3 down"),
            ]
        )
        authz, captured = await _build_started_authorizer(loader=loader)
        try:
            identity = Identity(principal_type="user", principal_id=principal_id)
            assert await authz.allows(identity, "audit.records.read") is True
            await captured[0](2, None)
            assert await authz.allows(identity, "audit.records.read") is True
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_uuid_string_in_loader_row_is_normalised(self) -> None:
        """rows with stringified UUIDs (e.g. asyncpg .Record edge) match."""
        principal_id = uuid4()
        loader = AsyncMock(
            return_value=[
                {"principal_id": str(principal_id), "permission": "product.conv.read"},
            ]
        )
        authz, _ = await _build_started_authorizer(loader=loader)
        try:
            identity = Identity(principal_type="user", principal_id=principal_id)
            assert await authz.allows(identity, "product.conv.read") is True
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_double_start_is_no_op(self) -> None:
        """calling start() twice does not re-subscribe or re-load."""
        loader = AsyncMock(return_value=[])
        client, listener, _ = _make_listener_capturing_subscribe()
        authz = LocalGrantAuthorizer(
            grant_loader=loader,
            epoch_client=client,
            epoch_listener=listener,
            catchup_interval_seconds=3600.0,
        )
        await authz.start()
        await authz.start()
        try:
            loader.assert_awaited_once()
            listener.subscribe.assert_awaited_once()
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_catchup_task_and_is_idempotent(self) -> None:
        """stop() cancels the spawned tick; second stop is a no-op."""
        loader = AsyncMock(return_value=[])
        authz, _ = await _build_started_authorizer(loader=loader)
        # first stop() cancels the task.
        await authz.stop()
        # second stop() is safe (no-op).
        await authz.stop()


class TestLocalGrantAuthorizerCatchupTick:
    """the periodic catch-up tick is the safety net for missed broadcasts."""

    @pytest.mark.asyncio
    async def test_catchup_loop_invokes_epoch_listener_catch_up(self) -> None:
        """the spawned tick calls EpochListener.catch_up on every interval.

        constructs the authorizer with a tiny interval and asserts
        catch_up was called at least once before stop. proves the
        background task wires the right method, even if we don't
        wait for many ticks.
        """
        import asyncio

        loader = AsyncMock(return_value=[])
        client, listener, _ = _make_listener_capturing_subscribe()
        authz = LocalGrantAuthorizer(
            grant_loader=loader,
            epoch_client=client,
            epoch_listener=listener,
            catchup_interval_seconds=0.01,  # 10ms -- fires almost immediately
        )
        await authz.start()
        try:
            # let the loop run a couple of ticks.
            await asyncio.sleep(0.05)
            assert listener.catch_up.await_count >= 1
            # the tick uses Subjects.mcp_rbac_epoch as the subject.
            from threetears.nats import Subjects

            for call in listener.catch_up.await_args_list:
                assert call.args[0] == Subjects.mcp_rbac_epoch()
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_catchup_loop_swallows_transient_errors(self) -> None:
        """a tick that raises does not kill the loop; subsequent ticks fire.

        proves the narrow exception scope around catch_up keeps the
        safety net alive across a transient L3 / NATS hiccup.
        """
        import asyncio

        loader = AsyncMock(return_value=[])
        client, listener, _ = _make_listener_capturing_subscribe()
        # first catch_up raises; subsequent ones return 0.
        listener.catch_up = AsyncMock(
            side_effect=[
                RuntimeError("transient"),
                0,
                0,
                0,
            ]
        )
        authz = LocalGrantAuthorizer(
            grant_loader=loader,
            epoch_client=client,
            epoch_listener=listener,
            catchup_interval_seconds=0.01,
        )
        await authz.start()
        try:
            await asyncio.sleep(0.05)
            # at least 2 calls means the loop survived the first error.
            assert listener.catch_up.await_count >= 2
        finally:
            await authz.stop()


class TestLocalGrantAuthorizerAdminLogging:
    """admin auto-grant logs the specific principal IDs at start-time."""

    @pytest.mark.asyncio
    async def test_admin_principal_ids_recorded(self, caplog: pytest.LogCaptureFixture) -> None:
        """admin_principal_ids surface in the start-time log payload."""
        admin_id = uuid4()
        loader = AsyncMock(return_value=[])
        with caplog.at_level("INFO"):
            authz, _ = await _build_started_authorizer(
                loader=loader,
                admin_principal_ids={admin_id},
            )
            try:
                # caplog accumulates records; one of them carries the ids.
                joined = " ".join(rec.message for rec in caplog.records)
                assert str(admin_id) in joined or any(
                    str(admin_id) in str(rec.__dict__.get("extra_data", "")) for rec in caplog.records
                )
            finally:
                await authz.stop()


class TestLocalGrantAuthorizerOptionalEpoch:
    """epoch_client / epoch_listener are jointly optional (single-process mode)."""

    @pytest.mark.asyncio
    async def test_start_without_epoch_loads_cache_and_skips_subscribe(self) -> None:
        """no epoch deps: start() loads cache, skips subscribe + catchup."""
        loader = AsyncMock(return_value=[])
        authz = LocalGrantAuthorizer(
            grant_loader=loader,
            # no epoch_client / epoch_listener: single-process mode
            catchup_interval_seconds=3600.0,
        )
        await authz.start()
        try:
            # cache primed (loader called once at start)
            assert loader.await_count == 1
            # no catchup task in single-process mode
            assert authz._catchup_task is None  # noqa: SLF001
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_stop_is_noop_when_no_catchup_task(self) -> None:
        """stop() handles the single-process case where no catchup task ever started."""
        loader = AsyncMock(return_value=[])
        authz = LocalGrantAuthorizer(grant_loader=loader)
        await authz.start()
        # should not raise; catchup_task is None
        await authz.stop()

    def test_constructor_rejects_only_epoch_client(self) -> None:
        """passing exactly one of (client, listener) is a usage error."""
        loader = AsyncMock(return_value=[])
        client, _listener, _captured = _make_listener_capturing_subscribe()
        with pytest.raises(ValueError, match="must be provided together"):
            LocalGrantAuthorizer(
                grant_loader=loader,
                epoch_client=client,
                # epoch_listener intentionally omitted
            )

    def test_constructor_rejects_only_epoch_listener(self) -> None:
        """passing exactly one of (client, listener) is a usage error."""
        loader = AsyncMock(return_value=[])
        _client, listener, _captured = _make_listener_capturing_subscribe()
        with pytest.raises(ValueError, match="must be provided together"):
            LocalGrantAuthorizer(
                grant_loader=loader,
                epoch_listener=listener,
                # epoch_client intentionally omitted
            )
