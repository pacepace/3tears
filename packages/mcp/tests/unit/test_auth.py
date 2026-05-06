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
        assert identity.is_admin is True

    @pytest.mark.asyncio
    async def test_env_var_provides_principal_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """env var carrying a UUID populates principal_id."""
        target = uuid4()
        monkeypatch.setenv("METALLM_ADMIN_USER_ID", str(target))
        provider = EnvVarIdentityProvider()
        identity = await provider.identify()
        assert identity.principal_id == target

    @pytest.mark.asyncio
    async def test_missing_env_var_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """no env var + no explicit principal_id raises a clear error."""
        monkeypatch.delenv("METALLM_ADMIN_USER_ID", raising=False)
        provider = EnvVarIdentityProvider()
        with pytest.raises(RuntimeError, match="METALLM_ADMIN_USER_ID"):
            await provider.identify()

    @pytest.mark.asyncio
    async def test_invalid_env_var_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """malformed env var value raises a UUID-parse error."""
        monkeypatch.setenv("METALLM_ADMIN_USER_ID", "not-a-uuid")
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


def _make_listener_capturing_subscribe() -> tuple[Any, Any, list[Any]]:
    """build a fake EpochClient + EpochListener; capture subscribe callback."""
    fake_client = MagicMock()
    fake_client.current = AsyncMock(return_value=0)

    fake_listener = MagicMock()
    captured: list[Any] = []

    async def _subscribe(subject: Any, on_bump: Any) -> None:  # noqa: ARG001
        captured.append(on_bump)

    fake_listener.subscribe = AsyncMock(side_effect=_subscribe)
    fake_listener.catch_up = AsyncMock(return_value=0)
    return fake_client, fake_listener, captured


class TestLocalGrantAuthorizer:
    """default-deny + admin auto-grant + cache reload on epoch bump."""

    @pytest.mark.asyncio
    async def test_admin_short_circuit_allows_without_grant(self) -> None:
        """is_admin=True identity allows every permission without DB lookup."""
        loader = AsyncMock(return_value=[])
        client, listener, _ = _make_listener_capturing_subscribe()
        authz = LocalGrantAuthorizer(
            grant_loader=loader, epoch_client=client, epoch_listener=listener,
        )
        await authz.start()

        identity = Identity(principal_type="user", principal_id=uuid4(), is_admin=True)
        assert await authz.allows(identity, "anything.at.all") is True
        # admin path does not consult the loader after start().
        loader.assert_awaited_once()  # called by start() reload

    @pytest.mark.asyncio
    async def test_default_deny_when_no_grant(self) -> None:
        """non-admin identity with no matching grant is denied."""
        loader = AsyncMock(return_value=[])
        client, listener, _ = _make_listener_capturing_subscribe()
        authz = LocalGrantAuthorizer(
            grant_loader=loader, epoch_client=client, epoch_listener=listener,
        )
        await authz.start()

        identity = Identity(principal_type="user", principal_id=uuid4())
        assert await authz.allows(identity, "metallm.conv.read") is False

    @pytest.mark.asyncio
    async def test_principal_grant_allows(self) -> None:
        """direct principal_id+permission grant authorizes the call."""
        principal_id = uuid4()
        loader = AsyncMock(return_value=[
            {"principal_id": principal_id, "permission": "metallm.conv.read"},
        ])
        client, listener, _ = _make_listener_capturing_subscribe()
        authz = LocalGrantAuthorizer(
            grant_loader=loader, epoch_client=client, epoch_listener=listener,
        )
        await authz.start()

        identity = Identity(principal_type="user", principal_id=principal_id)
        assert await authz.allows(identity, "metallm.conv.read") is True
        # other permissions still deny.
        assert await authz.allows(identity, "metallm.conv.write") is False

    @pytest.mark.asyncio
    async def test_group_grant_allows_via_membership(self) -> None:
        """grant on a group_id allows when identity carries that group."""
        group_id = uuid4()
        loader = AsyncMock(return_value=[
            {"principal_id": group_id, "permission": "hub.audit.read"},
        ])
        client, listener, _ = _make_listener_capturing_subscribe()
        authz = LocalGrantAuthorizer(
            grant_loader=loader, epoch_client=client, epoch_listener=listener,
        )
        await authz.start()

        identity = Identity(
            principal_type="user",
            principal_id=uuid4(),
            groups=frozenset({group_id}),
        )
        assert await authz.allows(identity, "hub.audit.read") is True

    @pytest.mark.asyncio
    async def test_role_grant_allows_via_assignment(self) -> None:
        """grant on a role_id allows when identity carries that role."""
        role_id = uuid4()
        loader = AsyncMock(return_value=[
            {"principal_id": role_id, "permission": "hub.audit.read"},
        ])
        client, listener, _ = _make_listener_capturing_subscribe()
        authz = LocalGrantAuthorizer(
            grant_loader=loader, epoch_client=client, epoch_listener=listener,
        )
        await authz.start()

        identity = Identity(
            principal_type="user",
            principal_id=uuid4(),
            roles=frozenset({role_id}),
        )
        assert await authz.allows(identity, "hub.audit.read") is True

    @pytest.mark.asyncio
    async def test_start_subscribes_to_rbac_epoch(self) -> None:
        """start() subscribes the listener to Subjects.mcp_rbac_epoch."""
        loader = AsyncMock(return_value=[])
        client, listener, _captured = _make_listener_capturing_subscribe()
        authz = LocalGrantAuthorizer(
            grant_loader=loader, epoch_client=client, epoch_listener=listener,
        )
        await authz.start()
        listener.subscribe.assert_awaited_once()
        # subject argument should be the canonical mcp.rbac epoch subject
        from threetears.nats import Subjects
        called_subject = listener.subscribe.await_args.args[0]
        assert called_subject == Subjects.mcp_rbac_epoch()

    @pytest.mark.asyncio
    async def test_epoch_bump_reloads_cache(self) -> None:
        """on rbac bump, reload pulls current grants and replaces cache."""
        principal_id = uuid4()
        loader = AsyncMock(side_effect=[
            [],  # cold-start load: no grants yet
            [{"principal_id": principal_id, "permission": "metallm.conv.read"}],  # post-bump
        ])
        client, listener, captured = _make_listener_capturing_subscribe()
        authz = LocalGrantAuthorizer(
            grant_loader=loader, epoch_client=client, epoch_listener=listener,
        )
        await authz.start()

        identity = Identity(principal_type="user", principal_id=principal_id)
        # before bump: deny.
        assert await authz.allows(identity, "metallm.conv.read") is False
        # simulate bump dispatch.
        on_bump = captured[0]
        await on_bump(7, {"hint": "added grant"})
        # after bump: allow.
        assert await authz.allows(identity, "metallm.conv.read") is True

    @pytest.mark.asyncio
    async def test_loader_failure_keeps_prior_cache(self) -> None:
        """transient L3 hiccup logs + leaves prior cache in place."""
        principal_id = uuid4()
        loader = AsyncMock(side_effect=[
            [{"principal_id": principal_id, "permission": "hub.audit.read"}],  # initial
            RuntimeError("L3 down"),  # bump-time failure
        ])
        client, listener, captured = _make_listener_capturing_subscribe()
        authz = LocalGrantAuthorizer(
            grant_loader=loader, epoch_client=client, epoch_listener=listener,
        )
        await authz.start()

        identity = Identity(principal_type="user", principal_id=principal_id)
        assert await authz.allows(identity, "hub.audit.read") is True

        await captured[0](2, None)  # bump that fails
        # cache should still hold the original grant.
        assert await authz.allows(identity, "hub.audit.read") is True

    @pytest.mark.asyncio
    async def test_uuid_string_in_loader_row_is_normalised(self) -> None:
        """rows with stringified UUIDs (e.g. asyncpg .Record edge) match."""
        principal_id = uuid4()
        loader = AsyncMock(return_value=[
            {"principal_id": str(principal_id), "permission": "metallm.conv.read"},
        ])
        client, listener, _ = _make_listener_capturing_subscribe()
        authz = LocalGrantAuthorizer(
            grant_loader=loader, epoch_client=client, epoch_listener=listener,
        )
        await authz.start()

        identity = Identity(principal_type="user", principal_id=principal_id)
        assert await authz.allows(identity, "metallm.conv.read") is True

    @pytest.mark.asyncio
    async def test_double_start_is_no_op(self) -> None:
        """calling start() twice does not re-subscribe or re-load."""
        loader = AsyncMock(return_value=[])
        client, listener, _ = _make_listener_capturing_subscribe()
        authz = LocalGrantAuthorizer(
            grant_loader=loader, epoch_client=client, epoch_listener=listener,
        )
        await authz.start()
        await authz.start()
        loader.assert_awaited_once()
        listener.subscribe.assert_awaited_once()
