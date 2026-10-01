"""the registry builds its host identity ONCE, hands every factory its token, and releases it on shutdown.

The host mints the token this process presents on its L3 reads, and keeps it fresh with a refresh
loop. Three properties, each of which used to live in host module state instead:

1. **one identity per process.** The rbac stack, the pod authenticator and the limit guard each
   build an L3 backend and all three present the same identity. The server asks the host for it
   once and hands the same token provider to each factory, so the host cannot run three
   handshakes and three refresh loops against one principal;
2. **a provider, never a value.** Each factory receives the identity's BOUND ``token``, so a token
   the host re-mints reaches every backend with no rewiring;
3. **the server owns the lifecycle.** ``shutdown`` closes the identity after everything that
   reads through it is stopped and before the connection drains, so the refresh loop stops with
   the process that started it rather than outliving it.
"""

from __future__ import annotations

import runpy
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from threetears.registry import server as server_module
from threetears.registry.auth import AllowAllAuthorizer, RegistryIdentity
from threetears.registry.server import RegistryServer


# parity-with: threetears.registry.auth.RegistryIdentity
class _FakeRegistryIdentity:
    """a host identity that records what the server did with it.

    :param events: the shared ordered log the test reads the shutdown order from
    :ptype events: list[str]
    """

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.closed = 0

    def token(self) -> str | None:
        """the current token.

        :return: a fixed token
        :rtype: str | None
        """
        return "host.minted.token"

    async def close(self) -> None:
        """record the close.

        :return: nothing
        :rtype: None
        """
        self.closed += 1
        self.events.append("identity.close")


def _nats_client() -> MagicMock:
    """a stand-in for the canonical wrapper client the factories receive.

    :return: a mock client
    :rtype: MagicMock
    """
    client = MagicMock()
    client.raw = MagicMock()
    return client


class TestOneIdentityPerProcess:
    """the identity is built once and every factory that reads L3 receives its bound token."""

    @pytest.mark.asyncio
    async def test_every_factory_gets_the_one_identitys_bound_token(self) -> None:
        """built once, and the same provider reaches the rbac, pod and limit factories."""
        identity = _FakeRegistryIdentity([])
        identity_factory = AsyncMock(return_value=identity)
        rbac_factory = AsyncMock(return_value=AllowAllAuthorizer())
        pod_factory = AsyncMock(return_value=None)
        limit_factory = AsyncMock(return_value=None)
        server = RegistryServer(
            namespace="testns",
            authorizer=AllowAllAuthorizer(),
            identity_factory=identity_factory,
            rbac_authorizer_factory=rbac_factory,
            pod_authenticator_factory=pod_factory,
            limit_guard_factory=limit_factory,
        )
        nc = _nats_client()

        assert await server.apply_identity_factory(nc) is identity
        await server.apply_rbac_factory(nc)
        await server.apply_pod_authenticator_factory(nc)
        await server.apply_limit_guard_factory(nc)

        identity_factory.assert_awaited_once_with(nc)
        for factory in (rbac_factory, pod_factory, limit_factory):
            factory.assert_awaited_once_with(nc, identity.token)
            provider = factory.await_args.args[1]
            # the BOUND method, not the token it returns today: a re-mint must reach the backend.
            assert provider.__self__ is identity

    @pytest.mark.asyncio
    async def test_without_an_identity_factory_the_factories_get_no_provider(self) -> None:
        """the pure-3tears / dev shape: nothing to present, and each factory decides what that means."""
        rbac_factory = AsyncMock(return_value=AllowAllAuthorizer())
        server = RegistryServer(
            namespace="testns",
            authorizer=AllowAllAuthorizer(),
            rbac_authorizer_factory=rbac_factory,
        )
        nc = _nats_client()

        assert await server.apply_identity_factory(nc) is None
        await server.apply_rbac_factory(nc)

        rbac_factory.assert_awaited_once_with(nc, None)


class TestShutdownReleasesTheIdentity:
    """the refresh loop stops with the process that started it."""

    @pytest.mark.asyncio
    async def test_shutdown_closes_the_identity_after_the_teardown_seam(self) -> None:
        """everything that reads through the token is stopped first; then the identity is released."""
        events: list[str] = []
        identity = _FakeRegistryIdentity(events)

        async def _teardown() -> None:
            events.append("on_shutdown")

        server = RegistryServer(
            namespace="testns",
            authorizer=AllowAllAuthorizer(),
            identity_factory=AsyncMock(return_value=identity),
            on_shutdown=_teardown,
        )
        await server.apply_identity_factory(_nats_client())

        await server.shutdown()

        assert identity.closed == 1
        assert events == ["on_shutdown", "identity.close"]

    @pytest.mark.asyncio
    async def test_shutdown_before_the_identity_was_built_closes_nothing(self) -> None:
        """a server torn down before startup reached the identity must not raise."""
        identity_factory = AsyncMock()
        server = RegistryServer(
            namespace="testns",
            authorizer=AllowAllAuthorizer(),
            identity_factory=identity_factory,
        )

        await server.shutdown()

        identity_factory.assert_not_awaited()

    def test_the_fake_is_a_registry_identity(self) -> None:
        """the protocol is runtime-checkable, so a host object missing ``close`` is caught here."""
        assert isinstance(_FakeRegistryIdentity([]), RegistryIdentity)


class TestTheEntryPointResolvesTheIdentityHook:
    """``python -m threetears.registry`` hands the server the host's identity factory, unawaited."""

    @staticmethod
    def _captured(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        """run the entry point and return the keyword arguments it built the server with.

        :param monkeypatch: pytest monkeypatch fixture
        :ptype monkeypatch: pytest.MonkeyPatch
        :return: the server's constructor keywords
        :rtype: dict[str, Any]
        """
        monkeypatch.delenv("THREETEARS_REGISTRY_ALLOW_ALL_TOOLS", raising=False)
        monkeypatch.delenv("THREETEARS_REGISTRY_FORCE_DENY_ALL", raising=False)
        captured: dict[str, Any] = {}

        class _CapturingServer:
            def __init__(self, **kwargs: Any) -> None:
                captured.update(kwargs)

            async def serve(self) -> None:
                return None

        monkeypatch.setattr(server_module, "RegistryServer", _CapturingServer)
        runpy.run_module("threetears.registry", run_name="__main__")
        return captured

    def test_the_configured_factory_is_handed_over_not_called(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """the server calls it once NATS is up; resolving it must not handshake."""
        host_factory = AsyncMock()
        monkeypatch.setattr(server_module, "HOST_IDENTITY_FACTORY_FOR_TEST", host_factory, raising=False)
        monkeypatch.setenv(
            "THREETEARS_REGISTRY_IDENTITY_TOKEN_PROVIDER_FACTORY",
            "threetears.registry.server:HOST_IDENTITY_FACTORY_FOR_TEST",
        )

        captured = self._captured(monkeypatch)

        assert captured["identity_factory"] is host_factory
        host_factory.assert_not_awaited()

    def test_unset_hands_over_no_factory(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """no host identity configured: the rbac stack then refuses at wiring, not here."""
        monkeypatch.delenv("THREETEARS_REGISTRY_IDENTITY_TOKEN_PROVIDER_FACTORY", raising=False)

        assert self._captured(monkeypatch)["identity_factory"] is None

    def test_a_malformed_spec_crashes_startup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """a misconfigured identity plugin must crash startup, never run unidentified."""
        monkeypatch.setenv("THREETEARS_REGISTRY_IDENTITY_TOKEN_PROVIDER_FACTORY", "not-a-dotted-path")

        with pytest.raises(ValueError, match="module:callable"):
            self._captured(monkeypatch)
