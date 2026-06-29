"""B5: the registry gates its k8s readiness on the Hub-JWKS cache being warm.

``RegistryServer._start_handlers`` wires a ``HealthCheck(name="jwks_warmed", probe=lambda: provider
is not None and provider.is_warmed)`` onto its ``HealthServer``. Until the cache's first successful
fetch, the proxy verifies every identity token against an EMPTY keyset and rejects fail-closed, so a
readiness probe that flipped ready too early would route calls the registry is guaranteed to fail.

These tests exercise the readiness CONTRACT end-to-end through the real primitives the registry wires
together -- a real :class:`CachedHubJwksProvider` (warmed via a mocked NATS fetch) driving a real
:class:`HealthServer` through the SAME probe predicate ``server.py`` uses -- so the NOT-READY ->
READY transition the registry depends on is verified rather than assumed. (The full ``serve()`` loop
that installs the check lives in the integration suite; it touches NATS connect + JetStream KV.)
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from threetears.core.security.jwks_provider import CachedHubJwksProvider
from threetears.nats.errors import RequestError
from threetears.observe import HealthCheck, HealthServer

_JWKS: dict[str, Any] = {
    "keys": [{"kid": "k1", "kty": "OKP", "crv": "Ed25519", "x": "abc", "use": "sig", "alg": "EdDSA"}]
}


def _registry_jwks_check(provider: CachedHubJwksProvider | None) -> HealthCheck:
    """the EXACT readiness probe ``RegistryServer._start_handlers`` wires onto its health server."""
    return HealthCheck(
        name="jwks_warmed",
        probe=lambda: provider is not None and provider.is_warmed,
    )


def _health_server(provider: CachedHubJwksProvider | None) -> HealthServer:
    return HealthServer(port=0, service_name="registry", checks=[_registry_jwks_check(provider)])


class TestRegistryJwksReadinessGate:
    """the registry reports NOT-READY until the Hub-JWKS cache completes its first successful fetch."""

    def test_not_ready_when_provider_absent(self) -> None:
        # before the provider is constructed the gate must read NOT-READY (never accept-then-fail).
        status = _health_server(None).get_status()
        assert status.healthy is False
        assert {c.name: c.healthy for c in status.components}["jwks_warmed"] is False

    @pytest.mark.asyncio
    async def test_not_ready_until_warmed_then_ready(self) -> None:
        nc = MagicMock()
        # the first fetch FAILS (Hub responder not up yet) so the cache stays empty + un-warmed;
        # the second fetch succeeds and warms it.
        nc.request_raw = AsyncMock(side_effect=[RequestError("hub down"), json.dumps(_JWKS).encode("utf-8")])
        provider = CachedHubJwksProvider(nc, request_timeout_seconds=1.0)
        server = _health_server(provider)

        await provider.refresh()  # first fetch fails -> NOT warmed
        not_ready = server.get_status()
        assert not_ready.healthy is False
        assert {c.name: c.healthy for c in not_ready.components}["jwks_warmed"] is False

        await provider.refresh()  # second fetch succeeds -> warmed
        ready = server.get_status()
        assert ready.healthy is True
        assert {c.name: c.healthy for c in ready.components}["jwks_warmed"] is True
