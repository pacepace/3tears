"""tests for CachedHubJwksProvider (verifier-side Hub JWKS fetch + cache, fail-closed)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from threetears.nats.errors import RequestError

from threetears.core.security.jwks_provider import CachedHubJwksProvider

_JWKS: dict[str, Any] = {
    "keys": [{"kid": "k1", "kty": "OKP", "crv": "Ed25519", "x": "abc", "use": "sig", "alg": "EdDSA"}]
}


def _client(reply: dict[str, Any] | None = None, *, error: Exception | None = None) -> MagicMock:
    nc = MagicMock()
    if error is not None:
        nc.request_raw = AsyncMock(side_effect=error)
    else:
        nc.request_raw = AsyncMock(return_value=json.dumps(reply).encode("utf-8"))
    return nc


def _provider(nc: MagicMock, **kwargs: Any) -> CachedHubJwksProvider:
    """build a provider; the request timeout is required (sourced from config in production)."""
    return CachedHubJwksProvider(nc, request_timeout_seconds=1.0, **kwargs)


class TestCachedHubJwksProvider:
    """the provider returns the cached JWKS synchronously and fails closed."""

    def test_fail_closed_before_any_fetch(self) -> None:
        provider = _provider(_client(_JWKS))
        assert provider() == {"keys": []}

    @pytest.mark.asyncio
    async def test_refresh_populates_cache(self) -> None:
        provider = _provider(_client(_JWKS))
        await provider.refresh()
        assert provider() == _JWKS

    @pytest.mark.asyncio
    async def test_failed_fetch_stays_empty_fail_closed(self) -> None:
        provider = _provider(_client(error=RequestError("hub down")))
        await provider.refresh()
        assert provider() == {"keys": []}

    @pytest.mark.asyncio
    async def test_malformed_reply_keeps_empty(self) -> None:
        provider = _provider(_client({"not": "a jwks"}))
        await provider.refresh()
        assert provider() == {"keys": []}

    @pytest.mark.asyncio
    async def test_failed_refresh_keeps_last_good(self) -> None:
        nc = MagicMock()
        nc.request_raw = AsyncMock(return_value=json.dumps(_JWKS).encode("utf-8"))
        provider = _provider(nc)
        await provider.refresh()
        assert provider() == _JWKS
        # a later refresh failure keeps the last good JWKS (overlap rotation / transient blip).
        nc.request_raw = AsyncMock(side_effect=RequestError("blip"))
        await provider.refresh()
        assert provider() == _JWKS

    @pytest.mark.asyncio
    async def test_start_fetches_then_stop_cancels(self) -> None:
        provider = _provider(_client(_JWKS), refresh_interval_seconds=3600)
        await provider.start()
        assert provider() == _JWKS
        await provider.stop()  # idempotent + must not raise
