"""tests for CachedHubJwksProvider (verifier-side Hub JWKS fetch + cache, fail-closed)."""

from __future__ import annotations

import asyncio
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

    @pytest.mark.asyncio
    async def test_cold_start_retries_fast_until_first_success(self) -> None:
        # the initial fetch races the Hub's responder at boot and FAILS; the steady refresh interval
        # is long (an hour), but the cold-start retry is short, so the cache warms within the short
        # interval instead of waiting an hour. under enforce an empty cache rejects every call, so a
        # slow warm is a multi-minute outage -- this is the regression guard for that.
        nc = MagicMock()
        nc.request_raw = AsyncMock(
            side_effect=[RequestError("hub responder not up yet"), json.dumps(_JWKS).encode("utf-8")]
        )
        provider = _provider(nc, refresh_interval_seconds=3600, initial_retry_interval_seconds=0.01)
        await provider.start()  # initial refresh fails -> cache empty, not warmed
        assert provider() == {"keys": []}
        await asyncio.sleep(0.05)  # >> the 0.01s cold-start retry, << the 3600s steady interval
        assert provider() == _JWKS  # the FAST retry warmed it, not the hour-long steady loop
        await provider.stop()

    @pytest.mark.asyncio
    async def test_steady_interval_after_warm_no_busy_retry(self) -> None:
        # once warmed, the loop uses the long steady interval -- a single fetch is enough and there
        # is no busy fast-retry afterwards (the side_effect would raise StopAsyncIteration on a 2nd).
        nc = MagicMock()
        nc.request_raw = AsyncMock(side_effect=[json.dumps(_JWKS).encode("utf-8")])
        provider = _provider(nc, refresh_interval_seconds=3600, initial_retry_interval_seconds=0.01)
        await provider.start()  # one successful fetch -> warmed -> steady 3600s loop
        assert provider() == _JWKS
        await asyncio.sleep(0.05)  # no further fetch (would exhaust the single side_effect)
        assert provider() == _JWKS
        await provider.stop()
