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
    async def test_empty_keyset_keeps_last_good_and_does_not_warm(self) -> None:
        # a shape-valid but EMPTY keyset verifies NOTHING; it must not warm (readiness would lie)
        # and must not clobber a good cache -- treat it like a transient blip.
        nc = MagicMock()
        nc.request_raw = AsyncMock(return_value=json.dumps(_JWKS).encode("utf-8"))
        provider = _provider(nc)
        await provider.refresh()
        assert provider.is_warmed is True
        nc.request_raw = AsyncMock(return_value=json.dumps({"keys": []}).encode("utf-8"))
        await provider.refresh()
        assert provider() == _JWKS  # last-good kept, NOT replaced with the empty keyset
        assert provider.is_warmed is True  # a working cache is not un-warmed by an empty reply
        # an empty keyset on a COLD provider never warms (it can verify nothing -> stays NOT-ready).
        cold = _provider(_client({"keys": []}))
        await cold.refresh()
        assert cold.is_warmed is False
        assert cold()["keys"] == []

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


class TestIsWarmedReadiness:
    """``is_warmed`` is the readiness signal a verifier gates its k8s readiness on."""

    def test_not_warmed_before_any_fetch(self) -> None:
        provider = _provider(_client(_JWKS))
        assert provider.is_warmed is False  # an empty cache cannot verify anything yet -> NOT-READY

    @pytest.mark.asyncio
    async def test_warms_on_first_successful_fetch(self) -> None:
        provider = _provider(_client(_JWKS))
        await provider.refresh()
        assert provider.is_warmed is True

    @pytest.mark.asyncio
    async def test_stays_not_warmed_on_failed_fetch(self) -> None:
        provider = _provider(_client(error=RequestError("hub down")))
        await provider.refresh()
        assert provider.is_warmed is False  # a failed fetch leaves it un-warm (still NOT-READY)


class TestReactiveRefresh:
    """``refresh_now`` is the reactive self-heal for a Hub re-key: one immediate, debounced,
    rate-limited refresh so a valid token signed under a freshly-rotated key heals on the FIRST
    failed-but-valid token rather than after a full steady interval."""

    @pytest.mark.asyncio
    async def test_refresh_now_fetches_and_returns_true(self) -> None:
        nc = MagicMock()
        nc.request_raw = AsyncMock(return_value=json.dumps(_JWKS).encode("utf-8"))
        provider = _provider(nc)
        ran = await provider.refresh_now()
        assert ran is True
        assert provider() == _JWKS
        assert nc.request_raw.await_count == 1  # exactly one Hub fetch

    @pytest.mark.asyncio
    async def test_concurrent_triggers_collapse_to_one_fetch(self) -> None:
        # a flood of kid-miss tokens fires many concurrent refresh_now()s; the lock + rate-limit must
        # collapse them to a SINGLE Hub request (no stampede). a plain async fn (not AsyncMock) holds
        # the in-flight fetch open so the other 7 triggers pile up behind the lock while it runs.
        nc = MagicMock()
        fetches = {"n": 0}

        async def _slow_request(*_a: Any, **_k: Any) -> bytes:
            fetches["n"] += 1
            await asyncio.sleep(0.01)  # hold the in-flight refresh so the others queue on the lock
            return json.dumps(_JWKS).encode("utf-8")

        nc.request_raw = _slow_request
        provider = _provider(nc, reactive_min_interval_seconds=60.0)
        results = await asyncio.gather(*(provider.refresh_now() for _ in range(8)))
        assert fetches["n"] == 1  # ONE Hub fetch despite 8 concurrent triggers
        assert sum(1 for r in results if r) == 1  # exactly one trigger reports it actually ran

    @pytest.mark.asyncio
    async def test_rate_limit_suppresses_a_second_refresh_in_window(self) -> None:
        nc = MagicMock()
        nc.request_raw = AsyncMock(return_value=json.dumps(_JWKS).encode("utf-8"))
        provider = _provider(nc, reactive_min_interval_seconds=60.0)
        assert await provider.refresh_now() is True
        # a second trigger inside the window is suppressed -> no second Hub fetch (flood protection).
        assert await provider.refresh_now() is False
        assert nc.request_raw.await_count == 1

    @pytest.mark.asyncio
    async def test_rate_limit_allows_again_after_window(self) -> None:
        nc = MagicMock()
        nc.request_raw = AsyncMock(return_value=json.dumps(_JWKS).encode("utf-8"))
        provider = _provider(nc, reactive_min_interval_seconds=0.01)
        assert await provider.refresh_now() is True
        await asyncio.sleep(0.05)  # window elapsed
        assert await provider.refresh_now() is True  # a genuine later re-key can still self-heal
        assert nc.request_raw.await_count == 2


class TestRefreshLoopUnkillable:
    """the background refresh loop is a supervisor loop: NO exception type may end it, or a Hub
    re-key would stop self-healing for the life of the pod."""

    @pytest.mark.asyncio
    async def test_loop_survives_an_arbitrary_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        nc = MagicMock()
        # nc fails the FIRST fetch (start's, at boot) so the cache is NOT warm when the loop begins;
        # every later fetch succeeds, so warming can only happen if the loop SURVIVES the raise below.
        seq: list[Exception | None] = [RequestError("hub responder not up at boot")]

        async def _request(*_a: Any, **_k: Any) -> bytes:
            if seq:
                raise seq.pop(0)  # type: ignore[misc]
            return json.dumps(_JWKS).encode("utf-8")

        nc.request_raw = _request
        provider = _provider(nc, refresh_interval_seconds=0.01, initial_retry_interval_seconds=0.01)

        calls = {"n": 0}
        real_refresh = provider.refresh

        async def _flaky_refresh() -> None:
            calls["n"] += 1
            # the FIRST loop iteration (call #2; call #1 is start's direct refresh) raises an
            # arbitrary, UNMODELLED exception straight out of refresh. if the loop were not a
            # supervisor loop, this would end the task and the cache would never warm.
            if calls["n"] == 2:
                raise RuntimeError("an arbitrary, unmodelled failure inside refresh")
            await real_refresh()

        monkeypatch.setattr(provider, "refresh", _flaky_refresh)
        await provider.start()  # call #1: real refresh -> nc raises -> swallowed -> NOT warmed
        assert provider.is_warmed is False
        # wait several loop iterations: iter1 raised RuntimeError; the loop must have SURVIVED and a
        # later iteration's successful fetch warmed the cache.
        await asyncio.sleep(0.1)
        assert provider.is_warmed is True
        assert calls["n"] >= 3  # start (#1) + the raising iteration (#2) + a surviving warm (#3+)
        await provider.stop()
