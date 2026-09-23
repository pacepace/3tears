"""a connection whose credential expires is renewed before it does, by the client itself (#514).

The auth-callout mints each connection's user JWT with a finite TTL, and at expiry the server
closes the connection in a way forever-reconnect does not cover. Three callers each grew their
own renewal loop and the copies diverged; the loop now lives on :class:`NatsClient`. Pinned here:

1. the schedule reconnects BEFORE expiry, and never busy-spins on a tiny TTL;
2. an unknown / non-positive TTL is re-checked on a cadence, never reconnected on a guess;
3. a standalone connection's TTL comes from ``FOURTEENAIBOTS_NATS_USER_JWT_TTL_SECONDS``;
4. a cadence too short to carry the caller's longest request is named, with the TTL;
5. the loop reconnects on schedule, runs the owner's ``before_renewal`` first, retries a
   failed renewal on the retry cadence rather than a full cycle, credits the owner's drain
   to the cadence, and never reconnects while the TTL is unknown;
6. a second ``renew_credential`` replaces the first, and ``shutdown`` stops the loop.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from threetears.nats import (
    DEFAULT_NATS_USER_JWT_TTL_SECONDS,
    PLATFORM_DEFAULT_NATS_USER_JWT_TTL_SECONDS,
    REAUTH_BUFFER_SECONDS,
    REAUTH_LEEWAY_SECONDS,
    REAUTH_MIN_SLEEP_SECONDS,
    REAUTH_UNKNOWN_TTL_RECHECK_SECONDS,
    SYNC_REPLY_BUDGET_SECONDS,
    NatsClient,
    has_schedulable_ttl,
    nats_user_jwt_ttl_seconds,
    seconds_until_reauth,
    unsafe_reauth_delay_reason,
)

_TTL_ENV = "FOURTEENAIBOTS_NATS_USER_JWT_TTL_SECONDS"


class TestTheSchedule:
    """a reconnect lands before expiry, clamped off the hot path, and never on a guess."""

    def test_it_lands_before_expiry(self) -> None:
        ttl = 180
        assert seconds_until_reauth(ttl) == pytest.approx(ttl - REAUTH_LEEWAY_SECONDS - REAUTH_BUFFER_SECONDS)

    def test_a_tiny_ttl_clamps_to_the_minimum_sleep(self) -> None:
        assert seconds_until_reauth(10) == REAUTH_MIN_SLEEP_SECONDS

    @pytest.mark.parametrize("ttl", [None, 0, -5])
    def test_an_unknown_ttl_is_rechecked(self, ttl: int | None) -> None:
        assert seconds_until_reauth(ttl) == REAUTH_UNKNOWN_TTL_RECHECK_SECONDS
        assert has_schedulable_ttl(ttl) is False

    def test_a_positive_ttl_is_schedulable(self) -> None:
        assert has_schedulable_ttl(1) is True


class TestTheEnvironmentTtl:
    """a connection with no handshake reads the TTL the platform mints with."""

    def test_unset_is_the_platform_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(_TTL_ENV, raising=False)
        assert nats_user_jwt_ttl_seconds() == 300

    def test_the_default_renews_outside_the_synchronous_reply_budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """a default inside the budget would recycle the connection under calls still owed a reply."""
        monkeypatch.delenv(_TTL_ENV, raising=False)
        assert seconds_until_reauth(nats_user_jwt_ttl_seconds()) > SYNC_REPLY_BUDGET_SECONDS

    def test_an_override_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_TTL_ENV, "600")
        assert nats_user_jwt_ttl_seconds() == 600

    @pytest.mark.parametrize("raw", ["0", "-30", "not-a-number"])
    def test_an_unusable_value_is_unknown_not_a_crash(self, monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
        monkeypatch.setenv(_TTL_ENV, raw)
        assert nats_user_jwt_ttl_seconds() is None


class TestAnUnsafeCadenceIsNamed:
    """a renewal drops every request in flight, so a cadence shorter than the longest is fatal to it."""

    def test_a_cadence_that_cannot_carry_the_longest_request_is_explained(self) -> None:
        reason = unsafe_reauth_delay_reason(60.0, 150, longest_request_seconds=120.0)
        assert reason is not None
        assert "150" in reason
        assert _TTL_ENV in reason

    def test_a_cadence_that_can_is_safe(self) -> None:
        assert unsafe_reauth_delay_reason(210.0, 300, longest_request_seconds=120.0) is None

    def test_an_unknown_ttl_is_not_called_unsafe(self) -> None:
        assert unsafe_reauth_delay_reason(60.0, None, longest_request_seconds=120.0) is None

    def test_the_reason_names_the_symptom_an_operator_sees(self) -> None:
        """nothing else in the logs connects "it answers, then hangs" to the TTL."""
        reason = unsafe_reauth_delay_reason(60.0, 150, longest_request_seconds=120.0)
        assert reason is not None
        assert "answers quickly, then hangs" in reason

    def test_a_drain_widens_the_window(self) -> None:
        """an owner that holds the connection open for requests in flight survives a shorter cadence."""
        delay = seconds_until_reauth(100)
        assert unsafe_reauth_delay_reason(delay, 100, longest_request_seconds=30.0) is not None
        assert unsafe_reauth_delay_reason(delay, 100, longest_request_seconds=30.0, drain_grace_seconds=30.0) is None

    def test_a_drain_never_widens_the_window_past_the_leeway(self) -> None:
        """the server closes the connection at ttl - leeway whatever the owner is waiting on."""
        ttl = 90
        delay = seconds_until_reauth(ttl)
        # delay is clamped to the minimum sleep, so delay + grace would overstate the window
        assert delay + 60.0 > ttl - REAUTH_LEEWAY_SECONDS
        reason = unsafe_reauth_delay_reason(delay, ttl, longest_request_seconds=30.0, drain_grace_seconds=60.0)
        assert reason is not None

    def test_the_boundary_is_exclusive(self) -> None:
        """a window equal to the longest request leaves it no time to deliver its answer."""
        assert unsafe_reauth_delay_reason(120.0, 210, longest_request_seconds=120.0) is not None


class TestOneDefaultTtl:
    """the TTL a connection assumes and the TTL the generic responder mints are one relation."""

    def test_the_assumed_ttl_never_exceeds_the_generic_mint(self) -> None:
        """assuming more than a minter mints is fatal; assuming less is only churn."""
        assert PLATFORM_DEFAULT_NATS_USER_JWT_TTL_SECONDS <= DEFAULT_NATS_USER_JWT_TTL_SECONDS


def _client() -> NatsClient:
    """a connected client over a stand-in raw connection.

    :return: the client
    :rtype: NatsClient
    """
    raw = MagicMock()
    raw.is_closed = False
    raw.is_connected = True
    raw.drain = AsyncMock()
    raw.close = AsyncMock()
    return NatsClient(raw=raw, namespace="ns", client_name="renewal-test")


@pytest.fixture
def fast(monkeypatch: pytest.MonkeyPatch) -> None:
    """shrink the schedule and the retry so a test sees several cycles in milliseconds.

    :param monkeypatch: pytest's patcher
    :ptype monkeypatch: pytest.MonkeyPatch
    :return: None
    :rtype: None
    """
    monkeypatch.setattr("threetears.nats.client.seconds_until_reauth", lambda _ttl: 0.01)
    monkeypatch.setattr("threetears.nats.client.REAUTH_RETRY_SECONDS", 0.01)


class TestTheClientRenewsItsOwnCredential:
    """the loop the three callers each carried, owned once by the client."""

    async def test_it_reconnects_on_schedule_after_the_owners_hook(
        self, monkeypatch: pytest.MonkeyPatch, fast: None
    ) -> None:
        del fast
        order: list[str] = []
        renewed = asyncio.Event()

        async def _reconnect(self: NatsClient) -> None:
            order.append("reconnect")
            renewed.set()

        async def _drain(ttl: int) -> None:
            order.append(f"drain:{ttl}")

        monkeypatch.setattr(NatsClient, "reconnect", _reconnect)
        client = _client()
        client.renew_credential(ttl_seconds=lambda: 300, before_renewal=_drain)
        try:
            async with asyncio.timeout(2.0):
                await renewed.wait()
        finally:
            await client.shutdown()

        assert order[:2] == ["drain:300", "reconnect"]

    async def test_a_failed_renewal_is_retried_not_fatal(self, monkeypatch: pytest.MonkeyPatch, fast: None) -> None:
        del fast
        calls: list[int] = []
        recovered = asyncio.Event()

        async def _reconnect(self: NatsClient) -> None:
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("transient reconnect failure")
            recovered.set()

        monkeypatch.setattr(NatsClient, "reconnect", _reconnect)
        client = _client()
        client.renew_credential(ttl_seconds=lambda: 300)
        try:
            async with asyncio.timeout(2.0):
                await recovered.wait()
        finally:
            await client.shutdown()

        assert len(calls) >= 2

    async def test_a_failure_after_the_scheduled_sleep_retries_without_a_second_cycle(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """the retry replaces the schedule: a second full cycle would land after the credential expired.

        the schedule and the retry differ by four orders of magnitude, so a loop that sleeps the
        retry AND then the schedule again misses the deadline instead of passing by accident.
        """
        schedule = iter([0.01])
        monkeypatch.setattr("threetears.nats.client.seconds_until_reauth", lambda _ttl: next(schedule, 60.0))
        monkeypatch.setattr("threetears.nats.client.REAUTH_RETRY_SECONDS", 0.01)
        calls: list[int] = []
        recovered = asyncio.Event()

        async def _reconnect(self: NatsClient) -> None:
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("transient reconnect failure")
            recovered.set()

        monkeypatch.setattr(NatsClient, "reconnect", _reconnect)
        client = _client()
        client.renew_credential(ttl_seconds=lambda: 300)
        try:
            async with asyncio.timeout(1.0):
                await recovered.wait()
        finally:
            await client.shutdown()

        assert len(calls) == 2

    async def test_a_drain_grace_is_credited_to_the_cadence(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """the owner's drain is part of the window, so a cadence it can carry is not called unsafe."""

        async def _reconnect(self: NatsClient) -> None:
            return None

        monkeypatch.setattr(NatsClient, "reconnect", _reconnect)
        client = _client()
        with caplog.at_level(logging.ERROR, logger="threetears.nats.client"):
            client.renew_credential(ttl_seconds=lambda: 100, longest_request_seconds=30.0, drain_grace_seconds=30.0)
            try:
                await asyncio.sleep(0.05)
            finally:
                await client.shutdown()

        assert not any("UNSAFE" in r.getMessage() for r in caplog.records)

    async def test_an_unknown_ttl_never_reconnects(self, monkeypatch: pytest.MonkeyPatch, fast: None) -> None:
        del fast
        calls: list[int] = []

        async def _reconnect(self: NatsClient) -> None:
            calls.append(1)

        monkeypatch.setattr(NatsClient, "reconnect", _reconnect)
        client = _client()
        client.renew_credential(ttl_seconds=lambda: None)
        try:
            await asyncio.sleep(0.1)
        finally:
            await client.shutdown()

        assert calls == []

    async def test_an_unsafe_cadence_is_logged_as_an_error(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        async def _reconnect(self: NatsClient) -> None:
            return None

        monkeypatch.setattr(NatsClient, "reconnect", _reconnect)
        client = _client()
        with caplog.at_level(logging.ERROR, logger="threetears.nats.client"):
            client.renew_credential(ttl_seconds=lambda: 150, longest_request_seconds=120.0)
            try:
                await asyncio.sleep(0.05)
            finally:
                await client.shutdown()

        assert any("UNSAFE NATS credential renewal cadence" in r.getMessage() for r in caplog.records)

    async def test_a_second_call_replaces_the_first(self, monkeypatch: pytest.MonkeyPatch, fast: None) -> None:
        del fast
        seen: list[int] = []
        second_seen = asyncio.Event()

        async def _reconnect(self: NatsClient) -> None:
            return None

        async def _hook(ttl: int) -> None:
            seen.append(ttl)
            if ttl == 600:
                second_seen.set()

        monkeypatch.setattr(NatsClient, "reconnect", _reconnect)
        client = _client()
        client.renew_credential(ttl_seconds=lambda: 300, before_renewal=_hook)
        client.renew_credential(ttl_seconds=lambda: 600, before_renewal=_hook)
        try:
            async with asyncio.timeout(2.0):
                await second_seen.wait()
            seen.clear()
            await asyncio.sleep(0.05)
        finally:
            await client.shutdown()

        assert seen and set(seen) == {600}

    async def test_shutdown_stops_the_loop(self, monkeypatch: pytest.MonkeyPatch, fast: None) -> None:
        del fast
        calls: list[int] = []

        async def _reconnect(self: NatsClient) -> None:
            calls.append(1)

        monkeypatch.setattr(NatsClient, "reconnect", _reconnect)
        client = _client()
        client.renew_credential(ttl_seconds=lambda: 300)
        await asyncio.sleep(0.05)

        client.raw.is_closed = True
        await client.shutdown()
        stopped_at = len(calls)
        await asyncio.sleep(0.05)

        assert len(calls) == stopped_at
