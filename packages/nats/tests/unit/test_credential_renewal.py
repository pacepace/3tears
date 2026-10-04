"""a connection whose credential expires is renewed before it does, by the client itself (hub issue #514).

The auth-callout mints each connection's user JWT with a finite TTL, and at expiry the server
closes the connection in a way forever-reconnect does not cover. The renewal is make-before-break
(:meth:`NatsClient.renew_connection`): a successor connection takes over and the replaced one is
kept open for the longest request it may be carrying. Pinned here:

1. the successor is opened ``ttl - leeway - buffer - longest`` after the current connection was
   established, never busy-spinning on a tiny TTL;
2. the replaced connection is held for the longest request, but never past its credential's expiry;
3. an unknown / non-positive TTL is re-checked on a cadence, never renewed on a guess;
4. a standalone connection's TTL comes from ``FOURTEENAIBOTS_NATS_USER_JWT_TTL_SECONDS``;
5. a TTL too short to carry the caller's longest request across a renewal is named, with the TTL;
6. the loop renews on schedule, measured from the connection's own age, retries a failed renewal
   on the retry cadence rather than a full cycle, and never renews while the TTL is unknown;
7. a second ``renew_credential`` replaces the first, and ``shutdown`` stops the loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from threetears.nats import (
    DEFAULT_NATS_USER_JWT_TTL_SECONDS,
    PLATFORM_DEFAULT_NATS_USER_JWT_TTL_SECONDS,
    REAUTH_BUFFER_SECONDS,
    REAUTH_LEEWAY_SECONDS,
    REAUTH_MARGIN_SECONDS,
    REAUTH_MIN_SLEEP_SECONDS,
    REAUTH_RETIRE_DRAIN_SECONDS,
    REAUTH_UNKNOWN_TTL_RECHECK_SECONDS,
    SYNC_REPLY_BUDGET_SECONDS,
    NatsClient,
    RequestTimeoutError,
    credential_lifetime_from_user_info,
    has_schedulable_ttl,
    nats_user_jwt_ttl_seconds,
    seconds_until_reauth,
    seconds_until_retirement,
    unsafe_renewal_reason,
)

_TTL_ENV = "FOURTEENAIBOTS_NATS_USER_JWT_TTL_SECONDS"


class TestTheSchedule:
    """a successor lands early enough for the replaced connection to finish what it carries."""

    def test_it_leaves_room_for_the_longest_request_before_expiry(self) -> None:
        ttl = 300
        expected = ttl - REAUTH_LEEWAY_SECONDS - REAUTH_BUFFER_SECONDS - 120.0
        assert seconds_until_reauth(ttl, longest_request_seconds=120.0) == pytest.approx(expected)

    def test_a_tiny_ttl_clamps_to_the_minimum_sleep(self) -> None:
        assert seconds_until_reauth(10, longest_request_seconds=30.0) == REAUTH_MIN_SLEEP_SECONDS

    @pytest.mark.parametrize("ttl", [None, 0, -5])
    def test_an_unknown_ttl_is_rechecked(self, ttl: int | None) -> None:
        assert seconds_until_reauth(ttl, longest_request_seconds=30.0) == REAUTH_UNKNOWN_TTL_RECHECK_SECONDS
        assert has_schedulable_ttl(ttl) is False

    def test_a_positive_ttl_is_schedulable(self) -> None:
        assert has_schedulable_ttl(1) is True


class TestTheRetirement:
    """the replaced connection is held for the longest request, and never past its credential."""

    def test_it_holds_for_the_longest_request(self) -> None:
        assert seconds_until_retirement(300, connection_age_seconds=90.0, longest_request_seconds=120.0) == 120.0

    def test_a_late_renewal_shortens_the_hold_to_beat_expiry(self) -> None:
        """a renewal that ran late after failures must not hold the connection past its credential."""
        age = 200.0
        hold = seconds_until_retirement(300, connection_age_seconds=age, longest_request_seconds=120.0)
        assert hold == pytest.approx(300 - REAUTH_LEEWAY_SECONDS - REAUTH_RETIRE_DRAIN_SECONDS - age)
        assert age + hold + REAUTH_RETIRE_DRAIN_SECONDS <= 300 - REAUTH_LEEWAY_SECONDS

    def test_past_the_point_it_retires_at_once(self) -> None:
        assert seconds_until_retirement(300, connection_age_seconds=280.0, longest_request_seconds=120.0) == 0.0

    def test_an_unknown_ttl_holds_for_the_longest_request(self) -> None:
        assert seconds_until_retirement(None, connection_age_seconds=1.0e6, longest_request_seconds=30.0) == 30.0

    def test_the_default_schedule_retires_before_expiry(self) -> None:
        """on schedule, the hold ends ``buffer`` short of the leeway, whatever the longest request."""
        ttl, longest = PLATFORM_DEFAULT_NATS_USER_JWT_TTL_SECONDS, 120.0
        swap = seconds_until_reauth(ttl, longest_request_seconds=longest)
        hold = seconds_until_retirement(ttl, connection_age_seconds=swap, longest_request_seconds=longest)
        assert hold == longest
        assert swap + hold + REAUTH_RETIRE_DRAIN_SECONDS < ttl - REAUTH_LEEWAY_SECONDS


class TestTheEnvironmentTtl:
    """a connection with no handshake reads the TTL the platform mints with."""

    def test_unset_is_the_platform_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """a day: the TTL is the backstop, and a kick is what takes access away now (owner ruling Q17)."""
        monkeypatch.delenv(_TTL_ENV, raising=False)
        assert nats_user_jwt_ttl_seconds() == 86_400

    def test_the_default_carries_the_synchronous_reply_budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """a default that could not hold a synchronous reply across a renewal would cut it off."""
        monkeypatch.delenv(_TTL_ENV, raising=False)
        ttl = nats_user_jwt_ttl_seconds()
        assert unsafe_renewal_reason(ttl, longest_request_seconds=SYNC_REPLY_BUDGET_SECONDS) is None

    def test_an_override_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_TTL_ENV, "600")
        assert nats_user_jwt_ttl_seconds() == 600

    @pytest.mark.parametrize("raw", ["0", "-30", "not-a-number"])
    def test_an_unusable_value_is_unknown_not_a_crash(self, monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
        monkeypatch.setenv(_TTL_ENV, raw)
        assert nats_user_jwt_ttl_seconds() is None


class TestAnUnsafeTtlIsNamed:
    """a TTL that cannot hold the longest request across a renewal cuts it off, and says why."""

    def test_a_ttl_that_cannot_carry_the_longest_request_is_explained(self) -> None:
        reason = unsafe_renewal_reason(150, longest_request_seconds=120.0)
        assert reason is not None
        assert "150" in reason
        assert _TTL_ENV in reason

    def test_a_ttl_that_can_is_safe(self) -> None:
        assert unsafe_renewal_reason(300, longest_request_seconds=120.0) is None

    def test_an_unknown_ttl_is_not_called_unsafe(self) -> None:
        assert unsafe_renewal_reason(None, longest_request_seconds=120.0) is None

    def test_the_reason_names_the_symptom_an_operator_sees(self) -> None:
        """nothing else in the logs connects "it answers, then hangs" to the TTL."""
        reason = unsafe_renewal_reason(150, longest_request_seconds=120.0)
        assert reason is not None
        assert "answers quickly, then hangs" in reason

    def test_the_boundary_is_exclusive(self) -> None:
        """the Hub refuses a TTL at or below longest + margin; the client agrees on the same number."""
        assert unsafe_renewal_reason(int(120 + REAUTH_MARGIN_SECONDS), longest_request_seconds=120.0) is not None
        assert unsafe_renewal_reason(int(121 + REAUTH_MARGIN_SECONDS), longest_request_seconds=120.0) is None


class TestOneDefaultTtl:
    """the TTL a connection assumes and the TTL the generic responder mints are one relation."""

    def test_the_assumed_ttl_never_exceeds_the_generic_mint(self) -> None:
        """assuming more than a minter mints is fatal; assuming less is only churn."""
        assert PLATFORM_DEFAULT_NATS_USER_JWT_TTL_SECONDS <= DEFAULT_NATS_USER_JWT_TTL_SECONDS

    def test_the_generic_mint_is_the_platform_backstop(self) -> None:
        """one number: a responder built without a TTL mints exactly what a client assumes."""
        assert DEFAULT_NATS_USER_JWT_TTL_SECONDS == PLATFORM_DEFAULT_NATS_USER_JWT_TTL_SECONDS


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
    monkeypatch.setattr("threetears.nats.client.seconds_until_reauth", lambda _ttl, **_kw: 0.01)
    monkeypatch.setattr("threetears.nats.client.REAUTH_MIN_SLEEP_SECONDS", 0.01)
    monkeypatch.setattr("threetears.nats.client.REAUTH_RETRY_SECONDS", 0.01)


def _user_info(expires_ns: object) -> bytes:
    """a ``$SYS.REQ.USER.INFO`` answer in the server's shape, carrying ``expires_ns``."""
    data: dict[str, object] = {"user": "u", "account": "$G"}
    if expires_ns is not _ABSENT:
        data["expires"] = expires_ns
    return json.dumps({"server": {"name": "nats-0"}, "data": data}).encode()


_ABSENT = object()


class TestTheServersAnswerIsRead:
    """``credential_lifetime_from_user_info``: the server's remaining time, made a whole lifetime."""

    def test_the_remaining_nanoseconds_plus_the_age_at_the_request_is_the_lifetime(self) -> None:
        reply = _user_info(240_500_000_000)  # 240.5 s remain
        assert credential_lifetime_from_user_info(reply, age_at_request=59.9, age_at_reply=60.1) == 300

    def test_it_rounds_down_so_any_error_renews_early(self) -> None:
        reply = _user_info(299_999_999_999)
        assert credential_lifetime_from_user_info(reply, age_at_request=0.0, age_at_reply=0.1) == 299

    @pytest.mark.parametrize("expires", [_ABSENT, 0])
    def test_a_credential_that_never_expires_is_none(self, expires: object) -> None:
        assert credential_lifetime_from_user_info(_user_info(expires), age_at_request=1.0, age_at_reply=1.0) is None

    def test_an_answer_that_spans_a_connection_swap_is_refused(self) -> None:
        """the new credential's remaining time plus the old connection's age overstates the lifetime."""
        reply = _user_info(300_000_000_000)
        with pytest.raises(ValueError, match="replaced while its credential lifetime was asked"):
            credential_lifetime_from_user_info(reply, age_at_request=250.0, age_at_reply=0.4)

    def test_a_server_error_is_refused(self) -> None:
        reply = json.dumps({"error": {"code": 503, "description": "events disabled"}}).encode()
        with pytest.raises(ValueError, match="refused the user-info request"):
            credential_lifetime_from_user_info(reply, age_at_request=1.0, age_at_reply=1.0)

    @pytest.mark.parametrize(
        "reply",
        [b"not json", b"[1, 2]", json.dumps({"server": {}}).encode(), _user_info("300s"), _user_info(True)],
    )
    def test_an_answer_not_in_the_user_info_shape_is_refused(self, reply: bytes) -> None:
        with pytest.raises(ValueError):
            credential_lifetime_from_user_info(reply, age_at_request=1.0, age_at_reply=1.0)


class TestTheClientRenewsItsOwnCredential:
    """the loop the three callers each carried, owned once by the client."""

    async def test_it_renews_on_schedule_holding_the_replaced_connection_for_the_longest_request(
        self, monkeypatch: pytest.MonkeyPatch, fast: None
    ) -> None:
        del fast
        holds: list[timedelta] = []
        renewed = asyncio.Event()

        async def _renew(self: NatsClient, *, retire_after: timedelta) -> None:
            holds.append(retire_after)
            renewed.set()

        monkeypatch.setattr(NatsClient, "renew_connection", _renew)
        client = _client()
        client.renew_credential(ttl_seconds=lambda: 300, longest_request_seconds=45.0)
        try:
            async with asyncio.timeout(2.0):
                await renewed.wait()
        finally:
            await client.shutdown()

        assert holds[0] == timedelta(seconds=45.0)

    async def test_the_schedule_is_measured_from_the_connection_not_the_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """armed late -- after a slow bootstrap -- the renewal must still beat the credential it has."""
        monkeypatch.setattr("threetears.nats.client.REAUTH_MIN_SLEEP_SECONDS", 0.01)
        renewed = asyncio.Event()

        async def _renew(self: NatsClient, *, retire_after: timedelta) -> None:
            renewed.set()

        monkeypatch.setattr(NatsClient, "renew_connection", _renew)
        client = _client()
        # the client's clock moves on as though the connection was established long enough ago
        # that its successor is already due; asyncio's own clock is left alone.
        later = SimpleNamespace(monotonic=lambda: time.monotonic() + 10_000.0)
        monkeypatch.setattr("threetears.nats.client.time", later)
        client.renew_credential(ttl_seconds=lambda: 300, longest_request_seconds=30.0)
        try:
            async with asyncio.timeout(1.0):
                await renewed.wait()
        finally:
            await client.shutdown()

    async def test_a_failed_renewal_is_retried_not_fatal(self, monkeypatch: pytest.MonkeyPatch, fast: None) -> None:
        del fast
        calls: list[int] = []
        recovered = asyncio.Event()

        async def _renew(self: NatsClient, *, retire_after: timedelta) -> None:
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("transient renewal failure")
            recovered.set()

        monkeypatch.setattr(NatsClient, "renew_connection", _renew)
        client = _client()
        client.renew_credential(ttl_seconds=lambda: 300)
        try:
            async with asyncio.timeout(2.0):
                await recovered.wait()
        finally:
            await client.shutdown()

        assert len(calls) >= 2

    async def test_a_server_that_does_not_answer_falls_back_to_the_configured_ttl(
        self, monkeypatch: pytest.MonkeyPatch, fast: None, caplog: pytest.LogCaptureFixture
    ) -> None:
        """an older grant without the user-info subject must not stop the renewal."""
        del fast
        renewed = asyncio.Event()

        async def _renew(self: NatsClient, *, retire_after: timedelta) -> None:
            renewed.set()

        async def _no_answer(self: NatsClient, *, timeout: timedelta = timedelta(seconds=2)) -> int | None:
            raise RequestTimeoutError("request timed out: subject=$SYS.REQ.USER.INFO")

        monkeypatch.setattr(NatsClient, "renew_connection", _renew)
        monkeypatch.setattr(NatsClient, "credential_ttl_from_server", _no_answer)
        client = _client()
        with caplog.at_level(logging.WARNING, logger="threetears.nats.client"):
            client.renew_credential(ttl_seconds=lambda: 300, ask_server=True)
            try:
                async with asyncio.timeout(2.0):
                    await renewed.wait()
            finally:
                await client.shutdown()

        assert any("did not report this connection's credential lifetime" in r.getMessage() for r in caplog.records)

    async def test_a_server_that_names_no_expiry_falls_back_and_says_so(
        self, monkeypatch: pytest.MonkeyPatch, fast: None, caplog: pytest.LogCaptureFixture
    ) -> None:
        """an answer with no lifetime must not change the schedule silently."""
        del fast
        renewed = asyncio.Event()

        async def _renew(self: NatsClient, *, retire_after: timedelta) -> None:
            renewed.set()

        async def _no_expiry(self: NatsClient, *, timeout: timedelta = timedelta(seconds=2)) -> int | None:
            return None

        monkeypatch.setattr(NatsClient, "renew_connection", _renew)
        monkeypatch.setattr(NatsClient, "credential_ttl_from_server", _no_expiry)
        client = _client()
        with caplog.at_level(logging.WARNING, logger="threetears.nats.client"):
            client.renew_credential(ttl_seconds=lambda: 300, ask_server=True)
            try:
                async with asyncio.timeout(2.0):
                    await renewed.wait()
            finally:
                await client.shutdown()

        assert any("reported no expiry" in r.getMessage() for r in caplog.records)

    async def test_the_servers_ttl_outranks_a_longer_configured_one(
        self, monkeypatch: pytest.MonkeyPatch, fast: None, caplog: pytest.LogCaptureFixture
    ) -> None:
        """a pod configured for a day renews on the 300 s the server reports it was minted."""
        del fast
        renewed = asyncio.Event()

        async def _renew(self: NatsClient, *, retire_after: timedelta) -> None:
            renewed.set()

        async def _server_says(self: NatsClient, *, timeout: timedelta = timedelta(seconds=2)) -> int | None:
            return 300

        monkeypatch.setattr(NatsClient, "renew_connection", _renew)
        monkeypatch.setattr(NatsClient, "credential_ttl_from_server", _server_says)
        client = _client()
        with caplog.at_level(logging.INFO, logger="threetears.nats.client"):
            client.renew_credential(ttl_seconds=lambda: 86_400, ask_server=True)
            try:
                async with asyncio.timeout(2.0):
                    await renewed.wait()
            finally:
                await client.shutdown()

        switched = [r for r in caplog.records if "shorter credential lifetime than configured" in r.getMessage()]
        assert switched, "the loop did not take the server's lifetime over the configured one"
        assert switched[0].extra_data == {  # type: ignore[attr-defined]
            "client_name": "renewal-test",
            "server_ttl_seconds": 300,
            "configured_ttl_seconds": 86_400,
        }

    async def test_a_shortfall_within_the_servers_rounding_is_not_reported_at_info(
        self, monkeypatch: pytest.MonkeyPatch, fast: None, caplog: pytest.LogCaptureFixture
    ) -> None:
        """the server's answer is rounded down by design, so a day reads back as 86399 s.

        Found on cobalt-dev: the INFO line fired on every renewal cycle for a lifetime that was
        exactly the configured one.
        """
        del fast
        renewed = asyncio.Event()

        async def _renew(self: NatsClient, *, retire_after: timedelta) -> None:
            renewed.set()

        async def _server_says(self: NatsClient, *, timeout: timedelta = timedelta(seconds=2)) -> int | None:
            return 86_399

        monkeypatch.setattr(NatsClient, "renew_connection", _renew)
        monkeypatch.setattr(NatsClient, "credential_ttl_from_server", _server_says)
        client = _client()
        with caplog.at_level(logging.DEBUG, logger="threetears.nats.client"):
            client.renew_credential(ttl_seconds=lambda: 86_400, ask_server=True)
            try:
                async with asyncio.timeout(2.0):
                    await renewed.wait()
            finally:
                await client.shutdown()

        switched = [r for r in caplog.records if "shorter credential lifetime than configured" in r.getMessage()]
        assert switched, "the shortfall within the rounding was not logged at all"
        assert all(r.levelno == logging.DEBUG for r in switched), [r.levelname for r in switched]

    async def test_a_steady_shortfall_is_reported_at_info_once(
        self, monkeypatch: pytest.MonkeyPatch, fast: None, caplog: pytest.LogCaptureFixture
    ) -> None:
        """the same server lifetime, cycle after cycle, is one fact, not one line per renewal."""
        del fast
        renewals = 0
        enough = asyncio.Event()

        async def _renew(self: NatsClient, *, retire_after: timedelta) -> None:
            nonlocal renewals
            renewals += 1
            if renewals >= 3:
                enough.set()

        async def _server_says(self: NatsClient, *, timeout: timedelta = timedelta(seconds=2)) -> int | None:
            return 300

        monkeypatch.setattr(NatsClient, "renew_connection", _renew)
        monkeypatch.setattr(NatsClient, "credential_ttl_from_server", _server_says)
        client = _client()
        with caplog.at_level(logging.DEBUG, logger="threetears.nats.client"):
            client.renew_credential(ttl_seconds=lambda: 86_400, ask_server=True)
            try:
                async with asyncio.timeout(5.0):
                    await enough.wait()
            finally:
                await client.shutdown()

        switched = [r for r in caplog.records if "shorter credential lifetime than configured" in r.getMessage()]
        assert len(switched) >= 3, "the loop did not measure the lifetime on every cycle"
        assert [r.levelno for r in switched].count(logging.INFO) == 1, [r.levelname for r in switched]

    async def test_a_failure_after_the_scheduled_sleep_retries_without_a_second_cycle(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """the retry replaces the schedule: a second full cycle would land after the credential expired.

        the schedule and the retry differ by four orders of magnitude, so a loop that sleeps the
        retry AND then the schedule again misses the deadline instead of passing by accident.
        """
        schedule = iter([0.01])
        monkeypatch.setattr("threetears.nats.client.seconds_until_reauth", lambda _ttl, **_kw: next(schedule, 60.0))
        monkeypatch.setattr("threetears.nats.client.REAUTH_MIN_SLEEP_SECONDS", 0.01)
        monkeypatch.setattr("threetears.nats.client.REAUTH_RETRY_SECONDS", 0.01)
        calls: list[int] = []
        recovered = asyncio.Event()

        async def _renew(self: NatsClient, *, retire_after: timedelta) -> None:
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("transient renewal failure")
            recovered.set()

        monkeypatch.setattr(NatsClient, "renew_connection", _renew)
        client = _client()
        client.renew_credential(ttl_seconds=lambda: 300)
        try:
            async with asyncio.timeout(1.0):
                await recovered.wait()
        finally:
            await client.shutdown()

        assert len(calls) == 2

    async def test_an_unknown_ttl_never_renews(self, monkeypatch: pytest.MonkeyPatch, fast: None) -> None:
        del fast
        calls: list[int] = []

        async def _renew(self: NatsClient, *, retire_after: timedelta) -> None:
            calls.append(1)

        monkeypatch.setattr(NatsClient, "renew_connection", _renew)
        client = _client()
        client.renew_credential(ttl_seconds=lambda: None)
        try:
            await asyncio.sleep(0.1)
        finally:
            await client.shutdown()

        assert calls == []

    async def test_an_unsafe_ttl_is_logged_as_an_error(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        async def _renew(self: NatsClient, *, retire_after: timedelta) -> None:
            return None

        monkeypatch.setattr(NatsClient, "renew_connection", _renew)
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
        ttl_now = {"value": 300}

        async def _renew(self: NatsClient, *, retire_after: timedelta) -> None:
            seen.append(ttl_now["value"])
            if ttl_now["value"] == 600:
                second_seen.set()

        def _first() -> int:
            ttl_now["value"] = 300
            return 300

        def _second() -> int:
            ttl_now["value"] = 600
            return 600

        monkeypatch.setattr(NatsClient, "renew_connection", _renew)
        client = _client()
        client.renew_credential(ttl_seconds=_first)
        client.renew_credential(ttl_seconds=_second)
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

        async def _renew(self: NatsClient, *, retire_after: timedelta) -> None:
            calls.append(1)

        monkeypatch.setattr(NatsClient, "renew_connection", _renew)
        client = _client()
        client.renew_credential(ttl_seconds=lambda: 300)
        await asyncio.sleep(0.05)

        client.raw.is_closed = True
        await client.shutdown()
        stopped_at = len(calls)
        await asyncio.sleep(0.05)

        assert len(calls) == stopped_at
