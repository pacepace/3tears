"""tests for the tool-pod proactive NATS-JWT re-auth lifecycle (platform-auth Option A, pod side).

A standalone tool pod opens its OWN NATS connection via the auth-callout, which mints a user JWT with a
finite TTL. At expiry the NATS server closes the connection in a way nats-py routes STRAIGHT to a
terminal close -- forever-reconnect does NOT cover it -- so the pod's data plane wedges and its
heartbeat supervisor eventually ``os._exit``s the process. The agent runtime already avoids this with a
proactive re-auth loop; these tests pin the SAME contract on the tool pod:

1. the schedule forces a reconnect BEFORE expiry (and never busy-spins on a tiny TTL);
2. an unknown / non-positive TTL re-checks on a cadence WITHOUT reconnecting on a guess;
3. the TTL comes from the pod's own config (``FOURTEENAIBOTS_NATS_USER_JWT_TTL_SECONDS``, default 150);
4. ``_reauth_nats_once`` drives the wrapper's ``reconnect()`` (which re-mints a fresh JWT);
5. the re-auth loop is UNKILLABLE -- a raise in the reconnect path is logged + retried, never fatal;
6. the loop does NOT reconnect while the TTL is unknown;
7. ``serve`` starts the loop only when the pod OWNS its connection, and ``shutdown`` cancels it.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from threetears.agent.tools import nats_reauth
from threetears.agent.tools.config import get_nats_user_jwt_ttl_seconds
from threetears.agent.tools.server import ToolServer

_TTL_ENV = "FOURTEENAIBOTS_NATS_USER_JWT_TTL_SECONDS"


class TestSecondsUntilReauth:
    """the schedule forces a reconnect BEFORE expiry, clamps a tiny TTL off the hot path, and
    re-checks (never reconnects) on an unknown TTL."""

    def test_schedules_before_expiry(self) -> None:
        """the scheduled re-auth lands strictly before the TTL elapses -- leeway + buffer early."""
        ttl = 180
        delay = nats_reauth.seconds_until_reauth(ttl)
        assert delay < ttl
        expected = ttl - nats_reauth.REAUTH_LEEWAY_SECONDS - nats_reauth.REAUTH_BUFFER_SECONDS
        assert delay == pytest.approx(expected)

    def test_default_ttl_schedules_a_positive_margin(self) -> None:
        """the platform-default TTL (150) reconnects at 60s -- comfortably before expiry."""
        delay = nats_reauth.seconds_until_reauth(150)
        assert delay == pytest.approx(60.0)

    def test_tiny_ttl_clamps_to_min_sleep(self) -> None:
        """a TTL smaller than the margin re-auths promptly but never busy-spins the event loop."""
        delay = nats_reauth.seconds_until_reauth(10)
        assert delay == nats_reauth.REAUTH_MIN_SLEEP_SECONDS

    def test_unknown_ttl_uses_recheck_interval(self) -> None:
        """no TTL (or a non-positive one) -> re-check soon, never reconnect on a guess."""
        assert nats_reauth.seconds_until_reauth(None) == nats_reauth.REAUTH_UNKNOWN_TTL_RECHECK_SECONDS
        assert nats_reauth.seconds_until_reauth(0) == nats_reauth.REAUTH_UNKNOWN_TTL_RECHECK_SECONDS
        assert nats_reauth.seconds_until_reauth(-5) == nats_reauth.REAUTH_UNKNOWN_TTL_RECHECK_SECONDS


class TestHasSchedulableTtl:
    """the shared predicate the scheduler AND the loop gate on (so they never diverge)."""

    def test_positive_ttl_is_schedulable(self) -> None:
        assert nats_reauth.has_schedulable_ttl(3600) is True
        assert nats_reauth.has_schedulable_ttl(1) is True

    def test_none_and_non_positive_are_not_schedulable(self) -> None:
        assert nats_reauth.has_schedulable_ttl(None) is False
        assert nats_reauth.has_schedulable_ttl(0) is False
        assert nats_reauth.has_schedulable_ttl(-5) is False


class TestNatsUserJwtTtlConfig:
    """the pod sources the TTL from its own env (the hub default value + env-var name)."""

    def test_defaults_to_platform_default_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """unset -> the platform default of 150 (mirrors the hub auth-callout responder default)."""
        monkeypatch.delenv(_TTL_ENV, raising=False)
        assert get_nats_user_jwt_ttl_seconds() == 150

    def test_env_override_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """a valid positive override is honored."""
        monkeypatch.setenv(_TTL_ENV, "300")
        assert get_nats_user_jwt_ttl_seconds() == 300

    def test_non_positive_is_unknown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """a <= 0 value is treated as unknown (None) so the loop re-checks rather than churns."""
        monkeypatch.setenv(_TTL_ENV, "0")
        assert get_nats_user_jwt_ttl_seconds() is None
        monkeypatch.setenv(_TTL_ENV, "-30")
        assert get_nats_user_jwt_ttl_seconds() is None

    def test_malformed_is_unknown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """a malformed value is treated as unknown (None), never a crash."""
        monkeypatch.setenv(_TTL_ENV, "not-a-number")
        assert get_nats_user_jwt_ttl_seconds() is None


def _server_with_nc(nc: object) -> ToolServer:
    """build a ToolServer with a NATS client bound on its slot for the re-auth methods.

    :param nc: the NATS client stand-in whose ``reconnect`` the re-auth drives
    :ptype nc: object
    :return: a server ready for the re-auth methods
    :rtype: ToolServer
    """
    server = ToolServer(nats_url="nats://localhost:4222", namespace_collection=None)
    # install the client on the server's NATS slot without dialing a real connection.
    setattr(server, "_nc", nc)
    return server


class TestReauthNatsOnce:
    """one re-auth pass forces a wrapper reconnect (which re-mints a fresh JWT)."""

    @pytest.mark.asyncio
    async def test_drives_wrapper_reconnect(self) -> None:
        """the per-pass re-auth calls ``NatsClient.reconnect`` exactly once."""
        nc = MagicMock()
        nc.reconnect = AsyncMock()
        server = _server_with_nc(nc)

        await server._reauth_nats_once()  # noqa: SLF001

        nc.reconnect.assert_awaited_once_with()

    @pytest.mark.asyncio
    async def test_skips_when_no_nats_client(self) -> None:
        """no nats_client -> no reconnect attempt (defensive no-op)."""
        server = ToolServer(nats_url="nats://localhost:4222", namespace_collection=None)
        # _nc is None before serve(); must not raise.
        await server._reauth_nats_once()  # noqa: SLF001


class TestNatsReauthLoop:
    """the loop reconnects on schedule, SURVIVES a failing reconnect, and never reconnects on an
    unknown TTL."""

    @pytest.mark.asyncio
    async def test_loop_reconnects_on_schedule(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """the running loop honors the schedule and forces a reconnect before expiry."""
        monkeypatch.setattr(nats_reauth, "seconds_until_reauth", lambda _ttl: 0.01)
        monkeypatch.setenv(_TTL_ENV, "150")

        nc = MagicMock()
        reconnected = asyncio.Event()

        async def _reconnect() -> None:
            reconnected.set()

        nc.reconnect = _reconnect
        server = _server_with_nc(nc)

        task = asyncio.ensure_future(server._nats_reauth_loop())  # noqa: SLF001
        try:
            async with asyncio.timeout(2.0):
                await reconnected.wait()
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        assert reconnected.is_set()

    @pytest.mark.asyncio
    async def test_loop_survives_exception_in_reconnect(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """a raise in one reconnect is logged + retried; the loop keeps running and re-auths next cycle."""
        monkeypatch.setattr(nats_reauth, "seconds_until_reauth", lambda _ttl: 0.0)
        monkeypatch.setattr(nats_reauth, "REAUTH_RETRY_SECONDS", 0.01)
        monkeypatch.setenv(_TTL_ENV, "150")

        server = _server_with_nc(MagicMock())

        calls: list[int] = []
        second_succeeded = asyncio.Event()

        async def _flaky_reauth() -> None:
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("transient reconnect failure")
            second_succeeded.set()

        monkeypatch.setattr(server, "_reauth_nats_once", _flaky_reauth)

        task = asyncio.ensure_future(server._nats_reauth_loop())  # noqa: SLF001
        try:
            async with asyncio.timeout(2.0):
                await second_succeeded.wait()
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        assert len(calls) >= 2

    @pytest.mark.asyncio
    async def test_loop_does_not_reconnect_when_ttl_unknown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """while the TTL is unknown the loop re-checks but NEVER forces a reconnect on a guess."""
        monkeypatch.setattr(nats_reauth, "seconds_until_reauth", lambda _ttl: 0.01)
        # unknown TTL: a malformed env value resolves to None.
        monkeypatch.setenv(_TTL_ENV, "0")

        reauth = AsyncMock()
        server = _server_with_nc(MagicMock())
        monkeypatch.setattr(server, "_reauth_nats_once", reauth)

        task = asyncio.ensure_future(server._nats_reauth_loop())  # noqa: SLF001
        try:
            await asyncio.sleep(0.1)  # let the loop re-check many times
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        reauth.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_loop_cancels_cleanly(self) -> None:
        """cancellation ends the loop with no escaped exception (clean CancelledError return)."""
        server = _server_with_nc(MagicMock())
        task = asyncio.ensure_future(server._nats_reauth_loop())  # noqa: SLF001
        await asyncio.sleep(0)  # let the loop start
        task.cancel()
        # the loop must swallow CancelledError and return cleanly.
        await task
        assert task.cancelled() is False
        assert task.done() is True


def _mock_owned_nc() -> AsyncMock:
    """build a mock NATS client sufficient to drive ``serve`` on the owned-connection path.

    :return: mock client whose serve dependencies parse (subscribe / publish / JWKS)
    :rtype: AsyncMock
    """
    mock_nc = AsyncMock()
    mock_nc.is_connected = True
    mock_nc.subscribe = AsyncMock()
    mock_nc.publish = AsyncMock()
    mock_nc.drain = AsyncMock()
    mock_nc.close = AsyncMock()
    mock_nc.reconnect = AsyncMock()
    # serve() self-provisions a Hub-JWKS provider (enforce-only); give the mock a JWKS reply so the
    # best-effort initial fetch parses instead of choking on a bare mock.
    mock_nc.request_raw = AsyncMock(return_value=json.dumps({"keys": []}).encode("utf-8"))
    return mock_nc


class TestServeWiring:
    """``serve`` starts the re-auth loop on the owned-connection path; ``shutdown`` cancels it."""

    @pytest.mark.asyncio
    async def test_serve_starts_reauth_loop_when_owning_connection(self) -> None:
        """a standalone pod (nats_url) owns its connection, so serve starts the proactive re-auth loop."""
        server = ToolServer(
            nats_url="nats://localhost:9999",
            namespace="testns",
            pod_id="reauth-pod",
            namespace_collection=None,
        )
        mock_nc = _mock_owned_nc()

        with patch("threetears.agent.tools.server.nats_connect", return_value=mock_nc):
            serve_task = asyncio.create_task(server.serve())
            await asyncio.sleep(0.05)
            try:
                assert server._nats_reauth_task is not None  # noqa: SLF001
                assert server._nats_reauth_task.done() is False  # noqa: SLF001
            finally:
                await server.shutdown()
                await asyncio.sleep(0.05)
                serve_task.cancel()
                try:
                    await serve_task
                except asyncio.CancelledError:
                    pass

        # shutdown cancelled + cleared the task.
        assert server._nats_reauth_task is None  # noqa: SLF001

    @pytest.mark.asyncio
    async def test_serve_skips_reauth_loop_on_injected_connection(self) -> None:
        """an agent-owned (injected) connection has its OWN re-auth loop; the pod must NOT double-drive it."""
        mock_nc = _mock_owned_nc()
        server = ToolServer(
            nats_client=mock_nc,
            namespace="testns",
            pod_id="shared-pod",
            namespace_collection=None,
        )

        serve_task = asyncio.create_task(server.serve())
        await asyncio.sleep(0.05)
        try:
            assert server._nats_reauth_task is None  # noqa: SLF001
        finally:
            await server.shutdown()
            await asyncio.sleep(0.05)
            serve_task.cancel()
            try:
                await serve_task
            except asyncio.CancelledError:
                pass
