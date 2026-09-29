"""a standalone tool pod renews the credential on the connection it opened; an agent's is left alone.

A standalone tool pod opens its OWN NATS connection via the auth-callout, which mints a user JWT with a
finite TTL, and at expiry the server closes it in a way forever-reconnect does not cover. The renewal
itself -- the schedule, the loop, surviving a failed renewal, never reconnecting on an unknown TTL -- is
:meth:`threetears.nats.NatsClient.renew_credential`'s, and is tested there. What the pod owns, and what
these tests pin:

1. ``serve`` asks the client to renew only when the pod OWNS its connection and the auth-callout
   minted its credential -- a static or anonymous credential never expires;
2. the TTL comes from the pod's environment, read every cycle;
3. before each renewal the pod drains the replies it still owes, since a reply cannot be delivered once
   the connection that received its request is gone, and the renewal loop credits that drain when it
   judges whether the cadence can carry a synchronous call;
4. an injected (agent-owned) connection is renewed by its owner, so the pod never double-drives it.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from threetears.agent.tools.server import ToolServer
from threetears.nats import REAUTH_BUFFER_SECONDS, SYNC_REPLY_BUDGET_SECONDS

_TTL_ENV = "FOURTEENAIBOTS_NATS_USER_JWT_TTL_SECONDS"


def _callout_pod() -> ToolServer:
    """a standalone pod that authenticates through the auth-callout, as production pods do.

    :return: the server
    :rtype: ToolServer
    """
    return ToolServer(
        nats_url="nats://localhost:9999",
        namespace="testns",
        pod_id="reauth-pod",
        auth_token=lambda: "identity-jwt",
    )


def _mock_nc() -> AsyncMock:
    """a mock NATS client sufficient to drive ``serve``.

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
    # renew_credential is synchronous: it starts a loop and returns.
    mock_nc.renew_credential = MagicMock()
    # serve() self-provisions a Hub-JWKS provider (enforce-only); give the mock a JWKS reply so the
    # best-effort initial fetch parses instead of choking on a bare mock.
    mock_nc.request_raw = AsyncMock(return_value=json.dumps({"keys": []}).encode("utf-8"))
    return mock_nc


async def _serve_briefly(server: ToolServer) -> None:
    """run ``serve`` long enough to finish wiring, then shut it down.

    :param server: the server to run
    :ptype server: ToolServer
    :return: nothing
    :rtype: None
    """
    serve_task = asyncio.create_task(server.serve())
    await asyncio.sleep(0.05)
    try:
        await server.shutdown()
        await asyncio.sleep(0.05)
    finally:
        serve_task.cancel()
        try:
            await serve_task
        except asyncio.CancelledError:
            pass


class TestServeWiring:
    """``serve`` asks the client it opened to renew its credential, and no other."""

    @pytest.mark.asyncio
    async def test_a_pod_renews_the_connection_it_opened(self) -> None:
        server = _callout_pod()
        mock_nc = _mock_nc()

        with patch("threetears.agent.tools.server.nats_connect", return_value=mock_nc):
            await _serve_briefly(server)

        mock_nc.renew_credential.assert_called_once()
        kwargs = mock_nc.renew_credential.call_args.kwargs
        assert kwargs["before_renewal"] == server.drain_before_reauth
        assert kwargs["longest_request_seconds"] == SYNC_REPLY_BUDGET_SECONDS
        # the drain's slack is credited to the cadence, so the one judge of the TTL sees it.
        assert kwargs["drain_grace_seconds"] == REAUTH_BUFFER_SECONDS

    @pytest.mark.asyncio
    async def test_a_static_credential_is_never_renewed(self) -> None:
        """a user/password connection holds a credential that does not expire."""
        server = ToolServer(
            nats_url="nats://localhost:9999",
            namespace="testns",
            pod_id="static-pod",
            nats_user="tool-pod",
            nats_password="not-a-real-password",
        )
        mock_nc = _mock_nc()

        with patch("threetears.agent.tools.server.nats_connect", return_value=mock_nc):
            await _serve_briefly(server)

        mock_nc.renew_credential.assert_not_called()

    @pytest.mark.asyncio
    async def test_the_ttl_is_read_from_the_environment_every_cycle(self, monkeypatch: pytest.MonkeyPatch) -> None:
        server = _callout_pod()
        mock_nc = _mock_nc()
        with patch("threetears.agent.tools.server.nats_connect", return_value=mock_nc):
            await _serve_briefly(server)
        ttl_source = mock_nc.renew_credential.call_args.kwargs["ttl_seconds"]

        monkeypatch.setenv(_TTL_ENV, "600")
        assert ttl_source() == 600
        monkeypatch.setenv(_TTL_ENV, "900")
        assert ttl_source() == 900

    @pytest.mark.asyncio
    async def test_an_injected_connection_is_left_to_its_owner(self) -> None:
        mock_nc = _mock_nc()
        server = ToolServer(nats_client=mock_nc, namespace="testns", pod_id="shared-pod")

        await _serve_briefly(server)

        mock_nc.renew_credential.assert_not_called()
