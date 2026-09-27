"""``ToolServer.shutdown`` releases ``serve`` even when a step of it raises.

Two tool pods logged ``NATS drain failed; forcing close`` with ``ConnectionResetError`` on SIGTERM
and stayed alive for two days: ``shutdown`` raised from the NATS drain before it set the event
``serve`` waits on, so ``serve`` never returned. The release now happens in ``finally``; the
exception still reaches the caller, which owns what a failed shutdown means.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from threetears.agent.tools.server import ToolServer


def _mock_nc(shutdown_error: BaseException) -> AsyncMock:
    """a NATS client double sufficient to drive ``serve``, whose own shutdown raises.

    :param shutdown_error: what the client's ``shutdown`` raises
    :ptype shutdown_error: BaseException
    :return: the double
    :rtype: AsyncMock
    """
    mock_nc = AsyncMock()
    mock_nc.is_connected = True
    mock_nc.subscribe = AsyncMock()
    mock_nc.publish = AsyncMock()
    mock_nc.renew_credential = MagicMock()
    mock_nc.request_raw = AsyncMock(return_value=json.dumps({"keys": []}).encode("utf-8"))
    mock_nc.shutdown = AsyncMock(side_effect=shutdown_error)
    return mock_nc


class TestAFailedDrainStillReleasesServe:
    """the pod's own serve loop ends; the failure is not swallowed."""

    @pytest.mark.asyncio
    async def test_serve_returns_and_the_error_propagates(self) -> None:
        """``shutdown`` raises the drain's error, and ``serve`` returns anyway.

        :return: nothing
        :rtype: None
        """
        server = ToolServer(nats_url="nats://localhost:9999", namespace="testns", pod_id="shutdown-pod")
        mock_nc = _mock_nc(ConnectionResetError("connection reset by peer"))

        with patch("threetears.agent.tools.server.nats_connect", return_value=mock_nc):
            serve_task = asyncio.create_task(server.serve())
            await asyncio.sleep(0.05)
            with pytest.raises(ConnectionResetError):
                await server.shutdown()
            await asyncio.wait_for(serve_task, timeout=1.0)

        mock_nc.shutdown.assert_awaited_once()
