"""in-flight-requests gauge wiring for the tool-pod :class:`ToolServer`.

the pod brackets every ``handle_call`` in a leak-safe in-flight gauge and
serves it through :meth:`ToolServer.render_metrics` (wired onto the shared
HealthServer's ``/metrics`` route by ``ToolServerBootstrap``) so KEDA's
prometheus scaler can autoscale the tool-pod Deployment on aggregate
in-flight call load. the leak-safe bracket semantics themselves are proven
in ``packages/observe/tests/test_inflight.py``; these tests prove the pod
actually exposes and wraps its dispatch, on both the clean and raising
path, through the public surface only.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from threetears.agent.tools.server import ToolServer
from threetears.nats import IncomingMessage

_METRIC_NAME = "threetears_tools_inflight_requests"


def _inflight_value(server: ToolServer) -> float:
    """extract the in-flight gauge value from the pod's rendered metrics.

    parses :meth:`ToolServer.render_metrics` (the public /metrics body) so
    the assertion reads the same exposition KEDA scrapes rather than a
    private attribute.

    :param server: tool server whose metrics to read
    :ptype server: ToolServer
    :return: the current in-flight gauge value, or ``-1.0`` when absent
    :rtype: float
    """
    _content_type, body = server.render_metrics()
    result = -1.0
    for line in body.decode("utf-8").splitlines():
        if line.startswith(f"{_METRIC_NAME} "):
            result = float(line.split()[1])
    return result


def _malformed_msg() -> IncomingMessage:
    """build a call message whose body fails ``CallRequest`` decode.

    the malformed-parse branch runs inside the in-flight bracket, so it
    exercises the gauge inc/dec without the full identity + proxy-
    assertion verification a well-formed call would need.

    :return: incoming envelope carrying non-JSON payload bytes
    :rtype: IncomingMessage
    """
    return IncomingMessage(data=b"not json", reply_subject="_INBOX.inflight", subject="3tears.tools.internal.pod")


@pytest.mark.asyncio
async def test_render_metrics_exposes_inflight_gauge_at_baseline() -> None:
    """a fresh pod's /metrics body carries the gauge at ``0``."""
    server = ToolServer(nats_url="nats://stub", namespace_collection=None)
    assert _inflight_value(server) == 0.0


@pytest.mark.asyncio
async def test_handle_call_returns_gauge_to_baseline_on_clean_path() -> None:
    """after a dispatched call the in-flight gauge is back to ``0``.

    the pod has no NATS client wired (``nats_url`` only), so the malformed
    branch's reply + audit are no-ops and the call completes cleanly
    through the bracket.
    """
    server = ToolServer(nats_url="nats://stub", namespace_collection=None)
    await server.handle_call(_malformed_msg())
    assert _inflight_value(server) == 0.0


@pytest.mark.asyncio
async def test_handle_call_returns_gauge_to_baseline_on_exception() -> None:
    """LEAK-SAFETY: a raising dispatch still decrements the gauge.

    the reply publish is wired to raise, so ``handle_call`` propagates an
    exception out of the in-flight bracket; the gauge must still unwind to
    baseline (a stranded counter would make KEDA over-scale forever).
    """
    nats = AsyncMock()
    nats.publish_reply.side_effect = RuntimeError("publish boom")
    server = ToolServer(nats_client=nats, namespace_collection=None)

    with pytest.raises(RuntimeError, match="publish boom"):
        await server.handle_call(_malformed_msg())

    assert _inflight_value(server) == 0.0
