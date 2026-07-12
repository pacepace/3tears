"""in-flight-requests gauge wiring for the registry :class:`CallProxy`.

the proxy brackets every ``_process_call`` in a leak-safe in-flight gauge
so KEDA's prometheus scaler can autoscale registry replicas on aggregate
in-flight tool-call load. the leak-safe bracket semantics themselves are
proven in ``packages/observe/tests/test_inflight.py``; these tests prove
the proxy actually wraps its dispatch (the gauge returns to baseline after
a call, on both the clean and the raising path).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from threetears.nats import IncomingMessage
from threetears.observe import InflightRequestsGauge

from ._dispatch_auth import make_proxy


def _malformed_msg() -> IncomingMessage:
    """build a call message whose body fails ``ProxyCallRequest`` decode.

    the malformed-parse branch runs entirely inside the in-flight
    bracket, so it exercises the gauge inc/dec without needing the full
    identity + pop verification a well-formed call would.

    :return: incoming envelope carrying non-JSON payload bytes
    :rtype: IncomingMessage
    """
    return IncomingMessage(data=b"not json", reply_subject="_INBOX.inflight", subject="3tears.tools.call")


@pytest.mark.asyncio
async def test_process_call_returns_gauge_to_baseline_on_clean_path() -> None:
    """after a dispatched call the in-flight gauge is back to ``0``."""
    from threetears.registry.catalog import ToolCatalog

    gauge = InflightRequestsGauge("threetears_registry_inflight_requests")
    proxy = make_proxy(ToolCatalog(), inflight_gauge=gauge)
    nc = AsyncMock()
    await proxy.start(nc)
    assert gauge.value == 0.0

    await proxy.handle_call(_malformed_msg())
    await proxy.stop()  # drains the spawned _process_call task

    assert gauge.value == 0.0


@pytest.mark.asyncio
async def test_process_call_returns_gauge_to_baseline_on_exception() -> None:
    """LEAK-SAFETY: a raising dispatch still decrements the gauge.

    the reply publish is wired to raise, so ``_process_call`` propagates
    an exception out of the in-flight bracket; the gauge must still unwind
    to its baseline (a stranded counter would make KEDA over-scale).
    """
    from threetears.registry.catalog import ToolCatalog

    gauge = InflightRequestsGauge("threetears_registry_inflight_requests")
    proxy = make_proxy(ToolCatalog(), inflight_gauge=gauge)
    nc = AsyncMock()
    nc.publish_reply.side_effect = RuntimeError("publish boom")
    await proxy.start(nc)

    await proxy.handle_call(_malformed_msg())
    await proxy.stop()  # gathers the raising task with return_exceptions

    assert gauge.value == 0.0
