"""the proxy binds its pop replay guard when it starts, before it can receive a call.

After a broker restart the guard refuses every proof issued before its bucket's creation time plus
its reach, and the bucket is created by whoever opens it first. Left to the first call, that call
creates the bucket and is refused as a replay it is not. Binding in :meth:`CallProxy.start`, before
the call subject is subscribed, puts the creation time before any proof this replica can see.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from threetears.registry.catalog import ToolCatalog

from ._dispatch_auth import make_proxy


class _OrderRecordingReplayGuard:
    """records its bind into the same ordered log the NATS double records its subscribe in."""

    def __init__(self, order: list[str]) -> None:
        self._order = order

    def require_covers(self, future_tolerance: timedelta) -> None:
        """a stub guard is sized for any verifier; the real check has its own tests."""

    async def bind(self) -> None:
        """record that the proxy bound the guard."""
        self._order.append("bind")

    async def record_unique(self, nonce: str, *, issued_at: datetime) -> bool:
        """never reached: no call is dispatched here."""
        del nonce, issued_at
        return True


@pytest.mark.asyncio
async def test_start_binds_the_pop_guard_before_subscribing() -> None:
    order: list[str] = []
    nc = AsyncMock()
    nc.subscribe = AsyncMock(side_effect=lambda **_: order.append("subscribe"))
    proxy = make_proxy(ToolCatalog(), pop_replay_guard=_OrderRecordingReplayGuard(order))

    await proxy.start(nc)

    assert order == ["bind", "subscribe"], order
