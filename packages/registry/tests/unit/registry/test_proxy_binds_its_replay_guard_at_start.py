"""the proxy binds its pop replay guard when it starts, before it can receive a call.

After a broker restart the guard refuses every proof issued before its bucket's creation time plus
its reach, and the bucket is created by whoever opens it first. Left to the first call, that call
creates the bucket and is refused as a replay it is not. Binding in :meth:`CallProxy.start`, before
the call subject is subscribed, puts the creation time before any proof this replica can see.

The same move puts the bucket's failure at startup: a proxy that cannot open its nonce bucket must
fail to start, with nothing subscribed, rather than come up and refuse every call.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from threetears.core.testing.replay_guard import FakeReplayGuard
from threetears.nats import KvError
from threetears.registry.catalog import ToolCatalog

from ._dispatch_auth import make_proxy


@pytest.mark.asyncio
async def test_start_binds_the_pop_guard_before_subscribing() -> None:
    order: list[str] = []
    nc = AsyncMock()
    nc.subscribe = AsyncMock(side_effect=lambda **_: order.append("subscribe"))
    proxy = make_proxy(ToolCatalog(), pop_replay_guard=FakeReplayGuard(events=order))

    await proxy.start(nc)

    assert order == ["bind", "subscribe"], order


@pytest.mark.asyncio
async def test_a_bucket_that_cannot_be_opened_fails_start_with_nothing_subscribed() -> None:
    # a later edit that caught this to keep the proxy up would reopen the first-use window and
    # leave every call refused; the failure belongs at startup, where a restart answers it.
    nc = AsyncMock()
    guard = FakeReplayGuard(bind_error=KvError("pop_nonces bucket unavailable"))
    proxy = make_proxy(ToolCatalog(), pop_replay_guard=guard)

    with pytest.raises(KvError, match="pop_nonces"):
        await proxy.start(nc)

    assert guard.binds == 1
    nc.subscribe.assert_not_awaited()
