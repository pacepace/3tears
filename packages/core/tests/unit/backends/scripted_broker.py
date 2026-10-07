"""a NatsProxyL3Backend whose broker answers each request with a scripted reply.

Shared by the tests that drive the proxy's real decode of a broker reply: each reply dict is
JSON-encoded as the hub sends it and handed back through the proxy's own request path.
"""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from threetears.nats import Subject

from threetears.core.backends.nats_proxy import NatsProxyL3Backend

#: a transaction id for tests that open one through the scripted broker.
TX_ID = "019d9a00-0000-7000-8000-000000000000"


def scripted_proxy(*replies: dict[str, Any]) -> NatsProxyL3Backend:
    """
    a proxy whose broker answers each request with the next scripted reply.

    :param replies: reply dicts, in request order
    :ptype replies: dict[str, Any]
    :return: the proxy
    :rtype: NatsProxyL3Backend
    """
    queue = list(replies)

    async def request_raw(*, subject: Subject, payload: bytes, timeout: timedelta | None = None) -> bytes:
        del subject, payload, timeout
        return json.dumps(queue.pop(0)).encode("utf-8")

    nats = MagicMock()
    nats.request_raw = AsyncMock(side_effect=request_raw)
    return NatsProxyL3Backend(
        nats_client=nats,
        namespace_prefix="test",
        agent_id="agent-123",
        identity_token=lambda: "test-identity-token",
    )
