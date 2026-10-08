"""a NATS L3 proxy whose broker answers from a script, shared by the proxy's decode tests.

Each test drives the proxy's own request path with reply dicts JSON-encoded as the hub sends them.
"""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from threetears.nats import Subject

from threetears.core.backends.nats_proxy import NatsProxyL3Backend

__all__ = ["TX_ID", "scripted_proxy"]

#: the transaction id a scripted ``begin`` answers with
TX_ID = "019d9a00-0000-7000-8000-000000000000"


def scripted_proxy(*replies: dict[str, Any]) -> NatsProxyL3Backend:
    """a proxy whose broker answers each request with the next scripted reply.

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
