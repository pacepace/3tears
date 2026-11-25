"""A transaction asks a deadline of its own, so a write queued at the broker behind others is not refused at a
serving read's five seconds.

The hub refuses ``QUERY_TIMEOUT`` a request that waited for a broker slot past the deadline its pod asked (a
transaction's requests: the ``statement_timeout_ms`` its ``tx.begin`` named), and the pod stops waiting for a reply
two seconds past it. Every transaction asked the backend's ``timeout_ms`` (five seconds, sized for a serving read), so
under the 2026-11-24 replay's load a page whose COPY takes under two seconds was refused after queueing, and after its
last try the whole load failed. ``conn.transaction(timeout_ms=...)`` asks the deadline sized to the write; a read
outside one keeps the backend's.
"""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from threetears.nats import Subject
from threetears.nats.errors import RequestTimeoutError

from threetears.core.backends.nats_proxy import NatsProxyL3Backend, _ProxyConnection
from threetears.core.collections.caller_transaction import CallerTransaction
from threetears.core.exceptions import DataLayerUnavailableError

TX_ID = "019d9a00-0000-7000-8000-0000000000aa"

#: how long each request waits for a broker slot before the hub reaches it: past a serving read's five seconds
QUEUED_S = 6.0


class _QueuedHub:
    """a hub where every request queues :data:`QUEUED_S` for a slot, and answers as the broker's
    ``_expired_while_queued`` does: a request whose deadline ran out while it queued is refused ``QUERY_TIMEOUT``;
    a pod that stopped waiting before then gets no reply at all.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any], float]] = []
        self._session_ms: int | None = None

    async def __call__(self, *, subject: Subject, payload: bytes, timeout: timedelta | None = None) -> bytes:
        path = subject.path
        body = json.loads(payload.decode("utf-8"))
        waited = timeout.total_seconds() if timeout is not None else 0.0
        self.calls.append((path, body, waited))
        if waited < QUEUED_S:
            raise RequestTimeoutError(f"request timed out: subject={path} timeout={waited:.1f}s")
        if path.endswith(".l3.tx.begin"):
            self._session_ms = int(body["statement_timeout_ms"])
            deadline_ms = self._session_ms
        elif ".l3.tx." in path:
            assert self._session_ms is not None, f"{path} outside a session"
            deadline_ms = self._session_ms
        else:
            deadline_ms = int(body["timeout_ms"])
        if QUEUED_S * 1000 > deadline_ms:
            reply: dict[str, Any] = {
                "success": False,
                "error_code": "QUERY_TIMEOUT",
                "error_message": f"request waited {QUEUED_S:.3f}s for the broker, past the {deadline_ms}ms timeout; "
                "not run",
            }
        elif path.endswith(".l3.tx.begin"):
            reply = {"success": True, "tx_id": TX_ID}
        elif path.endswith(".l3.tx.execute"):
            reply = {"success": True, "row_count": 1}
        elif ".l3.tx." in path:
            reply = {"success": True}
        else:
            reply = {"success": True, "rows": [], "row_count": 0}
        return json.dumps(reply).encode("utf-8")


def _proxy(hub: _QueuedHub) -> NatsProxyL3Backend:
    nc = MagicMock()
    nc.request_raw = hub
    return NatsProxyL3Backend(
        nats_client=nc,
        namespace_prefix="test",
        agent_id="agent-abc",
        identity_token=lambda: "test-identity-token",
        timeout_ms=5000,
    )


@pytest.mark.asyncio
async def test_a_write_queued_past_five_seconds_lands_within_its_own_deadline() -> None:
    """Flips if a transaction's deadline is not its own: its begin asks the backend's five seconds, or a request of it
    is waited for only that long, so a write that queued six seconds behind others is refused or given up on."""
    hub = _QueuedHub()
    proxy = _proxy(hub)
    async with proxy.acquire() as conn:
        async with conn.transaction(timeout_ms=30_000):
            await conn.execute("INSERT INTO t(x) VALUES ($1)", 1)
    assert [path.rsplit(".", 1)[-1] for path, _, _ in hub.calls] == ["begin", "execute", "commit"]
    assert hub.calls[0][1]["statement_timeout_ms"] == 30_000
    # every request of the session waited its deadline and the grace, not the backend's seven seconds
    assert {waited for _, _, waited in hub.calls} == {32.0}


@pytest.mark.asyncio
async def test_caller_transaction_hands_the_deadline_to_the_broker_connection() -> None:
    """Flips if CallerTransaction drops the deadline its caller sized (the pod's page writes open theirs so)."""
    hub = _QueuedHub()
    proxy = _proxy(hub)
    async with proxy.acquire() as conn:
        async with CallerTransaction(conn, timeout_ms=30_000):
            await conn.execute("INSERT INTO t(x) VALUES ($1)", 1)
    assert hub.calls[0][1]["statement_timeout_ms"] == 30_000


class _DirectConnection:
    """an asyncpg-shaped connection: its ``transaction()`` takes isolation and the like, never ``timeout_ms``."""

    def __init__(self) -> None:
        self.asked: list[dict[str, Any]] = []

    def transaction(self, *, isolation: str | None = None) -> AsyncMock:
        self.asked.append({"isolation": isolation})
        opened = AsyncMock()
        opened.__aexit__.return_value = False
        return opened


@pytest.mark.asyncio
async def test_caller_transaction_hands_no_deadline_to_a_direct_pool_connection() -> None:
    """Flips if a connection with no broker (asyncpg's, whose ``transaction()`` takes no ``timeout_ms``) is handed one,
    which it would refuse."""
    conn = _DirectConnection()
    async with CallerTransaction(conn, timeout_ms=30_000, isolation="serializable"):
        pass
    assert conn.asked == [{"isolation": "serializable"}]


@pytest.mark.asyncio
async def test_a_transaction_without_a_deadline_and_a_serving_read_keep_five_seconds() -> None:
    """Flips if the backend's five seconds stops being the default: a read outside a transaction, and a transaction
    that asks none, keep it, and so are refused after six seconds' queueing as before."""
    hub = _QueuedHub()
    proxy = _proxy(hub)
    with pytest.raises(DataLayerUnavailableError, match="QUERY_TIMEOUT"):
        await proxy.fetch("SELECT 1")
    assert hub.calls[-1][1]["timeout_ms"] == 5000
    assert hub.calls[-1][2] == 7.0
    with pytest.raises(DataLayerUnavailableError):
        async with proxy.acquire() as conn:
            async with conn.transaction():
                pass
    begin = next(body for path, body, _ in hub.calls if path.endswith(".l3.tx.begin"))
    assert begin["statement_timeout_ms"] == 5000


def test_a_deadline_under_one_millisecond_is_refused() -> None:
    """Flips if a transaction may ask a deadline no request can meet."""
    with pytest.raises(ValueError, match="at least 1"):
        _ProxyConnection(_proxy(_QueuedHub())).transaction(timeout_ms=0)
