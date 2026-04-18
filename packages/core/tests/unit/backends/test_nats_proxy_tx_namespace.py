"""unit tests for WS-ACL-06 ``namespace=`` threading through tx.begin.

covers the client-side portion of the Phase 5c work: the
``namespace`` kwarg on :meth:`_ProxyConnection.transaction` and the
pool-level :meth:`NatsProxyL3Backend.transaction` convenience wrapper.
every NATS interaction is mocked so these tests pin wire behavior
without requiring a live broker.

the broker-side tests that verify the handler honors the namespace
payload (sets search_path to owner schema, rejects unauthorized
namespaces with ``TX_NAMESPACE_UNAUTHORIZED``) live in
``14-eng-ai-bot/tests/unit/hub/broker/test_tx_namespace_routing.py``.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from threetears.core.backends.nats_proxy import NatsProxyL3Backend
from threetears.core.exceptions import DataLayerUnavailableError


__all__ = [
    "TX_ID",
]


TX_ID = "019d9a00-0000-7000-8000-000000000000"


def _make_reply(data: dict[str, Any]) -> MagicMock:
    """build mock NATS reply message with JSON payload.

    :param data: response payload dict
    :ptype data: dict[str, Any]
    :return: mock reply with ``.data`` attribute
    :rtype: MagicMock
    """
    reply = MagicMock()
    reply.data = json.dumps(data).encode("utf-8")
    return reply


class _ScriptedReplyPlan:
    """drive mock ``nc.request`` with a scripted list of responses.

    each call pops the next plan entry, asserts the subject matches,
    records the payload for later inspection, and returns the
    pre-built reply. lets tests assert on both "right subject called
    in the right order" and "right payload shipped" without hand-
    rolling stateful mocks.
    """

    def __init__(self, entries: list[tuple[str, dict[str, Any]]]) -> None:
        """capture ``(subject, response_dict)`` pairs in order.

        :param entries: list of ``(expected_subject, response_dict)``
        :ptype entries: list[tuple[str, dict[str, Any]]]
        :return: None
        :rtype: None
        """
        self._entries = list(entries)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def __call__(
        self,
        subject: str,
        payload: bytes,
        timeout: float = 0,
    ) -> MagicMock:
        """mock coroutine for ``nc.request``.

        :param subject: subject argument passed by the proxy
        :ptype subject: str
        :param payload: raw bytes payload
        :ptype payload: bytes
        :param timeout: ignored in tests
        :ptype timeout: float
        :return: pre-built reply
        :rtype: MagicMock
        :raises AssertionError: when a subject is called out of order
        """
        del timeout
        assert self._entries, f"no reply scripted for {subject}"
        expected_subject, response = self._entries.pop(0)
        assert subject == expected_subject, (
            f"expected {expected_subject!r}, got {subject!r}"
        )
        self.calls.append((subject, json.loads(payload.decode("utf-8"))))
        return _make_reply(response)


def _make_proxy(mock_nc: MagicMock) -> NatsProxyL3Backend:
    """build NatsProxyL3Backend with mock NATS client.

    :param mock_nc: mock NATS client
    :ptype mock_nc: MagicMock
    :return: configured proxy backend
    :rtype: NatsProxyL3Backend
    """
    return NatsProxyL3Backend(
        nats_client=mock_nc,
        namespace_prefix="test",
        agent_id="agent-abc",
    )


@pytest.mark.asyncio
async def test_transaction_with_namespace_ships_namespace_in_begin_payload() -> None:
    """``transaction(namespace="workspace.abc")`` puts namespace in tx.begin."""
    plan = _ScriptedReplyPlan(
        [
            ("test.l3.tx.begin", {"success": True, "tx_id": TX_ID}),
            ("test.l3.tx.execute", {"success": True, "row_count": 1}),
            ("test.l3.tx.commit", {"success": True}),
        ],
    )
    mock_nc = MagicMock()
    mock_nc.request = plan
    proxy = _make_proxy(mock_nc)

    async with proxy.acquire() as conn:
        async with conn.transaction(namespace="workspace.abc"):
            await conn.execute("INSERT INTO t(x) VALUES ($1)", 1)

    begin_subject, begin_payload = plan.calls[0]
    assert begin_subject == "test.l3.tx.begin"
    # WS-ACL-06: the namespace kwarg flows verbatim to the broker.
    assert begin_payload["namespace"] == "workspace.abc"


@pytest.mark.asyncio
async def test_transaction_without_namespace_uses_default() -> None:
    """``transaction()`` without namespace preserves pre-WS-ACL-06 behavior."""
    plan = _ScriptedReplyPlan(
        [
            ("test.l3.tx.begin", {"success": True, "tx_id": TX_ID}),
            ("test.l3.tx.commit", {"success": True}),
        ],
    )
    mock_nc = MagicMock()
    mock_nc.request = plan
    proxy = _make_proxy(mock_nc)

    async with proxy.acquire() as conn:
        async with conn.transaction():
            pass

    begin_payload = plan.calls[0][1]
    # default namespace = caller's default agent namespace, preserved.
    assert begin_payload["namespace"] == proxy.default_namespace


@pytest.mark.asyncio
async def test_pool_transaction_wrapper_threads_namespace() -> None:
    """pool-level ``pool.transaction(namespace=...)`` threads through begin."""
    plan = _ScriptedReplyPlan(
        [
            ("test.l3.tx.begin", {"success": True, "tx_id": TX_ID}),
            ("test.l3.tx.execute", {"success": True, "row_count": 2}),
            ("test.l3.tx.commit", {"success": True}),
        ],
    )
    mock_nc = MagicMock()
    mock_nc.request = plan
    proxy = _make_proxy(mock_nc)

    async with proxy.transaction(namespace="workspace.xyz") as conn:
        tag = await conn.execute("UPDATE t SET c = $1", "new")
        assert tag == "UPDATE 2"

    begin_payload = plan.calls[0][1]
    assert begin_payload["namespace"] == "workspace.xyz"
    subjects = [c[0] for c in plan.calls]
    assert subjects == [
        "test.l3.tx.begin",
        "test.l3.tx.execute",
        "test.l3.tx.commit",
    ]


@pytest.mark.asyncio
async def test_per_statement_namespace_inside_tx_rejected_on_execute() -> None:
    """statement-level ``namespace=`` inside a tx raises ValueError on execute."""
    plan = _ScriptedReplyPlan(
        [
            ("test.l3.tx.begin", {"success": True, "tx_id": TX_ID}),
            ("test.l3.tx.rollback", {"success": True}),
        ],
    )
    mock_nc = MagicMock()
    mock_nc.request = plan
    proxy = _make_proxy(mock_nc)

    with pytest.raises(ValueError, match="per-statement namespace"):
        async with proxy.acquire() as conn:
            async with conn.transaction(namespace="workspace.abc"):
                await conn.execute(
                    "INSERT INTO t(x) VALUES ($1)",
                    1,
                    namespace="workspace.other",
                )


@pytest.mark.asyncio
async def test_per_statement_namespace_inside_tx_rejected_on_fetchrow() -> None:
    """statement-level ``namespace=`` inside a tx raises ValueError on fetchrow."""
    plan = _ScriptedReplyPlan(
        [
            ("test.l3.tx.begin", {"success": True, "tx_id": TX_ID}),
            ("test.l3.tx.rollback", {"success": True}),
        ],
    )
    mock_nc = MagicMock()
    mock_nc.request = plan
    proxy = _make_proxy(mock_nc)

    with pytest.raises(ValueError, match="per-statement namespace"):
        async with proxy.acquire() as conn:
            async with conn.transaction(namespace="workspace.abc"):
                await conn.fetchrow(
                    "SELECT 1",
                    namespace="workspace.other",
                )


@pytest.mark.asyncio
async def test_per_statement_namespace_inside_tx_rejected_on_fetch() -> None:
    """statement-level ``namespace=`` inside a tx raises ValueError on fetch."""
    plan = _ScriptedReplyPlan(
        [
            ("test.l3.tx.begin", {"success": True, "tx_id": TX_ID}),
            ("test.l3.tx.rollback", {"success": True}),
        ],
    )
    mock_nc = MagicMock()
    mock_nc.request = plan
    proxy = _make_proxy(mock_nc)

    with pytest.raises(ValueError, match="per-statement namespace"):
        async with proxy.acquire() as conn:
            async with conn.transaction(namespace="workspace.abc"):
                await conn.fetch(
                    "SELECT 1",
                    namespace="workspace.other",
                )


@pytest.mark.asyncio
async def test_tx_namespace_unauthorized_surfaces_as_data_layer_error() -> None:
    """broker ``TX_NAMESPACE_UNAUTHORIZED`` raises :class:`DataLayerUnavailableError`."""
    plan = _ScriptedReplyPlan(
        [
            (
                "test.l3.tx.begin",
                {
                    "success": False,
                    "error_code": "TX_NAMESPACE_UNAUTHORIZED",
                    "error_message": "no grant",
                },
            ),
        ],
    )
    mock_nc = MagicMock()
    mock_nc.request = plan
    proxy = _make_proxy(mock_nc)

    with pytest.raises(DataLayerUnavailableError) as excinfo:
        async with proxy.acquire() as conn:
            async with conn.transaction(namespace="workspace.forbidden"):
                pass

    assert "TX_NAMESPACE_UNAUTHORIZED" in str(excinfo.value)
