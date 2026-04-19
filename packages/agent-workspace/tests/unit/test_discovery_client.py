"""tests for :class:`NamespaceDiscoveryClient`.

covers the generalized ``{ns}.namespace.discover`` client shipped by
namespace-task-01 Phase 1. the test double replaces the NATS client;
assertions focus on the subject string the client publishes to, the
request payload it serializes, and its translation of broker responses
(success envelopes, error envelopes, malformed bytes, transport
failures) into :class:`DiscoveryClientError` versus typed summaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

import pytest

from threetears.agent.workspace.discovery_client import (
    DiscoveryClientError,
    NamespaceDiscoveryClient,
    NamespaceDiscoveryRequest,
    NamespaceDiscoveryResponse,
    NamespaceDiscoverySummary,
)


@dataclass
class _Reply:
    """stand-in for a NATS reply object exposing ``.data``."""

    data: bytes


class _FakeNatsClient:
    """fake NATS client recording publish subjects + payloads.

    :param reply_bytes: bytes to return as ``reply.data`` on request
    :ptype reply_bytes: bytes
    :param raise_exc: optional exception to raise on request instead
    :ptype raise_exc: Exception | None
    """

    def __init__(
        self,
        reply_bytes: bytes = b"",
        raise_exc: Exception | None = None,
    ) -> None:
        """initialize the fake client.

        :param reply_bytes: bytes to return as ``reply.data``
        :ptype reply_bytes: bytes
        :param raise_exc: optional exception to raise on request
        :ptype raise_exc: Exception | None
        """
        self._reply_bytes = reply_bytes
        self._raise_exc = raise_exc
        self.calls: list[tuple[str, bytes, float]] = []

    async def request(
        self, subject: str, payload: bytes, timeout: float,
    ) -> _Reply:
        """record the call and return a canned reply.

        :param subject: NATS subject the client publishes to
        :ptype subject: str
        :param payload: request body bytes
        :ptype payload: bytes
        :param timeout: per-call timeout seconds
        :ptype timeout: float
        :return: canned reply
        :rtype: _Reply
        :raises Exception: when the fake is configured to raise
        """
        self.calls.append((subject, payload, timeout))
        if self._raise_exc is not None:
            raise self._raise_exc
        return _Reply(data=self._reply_bytes)


def _success_bytes(summaries: list[NamespaceDiscoverySummary]) -> bytes:
    """build a success envelope as bytes.

    :param summaries: list of summaries to carry in the response
    :ptype summaries: list[NamespaceDiscoverySummary]
    :return: serialized response bytes
    :rtype: bytes
    """
    return NamespaceDiscoveryResponse(items=summaries).model_dump_json().encode()


@pytest.mark.asyncio
async def test_discover_publishes_on_generalized_subject() -> None:
    """client publishes to ``{ns}.namespace.discover`` exactly."""
    reply = _success_bytes([])
    fake = _FakeNatsClient(reply_bytes=reply)
    client = NamespaceDiscoveryClient(nats_client=fake, namespace="test-ns")

    await client.discover(
        correlation_id=uuid4(),
        agent_id=uuid4(),
        customer_id=uuid4(),
        user_id=None,
    )

    assert len(fake.calls) == 1
    subject, _payload, _timeout = fake.calls[0]
    assert subject == "test-ns.namespace.discover"


@pytest.mark.asyncio
async def test_discover_payload_carries_namespace_type_filter() -> None:
    """client serializes the namespace_type filter verbatim."""
    reply = _success_bytes([])
    fake = _FakeNatsClient(reply_bytes=reply)
    client = NamespaceDiscoveryClient(nats_client=fake, namespace="ns")

    await client.discover(
        correlation_id=uuid4(),
        agent_id=uuid4(),
        customer_id=uuid4(),
        user_id=None,
        namespace_type="workspace",
    )

    _subject, payload, _timeout = fake.calls[0]
    parsed = NamespaceDiscoveryRequest.model_validate_json(payload)
    assert parsed.namespace_type == "workspace"


@pytest.mark.asyncio
async def test_discover_unfiltered_payload_has_none_filter() -> None:
    """omitting the filter leaves ``namespace_type`` as None on the wire."""
    reply = _success_bytes([])
    fake = _FakeNatsClient(reply_bytes=reply)
    client = NamespaceDiscoveryClient(nats_client=fake, namespace="ns")

    await client.discover(
        correlation_id=uuid4(),
        agent_id=uuid4(),
        customer_id=uuid4(),
        user_id=None,
    )

    _subject, payload, _timeout = fake.calls[0]
    parsed = NamespaceDiscoveryRequest.model_validate_json(payload)
    assert parsed.namespace_type is None


@pytest.mark.asyncio
async def test_discover_parses_summaries_from_success_envelope() -> None:
    """each summary in the response surfaces typed on the client."""
    agent_id = uuid4()
    customer_id = uuid4()
    expected = [
        NamespaceDiscoverySummary(
            id=uuid4(),
            name="workspace.alpha",
            namespace_type="workspace",
            owner_agent_id=agent_id,
            customer_id=customer_id,
        ),
        NamespaceDiscoverySummary(
            id=uuid4(),
            name="memory.beta",
            namespace_type="memory",
            owner_agent_id=agent_id,
            customer_id=customer_id,
        ),
    ]
    fake = _FakeNatsClient(reply_bytes=_success_bytes(expected))
    client = NamespaceDiscoveryClient(nats_client=fake, namespace="ns")

    items = await client.discover(
        correlation_id=uuid4(),
        agent_id=agent_id,
        customer_id=customer_id,
        user_id=None,
    )

    assert [item.name for item in items] == ["workspace.alpha", "memory.beta"]
    assert [item.namespace_type for item in items] == ["workspace", "memory"]


@pytest.mark.asyncio
async def test_discover_transport_failure_raises_client_error() -> None:
    """NATS exception surfaces as :class:`DiscoveryClientError`."""
    fake = _FakeNatsClient(raise_exc=RuntimeError("nats unreachable"))
    client = NamespaceDiscoveryClient(nats_client=fake, namespace="ns")

    with pytest.raises(DiscoveryClientError) as excinfo:
        await client.discover(
            correlation_id=uuid4(),
            agent_id=uuid4(),
            customer_id=uuid4(),
            user_id=None,
        )
    assert "nats unreachable" in str(excinfo.value)


@pytest.mark.asyncio
async def test_discover_malformed_reply_raises_client_error() -> None:
    """non-JSON reply bytes surface as :class:`DiscoveryClientError`."""
    fake = _FakeNatsClient(reply_bytes=b"not-json")
    client = NamespaceDiscoveryClient(nats_client=fake, namespace="ns")

    with pytest.raises(DiscoveryClientError):
        await client.discover(
            correlation_id=uuid4(),
            agent_id=uuid4(),
            customer_id=uuid4(),
            user_id=None,
        )


@pytest.mark.asyncio
async def test_discover_requires_nats_client() -> None:
    """constructing with ``None`` nats client fails the call up front."""
    client = NamespaceDiscoveryClient(nats_client=None, namespace="ns")

    with pytest.raises(DiscoveryClientError) as excinfo:
        await client.discover(
            correlation_id=uuid4(),
            agent_id=uuid4(),
            customer_id=uuid4(),
            user_id=None,
        )
    assert "no" in str(excinfo.value).lower() or "nats" in str(excinfo.value).lower()


@pytest.mark.asyncio
async def test_discover_carries_correlation_agent_user_customer_on_wire() -> None:
    """every identity field from the caller round-trips into the payload."""
    fake = _FakeNatsClient(reply_bytes=_success_bytes([]))
    client = NamespaceDiscoveryClient(nats_client=fake, namespace="ns")

    correlation_id = uuid4()
    agent_id = uuid4()
    customer_id = uuid4()
    user_id = uuid4()

    await client.discover(
        correlation_id=correlation_id,
        agent_id=agent_id,
        customer_id=customer_id,
        user_id=user_id,
        namespace_type="memory",
    )

    _subject, payload, _timeout = fake.calls[0]
    parsed = NamespaceDiscoveryRequest.model_validate_json(payload)
    assert parsed.correlation_id == correlation_id
    assert parsed.agent_id == agent_id
    assert parsed.customer_id == customer_id
    assert parsed.user_id == user_id
    assert parsed.namespace_type == "memory"
