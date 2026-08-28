"""tests for the agent-side memory-namespace ensure client.

The client's job is small and every part of it is a refusal: present the
CURRENT identity token, send one typed request, and turn anything that is not
a matching success into :class:`MemoryNamespaceUnavailableError`. So the NATS
stand-in here records what was sent rather than merely accepting it -- "was
refused" and "was never sent" are different claims, and a client that sends
nothing would otherwise satisfy every refusal case.

Each refusal is paired with the admitted twin built from the same helper, so a
client that raised unconditionally fails the suite.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any
from uuid import UUID, uuid4

import pytest

from threetears.nats import RequestError, Subjects

from threetears.agent.memory.namespace_client import (
    HubMemoryNamespaceProvisioner,
    MemoryNamespaceEnsureReply,
    MemoryNamespaceEnsureRequest,
    MemoryNamespaceRef,
    MemoryNamespaceUnavailableError,
)


class _NatsClientFake:
    """records every request and replays a scripted reply.

    :ivar sent: one :class:`MemoryNamespaceEnsureRequest` per request, in order
    :ivar subjects: subject path per request, in order
    :ivar reply: reply handed back, when no ``failure`` is configured
    :ivar failure: exception raised instead of replying
    """

    def __init__(
        self,
        *,
        reply: MemoryNamespaceEnsureReply | None = None,
        failure: Exception | None = None,
    ) -> None:
        """store the scripted outcome.

        :param reply: reply to return
        :ptype reply: MemoryNamespaceEnsureReply | None
        :param failure: exception to raise instead of replying
        :ptype failure: Exception | None
        :return: nothing
        :rtype: None
        """
        self.sent: list[MemoryNamespaceEnsureRequest] = []
        self.subjects: list[str] = []
        self.reply = reply
        self.failure = failure

    async def request(
        self,
        *,
        subject: Any,
        message: Any,
        response_type: Any,
        timeout: timedelta,
    ) -> Any:
        """record the outbound request and replay the scripted outcome.

        :param subject: target subject
        :ptype subject: Any
        :param message: typed request body
        :ptype message: Any
        :param response_type: model the reply decodes into
        :ptype response_type: Any
        :param timeout: request timeout
        :ptype timeout: timedelta
        :return: scripted reply
        :rtype: Any
        :raises Exception: the configured failure, when one is set
        """
        _ = response_type, timeout
        self.sent.append(message)
        self.subjects.append(subject.path)
        if self.failure is not None:
            raise self.failure
        return self.reply


def _reply(*, agent_id: UUID, customer_id: UUID, namespace_id: UUID | None = None) -> MemoryNamespaceEnsureReply:
    """build the success reply a hub would send for one pair.

    :param agent_id: owning agent UUID
    :ptype agent_id: UUID
    :param customer_id: owning customer UUID
    :ptype customer_id: UUID
    :param namespace_id: namespace UUID; minted when omitted
    :ptype namespace_id: UUID | None
    :return: success reply
    :rtype: MemoryNamespaceEnsureReply
    """
    return MemoryNamespaceEnsureReply(
        success=True,
        correlation_id=uuid4(),
        namespace_id=namespace_id if namespace_id is not None else uuid4(),
        name="memories.aaaaaaaa.bbbbbbbb",
        namespace_type="memory",
        owner_agent_id=agent_id,
        customer_id=customer_id,
        owner_namespace=f"agents.{agent_id}",
    )


def _provisioner(nc: _NatsClientFake, token: str = "tok-1") -> HubMemoryNamespaceProvisioner:
    """build a provisioner over the fake transport with a fixed token.

    :param nc: fake NATS client
    :ptype nc: _NatsClientFake
    :param token: token the provider returns
    :ptype token: str
    :return: wired provisioner
    :rtype: HubMemoryNamespaceProvisioner
    """
    return HubMemoryNamespaceProvisioner(
        nats_client=nc,  # type: ignore[arg-type]
        identity_token_provider=lambda: token,
    )


class TestEnsureSendsAVerifiableRequest:
    async def test_success_returns_the_resolved_reference(self) -> None:
        """the admitted path: a matching success becomes a namespace reference."""
        agent_id, customer_id, namespace_id = uuid4(), uuid4(), uuid4()
        nc = _NatsClientFake(reply=_reply(agent_id=agent_id, customer_id=customer_id, namespace_id=namespace_id))

        ref = await _provisioner(nc).ensure(agent_id=agent_id, customer_id=customer_id)

        assert isinstance(ref, MemoryNamespaceRef)
        assert ref.id == namespace_id
        assert ref.owner_agent_id == agent_id
        assert ref.customer_id == customer_id
        assert ref.namespace_type == "memory"
        assert ref.owner_namespace == f"agents.{agent_id}"

    async def test_request_carries_the_token_and_the_pair(self) -> None:
        """the wire request names the token and the pair being asked about."""
        agent_id, customer_id = uuid4(), uuid4()
        nc = _NatsClientFake(reply=_reply(agent_id=agent_id, customer_id=customer_id))

        await _provisioner(nc, token="tok-abc").ensure(agent_id=agent_id, customer_id=customer_id)

        assert len(nc.sent) == 1
        sent = nc.sent[0]
        assert sent.identity_token == "tok-abc"
        assert sent.agent_id == agent_id
        assert sent.customer_id == customer_id
        assert nc.subjects == [Subjects.hub_memory_namespace_ensure().path]
        assert nc.subjects[0].endswith(".hub.memory.namespace.ensure")

    async def test_token_is_read_live_on_every_ensure(self) -> None:
        """a re-handshake replaces the token, and the hub accepts only the current one.

        a provisioner that captured the bootstrap token would be refused on
        every ensure after the first identity refresh, so the provider is
        called per request rather than once at construction.
        """
        agent_id, customer_id = uuid4(), uuid4()
        nc = _NatsClientFake(reply=_reply(agent_id=agent_id, customer_id=customer_id))
        tokens = iter(["tok-first", "tok-second"])
        provisioner = HubMemoryNamespaceProvisioner(
            nats_client=nc,  # type: ignore[arg-type]
            identity_token_provider=lambda: next(tokens),
        )

        await provisioner.ensure(agent_id=agent_id, customer_id=customer_id)
        await provisioner.ensure(agent_id=agent_id, customer_id=customer_id)

        assert [r.identity_token for r in nc.sent] == ["tok-first", "tok-second"]


class TestEnsureFailsClosed:
    async def test_absent_token_sends_nothing(self) -> None:
        """with no handshake completed there is no credential to present."""
        nc = _NatsClientFake(reply=_reply(agent_id=uuid4(), customer_id=uuid4()))
        provisioner = HubMemoryNamespaceProvisioner(
            nats_client=nc,  # type: ignore[arg-type]
            identity_token_provider=lambda: None,
        )

        with pytest.raises(MemoryNamespaceUnavailableError, match="no identity token"):
            await provisioner.ensure(agent_id=uuid4(), customer_id=uuid4())
        assert nc.sent == []

    async def test_raising_token_provider_sends_nothing(self) -> None:
        """the L3 backend's provider RAISES before a handshake; same outcome."""

        def _raise() -> str:
            """stand in for a provider that refuses before handshake.

            :return: never returns
            :rtype: str
            :raises RuntimeError: always
            """
            raise RuntimeError("handshake has not completed")

        nc = _NatsClientFake(reply=_reply(agent_id=uuid4(), customer_id=uuid4()))
        provisioner = HubMemoryNamespaceProvisioner(
            nats_client=nc,  # type: ignore[arg-type]
            identity_token_provider=_raise,
        )

        with pytest.raises(MemoryNamespaceUnavailableError, match="no identity token"):
            await provisioner.ensure(agent_id=uuid4(), customer_id=uuid4())
        assert nc.sent == []

    async def test_transport_failure_is_reported_as_unavailable(self) -> None:
        """a timeout or a missing responder is not a namespace."""
        nc = _NatsClientFake(failure=RequestError("no responders available"))

        with pytest.raises(MemoryNamespaceUnavailableError, match="request failed"):
            await _provisioner(nc).ensure(agent_id=uuid4(), customer_id=uuid4())

    async def test_hub_refusal_is_reported_with_its_code(self) -> None:
        """a refusal names the hub's own code so an operator can match the log."""
        nc = _NatsClientFake(
            reply=MemoryNamespaceEnsureReply(
                success=False,
                correlation_id=uuid4(),
                error_code="AGENT_MISMATCH",
                error_message="request names an agent that is not the verified caller",
            )
        )

        with pytest.raises(MemoryNamespaceUnavailableError, match="AGENT_MISMATCH"):
            await _provisioner(nc).ensure(agent_id=uuid4(), customer_id=uuid4())

    async def test_success_with_no_namespace_is_refused(self) -> None:
        """a success carrying no namespace id is a malformed answer, not a row."""
        agent_id, customer_id = uuid4(), uuid4()
        nc = _NatsClientFake(
            reply=MemoryNamespaceEnsureReply(
                success=True,
                correlation_id=uuid4(),
                namespace_id=None,
                namespace_type="memory",
                owner_agent_id=agent_id,
                customer_id=customer_id,
            )
        )

        with pytest.raises(MemoryNamespaceUnavailableError, match="carrying no namespace"):
            await _provisioner(nc).ensure(agent_id=agent_id, customer_id=customer_id)

    async def test_reply_about_another_agent_is_refused(self) -> None:
        """an answer about a peer must not become the subject of a decision.

        the caller evaluates ``memory.read`` / ``memory.write`` against whatever
        comes back, so silently accepting a namespace owned by somebody else
        would authorize against the wrong row.
        """
        agent_id, peer_id, customer_id = uuid4(), uuid4(), uuid4()
        nc = _NatsClientFake(reply=_reply(agent_id=peer_id, customer_id=customer_id))

        with pytest.raises(MemoryNamespaceUnavailableError, match="different agent or customer"):
            await _provisioner(nc).ensure(agent_id=agent_id, customer_id=customer_id)

    async def test_reply_about_another_customer_is_refused(self) -> None:
        """same rule on the tenancy axis."""
        agent_id, customer_id, other_customer = uuid4(), uuid4(), uuid4()
        nc = _NatsClientFake(reply=_reply(agent_id=agent_id, customer_id=other_customer))

        with pytest.raises(MemoryNamespaceUnavailableError, match="different agent or customer"):
            await _provisioner(nc).ensure(agent_id=agent_id, customer_id=customer_id)
