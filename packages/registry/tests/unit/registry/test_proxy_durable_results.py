"""The registry sits on both sides of the answer that cannot ride a reply inbox.

Fixing only the pod would relocate the failure rather than end it. The chain is agent -> registry ->
pod, and all three connections run the same proactive credential refresh, so ``allow_responses`` --
which belongs to the connection that RECEIVED a request -- is destroyed on every hop by the very act
of staying authenticated. The registry therefore both COLLECTS a pod's answer off a subject the pod
owns, and DELIVERS its own answer onto a subject the calling agent owns.

The two hops key their subjects differently, and the asymmetry is forced. A pod is a named responder,
so ``tools.result.{pod_id}.{call_id}`` names the answerer and only that pod holds the publish grant.
The registry is an anonymous member of a queue group, so an agent cannot know which replica will take
its call and cannot subscribe a responder-named subject in advance; ``tools.reply.{agent_id}.{call_id}``
names the CALLER instead, and containment moves to the registry refusing any subject that does not
name the call's VERIFIED agent id.
"""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any
from uuid import uuid4

import pytest
from threetears.agent.tools.context_envelope import CallContext

from threetears.agent.tools.server import CallAccepted, CallRequest
from threetears.nats import (
    RESULT_ACK_TIMEOUT_SECONDS,
    IncomingMessage,
    RequestError,
    RequestTimeoutError,
    Subject,
    Subjects,
    set_default_namespace,
)
from threetears.registry.catalog import CatalogEntry, ToolCatalog, ToolEndpoint
from threetears.registry.proxy import ProxyCallAccepted, ProxyCallResponse

from ._dispatch_auth import DEFAULT_AGENT_ID, make_authed_request, make_proxy

_NS = "test"
_LONG = 1200.0
_SHORT = 5.0


@pytest.fixture(autouse=True)
def _bind_namespace() -> None:
    set_default_namespace(_NS)


class _Waiter:
    """a result waiter that yields a scripted outcome."""

    def __init__(self, outcome: Any, subject: Subject) -> None:
        self._outcome = outcome
        self.subject = subject
        self.closed = False

    async def wait(self, *, timeout: timedelta) -> bytes:
        del timeout
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        return self._outcome

    async def close(self) -> None:
        self.closed = True


class _Nats:
    """records everything the durable path touches, and scripts the pod's side of it."""

    def __init__(
        self,
        *,
        accept: Any = None,
        delivered: Any = b"",
    ) -> None:
        self._accept = accept
        self._delivered = delivered
        self.request_raw_calls: list[dict[str, Any]] = []
        self.forwarded: list[CallRequest] = []
        self.waiters: list[_Waiter] = []
        self.waiter_subjects: list[str] = []
        self.waiter_budgets: list[float] = []
        self.replies: list[Any] = []
        self.published: list[tuple[str, bytes]] = []
        self.order: list[str] = []

    async def subscribe(self, **kwargs: Any) -> Any:
        del kwargs
        return object()

    async def unsubscribe(self, sub: Any) -> None:
        del sub

    async def request_raw(self, *, subject: Subject, payload: bytes, timeout: timedelta) -> bytes:
        self.order.append("dispatch")
        self.request_raw_calls.append({"subject": subject, "timeout": timeout})
        self.forwarded.append(CallRequest.model_validate_json(payload))
        if isinstance(self._accept, BaseException):
            raise self._accept
        if self._accept is not None:
            return self._accept
        return CallAccepted(accepted=True, pod_id="pod-1", result_subject="x").model_dump_json().encode("utf-8")

    async def jetstream_result_waiter(self, *, subject: Subject, stream: str, wait_budget: timedelta) -> _Waiter:
        self.order.append("open-waiter")
        self.waiter_subjects.append(subject.path)
        self.waiter_budgets.append(wait_budget.total_seconds())
        self.stream = stream
        waiter = _Waiter(self._delivered, subject)
        self.waiters.append(waiter)
        return waiter

    async def jetstream_publish(self, *, subject: Subject, payload: bytes) -> None:
        self.published.append((subject.path, payload))

    async def publish_reply(self, *, reply_subject: str, message: Any) -> None:
        del reply_subject
        self.replies.append(message)


async def _registered_catalog(timeout_seconds: float, *, pods: tuple[str, ...] = ("pod-1",)) -> ToolCatalog:
    """a catalog holding one tool whose DECLARED timeout decides which delivery path it takes."""
    catalog = ToolCatalog()
    await catalog.register(
        CatalogEntry(
            tool_name="threetears.calculator",
            tool_version="1.0.0",
            full_name="threetears.calculator@1.0.0",
            description="test tool",
            input_schema={"type": "object", "properties": {}},
            timeout_seconds=timeout_seconds,
            endpoints=[ToolEndpoint(pod_id=pod, status="available") for pod in pods],
        )
    )
    return catalog


def _msg(request: Any) -> IncomingMessage:
    return IncomingMessage(
        data=request.model_dump_json().encode("utf-8"),
        reply_subject="_INBOX_agent_pod_a.1",
        subject=f"{_NS}.tools.call",
    )


def _pod_answer(content: str = "68KB of results", success: bool = True) -> bytes:
    return ProxyCallResponse(success=success, content=content, context=CallContext()).model_dump_json().encode("utf-8")


async def _dispatch(proxy: Any, nats: Any, request: Any) -> None:
    """drive one call end to end, draining the background dispatch task before asserting."""
    await proxy.start(nats)
    await proxy.handle_call(_msg(request))
    await proxy.stop()


class TestCollectingAPodsAnswer:
    """The registry -> pod hop: a long tool's answer comes off the pod's own durable subject."""

    @pytest.mark.asyncio
    async def test_a_long_call_waits_on_the_pods_delivery_subject(self) -> None:
        catalog = await _registered_catalog(_LONG)
        nats = _Nats(delivered=_pod_answer())
        await _dispatch(make_proxy(catalog, namespace=_NS), nats, make_authed_request())

        assert len(nats.waiter_subjects) == 1
        assert nats.waiter_subjects[0].startswith(f"{_NS}.tools.result.pod-1.")
        assert nats.replies[0].content == "68KB of results"

    @pytest.mark.asyncio
    async def test_the_pod_is_told_where_to_deliver(self) -> None:
        """the forwarded request carries the same subject the waiter is bound to."""
        catalog = await _registered_catalog(_LONG)
        nats = _Nats(delivered=_pod_answer())
        await _dispatch(make_proxy(catalog, namespace=_NS), nats, make_authed_request())

        assert nats.forwarded[0].result_subject == nats.waiter_subjects[0]

    @pytest.mark.asyncio
    async def test_the_waiter_opens_before_the_call_is_dispatched(self) -> None:
        """ordering is the contract: a pod must never answer with nothing listening."""
        catalog = await _registered_catalog(_LONG)
        nats = _Nats(delivered=_pod_answer())
        await _dispatch(make_proxy(catalog, namespace=_NS), nats, make_authed_request())

        assert nats.order == ["open-waiter", "dispatch"]

    @pytest.mark.asyncio
    async def test_the_delivery_subject_is_per_call_not_per_turn(self) -> None:
        """two calls in ONE turn must not share a subject.

        ``correlation_id`` is per TURN, and a turn routinely makes several tool calls; keying on it
        would hand the second waiter the first call's result. the constant correlation id here is
        exactly the shape that would collide.
        """
        catalog = await _registered_catalog(_LONG)
        nats = _Nats(delivered=_pod_answer())
        proxy = make_proxy(catalog, namespace=_NS)
        await proxy.start(nats)
        await proxy.handle_call(_msg(make_authed_request()))
        await proxy.handle_call(_msg(make_authed_request()))
        await proxy.stop()

        assert len(set(nats.waiter_subjects)) == 2

    @pytest.mark.asyncio
    async def test_the_acknowledgement_window_is_short(self) -> None:
        """the accept is answered before any work, so a dead pod is detected in milliseconds.

        if the accept shared the tool's budget, failover to a sibling endpoint could not fire until
        the whole 20 minutes had elapsed -- which is the behaviour the failover loop exists to avoid.
        """
        catalog = await _registered_catalog(_LONG)
        nats = _Nats(delivered=_pod_answer())
        await _dispatch(make_proxy(catalog, namespace=_NS), nats, make_authed_request())

        assert nats.request_raw_calls[0]["timeout"].total_seconds() == RESULT_ACK_TIMEOUT_SECONDS
        assert nats.waiter_budgets == [_LONG]

    @pytest.mark.asyncio
    async def test_a_short_call_keeps_the_synchronous_path(self) -> None:
        """non-vacuous: nothing changes for calls the connection can genuinely outlive."""
        catalog = await _registered_catalog(_SHORT)
        nats = _Nats(accept=_pod_answer(content="quick"))
        await _dispatch(make_proxy(catalog, namespace=_NS), nats, make_authed_request())

        assert nats.waiter_subjects == []
        assert nats.forwarded[0].result_subject is None
        assert nats.request_raw_calls[0]["timeout"].total_seconds() == _SHORT
        assert nats.replies[0].content == "quick"

    @pytest.mark.asyncio
    async def test_the_waiter_is_always_closed(self) -> None:
        """an abandoned consumer per long call would accumulate for the life of the process."""
        catalog = await _registered_catalog(_LONG)
        nats = _Nats(delivered=_pod_answer())
        await _dispatch(make_proxy(catalog, namespace=_NS), nats, make_authed_request())

        assert nats.waiters[0].closed is True

    @pytest.mark.asyncio
    async def test_the_waiter_is_closed_even_when_no_result_arrives(self) -> None:
        catalog = await _registered_catalog(_LONG)
        nats = _Nats(delivered=RequestTimeoutError("no result"))
        await _dispatch(make_proxy(catalog, namespace=_NS), nats, make_authed_request())

        assert nats.waiters[0].closed is True
        assert nats.replies[0].error_code == "TOOL_TIMEOUT"


class TestWhenTheOtherEndDoesNotPlayAlong:
    @pytest.mark.asyncio
    async def test_an_unacknowledged_call_reads_as_a_dead_endpoint(self) -> None:
        """TOOL_UNAVAILABLE, so the failover loop can try a sibling endpoint.

        unlike the synchronous path -- where a timeout might mean "the pod is running your tool", so
        retrying risks a double execution -- the pod had only to acknowledge before starting work.
        silence there means it is not there.
        """
        catalog = await _registered_catalog(_LONG)
        nats = _Nats(accept=RequestError("no responders available"))
        await _dispatch(make_proxy(catalog, namespace=_NS), nats, make_authed_request())

        assert nats.replies[0].error_code == "TOOL_UNAVAILABLE"

    @pytest.mark.asyncio
    async def test_an_unacknowledged_call_fails_over_to_a_sibling_pod(self) -> None:
        """the acknowledgement exists to keep this working; prove it still does."""
        catalog = await _registered_catalog(_LONG, pods=("pod-dead", "pod-alive"))
        nats = _Nats(accept=RequestError("no responders available"))
        await _dispatch(make_proxy(catalog, namespace=_NS), nats, make_authed_request())

        attempted = {call["subject"].path for call in nats.request_raw_calls}
        assert len(attempted) == 2, f"failover did not try a second endpoint: {attempted}"

    @pytest.mark.asyncio
    async def test_a_pod_that_refuses_the_subject_is_not_retried_elsewhere(self) -> None:
        """a refusal is a wiring fault, not a dead endpoint.

        the pod and this proxy disagree about the pod's own identity. failing over would paper over
        that until it applied to every pod at once, so it surfaces as its own error code instead.
        """
        catalog = await _registered_catalog(_LONG, pods=("pod-1", "pod-2"))
        refusal = (
            CallAccepted(accepted=False, pod_id="pod-1", error="not mine to publish").model_dump_json().encode("utf-8")
        )
        nats = _Nats(accept=refusal)
        await _dispatch(make_proxy(catalog, namespace=_NS), nats, make_authed_request())

        assert nats.replies[0].error_code == "TOOL_RESULT_SUBJECT_REFUSED"
        assert len(nats.request_raw_calls) == 1

    @pytest.mark.asyncio
    async def test_a_pod_that_answers_outright_is_taken_at_its_word(self) -> None:
        """a malformed-request rejection comes back as the ANSWER, not as an acknowledgement.

        the pod could not parse far enough to learn where to deliver, so it replies on the inbox.
        treating that as an accept would leave the caller waiting for a result already in hand.
        """
        catalog = await _registered_catalog(_LONG)
        nats = _Nats(accept=_pod_answer(content="malformed call request", success=False))
        await _dispatch(make_proxy(catalog, namespace=_NS), nats, make_authed_request())

        assert nats.replies[0].success is False
        assert nats.replies[0].content == "malformed call request"

    @pytest.mark.asyncio
    async def test_a_result_that_never_arrives_is_not_retried_elsewhere(self) -> None:
        """the pod ACCEPTED, so it may be running the tool; a retry risks a double execution."""
        catalog = await _registered_catalog(_LONG, pods=("pod-1", "pod-2"))
        nats = _Nats(delivered=RequestTimeoutError("nothing delivered"))
        await _dispatch(make_proxy(catalog, namespace=_NS), nats, make_authed_request())

        assert nats.replies[0].error_code == "TOOL_TIMEOUT"
        assert len(nats.request_raw_calls) == 1


class TestDeliveringToTheAgent:
    """The agent -> registry hop, where the responder is anonymous and the subject names the caller."""

    def _long_request(self, *, result_subject: str) -> Any:
        return make_authed_request().model_copy(update={"result_subject": result_subject})

    @pytest.mark.asyncio
    async def test_the_answer_is_delivered_on_the_callers_subject(self) -> None:
        catalog = await _registered_catalog(_LONG)
        nats = _Nats(delivered=_pod_answer())
        subject = Subjects.tools_reply(DEFAULT_AGENT_ID, "call-1").path
        await _dispatch(make_proxy(catalog, namespace=_NS), nats, self._long_request(result_subject=subject))

        assert len(nats.published) == 1
        published_subject, payload = nats.published[0]
        assert published_subject == subject
        assert json.loads(payload)["content"] == "68KB of results"

    @pytest.mark.asyncio
    async def test_the_call_is_acknowledged_on_the_inbox(self) -> None:
        """the agent must be able to tell "the registry has your call" from "nobody answered"."""
        catalog = await _registered_catalog(_LONG)
        nats = _Nats(delivered=_pod_answer())
        subject = Subjects.tools_reply(DEFAULT_AGENT_ID, "call-1").path
        await _dispatch(make_proxy(catalog, namespace=_NS), nats, self._long_request(result_subject=subject))

        assert isinstance(nats.replies[0], ProxyCallAccepted)
        assert nats.replies[0].accepted is True
        assert nats.replies[0].result_subject == subject

    @pytest.mark.asyncio
    async def test_the_answer_never_also_goes_to_the_inbox(self) -> None:
        """a second copy on the inbox would mask a broken delivery in every test below."""
        catalog = await _registered_catalog(_LONG)
        nats = _Nats(delivered=_pod_answer())
        subject = Subjects.tools_reply(DEFAULT_AGENT_ID, "call-1").path
        await _dispatch(make_proxy(catalog, namespace=_NS), nats, self._long_request(result_subject=subject))

        assert not [reply for reply in nats.replies if isinstance(reply, ProxyCallResponse)]

    @pytest.mark.asyncio
    async def test_a_subject_naming_another_agent_is_refused(self) -> None:
        """THE CONTAINMENT. The registry's wildcard publish grant must not be a redirect primitive.

        One registry connection fronts every agent, so the grant on the reply family has to be
        ``tools.reply.*.*`` -- there is no per-connection list of agent ids to mint literals from.
        What stops that being usable to inject a result into a PEER's in-flight call is this check,
        against the identity the token proved rather than the one the envelope claimed.
        """
        catalog = await _registered_catalog(_LONG)
        nats = _Nats(delivered=_pod_answer())
        peer = Subjects.tools_reply(uuid4(), "call-1").path
        await _dispatch(make_proxy(catalog, namespace=_NS), nats, self._long_request(result_subject=peer))

        assert nats.published == []
        assert isinstance(nats.replies[0], ProxyCallAccepted)
        assert nats.replies[0].accepted is False
        assert peer in (nats.replies[0].error or "")

    @pytest.mark.asyncio
    async def test_a_refused_subject_costs_no_tool_call(self) -> None:
        """the refusal happens before routing: a rejected call must not run anything."""
        catalog = await _registered_catalog(_LONG)
        nats = _Nats(delivered=_pod_answer())
        peer = Subjects.tools_reply(uuid4(), "call-1").path
        await _dispatch(make_proxy(catalog, namespace=_NS), nats, self._long_request(result_subject=peer))

        assert nats.request_raw_calls == []

    @pytest.mark.asyncio
    async def test_a_pod_result_subject_cannot_be_smuggled_in(self) -> None:
        """an agent must not be able to make the registry publish into the pod-result family."""
        catalog = await _registered_catalog(_LONG)
        nats = _Nats(delivered=_pod_answer())
        pod_family = Subjects.tools_result("pod-1", "call-1").path
        await _dispatch(make_proxy(catalog, namespace=_NS), nats, self._long_request(result_subject=pod_family))

        assert nats.published == []

    @pytest.mark.asyncio
    async def test_a_wildcard_subject_is_refused(self) -> None:
        """a wildcard would land one answer on every waiting agent's consumer at once."""
        catalog = await _registered_catalog(_LONG)
        nats = _Nats(delivered=_pod_answer())
        wildcard = f"{_NS}.tools.reply.{DEFAULT_AGENT_ID}.*"
        await _dispatch(make_proxy(catalog, namespace=_NS), nats, self._long_request(result_subject=wildcard))

        assert nats.published == []
        assert nats.replies[0].accepted is False

    @pytest.mark.asyncio
    async def test_an_error_outcome_travels_the_delivery_route_too(self) -> None:
        """every branch must land where the caller is listening, not just the happy one.

        an error published on the inbox after the call was acknowledged for delivery would leave the
        agent waiting out its whole timeout on a subject nothing publishes to.
        """
        catalog = await _registered_catalog(_LONG)
        nats = _Nats(accept=RequestError("no responders available"))
        subject = Subjects.tools_reply(DEFAULT_AGENT_ID, "call-1").path
        await _dispatch(make_proxy(catalog, namespace=_NS), nats, self._long_request(result_subject=subject))

        assert len(nats.published) == 1
        assert json.loads(nats.published[0][1])["error_code"] == "TOOL_UNAVAILABLE"

    @pytest.mark.asyncio
    async def test_an_unknown_tool_travels_the_delivery_route_too(self) -> None:
        catalog = ToolCatalog()
        nats = _Nats(delivered=_pod_answer())
        subject = Subjects.tools_reply(DEFAULT_AGENT_ID, "call-1").path
        await _dispatch(make_proxy(catalog, namespace=_NS), nats, self._long_request(result_subject=subject))

        assert len(nats.published) == 1
        assert json.loads(nats.published[0][1])["error_code"] == "TOOL_UNAVAILABLE"
