"""A tool that outlives its connection must still be able to hand back its answer.

The production incident, in full: a pentest pod ran ``testssl`` for 92 seconds, the scan succeeded
with exit code 0 and 67902 bytes of output, and the publish was refused --

    03:26:12 running testssl
    03:26:48 NATS re-authenticated (proactive reconnect)
    03:27:44 scanner finished {"exit_code": 0, "duration_seconds": 91.964, "stdout_bytes": 67902}
    03:27:44 NATS error: permissions violation for publish to "_inbox...."

``allow_responses`` -- the right to answer a request without a standing grant on the requester's
inbox -- belongs to the connection that RECEIVED the request. NATS has no in-band re-auth, so
refreshing a credential is a reconnect, and the refresh that keeps the pod authenticated is the same
event that destroys its right to answer. The two obvious levers were both refused: a JWT TTL long
enough to cover a 1200-second scan means 20-minute credentials, and a standing grant on the inbox
tree would let any tool pod forge a reply into any other pod's in-flight call.

So a long call is ACCEPTED and then DELIVERED: acknowledged immediately on the inbox while that is
still guaranteed to work, and answered on a subject the pod holds a standing grant on -- one derived
from its own pod id, so every refresh re-mints it identically.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from uuid import uuid4

import pytest

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.agent.tools.server import CallAccepted, CallResponse, ToolServer
from threetears.nats import IncomingMessage, Subject, Subjects, set_default_namespace

from unit.tools._pod_auth import StubReplayGuard as _PodReplayGuard
from unit.tools._pod_auth import jwks_provider as _pod_jwks_provider
from unit.tools._pod_auth import signed_call_payload as _signed_call_payload

_NS = "3tears"
_POD = "scan-pod-1"


@pytest.fixture(autouse=True)
def _bind_namespace() -> None:
    set_default_namespace(_NS)


# parity-exempt: stands in for exactly the three NatsClient methods a dispatch can answer through (publish / jetstream_publish / publish_reply); a full declaration would drag in the whole connection lifecycle no branch of handle_call touches, and drift in any of those three fails these tests directly  # noqa: E501
class _FakeNats:
    """records the two publish surfaces a dispatch can answer through.

    Deliberately narrow: what these tests observe is WHICH of the two publish paths each branch of a
    dispatch chose, so the double records both and nothing else.
    """

    def __init__(self, *, jetstream_failures: int = 0) -> None:
        self.replies: list[tuple[str, Any]] = []
        self.delivered: list[tuple[str, bytes]] = []
        self.audits: list[tuple[str, bytes]] = []
        self._jetstream_failures = jetstream_failures

    async def publish(self, *, subject: Subject, message: Any, reply_to: Subject | None = None) -> None:
        del reply_to, subject, message

    async def jetstream_publish(self, *, subject: Subject, payload: bytes) -> None:
        if ".audit." in subject.path:
            self.audits.append((subject.path, payload))
            return
        if self._jetstream_failures > 0:
            self._jetstream_failures -= 1
            raise RuntimeError("broker unreachable")
        self.delivered.append((subject.path, payload))

    async def publish_reply(self, *, reply_subject: str, message: Any) -> None:
        self.replies.append((reply_subject, message))


class _StubTool(TearsTool):
    """a tool standing in for a long scan; the duration is irrelevant to the routing decision."""

    def __init__(self, *, content: str = "68KB of results", raise_exc: BaseException | None = None) -> None:
        self._content = content
        self._raise_exc = raise_exc

    async def execute(self, **kwargs: Any) -> ToolResult:
        del kwargs
        if self._raise_exc is not None:
            raise self._raise_exc
        return ToolResult(success=True, content=self._content)

    def mcp_schema(self) -> MCPToolDefinition:
        return MCPToolDefinition(
            name="test.stub",
            version="1.0",
            description="stub",
            input_schema={"type": "object", "properties": {}},
        )

    def mcp_name(self) -> str:
        return "test.stub"

    def mcp_version(self) -> str:
        return "1.0"


def _server(nats: _FakeNats, *, tool: TearsTool | None = None) -> ToolServer:
    server = ToolServer(
        namespace=_NS,
        nats_client=nats,  # type: ignore[arg-type]
        pod_id=_POD,
        namespace_collection=None,
        jwks_provider=_pod_jwks_provider,
        assertion_replay_guard=_PodReplayGuard(),
    )
    server.register(tool if tool is not None else _StubTool())
    return server


def _msg(payload: dict[str, Any]) -> IncomingMessage:
    return IncomingMessage(
        data=json.dumps(payload).encode("utf-8"),
        reply_subject="_INBOX_registry_reg-1.abc",
        subject=f"{_NS}.tools.internal.{_POD}",
    )


def _long_call(*, result_subject: str) -> IncomingMessage:
    payload = _signed_call_payload(pod_id=_POD, conversation_id=uuid4(), user_id=uuid4())
    payload["result_subject"] = result_subject
    return _msg(payload)


def _delivery_subject(call_id: str = "call-1") -> str:
    return Subjects.tools_result(_POD, call_id).path


def _decoded(payload: bytes) -> dict[str, Any]:
    return json.loads(payload.decode("utf-8"))


class TestTheAnswerLeavesOnASubjectThePodOwns:
    @pytest.mark.asyncio
    async def test_the_result_is_delivered_not_replied(self) -> None:
        """THE PRODUCTION BUG. The scan's output leaves on the pod's own standing-grant subject."""
        nats = _FakeNats()
        server = _server(nats)

        await server.handle_call(_long_call(result_subject=_delivery_subject()))

        assert len(nats.delivered) == 1
        subject, payload = nats.delivered[0]
        assert subject == _delivery_subject()
        assert _decoded(payload)["content"] == "68KB of results"

    @pytest.mark.asyncio
    async def test_the_call_is_acknowledged_before_the_tool_runs(self) -> None:
        """the caller must be able to tell "running your long tool" from "this pod is gone".

        without the acknowledgement, the registry's failover to a sibling endpoint could only fire
        after the whole tool budget elapsed -- so a dead pod would cost 20 minutes rather than
        milliseconds.
        """
        nats = _FakeNats()
        server = _server(nats)

        await server.handle_call(_long_call(result_subject=_delivery_subject()))

        assert len(nats.replies) == 1
        _reply_subject, ack = nats.replies[0]
        assert isinstance(ack, CallAccepted)
        assert ack.accepted is True
        assert ack.pod_id == _POD
        assert ack.result_subject == _delivery_subject()

    @pytest.mark.asyncio
    async def test_the_result_never_goes_to_the_reply_inbox(self) -> None:
        """non-vacuous: the answer must not ALSO be replied, which would mask a broken delivery."""
        nats = _FakeNats()
        server = _server(nats)

        await server.handle_call(_long_call(result_subject=_delivery_subject()))

        assert not [msg for _subject, msg in nats.replies if isinstance(msg, CallResponse)]

    @pytest.mark.asyncio
    async def test_a_short_call_keeps_the_fast_reply_path(self) -> None:
        """no ``result_subject`` means the caller judged the call short; nothing changes for it."""
        nats = _FakeNats()
        server = _server(nats)
        payload = _signed_call_payload(pod_id=_POD, conversation_id=uuid4(), user_id=uuid4())

        await server.handle_call(_msg(payload))

        assert not nats.delivered
        assert len(nats.replies) == 1
        _reply_subject, response = nats.replies[0]
        assert isinstance(response, CallResponse)
        assert response.content == "68KB of results"


class TestTheConnectionIsFreedImmediately:
    """Acknowledging is what releases the connection, and releasing it is the point.

    If a durably-delivered call kept owing a reply, a 20-minute scan would hold the connection
    against re-auth for 20 minutes, the bounded drain would give up anyway, and the pod would log a
    reply about to be lost that was in fact perfectly safe.
    """

    @pytest.mark.asyncio
    async def test_the_pod_owes_nothing_once_the_call_is_accepted(self) -> None:
        nats = _FakeNats()
        tool = _SlowTool()
        server = _server(nats, tool=tool)

        dispatch = asyncio.create_task(server.handle_call(_long_call(result_subject=_delivery_subject())))
        await asyncio.sleep(0.05)
        try:
            assert nats.replies, "the call was not acknowledged"
            assert server.sync_replies_in_flight == 0
            assert await server.await_sync_replies(timeout=0.05) is True
        finally:
            tool.release()
            await asyncio.wait_for(dispatch, timeout=1.0)

    @pytest.mark.asyncio
    async def test_a_short_call_still_holds_the_connection(self) -> None:
        """non-vacuous: the drain must still defend the calls that DO answer on the inbox."""
        nats = _FakeNats()
        tool = _SlowTool()
        server = _server(nats, tool=tool)
        payload = _signed_call_payload(pod_id=_POD, conversation_id=uuid4(), user_id=uuid4())

        dispatch = asyncio.create_task(server.handle_call(_msg(payload)))
        await asyncio.sleep(0.05)
        try:
            assert server.sync_replies_in_flight == 1
            assert await server.await_sync_replies(timeout=0.05) is False
        finally:
            tool.release()
            await asyncio.wait_for(dispatch, timeout=1.0)
        assert server.sync_replies_in_flight == 0


class TestAPodWillNotPublishWhereItWasTold:
    """The grant is per-pod; a delivery subject naming someone else is refused before any work.

    The broker denies it too -- this pod's JWT carries a grant on nothing else -- but that denial
    arrives as an opaque publish error AFTER the tool has run, which is the exact shape of the
    incident this design exists to end.
    """

    @pytest.mark.asyncio
    async def test_a_peers_delivery_subject_is_refused(self) -> None:
        nats = _FakeNats()
        server = _server(nats)
        foreign = Subjects.tools_result("some-other-pod", "call-1").path

        await server.handle_call(_long_call(result_subject=foreign))

        assert not nats.delivered
        _reply_subject, ack = nats.replies[0]
        assert isinstance(ack, CallAccepted)
        assert ack.accepted is False
        assert foreign in (ack.error or "")

    @pytest.mark.asyncio
    async def test_the_refusal_happens_before_the_tool_runs(self) -> None:
        """a refused call must cost nothing: the point is to fail before the work, not after it."""
        nats = _FakeNats()
        ran: list[bool] = []

        class _Recording(_StubTool):
            async def execute(self, **kwargs: Any) -> ToolResult:
                ran.append(True)
                return await super().execute(**kwargs)

        server = _server(nats, tool=_Recording())
        await server.handle_call(_long_call(result_subject=f"{_NS}.tools.reply.agent-A.call-1"))

        assert ran == []

    @pytest.mark.asyncio
    async def test_a_wildcard_delivery_subject_is_refused(self) -> None:
        """a wildcard tail would land one publish on every waiter's consumer at once."""
        nats = _FakeNats()
        server = _server(nats)

        await server.handle_call(_long_call(result_subject=f"{_NS}.tools.result.{_POD}.*"))

        assert not nats.delivered


class TestEveryOutcomeTravelsTheSameRoute:
    """A rejection published on the inbox after the call was accepted for delivery would strand the
    caller for its whole timeout on a subject nothing publishes to."""

    @pytest.mark.asyncio
    async def test_an_unknown_tool_is_reported_on_the_delivery_subject(self) -> None:
        nats = _FakeNats()
        server = _server(nats)
        payload = _signed_call_payload(pod_id=_POD, tool_name="test.missing", conversation_id=uuid4(), user_id=uuid4())
        payload["result_subject"] = _delivery_subject()

        await server.handle_call(_msg(payload))

        assert len(nats.delivered) == 1
        body = _decoded(nats.delivered[0][1])
        assert body["success"] is False
        assert "unknown tool" in body["error"]

    @pytest.mark.asyncio
    async def test_a_raising_tool_is_reported_on_the_delivery_subject(self) -> None:
        nats = _FakeNats()
        server = _server(nats, tool=_StubTool(raise_exc=RuntimeError("scanner exploded")))

        await server.handle_call(_long_call(result_subject=_delivery_subject()))

        assert len(nats.delivered) == 1
        body = _decoded(nats.delivered[0][1])
        assert body["success"] is False
        assert "scanner exploded" in body["error"]

    @pytest.mark.asyncio
    async def test_an_unverified_identity_is_reported_on_the_delivery_subject(self) -> None:
        nats = _FakeNats()
        server = _server(nats)
        payload = _signed_call_payload(pod_id=_POD, conversation_id=uuid4(), user_id=uuid4())
        payload["result_subject"] = _delivery_subject()
        payload["context"]["identity_token"] = "not-a-jws"

        await server.handle_call(_msg(payload))

        assert len(nats.delivered) == 1
        assert _decoded(nats.delivered[0][1])["success"] is False


class TestADeliveryFailureIsRetriedRatherThanDiscarded:
    """By this point the tool has already run. Losing the publish to a blip reproduces the incident
    exactly -- work done, answer gone -- and the caller still has a long timeout left to wait in."""

    @pytest.mark.asyncio
    async def test_a_transient_publish_failure_is_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from threetears.agent.tools import server as server_module

        monkeypatch.setattr(server_module, "_RESULT_DELIVERY_RETRY_SECONDS", 0.0)
        nats = _FakeNats(jetstream_failures=2)
        server = _server(nats)

        await server.handle_call(_long_call(result_subject=_delivery_subject()))

        assert len(nats.delivered) == 1
        assert _decoded(nats.delivered[0][1])["content"] == "68KB of results"

    @pytest.mark.asyncio
    async def test_exhausted_retries_do_not_raise_out_of_the_dispatch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """the answer is lost and logged as lost; the dispatch must not also take the pod down."""
        from threetears.agent.tools import server as server_module

        monkeypatch.setattr(server_module, "_RESULT_DELIVERY_RETRY_SECONDS", 0.0)
        nats = _FakeNats(jetstream_failures=99)
        server = _server(nats)

        await server.handle_call(_long_call(result_subject=_delivery_subject()))

        assert not nats.delivered


class _SlowTool(_StubTool):
    """a tool that blocks until the test releases it, standing in for a long scan."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = asyncio.Event()

    async def execute(self, **kwargs: Any) -> ToolResult:
        del kwargs
        await self.gate.wait()
        return ToolResult(success=True, content="68KB of results")

    def release(self) -> None:
        self.gate.set()
