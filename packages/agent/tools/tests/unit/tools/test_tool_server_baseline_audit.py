"""unit tests for the baseline ``tool.call`` audit emission on dispatch.

audit-task-01 Phase 3 (AUD-03): :meth:`ToolServer.handle_call` emits
one :class:`AuditEvent` per dispatch with ``event_type='tool.call'``,
outcome derived from the response / exception path, and identity axes
pulled from the inbound :class:`CallContext`. the baseline event is
published to ``{namespace}.audit.tool.call`` via :func:`publish_audit`
and is additive to any per-tool domain event the tool itself emits.

the tests in this module wire a pre-connected fake NATS client so the
audit publish path fires; existing :class:`ToolServer` tests construct
the server with ``nats_url`` alone (so ``self._nc`` stays ``None``
until ``serve`` runs) which naturally suppresses baseline emission.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, cast
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.agent.tools.context import ToolContextManager
from threetears.agent.tools.server import ToolServer
from threetears.nats import IncomingMessage, Subject, set_default_namespace

from unit.tools._pod_auth import StubReplayGuard as _PodReplayGuard
from unit.tools._pod_auth import jwks_provider as _pod_jwks_provider
from unit.tools._pod_auth import mint_user_assertion as _mint_user_assertion
from unit.tools._pod_auth import signed_call_payload as _signed_call_payload


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeNats:
    """records :meth:`publish` + :meth:`publish_reply` calls.

    matches the canonical :class:`threetears.nats.NatsClient` wrapper
    surface ToolServer touches: kw-only ``subject`` + ``message`` for
    typed publishes, kw-only ``reply_subject`` + ``message`` for
    request-reply responses. ``raise_on_publish`` lets tests simulate
    a transient publish failure without breaking the publish_reply
    path that the inbound dispatch needs.
    """

    published: list[tuple[Subject, BaseModel]] = field(default_factory=list)
    replies: list[tuple[str, BaseModel]] = field(default_factory=list)
    raise_on_publish: BaseException | None = None

    async def publish(
        self,
        *,
        subject: Subject,
        message: BaseModel,
        reply_to: Subject | None = None,
    ) -> None:
        del reply_to
        self.published.append((subject, message))
        if self.raise_on_publish is not None:
            raise self.raise_on_publish

    async def publish_reply(self, *, reply_subject: str, message: BaseModel) -> None:
        self.replies.append((reply_subject, message))


class _StubTool(TearsTool):
    """TearsTool returning a fixed :class:`ToolResult`."""

    def __init__(
        self,
        *,
        name: str = "test.stub",
        version: str = "1.0",
        result: ToolResult | None = None,
        raise_exc: BaseException | None = None,
    ) -> None:
        self._name = name
        self._version = version
        self._result = result or ToolResult(success=True, content="ok")
        self._raise_exc = raise_exc

    async def execute(self, **kwargs: Any) -> ToolResult:
        del kwargs
        if self._raise_exc is not None:
            raise self._raise_exc
        return self._result

    def mcp_schema(self) -> MCPToolDefinition:
        return MCPToolDefinition(
            name=self._name,
            version=self._version,
            description="stub",
            input_schema={"type": "object", "properties": {}},
        )

    def mcp_name(self) -> str:
        return self._name

    def mcp_version(self) -> str:
        return self._version


def _make_msg(payload: dict[str, Any] | bytes) -> IncomingMessage:
    """build an :class:`IncomingMessage` envelope with ``payload``.

    :param payload: either a dict (JSON-serialized) or pre-encoded bytes
    :ptype payload: dict[str, Any] | bytes
    :return: wrapper-shaped envelope with a deterministic reply subject
    :rtype: IncomingMessage
    """
    if isinstance(payload, bytes):
        data = payload
    else:
        data = json.dumps(payload).encode("utf-8")
    return IncomingMessage(
        data=data,
        reply_subject="_INBOX.audit-test",
        subject="aibots.tools.internal.test-pod",
    )


def _audit_envelopes(nats: _FakeNats, subject_suffix: str) -> list[dict[str, Any]]:
    """extract decoded envelopes whose subject path ends with ``subject_suffix``."""
    result: list[dict[str, Any]] = []
    for subject, message in nats.published:
        if subject.path.endswith(subject_suffix):
            result.append(json.loads(message.model_dump_json()))
    return result


@pytest.fixture(autouse=True)
def _bind_namespace() -> None:
    """default ``aibots`` namespace so :class:`Subjects` builders are deterministic."""
    set_default_namespace("aibots")


# ---------------------------------------------------------------------------
# success path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_baseline_audit_emitted_on_success_path() -> None:
    """success dispatch publishes one ``tool.call`` envelope with outcome=success."""
    set_default_namespace("ns")
    nats = _FakeNats()
    owner_agent_id = uuid4()
    server = ToolServer(
        namespace="ns",
        nats_client=nats,
        pod_id="audit-pod",
        agent_id=owner_agent_id,
        namespace_collection=None,
        jwks_provider=_pod_jwks_provider,
        assertion_replay_guard=_PodReplayGuard(),
    )
    server.register(_StubTool())

    calling_agent_id = uuid4()
    user_id = uuid4()
    customer_id = uuid4()
    correlation_id = uuid4()
    # enforce-only, PRODUCTION shape: the handshake identity token is user-LESS; the audited
    # calling_agent / customer come from the VERIFIED handshake token, and the audited actor_user_id
    # comes from a SEPARATE bound user-assertion (``user_id`` drives the helper to mint one). this is
    # the shape a user-driven turn actually has -- the handshake token NEVER carries the per-turn
    # user -- so the audit actor attribution is exercised on a production-possible token shape.
    msg = _make_msg(
        _signed_call_payload(
            pod_id="audit-pod",
            tool_name="test.stub",
            tool_version="1.0",
            agent_id=calling_agent_id,
            customer_id=customer_id,
            user_id=user_id,
            correlation_id=str(correlation_id),
        )
    )

    await server.handle_call(msg)

    envelopes = _audit_envelopes(nats, ".audit.tool.call")
    assert len(envelopes) == 1
    env = envelopes[0]
    assert env["event_type"] == "tool.call"
    assert env["outcome"] == "success"
    assert env["action"] == "call"
    assert env["actor_user_id"] == str(user_id)
    assert env["calling_agent_id"] == str(calling_agent_id)
    assert env["owner_agent_id"] == str(owner_agent_id)
    assert env["customer_id"] == str(customer_id)
    assert env["resource_namespace_id"] is None
    assert env["resource_namespace_type"] is None
    assert env["correlation_id"] == str(correlation_id)
    assert env["details"]["tool_name"] == "test.stub"
    assert env["details"]["tool_version"] == "1.0"
    assert isinstance(env["details"]["duration_ms"], (int, float))
    assert env["details"]["duration_ms"] >= 0.0
    assert "failure_reason" not in env["details"]


@pytest.mark.asyncio
async def test_baseline_audit_subject_uses_namespace() -> None:
    """the published subject is ``{namespace}.audit.tool.call``."""
    set_default_namespace("proj")
    nats = _FakeNats()
    server = ToolServer(namespace="proj", nats_client=nats, namespace_collection=None)
    server.register(_StubTool())
    msg = _make_msg(
        {
            "tool_name": "test.stub",
            "tool_version": "1.0",
            "arguments": {},
            "context": {"correlation_id": str(uuid4())},
        },
    )

    await server.handle_call(msg)

    assert any(s.path == "proj.audit.tool.call" for s, _ in nats.published)


# ---------------------------------------------------------------------------
# failure paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_baseline_audit_outcome_failure_when_tool_returns_false() -> None:
    """tool returning success=False surfaces as outcome=failure."""
    set_default_namespace("ns")
    nats = _FakeNats()
    server = ToolServer(
        namespace="ns",
        nats_client=nats,
        pod_id="audit-pod",
        namespace_collection=None,
        jwks_provider=_pod_jwks_provider,
        assertion_replay_guard=_PodReplayGuard(),
    )
    server.register(
        _StubTool(result=ToolResult(success=False, content="", error="nope")),
    )
    msg = _make_msg(_signed_call_payload(pod_id="audit-pod", tool_name="test.stub", tool_version="1.0"))

    await server.handle_call(msg)

    envelopes = _audit_envelopes(nats, ".audit.tool.call")
    assert len(envelopes) == 1
    env = envelopes[0]
    assert env["outcome"] == "failure"
    assert env["details"]["failure_reason"] == "nope"


@pytest.mark.asyncio
async def test_baseline_audit_outcome_error_when_tool_raises() -> None:
    """tool raising an exception surfaces as outcome=error."""
    set_default_namespace("ns")
    nats = _FakeNats()
    server = ToolServer(
        namespace="ns",
        nats_client=nats,
        pod_id="audit-pod",
        namespace_collection=None,
        jwks_provider=_pod_jwks_provider,
        assertion_replay_guard=_PodReplayGuard(),
    )
    server.register(_StubTool(raise_exc=RuntimeError("boom")))
    msg = _make_msg(_signed_call_payload(pod_id="audit-pod", tool_name="test.stub", tool_version="1.0"))

    await server.handle_call(msg)

    envelopes = _audit_envelopes(nats, ".audit.tool.call")
    assert len(envelopes) == 1
    env = envelopes[0]
    assert env["outcome"] == "error"
    assert "boom" in env["details"]["failure_reason"]


@pytest.mark.asyncio
async def test_baseline_audit_outcome_failure_on_unknown_tool() -> None:
    """unknown tool key emits ``tool.call`` with outcome=failure."""
    set_default_namespace("ns")
    nats = _FakeNats()
    server = ToolServer(
        namespace="ns",
        nats_client=nats,
        pod_id="audit-pod",
        namespace_collection=None,
        jwks_provider=_pod_jwks_provider,
        assertion_replay_guard=_PodReplayGuard(),
    )
    # no tool registered
    msg = _make_msg(_signed_call_payload(pod_id="audit-pod", tool_name="missing.tool", tool_version="2.0"))

    await server.handle_call(msg)

    envelopes = _audit_envelopes(nats, ".audit.tool.call")
    assert len(envelopes) == 1
    env = envelopes[0]
    assert env["outcome"] == "failure"
    assert env["details"]["tool_name"] == "missing.tool"
    assert env["details"]["tool_version"] == "2.0"
    assert "missing.tool@2.0" in env["details"]["failure_reason"]


@pytest.mark.asyncio
async def test_baseline_audit_outcome_failure_on_malformed_request() -> None:
    """malformed JSON emits a baseline envelope with minimal fields."""
    set_default_namespace("ns")
    nats = _FakeNats()
    server = ToolServer(namespace="ns", nats_client=nats, namespace_collection=None)

    msg = _make_msg(b"<<not valid json>>")

    await server.handle_call(msg)

    envelopes = _audit_envelopes(nats, ".audit.tool.call")
    assert len(envelopes) == 1
    env = envelopes[0]
    assert env["outcome"] == "failure"
    assert env["actor_user_id"] is None
    assert env["calling_agent_id"] is None
    assert env["customer_id"] is None
    assert env["details"]["tool_name"] == ""
    assert env["details"]["tool_version"] == ""
    assert "malformed" in env["details"]["failure_reason"]


# ---------------------------------------------------------------------------
# emission gated on namespace + nats client
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_baseline_audit_skipped_when_no_nats_client() -> None:
    """server constructed with ``nats_url`` and no client skips emission."""
    # nats_url set but no connected client (serve not called) -> _nc is None
    server = ToolServer(
        nats_url="nats://localhost:1234",
        namespace="ns",
        namespace_collection=None,
    )
    server.register(_StubTool())

    msg = _make_msg(
        {
            "tool_name": "test.stub",
            "tool_version": "1.0",
            "arguments": {},
            "context": {"correlation_id": str(uuid4())},
        },
    )

    # no exception, and there is no nats client to observe publishes on
    await server.handle_call(msg)
    # sanity: dispatch returned cleanly even though no reply could be sent


# ---------------------------------------------------------------------------
# publish failure never breaks the dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_baseline_audit_publish_failure_does_not_taint_response() -> None:
    """a raising NATS publish is swallowed; reply still carries success."""
    set_default_namespace("ns")
    nats = _FakeNats(raise_on_publish=RuntimeError("nats offline"))
    server = ToolServer(
        namespace="ns",
        nats_client=nats,
        pod_id="audit-pod",
        namespace_collection=None,
        jwks_provider=_pod_jwks_provider,
        assertion_replay_guard=_PodReplayGuard(),
    )
    server.register(_StubTool())
    msg = _make_msg(_signed_call_payload(pod_id="audit-pod", tool_name="test.stub", tool_version="1.0"))

    # does not raise
    await server.handle_call(msg)

    # the dispatch reply landed via publish_reply (which does NOT
    # share the audit publish failure path) so the caller still sees
    # a success response.
    assert len(nats.replies) == 1
    _reply_subject, response = nats.replies[0]
    response_data = json.loads(response.model_dump_json())
    assert response_data["success"] is True


# ---------------------------------------------------------------------------
# correlation id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_baseline_audit_correlation_id_lifted_from_context() -> None:
    """when CallContext carries a correlation_id, the envelope reuses it."""
    set_default_namespace("ns")
    nats = _FakeNats()
    server = ToolServer(namespace="ns", nats_client=nats, namespace_collection=None)
    server.register(_StubTool())
    correlation_id = uuid4()
    msg = _make_msg(
        {
            "tool_name": "test.stub",
            "tool_version": "1.0",
            "arguments": {},
            "context": {"correlation_id": str(correlation_id)},
        },
    )

    await server.handle_call(msg)

    envelopes = _audit_envelopes(nats, ".audit.tool.call")
    assert envelopes[0]["correlation_id"] == str(correlation_id)


@pytest.mark.asyncio
async def test_baseline_audit_correlation_id_synthesized_when_context_missing() -> None:
    """no-context dispatch still emits with a freshly minted correlation id."""
    set_default_namespace("ns")
    nats = _FakeNats()
    server = ToolServer(namespace="ns", nats_client=nats, namespace_collection=None)
    server.register(_StubTool())
    msg = _make_msg(
        {
            "tool_name": "test.stub",
            "tool_version": "1.0",
            "arguments": {},
        },
    )

    await server.handle_call(msg)

    envelopes = _audit_envelopes(nats, ".audit.tool.call")
    assert len(envelopes) == 1
    env = envelopes[0]
    # parses as a UUID (any non-empty UUID is acceptable)
    assert UUID(env["correlation_id"])


# ---------------------------------------------------------------------------
# v0.13.9 user-assertion: the pod MIRRORS the registry proxy's user-assertion gate.
#
# the handshake identity token is one-per-pod and user-LESS, so a user-driven turn carries the
# per-turn VERIFIED user_id as a SECOND, Hub-minted, cnf-LESS user-assertion at
# ``context.user_identity_token``. ``ToolServer._verify_identity`` verifies it against the SAME
# issuer/JWKS, BINDS it to the handshake token (``sub`` + ``customer_id`` must match), and re-stamps
# ``user_id`` -- otherwise its defense-in-depth re-stamp would clobber the proxy-verified user_id
# back to None, dropping the audited actor_user_id and starving the per-user ToolContextManager.
# these mirror registry ``test_identity_verification.TestDispatchUserAssertion`` on the POD side,
# asserting through the baseline audit actor + the context-factory gate.
# ---------------------------------------------------------------------------


def _recording_factory(calls: list[tuple[UUID, UUID]]):
    """an async context_factory that records its (conversation_id, user_id) args.

    returns a sentinel cast to :class:`ToolContextManager` -- the stub tool ignores the manager, so
    the recorded CALL (with the verified, bound user_id) is what proves the per-user gate opened.
    """

    async def factory(conversation_id: UUID, user_id: UUID) -> ToolContextManager:
        calls.append((conversation_id, user_id))
        return cast(ToolContextManager, object())

    return factory


@pytest.mark.asyncio
async def test_bound_user_assertion_restamps_actor_user_id() -> None:
    """a bound user-assertion supplies the audited actor_user_id; the handshake token is user-less."""
    set_default_namespace("ns")
    nats = _FakeNats()
    agent_id, customer_id, real_user = uuid4(), uuid4(), uuid4()
    server = ToolServer(
        namespace="ns",
        nats_client=nats,
        pod_id="ua-pod",
        namespace_collection=None,
        jwks_provider=_pod_jwks_provider,
        assertion_replay_guard=_PodReplayGuard(),
    )
    server.register(_StubTool())
    msg = _make_msg(
        _signed_call_payload(
            pod_id="ua-pod",
            tool_name="test.stub",
            tool_version="1.0",
            agent_id=agent_id,
            customer_id=customer_id,
            user_id=real_user,  # mints a bound user-assertion (sub=agent_id, customer=customer_id)
        )
    )

    await server.handle_call(msg)

    env = _audit_envelopes(nats, ".audit.tool.call")[0]
    assert env["outcome"] == "success"  # the call DISPATCHED (was not denied)
    assert env["actor_user_id"] == str(real_user)  # from the bound user-assertion
    assert env["calling_agent_id"] == str(agent_id)  # from the handshake token
    assert env["customer_id"] == str(customer_id)


@pytest.mark.asyncio
async def test_bound_user_assertion_builds_per_user_context_manager() -> None:
    """a verified user_id + a conversation_id opens the per-user ToolContextManager factory gate."""
    set_default_namespace("ns")
    nats = _FakeNats()
    agent_id, customer_id, real_user, conv_id = uuid4(), uuid4(), uuid4(), uuid4()
    factory_calls: list[tuple[UUID, UUID]] = []
    server = ToolServer(
        namespace="ns",
        nats_client=nats,
        pod_id="ua-pod",
        namespace_collection=None,
        jwks_provider=_pod_jwks_provider,
        assertion_replay_guard=_PodReplayGuard(),
        context_factory=_recording_factory(factory_calls),
    )
    server.register(_StubTool())
    msg = _make_msg(
        _signed_call_payload(
            pod_id="ua-pod",
            tool_name="test.stub",
            tool_version="1.0",
            agent_id=agent_id,
            customer_id=customer_id,
            conversation_id=conv_id,
            user_id=real_user,
        )
    )

    await server.handle_call(msg)

    # the factory ran with the conversation + the VERIFIED, bound user_id (NOT None) -- proving the
    # user-assertion re-stamp reached the per-user context-manager gate.
    assert factory_calls == [(conv_id, real_user)]


@pytest.mark.asyncio
async def test_absent_user_assertion_leaves_actor_none_and_no_context_manager() -> None:
    """no user-assertion (agent on its own behalf): user_id stays None -> no actor, gate stays shut."""
    set_default_namespace("ns")
    nats = _FakeNats()
    conv_id = uuid4()
    factory_calls: list[tuple[UUID, UUID]] = []
    server = ToolServer(
        namespace="ns",
        nats_client=nats,
        pod_id="ua-pod",
        namespace_collection=None,
        jwks_provider=_pod_jwks_provider,
        assertion_replay_guard=_PodReplayGuard(),
        context_factory=_recording_factory(factory_calls),
    )
    server.register(_StubTool())
    msg = _make_msg(
        _signed_call_payload(
            pod_id="ua-pod",
            tool_name="test.stub",
            tool_version="1.0",
            conversation_id=conv_id,  # present, but with no user the per-user gate stays closed
        )
    )

    await server.handle_call(msg)

    env = _audit_envelopes(nats, ".audit.tool.call")[0]
    assert env["outcome"] == "success"  # an agent-on-own-behalf call still dispatches
    assert env["actor_user_id"] is None  # no per-turn user
    assert factory_calls == []  # the per-user context-manager gate stayed shut


@pytest.mark.asyncio
@pytest.mark.parametrize("flavor", ["sub_mismatch", "customer_mismatch", "expired", "invalid", "null_user"])
async def test_user_assertion_failclosed_denies(flavor: str) -> None:
    """a mis-bound / expired / invalid / user-less user-assertion is denied fail-closed (no dispatch).

    mirrors the proxy's TOOL_USER_IDENTITY_UNVERIFIED: the pod rejects the call before the tool runs,
    so the reply is an error and the baseline audit records a ``user-assertion verification failed``
    failure rather than a successful tool.call.
    """
    set_default_namespace("ns")
    nats = _FakeNats()
    agent_id, customer_id = uuid4(), uuid4()
    if flavor == "sub_mismatch":
        # minted for a DIFFERENT agent -> cross-agent replay
        bad = _mint_user_assertion(sub=uuid4(), customer_id=customer_id, user_id=uuid4())
    elif flavor == "customer_mismatch":
        # minted for a DIFFERENT customer -> cross-customer replay
        bad = _mint_user_assertion(sub=agent_id, customer_id=uuid4(), user_id=uuid4())
    elif flavor == "expired":
        bad = _mint_user_assertion(sub=agent_id, customer_id=customer_id, user_id=uuid4(), exp_delta=-120)
    elif flavor == "null_user":
        # bound but carries no user_id -> the proxy/pod's "carries no user_id" deny
        bad = _mint_user_assertion(sub=agent_id, customer_id=customer_id, user_id=None)
    else:  # invalid: not a verifiable JWS
        bad = "not.a.valid.jws"
    server = ToolServer(
        namespace="ns",
        nats_client=nats,
        pod_id="ua-pod",
        namespace_collection=None,
        jwks_provider=_pod_jwks_provider,
        assertion_replay_guard=_PodReplayGuard(),
    )
    server.register(_StubTool())
    msg = _make_msg(
        _signed_call_payload(
            pod_id="ua-pod",
            tool_name="test.stub",
            tool_version="1.0",
            agent_id=agent_id,
            customer_id=customer_id,
            user_assertion=bad,
        )
    )

    await server.handle_call(msg)

    # the tool never ran -> the reply is an error response.
    assert len(nats.replies) == 1
    _reply_subject, response = nats.replies[0]
    response_data = json.loads(response.model_dump_json())
    assert response_data["success"] is False
    assert "user-assertion verification failed" in response_data["error"]
    # ...and the baseline audit records the fail-closed denial, not a success.
    env = _audit_envelopes(nats, ".audit.tool.call")[0]
    assert env["outcome"] == "failure"
    assert env["actor_user_id"] is None
    assert "user-assertion verification failed" in env["details"]["failure_reason"]
