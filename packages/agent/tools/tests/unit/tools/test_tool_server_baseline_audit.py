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
from typing import Any
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.agent.tools.server import ToolServer
from threetears.nats import IncomingMessage, Subject, set_default_namespace


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
        subject="3tears.tools.internal.test-pod",
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
    """default ``3tears`` namespace so :class:`Subjects` builders are deterministic."""
    set_default_namespace("3tears")


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
        agent_id=owner_agent_id,
        namespace_collection=None,
    )
    server.register(_StubTool())

    calling_agent_id = uuid4()
    user_id = uuid4()
    customer_id = uuid4()
    correlation_id = uuid4()
    msg = _make_msg(
        {
            "tool_name": "test.stub",
            "tool_version": "1.0",
            "arguments": {},
            "context": {
                "user_id": str(user_id),
                "customer_id": str(customer_id),
                "agent_id": str(calling_agent_id),
                "correlation_id": str(correlation_id),
            },
        },
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
    server = ToolServer(namespace="ns", nats_client=nats, namespace_collection=None)
    server.register(
        _StubTool(result=ToolResult(success=False, content="", error="nope")),
    )
    msg = _make_msg(
        {
            "tool_name": "test.stub",
            "tool_version": "1.0",
            "arguments": {},
            "context": {"correlation_id": str(uuid4())},
        },
    )

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
    server = ToolServer(namespace="ns", nats_client=nats, namespace_collection=None)
    server.register(_StubTool(raise_exc=RuntimeError("boom")))
    msg = _make_msg(
        {
            "tool_name": "test.stub",
            "tool_version": "1.0",
            "arguments": {},
            "context": {"correlation_id": str(uuid4())},
        },
    )

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
    server = ToolServer(namespace="ns", nats_client=nats, namespace_collection=None)
    # no tool registered
    msg = _make_msg(
        {
            "tool_name": "missing.tool",
            "tool_version": "2.0",
            "arguments": {},
            "context": {"correlation_id": str(uuid4())},
        },
    )

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
    server = ToolServer(namespace="ns", nats_client=nats, namespace_collection=None)
    server.register(_StubTool())
    msg = _make_msg(
        {
            "tool_name": "test.stub",
            "tool_version": "1.0",
            "arguments": {},
            "context": {"correlation_id": str(uuid4())},
        },
    )

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
