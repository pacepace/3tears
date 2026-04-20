"""unit tests for the baseline ``tool.call`` audit emission on dispatch.

audit-task-01 Phase 3 (AUD-03): :meth:`ToolServer._handle_call` emits
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
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.agent.tools.server import ToolServer


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeNats:
    """records ``publish`` calls; optionally raises on publish."""

    published: list[tuple[str, bytes]] = field(default_factory=list)
    raise_on_publish: BaseException | None = None

    async def publish(self, subject: str, payload: bytes) -> None:
        self.published.append((subject, payload))
        if self.raise_on_publish is not None:
            raise self.raise_on_publish


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


def _make_msg(payload: dict[str, Any] | bytes) -> MagicMock:
    msg = MagicMock()
    if isinstance(payload, bytes):
        msg.data = payload
    else:
        msg.data = json.dumps(payload).encode("utf-8")
    msg.respond = AsyncMock()
    return msg


def _audit_envelopes(nats: _FakeNats, subject_suffix: str) -> list[dict[str, Any]]:
    """extract decoded JSON envelopes with matching subject suffix."""
    result: list[dict[str, Any]] = []
    for subject, payload in nats.published:
        if subject.endswith(subject_suffix):
            result.append(json.loads(payload.decode("utf-8")))
    return result


# ---------------------------------------------------------------------------
# success path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_baseline_audit_emitted_on_success_path() -> None:
    """success dispatch publishes one ``tool.call`` envelope with outcome=success."""
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

    await server._handle_call(msg)

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

    await server._handle_call(msg)

    assert any(s == "proj.audit.tool.call" for s, _ in nats.published)


# ---------------------------------------------------------------------------
# failure paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_baseline_audit_outcome_failure_when_tool_returns_false() -> None:
    """tool returning success=False surfaces as outcome=failure."""
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

    await server._handle_call(msg)

    envelopes = _audit_envelopes(nats, ".audit.tool.call")
    assert len(envelopes) == 1
    env = envelopes[0]
    assert env["outcome"] == "failure"
    assert env["details"]["failure_reason"] == "nope"


@pytest.mark.asyncio
async def test_baseline_audit_outcome_error_when_tool_raises() -> None:
    """tool raising an exception surfaces as outcome=error."""
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

    await server._handle_call(msg)

    envelopes = _audit_envelopes(nats, ".audit.tool.call")
    assert len(envelopes) == 1
    env = envelopes[0]
    assert env["outcome"] == "error"
    assert "boom" in env["details"]["failure_reason"]


@pytest.mark.asyncio
async def test_baseline_audit_outcome_failure_on_unknown_tool() -> None:
    """unknown tool key emits ``tool.call`` with outcome=failure."""
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

    await server._handle_call(msg)

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
    nats = _FakeNats()
    server = ToolServer(namespace="ns", nats_client=nats, namespace_collection=None)

    msg = _make_msg(b"<<not valid json>>")

    await server._handle_call(msg)

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
        nats_url="nats://localhost:1234", namespace="ns", namespace_collection=None,
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
    await server._handle_call(msg)
    # sanity: response still succeeded
    msg.respond.assert_called_once()


# ---------------------------------------------------------------------------
# publish failure never breaks the dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_baseline_audit_publish_failure_does_not_taint_response() -> None:
    """a raising NATS publish is swallowed; response still carries success."""
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
    await server._handle_call(msg)

    response_data = json.loads(msg.respond.call_args[0][0])
    assert response_data["success"] is True


# ---------------------------------------------------------------------------
# correlation id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_baseline_audit_correlation_id_lifted_from_context() -> None:
    """when CallContext carries a correlation_id, the envelope reuses it."""
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

    await server._handle_call(msg)

    envelopes = _audit_envelopes(nats, ".audit.tool.call")
    assert envelopes[0]["correlation_id"] == str(correlation_id)


@pytest.mark.asyncio
async def test_baseline_audit_correlation_id_synthesized_when_context_missing() -> None:
    """no-context dispatch still emits with a freshly minted correlation id."""
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

    await server._handle_call(msg)

    envelopes = _audit_envelopes(nats, ".audit.tool.call")
    assert len(envelopes) == 1
    env = envelopes[0]
    # parses as a UUID (any non-empty UUID is acceptable)
    assert UUID(env["correlation_id"])
