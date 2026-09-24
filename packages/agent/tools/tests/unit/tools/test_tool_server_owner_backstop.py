"""an agent's in-process tool server serves its own agent and refuses every other caller.

The registry routes an agent's in-process tool only to that agent's own process. The serving pod
holds the same line on its own side, so a call that reaches it by any other path -- a registry
predating owner routing, a routing fault -- is refused rather than answered from this agent's
state for somebody else. A Tool Pod's server (a single-token pod-id) serves every caller, as
before.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID, uuid4

import pytest

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.agent.tools.call_scope import current_scope
from threetears.agent.tools.server import CallResponse, ToolServer
from threetears.core.security import PLATFORM_CUSTOMER_SENTINEL
from threetears.nats import IncomingMessage, Subjects

from unit.tools._pod_auth import StubReplayGuard, jwks_provider, signed_call_payload

_OWNER = UUID("01948a00-aaaa-7000-8000-00000000000a")
_PEER = UUID("01948a00-aaaa-7000-8000-00000000000b")
_IN_PROCESS_POD = Subjects.agent_inprocess_pod_id(_OWNER, "inst-1")
_TOOL_POD = "01948a00-dddd-7000-8000-0000000000d1"


class _RecordingTool(TearsTool):
    """records the verified caller of every call it runs."""

    def __init__(self) -> None:
        self.callers: list[UUID | None] = []

    async def execute(self, **kwargs: Any) -> ToolResult:
        scope = current_scope()
        self.callers.append(scope.context.agent_id if scope is not None and scope.context is not None else None)
        return ToolResult(success=True, content=json.dumps(kwargs))

    def mcp_schema(self) -> MCPToolDefinition:
        return MCPToolDefinition(
            name="test.stub", version="1.0", description="records", input_schema={"type": "object"}
        )

    def mcp_name(self) -> str:
        return "test.stub"

    def mcp_version(self) -> str:
        return "1.0"


# parity-exempt: subset stand-in for NatsClient exposing only the publish_reply the pod's handler answers on
class _RecordingNatsClient:
    def __init__(self) -> None:
        self.replies: list[Any] = []

    async def publish_reply(self, *, reply_subject: str, message: Any) -> None:
        self.replies.append(message)


def _server(pod_id: str) -> tuple[ToolServer, _RecordingTool, _RecordingNatsClient]:
    server = ToolServer(
        nats_url="nats://localhost:9999",
        pod_id=pod_id,
        jwks_provider=jwks_provider,
        assertion_replay_guard=StubReplayGuard(),
    )
    tool = _RecordingTool()
    server.register(tool)
    rec = _RecordingNatsClient()
    # the handler answers on ``self._nc``; installed directly rather than through ``serve``, which
    # would dial a real connection.
    setattr(server, "_nc", rec)
    return server, tool, rec


async def _call(
    server: ToolServer, rec: _RecordingNatsClient, pod_id: str, caller: UUID, **kwargs: Any
) -> CallResponse:
    await server.handle_call(
        IncomingMessage(
            data=json.dumps(signed_call_payload(pod_id=pod_id, agent_id=caller, **kwargs)).encode("utf-8"),
            reply_subject="_INBOX.test",
            subject=f"3tears.tools.internal.{pod_id}",
        )
    )
    reply: CallResponse = rec.replies[-1]
    return reply


class TestAnInProcessServerServesOnlyItsAgent:
    @pytest.mark.asyncio
    async def test_the_owning_agents_call_runs(self) -> None:
        server, tool, rec = _server(_IN_PROCESS_POD)

        reply = await _call(server, rec, _IN_PROCESS_POD, _OWNER)

        assert reply.success is True, reply.error
        assert reply.error_code is None
        assert tool.callers == [_OWNER]

    @pytest.mark.asyncio
    async def test_another_agents_call_is_refused_before_the_tool_runs(self) -> None:
        server, tool, rec = _server(_IN_PROCESS_POD)

        reply = await _call(server, rec, _IN_PROCESS_POD, _PEER)

        assert reply.success is False
        assert reply.error_code == "TOOL_CALLER_NOT_OWNER"
        assert reply.error is not None
        assert str(_PEER) in reply.error
        assert str(_OWNER) in reply.error
        assert tool.callers == []

    @pytest.mark.asyncio
    async def test_a_tool_pods_call_is_refused_too(self) -> None:
        """a tool pod has no agent at all, so it owns no agent's in-process server."""
        server, tool, rec = _server(_IN_PROCESS_POD)

        reply = await _call(server, rec, _IN_PROCESS_POD, uuid4(), customer_id=PLATFORM_CUSTOMER_SENTINEL)

        assert reply.error_code == "TOOL_CALLER_NOT_OWNER"
        assert tool.callers == []


class TestAToolPodServerServesEveryone:
    @pytest.mark.asyncio
    async def test_any_agent_and_any_tool_pod_reach_it(self) -> None:
        server, tool, rec = _server(_TOOL_POD)
        tool_pod_caller = uuid4()

        for caller, customer in ((_OWNER, None), (_PEER, None), (tool_pod_caller, PLATFORM_CUSTOMER_SENTINEL)):
            reply = await _call(server, rec, _TOOL_POD, caller, customer_id=customer)
            assert reply.success is True, reply.error

        assert tool.callers == [_OWNER, _PEER, tool_pod_caller]


class TestAPodIdNamingNoAgentIsRefusedAtConstruction:
    def test_a_dotted_id_without_an_agent_uuid_cannot_build_a_server(self) -> None:
        """such a server could never be probed under any agent's grant, so it fails loudly now."""
        with pytest.raises(ValueError, match="agent-A.inst-1"):
            ToolServer(nats_url="nats://localhost:9999", pod_id="agent-A.inst-1")
