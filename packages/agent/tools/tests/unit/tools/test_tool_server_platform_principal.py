"""the pod's mirror gate admits the principal the registry admits, and refuses what it refuses.

The registry reads a token whose customer claim is the platform sentinel as a
tool pod with no customer and evaluates it on its own grant. The serving pod
re-verifies the same token at its own door; a pod that parsed the claim as a
customer UUID there would refuse every call the registry had just admitted,
which is exactly the composed failure that shipped once. So the pod reads the
principal through the same function the registry does, and the two cannot
disagree about the same token.
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
from threetears.nats import IncomingMessage

from unit.tools._pod_auth import StubReplayGuard, jwks_provider, signed_call_payload

_POD_ID = "test-pod"


class _ScopeRecordingTool(TearsTool):
    """echoes its arguments and records the call scope's context it ran under."""

    def __init__(self) -> None:
        self.contexts: list[Any] = []

    async def execute(self, **kwargs: Any) -> ToolResult:
        scope = current_scope()
        self.contexts.append(scope.context if scope is not None else None)
        return ToolResult(success=True, content=json.dumps(kwargs))

    def mcp_schema(self) -> MCPToolDefinition:
        return MCPToolDefinition(
            name="test.stub", version="1.0", description="records scope", input_schema={"type": "object"}
        )

    def mcp_name(self) -> str:
        return "test.stub"

    def mcp_version(self) -> str:
        return "1.0"


# parity-exempt: subset stand-in for NatsClient exposing only the publish_reply the pod's handler answers on
class _RecordingNatsClient:
    def __init__(self) -> None:
        self.replies: list[tuple[str, Any]] = []

    async def publish_reply(self, *, reply_subject: str, message: Any) -> None:
        self.replies.append((reply_subject, message))


def _server() -> tuple[ToolServer, _ScopeRecordingTool, _RecordingNatsClient]:
    server = ToolServer(
        nats_url="nats://localhost:9999",
        pod_id=_POD_ID,
        jwks_provider=jwks_provider,
        assertion_replay_guard=StubReplayGuard(),
    )
    tool = _ScopeRecordingTool()
    server.register(tool)
    rec = _RecordingNatsClient()
    # the handler answers on ``self._nc``; installed directly rather than through ``serve``, which
    # would dial a real connection. setattr keeps the test from binding to the private slot's name.
    setattr(server, "_nc", rec)
    return server, tool, rec


def _msg(payload: dict[str, Any]) -> IncomingMessage:
    return IncomingMessage(
        data=json.dumps(payload).encode("utf-8"),
        reply_subject="_INBOX.test",
        subject=f"3tears.tools.internal.{_POD_ID}",
    )


class TestATooPodPrincipalReachesTheTool:
    @pytest.mark.asyncio
    async def test_a_sentinel_customer_token_is_admitted_and_the_tool_runs_with_no_customer(self) -> None:
        server, tool, rec = _server()
        pod_id = uuid4()

        await server.handle_call(
            _msg(
                signed_call_payload(
                    pod_id=_POD_ID,
                    arguments={"text": "hi"},
                    agent_id=pod_id,
                    customer_id=PLATFORM_CUSTOMER_SENTINEL,
                )
            )
        )

        _subject, reply = rec.replies[-1]
        assert isinstance(reply, CallResponse)
        assert reply.success is True, reply.error
        assert json.loads(reply.content) == {"text": "hi"}
        assert len(tool.contexts) == 1
        context = tool.contexts[0]
        assert context.agent_id == pod_id  # the VERIFIED principal, re-stamped
        assert context.customer_id is None  # the sentinel was read as no customer
        assert context.user_id is None

    @pytest.mark.asyncio
    async def test_a_customer_uuid_token_still_stamps_its_customer(self) -> None:
        """the A/B: the same gate over an agent's token keeps the customer."""
        server, tool, rec = _server()
        agent_id, customer_id = uuid4(), uuid4()

        await server.handle_call(_msg(signed_call_payload(pod_id=_POD_ID, agent_id=agent_id, customer_id=customer_id)))

        assert rec.replies[-1][1].success is True
        assert tool.contexts[0].agent_id == agent_id
        assert tool.contexts[0].customer_id == customer_id


class TestATooPodPrincipalCannotBorrowAUser:
    @pytest.mark.asyncio
    async def test_a_user_assertion_on_a_sentinel_token_is_refused_before_the_tool(self) -> None:
        # a pod acts on nobody's behalf. the assertion here is minted BOUND to the pod token --
        # same sub, same sentinel customer, same conversation -- so the claim-equality binding
        # alone would admit it; the refusal has to be the pod's own.
        server, tool, rec = _server()
        pod_id, conversation_id = uuid4(), uuid4()

        await server.handle_call(
            _msg(
                signed_call_payload(
                    pod_id=_POD_ID,
                    agent_id=pod_id,
                    customer_id=PLATFORM_CUSTOMER_SENTINEL,
                    conversation_id=conversation_id,
                    user_id=UUID("01948a00-0000-7000-8000-00000000cafe"),
                )
            )
        )

        _subject, reply = rec.replies[-1]
        assert reply.success is False
        assert reply.error is not None
        assert "user-assertion verification failed" in reply.error
        assert tool.contexts == []  # the tool never ran
