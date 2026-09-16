"""the pod hands its replay guard the proxy assertion's signed issue time.

The guard refuses an assertion issued before its nonce bucket was created -- the only thing that
still refuses a replay after a broker restart empties that bucket. It can only do that with the
time the registry SIGNED. Handing it the pod's own clock would make every replay look new.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import jwt
import pytest

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.agent.tools.server import CallResponse, ToolServer
from threetears.nats import IncomingMessage

from unit.tools._pod_auth import StubReplayGuard, jwks_provider, signed_call_payload

_POD_ID = "test-pod"


class _EchoTool(TearsTool):
    """echoes its arguments."""

    async def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, content=json.dumps(kwargs))

    def mcp_schema(self) -> MCPToolDefinition:
        return MCPToolDefinition(name="test.stub", version="1.0", description="echo", input_schema={"type": "object"})

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


class TestTheGuardSeesTheSignedIssueTime:
    @pytest.mark.asyncio
    async def test_the_assertions_signed_iat_reaches_the_guard(self) -> None:
        guard = StubReplayGuard()
        server = ToolServer(
            nats_url="nats://localhost:9999",
            pod_id=_POD_ID,
            jwks_provider=jwks_provider,
            assertion_replay_guard=guard,
        )
        server.register(_EchoTool())
        rec = _RecordingNatsClient()
        # the handler answers on ``self._nc``; installed directly rather than through ``serve``,
        # which would dial a real connection.
        setattr(server, "_nc", rec)
        payload = signed_call_payload(pod_id=_POD_ID, agent_id=uuid4(), customer_id=uuid4())

        await server.handle_call(
            IncomingMessage(
                data=json.dumps(payload).encode("utf-8"),
                reply_subject="_INBOX.test",
                subject=f"3tears.tools.internal.{_POD_ID}",
            )
        )

        reply = rec.replies[-1][1]
        assert isinstance(reply, CallResponse)
        assert reply.success is True, reply.error
        signed_iat = jwt.decode(payload["proxy_assertion"], options={"verify_signature": False})["iat"]
        assert guard.issued_at == [datetime.fromtimestamp(signed_iat, UTC)]
