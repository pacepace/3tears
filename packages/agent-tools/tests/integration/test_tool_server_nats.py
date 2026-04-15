"""integration tests for ToolServer with real NATS at localhost:4222.

requires NATS server running at nats://localhost:4222.
"""

from __future__ import annotations

import asyncio
import json
import socket
from typing import Any
from uuid import uuid4

import pytest
from nats.aio.client import Client as NatsClient

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.agent.tools.server import ToolServer


def _nats_reachable() -> bool:
    """check whether NATS is listening on localhost:4222.

    :return: True when a TCP connect succeeds within one second
    :rtype: bool
    """
    try:
        with socket.create_connection(("localhost", 4222), timeout=1):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _nats_reachable(), reason="NATS not running on localhost:4222")

_NATS_URL = "nats://localhost:4222"


# -- helpers --


class IntegrationStubTool(TearsTool):
    """stub TearsTool for integration testing."""

    async def execute(self, **kwargs: Any) -> ToolResult:
        """execute stub tool echoing arguments.

        :param kwargs: tool input parameters
        :ptype kwargs: Any
        :return: success result with echoed arguments
        :rtype: ToolResult
        """
        result = ToolResult(
            success=True,
            content=json.dumps(kwargs),
        )
        return result

    def mcp_schema(self) -> MCPToolDefinition:
        """return stub MCP schema.

        :return: tool definition
        :rtype: MCPToolDefinition
        """
        result = MCPToolDefinition(
            name="integration.stub",
            version="1.0",
            description="integration test stub tool",
            input_schema={
                "type": "object",
                "properties": {
                    "message": {"type": "string"},
                },
            },
        )
        return result

    def mcp_name(self) -> str:
        """return namespaced tool name.

        :return: namespaced tool name
        :rtype: str
        """
        return "integration.stub"

    def mcp_version(self) -> str:
        """return tool version.

        :return: version string
        :rtype: str
        """
        return "1.0"


# -- integration tests --


class TestToolServerNatsIntegration:
    """integration tests for ToolServer with real NATS."""

    @pytest.mark.asyncio
    async def test_register_and_call_tool_via_nats(self) -> None:
        """register tool, send call request via NATS, receive response."""
        pod_id = f"integ-{uuid4().hex[:8]}"
        namespace = f"integ_{uuid4().hex[:8]}"

        server = ToolServer(
            nats_url=_NATS_URL,
            namespace=namespace,
            pod_id=pod_id,
            heartbeat_interval=60.0,
        )
        tool = IntegrationStubTool()
        server.register(tool)

        serve_task = asyncio.create_task(server.serve())
        await asyncio.sleep(0.2)

        try:
            nc = NatsClient()
            await nc.connect(_NATS_URL)

            call_subject = f"{namespace}.tools.internal.{pod_id}"
            correlation_id = str(uuid4())
            request_data = json.dumps({
                "tool_name": "integration.stub",
                "tool_version": "1.0",
                "arguments": {"message": "hello"},
                "correlation_id": correlation_id,
            }).encode("utf-8")

            response = await nc.request(call_subject, request_data, timeout=5.0)
            response_data = json.loads(response.data)

            assert response_data["success"] is True
            assert response_data["correlation_id"] == correlation_id
            content = json.loads(response_data["content"])
            assert content == {"message": "hello"}

            await nc.close()
        finally:
            await server.shutdown()
            await asyncio.sleep(0.1)
            serve_task.cancel()
            try:
                await serve_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_heartbeat_visible_on_nats(self) -> None:
        """heartbeat messages are published and receivable on NATS."""
        pod_id = f"hb-integ-{uuid4().hex[:8]}"
        namespace = f"hb_{uuid4().hex[:8]}"

        server = ToolServer(
            nats_url=_NATS_URL,
            namespace=namespace,
            pod_id=pod_id,
            heartbeat_interval=0.1,
        )
        tool = IntegrationStubTool()
        server.register(tool)

        heartbeats: list[dict[str, Any]] = []

        nc = NatsClient()
        await nc.connect(_NATS_URL)

        heartbeat_subject = f"{namespace}.tools.heartbeat.{pod_id}"

        async def _on_heartbeat(msg: Any) -> None:
            """collect heartbeat messages."""
            heartbeats.append(json.loads(msg.data))

        await nc.subscribe(heartbeat_subject, cb=_on_heartbeat)

        serve_task = asyncio.create_task(server.serve())
        await asyncio.sleep(0.5)

        try:
            assert len(heartbeats) >= 1
            assert heartbeats[0]["pod_id"] == pod_id
            assert heartbeats[0]["tools_count"] == 1
            assert "timestamp" in heartbeats[0]
        finally:
            await server.shutdown()
            await asyncio.sleep(0.1)
            serve_task.cancel()
            try:
                await serve_task
            except asyncio.CancelledError:
                pass
            await nc.close()
