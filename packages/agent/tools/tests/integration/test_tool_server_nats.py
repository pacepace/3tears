"""integration tests for ToolServer against a NATS testcontainer.

uses the canonical session-scoped ``nats_container`` fixture from
:mod:`threetears.core.testing.fixtures` (registered at the workspace
root conftest) so the test suite spins up its own JetStream-enabled
NATS server rather than depending on a pre-running ``localhost:4222``.
``check_docker_available`` inside the fixture gates the suite on the
docker daemon -- a fresh checkout without docker still skips
gracefully, but with docker (the normal case) the tests run.
"""

from __future__ import annotations

import asyncio
import json
from datetime import timedelta
from typing import Any
from uuid import uuid4

import pytest

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.agent.tools.server import ToolServer
from threetears.nats import (
    IncomingMessage,
    NatsClient,
    Subject,
    Subjects,
    set_default_namespace,
)

# the canonical ``nats_container`` fixture is session-scoped on the
# session event loop. align the test event loop with that scope so the
# NATS connections established here land on the same loop that the
# fixture set up the container on.
pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio(loop_scope="session"),
]


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

    async def test_register_and_call_tool_via_nats(
        self,
        nats_container: str,
    ) -> None:
        """register tool, send call request via NATS, receive response.

        :param nats_container: NATS URL from the canonical testcontainer fixture
        :ptype nats_container: str
        :return: nothing
        :rtype: None
        """
        pod_id = f"integ-{uuid4().hex[:8]}"
        namespace = f"integ_{uuid4().hex[:8]}"

        server = ToolServer(
            nats_url=nats_container,
            namespace=namespace,
            pod_id=pod_id,
            heartbeat_interval=60.0,
            namespace_collection=None,
        )
        tool = IntegrationStubTool()
        server.register(tool)

        serve_task = asyncio.create_task(server.serve())
        await asyncio.sleep(0.2)

        try:
            set_default_namespace(namespace)
            nc = await NatsClient.connect(
                nats_url=nats_container,
                nats_subject_namespace=namespace,
                client_name="tool-server-itest",
            )

            call_subject = Subjects.tools_internal(pod_id)
            correlation_id = str(uuid4())
            # the CallRequest envelope moved correlation_id into the
            # nested CallContext. top-level ``correlation_id`` on the
            # request is explicitly rejected (extra="forbid").
            request_data = json.dumps(
                {
                    "tool_name": "integration.stub",
                    "tool_version": "1.0",
                    "arguments": {"message": "hello"},
                    "context": {"correlation_id": correlation_id},
                }
            ).encode("utf-8")

            response_bytes = await nc.request_raw(
                subject=call_subject,
                payload=request_data,
                timeout=timedelta(seconds=5),
            )
            response_data = json.loads(response_bytes)

            assert response_data["success"] is True, response_data.get("error")
            assert response_data["context"]["correlation_id"] == correlation_id
            content = json.loads(response_data["content"])
            assert content == {"message": "hello"}

            await nc.shutdown()
        finally:
            await server.shutdown()
            await asyncio.sleep(0.1)
            serve_task.cancel()
            try:
                await serve_task
            except asyncio.CancelledError:
                pass

    async def test_heartbeat_visible_on_nats(
        self,
        nats_container: str,
    ) -> None:
        """heartbeat messages are published and receivable on NATS.

        :param nats_container: NATS URL from the canonical testcontainer fixture
        :ptype nats_container: str
        :return: nothing
        :rtype: None
        """
        pod_id = f"hb-integ-{uuid4().hex[:8]}"
        namespace = f"hb_{uuid4().hex[:8]}"

        server = ToolServer(
            nats_url=nats_container,
            namespace=namespace,
            pod_id=pod_id,
            heartbeat_interval=0.1,
            namespace_collection=None,
        )
        tool = IntegrationStubTool()
        server.register(tool)

        heartbeats: list[dict[str, Any]] = []

        set_default_namespace(namespace)
        nc = await NatsClient.connect(
            nats_url=nats_container,
            nats_subject_namespace=namespace,
            client_name="tool-heartbeat-itest",
        )

        heartbeat_subject = Subjects.tools_heartbeat(pod_id)

        async def _on_heartbeat(msg: IncomingMessage) -> None:
            """collect heartbeat messages."""
            heartbeats.append(json.loads(msg.data))

        await nc.subscribe(subject=heartbeat_subject, cb=_on_heartbeat)

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
            await nc.shutdown()
