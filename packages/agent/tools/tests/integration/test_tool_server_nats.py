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
from unit.tools._pod_auth import jwks_provider as _pod_jwks_provider
from unit.tools._pod_auth import signed_call_payload as _signed_call_payload

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
        """register tool, send a FULLY-AUTHENTICATED call request via NATS, receive response.

        enforce-only (v0.13.9): the pod re-verifies the Hub identity token AND the proxy's
        body-bound assertion on every inbound call, unconditionally and fail-closed, before
        dispatch. so a real round-trip must present both -- a bare envelope is correctly rejected.
        the call is built with the shared :func:`_signed_call_payload` scaffolding and the server is
        wired with the matching :func:`_pod_jwks_provider` (the combined Hub-identity + proxy-assertion
        JWKS), exercising the live-NATS happy path under the production verification contract. the
        pod provisions its own NATS-KV replay guard in ``serve()`` against the JetStream container.

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
            jwks_provider=_pod_jwks_provider,
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
            # the enforce-only pod verifies the identity token + the proxy assertion (bound to THIS
            # pod_id + the call body) before dispatch; ``_signed_call_payload`` builds the matching
            # production-shape envelope. correlation_id lives on the nested CallContext (top-level
            # ``correlation_id`` is rejected, extra="forbid").
            request_data = json.dumps(
                _signed_call_payload(
                    pod_id=pod_id,
                    tool_name="integration.stub",
                    tool_version="1.0",
                    arguments={"message": "hello"},
                    correlation_id=correlation_id,
                )
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

    async def test_composite_agent_pod_id_routes_end_to_end(
        self,
        nats_container: str,
    ) -> None:
        """an agent IN-PROCESS tool server keyed on the ``{agent_id}.{instance}`` composite routes.

        the isolation fix scopes the agent's in-process tool subjects on the AUTHENTICATED agent
        subtree by running the in-process :class:`ToolServer` under the two-token
        ``{agent_id}.{instance}`` composite pod-id (``Subjects.agent_inprocess_pod_id``). this proves
        the composite still routes end-to-end against real NATS: the server subscribes
        ``tools.internal.{agent_id}.{instance}`` (structural dot preserved) and a forward to that
        exact subject -- with a proxy assertion bound to the composite pod-id -- round-trips. a
        sanitize-collapsed single token would have broken the registry-grant match (a
        ToolReadinessTimeout); the two-token subject must survive intact for routing to work.

        :param nats_container: NATS URL from the canonical testcontainer fixture
        :ptype nats_container: str
        :return: nothing
        :rtype: None
        """
        agent_id = str(uuid4())
        instance_id = str(uuid4())
        composite_pod_id = Subjects.agent_inprocess_pod_id(agent_id, instance_id)
        assert composite_pod_id == f"{agent_id}.{instance_id}"  # two tokens, structural dot intact
        namespace = f"composite_{uuid4().hex[:8]}"

        server = ToolServer(
            nats_url=nats_container,
            namespace=namespace,
            pod_id=composite_pod_id,
            heartbeat_interval=60.0,
            namespace_collection=None,
            jwks_provider=_pod_jwks_provider,
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
                client_name="composite-itest",
            )

            # the registry would forward on the registered (composite) pod-id verbatim; the rendered
            # subject is a TWO-token ``tools.internal.{agent_id}.{instance}`` under the agent subtree.
            call_subject = Subjects.tools_internal(composite_pod_id)
            assert call_subject.path == f"{namespace}.tools.internal.{agent_id}.{instance_id}"
            correlation_id = str(uuid4())
            request_data = json.dumps(
                _signed_call_payload(
                    pod_id=composite_pod_id,
                    tool_name="integration.stub",
                    tool_version="1.0",
                    arguments={"message": "composite"},
                    correlation_id=correlation_id,
                )
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
            assert content == {"message": "composite"}

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
