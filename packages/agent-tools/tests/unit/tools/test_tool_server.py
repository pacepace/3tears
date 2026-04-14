"""tests for ToolServer NATS-based tool serving."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.agent.tools.server import (
    CallRequest,
    CallResponse,
    HeartbeatMessage,
    RegistrationManifest,
    ToolManifestEntry,
    ToolServer,
)


# -- helpers --


class StubTool(TearsTool):
    """stub TearsTool for testing ToolServer."""

    def __init__(self, name: str = "test.stub", version: str = "1.0") -> None:
        """initialize stub tool.

        :param name: namespaced tool name
        :ptype name: str
        :param version: version string
        :ptype version: str
        """
        self._name = name
        self._version = version

    async def _execute(self, **kwargs: Any) -> ToolResult:
        """execute stub tool.

        :param kwargs: tool input parameters
        :ptype kwargs: Any
        :return: success result echoing arguments
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
            name=self._name,
            version=self._version,
            description="stub tool for testing",
            input_schema={"type": "object", "properties": {}},
        )
        return result

    def mcp_name(self) -> str:
        """return namespaced tool name.

        :return: namespaced tool name
        :rtype: str
        """
        return self._name

    def mcp_version(self) -> str:
        """return tool version.

        :return: version string
        :rtype: str
        """
        return self._version


class FailingTool(TearsTool):
    """stub TearsTool that always raises on execute."""

    async def _execute(self, **kwargs: Any) -> ToolResult:
        """execute and raise.

        :param kwargs: ignored
        :ptype kwargs: Any
        :return: never returns
        :rtype: ToolResult
        :raises RuntimeError: always
        """
        raise RuntimeError("intentional failure")

    def mcp_schema(self) -> MCPToolDefinition:
        """return failing tool MCP schema.

        :return: tool definition
        :rtype: MCPToolDefinition
        """
        result = MCPToolDefinition(
            name="test.failing",
            version="1.0",
            description="tool that always fails",
            input_schema={"type": "object", "properties": {}},
        )
        return result

    def mcp_name(self) -> str:
        """return namespaced tool name.

        :return: namespaced tool name
        :rtype: str
        """
        return "test.failing"

    def mcp_version(self) -> str:
        """return tool version.

        :return: version string
        :rtype: str
        """
        return "1.0"


def _make_nats_msg(data: dict[str, Any]) -> MagicMock:
    """create mock NATS message with given data payload.

    :param data: payload dict to serialize as JSON
    :ptype data: dict[str, Any]
    :return: mock message with .data and .respond attributes
    :rtype: MagicMock
    """
    msg = MagicMock()
    msg.data = json.dumps(data).encode("utf-8")
    msg.respond = AsyncMock()
    return msg


# -- registration tests --


class TestToolServerRegister:
    """tests for ToolServer.register method."""

    def test_register_adds_tool(self) -> None:
        """register stores tool keyed by name@version."""
        server = ToolServer(nats_url="nats://localhost:4222")
        tool = StubTool(name="test.stub", version="1.0")
        server.register(tool)
        assert "test.stub@1.0" in server._tools
        assert server._tools["test.stub@1.0"] is tool

    def test_register_multiple_tools(self) -> None:
        """register stores multiple distinct tools."""
        server = ToolServer(nats_url="nats://localhost:4222")
        tool_a = StubTool(name="test.alpha", version="1.0")
        tool_b = StubTool(name="test.beta", version="2.0")
        server.register(tool_a)
        server.register(tool_b)
        assert len(server._tools) == 2
        assert "test.alpha@1.0" in server._tools
        assert "test.beta@2.0" in server._tools

    def test_register_same_key_overwrites(self) -> None:
        """registering tool with same name@version overwrites previous."""
        server = ToolServer(nats_url="nats://localhost:4222")
        tool_a = StubTool(name="test.stub", version="1.0")
        tool_b = StubTool(name="test.stub", version="1.0")
        server.register(tool_a)
        server.register(tool_b)
        assert server._tools["test.stub@1.0"] is tool_b


# -- serve tests --


class TestToolServerServe:
    """tests for ToolServer.serve method (mocked NATS)."""

    @pytest.mark.asyncio
    async def test_serve_connects_to_nats(self) -> None:
        """serve connects to NATS server at configured URL."""
        server = ToolServer(nats_url="nats://localhost:9999", pod_id="test-pod-1")
        tool = StubTool()
        server.register(tool)

        mock_nc = AsyncMock()
        mock_nc.is_connected = True
        mock_nc.subscribe = AsyncMock()
        mock_nc.publish = AsyncMock()
        mock_nc.drain = AsyncMock()
        mock_nc.close = AsyncMock()

        with patch("threetears.agent.tools.server.nats_connect", return_value=mock_nc):
            # start serve in background, then shut down quickly
            serve_task = asyncio.create_task(server.serve())
            # give event loop a tick for serve to start
            await asyncio.sleep(0.05)
            await server.shutdown()
            await asyncio.sleep(0.05)
            serve_task.cancel()
            try:
                await serve_task
            except asyncio.CancelledError:
                pass

        mock_nc.subscribe.assert_called()

    @pytest.mark.asyncio
    async def test_serve_sends_registration_manifest(self) -> None:
        """serve publishes registration manifest on connect."""
        server = ToolServer(
            nats_url="nats://localhost:9999",
            namespace="testns",
            pod_id="test-pod-2",
        )
        tool = StubTool(name="test.stub", version="1.0")
        server.register(tool)

        mock_nc = AsyncMock()
        mock_nc.is_connected = True
        mock_nc.subscribe = AsyncMock()
        mock_nc.publish = AsyncMock()
        mock_nc.drain = AsyncMock()
        mock_nc.close = AsyncMock()

        with patch("threetears.agent.tools.server.nats_connect", return_value=mock_nc):
            serve_task = asyncio.create_task(server.serve())
            await asyncio.sleep(0.05)
            await server.shutdown()
            await asyncio.sleep(0.05)
            serve_task.cancel()
            try:
                await serve_task
            except asyncio.CancelledError:
                pass

        # find the registration publish call
        publish_calls = mock_nc.publish.call_args_list
        registration_call = None
        for call in publish_calls:
            subject = call[0][0] if call[0] else call[1].get("subject", "")
            if "tools.register" in subject:
                registration_call = call
                break

        assert registration_call is not None
        subject = registration_call[0][0]
        assert subject == "testns.tools.register"
        payload = json.loads(registration_call[0][1])
        assert payload["pod_id"] == "test-pod-2"
        assert len(payload["tools"]) == 1
        assert payload["tools"][0]["name"] == "test.stub"
        assert payload["tools"][0]["version"] == "1.0"

    @pytest.mark.asyncio
    async def test_serve_subscribes_to_call_subject(self) -> None:
        """serve subscribes to tool call subject with pod_id."""
        server = ToolServer(
            nats_url="nats://localhost:9999",
            namespace="testns",
            pod_id="test-pod-3",
        )
        tool = StubTool()
        server.register(tool)

        mock_nc = AsyncMock()
        mock_nc.is_connected = True
        mock_nc.subscribe = AsyncMock()
        mock_nc.publish = AsyncMock()
        mock_nc.drain = AsyncMock()
        mock_nc.close = AsyncMock()

        with patch("threetears.agent.tools.server.nats_connect", return_value=mock_nc):
            serve_task = asyncio.create_task(server.serve())
            await asyncio.sleep(0.05)
            await server.shutdown()
            await asyncio.sleep(0.05)
            serve_task.cancel()
            try:
                await serve_task
            except asyncio.CancelledError:
                pass

        subscribe_calls = mock_nc.subscribe.call_args_list
        subjects = [c[0][0] for c in subscribe_calls]
        assert "testns.tools.internal.test-pod-3" in subjects


# -- _handle_call tests --


class TestToolServerHandleCall:
    """tests for ToolServer._handle_call method."""

    @pytest.mark.asyncio
    async def test_handle_call_routes_to_correct_tool(self) -> None:
        """_handle_call dispatches to tool matching name@version."""
        server = ToolServer(nats_url="nats://localhost:9999")
        tool = StubTool(name="test.stub", version="1.0")
        server.register(tool)

        correlation_id = str(uuid4())
        msg = _make_nats_msg({
            "tool_name": "test.stub",
            "tool_version": "1.0",
            "arguments": {"key": "value"},
            "correlation_id": correlation_id,
        })

        await server._handle_call(msg)

        msg.respond.assert_called_once()
        response_data = json.loads(msg.respond.call_args[0][0])
        assert response_data["success"] is True
        assert response_data["correlation_id"] == correlation_id
        content = json.loads(response_data["content"])
        assert content == {"key": "value"}

    @pytest.mark.asyncio
    async def test_handle_call_returns_tool_result_on_success(self) -> None:
        """_handle_call returns serialized ToolResult with success=True."""
        server = ToolServer(nats_url="nats://localhost:9999")
        tool = StubTool(name="test.stub", version="1.0")
        server.register(tool)

        msg = _make_nats_msg({
            "tool_name": "test.stub",
            "tool_version": "1.0",
            "arguments": {},
            "correlation_id": "corr-1",
        })

        await server._handle_call(msg)

        response_data = json.loads(msg.respond.call_args[0][0])
        assert response_data["success"] is True
        assert response_data["error"] is None

    @pytest.mark.asyncio
    async def test_handle_call_returns_error_on_unknown_tool(self) -> None:
        """_handle_call returns error response for unregistered tool."""
        server = ToolServer(nats_url="nats://localhost:9999")

        msg = _make_nats_msg({
            "tool_name": "nonexistent.tool",
            "tool_version": "1.0",
            "arguments": {},
            "correlation_id": "corr-2",
        })

        await server._handle_call(msg)

        response_data = json.loads(msg.respond.call_args[0][0])
        assert response_data["success"] is False
        assert "nonexistent.tool@1.0" in response_data["error"]
        assert response_data["correlation_id"] == "corr-2"

    @pytest.mark.asyncio
    async def test_handle_call_returns_error_on_execution_failure(self) -> None:
        """_handle_call returns error response when tool raises exception."""
        server = ToolServer(nats_url="nats://localhost:9999")
        tool = FailingTool()
        server.register(tool)

        msg = _make_nats_msg({
            "tool_name": "test.failing",
            "tool_version": "1.0",
            "arguments": {},
            "correlation_id": "corr-3",
        })

        await server._handle_call(msg)

        response_data = json.loads(msg.respond.call_args[0][0])
        assert response_data["success"] is False
        assert "intentional failure" in response_data["error"]
        assert response_data["correlation_id"] == "corr-3"

    @pytest.mark.asyncio
    async def test_handle_call_returns_error_on_malformed_request(self) -> None:
        """_handle_call returns error response for invalid JSON payload."""
        server = ToolServer(nats_url="nats://localhost:9999")

        msg = MagicMock()
        msg.data = b"not valid json"
        msg.respond = AsyncMock()

        await server._handle_call(msg)

        msg.respond.assert_called_once()
        response_data = json.loads(msg.respond.call_args[0][0])
        assert response_data["success"] is False
        assert response_data["error"] is not None


# -- heartbeat tests --


class TestToolServerHeartbeat:
    """tests for ToolServer heartbeat loop."""

    @pytest.mark.asyncio
    async def test_heartbeat_published_at_interval(self) -> None:
        """heartbeat publishes to correct subject with pod_id and tools_count."""
        server = ToolServer(
            nats_url="nats://localhost:9999",
            namespace="testns",
            pod_id="hb-pod",
            heartbeat_interval=0.05,
        )
        tool = StubTool()
        server.register(tool)

        mock_nc = AsyncMock()
        mock_nc.is_connected = True
        mock_nc.subscribe = AsyncMock()
        mock_nc.publish = AsyncMock()
        mock_nc.drain = AsyncMock()
        mock_nc.close = AsyncMock()

        with patch("threetears.agent.tools.server.nats_connect", return_value=mock_nc):
            serve_task = asyncio.create_task(server.serve())
            # wait for at least one heartbeat
            await asyncio.sleep(0.15)
            await server.shutdown()
            await asyncio.sleep(0.05)
            serve_task.cancel()
            try:
                await serve_task
            except asyncio.CancelledError:
                pass

        heartbeat_calls = [
            c for c in mock_nc.publish.call_args_list
            if "heartbeat" in (c[0][0] if c[0] else "")
        ]
        assert len(heartbeat_calls) >= 1
        subject = heartbeat_calls[0][0][0]
        assert subject == "testns.tools.heartbeat.hb-pod"
        payload = json.loads(heartbeat_calls[0][0][1])
        assert payload["pod_id"] == "hb-pod"
        assert payload["tools_count"] == 1
        assert "timestamp" in payload


# -- shutdown tests --


class TestToolServerShutdown:
    """tests for ToolServer.shutdown method."""

    @pytest.mark.asyncio
    async def test_shutdown_stops_heartbeat_and_drains_nats(self) -> None:
        """shutdown cancels heartbeat task and drains NATS connection."""
        server = ToolServer(
            nats_url="nats://localhost:9999",
            pod_id="shutdown-pod",
            heartbeat_interval=0.05,
        )
        tool = StubTool()
        server.register(tool)

        mock_nc = AsyncMock()
        mock_nc.is_connected = True
        mock_nc.subscribe = AsyncMock()
        mock_nc.publish = AsyncMock()
        mock_nc.drain = AsyncMock()
        mock_nc.close = AsyncMock()

        with patch("threetears.agent.tools.server.nats_connect", return_value=mock_nc):
            serve_task = asyncio.create_task(server.serve())
            await asyncio.sleep(0.05)
            assert server._running is True
            await server.shutdown()
            await asyncio.sleep(0.05)
            serve_task.cancel()
            try:
                await serve_task
            except asyncio.CancelledError:
                pass

        assert server._running is False
        mock_nc.drain.assert_called_once()
        mock_nc.close.assert_called_once()


# -- wire format model tests --


class TestWireFormatModels:
    """tests for Pydantic wire format models."""

    def test_call_request_serialization(self) -> None:
        """CallRequest serializes to JSON with all required fields."""
        req = CallRequest(
            tool_name="test.stub",
            tool_version="1.0",
            arguments={"key": "value"},
            correlation_id="corr-123",
        )
        data = json.loads(req.model_dump_json())
        assert data["tool_name"] == "test.stub"
        assert data["tool_version"] == "1.0"
        assert data["arguments"] == {"key": "value"}
        assert data["correlation_id"] == "corr-123"

    def test_call_response_serialization(self) -> None:
        """CallResponse serializes to JSON with all fields."""
        resp = CallResponse(
            success=True,
            content="result text",
            metadata=None,
            error=None,
            correlation_id="corr-456",
        )
        data = json.loads(resp.model_dump_json())
        assert data["success"] is True
        assert data["content"] == "result text"
        assert data["metadata"] is None
        assert data["error"] is None
        assert data["correlation_id"] == "corr-456"

    def test_registration_manifest_serialization(self) -> None:
        """RegistrationManifest serializes with pod_id and tools list."""
        manifest = RegistrationManifest(
            pod_id="pod-abc",
            tools=[
                ToolManifestEntry(
                    name="test.stub",
                    version="1.0",
                    description="test tool",
                    input_schema={"type": "object"},
                ),
            ],
        )
        data = json.loads(manifest.model_dump_json())
        assert data["pod_id"] == "pod-abc"
        assert len(data["tools"]) == 1
        assert data["tools"][0]["name"] == "test.stub"

    def test_heartbeat_message_serialization(self) -> None:
        """HeartbeatMessage serializes with pod_id, timestamp, and tools_count."""
        hb = HeartbeatMessage(
            pod_id="pod-xyz",
            timestamp="2026-01-01T00:00:00+00:00",
            tools_count=3,
        )
        data = json.loads(hb.model_dump_json())
        assert data["pod_id"] == "pod-xyz"
        assert data["tools_count"] == 3
        assert data["timestamp"] == "2026-01-01T00:00:00+00:00"


# -- probe and wait_until_ready tests --


class TestToolServerProbe:
    """tests for reachability probe handling and wait_until_ready."""

    @pytest.mark.asyncio
    async def test_serve_subscribes_to_probe_subject(self) -> None:
        """serve subscribes to {namespace}.tools.probe.{pod_id} before publishing registration."""
        server = ToolServer(
            nats_url="nats://localhost:9999",
            namespace="testns",
            pod_id="probe-pod-1",
        )
        tool = StubTool()
        server.register(tool)

        mock_nc = AsyncMock()
        mock_nc.is_connected = True
        mock_nc.subscribe = AsyncMock()
        mock_nc.publish = AsyncMock()
        mock_nc.drain = AsyncMock()
        mock_nc.close = AsyncMock()

        with patch("threetears.agent.tools.server.nats_connect", return_value=mock_nc):
            serve_task = asyncio.create_task(server.serve())
            await asyncio.sleep(0.05)
            await server.shutdown()
            await asyncio.sleep(0.05)
            serve_task.cancel()
            try:
                await serve_task
            except asyncio.CancelledError:
                pass

        subscribe_calls = mock_nc.subscribe.call_args_list
        subjects = [c[0][0] for c in subscribe_calls]
        assert "testns.tools.probe.probe-pod-1" in subjects

    @pytest.mark.asyncio
    async def test_serve_subscribes_before_publishing_registration(self) -> None:
        """probe and call subscriptions must be active before registration publish."""
        server = ToolServer(
            nats_url="nats://localhost:9999",
            namespace="testns",
            pod_id="order-pod",
        )
        tool = StubTool()
        server.register(tool)

        order: list[str] = []

        async def record_subscribe(*args: Any, **kwargs: Any) -> AsyncMock:
            """capture subscribe call order."""
            subject = args[0] if args else kwargs.get("subject", "")
            order.append(f"subscribe:{subject}")
            return AsyncMock()

        async def record_publish(*args: Any, **kwargs: Any) -> None:
            """capture publish call order."""
            subject = args[0] if args else kwargs.get("subject", "")
            order.append(f"publish:{subject}")

        mock_nc = AsyncMock()
        mock_nc.is_connected = True
        mock_nc.subscribe = AsyncMock(side_effect=record_subscribe)
        mock_nc.publish = AsyncMock(side_effect=record_publish)
        mock_nc.drain = AsyncMock()
        mock_nc.close = AsyncMock()

        with patch("threetears.agent.tools.server.nats_connect", return_value=mock_nc):
            serve_task = asyncio.create_task(server.serve())
            await asyncio.sleep(0.05)
            await server.shutdown()
            await asyncio.sleep(0.05)
            serve_task.cancel()
            try:
                await serve_task
            except asyncio.CancelledError:
                pass

        # find first registration publish index and ensure probe subscribe precedes it
        first_register_idx = next(
            (i for i, evt in enumerate(order) if "publish:testns.tools.register" in evt),
            -1,
        )
        assert first_register_idx > 0, "registration publish must happen after subscriptions"
        assert any(
            evt == "subscribe:testns.tools.probe.order-pod"
            for evt in order[:first_register_idx]
        ), "probe subscription must be established before registration publish"

    @pytest.mark.asyncio
    async def test_handle_probe_responds_with_ack(self) -> None:
        """_handle_probe replies with ProbeAck carrying pod_id and ready=True."""
        server = ToolServer(nats_url="nats://localhost:9999", pod_id="ack-pod")

        msg = MagicMock()
        msg.data = b'{"pod_id": "ack-pod"}'
        msg.respond = AsyncMock()

        await server._handle_probe(msg)

        msg.respond.assert_called_once()
        payload = json.loads(msg.respond.call_args[0][0])
        assert payload["pod_id"] == "ack-pod"
        assert payload["ready"] is True

    @pytest.mark.asyncio
    async def test_handle_probe_sets_ready_event(self) -> None:
        """_handle_probe sets the internal ready event on first successful probe."""
        server = ToolServer(nats_url="nats://localhost:9999", pod_id="ready-pod")
        assert server._ready_event.is_set() is False

        msg = MagicMock()
        msg.data = b'{"pod_id": "ready-pod"}'
        msg.respond = AsyncMock()

        await server._handle_probe(msg)

        assert server._ready_event.is_set() is True

    @pytest.mark.asyncio
    async def test_wait_until_ready_unblocks_after_probe(self) -> None:
        """wait_until_ready returns True once a probe has been handled."""
        server = ToolServer(nats_url="nats://localhost:9999", pod_id="wait-pod")

        msg = MagicMock()
        msg.data = b'{"pod_id": "wait-pod"}'
        msg.respond = AsyncMock()

        async def probe_later() -> None:
            """deliver probe after a small delay."""
            await asyncio.sleep(0.02)
            await server._handle_probe(msg)

        probe_task = asyncio.create_task(probe_later())
        ready = await server.wait_until_ready(timeout=1.0)
        await probe_task

        assert ready is True

    @pytest.mark.asyncio
    async def test_wait_until_ready_returns_false_on_timeout(self) -> None:
        """wait_until_ready returns False when no probe arrives within timeout."""
        server = ToolServer(nats_url="nats://localhost:9999", pod_id="slow-pod")
        ready = await server.wait_until_ready(timeout=0.05)
        assert ready is False
