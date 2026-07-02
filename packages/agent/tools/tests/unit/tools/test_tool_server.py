"""tests for ToolServer NATS-based tool serving."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from pydantic import BaseModel

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.agent.tools.server import (
    CallRequest,
    CallResponse,
    HeartbeatMessage,
    RegistrationManifest,
    ToolManifestEntry,
    ToolServer,
)
from threetears.nats import IncomingMessage, Subject


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

    async def execute(self, **kwargs: Any) -> ToolResult:
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

    async def execute(self, **kwargs: Any) -> ToolResult:
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


def _make_nats_msg(
    data: dict[str, Any],
    *,
    reply_subject: str = "_INBOX.test",
) -> IncomingMessage:
    """build an :class:`IncomingMessage` envelope for handler tests.

    handlers receive the wrapper's typed envelope (``data`` +
    ``reply_subject``); the legacy ``msg.respond(...)`` shape was
    replaced by ``self._respond(msg, response)`` which routes through
    :meth:`NatsClient.publish_reply` against the inbound reply
    subject. tests inject a stub ``NatsClient`` on ``server._nc`` so
    they can assert which response landed on which reply subject.

    :param data: payload dict to serialize as JSON
    :ptype data: dict[str, Any]
    :param reply_subject: reply inbox subject the handler will respond on
    :ptype reply_subject: str
    :return: wrapper-shaped envelope
    :rtype: IncomingMessage
    """
    return IncomingMessage(
        data=json.dumps(data).encode("utf-8"),
        reply_subject=reply_subject,
        subject="3tears.tools.internal.test-pod",
    )


class _RecordingNatsClient:
    """tiny stand-in for :class:`threetears.nats.NatsClient`.

    captures :meth:`publish_reply` invocations so handler tests can
    assert which response landed on which reply subject. mirrors only
    the surface ToolServer's handlers actually touch.
    """

    def __init__(self) -> None:
        self.replies: list[tuple[str, BaseModel]] = []

    async def publish_reply(self, *, reply_subject: str, message: BaseModel) -> None:
        """record the reply publish call."""
        self.replies.append((reply_subject, message))

    @property
    def last_reply(self) -> tuple[str, BaseModel]:
        """return most recent ``(reply_subject, message)`` recorded."""
        return self.replies[-1]


def _attach_recording_nc(server: ToolServer) -> _RecordingNatsClient:
    """install a recording NATS stub on ``server._nc`` and return it.

    handlers under test call ``self._respond(msg, response)`` which
    requires ``self._nc.publish_reply`` to exist; tests that drive
    handlers directly without going through :meth:`serve` use this
    helper to satisfy that wiring without standing up a real connection.

    :param server: tool server under test
    :ptype server: ToolServer
    :return: the recording stub bound on ``server._nc``
    :rtype: _RecordingNatsClient
    """
    rec = _RecordingNatsClient()
    # setattr bypasses ruff SLF001; the unit test legitimately needs to
    # install a recording stub on the server's NATS slot without going
    # through :meth:`serve` (which would dial a real connection).
    setattr(server, "_nc", rec)
    return rec


# -- registration tests --


class TestToolServerRegister:
    """tests for ToolServer.register method."""

    def test_register_adds_tool(self) -> None:
        """register stores tool keyed by name@version."""
        server = ToolServer(nats_url="nats://localhost:4222", namespace_collection=None)
        tool = StubTool(name="test.stub", version="1.0")
        server.register(tool)
        assert "test.stub@1.0" in server.tool_names
        assert server.tools_count == 1

    def test_register_multiple_tools(self) -> None:
        """register stores multiple distinct tools."""
        server = ToolServer(nats_url="nats://localhost:4222", namespace_collection=None)
        tool_a = StubTool(name="test.alpha", version="1.0")
        tool_b = StubTool(name="test.beta", version="2.0")
        server.register(tool_a)
        server.register(tool_b)
        assert server.tools_count == 2
        assert "test.alpha@1.0" in server.tool_names
        assert "test.beta@2.0" in server.tool_names

    def test_register_same_key_overwrites(self) -> None:
        """registering tool with same name@version overwrites previous.

        the ``name@version`` key stays at one registration (the later
        ``register`` call wins) so ``tools_count`` remains 1 and
        ``tool_names`` carries exactly one entry. dispatch semantics
        (``tool_b`` replaces ``tool_a``) are covered in
        :class:`TestToolServerHandleCall`.
        """
        server = ToolServer(nats_url="nats://localhost:4222", namespace_collection=None)
        tool_a = StubTool(name="test.stub", version="1.0")
        tool_b = StubTool(name="test.stub", version="1.0")
        server.register(tool_a)
        server.register(tool_b)
        assert server.tools_count == 1
        assert server.tool_names == ("test.stub@1.0",)


# -- serve tests --


class TestToolServerServe:
    """tests for ToolServer.serve method (mocked NATS)."""

    @pytest.mark.asyncio
    async def test_serve_connects_to_nats(self) -> None:
        """serve connects to NATS server at configured URL."""
        server = ToolServer(
            nats_url="nats://localhost:9999",
            pod_id="test-pod-1",
            namespace_collection=None,
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
            namespace_collection=None,
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

        # find the registration publish call (kw-only ``subject=Subject(...)``)
        publish_calls = mock_nc.publish.call_args_list
        registration_call = None
        for call in publish_calls:
            subj_arg = call.kwargs.get("subject")
            subject_path = subj_arg.path if subj_arg is not None else ""
            if "tools.register" in subject_path:
                registration_call = call
                break

        assert registration_call is not None
        subject_arg = registration_call.kwargs["subject"]
        assert subject_arg.path == "testns.tools.register"
        message = registration_call.kwargs["message"]
        payload = json.loads(message.model_dump_json())
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
            namespace_collection=None,
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
        subject_paths = [c.kwargs["subject"].path for c in subscribe_calls]
        assert "testns.tools.internal.test-pod-3" in subject_paths


# -- handle_call tests --


class TestToolServerHandleCall:
    """tests for ToolServer.handle_call method."""

    @pytest.mark.asyncio
    async def test_handle_call_routes_to_correct_tool(self) -> None:
        """handle_call dispatches to tool matching name@version.

        correlation_id rides on ``context.correlation_id`` (UUID) per
        context-task-01; the server echoes the whole
        :class:`CallContext` back on :class:`CallResponse.context` so
        the response shape matches the request.
        """
        server = ToolServer(nats_url="nats://localhost:9999", namespace_collection=None)
        tool = StubTool(name="test.stub", version="1.0")
        server.register(tool)

        correlation_id = uuid4()
        msg = _make_nats_msg(
            {
                "tool_name": "test.stub",
                "tool_version": "1.0",
                "arguments": {"key": "value"},
                "context": {"correlation_id": str(correlation_id)},
            }
        )
        rec = _attach_recording_nc(server)

        await server.handle_call(msg)

        assert len(rec.replies) == 1
        reply_subject, response = rec.last_reply
        assert reply_subject == "_INBOX.test"
        response_data = json.loads(response.model_dump_json())
        assert response_data["success"] is True
        assert "correlation_id" not in response_data
        assert response_data["context"]["correlation_id"] == str(correlation_id)
        content = json.loads(response_data["content"])
        assert content == {"key": "value"}

    @pytest.mark.asyncio
    async def test_handle_call_returns_tool_result_on_success(self) -> None:
        """handle_call returns serialized ToolResult with success=True."""
        server = ToolServer(nats_url="nats://localhost:9999", namespace_collection=None)
        tool = StubTool(name="test.stub", version="1.0")
        server.register(tool)

        msg = _make_nats_msg(
            {
                "tool_name": "test.stub",
                "tool_version": "1.0",
                "arguments": {},
                "context": {"correlation_id": str(uuid4())},
            }
        )
        rec = _attach_recording_nc(server)

        await server.handle_call(msg)

        response_data = json.loads(rec.last_reply[1].model_dump_json())
        assert response_data["success"] is True
        assert response_data["error"] is None

    @pytest.mark.asyncio
    async def test_handle_call_returns_error_on_unknown_tool(self) -> None:
        """handle_call returns error response for unregistered tool."""
        server = ToolServer(nats_url="nats://localhost:9999", namespace_collection=None)

        correlation_id = uuid4()
        msg = _make_nats_msg(
            {
                "tool_name": "nonexistent.tool",
                "tool_version": "1.0",
                "arguments": {},
                "context": {"correlation_id": str(correlation_id)},
            }
        )
        rec = _attach_recording_nc(server)

        await server.handle_call(msg)

        response_data = json.loads(rec.last_reply[1].model_dump_json())
        assert response_data["success"] is False
        assert "nonexistent.tool@1.0" in response_data["error"]
        assert response_data["context"]["correlation_id"] == str(correlation_id)

    @pytest.mark.asyncio
    async def test_handle_call_returns_error_on_execution_failure(self) -> None:
        """handle_call returns error response when tool raises exception."""
        server = ToolServer(nats_url="nats://localhost:9999", namespace_collection=None)
        tool = FailingTool()
        server.register(tool)

        correlation_id = uuid4()
        msg = _make_nats_msg(
            {
                "tool_name": "test.failing",
                "tool_version": "1.0",
                "arguments": {},
                "context": {"correlation_id": str(correlation_id)},
            }
        )
        rec = _attach_recording_nc(server)

        await server.handle_call(msg)

        response_data = json.loads(rec.last_reply[1].model_dump_json())
        assert response_data["success"] is False
        assert "intentional failure" in response_data["error"]
        assert response_data["context"]["correlation_id"] == str(correlation_id)

    @pytest.mark.asyncio
    async def test_handle_call_returns_error_on_malformed_request(self) -> None:
        """handle_call returns error response for invalid JSON payload."""
        server = ToolServer(nats_url="nats://localhost:9999", namespace_collection=None)

        msg = IncomingMessage(
            data=b"not valid json",
            reply_subject="_INBOX.test",
            subject="3tears.tools.internal.test-pod",
        )
        rec = _attach_recording_nc(server)

        await server.handle_call(msg)

        assert len(rec.replies) == 1
        response_data = json.loads(rec.last_reply[1].model_dump_json())
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
            namespace_collection=None,
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

        heartbeat_calls = [c for c in mock_nc.publish.call_args_list if "heartbeat" in c.kwargs.get("subject").path]
        assert len(heartbeat_calls) >= 1
        subject_arg = heartbeat_calls[0].kwargs["subject"]
        assert subject_arg.path == "testns.tools.heartbeat.hb-pod"
        message = heartbeat_calls[0].kwargs["message"]
        payload = json.loads(message.model_dump_json())
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
            namespace_collection=None,
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
            assert server.is_running is True
            await server.shutdown()
            await asyncio.sleep(0.05)
            serve_task.cancel()
            try:
                await serve_task
            except asyncio.CancelledError:
                pass

        assert server.is_running is False
        # the wrapper exposes a single ``shutdown()`` that internally
        # drains + closes the underlying nats-py connection.
        mock_nc.shutdown.assert_called_once()


# -- wire format model tests --


class TestWireFormatModels:
    """tests for Pydantic wire format models."""

    def test_call_request_serialization(self) -> None:
        """CallRequest serializes to JSON with required fields.

        correlation_id is no longer a flat field on :class:`CallRequest`;
        it rides on :class:`CallContext.correlation_id` since
        context-task-01.
        """
        from threetears.agent.tools.context_envelope import CallContext

        correlation_id = uuid4()
        req = CallRequest(
            tool_name="test.stub",
            tool_version="1.0",
            arguments={"key": "value"},
            context=CallContext(correlation_id=correlation_id),
        )
        data = json.loads(req.model_dump_json())
        assert data["tool_name"] == "test.stub"
        assert data["tool_version"] == "1.0"
        assert data["arguments"] == {"key": "value"}
        assert "correlation_id" not in data
        assert data["context"]["correlation_id"] == str(correlation_id)

    def test_call_response_serialization(self) -> None:
        """CallResponse serializes to JSON with all fields.

        correlation_id is no longer a flat field on :class:`CallResponse`;
        it rides on :class:`CallContext.correlation_id` since
        context-task-01 unified request + response identity on a single
        :class:`CallContext` shape.
        """
        from threetears.agent.tools.context_envelope import CallContext

        correlation_id = uuid4()
        resp = CallResponse(
            success=True,
            content="result text",
            metadata=None,
            error=None,
            context=CallContext(correlation_id=correlation_id),
        )
        data = json.loads(resp.model_dump_json())
        assert data["success"] is True
        assert data["content"] == "result text"
        assert data["metadata"] is None
        assert data["error"] is None
        assert "correlation_id" not in data
        assert data["context"]["correlation_id"] == str(correlation_id)

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
            namespace_collection=None,
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
        subject_paths = [c.kwargs["subject"].path for c in subscribe_calls]
        assert "testns.tools.probe.probe-pod-1" in subject_paths

    @pytest.mark.asyncio
    async def test_serve_subscribes_before_publishing_registration(self) -> None:
        """probe and call subscriptions must be active before registration publish."""
        server = ToolServer(
            nats_url="nats://localhost:9999",
            namespace="testns",
            pod_id="order-pod",
            namespace_collection=None,
        )
        tool = StubTool()
        server.register(tool)

        order: list[str] = []

        async def record_subscribe(*args: Any, **kwargs: Any) -> AsyncMock:
            """capture subscribe call order (kw-only ``subject=Subject(...)``)."""
            subject_arg = kwargs.get("subject")
            path = subject_arg.path if subject_arg is not None else ""
            order.append(f"subscribe:{path}")
            return AsyncMock()

        async def record_publish(*args: Any, **kwargs: Any) -> None:
            """capture publish call order (kw-only ``subject=Subject(...)``)."""
            subject_arg = kwargs.get("subject")
            path = subject_arg.path if subject_arg is not None else ""
            order.append(f"publish:{path}")

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
        assert any(evt == "subscribe:testns.tools.probe.order-pod" for evt in order[:first_register_idx]), (
            "probe subscription must be established before registration publish"
        )

    @pytest.mark.asyncio
    async def test_handle_probe_responds_with_ack(self) -> None:
        """handle_probe replies with ProbeAck carrying pod_id and ready=True."""
        server = ToolServer(
            nats_url="nats://localhost:9999",
            pod_id="ack-pod",
            namespace_collection=None,
        )
        rec = _attach_recording_nc(server)

        msg = IncomingMessage(
            data=b'{"pod_id": "ack-pod"}',
            reply_subject="_INBOX.probe",
            subject="3tears.tools.probe.ack-pod",
        )

        await server.handle_probe(msg)

        assert len(rec.replies) == 1
        reply_subject, response = rec.last_reply
        assert reply_subject == "_INBOX.probe"
        payload = json.loads(response.model_dump_json())
        assert payload["pod_id"] == "ack-pod"
        assert payload["ready"] is True

    @pytest.mark.asyncio
    async def test_handle_probe_does_not_mutate_server_state(self) -> None:
        """handle_probe is a pure responder -- readiness is driven by discovery."""
        server = ToolServer(
            nats_url="nats://localhost:9999",
            pod_id="ready-pod",
            namespace_collection=None,
        )
        rec = _attach_recording_nc(server)

        msg = IncomingMessage(
            data=b'{"pod_id": "ready-pod"}',
            reply_subject="_INBOX.probe",
            subject="3tears.tools.probe.ready-pod",
        )

        await server.handle_probe(msg)

        # probe handler only responds; it must not flip the serve()-caller
        # ready signal. readiness from the caller's perspective is
        # established by serve() finishing subscription (observable via
        # ``wait_ready`` / ``is_running``) and by polling the registry's
        # discovery subject (see wait_until_ready). we assert here that
        # a bare handle_probe call (no serve() yet) has not caused
        # wait_ready to unblock -- the quickest observable check is that
        # ``wait_ready`` still times out.
        assert len(rec.replies) == 1
        assert server.is_running is False
        with pytest.raises(asyncio.TimeoutError):
            await server.wait_ready(timeout=0.01)

    @pytest.mark.asyncio
    async def test_wait_until_ready_unblocks_when_discovery_reports_available(self) -> None:
        """wait_until_ready returns True once discovery reports every tool available."""
        from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool

        class _FakeTool(TearsTool):
            """minimal TearsTool fake for readiness polling tests."""

            def mcp_name(self) -> str:
                """return name."""
                return "test.probe"

            def mcp_version(self) -> str:
                """return version."""
                return "1.0.0"

            def mcp_schema(self) -> MCPToolDefinition:
                """return a trivial manifest entry."""
                return MCPToolDefinition(
                    name="test.probe",
                    version="1.0.0",
                    description="probe fake",
                    input_schema={"type": "object", "properties": {}},
                )

            async def execute(self, **_kwargs: Any) -> dict[str, Any]:
                """no-op execution path."""
                return {"ok": True}

        from threetears.agent.tools.server import (
            DiscoveryProbeResponse,
            DiscoveryProbeResultEntry,
        )

        discovery_response = DiscoveryProbeResponse(
            agent_id="wait-pod",
            tools=[
                DiscoveryProbeResultEntry(name="test.probe", version="1.0.0", status="available"),
            ],
        )

        from threetears.nats import set_default_namespace

        set_default_namespace("testns")
        nc = MagicMock()
        nc.request = AsyncMock(return_value=discovery_response)
        server = ToolServer(
            nats_client=nc,
            pod_id="wait-pod",
            namespace_collection=None,
        )
        server.register(_FakeTool())

        ready = await server.wait_until_ready(timeout=1.0)
        assert ready is True
        nc.request.assert_called()

    @pytest.mark.asyncio
    async def test_wait_until_ready_returns_false_on_timeout(self) -> None:
        """wait_until_ready returns False when discovery never reports available."""
        from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool

        class _FakeTool(TearsTool):
            """minimal TearsTool fake that stays unavailable."""

            def mcp_name(self) -> str:
                """return name."""
                return "test.slow"

            def mcp_version(self) -> str:
                """return version."""
                return "1.0.0"

            def mcp_schema(self) -> MCPToolDefinition:
                """return a trivial manifest entry."""
                return MCPToolDefinition(
                    name="test.slow",
                    version="1.0.0",
                    description="slow fake",
                    input_schema={"type": "object", "properties": {}},
                )

            async def execute(self, **_kwargs: Any) -> dict[str, Any]:
                """no-op execution path."""
                return {"ok": True}

        from threetears.agent.tools.server import (
            DiscoveryProbeResponse,
            DiscoveryProbeResultEntry,
        )

        discovery_response = DiscoveryProbeResponse(
            agent_id="slow-pod",
            tools=[
                DiscoveryProbeResultEntry(name="test.slow", version="1.0.0", status="unavailable"),
            ],
        )

        nc = MagicMock()
        nc.request = AsyncMock(return_value=discovery_response)
        server = ToolServer(
            nats_client=nc,
            pod_id="slow-pod",
            namespace_collection=None,
        )
        server.register(_FakeTool())

        ready = await server.wait_until_ready(timeout=0.05)
        assert ready is False


# -- public-surface promotions for hub cross-class collaboration --


class TestToolsCountProperty:
    """``tools_count`` exposes ``len(self._tools)`` as a public read."""

    def test_tools_count_zero_on_fresh_server(self) -> None:
        server = ToolServer(nats_url="nats://localhost:4222", namespace_collection=None)
        assert server.tools_count == 0

    def test_tools_count_reflects_registrations(self) -> None:
        server = ToolServer(nats_url="nats://localhost:4222", namespace_collection=None)
        server.register(StubTool(name="a", version="1.0"))
        server.register(StubTool(name="b", version="1.0"))
        assert server.tools_count == 2

    def test_tools_count_decreases_on_unregister(self) -> None:
        server = ToolServer(nats_url="nats://localhost:4222", namespace_collection=None)
        server.register(StubTool(name="a", version="1.0"))
        server.register(StubTool(name="b", version="1.0"))
        server.unregister("a")
        assert server.tools_count == 1


class TestToolNamesProperty:
    """``tool_names`` returns an immutable snapshot of registration keys."""

    def test_tool_names_empty_on_fresh_server(self) -> None:
        server = ToolServer(nats_url="nats://localhost:4222", namespace_collection=None)
        assert server.tool_names == ()

    def test_tool_names_contains_registered_keys(self) -> None:
        server = ToolServer(nats_url="nats://localhost:4222", namespace_collection=None)
        server.register(StubTool(name="a", version="1.0"))
        server.register(StubTool(name="b", version="2.0"))
        names = server.tool_names
        assert set(names) == {"a@1.0", "b@2.0"}

    def test_tool_names_returns_tuple_not_dict_keys(self) -> None:
        """snapshot is a tuple so callers cannot mutate server state."""
        server = ToolServer(nats_url="nats://localhost:4222", namespace_collection=None)
        server.register(StubTool(name="a", version="1.0"))
        names = server.tool_names
        assert isinstance(names, tuple)
        # tuples have no ``add``/``pop``; this assertion pins the contract
        # against a regression that returned ``self._tools.keys()``.
        assert not hasattr(names, "pop")

    def test_tool_names_snapshot_is_stable_across_mutations(self) -> None:
        """snapshot reflects state at call time; later mutations do not echo."""
        server = ToolServer(nats_url="nats://localhost:4222", namespace_collection=None)
        server.register(StubTool(name="a", version="1.0"))
        snapshot = server.tool_names
        server.register(StubTool(name="b", version="1.0"))
        assert snapshot == ("a@1.0",)


class TestIsConnectedProperty:
    """``is_connected`` reflects whether NATS client is live."""

    def test_is_connected_false_before_serve(self) -> None:
        server = ToolServer(nats_url="nats://localhost:4222", namespace_collection=None)
        assert server.is_connected is False

    def test_is_connected_true_when_nc_set(self) -> None:
        server = ToolServer(nats_client=MagicMock(), namespace_collection=None)
        assert server.is_connected is True


class TestPublishRegistrationPublicMethod:
    """``publish_registration`` is the public name for the manifest publish."""

    @pytest.mark.asyncio
    async def test_publish_registration_raises_when_not_connected(self) -> None:
        server = ToolServer(nats_url="nats://localhost:4222", namespace_collection=None)
        with pytest.raises(RuntimeError, match="publish_registration"):
            await server.publish_registration()

    @pytest.mark.asyncio
    async def test_publish_registration_publishes_on_configured_subject(self) -> None:
        from threetears.nats import set_default_namespace

        set_default_namespace("testns")
        nc = AsyncMock()
        server = ToolServer(
            nats_client=nc,
            namespace="testns",
            pod_id="pod-7",
            namespace_collection=None,
        )
        server.register(StubTool(name="alpha", version="1.0"))
        await server.publish_registration()
        nc.publish.assert_awaited_once()
        subject_arg = nc.publish.await_args.kwargs["subject"]
        message = nc.publish.await_args.kwargs["message"]
        assert subject_arg.path == "testns.tools.register"
        parsed = json.loads(message.model_dump_json())
        assert parsed["pod_id"] == "pod-7"
        assert parsed["tools"][0]["name"] == "alpha"


class TestRegisterToolDeregisterTool:
    """``register_tool`` / ``deregister_tool`` are the dynamic-lifecycle
    public methods: mutate the registration set AND publish the updated
    manifest when NATS is connected.
    """

    @pytest.mark.asyncio
    async def test_register_tool_adds_and_publishes(self) -> None:
        nc = AsyncMock()
        server = ToolServer(nats_client=nc, namespace_collection=None)
        await server.register_tool(StubTool(name="x", version="1.0"))
        assert server.tools_count == 1
        nc.publish.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_register_tool_skips_publish_when_not_connected(self) -> None:
        # no nats_client injected: pre-serve() registration path
        server = ToolServer(nats_url="nats://localhost:4222", namespace_collection=None)
        await server.register_tool(StubTool(name="x", version="1.0"))
        assert server.tools_count == 1  # tool still registered

    @pytest.mark.asyncio
    async def test_deregister_tool_removes_and_publishes(self) -> None:
        nc = AsyncMock()
        server = ToolServer(nats_client=nc, namespace_collection=None)
        server.register(StubTool(name="x", version="1.0"))
        removed = await server.deregister_tool("x")
        assert removed is True
        assert server.tools_count == 0
        nc.publish.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_deregister_tool_returns_false_when_missing(self) -> None:
        nc = AsyncMock()
        server = ToolServer(nats_client=nc, namespace_collection=None)
        removed = await server.deregister_tool("never-registered")
        assert removed is False
        nc.publish.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_deregister_tool_no_publish_when_not_connected(self) -> None:
        server = ToolServer(nats_url="nats://localhost:4222", namespace_collection=None)
        server.register(StubTool(name="x", version="1.0"))
        removed = await server.deregister_tool("x")
        assert removed is True
        assert server.tools_count == 0


class TestToolServerInjectedNatsClient:
    """``nats_client`` ctor parameter supports shared-connection callers
    (agent bootstrap) without competing with the owner's lifecycle.
    """

    def test_constructor_without_either_raises(self) -> None:
        """omitting both ``nats_url`` and ``nats_client`` is a config error."""
        with pytest.raises(ValueError, match="nats_url or nats_client"):
            ToolServer(namespace_collection=None)

    def test_constructor_with_injected_client_records_non_ownership(
        self,
    ) -> None:
        """supplying ``nats_client`` flips the ownership flag.

        the injected client is already live from the caller's
        perspective, so ``is_connected`` is ``True`` at construction
        time and ``owns_nats_connection`` is ``False`` (shutdown will
        not close it).
        """
        nc = AsyncMock()
        server = ToolServer(nats_client=nc, namespace_collection=None)
        assert server.is_connected is True
        assert server.owns_nats_connection is False

    def test_constructor_with_url_records_ownership(self) -> None:
        """supplying only ``nats_url`` means the server owns the connection.

        ``is_connected`` stays ``False`` until :meth:`serve` opens the
        connection; ``owns_nats_connection`` is ``True`` so shutdown
        will drain + close it.
        """
        server = ToolServer(nats_url="nats://localhost:4222", namespace_collection=None)
        assert server.is_connected is False
        assert server.owns_nats_connection is True

    @pytest.mark.asyncio
    async def test_serve_skips_connect_when_client_injected(self) -> None:
        """``serve()`` reuses the injected client rather than opening a new one.

        identity of the injected client is observed behaviorally: the
        mock's ``subscribe`` + ``publish`` receive the calls that serve
        would otherwise dispatch to a freshly-opened client, and
        ``nats_connect`` is never awaited.
        """
        nc = AsyncMock()
        server = ToolServer(
            nats_client=nc,
            heartbeat_interval=3600.0,
            namespace_collection=None,
        )
        with patch(
            "threetears.agent.tools.server.nats_connect",
            new=AsyncMock(),
        ) as connect_mock:
            serve_task = asyncio.create_task(server.serve())
            await asyncio.sleep(0)
            try:
                await server.wait_ready(timeout=1.0)
            finally:
                await server.shutdown()
                await asyncio.wait_for(serve_task, timeout=1.0)
        connect_mock.assert_not_awaited()
        assert server.is_connected is True
        assert server.owns_nats_connection is False
        nc.subscribe.assert_awaited()

    @pytest.mark.asyncio
    async def test_shutdown_does_not_close_injected_client(self) -> None:
        """caller-owned connection stays open after ``shutdown()``."""
        nc = AsyncMock()
        server = ToolServer(
            nats_client=nc,
            heartbeat_interval=3600.0,
            namespace_collection=None,
        )
        serve_task = asyncio.create_task(server.serve())
        await asyncio.sleep(0)
        await server.wait_ready(timeout=1.0)
        await server.shutdown()
        await asyncio.wait_for(serve_task, timeout=1.0)
        # the wrapper exposes ``shutdown()``, not the legacy
        # ``drain()`` + ``close()`` pair; injected clients must be
        # left open for the owning caller to dispose.
        nc.shutdown.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_shutdown_closes_self_owned_client(self) -> None:
        """server-owned connection is shut down (drain + close) on ``shutdown()``."""
        nc = AsyncMock()
        server = ToolServer(
            nats_url="nats://localhost:4222",
            heartbeat_interval=3600.0,
            namespace_collection=None,
        )
        with patch(
            "threetears.agent.tools.server.nats_connect",
            new=AsyncMock(return_value=nc),
        ):
            serve_task = asyncio.create_task(server.serve())
            await asyncio.sleep(0)
            await server.wait_ready(timeout=1.0)
            await server.shutdown()
            await asyncio.wait_for(serve_task, timeout=1.0)
        # the wrapper's ``shutdown()`` internally drains + closes the
        # underlying nats-py connection.
        nc.shutdown.assert_awaited_once()
