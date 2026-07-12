"""tests for ToolServer NATS-based tool serving."""

from __future__ import annotations

import asyncio
import json
import logging
import time
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
from threetears.core.security.identity_token import (
    IdentityClaims,
    build_jwks,
    generate_signing_keypair,
    sign_identity_token,
)
from threetears.nats import IncomingMessage, Subject

from unit.tools._pod_auth import StubReplayGuard as _PodReplayGuard
from unit.tools._pod_auth import jwks_provider as _pod_jwks_provider
from unit.tools._pod_auth import mint_user_assertion as _pod_mint_hub_token
from unit.tools._pod_auth import signed_call_payload as _signed_call_payload


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
        # serve() now always self-provisions a Hub-JWKS provider (enforce-only); give the mock a
        # JWKS reply so the best-effort initial fetch parses instead of choking on a bare mock.
        mock_nc.request_raw = AsyncMock(return_value=json.dumps({"keys": []}).encode("utf-8"))

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
        # serve() now always self-provisions a Hub-JWKS provider (enforce-only); give the mock a
        # JWKS reply so the best-effort initial fetch parses instead of choking on a bare mock.
        mock_nc.request_raw = AsyncMock(return_value=json.dumps({"keys": []}).encode("utf-8"))

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
    async def test_serve_self_provisions_jwks_provider(self) -> None:
        """with no injected provider, serve self-provisions a Hub-JWKS provider (enforce-only).

        observable proof (no private access): the self-provisioned provider fetches the Hub JWKS
        over this pod's own connection during serve.
        """
        server = ToolServer(
            nats_url="nats://localhost:9999",
            namespace="testns",
            pod_id="test-pod-jwks",
            namespace_collection=None,
        )
        server.register(StubTool(name="test.stub", version="1.0"))

        mock_nc = AsyncMock()
        mock_nc.is_connected = True
        mock_nc.subscribe = AsyncMock()
        mock_nc.publish = AsyncMock()
        mock_nc.drain = AsyncMock()
        mock_nc.close = AsyncMock()
        # serve() now always self-provisions a Hub-JWKS provider (enforce-only); give the mock a
        # JWKS reply so the best-effort initial fetch parses instead of choking on a bare mock.
        mock_nc.request_raw = AsyncMock(return_value=json.dumps({"keys": []}).encode("utf-8"))

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

        fetched_jwks = any(
            getattr(call.kwargs.get("subject"), "path", "").endswith(".hub.jwks")
            for call in mock_nc.request_raw.await_args_list
        )
        assert fetched_jwks, "serve must self-provision a provider that fetches the Hub JWKS"

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
        # serve() now always self-provisions a Hub-JWKS provider (enforce-only); give the mock a
        # JWKS reply so the best-effort initial fetch parses instead of choking on a bare mock.
        mock_nc.request_raw = AsyncMock(return_value=json.dumps({"keys": []}).encode("utf-8"))

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
        server = ToolServer(
            nats_url="nats://localhost:9999",
            pod_id="test-pod",
            namespace_collection=None,
            jwks_provider=_pod_jwks_provider,
            assertion_replay_guard=_PodReplayGuard(),
        )
        tool = StubTool(name="test.stub", version="1.0")
        server.register(tool)

        correlation_id = uuid4()
        msg = _make_nats_msg(
            _signed_call_payload(
                pod_id="test-pod",
                tool_name="test.stub",
                tool_version="1.0",
                arguments={"key": "value"},
                correlation_id=str(correlation_id),
            )
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
        server = ToolServer(
            nats_url="nats://localhost:9999",
            pod_id="test-pod",
            namespace_collection=None,
            jwks_provider=_pod_jwks_provider,
            assertion_replay_guard=_PodReplayGuard(),
        )
        tool = StubTool(name="test.stub", version="1.0")
        server.register(tool)

        msg = _make_nats_msg(_signed_call_payload(pod_id="test-pod", tool_name="test.stub", tool_version="1.0"))
        rec = _attach_recording_nc(server)

        await server.handle_call(msg)

        response_data = json.loads(rec.last_reply[1].model_dump_json())
        assert response_data["success"] is True
        assert response_data["error"] is None

    @pytest.mark.asyncio
    async def test_handle_call_returns_error_on_unknown_tool(self) -> None:
        """handle_call returns error response for unregistered tool."""
        server = ToolServer(
            nats_url="nats://localhost:9999",
            pod_id="test-pod",
            namespace_collection=None,
            jwks_provider=_pod_jwks_provider,
            assertion_replay_guard=_PodReplayGuard(),
        )

        correlation_id = uuid4()
        msg = _make_nats_msg(
            _signed_call_payload(
                pod_id="test-pod",
                tool_name="nonexistent.tool",
                tool_version="1.0",
                correlation_id=str(correlation_id),
            )
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
        server = ToolServer(
            nats_url="nats://localhost:9999",
            pod_id="test-pod",
            namespace_collection=None,
            jwks_provider=_pod_jwks_provider,
            assertion_replay_guard=_PodReplayGuard(),
        )
        tool = FailingTool()
        server.register(tool)

        correlation_id = uuid4()
        msg = _make_nats_msg(
            _signed_call_payload(
                pod_id="test-pod",
                tool_name="test.failing",
                tool_version="1.0",
                correlation_id=str(correlation_id),
            )
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
        # Healthy NATS: the heartbeat loop's liveness supervisor os._exit(1)s the
        # PROCESS after a sustained-unhealthy streak, and ToolServer.is_healthy is
        # `not is_closed and is_healthy`. A bare AsyncMock leaves both truthy ->
        # is_closed truthy -> unhealthy -> the supervisor kills the test process
        # (exit 1, no traceback). This test runs the loop long enough to hit the
        # threshold, so pin the client healthy (the publish path it exercises).
        mock_nc.is_closed = False
        mock_nc.is_healthy = True
        mock_nc.subscribe = AsyncMock()
        mock_nc.publish = AsyncMock()
        mock_nc.drain = AsyncMock()
        mock_nc.close = AsyncMock()
        # serve() now always self-provisions a Hub-JWKS provider (enforce-only); give the mock a
        # JWKS reply so the best-effort initial fetch parses instead of choking on a bare mock.
        mock_nc.request_raw = AsyncMock(return_value=json.dumps({"keys": []}).encode("utf-8"))

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
        # healthy connection so the heartbeat-loop liveness supervisor does not trip its os._exit
        # crash-recycle during this short-interval test (bare AsyncMock leaves is_closed truthy).
        mock_nc.is_closed = False
        mock_nc.is_healthy = True
        mock_nc.subscribe = AsyncMock()
        mock_nc.publish = AsyncMock()
        mock_nc.drain = AsyncMock()
        mock_nc.close = AsyncMock()
        # serve() now always self-provisions a Hub-JWKS provider (enforce-only); give the mock a
        # JWKS reply so the best-effort initial fetch parses instead of choking on a bare mock.
        mock_nc.request_raw = AsyncMock(return_value=json.dumps({"keys": []}).encode("utf-8"))

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
        # serve() now always self-provisions a Hub-JWKS provider (enforce-only); give the mock a
        # JWKS reply so the best-effort initial fetch parses instead of choking on a bare mock.
        mock_nc.request_raw = AsyncMock(return_value=json.dumps({"keys": []}).encode("utf-8"))

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
        # serve() now always self-provisions a Hub-JWKS provider (enforce-only); give the mock a
        # JWKS reply so the best-effort initial fetch parses instead of choking on a bare mock.
        mock_nc.request_raw = AsyncMock(return_value=json.dumps({"keys": []}).encode("utf-8"))

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

            async def execute(self, **_kwargs: Any) -> ToolResult:
                """no-op execution path."""
                return ToolResult(success=True, content="")

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

            async def execute(self, **_kwargs: Any) -> ToolResult:
                """no-op execution path."""
                return ToolResult(success=True, content="")

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
        # serve() self-provisions a Hub-JWKS provider over the injected client (enforce-only); feed
        # the mock a JWKS reply so the best-effort initial fetch parses.
        nc.request_raw = AsyncMock(return_value=json.dumps({"keys": []}).encode("utf-8"))
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
        # serve() self-provisions a Hub-JWKS provider over the injected client (enforce-only); feed
        # the mock a JWKS reply so the best-effort initial fetch parses.
        nc.request_raw = AsyncMock(return_value=json.dumps({"keys": []}).encode("utf-8"))
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
        # serve() self-provisions a Hub-JWKS provider over the opened client (enforce-only); feed
        # the mock a JWKS reply so the best-effort initial fetch parses.
        nc.request_raw = AsyncMock(return_value=json.dumps({"keys": []}).encode("utf-8"))
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


class TestToolServerIdentityVerification:
    """v0.13.9 enforce-only: the pod re-verifies the Hub identity token (defense in depth).

    closes the direct-internal-subject bypass -- a publisher straight to the pod without a valid
    Hub-signed token (or one that forged a different identity onto a captured token) is rejected.
    verification is UNCONDITIONAL; there is no off/warn path. the pod ALSO verifies the proxy's
    body-bound assertion on every call (its own class below), so the cases that assert the tool
    RAN carry a valid assertion too; the identity-gate REJECTION cases short-circuit at the
    identity gate (the first gate) and need none.
    """

    @staticmethod
    def _hub_and_proxy() -> tuple[Any, Any, dict[str, Any]]:
        """a Hub identity keypair + a proxy-assertion signer + the COMBINED JWKS the pod verifies
        BOTH the identity token and the proxy assertion against (one ``jwks_provider`` feeds both
        gates, so the document carries both public keys under distinct kids)."""
        import base64

        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from pydantic import SecretStr

        from threetears.core.security import ProxyAssertionSigner

        priv, pub = generate_signing_keypair()
        seed = base64.urlsafe_b64encode(Ed25519PrivateKey.generate().private_bytes_raw()).decode("ascii")
        signer = ProxyAssertionSigner.from_secret(SecretStr(seed))
        combined = {"keys": [*build_jwks({"kid-1": pub})["keys"], *signer.public_jwks()["keys"]]}
        return priv, signer, combined

    @staticmethod
    def _token(priv: Any, *, sub: Any, customer_id: Any) -> str:
        now = int(time.time())
        claims = IdentityClaims(
            sub=str(sub),
            customer_id=str(customer_id),
            sid="sid-1",
            pod_id="pod-1",
            iss="hub",
            iat=now,
            exp=now + 600,
        )
        return sign_identity_token(claims, signing_key=priv, kid="kid-1")

    @staticmethod
    def _server(*, jwks_provider: Any) -> tuple[ToolServer, Any]:
        server = ToolServer(
            nats_url="nats://localhost:9999",
            namespace_collection=None,
            pod_id="test-pod",
            jwks_provider=jwks_provider,
            assertion_replay_guard=_PodReplayGuard(),
        )
        server.register(StubTool(name="test.stub", version="1.0"))
        return server, _attach_recording_nc(server)

    @staticmethod
    def _assertion(signer: Any, *, body_hash: str) -> str:
        return signer.mint(
            pod_id="test-pod",
            agent_id=str(uuid4()),
            customer_id=str(uuid4()),
            body_hash=body_hash,
            nonce=str(uuid4()),
            now=int(time.time()),
        )

    @staticmethod
    def _msg(
        *,
        agent_id: Any = None,
        customer_id: Any = None,
        token: str | None = None,
        assertion: str | None = None,
        correlation_id: str | None = None,
    ) -> IncomingMessage:
        context: dict[str, Any] = {"correlation_id": correlation_id or str(uuid4())}
        if agent_id is not None:
            context["agent_id"] = str(agent_id)
        if customer_id is not None:
            context["customer_id"] = str(customer_id)
        if token is not None:
            context["identity_token"] = token
        envelope: dict[str, Any] = {
            "tool_name": "test.stub",
            "tool_version": "1.0",
            "arguments": {"key": "value"},
            "context": context,
        }
        if assertion is not None:
            envelope["proxy_assertion"] = assertion
        return _make_nats_msg(envelope)

    @classmethod
    def _signed_msg(
        cls,
        priv: Any,
        signer: Any,
        *,
        envelope_agent_id: Any,
        envelope_customer_id: Any,
        token_sub: Any,
        token_customer: Any,
    ) -> IncomingMessage:
        """a full happy-path message: a valid identity token + a valid proxy assertion bound to the
        exact call body, with the ENVELOPE free to claim a different (or absent) identity so the
        re-stamp behaviour is observable through the public dispatch surface."""
        from threetears.core.security import canonical_call_hash

        correlation_id = str(uuid4())
        token = cls._token(priv, sub=token_sub, customer_id=token_customer)
        body_hash = canonical_call_hash("test.stub", {"key": "value"}, correlation_id)
        assertion = cls._assertion(signer, body_hash=body_hash)
        return cls._msg(
            agent_id=envelope_agent_id,
            customer_id=envelope_customer_id,
            token=token,
            assertion=assertion,
            correlation_id=correlation_id,
        )

    @staticmethod
    def _response(rec: Any) -> dict[str, Any]:
        result: dict[str, Any] = json.loads(rec.last_reply[1].model_dump_json())
        return result

    @pytest.mark.asyncio
    async def test_enforce_rejects_missing_token(self) -> None:
        _priv, _signer, jwks = self._hub_and_proxy()
        server, rec = self._server(jwks_provider=lambda: jwks)
        await server.handle_call(self._msg(agent_id=uuid4(), token=None))
        response = self._response(rec)
        assert response["success"] is False
        assert "identity verification failed" in response["error"]

    @pytest.mark.asyncio
    async def test_enforce_accepts_valid_matching_token(self) -> None:
        priv, signer, jwks = self._hub_and_proxy()
        agent, cust = uuid4(), uuid4()
        server, rec = self._server(jwks_provider=lambda: jwks)
        await server.handle_call(
            self._signed_msg(
                priv,
                signer,
                envelope_agent_id=agent,
                envelope_customer_id=cust,
                token_sub=agent,
                token_customer=cust,
            )
        )
        response = self._response(rec)
        assert response["success"] is True  # verified -> the tool ran
        # the dispatched call carries the verified identity (here identical to the envelope's claim).
        assert response["context"]["agent_id"] == str(agent)
        assert response["context"]["customer_id"] == str(cust)

    @pytest.mark.asyncio
    async def test_enforce_restamps_a_forged_identity_to_the_token(self) -> None:
        # a captured token re-pointed at a forged agent/customer runs under the TOKEN's true
        # identity -- the forged envelope claim is discarded, not honoured (and not merely rejected).
        priv, signer, jwks = self._hub_and_proxy()
        true_agent, true_cust = uuid4(), uuid4()  # token minted for agent A
        server, rec = self._server(jwks_provider=lambda: jwks)
        await server.handle_call(  # envelope forges agent/customer B
            self._signed_msg(
                priv,
                signer,
                envelope_agent_id=uuid4(),
                envelope_customer_id=uuid4(),
                token_sub=true_agent,
                token_customer=true_cust,
            )
        )
        response = self._response(rec)
        assert response["success"] is True  # verified -> the tool ran under the re-stamped identity
        assert response["context"]["agent_id"] == str(true_agent)  # B was overwritten with A
        assert response["context"]["customer_id"] == str(true_cust)

    @pytest.mark.asyncio
    async def test_enforce_restamps_when_envelope_identity_is_absent(self) -> None:
        # the bypass guard: a valid token presented with NO claimed agent/customer (stripped to
        # skip any comparison) must NOT dispatch under a null identity -- it is re-stamped from the
        # verified token, so the tool runs under the authenticated identity, never an unbound one.
        priv, signer, jwks = self._hub_and_proxy()
        true_agent, true_cust = uuid4(), uuid4()
        server, rec = self._server(jwks_provider=lambda: jwks)
        await server.handle_call(
            self._signed_msg(
                priv,
                signer,
                envelope_agent_id=None,
                envelope_customer_id=None,
                token_sub=true_agent,
                token_customer=true_cust,
            )
        )
        response = self._response(rec)
        assert response["success"] is True
        assert response["context"]["agent_id"] == str(true_agent)  # null -> verified identity
        assert response["context"]["customer_id"] == str(true_cust)

    @pytest.mark.asyncio
    async def test_enforce_rejects_when_jwks_provider_raises(self) -> None:
        # a flaky provider must become a fail-closed rejection at the identity gate, never an
        # escaped exception that hangs the dispatch with no reply.
        priv, _signer, _jwks = self._hub_and_proxy()
        token = self._token(priv, sub=uuid4(), customer_id=uuid4())

        def _boom() -> dict[str, Any]:
            raise RuntimeError("provider down")

        server, rec = self._server(jwks_provider=_boom)
        await server.handle_call(self._msg(agent_id=uuid4(), customer_id=uuid4(), token=token))
        response = self._response(rec)
        assert response["success"] is False
        assert "identity verification failed" in response["error"]


class TestToolServerProxyAssertionVerification:
    """v0.13.9 enforce-only: the pod verifies the proxy's body-bound assertion (the pod's PRIMARY
    gate).

    a direct publisher to the internal subject -- without a valid proxy assertion for THIS call
    body -- is rejected. verification is unconditional. because the pod verifies the identity token
    FIRST, every message here also carries a valid token so the ASSERTION gate is the one under
    test (a tokenless message would short-circuit at the identity gate instead).
    """

    @staticmethod
    def _hub_and_proxy() -> tuple[Any, Any, dict[str, Any]]:
        """a Hub identity keypair + a proxy-assertion signer + the COMBINED JWKS the pod verifies
        the identity token AND the proxy assertion against."""
        import base64

        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from pydantic import SecretStr

        from threetears.core.security import ProxyAssertionSigner

        priv, pub = generate_signing_keypair()
        seed = base64.urlsafe_b64encode(Ed25519PrivateKey.generate().private_bytes_raw()).decode("ascii")
        signer = ProxyAssertionSigner.from_secret(SecretStr(seed))
        combined = {"keys": [*build_jwks({"kid-1": pub})["keys"], *signer.public_jwks()["keys"]]}
        return priv, signer, combined

    @staticmethod
    def _token(priv: Any) -> str:
        now = int(time.time())
        claims = IdentityClaims(
            sub=str(uuid4()),
            customer_id=str(uuid4()),
            sid="sid-1",
            pod_id="pod-1",
            iss="hub",
            iat=now,
            exp=now + 600,
        )
        return sign_identity_token(claims, signing_key=priv, kid="kid-1")

    @staticmethod
    def _server(jwks: dict[str, Any], *, assertion_replay_guard: Any = None) -> tuple[ToolServer, Any]:
        server = ToolServer(
            nats_url="nats://localhost:9999",
            namespace_collection=None,
            pod_id="test-pod",
            jwks_provider=lambda: jwks,
            assertion_replay_guard=assertion_replay_guard if assertion_replay_guard is not None else _PodReplayGuard(),
        )
        server.register(StubTool(name="test.stub", version="1.0"))
        return server, _attach_recording_nc(server)

    @staticmethod
    def _msg(*, token: str, assertion: str | None, correlation_id: str) -> IncomingMessage:
        context: dict[str, Any] = {"correlation_id": correlation_id, "identity_token": token}
        envelope: dict[str, Any] = {
            "tool_name": "test.stub",
            "tool_version": "1.0",
            "arguments": {"key": "value"},
            "context": context,
        }
        if assertion is not None:
            envelope["proxy_assertion"] = assertion
        return _make_nats_msg(envelope)

    @staticmethod
    def _response(rec: Any) -> dict[str, Any]:
        return json.loads(rec.last_reply[1].model_dump_json())

    @staticmethod
    def _assertion_for(signer: Any, *, body_hash: str) -> str:
        return signer.mint(
            pod_id="test-pod",
            agent_id=str(uuid4()),
            customer_id=str(uuid4()),
            body_hash=body_hash,
            nonce=str(uuid4()),
            now=int(time.time()),
        )

    @pytest.mark.asyncio
    async def test_enforce_accepts_a_valid_assertion(self) -> None:
        from threetears.core.security import canonical_call_hash

        priv, signer, jwks = self._hub_and_proxy()
        server, rec = self._server(jwks)
        corr = str(uuid4())
        body_hash = canonical_call_hash("test.stub", {"key": "value"}, corr)
        await server.handle_call(
            self._msg(
                token=self._token(priv),
                assertion=self._assertion_for(signer, body_hash=body_hash),
                correlation_id=corr,
            )
        )
        assert self._response(rec)["success"] is True

    @pytest.mark.asyncio
    async def test_enforce_rejects_a_missing_assertion(self) -> None:
        priv, _signer, jwks = self._hub_and_proxy()
        server, rec = self._server(jwks)
        # a valid token clears the identity gate; the absent assertion is what the assertion gate
        # must reject (proving the two gates are both live).
        await server.handle_call(self._msg(token=self._token(priv), assertion=None, correlation_id=str(uuid4())))
        response = self._response(rec)
        assert response["success"] is False
        assert "proxy assertion verification failed" in response["error"]

    @pytest.mark.asyncio
    async def test_enforce_rejects_an_assertion_for_a_different_body(self) -> None:
        priv, signer, jwks = self._hub_and_proxy()
        server, rec = self._server(jwks)
        # an assertion bound to a DIFFERENT body than the actual call -> splice rejected.
        await server.handle_call(
            self._msg(
                token=self._token(priv),
                assertion=self._assertion_for(signer, body_hash="WRONG-BODY"),
                correlation_id=str(uuid4()),
            )
        )
        assert self._response(rec)["success"] is False

    @pytest.mark.asyncio
    async def test_enforce_requires_an_assertion_replay_guard(self) -> None:
        # mirror the registry's required pop replay guard: a pod with NO assertion replay guard must
        # NOT silently skip single-use enforcement -- even a VALID token + VALID assertion is
        # rejected (fail closed). serve() always provisions the guard; this pins the verify-site
        # guard so a regression that drops it surfaces as a rejection, not a silent replay window.
        from threetears.core.security import canonical_call_hash

        priv, signer, jwks = self._hub_and_proxy()
        server = ToolServer(
            nats_url="nats://localhost:9999",
            namespace_collection=None,
            pod_id="test-pod",
            jwks_provider=lambda: jwks,
            # NOTE: assertion_replay_guard omitted -> None -> the verify site must fail closed.
        )
        server.register(StubTool(name="test.stub", version="1.0"))
        rec = _attach_recording_nc(server)
        corr = str(uuid4())
        body_hash = canonical_call_hash("test.stub", {"key": "value"}, corr)
        await server.handle_call(
            self._msg(
                token=self._token(priv),
                assertion=self._assertion_for(signer, body_hash=body_hash),
                correlation_id=corr,
            )
        )
        response = self._response(rec)
        # the call is rejected at the proxy-assertion gate (the missing guard makes it fail closed);
        # the wire error carries the gate name + exception type, not the internal message.
        assert response["success"] is False
        assert "proxy assertion verification failed" in response["error"]


class _PodRekeyingProvider:
    """models a tool-pod JWKS cache that is STALE for the Hub identity key until ONE reactive refresh
    brings it current (a Hub re-key the cache had not caught up to). counts reactive refreshes so a
    test can assert "exactly one, never a stampede"."""

    def __init__(self, *, stale: dict[str, Any], fresh: dict[str, Any]) -> None:
        self._jwks = stale
        self._fresh = fresh
        self.refresh_calls = 0

    def __call__(self) -> dict[str, Any]:
        return self._jwks

    async def refresh_now(self) -> bool:
        self.refresh_calls += 1
        self._jwks = self._fresh
        return True


def _pod_jwks_without_hub_kid() -> dict[str, Any]:
    """the combined pod JWKS with the Hub identity key (kid-1) REMOVED.

    leaves the proxy-assertion signer key(s) intact, so the proxy-assertion gate still verifies; only
    the Hub identity-token gate sees a kid-miss -- exactly the stale-after-re-key shape B5 self-heals.
    """
    fresh = _pod_jwks_provider()
    return {"keys": [k for k in fresh["keys"] if k.get("kid") != "kid-1"]}


class TestToolServerJwksWarmedReadiness:
    """B5: ``jwks_warmed`` is the readiness signal the tool-pod gates its k8s readiness on -- it must
    report NOT-READY until the JWKS provider can actually verify a token."""

    def test_not_warmed_before_serve_provisions_a_provider(self) -> None:
        # no provider yet (serve() self-provisions it) -> the pod must report NOT-READY.
        server = ToolServer(nats_url="nats://localhost:9999", namespace_collection=None)
        assert server.jwks_warmed is False

    def test_reflects_an_injected_providers_warmth(self) -> None:
        class _Provider:
            def __init__(self) -> None:
                self.is_warmed = False

            def __call__(self) -> dict[str, Any]:
                return {"keys": []}

        provider = _Provider()
        server = ToolServer(nats_url="nats://localhost:9999", namespace_collection=None, jwks_provider=provider)
        assert server.jwks_warmed is False  # unwarmed cache -> NOT-READY
        provider.is_warmed = True
        assert server.jwks_warmed is True  # first successful fetch -> READY

    def test_static_injected_provider_with_no_warmth_signal_is_ready(self) -> None:
        # a static JWKS callable (tests) has no warm-up phase: it returns keys synchronously, so the
        # readiness gate treats it as ready rather than wedging NOT-READY forever.
        server = ToolServer(
            nats_url="nats://localhost:9999", namespace_collection=None, jwks_provider=_pod_jwks_provider
        )
        assert server.jwks_warmed is True


class TestToolServerReactiveJwksRefresh:
    """B5: pod-side reactive self-heal mirrors the proxy -- a kid-not-in-cache miss triggers exactly
    ONE reactive refresh + re-verify; an expired token does NOT trigger a refresh."""

    def _server(self, provider: Any) -> ToolServer:
        return ToolServer(
            nats_url="nats://localhost:9999",
            pod_id="test-pod",
            namespace_collection=None,
            jwks_provider=provider,
            jwks_refresh=provider.refresh_now,
            assertion_replay_guard=_PodReplayGuard(),
        )

    @pytest.mark.asyncio
    async def test_kid_miss_triggers_one_reactive_refresh_then_dispatches(self) -> None:
        # the cache lacks the Hub identity key after a re-key; the FIRST reactive refresh brings it,
        # so a valid call self-heals and dispatches rather than being rejected for a steady interval.
        provider = _PodRekeyingProvider(stale=_pod_jwks_without_hub_kid(), fresh=_pod_jwks_provider())
        server = self._server(provider)
        server.register(StubTool(name="test.stub", version="1.0"))
        rec = _attach_recording_nc(server)
        msg = _make_nats_msg(_signed_call_payload(pod_id="test-pod", tool_name="test.stub", tool_version="1.0"))

        await server.handle_call(msg)

        assert provider.refresh_calls == 1  # EXACTLY one reactive refresh
        response = json.loads(rec.last_reply[1].model_dump_json())
        assert response["success"] is True  # re-verify succeeded -> the tool ran

    @pytest.mark.asyncio
    async def test_expired_token_does_not_trigger_refresh(self) -> None:
        # an expired handshake token is signed under a key the cache HOLDS -> the failure is expiry,
        # not a kid-miss, so it must NOT provoke a Hub refresh (else every bad token hits the Hub).
        provider = _PodRekeyingProvider(stale=_pod_jwks_provider(), fresh=_pod_jwks_provider())
        server = self._server(provider)
        server.register(StubTool(name="test.stub", version="1.0"))
        rec = _attach_recording_nc(server)
        # swap the payload's handshake token for an EXPIRED one signed by the same Hub key the cache
        # holds (mint_user_assertion signs with the pod's Hub key under kid-1, so its kid IS present).
        payload = _signed_call_payload(pod_id="test-pod", tool_name="test.stub", tool_version="1.0")
        payload["context"]["identity_token"] = _pod_mint_hub_token(
            sub=uuid4(), customer_id=uuid4(), user_id=None, exp_delta=-3600
        )

        await server.handle_call(_make_nats_msg(payload))

        assert provider.refresh_calls == 0  # NO reactive refresh on an expired token
        response = json.loads(rec.last_reply[1].model_dump_json())
        assert response["success"] is False
        assert "identity verification failed" in response["error"]


class TestToolServerVerificationObservability:
    """B8: the pod's verify-failure log carries the exception MESSAGE so a stale-JWKS (kid-miss)
    failure is distinguishable from an absent/expired-token failure (the gap that masked the
    datasource failure). The message is the STRUCTURAL reason, never token or key material."""

    def _server(self, provider: Any) -> ToolServer:
        return ToolServer(
            nats_url="nats://localhost:9999",
            pod_id="test-pod",
            namespace_collection=None,
            jwks_provider=provider,
            assertion_replay_guard=_PodReplayGuard(),
        )

    @staticmethod
    def _detail(caplog: pytest.LogCaptureFixture) -> str:
        rec = next(r for r in caplog.records if r.getMessage() == "pod identity verification failed; rejecting call")
        extra = getattr(rec, "extra_data", None)
        assert extra is not None, "the pod identity-verification-failed log must carry structured extra_data"
        detail: str = extra["detail"]
        return detail

    @pytest.mark.asyncio
    async def test_kid_miss_vs_absent_logs_are_distinguishable(self, caplog: pytest.LogCaptureFixture) -> None:
        # (1) kid-MISS: the cache lacks the Hub identity key (stale after a re-key), no reactive
        # refresh wired -> the failure logs "no JWKS key matches the token kid".
        stale = _pod_jwks_without_hub_kid()
        server = self._server(lambda: stale)
        server.register(StubTool(name="test.stub", version="1.0"))
        _attach_recording_nc(server)
        msg = _make_nats_msg(_signed_call_payload(pod_id="test-pod", tool_name="test.stub", tool_version="1.0"))
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="threetears.agent.tools.server"):
            await server.handle_call(msg)
        kid_miss_detail = self._detail(caplog)

        # (2) token ABSENT: a different, distinct reason.
        server2 = self._server(_pod_jwks_provider)
        server2.register(StubTool(name="test.stub", version="1.0"))
        _attach_recording_nc(server2)
        bad = _signed_call_payload(pod_id="test-pod", tool_name="test.stub", tool_version="1.0")
        bad["context"].pop("identity_token", None)
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="threetears.agent.tools.server"):
            await server2.handle_call(_make_nats_msg(bad))
        absent_detail = self._detail(caplog)

        assert "no JWKS key matches the token kid" in kid_miss_detail
        assert "no identity token" in absent_detail
        assert kid_miss_detail != absent_detail  # the two failure modes are distinguishable


class TestToolServerAuthToken:
    """per-key-identity connect: the self-minted ``auth_token`` provider drives BOTH the NATS
    connect credential and a FRESH registration-manifest token on every publish."""

    @staticmethod
    def _mock_nc() -> AsyncMock:
        """a mock NatsClient wired enough for serve()'s JWKS warm-up + publishes."""
        mock_nc = AsyncMock()
        mock_nc.is_connected = True
        mock_nc.subscribe = AsyncMock()
        mock_nc.publish = AsyncMock()
        mock_nc.drain = AsyncMock()
        mock_nc.close = AsyncMock()
        mock_nc.request_raw = AsyncMock(return_value=json.dumps({"keys": []}).encode("utf-8"))
        return mock_nc

    @staticmethod
    async def _run_serve(server: ToolServer, mock_nc: AsyncMock) -> None:
        """drive serve() to first-publish then shut down."""
        serve_task = asyncio.create_task(server.serve())
        await asyncio.sleep(0.05)
        await server.shutdown()
        await asyncio.sleep(0.05)
        serve_task.cancel()
        try:
            await serve_task
        except asyncio.CancelledError:
            pass

    @staticmethod
    def _manifest_token(mock_nc: AsyncMock) -> str:
        """extract the bootstrap_token from the MOST RECENT published registration manifest."""
        for call in reversed(mock_nc.publish.call_args_list):
            subj = call.kwargs.get("subject")
            if subj is not None and "tools.register" in subj.path:
                payload = json.loads(call.kwargs["message"].model_dump_json())
                return payload["bootstrap_token"]
        raise AssertionError("no registration manifest was published")

    @pytest.mark.asyncio
    async def test_auth_token_provider_passed_to_nats_connect(self) -> None:
        """the auth_token provider is threaded into nats_connect (presented INSTEAD of creds)."""
        provider = lambda: "minted-token-abc"  # noqa: E731 -- terse test provider
        server = ToolServer(
            nats_url="nats://localhost:9999",
            pod_id="pod-authtoken",
            auth_token=provider,
            namespace_collection=None,
        )
        server.register(StubTool(name="test.stub", version="1.0"))
        mock_nc = self._mock_nc()
        with patch("threetears.agent.tools.server.nats_connect", return_value=mock_nc) as connect_spy:
            await self._run_serve(server, mock_nc)
        connect_spy.assert_called_once()
        assert connect_spy.call_args.kwargs["auth_token"] is provider
        # user/password are NOT set on the identity path -> they thread through as None
        assert connect_spy.call_args.kwargs["user"] is None
        assert connect_spy.call_args.kwargs["password"] is None

    @pytest.mark.asyncio
    async def test_manifest_carries_freshly_minted_token(self) -> None:
        """each publish mints a FRESH manifest token from the same provider (not a cached string)."""
        counter = {"n": 0}

        def provider() -> str:
            counter["n"] += 1
            return f"minted-{counter['n']}"

        server = ToolServer(
            nats_url="nats://localhost:9999",
            pod_id="pod-fresh",
            auth_token=provider,
            namespace_collection=None,
        )
        server.register(StubTool(name="test.stub", version="1.0"))
        mock_nc = self._mock_nc()
        with patch("threetears.agent.tools.server.nats_connect", return_value=mock_nc):
            await self._run_serve(server, mock_nc)
        token = self._manifest_token(mock_nc)
        # the manifest carried a provider-minted token (not the None/static fallback)
        assert token.startswith("minted-")
        # a subsequent republish re-invokes the provider -> a DIFFERENT fresh token
        await server.publish_registration()
        second = self._manifest_token(mock_nc)
        assert second.startswith("minted-")
        assert second != token

    @pytest.mark.asyncio
    async def test_static_bootstrap_token_fallback_when_no_provider(self) -> None:
        """with no auth_token provider, the manifest carries the static bootstrap_token (dev)."""
        server = ToolServer(
            nats_url="nats://localhost:9999",
            pod_id="pod-static",
            bootstrap_token="static-dev-token",
            namespace_collection=None,
        )
        server.register(StubTool(name="test.stub", version="1.0"))
        mock_nc = self._mock_nc()
        with patch("threetears.agent.tools.server.nats_connect", return_value=mock_nc) as connect_spy:
            await self._run_serve(server, mock_nc)
        assert connect_spy.call_args.kwargs["auth_token"] is None
        assert self._manifest_token(mock_nc) == "static-dev-token"


class TestNatsConnectAuthToken:
    """``nats_connect`` presents the ``auth_token`` provider INSTEAD of static user/password."""

    @pytest.mark.asyncio
    async def test_auth_token_presented_without_creds(self) -> None:
        """when auth_token is set, connect is called with it and WITHOUT user/password."""
        from threetears.agent.tools.server import nats_connect

        provider = lambda: "tok"  # noqa: E731 -- terse test provider
        sentinel = object()
        with patch(
            "threetears.agent.tools.server.NatsClient.connect",
            new=AsyncMock(return_value=sentinel),
        ) as connect_mock:
            result = await nats_connect(
                "nats://localhost:4222",
                namespace="ns",
                user="ignored",
                password="ignored",
                auth_token=provider,
            )
        assert result is sentinel
        kwargs = connect_mock.call_args.kwargs
        assert kwargs["auth_token"] is provider
        assert "user" not in kwargs
        assert "password" not in kwargs

    @pytest.mark.asyncio
    async def test_falls_back_to_user_password_when_no_token(self) -> None:
        """with no auth_token, connect is called with the static user/password (legacy/dev)."""
        from threetears.agent.tools.server import nats_connect

        sentinel = object()
        with patch(
            "threetears.agent.tools.server.NatsClient.connect",
            new=AsyncMock(return_value=sentinel),
        ) as connect_mock:
            await nats_connect(
                "nats://localhost:4222",
                namespace="ns",
                user="u",
                password="p",
            )
        kwargs = connect_mock.call_args.kwargs
        assert kwargs["user"] == "u"
        assert kwargs["password"] == "p"
        assert "auth_token" not in kwargs
