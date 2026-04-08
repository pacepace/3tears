"""ToolServer -- serves TearsTool instances via NATS.

registers tools, subscribes to call subject, publishes heartbeats,
handles graceful shutdown. each tool pod runs one ToolServer.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any
from uuid import uuid7

from nats.aio.client import Client as NatsClient
from nats.aio.msg import Msg as NatsMsg
from pydantic import BaseModel

from threetears.agent.tools.base_tool import TearsTool
from threetears.observe import get_logger, traced

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# NATS connection helper (patched in tests)
# ---------------------------------------------------------------------------


async def nats_connect(url: str) -> NatsClient:
    """connect to NATS server at given URL.

    :param url: NATS server URL
    :ptype url: str
    :return: connected NATS client
    :rtype: NatsClient
    """
    nc = NatsClient()
    await nc.connect(url)
    return nc


# ---------------------------------------------------------------------------
# Wire-format Pydantic models
# ---------------------------------------------------------------------------


class ToolManifestEntry(BaseModel):
    """single tool entry in registration manifest.

    :param name: namespaced tool name
    :ptype name: str
    :param version: semver-compatible version string
    :ptype version: str
    :param description: human-readable tool description
    :ptype description: str
    :param input_schema: JSON Schema for tool input parameters
    :ptype input_schema: dict[str, Any]
    :param timeout_seconds: expected maximum execution time, None uses caller default
    :ptype timeout_seconds: float | None
    """

    name: str
    version: str
    description: str
    input_schema: dict[str, Any]
    timeout_seconds: float | None = None


class RegistrationManifest(BaseModel):
    """manifest sent on connect to register all tools.

    :param pod_id: unique identifier for this tool pod
    :ptype pod_id: str
    :param tools: list of tool definitions served by this pod
    :ptype tools: list[ToolManifestEntry]
    :param bootstrap_token: optional authentication token for registry verification
    :ptype bootstrap_token: str | None
    """

    pod_id: str
    tools: list[ToolManifestEntry]
    bootstrap_token: str | None = None


class CallRequest(BaseModel):
    """incoming tool call request from NATS.

    :param tool_name: namespaced name of tool to invoke
    :ptype tool_name: str
    :param tool_version: version of tool to invoke
    :ptype tool_version: str
    :param arguments: tool input parameters
    :ptype arguments: dict[str, Any]
    :param correlation_id: request correlation identifier
    :ptype correlation_id: str
    """

    tool_name: str
    tool_version: str
    arguments: dict[str, Any]
    correlation_id: str


class CallResponse(BaseModel):
    """outgoing tool call response to NATS.

    :param success: whether tool execution succeeded
    :ptype success: bool
    :param content: result content string
    :ptype content: str
    :param metadata: optional additional metadata
    :ptype metadata: dict[str, Any] | None
    :param error: error message if execution failed
    :ptype error: str | None
    :param correlation_id: request correlation identifier
    :ptype correlation_id: str
    """

    success: bool
    content: str
    metadata: dict[str, Any] | None = None
    error: str | None = None
    correlation_id: str = ""


class HeartbeatMessage(BaseModel):
    """periodic heartbeat published by tool server.

    :param pod_id: unique identifier for this tool pod
    :ptype pod_id: str
    :param timestamp: ISO 8601 timestamp of heartbeat
    :ptype timestamp: str
    :param tools_count: number of tools registered in this pod
    :ptype tools_count: int
    """

    pod_id: str
    timestamp: str
    tools_count: int


# ---------------------------------------------------------------------------
# ToolServer
# ---------------------------------------------------------------------------


class ToolServer:
    """serves TearsTool instances via NATS.

    registers tools, subscribes to call subject, publishes heartbeats,
    handles graceful shutdown. each tool pod runs one ToolServer.
    """

    def __init__(
        self,
        nats_url: str,
        namespace: str = "aibots",
        pod_id: str | None = None,
        heartbeat_interval: float = 15.0,
        bootstrap_token: str | None = None,
    ) -> None:
        """initialize tool server.

        :param nats_url: NATS server connection URL
        :ptype nats_url: str
        :param namespace: NATS subject namespace prefix
        :ptype namespace: str
        :param pod_id: unique pod identifier (generated if not provided)
        :ptype pod_id: str | None
        :param heartbeat_interval: seconds between heartbeat publishes
        :ptype heartbeat_interval: float
        :param bootstrap_token: authentication token for registry verification
        :ptype bootstrap_token: str | None
        """
        self._nats_url = nats_url
        self._namespace = namespace
        self._pod_id = pod_id or str(uuid7())
        self._heartbeat_interval = heartbeat_interval
        self._bootstrap_token = bootstrap_token
        self._tools: dict[str, TearsTool] = {}
        self._nc: NatsClient | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._running = False
        self._shutdown_event = asyncio.Event()

    def register(self, tool: TearsTool) -> None:
        """register tool for serving via NATS.

        :param tool: TearsTool instance to register
        :ptype tool: TearsTool
        """
        key = f"{tool.mcp_name()}@{tool.mcp_version()}"
        self._tools[key] = tool
        log.info(
            "registered tool",
            extra={"extra_data": {"tool_key": key, "pod_id": self._pod_id}},
        )

    @traced()
    async def serve(self) -> None:
        """connect to NATS and begin serving registered tools.

        connects to NATS, publishes registration manifest, subscribes
        to call subject, starts heartbeat loop, then waits for shutdown
        signal.
        """
        self._nc = await nats_connect(self._nats_url)
        self._running = True
        log.info(
            "connected to NATS",
            extra={"extra_data": {
                "nats_url": self._nats_url,
                "pod_id": self._pod_id,
            }},
        )

        await self._publish_registration()

        call_subject = f"{self._namespace}.tools.internal.{self._pod_id}"
        await self._nc.subscribe(call_subject, cb=self._handle_call)
        log.info(
            "subscribed to call subject",
            extra={"extra_data": {"subject": call_subject}},
        )

        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        await self._shutdown_event.wait()

    @traced()
    async def _publish_registration(self) -> None:
        """publish registration manifest to NATS.

        sends manifest containing all registered tool definitions
        to registration subject for discovery by registry.
        """
        tools_list: list[ToolManifestEntry] = []
        for tool in self._tools.values():
            schema = tool.mcp_schema()
            entry = ToolManifestEntry(
                name=schema.name,
                version=schema.version,
                description=schema.description,
                input_schema=schema.input_schema,
                timeout_seconds=schema.timeout_seconds,
            )
            tools_list.append(entry)

        manifest = RegistrationManifest(
            pod_id=self._pod_id,
            tools=tools_list,
            bootstrap_token=self._bootstrap_token,
        )

        subject = f"{self._namespace}.tools.register"
        await self._nc.publish(subject, manifest.model_dump_json().encode("utf-8"))
        log.info(
            "published registration manifest",
            extra={"extra_data": {
                "subject": subject,
                "pod_id": self._pod_id,
                "tools_count": len(tools_list),
            }},
        )

    @traced(record_args=True)
    async def _handle_call(self, msg: NatsMsg) -> None:
        """handle incoming tool call request.

        parses call request, dispatches to matching tool, and sends
        response back via NATS reply.

        :param msg: incoming NATS message containing call request
        :ptype msg: NatsMsg
        """
        try:
            request = CallRequest.model_validate_json(msg.data)
        except Exception as exc:
            error_response = CallResponse(
                success=False,
                content="",
                error=f"malformed call request: {exc}",
            )
            await msg.respond(error_response.model_dump_json().encode("utf-8"))
            return

        tool_key = f"{request.tool_name}@{request.tool_version}"
        tool = self._tools.get(tool_key)

        if tool is None:
            error_response = CallResponse(
                success=False,
                content="",
                error=f"unknown tool: {tool_key}",
                correlation_id=request.correlation_id,
            )
            await msg.respond(error_response.model_dump_json().encode("utf-8"))
            log.warning(
                "unknown tool requested",
                extra={"extra_data": {
                    "tool_key": tool_key,
                    "correlation_id": request.correlation_id,
                }},
            )
            return

        try:
            tool_result = await tool.execute(**request.arguments)
            response = CallResponse(
                success=tool_result.success,
                content=tool_result.content,
                metadata=tool_result.metadata,
                error=tool_result.error,
                correlation_id=request.correlation_id,
            )
        except Exception as exc:
            log.error(
                "tool execution failed",
                extra={"extra_data": {
                    "tool_key": tool_key,
                    "correlation_id": request.correlation_id,
                    "error": str(exc),
                }},
            )
            response = CallResponse(
                success=False,
                content="",
                error=f"tool execution failed: {exc}",
                correlation_id=request.correlation_id,
            )

        await msg.respond(response.model_dump_json().encode("utf-8"))

    async def _heartbeat_loop(self) -> None:
        """publish periodic heartbeat and re-registration until shutdown.

        publishes heartbeat containing pod_id, timestamp, and tools_count
        to heartbeat subject at configured interval. re-publishes full
        registration manifest alongside each heartbeat so the registry
        recovers automatically if it restarts.
        """
        subject = f"{self._namespace}.tools.heartbeat.{self._pod_id}"
        while self._running:
            heartbeat = HeartbeatMessage(
                pod_id=self._pod_id,
                timestamp=datetime.now(UTC).isoformat(),
                tools_count=len(self._tools),
            )
            try:
                await self._nc.publish(subject, heartbeat.model_dump_json().encode("utf-8"))
            except Exception as exc:
                log.warning(
                    "heartbeat publish failed",
                    extra={"extra_data": {"error": str(exc)}},
                )
            try:
                await self._publish_registration()
            except Exception as exc:
                log.warning(
                    "periodic re-registration failed",
                    extra={"extra_data": {"error": str(exc)}},
                )
            await asyncio.sleep(self._heartbeat_interval)

    @traced()
    async def shutdown(self) -> None:
        """gracefully shut down tool server.

        stops heartbeat loop, drains NATS subscriptions, and closes
        NATS connection.
        """
        log.info(
            "shutting down tool server",
            extra={"extra_data": {"pod_id": self._pod_id}},
        )
        self._running = False

        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        if self._nc is not None:
            await self._nc.drain()
            await self._nc.close()

        self._shutdown_event.set()
