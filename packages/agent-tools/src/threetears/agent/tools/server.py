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
from threetears.agent.tools.config import (
    get_ready_poll_interval as _get_ready_poll_interval,
)
from threetears.agent.tools.config import (
    get_ready_timeout as _get_ready_timeout,
)
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


class ProbeAck(BaseModel):
    """acknowledgment of a reachability probe from the registry.

    :param pod_id: unique identifier for this tool pod
    :ptype pod_id: str
    :param ready: whether pod is ready to serve calls
    :ptype ready: bool
    """

    pod_id: str
    ready: bool = True


class DiscoveryProbeToolEntry(BaseModel):
    """single tool in a discovery probe request.

    :param name: namespaced tool name
    :ptype name: str
    :param version: semver-compatible version string
    :ptype version: str
    """

    name: str
    version: str


class DiscoveryProbeRequest(BaseModel):
    """discovery request used by :meth:`ToolServer.wait_until_ready`.

    mirrors :class:`threetears.registry.discovery.DiscoverRequest` so
    the pod can poll the registry without importing from the registry
    package (which would create a circular dependency).

    :param agent_id: pod identifier standing in for agent_id in the wire
    :ptype agent_id: str
    :param tool_manifest: list of pinned tools to resolve
    :ptype tool_manifest: list[DiscoveryProbeToolEntry]
    """

    agent_id: str
    tool_manifest: list[DiscoveryProbeToolEntry]


class DiscoveryProbeResultEntry(BaseModel):
    """single tool result in a discovery probe response.

    only the fields needed by readiness polling are modeled; extra
    fields in the wire are ignored by pydantic default.

    :param name: namespaced tool name
    :ptype name: str
    :param version: semver-compatible version string
    :ptype version: str
    :param status: availability status reported by registry
    :ptype status: str
    """

    name: str
    version: str
    status: str


class DiscoveryProbeResponse(BaseModel):
    """discovery response used by :meth:`ToolServer.wait_until_ready`.

    :param agent_id: identifier of requester echoed back
    :ptype agent_id: str
    :param tools: list of resolved tool results
    :ptype tools: list[DiscoveryProbeResultEntry]
    """

    agent_id: str
    tools: list[DiscoveryProbeResultEntry]


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

        connects to NATS, subscribes to call and probe subjects
        first so both are live before the registry can attempt
        a reachability probe, publishes the registration manifest,
        starts the heartbeat loop, then waits for shutdown signal.
        ordering matters: subscribing before publishing eliminates
        the race where the registry issues a probe to a subject
        the pod has not yet bound.
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

        call_subject = f"{self._namespace}.tools.internal.{self._pod_id}"
        await self._nc.subscribe(call_subject, cb=self._handle_call)
        log.info(
            "subscribed to call subject",
            extra={"extra_data": {"subject": call_subject}},
        )

        probe_subject = f"{self._namespace}.tools.probe.{self._pod_id}"
        await self._nc.subscribe(probe_subject, cb=self._handle_probe)
        log.info(
            "subscribed to probe subject",
            extra={"extra_data": {"subject": probe_subject}},
        )

        await self._publish_registration()

        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        await self._shutdown_event.wait()

    async def _handle_probe(self, msg: NatsMsg) -> None:
        """respond to reachability probe from registry.

        replies with a ProbeAck carrying pod_id and ready=True so the
        registry can promote this pod's pending endpoints to available.
        the probe handler does NOT set the readiness event directly --
        readiness is determined by polling the registry's discovery
        response until every registered tool reports as 'available',
        which guarantees the registry's catalog state has completed
        the pending -> available transition before ``wait_until_ready``
        unblocks.

        :param msg: incoming NATS message containing probe request
        :ptype msg: NatsMsg
        """
        ack = ProbeAck(pod_id=self._pod_id, ready=True)
        await msg.respond(ack.model_dump_json().encode("utf-8"))

    async def wait_until_ready(self, timeout: float | None = None) -> bool:
        """block until registry catalog reports every tool as available.

        polls the registry's discovery subject with this pod's tool
        manifest until the catalog reports every entry as 'available',
        then returns True. unlike an event-driven probe-arrival signal,
        this waits for the full probe -> mark_ready -> discovery round-
        trip so routable state is guaranteed when the function returns
        (no residual race where ``TOOL_NOT_READY`` could still fire for
        a fresh caller). returns False on timeout. intended as the
        developer-friendly substitute for ``asyncio.sleep(1.0)`` after
        ``serve``.

        :param timeout: seconds to wait before giving up. sourced
            from THREETEARS_TOOLSERVER_READY_TIMEOUT env var if not
            provided.
        :ptype timeout: float | None
        :return: True if ready within timeout, False on timeout
        :rtype: bool
        :raises RuntimeError: if called before ``serve`` connects NATS
        """
        if self._nc is None:
            raise RuntimeError("wait_until_ready called before serve() connected NATS")
        effective_timeout = timeout if timeout is not None else _get_ready_timeout()
        deadline = asyncio.get_event_loop().time() + effective_timeout
        manifest_names = [
            ToolManifestEntry(
                name=t.mcp_schema().name,
                version=t.mcp_schema().version,
                description=t.mcp_schema().description,
                input_schema=t.mcp_schema().input_schema,
                timeout_seconds=t.mcp_schema().timeout_seconds,
            )
            for t in self._tools.values()
        ]
        # a tool-less server has nothing to become ready for -- return True
        # immediately rather than timing out. callers still get the guarantee
        # that whatever tools ARE registered have transitioned to available.
        if not manifest_names:
            return True
        ready = False
        poll_interval = _get_ready_poll_interval()
        expected_count = len(manifest_names)
        while asyncio.get_event_loop().time() < deadline:
            try:
                request = DiscoveryProbeRequest(
                    agent_id=self._pod_id,
                    tool_manifest=[
                        DiscoveryProbeToolEntry(name=m.name, version=m.version)
                        for m in manifest_names
                    ],
                )
                reply = await self._nc.request(
                    f"{self._namespace}.tools.discover",
                    request.model_dump_json().encode("utf-8"),
                    timeout=min(1.0, max(deadline - asyncio.get_event_loop().time(), 0.01)),
                )
                response = DiscoveryProbeResponse.model_validate_json(reply.data)
                available_count = sum(
                    1 for tool in response.tools if tool.status == "available"
                )
                if available_count == expected_count:
                    ready = True
                    break
            except Exception as exc:
                # intentional: readiness polling must tolerate transient NATS
                # hiccups and discovery schema drift without crashing the
                # caller. log at debug so the symptom surfaces in diagnostics
                # rather than a blanket silent swallow.
                log.debug(
                    "wait_until_ready poll iteration failed",
                    extra={"extra_data": {
                        "pod_id": self._pod_id,
                        "error": str(exc),
                    }},
                )
            await asyncio.sleep(poll_interval)
        return ready

    @traced()
    async def _publish_registration(self) -> None:
        """publish registration manifest to NATS.

        sends manifest containing all registered tool definitions
        to registration subject for discovery by registry. requires
        ``serve()`` to have established the NATS connection first.

        :raises RuntimeError: if called before ``serve`` connects NATS
        """
        nc = self._nc
        if nc is None:
            raise RuntimeError("_publish_registration called before NATS connected")
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
        await nc.publish(subject, manifest.model_dump_json().encode("utf-8"))
        log.debug(
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
            tool_result = await tool.run(**request.arguments)
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
        recovers automatically if it restarts. requires ``serve()`` to
        have connected NATS first.

        :raises RuntimeError: if called before ``serve`` connects NATS
        """
        nc = self._nc
        if nc is None:
            raise RuntimeError("_heartbeat_loop started before NATS connected")
        subject = f"{self._namespace}.tools.heartbeat.{self._pod_id}"
        while self._running:
            heartbeat = HeartbeatMessage(
                pod_id=self._pod_id,
                timestamp=datetime.now(UTC).isoformat(),
                tools_count=len(self._tools),
            )
            try:
                await nc.publish(subject, heartbeat.model_dump_json().encode("utf-8"))
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
