"""call proxy for routing tool calls to tool pods.

subscribes to NATS call subject, validates tool availability
in catalog, selects endpoint via pluggable routing strategy,
tracks in-flight calls, and forwards to tool pod via
NATS request-reply.
"""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel

from threetears.observe import get_logger
from threetears.registry.auth import AgentToolAuthorizer
from threetears.registry.catalog import ToolCatalog
from threetears.registry.routing import LeastConnectionsStrategy, RoutingStrategy

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Wire-format Pydantic models
# ---------------------------------------------------------------------------


class ProxyCallRequest(BaseModel):
    """incoming tool call request from agent.

    optional ``conversation_id`` and ``user_id`` are passed through to
    the target tool pod so conversation-aware tools (workspace_*,
    pin-backed builtins) can resolve per-call context. the fields are
    optional because stateless tools do not need identity scope.

    :param agent_id: unique identifier of requesting agent
    :ptype agent_id: str
    :param tool_name: namespaced name of tool to invoke
    :ptype tool_name: str
    :param tool_version: version of tool to invoke
    :ptype tool_version: str
    :param arguments: tool input parameters
    :ptype arguments: dict[str, Any]
    :param correlation_id: request correlation identifier
    :ptype correlation_id: str
    :param conversation_id: conversation identifier for per-call context
        resolution; string form of a UUID
    :ptype conversation_id: str | None
    :param user_id: invoking user identifier for per-call context
        resolution; string form of a UUID
    :ptype user_id: str | None
    """

    agent_id: str
    tool_name: str
    tool_version: str
    arguments: dict[str, Any]
    correlation_id: str
    conversation_id: str | None = None
    user_id: str | None = None


class ProxyCallResponse(BaseModel):
    """outgoing tool call response to agent.

    :param success: whether tool execution succeeded
    :ptype success: bool
    :param content: result content string
    :ptype content: str
    :param metadata: optional additional metadata
    :ptype metadata: dict[str, Any] | None
    :param error: error message if execution failed
    :ptype error: str | None
    :param error_code: machine-readable error code
    :ptype error_code: str | None
    :param correlation_id: request correlation identifier
    :ptype correlation_id: str
    """

    success: bool
    content: str
    metadata: dict[str, Any] | None = None
    error: str | None = None
    error_code: str | None = None
    correlation_id: str = ""


# ---------------------------------------------------------------------------
# CallProxy
# ---------------------------------------------------------------------------


class CallProxy:
    """proxies tool calls from agents to tool pods.

    subscribes to call subject with queue group for HA,
    validates tool availability in catalog, selects endpoint
    via configurable routing strategy, tracks in-flight call
    counts, and forwards calls via NATS request-reply.
    """

    def __init__(
        self,
        catalog: ToolCatalog,
        namespace: str = "aibots",
        timeout: float | None = None,
        authorizer: AgentToolAuthorizer | None = None,
        routing_strategy: RoutingStrategy | None = None,
    ) -> None:
        """initialize call proxy.

        :param catalog: tool catalog for tool lookup
        :ptype catalog: ToolCatalog
        :param namespace: NATS subject namespace prefix
        :ptype namespace: str
        :param timeout: default timeout in seconds for forwarded NATS requests.
            sourced from THREETEARS_REGISTRY_CALL_TIMEOUT env var if not provided.
        :ptype timeout: float | None
        :param authorizer: optional agent tool authorizer for access control
        :ptype authorizer: AgentToolAuthorizer | None
        :param routing_strategy: endpoint selection strategy (defaults to least-connections)
        :ptype routing_strategy: RoutingStrategy | None
        """
        from threetears.registry.config import get_call_timeout

        self._catalog = catalog
        self._namespace = namespace
        self._timeout = timeout if timeout is not None else get_call_timeout()
        self._authorizer = authorizer
        self._routing_strategy: RoutingStrategy = routing_strategy or LeastConnectionsStrategy()
        self._nc: Any | None = None
        self._sub: Any | None = None
        self._active_tasks: set[asyncio.Task[None]] = set()

    async def start(self, nc: Any) -> None:
        """start listening for tool call requests.

        :param nc: connected NATS client
        :ptype nc: Any
        """
        self._nc = nc
        subject = f"{self._namespace}.tools.call"
        self._sub = await nc.subscribe(
            subject,
            queue="registry",
            cb=self._handle_call,
        )
        log.info(
            "call proxy started",
            extra={"extra_data": {"subject": subject, "timeout": self._timeout}},
        )

    async def stop(self) -> None:
        """stop listening and drain in-flight tool call tasks."""
        if self._sub is not None:
            await self._sub.unsubscribe()
            self._sub = None
        if self._active_tasks:
            log.info(
                "draining in-flight tool call tasks",
                extra={"extra_data": {"count": len(self._active_tasks)}},
            )
            await asyncio.gather(*self._active_tasks, return_exceptions=True)
            self._active_tasks.clear()
        log.info("call proxy stopped")

    async def _handle_call(self, msg: Any) -> None:
        """dispatch incoming tool call to background task.

        spawns _process_call as concurrent task so the NATS
        subscription callback returns immediately, allowing
        parallel processing of multiple tool call requests.

        :param msg: incoming NATS message containing call request
        :ptype msg: Any
        """
        task = asyncio.create_task(self._process_call(msg))
        self._active_tasks.add(task)
        task.add_done_callback(self._active_tasks.discard)

    async def _process_call(self, msg: Any) -> None:
        """process tool call request concurrently.

        validates tool exists, selects endpoint via routing strategy,
        tracks in-flight count, forwards call to tool pod, and
        returns result transparently.

        :param msg: incoming NATS message containing call request
        :ptype msg: Any
        :raises RuntimeError: when invoked before ``start`` connects NATS
        """
        if self._nc is None:
            raise RuntimeError("_process_call invoked before NATS connected")
        try:
            request = ProxyCallRequest.model_validate_json(msg.data)
        except Exception as exc:
            response = ProxyCallResponse(
                success=False,
                content="",
                error=f"malformed call request: {exc}",
                error_code="MALFORMED_REQUEST",
            )
            if msg.reply:
                await self._nc.publish(
                    msg.reply,
                    response.model_dump_json().encode("utf-8"),
                )
            return

        if self._authorizer is not None:
            authorized = await self._authorizer.is_authorized(
                request.agent_id, request.tool_name,
            )
            if not authorized:
                response = ProxyCallResponse(
                    success=False,
                    content="",
                    error=f"agent not authorized for tool {request.tool_name}",
                    error_code="TOOL_NOT_AUTHORIZED",
                    correlation_id=request.correlation_id,
                )
                if msg.reply:
                    await self._nc.publish(
                        msg.reply,
                        response.model_dump_json().encode("utf-8"),
                    )
                log.warning(
                    "agent tool call denied",
                    extra={"extra_data": {
                        "agent_id": request.agent_id,
                        "tool_name": request.tool_name,
                        "correlation_id": request.correlation_id,
                    }},
                )
                return

        full_name = f"{request.tool_name}@{request.tool_version}"
        entry = self._catalog.get(full_name)

        if entry is None:
            response = ProxyCallResponse(
                success=False,
                content="",
                error=f"tool {full_name} is not available",
                error_code="TOOL_UNAVAILABLE",
                correlation_id=request.correlation_id,
            )
            if msg.reply:
                await self._nc.publish(
                    msg.reply,
                    response.model_dump_json().encode("utf-8"),
                )
            log.warning(
                "tool not found for call",
                extra={"extra_data": {
                    "full_name": full_name,
                    "agent_id": request.agent_id,
                    "correlation_id": request.correlation_id,
                }},
            )
            return

        endpoint = self._routing_strategy.select(entry.endpoints)

        if endpoint is None:
            # TOOL_NOT_READY takes priority over TOOL_UNAVAILABLE: if ANY
            # endpoint is still pending its probe confirmation, the caller
            # should retry shortly rather than give up. TOOL_UNAVAILABLE is
            # only reported when no pending endpoints exist either.
            has_pending = any(ep.status == "pending" for ep in entry.endpoints)
            if has_pending:
                response = ProxyCallResponse(
                    success=False,
                    content="",
                    error=(
                        f"tool {full_name} endpoints have not yet "
                        "confirmed reachability"
                    ),
                    error_code="TOOL_NOT_READY",
                    correlation_id=request.correlation_id,
                )
                if msg.reply:
                    await self._nc.publish(
                        msg.reply,
                        response.model_dump_json().encode("utf-8"),
                    )
                log.warning(
                    "tool endpoints still pending probe confirmation",
                    extra={"extra_data": {
                        "full_name": full_name,
                        "endpoint_count": len(entry.endpoints),
                        "agent_id": request.agent_id,
                        "correlation_id": request.correlation_id,
                    }},
                )
                return
            response = ProxyCallResponse(
                success=False,
                content="",
                error=f"tool {full_name} has no available endpoints",
                error_code="TOOL_UNAVAILABLE",
                correlation_id=request.correlation_id,
            )
            if msg.reply:
                await self._nc.publish(
                    msg.reply,
                    response.model_dump_json().encode("utf-8"),
                )
            log.warning(
                "no available endpoints for call",
                extra={"extra_data": {
                    "full_name": full_name,
                    "endpoint_count": len(entry.endpoints),
                    "agent_id": request.agent_id,
                    "correlation_id": request.correlation_id,
                }},
            )
            return

        # in_flight is read by routing strategies during endpoint selection
        # and incremented/decremented here. the +=/-= pair is safe under
        # asyncio (no preemption between the read and the store within a
        # single bytecode op) but would race under threaded execution. if
        # this proxy is ever moved off a single event loop, wrap these
        # ops in an asyncio.Lock or swap to a threadsafe counter.
        endpoint.in_flight += 1
        try:
            response = await self._forward_call(request, endpoint.pod_id)
        finally:
            endpoint.in_flight -= 1
        if msg.reply:
            await self._nc.publish(
                msg.reply,
                response.model_dump_json().encode("utf-8"),
            )

    def _resolve_timeout(self, tool_name: str, tool_version: str) -> float:
        """resolve effective timeout for a tool call.

        checks catalog entry for per-tool declared timeout, falls back
        to proxy default (from env var or platform default).

        :param tool_name: namespaced tool name
        :ptype tool_name: str
        :param tool_version: tool version string
        :ptype tool_version: str
        :return: effective timeout in seconds
        :rtype: float
        """
        full_name = f"{tool_name}@{tool_version}"
        entry = self._catalog.get(full_name)
        if entry is not None and entry.timeout_seconds is not None:
            result: float = entry.timeout_seconds
            return result
        return self._timeout

    async def _forward_call(
        self,
        request: ProxyCallRequest,
        pod_id: str,
    ) -> ProxyCallResponse:
        """forward tool call to target tool pod via NATS request-reply.

        uses per-tool timeout from catalog if declared, otherwise
        falls back to proxy default.

        :param request: original call request from agent
        :ptype request: ProxyCallRequest
        :param pod_id: identifier of target tool pod
        :ptype pod_id: str
        :return: response from tool pod or error response on timeout
        :rtype: ProxyCallResponse
        :raises RuntimeError: when invoked before ``start`` connects NATS
        """
        if self._nc is None:
            raise RuntimeError("_forward_call invoked before NATS connected")
        internal_subject = f"{self._namespace}.tools.internal.{pod_id}"
        internal_payload = _build_internal_payload(request)
        effective_timeout = self._resolve_timeout(request.tool_name, request.tool_version)

        try:
            reply = await self._nc.request(
                internal_subject,
                internal_payload,
                timeout=effective_timeout,
            )
            response = ProxyCallResponse.model_validate_json(reply.data)
        except TimeoutError:
            log.warning(
                "tool call timed out",
                extra={"extra_data": {
                    "pod_id": pod_id,
                    "tool_name": request.tool_name,
                    "correlation_id": request.correlation_id,
                    "timeout": effective_timeout,
                }},
            )
            response = ProxyCallResponse(
                success=False,
                content="",
                error=f"tool call timed out after {effective_timeout}s",
                error_code="TOOL_TIMEOUT",
                correlation_id=request.correlation_id,
            )
        return response


def _build_internal_payload(request: ProxyCallRequest) -> bytes:
    """build internal NATS payload for forwarding to tool pod.

    constructs CallRequest-compatible payload using tool_name,
    tool_version, arguments, and correlation_id from proxy request.

    :param request: original proxy call request
    :ptype request: ProxyCallRequest
    :return: serialized internal call request bytes
    :rtype: bytes
    """
    from threetears.agent.tools.server import CallRequest

    internal_request = CallRequest(
        tool_name=request.tool_name,
        tool_version=request.tool_version,
        arguments=request.arguments,
        correlation_id=request.correlation_id,
        conversation_id=request.conversation_id,
        user_id=request.user_id,
    )
    result = internal_request.model_dump_json().encode("utf-8")
    return result
