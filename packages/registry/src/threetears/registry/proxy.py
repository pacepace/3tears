"""call proxy for routing tool calls to tool pods.

subscribes to NATS call subject, validates tool availability
in catalog, selects endpoint via pluggable routing strategy,
tracks in-flight calls, and forwards to tool pod via
NATS request-reply.
"""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

from threetears.agent.tools.context_envelope import CallContext, bind_log_context
from threetears.observe import clear_context, get_logger
from threetears.registry.auth import AgentToolAuthorizer
from threetears.registry.catalog import ToolCatalog
from threetears.registry.routing import LeastConnectionsStrategy, RoutingStrategy

__all__ = [
    "CallProxy",
    "ProxyCallRequest",
    "ProxyCallResponse",
]

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Wire-format Pydantic models
# ---------------------------------------------------------------------------


_LEGACY_FLAT_IDENTITY_FIELDS: frozenset[str] = frozenset(
    {"conversation_id", "user_id", "customer_id", "correlation_id", "agent_id"}
)


class ProxyCallRequest(BaseModel):
    """incoming tool call request from agent.

    all per-call identity dimensions (conversation_id, user_id,
    customer_id, correlation_id, agent_id) ride as a single nested
    :class:`CallContext` under ``context`` and are forwarded to the
    target tool pod untouched. :class:`CallContext.agent_id` is the
    single source of truth for the originating agent identity -- the
    proxy reads it for authorization + routing decisions after
    deserialization, which happens at the same moment the context
    becomes available, so a separate top-level ``agent_id`` field was a
    duplicate representation and has been removed.

    :param tool_name: namespaced name of tool to invoke
    :ptype tool_name: str
    :param tool_version: version of tool to invoke
    :ptype tool_version: str
    :param arguments: tool input parameters
    :ptype arguments: dict[str, Any]
    :param context: unified identity + trace envelope forwarded
        verbatim to the tool pod. must be present and carry
        ``agent_id`` for the proxy to route the call; stateless
        utility calls still populate :class:`CallContext` even if only
        with ``agent_id`` + ``correlation_id``
    :ptype context: CallContext | None
    """

    model_config = ConfigDict(extra="forbid")

    tool_name: str
    tool_version: str
    arguments: dict[str, Any]
    context: CallContext | None = None

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_flat_identity_fields(cls, data: Any) -> Any:
        """reject removed flat identity fields with a migration pointer.

        all legacy flat identity fields -- ``conversation_id``,
        ``user_id``, ``customer_id``, ``correlation_id``, and
        ``agent_id`` -- have moved onto :class:`CallContext`. callers
        sending any of them as top-level wire fields hit this rejector
        with a message naming the offender so the migration point is
        obvious.

        :param data: raw input dict (mode='before' runs pre-coercion)
        :ptype data: Any
        :return: unchanged input when no legacy fields are present
        :rtype: Any
        :raises ValueError: when any legacy flat identity field is
            present on the wire
        """
        if isinstance(data, dict):
            offending = sorted(_LEGACY_FLAT_IDENTITY_FIELDS & data.keys())
            if offending:
                fields_list = ", ".join(offending)
                raise ValueError(
                    f"legacy flat identity field(s) {fields_list} rejected on "
                    f"ProxyCallRequest; migrated to CallContext, see "
                    f"threetears.agent.tools.context_envelope.CallContext"
                )
        return data


class ProxyCallResponse(BaseModel):
    """outgoing tool call response to agent.

    the response echoes the inbound :class:`CallContext` verbatim so
    identity has one shape on both sides of the proxy hop. there is no
    top-level ``correlation_id`` string; log-border stringification
    reads ``str(response.context.correlation_id)`` when needed. the
    field is ``None`` only when the inbound request carried no context
    at all (which is also rejected upstream because routing requires
    ``context.agent_id``; ``None`` survives only in error responses
    built from a malformed inbound request).

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
    :param context: unified identity + trace envelope echoed from the
        inbound :class:`ProxyCallRequest`; ``None`` only on
        malformed-request error responses where no context could be
        parsed
    :ptype context: CallContext | None
    """

    success: bool
    content: str
    metadata: dict[str, Any] | None = None
    error: str | None = None
    error_code: str | None = None
    context: CallContext | None = None


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
        returns result transparently. :attr:`CallContext.agent_id` is
        the single source of truth for routing / authorization; a
        missing context or missing ``context.agent_id`` surfaces as a
        ``MALFORMED_REQUEST`` response with a pointer to the rename.
        binds the canonical logging context tags (``cid``/``conv``/
        ``user``/``agent``/``customer``) from the inbound
        :class:`CallContext` for the duration of the dispatch so every
        log line in this handler and its callees renders with those
        tags; cleared in ``finally`` to avoid bleeding identifiers
        across concurrently-handled calls on the same asyncio task.

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

        bind_log_context(request.context)
        try:
            await self._dispatch_call(request, msg)
        finally:
            clear_context()

    async def _dispatch_call(
        self,
        request: "ProxyCallRequest",
        msg: Any,
    ) -> None:
        """body of :meth:`_process_call` after the logging-context bind.

        kept separate so the ``try``/``finally`` wrapping the
        :func:`bind_log_context` / :func:`clear_context` pair stays
        shallow; the operational flow (auth, catalog lookup, routing,
        forward) lives here untouched.

        :param request: parsed + identity-bound call request
        :ptype request: ProxyCallRequest
        :param msg: incoming NATS message (for reply subject)
        :ptype msg: Any
        :return: nothing; response bytes are published to ``msg.reply``
            by each branch below
        :rtype: None
        """
        assert self._nc is not None
        if request.context is None or request.context.agent_id is None:
            response = ProxyCallResponse(
                success=False,
                content="",
                error=(
                    "ProxyCallRequest received without context.agent_id; "
                    "cannot route"
                ),
                error_code="MALFORMED_REQUEST",
                context=request.context,
            )
            if msg.reply:
                await self._nc.publish(
                    msg.reply,
                    response.model_dump_json().encode("utf-8"),
                )
            log.warning(
                "proxy call missing agent_id in context",
                extra={
                    "extra_data": {
                        "tool_name": request.tool_name,
                        "correlation_id": _correlation_id_str(request),
                    }
                },
            )
            return

        # log-border stringification of identity dimensions; the
        # ProxyCallResponse echoes the whole context so these string
        # forms are for log records only. user_id rides on the same
        # CallContext envelope (context-task-01) and is plumbed to
        # the authorizer so rbac-evaluator implementations can
        # resolve user-side grants; ``None`` when the dispatch
        # carries no user identity (authorizer will deny).
        correlation_id_log = _correlation_id_str(request)
        agent_id_log = str(request.context.agent_id)
        user_id_log: str | None = (
            str(request.context.user_id)
            if request.context.user_id is not None
            else None
        )

        if self._authorizer is not None:
            authorized = await self._authorizer.is_authorized(
                agent_id_log,
                user_id_log,
                request.tool_name,
            )
            if not authorized:
                response = ProxyCallResponse(
                    success=False,
                    content="",
                    error=f"agent not authorized for tool {request.tool_name}",
                    error_code="TOOL_NOT_AUTHORIZED",
                    context=request.context,
                )
                if msg.reply:
                    await self._nc.publish(
                        msg.reply,
                        response.model_dump_json().encode("utf-8"),
                    )
                log.warning(
                    "agent tool call denied",
                    extra={
                        "extra_data": {
                            "agent_id": agent_id_log,
                            "user_id": user_id_log,
                            "tool_name": request.tool_name,
                            "correlation_id": correlation_id_log,
                        }
                    },
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
                context=request.context,
            )
            if msg.reply:
                await self._nc.publish(
                    msg.reply,
                    response.model_dump_json().encode("utf-8"),
                )
            log.warning(
                "tool not found for call",
                extra={
                    "extra_data": {
                        "full_name": full_name,
                        "agent_id": agent_id_log,
                        "correlation_id": correlation_id_log,
                    }
                },
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
                    error=(f"tool {full_name} endpoints have not yet confirmed reachability"),
                    error_code="TOOL_NOT_READY",
                    context=request.context,
                )
                if msg.reply:
                    await self._nc.publish(
                        msg.reply,
                        response.model_dump_json().encode("utf-8"),
                    )
                log.warning(
                    "tool endpoints still pending probe confirmation",
                    extra={
                        "extra_data": {
                            "full_name": full_name,
                            "endpoint_count": len(entry.endpoints),
                            "agent_id": agent_id_log,
                            "correlation_id": correlation_id_log,
                        }
                    },
                )
                return
            response = ProxyCallResponse(
                success=False,
                content="",
                error=f"tool {full_name} has no available endpoints",
                error_code="TOOL_UNAVAILABLE",
                context=request.context,
            )
            if msg.reply:
                await self._nc.publish(
                    msg.reply,
                    response.model_dump_json().encode("utf-8"),
                )
            log.warning(
                "no available endpoints for call",
                extra={
                    "extra_data": {
                        "full_name": full_name,
                        "endpoint_count": len(entry.endpoints),
                        "agent_id": agent_id_log,
                        "correlation_id": correlation_id_log,
                    }
                },
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
        correlation_id_log = _correlation_id_str(request)

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
                extra={
                    "extra_data": {
                        "pod_id": pod_id,
                        "tool_name": request.tool_name,
                        "correlation_id": correlation_id_log,
                        "timeout": effective_timeout,
                    }
                },
            )
            response = ProxyCallResponse(
                success=False,
                content="",
                error=f"tool call timed out after {effective_timeout}s",
                error_code="TOOL_TIMEOUT",
                context=request.context,
            )
        return response


def _correlation_id_str(request: ProxyCallRequest) -> str:
    """stringify the correlation id riding on ``request.context``.

    the wire-level correlation id lives on
    :attr:`CallContext.correlation_id`. log records carry it as a
    string for human consumption; :class:`ProxyCallResponse` itself
    echoes the whole :class:`CallContext` so the response shape stays
    identical to the request. this helper centralizes the log-border
    conversion: returns ``str(request.context.correlation_id)`` when
    present, else the empty string.

    :param request: parsed proxy call request
    :ptype request: ProxyCallRequest
    :return: stringified correlation id or ``""`` when absent
    :rtype: str
    """
    result = ""
    if request.context is not None and request.context.correlation_id is not None:
        result = str(request.context.correlation_id)
    return result


def _build_internal_payload(request: ProxyCallRequest) -> bytes:
    """build internal NATS payload for forwarding to tool pod.

    constructs :class:`CallRequest` from the proxy request, copying
    ``context`` through verbatim so identity dimensions (including
    ``correlation_id`` which now lives exclusively on
    :class:`CallContext`) survive the hop from registry to tool pod.

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
        context=request.context,
    )
    result = internal_request.model_dump_json().encode("utf-8")
    return result
