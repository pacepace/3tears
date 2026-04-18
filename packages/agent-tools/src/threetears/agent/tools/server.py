"""ToolServer -- serves TearsTool instances via NATS.

registers tools, subscribes to call subject, publishes heartbeats,
handles graceful shutdown. each tool pod runs one ToolServer.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Awaitable, Callable
from uuid import UUID, uuid7

from nats.aio.client import Client as NatsClient
from nats.aio.msg import Msg as NatsMsg
from pydantic import BaseModel, ConfigDict, model_validator

from threetears.agent.tools.base_tool import TearsTool
from threetears.agent.tools.call_scope import (
    ToolCallScope,
    enter_call_scope,
)
from threetears.agent.tools.context_envelope import CallContext, bind_log_context
from threetears.agent.tools.config import (
    get_ready_poll_interval as _get_ready_poll_interval,
)
from threetears.agent.tools.config import (
    get_ready_timeout as _get_ready_timeout,
)
from threetears.agent.tools.config import (
    get_serve_ready_timeout,
)
from threetears.observe import clear_context, get_logger, traced

__all__ = [
    "CallRequest",
    "CallResponse",
    "DiscoveryProbeRequest",
    "DiscoveryProbeResponse",
    "DiscoveryProbeResultEntry",
    "DiscoveryProbeToolEntry",
    "HeartbeatMessage",
    "ProbeAck",
    "RegistrationManifest",
    "ToolManifestEntry",
    "ToolServer",
    "nats_connect",
]

# sentinel tuple used by ``tool_names`` so callers always get an
# immutable shape back (prevents accidental mutation of the internal
# dict through the public accessor).
_EMPTY_TOOL_NAMES: tuple[str, ...] = ()

if TYPE_CHECKING:
    from threetears.agent.tools.context import ToolContextManager

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


_LEGACY_FLAT_IDENTITY_FIELDS: frozenset[str] = frozenset(
    {"conversation_id", "user_id", "customer_id", "agent_id", "correlation_id"}
)


class CallRequest(BaseModel):
    """incoming tool call request from NATS.

    per-call identity dimensions (conversation_id, user_id, customer_id,
    agent_id, correlation_id) ride as a single nested
    :class:`CallContext` under ``context``. this replaces the previous
    shape where each dimension was a flat field; see
    :mod:`threetears.agent.tools.context_envelope`. ``correlation_id``
    lives exclusively on :attr:`CallContext.correlation_id`; the
    matching :class:`CallResponse` also carries a nested
    :class:`CallContext` (no top-level ``correlation_id`` string), so
    there is one shape for identity in both directions.

    the ``context`` field is optional because pure stateless tools
    (math, web search) do not require identity scope and the tool
    server degrades gracefully when it is omitted.

    :param tool_name: namespaced name of tool to invoke
    :ptype tool_name: str
    :param tool_version: version of tool to invoke
    :ptype tool_version: str
    :param arguments: tool input parameters
    :ptype arguments: dict[str, Any]
    :param context: unified identity + trace envelope for this call;
        ``None`` for stateless tool invocations. includes the
        ``correlation_id`` used for response routing and log correlation
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

        when a caller still emits ``conversation_id`` / ``user_id`` /
        ``customer_id`` / ``agent_id`` / ``correlation_id`` as top-level
        fields on the wire, pydantic's generic ``extra='forbid'`` error
        is unhelpful for diagnosing the rename. this validator
        intercepts the common legacy shapes and raises a message that
        names the offending field and points at :class:`CallContext` so
        the fix site is obvious. any other unknown field falls through
        to the standard ``extra='forbid'`` error.

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
                    f"CallRequest; migrated to CallContext, see "
                    f"threetears.agent.tools.context_envelope.CallContext"
                )
        return data


class CallResponse(BaseModel):
    """outgoing tool call response to NATS.

    responses carry the same :class:`CallContext` envelope as the
    inbound :class:`CallRequest`. the responder echoes
    ``request.context`` verbatim (or a minimally-populated
    :class:`CallContext` with just the correlation_id when that's all
    the responder knows) so downstream log consumers can correlate the
    reply to the inbound request. ``None`` when the inbound request
    carried no context (fully stateless call). identity never splits
    between "flat echo field" and "nested envelope" -- one shape in
    both directions.

    :param success: whether tool execution succeeded
    :ptype success: bool
    :param content: result content string
    :ptype content: str
    :param metadata: optional additional metadata
    :ptype metadata: dict[str, Any] | None
    :param error: error message if execution failed
    :ptype error: str | None
    :param context: unified identity + trace envelope echoed from the
        inbound :class:`CallRequest`; ``None`` when the inbound request
        carried no context
    :ptype context: CallContext | None
    """

    success: bool
    content: str
    metadata: dict[str, Any] | None = None
    error: str | None = None
    context: CallContext | None = None


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
        nats_url: str = "",
        namespace: str = "aibots",
        pod_id: str | None = None,
        heartbeat_interval: float = 15.0,
        bootstrap_token: str | None = None,
        context_factory: ("Callable[[UUID, UUID], Awaitable[ToolContextManager]] | None") = None,
        nats_client: NatsClient | None = None,
    ) -> None:
        """initialize tool server.

        the NATS connection can be supplied two ways. callers that own
        a connection lifecycle (bootstrap, orchestrator) pass
        ``nats_client`` and leave ``nats_url`` at its default; the
        server attaches to that client in :meth:`serve` and will NOT
        disconnect it in :meth:`shutdown` (lifecycle belongs to the
        caller). standalone callers pass ``nats_url``; the server
        opens its own connection in :meth:`serve` and closes it in
        :meth:`shutdown`. exactly one of the two must be supplied
        with a non-empty value.

        :param nats_url: NATS server connection URL; leave empty when
            supplying ``nats_client``
        :ptype nats_url: str
        :param namespace: NATS subject namespace prefix
        :ptype namespace: str
        :param pod_id: unique pod identifier (generated if not provided)
        :ptype pod_id: str | None
        :param heartbeat_interval: seconds between heartbeat publishes
        :ptype heartbeat_interval: float
        :param bootstrap_token: authentication token for registry verification
        :ptype bootstrap_token: str | None
        :param context_factory: optional async factory taking
            ``(conversation_id, user_id)`` and returning a
            :class:`ToolContextManager` scoped to that conversation.
            when supplied, the server constructs a
            :class:`ToolCallScope` per incoming call and installs it
            via :func:`enter_call_scope` so conversation-aware tools
            (workspace_*, pin-backed builtins) can resolve their
            context through :func:`tool_context_provider`. when
            omitted, tools that require a context crash with a
            :class:`RuntimeError` at first use, same as today
        :ptype context_factory: Callable[[UUID, UUID], Awaitable[ToolContextManager]] | None
        :param nats_client: pre-connected NATS client supplied by a
            caller that owns its lifecycle (typically the agent
            bootstrap sharing one connection across strategy,
            handler, and heartbeat). when set, ``nats_url`` is
            ignored and the server will not disconnect the client on
            shutdown
        :ptype nats_client: NatsClient | None
        :raises ValueError: when neither ``nats_url`` nor
            ``nats_client`` carries a usable value
        """
        if not nats_url and nats_client is None:
            raise ValueError(
                "ToolServer requires either nats_url or nats_client; neither "
                "was supplied"
            )
        self._nats_url = nats_url
        self._namespace = namespace
        self._pod_id = pod_id or str(uuid7())
        self._heartbeat_interval = heartbeat_interval
        self._bootstrap_token = bootstrap_token
        self._context_factory = context_factory
        self._tools: dict[str, TearsTool] = {}
        self._nc: NatsClient | None = nats_client
        self._owns_nats_connection: bool = nats_client is None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._ready_event = asyncio.Event()

    @property
    def pod_id(self) -> str:
        """return the unique pod identifier this server was constructed with.

        exposed as a public property so callers (agent runtime bootstrap)
        can derive a UUID from it without reaching into ``_pod_id``.

        :return: pod identifier string (UUID hex form)
        :rtype: str
        """
        return self._pod_id

    @property
    def tools_count(self) -> int:
        """return number of tools currently registered on this server.

        used by hub observability code (datasource tool pod, delegation
        manager) that logs ``tools_count=N`` on startup and by readiness
        checks that decide whether to start ``serve()`` at all. reading
        this property is O(1) and takes no locks; it is safe to call at
        any point in the server's lifecycle, including before
        ``serve()`` has connected and after ``shutdown()`` has
        completed.

        :return: number of registered tools
        :rtype: int
        """
        return len(self._tools)

    @property
    def tool_names(self) -> tuple[str, ...]:
        """return an immutable snapshot of registered tool keys.

        keys are the internal ``name@version`` form the server uses for
        dispatch. returns a tuple (not the internal dict) so callers
        cannot mutate the server's state through the accessor: the
        snapshot reflects the registration set at call time and does
        not update when subsequent :meth:`register_tool` /
        :meth:`deregister_tool` calls change the underlying dict.
        iteration order follows registration order (dict insertion
        order) but callers MUST NOT rely on it for correctness.

        :return: tuple of ``name@version`` strings
        :rtype: tuple[str, ...]
        """
        if not self._tools:
            return _EMPTY_TOOL_NAMES
        return tuple(self._tools.keys())

    @property
    def is_connected(self) -> bool:
        """return whether this server has an active NATS connection.

        ``True`` between the moment :meth:`serve` completes
        ``nats_connect`` and the moment :meth:`shutdown` calls
        ``close()``; ``False`` otherwise. callers that need to gate
        publish work on the server's connectivity state should use this
        property rather than reaching into ``_nc``. this is the only
        public view on the NATS client — the client itself is NOT
        exposed because tool callers have no legitimate need to
        ``subscribe``/``request``/``publish`` on the server's
        connection (those flows happen via NATS proxies or their own
        clients).

        :return: true iff ``serve()`` has connected and ``shutdown()``
            has not yet closed the client
        :rtype: bool
        """
        return self._nc is not None

    async def wait_ready(self, timeout: float | None = None) -> None:
        """block until serve() has subscribed to NATS and published registration.

        callers that spawn serve() in a background task should await this
        before sending tool calls to avoid the race where the first call
        arrives before the subscription is live. when ``timeout`` is
        ``None`` the value is sourced from
        ``THREETEARS_TOOLSERVER_SERVE_READY_TIMEOUT`` (platform default
        applied when the variable is unset or malformed).

        :param timeout: maximum seconds to wait; ``None`` reads from config
        :ptype timeout: float | None
        :raises asyncio.TimeoutError: if serve() does not become ready in time
        """
        resolved = get_serve_ready_timeout() if timeout is None else timeout
        await asyncio.wait_for(self._ready_event.wait(), timeout=resolved)

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

    def unregister(self, mcp_name: str) -> bool:
        """remove tool registration by mcp_name, regardless of version.

        supports atomic swap flows (hot-reload of workspace config,
        per-agent plugin refresh) where a tool family is registered,
        then replaced with a rebuilt instance bound to new dependencies.
        matches on mcp_name prefix of the internal ``name@version`` key so
        callers do not have to know the version. returns True when one or
        more keys were removed; False when nothing matched so callers can
        distinguish a no-op from a successful removal without silently
        swallowing an invariant break.

        :param mcp_name: namespaced tool name to remove
        :ptype mcp_name: str
        :return: True when one or more keys were removed
        :rtype: bool
        """
        prefix = f"{mcp_name}@"
        matched_keys = [key for key in self._tools if key.startswith(prefix)]
        for key in matched_keys:
            del self._tools[key]
        removed = len(matched_keys) > 0
        if removed:
            log.info(
                "unregistered tool",
                extra={
                    "extra_data": {
                        "mcp_name": mcp_name,
                        "removed_keys": matched_keys,
                        "pod_id": self._pod_id,
                    }
                },
            )
        return removed

    @traced()
    async def serve(self) -> None:
        """begin serving registered tools on NATS.

        when the server was constructed with an injected
        ``nats_client`` the connection is already open and the server
        attaches to it. when the server was constructed with a
        ``nats_url`` it opens its own connection here. either way the
        server subscribes to call and probe subjects first so both
        are live before the registry can attempt a reachability
        probe, publishes the registration manifest, starts the
        heartbeat loop, then waits for the shutdown signal. ordering
        matters: subscribing before publishing eliminates the race
        where the registry issues a probe to a subject the pod has
        not yet bound.
        """
        if self._nc is None:
            self._nc = await nats_connect(self._nats_url)
            log.info(
                "connected to NATS",
                extra={
                    "extra_data": {
                        "nats_url": self._nats_url,
                        "pod_id": self._pod_id,
                    }
                },
            )
        else:
            log.info(
                "using injected NATS connection",
                extra={
                    "extra_data": {
                        "pod_id": self._pod_id,
                    }
                },
            )
        self._running = True

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

        await self.publish_registration()

        self._ready_event.set()

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
                    tool_manifest=[DiscoveryProbeToolEntry(name=m.name, version=m.version) for m in manifest_names],
                )
                reply = await self._nc.request(
                    f"{self._namespace}.tools.discover",
                    request.model_dump_json().encode("utf-8"),
                    timeout=min(1.0, max(deadline - asyncio.get_event_loop().time(), 0.01)),
                )
                response = DiscoveryProbeResponse.model_validate_json(reply.data)
                available_count = sum(1 for tool in response.tools if tool.status == "available")
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
                    extra={
                        "extra_data": {
                            "pod_id": self._pod_id,
                            "error": str(exc),
                        }
                    },
                )
            await asyncio.sleep(poll_interval)
        return ready

    @traced()
    async def register_tool(self, tool: TearsTool) -> None:
        """register a tool and publish the updated manifest if connected.

        atomic public helper for dynamic tool-pod lifecycle (hub
        delegation manager, datasource tool pod). equivalent to calling
        :meth:`register` followed by :meth:`publish_registration` while
        holding the server's invariant that the manifest on the wire
        stays in sync with the in-memory registry. safe to call before
        :meth:`serve` has connected NATS: in that case the tool is
        still registered and the publish step is skipped (no-op) so
        the initial registration manifest published by :meth:`serve`
        will include it. safe to call multiple times with the same
        tool; duplicate ``name@version`` keys overwrite.

        :param tool: TearsTool instance to register
        :ptype tool: TearsTool
        """
        self.register(tool)
        if self._nc is not None:
            await self.publish_registration()

    @traced()
    async def deregister_tool(self, tool_name: str) -> bool:
        """remove all versions of a tool and publish the updated manifest.

        atomic public helper for dynamic tool-pod lifecycle (hub
        delegation manager deregistering an agent, datasource tool pod
        deregistering a data source). matches on ``mcp_name`` prefix
        of the internal ``name@version`` key, removes every matching
        entry, then publishes the reduced manifest if connected.
        returns ``True`` when at least one entry was removed so callers
        can distinguish a no-op deregister from a real one without
        silently swallowing an invariant break (e.g. "I thought that
        tool was registered but the key was missing"). safe to call
        before :meth:`serve` has connected NATS: the removal still
        happens and the publish step is skipped.

        :param tool_name: namespaced ``mcp_name`` (without the
            ``@version`` suffix) identifying the family of tool
            registrations to remove
        :ptype tool_name: str
        :return: true when one or more registrations were removed
        :rtype: bool
        """
        removed = self.unregister(tool_name)
        if removed and self._nc is not None:
            await self.publish_registration()
        return removed

    @traced()
    async def publish_registration(self) -> None:
        """publish registration manifest to NATS.

        sends manifest containing all registered tool definitions
        to registration subject for discovery by registry. requires
        ``serve()`` to have established the NATS connection first.
        use :meth:`register_tool` / :meth:`deregister_tool` for the
        common "mutate+publish" dynamic flows; call this directly only
        when you need to re-publish the current manifest without
        changing it (e.g. on registry recovery).

        :raises RuntimeError: if called before ``serve`` connects NATS
        """
        nc = self._nc
        if nc is None:
            raise RuntimeError("publish_registration called before NATS connected")
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
            extra={
                "extra_data": {
                    "subject": subject,
                    "pod_id": self._pod_id,
                    "tools_count": len(tools_list),
                }
            },
        )

    @traced(record_args=True)
    async def _handle_call(self, msg: NatsMsg) -> None:
        """handle incoming tool call request.

        parses call request, dispatches to matching tool, and sends
        response back via NATS reply. the inbound :class:`CallContext`
        is echoed verbatim on the :class:`CallResponse` so the response
        carries identity in the same shape as the request. binds the
        canonical logging context tags (``cid``/``conv``/``user``/
        ``agent``/``customer``) from the :class:`CallContext` for the
        duration of the dispatch so every log line in this handler and
        its callees renders with those tags.

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

        bind_log_context(request.context)
        try:
            # log-border stringification of the correlation id lifted
            # off the inbound context; the response itself echoes the
            # whole context (one shape in both directions), this
            # variable exists only to tag log records that the
            # set_context binding does not already cover.
            correlation_id_log = (
                str(request.context.correlation_id)
                if request.context is not None and request.context.correlation_id is not None
                else ""
            )

            tool_key = f"{request.tool_name}@{request.tool_version}"
            tool = self._tools.get(tool_key)

            if tool is None:
                error_response = CallResponse(
                    success=False,
                    content="",
                    error=f"unknown tool: {tool_key}",
                    context=request.context,
                )
                await msg.respond(error_response.model_dump_json().encode("utf-8"))
                log.warning(
                    "unknown tool requested",
                    extra={
                        "extra_data": {
                            "tool_key": tool_key,
                            "correlation_id": correlation_id_log,
                        }
                    },
                )
                return

            try:
                scope = await self._build_call_scope(request)
                async with enter_call_scope(scope):
                    tool_result = await tool.run(**request.arguments)
                response = CallResponse(
                    success=tool_result.success,
                    content=tool_result.content,
                    metadata=tool_result.metadata,
                    error=tool_result.error,
                    context=request.context,
                )
            except Exception as exc:
                log.error(
                    "tool execution failed",
                    extra={
                        "extra_data": {
                            "tool_key": tool_key,
                            "correlation_id": correlation_id_log,
                            "error": str(exc),
                        }
                    },
                )
                response = CallResponse(
                    success=False,
                    content="",
                    error=f"tool execution failed: {exc}",
                    context=request.context,
                )

            await msg.respond(response.model_dump_json().encode("utf-8"))
        finally:
            clear_context()

    async def _build_call_scope(
        self,
        request: CallRequest,
    ) -> ToolCallScope:
        """construct per-call scope from envelope :class:`CallContext`.

        reads identity dimensions off ``request.context`` (which arrives
        as UUIDs already coerced by pydantic at the wire boundary) and
        resolves a :class:`ToolContextManager` by calling the server's
        ``context_factory`` when both ``conversation_id`` and
        ``user_id`` are present. callers that do not need the context
        (stateless tools) can safely omit ``context`` entirely: the
        resulting scope carries ``context_manager=None`` and any tool
        that requires it raises at first use.

        factory exceptions propagate to :meth:`_handle_call`'s except
        block so the call is surfaced as a failed tool result rather
        than a silent no-context handoff.

        :param request: parsed call request
        :ptype request: CallRequest
        :return: populated :class:`ToolCallScope`
        :rtype: ToolCallScope
        """
        context = request.context if request.context is not None else CallContext()
        context_manager: ToolContextManager | None = None
        log.debug(
            "building call scope",
            extra={
                "extra_data": {
                    "factory_present": self._context_factory is not None,
                    "conv_present": context.conversation_id is not None,
                    "user_present": context.user_id is not None,
                }
            },
        )
        if (
            self._context_factory is not None
            and context.conversation_id is not None
            and context.user_id is not None
        ):
            context_manager = await self._context_factory(
                context.conversation_id,
                context.user_id,
            )
        return ToolCallScope(
            context=context,
            context_manager=context_manager,
        )

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
                await self.publish_registration()
            except Exception as exc:
                log.warning(
                    "periodic re-registration failed",
                    extra={"extra_data": {"error": str(exc)}},
                )
            await asyncio.sleep(self._heartbeat_interval)

    @traced()
    async def shutdown(self) -> None:
        """gracefully shut down tool server.

        stops the heartbeat loop and drains NATS subscriptions the
        server owns. the NATS connection itself is closed ONLY when
        the server opened it (i.e. was constructed with ``nats_url``).
        when the connection was injected via ``nats_client`` the
        caller owns the lifecycle and shutdown leaves the connection
        open so other subscribers (graph handler, heartbeat loop on
        the bootstrap side) continue to work until the caller closes
        the connection itself.
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

        if self._nc is not None and self._owns_nats_connection:
            await self._nc.drain()
            await self._nc.close()

        self._shutdown_event.set()
