"""``McpServer`` -- thin wrapper over the official ``mcp.server.Server``.

owns three things on top of the SDK:

1. tool registration: pulls :class:`McpTool` instances from a
   :class:`ToolRegistry` and registers them with the underlying SDK
   server so the client sees them at ``tools/list`` time.
2. RBAC gating: every dispatch flows through
   :meth:`Authorizer.allows(identity, tool.required_permission)`, or
   :meth:`CallAuthorizer.authorize_call` for an authorizer that has it,
   before the handler runs. denied dispatches return a
   :class:`mcp.types.CallToolResult` with ``isError=True`` so the
   MCP client sees the failure at the protocol level (not as
   ``isError=False`` text content carrying inner JSON).
3. error mapping: handler exceptions are caught at the framework
   border, logged with context, and returned as ``CallToolResult
   (isError=True, content=[...])``. nothing leaks past the border
   as a raw traceback.

stdio is the v1 transport. v2 HTTP slots in by adding a
``serve_http`` method alongside :meth:`serve_stdio`; the rest of
the framework (RBAC, identity, error mapping) is unchanged.

logging discipline: stdio servers MUST NOT write to stdout / stderr.
the framework configures nothing on its own -- callers wire logging
to a file or to NATS via ``threetears.observe.configure_logging``
BEFORE calling :meth:`serve_stdio`. an enforcement test guards
against any module under ``threetears.mcp`` writing to
``sys.stdout`` / ``sys.stderr`` directly.
"""

from __future__ import annotations

import asyncio
import json
import warnings
from collections.abc import Awaitable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast, get_args

import jsonschema
import mcp.types as mcp_types
from mcp.server import Server as _SdkServer
from mcp.server.connection import allowed_log_levels
from mcp.server.context import ServerRequestContext
from mcp.server.context import CallNext, HandlerResult
from mcp.server.stdio import stdio_server as _stdio_server
from mcp.server.subscriptions import ListenHandler
from mcp.shared.exceptions import MCPDeprecationWarning
from mcp.types.version import MODERN_PROTOCOL_VERSIONS
from threetears.observe import BuildOnce, contain, get_logger

from threetears.mcp._mcp_sdk_internals import connection_of
from threetears.mcp.auth import (
    Authorizer,
    CallAuthorizer,
    GrantCacheUnavailableError,
    IdentityProvider,
    Refusal,
    ToolCall,
)
from threetears.mcp.context import CallContext, LogLevel
from threetears.mcp.errors import (
    AUTHZ_ERROR,
    ERROR_META_KEY,
    HANDLER_ERROR,
    IDENTITY_UNAVAILABLE,
    INVALID_ARGUMENTS,
    OUTPUT_VALIDATION_ERROR,
    PERMISSION_DENIED,
    TOOL_ERROR,
    TOOLS_UNAVAILABLE,
    UNKNOWN_TOOL,
    ErrorCode,
    ErrorCodes,
    McpToolError,
    error_detail,
)
from threetears.mcp.http_server import (
    McpHttpTransport,
    build_mcp_http_app,
    current_request_streams,
    current_transport_pushes,
    on_request_drop,
    serve_mcp_http,
)
from threetears.mcp.push import Pushes
from threetears.mcp.resources import (
    ResourceProvider,
    ResourceRegistry,
    ResourceService,
    identify_or_refuse,
    refuse_unknown_resource,
)
from threetears.mcp.tool import McpTool, ToolProvider, ToolRegistry

if TYPE_CHECKING:
    from starlette.applications import Starlette

__all__ = [
    "McpServer",
    "tool_error",
]

log = get_logger(__name__)

#: what a client is told when the tool provider fails; the provider's own error is logged, not sent
_TOOLS_UNAVAILABLE_MESSAGE = "the server's tool set could not be read"

#: how much of a refused result's structured content the refusal's log line keeps: enough for an
#: operator to see what the call returned, without one oversized result flooding the log
_MAX_LOGGED_STRUCTURED_CHARS = 4000


#: how much of a tool's own failure text a failure that named no code carries as its error message
_MAX_TOOL_ERROR_MESSAGE_CHARS = 500


#: the protocol's log levels, least severe first: a client's ``logging/setLevel`` asks for its level and every one after it
_LOG_LEVELS: tuple[LogLevel, ...] = get_args(LogLevel)


class _HandshakeLoggingServer(_SdkServer[Any]):
    """the SDK server, declaring the ``logging`` capability only on the handshake-era protocols.

    The SDK declares it whenever a ``logging/setLevel`` handler is registered; 2026-07-28 removed both the capability
    and the request (a client there opts in per request), so on those protocols it is left out.
    """

    def get_capabilities(self, *args: Any, **kwargs: Any) -> mcp_types.ServerCapabilities:
        """the SDK's capabilities, less ``logging`` on a 2026-07-28-era protocol.

        :param args: the SDK's positional arguments
        :ptype args: Any
        :param kwargs: the SDK's keyword arguments; ``protocol_version`` names the era when given
        :ptype kwargs: Any
        :return: the capabilities
        :rtype: mcp.types.ServerCapabilities
        """
        capabilities = super().get_capabilities(*args, **kwargs)
        if kwargs.get("protocol_version") in MODERN_PROTOCOL_VERSIONS:
            return capabilities.model_copy(update={"logging": None})
        # a list change is pushed to a connection that outlives its requests (a stateful session, stdio); a stateless
        # request has nothing to push to, so it is not told to expect them
        pushes = current_transport_pushes()
        update: dict[str, Any] = {}
        if capabilities.tools is not None:
            update["tools"] = capabilities.tools.model_copy(update={"list_changed": pushes})
        if capabilities.resources is not None:
            update["resources"] = capabilities.resources.model_copy(
                update={"list_changed": pushes, "subscribe": pushes}
            )
        return capabilities.model_copy(update=update)


class McpServer:
    """RBAC-gated MCP server wrapping the official SDK server.

    composition not inheritance: the SDK's :class:`mcp.server.Server`
    handles the protocol-level concerns (tools/list, tools/call,
    initialize handshake, notifications). this class owns the
    framework-level concerns (RBAC, identity, error mapping) and
    delegates protocol to the SDK.

    :param name: server name advertised to the MCP client at
        initialization
    :ptype name: str
    :param identity_provider: resolves the calling identity for
        every dispatch (v1 returns one identity per server lifetime)
    :ptype identity_provider: IdentityProvider
    :param authorizer: evaluates ``(identity, required_permission)``
        before each tool dispatch; framework default is
        :class:`LocalGrantAuthorizer`
    :ptype authorizer: Authorizer
    :param registry: tool registry to expose. per-product servers
        construct their own via a ``build_registry()`` factory so
        registry lifetime is tied to one server process and tests get
        isolation by construction (there is no module-level default).
        exactly one of ``registry`` and ``tool_provider`` is given
    :ptype registry: ToolRegistry | None
    :param tool_provider: called on every ``tools/list`` and
        ``tools/call`` for the tools as they stand, for a tool set that
        follows a live registry; in place of ``registry``
    :ptype tool_provider: ToolProvider | None
    :param error_codes: the server's closed set of error codes and the exceptions mapped to them; the
        built-in codes alone when not given
    :ptype error_codes: ErrorCodes | None
    :param resources: a fixed set of resources and templates to serve; the server serves none (and declares no
        resources capability) when neither this nor ``resource_provider`` is given
    :ptype resources: ResourceRegistry | None
    :param resource_provider: called on every resource request for the resources as they stand, in place of
        ``resources``
    :ptype resource_provider: ResourceProvider | None
    :param version: the server's version, advertised as ``serverInfo.version``: in the ``initialize`` result on the
        handshake-era protocols, in each result's ``serverInfo`` stamp on 2026-07-28. Empty is advertised empty, as
        before
    :ptype version: str
    :param instructions: how a client should use the server, for its model: in the ``initialize`` result on the
        handshake-era protocols, in the ``server/discover`` result on 2026-07-28. None sends none, as before
    :ptype instructions: str | None
    """

    def __init__(
        self,
        *,
        name: str,
        identity_provider: IdentityProvider,
        authorizer: Authorizer,
        registry: ToolRegistry | None = None,
        tool_provider: ToolProvider | None = None,
        error_codes: ErrorCodes | None = None,
        resources: ResourceRegistry | None = None,
        resource_provider: ResourceProvider | None = None,
        version: str = "",
        instructions: str | None = None,
    ) -> None:
        """capture deps; no I/O until :meth:`serve_stdio`.

        :param name: server name string
        :ptype name: str
        :param identity_provider: identity resolver
        :ptype identity_provider: IdentityProvider
        :param authorizer: permission evaluator
        :ptype authorizer: Authorizer
        :param registry: a fixed tool registry
        :ptype registry: ToolRegistry | None
        :param tool_provider: the live tool set, in place of ``registry``
        :ptype tool_provider: ToolProvider | None
        :param error_codes: the server's error codes
        :ptype error_codes: ErrorCodes | None
        :param resources: a fixed set of resources
        :ptype resources: ResourceRegistry | None
        :param resource_provider: the live resource set, in place of ``resources``
        :ptype resource_provider: ResourceProvider | None
        :param version: the server's advertised version
        :ptype version: str
        :param instructions: the server's instructions to a client
        :ptype instructions: str | None
        :return: nothing
        :rtype: None
        :raises ValueError: when ``name`` is empty, not exactly one of
            ``registry`` and ``tool_provider`` is given, or both
            ``resources`` and ``resource_provider`` are
        """
        if not name:
            raise ValueError("server name must be non-empty")
        # a fixed registry is the provider that always answers the same tools
        if registry is not None and tool_provider is None:
            self._tool_provider: ToolProvider = registry.all
        elif tool_provider is not None and registry is None:
            self._tool_provider = tool_provider
        else:
            raise ValueError("McpServer takes exactly one of registry= and tool_provider=")
        self._name = name
        self._identity_provider = identity_provider
        self._authorizer = authorizer
        # an authorizer whose class defines authorize_call is given the whole call; read from the class, so a
        # stand-in that answers any attribute is not taken for one
        self._call_authorizer: CallAuthorizer | None = (
            cast("CallAuthorizer", authorizer) if callable(getattr(type(authorizer), "authorize_call", None)) else None
        )
        self._error_codes = error_codes if error_codes is not None else ErrorCodes()
        # a failure naming a code the set does not hold is logged once per code: every such call is the same bug.
        # Bounded: a tool naming codes without end is logged on each past the bound, never held without end
        self._unregistered_codes_logged: BuildOnce[str, bool] = BuildOnce(cost=lambda _: 1, max_cost=256)
        self._started = False
        # one ERROR per grant-cache outage, not one traceback per refused call: the authorizer
        # already logged the failed reload with its traceback, and an outage refuses every call.
        self._grant_cache_outage_logged = False
        # one WARNING per server, the first time a handler's progress or log entry is dropped because
        # the HTTP transport answers in plain JSON: every such call drops them the same way
        self._undeliverable_notifications_logged = False
        # the level each connection asked for with logging/setLevel (handshake-era protocols), keyed by the SDK's
        # connection, whose teardown stack drops the entry as the connection closes (the push registry's lifecycle,
        # threetears.mcp.push). A connection that never asked is absent: every level is sent, as before. Changed only
        # on the serving loop (a request's handler, a connection's teardown), so a snapshot replaced whole needs no lock
        self._log_levels: Mapping[int, LogLevel] = MappingProxyType({})
        # the connections and listen streams this server's change notifications go to
        self._pushes = Pushes()
        if resources is not None and resource_provider is not None:
            raise ValueError("McpServer takes at most one of resources= and resource_provider=")
        provider = resources.all if resources is not None else resource_provider
        self._resources: ResourceService | None = (
            None
            if provider is None
            else ResourceService(
                provider=provider,
                identity_provider=identity_provider,
                authorizer=authorizer,
                pushes=self._pushes,
                error_codes=self._error_codes,
            )
        )
        self._listen = ListenHandler(self._pushes.bus)
        self._sdk_server: _SdkServer[Any] = self._build_sdk_server(name, version=version, instructions=instructions)

    @property
    def name(self) -> str:
        """advertised server name.

        :return: server name
        :rtype: str
        """
        return self._name

    @property
    def error_codes(self) -> ErrorCodes:
        """the server's closed set of error codes: what a failed call's ``_meta`` error may name.

        :return: the set
        :rtype: ErrorCodes
        """
        return self._error_codes

    def _build_sdk_server(self, name: str, *, version: str, instructions: str | None) -> _SdkServer[Any]:
        """build the SDK server with this server's ``tools/list`` + ``tools/call`` handlers.

        each :class:`McpServer` gets its own SDK server (no cross-pollination); both handlers
        answer from :meth:`_tools` and :meth:`_dispatch`.

        :param name: advertised server name
        :ptype name: str
        :param version: advertised server version
        :ptype version: str
        :param instructions: the server's instructions to a client
        :ptype instructions: str | None
        :return: the wired SDK server
        :rtype: mcp.server.Server
        """

        async def _on_list_tools(_ctx: Any, _params: Any) -> mcp_types.ListToolsResult:
            return mcp_types.ListToolsResult(tools=self._tools())

        async def _on_call_tool(ctx: ServerRequestContext[Any, Any], params: Any) -> mcp_types.CallToolResult:
            # the SDK passes arguments=None for a call that sent none. what escapes _dispatch is a
            # bug (it answers every refusal itself), so it is not caught here: the SDK answers the
            # request with a JSON-RPC error
            return await self._dispatch(params.name, params.arguments or {}, request=ctx)

        async def _on_set_logging_level(
            ctx: ServerRequestContext[Any, Any], params: mcp_types.SetLevelRequestParams
        ) -> mcp_types.EmptyResult:
            self._set_log_level(ctx, params.level)
            return mcp_types.EmptyResult()

        # the SDK warns that the logging capability is deprecated as of 2026-07-28; on the handshake-era protocols it
        # is the specification's, and _HandshakeLoggingServer leaves it out of the later ones
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", MCPDeprecationWarning)
            sdk_server = _HandshakeLoggingServer(
                name,
                version=version,
                instructions=instructions,
                on_list_tools=_on_list_tools,
                on_call_tool=_on_call_tool,
                on_set_logging_level=_on_set_logging_level,
                on_subscriptions_listen=self._on_listen,
                **self._resource_handlers(),
            )
        sdk_server.middleware.append(self._track_connection)
        return sdk_server

    def _resource_handlers(self) -> dict[str, Any]:
        """the SDK handlers of the resource requests, when this server serves resources; none otherwise.

        :return: the handlers, by the SDK server's keyword
        :rtype: dict[str, Any]
        """
        service = self._resources
        if service is None:
            return {}

        async def _on_list_resources(_ctx: Any, _params: Any) -> mcp_types.ListResourcesResult:
            return await service.list_resources()

        async def _on_list_templates(_ctx: Any, _params: Any) -> mcp_types.ListResourceTemplatesResult:
            return await service.list_templates()

        async def _on_read(_ctx: Any, params: mcp_types.ReadResourceRequestParams) -> mcp_types.ReadResourceResult:
            return await service.read(str(params.uri))

        async def _on_subscribe(
            ctx: ServerRequestContext[Any, Any], params: mcp_types.SubscribeRequestParams
        ) -> mcp_types.EmptyResult:
            return await service.watch(ctx, str(params.uri))

        async def _on_unsubscribe(
            ctx: ServerRequestContext[Any, Any], params: mcp_types.UnsubscribeRequestParams
        ) -> mcp_types.EmptyResult:
            return service.unwatch(ctx, str(params.uri))

        return {
            "on_list_resources": _on_list_resources,
            "on_list_resource_templates": _on_list_templates,
            "on_read_resource": _on_read,
            "on_subscribe_resource": _on_subscribe,
            "on_unsubscribe_resource": _on_unsubscribe,
        }

    async def _on_listen(
        self, ctx: ServerRequestContext[Any, Any], params: mcp_types.SubscriptionsListenRequestParams
    ) -> mcp_types.SubscriptionsListenResult:
        """serve a 2026-07-28 ``subscriptions/listen`` stream to a caller with an identity.

        a stream that names resources is served only when the caller may read each of them.

        :param ctx: the request's context
        :ptype ctx: mcp.server.context.ServerRequestContext
        :param params: the listen request
        :ptype params: mcp.types.SubscriptionsListenRequestParams
        :return: the stream's closing result
        :rtype: mcp.types.SubscriptionsListenResult
        """
        if self._resources is not None:
            return await self._resources.listen(ctx, params)
        await identify_or_refuse(self._identity_provider)
        for uri in params.notifications.resource_subscriptions or ():
            raise refuse_unknown_resource(uri)
        return await self._listen(ctx, params)

    async def notify_resources_changed(self) -> None:
        """tell every connected client that the resource list changed, as :meth:`notify_tools_changed` does for tools.

        :return: nothing
        :rtype: None
        """
        await self._pushes.resources_changed()

    async def notify_resource_updated(self, uri: str) -> None:
        """tell the clients subscribed to ``uri`` that it changed, so they read it again.

        sends ``notifications/resources/updated`` to each stateful session that subscribed to it with
        ``resources/subscribe`` and on each 2026-07-28 listen stream that named it. only the URI is sent; reading
        it again is a request of its own, resolved and authorized as every request is.

        :param uri: the resource's URI
        :ptype uri: str
        :return: nothing
        :rtype: None
        """
        await self._pushes.resource_updated(uri)

    async def _track_connection(self, ctx: ServerRequestContext[Any, Any], call_next: CallNext) -> HandlerResult:
        """keep the connection every inbound message arrives on, when it is one the server can push to.

        a handshake-era connection that outlives its requests is kept until it closes, for
        :meth:`notify_tools_changed` and the like.

        :param ctx: the SDK's context for the message
        :ptype ctx: mcp.server.context.ServerRequestContext
        :param call_next: the rest of the SDK's chain
        :ptype call_next: mcp.server.context.CallNext
        :return: what the chain answers
        :rtype: mcp.server.context.HandlerResult
        """
        if current_transport_pushes() and ctx.protocol_version not in MODERN_PROTOCOL_VERSIONS:
            self._pushes.track(ctx.session)
        return await call_next(ctx)

    async def notify_tools_changed(self) -> None:
        """tell every connected client that the tool list changed, so it lists the tools again.

        sends ``notifications/tools/list_changed`` to each handshake-era connection that outlives its requests (a
        stateful Streamable-HTTP session, stdio) and on each 2026-07-28 ``subscriptions/listen`` stream that asked
        for it. a stateless request has nothing to push to: its client sees the change on its next ``tools/list``.
        never raises for a client that cannot be reached; that is logged.

        :return: nothing
        :rtype: None
        """
        await self._pushes.tools_changed()

    def _set_log_level(self, ctx: ServerRequestContext[Any, Any], level: LogLevel) -> None:
        """remember the level a connection asked for, for the log entries its later calls send.

        The level is the connection's (:func:`~threetears.mcp._mcp_sdk_internals.connection_of`), kept until its
        teardown: a stateless HTTP request is a connection of its own, so a level it asks for is gone with it and narrows
        no later call.

        :param ctx: the SDK's context for the ``logging/setLevel`` request
        :ptype ctx: mcp.server.context.ServerRequestContext
        :param level: the level asked for
        :ptype level: LogLevel
        :return: nothing
        :rtype: None
        """
        connection = connection_of(ctx.session)
        key = id(connection)
        if key not in self._log_levels:
            connection.exit_stack.callback(self._forget_log_level, key)
        self._log_levels = MappingProxyType({**self._log_levels, key: level})

    def _forget_log_level(self, key: int) -> None:
        """drop a closed connection's level.

        :param key: the connection's key
        :ptype key: int
        :return: nothing
        :rtype: None
        """
        self._log_levels = MappingProxyType({held: level for held, level in self._log_levels.items() if held != key})

    def _log_levels_for(self, request: ServerRequestContext[Any, Any]) -> frozenset[LogLevel]:
        """the log levels a call may send: the protocol's rule, narrowed by the connection's ``logging/setLevel``.

        :param request: the SDK's context for the call
        :ptype request: mcp.server.context.ServerRequestContext
        :return: the levels
        :rtype: frozenset[LogLevel]
        """
        # the SDK's own reading of the protocol's rule: on 2026-07-28+ a client opts in per request, at a level; on
        # earlier versions every level may be sent, unless the connection asked for a level with logging/setLevel
        allowed: frozenset[LogLevel] = allowed_log_levels(request.protocol_version, request.meta)
        chosen = self._log_levels.get(id(connection_of(request.session)))
        if chosen is None or request.protocol_version in MODERN_PROTOCOL_VERSIONS:
            return allowed
        return allowed & frozenset(_LOG_LEVELS[_LOG_LEVELS.index(chosen) :])

    def _current_tools(self) -> dict[str, McpTool]:
        """the tools as the provider answers them now, by name, in its order.

        :return: tool by name
        :rtype: dict[str, McpTool]
        :raises ValueError: when the provider answers two tools with one name, which a client
            could not tell apart
        """
        tools: dict[str, McpTool] = {}
        for tool in self._tool_provider():
            if tool.name in tools:
                raise ValueError(f"the tool provider answered two tools named {tool.name!r}")
            tools[tool.name] = tool
        return tools

    def _tools(self) -> list[mcp_types.Tool]:
        """the tool inventory, in the provider's order.

        :return: one MCP tool per tool the provider answers
        :rtype: list[mcp_types.Tool]
        :raises RuntimeError: when the provider fails, after logging it, with the provider's
            exception as the cause; the SDK answers the ``tools/list`` with a JSON-RPC error
        """
        try:
            tools = self._current_tools()
        except Exception as exc:
            log.exception("MCP tools/list: the tool provider failed")
            # the SDK puts the exception's text in the JSON-RPC error it answers:
            # a fixed message keeps the provider's internals off the wire, as tools/call's
            # TOOLS_UNAVAILABLE does
            raise RuntimeError(_TOOLS_UNAVAILABLE_MESSAGE) from exc
        return [_listed_tool(tool) for tool in tools.values()]

    async def _dispatch(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        request: ServerRequestContext[Any, Any],
    ) -> mcp_types.CallToolResult:
        """resolve identity, gate via authorizer, run handler, map errors.

        the gate fires BEFORE the handler so denied callers cannot
        observe handler-side state (timing, log output, etc.). every
        framework error path -- unknown tool, denied permission,
        handler exception -- returns a :class:`CallToolResult` with
        ``isError=True`` so the MCP client sees the failure at the
        protocol level. the happy path returns one too, with
        ``isError=False``: the SDK does not wrap a bare content list.

        the structured error detail (machine-readable ``code`` field)
        is preserved in the content text as JSON so callers that want
        to pattern-match on ``code`` can; the protocol-level
        ``isError`` flag is what gates "did it succeed" decisions.

        :param tool_name: requested tool name from the MCP client
        :ptype tool_name: str
        :param arguments: validated input arguments
        :ptype arguments: dict[str, Any]
        :param request: the SDK's context for the request, from which a handler that asks for one
            gets its :class:`CallContext`
        :ptype request: mcp.server.context.ServerRequestContext
        :return: the call's result, failed or not
        :rtype: mcp_types.CallToolResult
        """
        try:
            tool = self._current_tools().get(tool_name)
        except Exception as exc:
            contain(
                exc,
                log=log,
                what="MCP dispatch: the tool provider failed",
                extra={"tool_name": tool_name},
                level="error",
            )
            return _error_result(tool_name=tool_name, code=TOOLS_UNAVAILABLE, message=_TOOLS_UNAVAILABLE_MESSAGE)
        if tool is None:
            log.warning(
                "MCP dispatch: unknown tool",
                extra={"extra_data": {"tool_name": tool_name}},
            )
            return _error_result(
                tool_name=tool_name, code=UNKNOWN_TOOL, message=f"no tool named {tool_name!r} registered"
            )

        # the SDK's low-level server does not check arguments against the tool's inputSchema: checked
        # here, before the handler runs, against the tool as the provider answers it now
        try:
            jsonschema.validate(instance=arguments, schema=tool.input_schema)
        except jsonschema.ValidationError as exc:
            return _invalid_arguments_result(exc.message, _argument_at_fault(exc))
        if tool.context_parameter is not None and tool.context_parameter in arguments:
            # a schema that admits extra properties let through an argument named as the context
            return _invalid_arguments_result(
                f"{tool.context_parameter!r} is not an argument of this tool", tool.context_parameter
            )

        try:
            identity = await self._identity_provider.identify()
        except RuntimeError as exc:
            # the one type the IdentityProvider protocol raises for "no identity"; anything else is a bug
            contain(
                exc,
                log=log,
                what="MCP dispatch: identity resolution failed",
                extra={"tool_name": tool_name},
            )
            return _error_result(
                tool_name=tool_name, code=IDENTITY_UNAVAILABLE, message="caller identity could not be resolved"
            )

        try:
            if self._call_authorizer is not None:
                call = ToolCall(
                    tool_name=tool_name,
                    arguments=MappingProxyType(dict(arguments)),
                    required_permission=tool.required_permission,
                )
                decision: bool | Refusal = await self._call_authorizer.authorize_call(identity, call)
            else:
                decision = await self._authorizer.allows(identity, tool.required_permission)
        except GrantCacheUnavailableError:
            if not self._grant_cache_outage_logged:
                self._grant_cache_outage_logged = True
                log.error(
                    "MCP dispatch: the grant cache is not current; refusing non-admin calls until it reloads",
                    extra={"extra_data": {"tool_name": tool_name, "required_permission": tool.required_permission}},
                )
            return _error_result(
                tool_name=tool_name,
                code=AUTHZ_ERROR,
                message="authorization is temporarily unavailable (the grant cache is reloading); retry",
            )
        except Exception as exc:
            contain(
                exc,
                log=log,
                what="MCP dispatch: authorizer raised; denying by default",
                extra={
                    "tool_name": tool_name,
                    "principal_id": str(identity.principal_id),
                    "required_permission": tool.required_permission,
                },
            )
            return _error_result(tool_name=tool_name, code=AUTHZ_ERROR, message="authorizer evaluation failed")

        if self._grant_cache_outage_logged and not identity.is_admin:
            # an admin's call never reads the cache, so only a non-admin answer ends the outage
            self._grant_cache_outage_logged = False
            log.info("MCP dispatch: the grant cache answers again; calls are being authorized")
        if isinstance(decision, Refusal):
            refused = decision.code.code if isinstance(decision.code, ErrorCode) else decision.code
            code = self._registered(refused, tool_name, fallback=PERMISSION_DENIED)
            log.info(
                "MCP dispatch: refused by the call authorizer",
                extra={
                    "extra_data": {
                        "tool_name": tool_name,
                        "principal_id": str(identity.principal_id),
                        "code": code.code,
                    }
                },
            )
            message = decision.message if code.code == refused else f"refused: {decision.message}"
            if decision.content is not None:
                # the server's own refusal shape is the content, as a tool's failure is; the code rides _meta
                content = decision.content if isinstance(decision.content, str) else dict(decision.content)
                answered = tool_error(content, code=code, message=message, argument=decision.argument)
                return answered.model_copy(
                    update={"meta": {ERROR_META_KEY: error_detail(code, message, decision.argument)}}
                )
            return _error_result(tool_name=tool_name, code=code, message=message, argument=decision.argument)
        if decision is not True:
            log.info(
                "MCP dispatch: denied",
                extra={
                    "extra_data": {
                        "tool_name": tool_name,
                        "principal_id": str(identity.principal_id),
                        "required_permission": tool.required_permission,
                    }
                },
            )
            return _error_result(
                tool_name=tool_name,
                code=PERMISSION_DENIED,
                message=f"identity lacks required permission {tool.required_permission!r}",
            )

        cancelled = asyncio.Event()
        kwargs = dict(arguments)
        if tool.context_parameter is not None:
            kwargs[tool.context_parameter] = self._call_context(tool_name, request, cancelled)
        dropped: list[bool] = []
        try:
            result = await _handled(tool.handler(**kwargs), dropped)
        except asyncio.CancelledError:
            # the client cancelled the call or went away; work the call started outside its own task
            # sees it through the context
            cancelled.set()
            current = asyncio.current_task()
            if dropped and (current is None or current.cancelling() == 0):
                # the call's own connection dropped: the handler was cancelled, this dispatch was not. Answered for
                # the protocol's sake; the request that would read it is gone
                log.info(
                    "MCP dispatch: the call's connection dropped; its handler was cancelled",
                    extra={"extra_data": {"tool_name": tool_name}},
                )
                return _error_result(
                    tool_name=tool_name, code=HANDLER_ERROR, message="the call's connection dropped; it was cancelled"
                )
            raise
        except McpToolError as exc:
            # the handler named its failure: not a bug, so not logged as one
            code = self._registered(exc.code, tool_name)
            message = exc.message if code is not HANDLER_ERROR else f"{type(exc).__name__}: {exc}"
            return _error_result(tool_name=tool_name, code=code, message=message, argument=exc.argument)
        except Exception as exc:
            mapped = self._error_codes.for_exception(exc)
            if mapped is not None:
                # a failure the server mapped to a code of its own: an expected condition, logged as one
                log.info(
                    "MCP dispatch: handler raised a mapped exception",
                    extra={"extra_data": {"tool_name": tool_name, "code": mapped.code, "type": type(exc).__name__}},
                )
                return _error_result(tool_name=tool_name, code=mapped, message=str(exc) or type(exc).__name__)
            contain(
                exc,
                log=log,
                what="MCP dispatch: handler exception",
                extra={
                    "tool_name": tool_name,
                    "principal_id": str(identity.principal_id),
                },
            )
            return _error_result(tool_name=tool_name, code=HANDLER_ERROR, message=f"{type(exc).__name__}: {exc}")

        call_result = _as_call_tool_result(result)
        if call_result.is_error:
            return self._with_error_code(call_result, tool_name)
        if tool.output_schema is not None:
            # the result as the client receives it: the SDK sends it in JSON mode, where a UUID or a
            # datetime is a string
            wire = call_result.model_dump(mode="json", by_alias=True)
            refusal = _output_schema_refusal(wire, tool.output_schema)
            if refusal is not None:
                # the handler has run, so for a tool with side effects this log is the only record
                # of what it returned
                log.error(
                    "MCP dispatch: the tool's result does not match its outputSchema",
                    extra={
                        "extra_data": {
                            "tool_name": tool_name,
                            "reason": refusal,
                            "structured_content": _loggable_structured(wire.get("structuredContent")),
                        }
                    },
                )
                return _error_result(tool_name=tool_name, code=OUTPUT_VALIDATION_ERROR, message=refusal)
        return call_result

    def _call_context(
        self,
        tool_name: str,
        request: ServerRequestContext[Any, Any],
        cancelled: asyncio.Event,
    ) -> CallContext:
        """the :class:`CallContext` for one call, its channels bound to the SDK request it arrived on.

        progress and log messages go out on the request's own channel, so over Streamable HTTP they
        ride the call's response stream and reach the client before its result. a transport that
        answers in plain JSON has no such stream: what the handler sends is not sent, and the first
        time that happens the server logs a warning naming the fix.

        :param tool_name: the tool being called
        :ptype tool_name: str
        :param request: the SDK's context for the request
        :ptype request: mcp.server.context.ServerRequestContext
        :param cancelled: the event the dispatch sets when the call is cancelled
        :ptype cancelled: asyncio.Event
        :return: the call's context
        :rtype: CallContext
        """
        session = request.session
        request_id = request.request_id
        meta = request.meta or {}
        progress_token = meta.get("progress_token")
        allowed_levels = self._log_levels_for(request)
        streams = current_request_streams()

        async def _send_progress(progress: float, total: float | None, message: str | None) -> None:
            if not streams:
                self._note_undeliverable(tool_name, "progress")
                return
            await session.report_progress(progress, total, message)

        async def _send_log(level: LogLevel, data: Any, logger: str | None) -> None:
            if level not in allowed_levels:
                return
            if not streams:
                self._note_undeliverable(tool_name, "log")
                return
            # sent as the notification itself rather than through the SDK's send_log_message, which
            # warns on every call (the logging capability is deprecated in 2026-07-28; the message is
            # not)
            await session.send_notification(
                mcp_types.LoggingMessageNotification(
                    params=mcp_types.LoggingMessageNotificationParams(level=level, data=data, logger=logger),
                ),
                related_request_id=request_id,
            )

        return CallContext(
            tool_name=tool_name,
            request_id=request_id,
            progress_token=progress_token,
            progress_sender=_send_progress,
            log_sender=_send_log,
            cancelled=cancelled,
        )

    def _note_undeliverable(self, tool_name: str, kind: str) -> None:
        """log, once per server, that a handler's notification cannot reach the client.

        :param tool_name: the tool whose handler sent it
        :ptype tool_name: str
        :param kind: ``"progress"`` or ``"log"``
        :ptype kind: str
        :return: nothing
        :rtype: None
        """
        if self._undeliverable_notifications_logged:
            return
        self._undeliverable_notifications_logged = True
        log.warning(
            "MCP dispatch: a handler's progress and log notifications are not reaching the client: the HTTP "
            "transport answers in plain JSON (json_response=True). Build it with json_response=False to send "
            "them on the call's SSE stream. Logged once per server.",
            extra={"extra_data": {"tool_name": tool_name, "kind": kind}},
        )

    def _registered(self, code: str, tool_name: str, *, fallback: ErrorCode = HANDLER_ERROR) -> ErrorCode:
        """the set's code of that name; ``fallback`` for a name the set does not hold, logged once per name.

        :param code: the code a handler or an authorizer named
        :ptype code: str
        :param tool_name: the tool whose call named it
        :ptype tool_name: str
        :param fallback: what the call answers when the set does not hold the name
        :ptype fallback: ErrorCode
        :return: the code the call answers with
        :rtype: ErrorCode
        """
        known = self._error_codes.get(code)
        if known is not None:
            return known

        def warn() -> bool:
            log.error(
                "MCP dispatch: a call failed with an error code the server never registered; answering %s. "
                "Register it in the server's ErrorCodes. Logged once per code.",
                fallback.code,
                extra={"extra_data": {"tool_name": tool_name, "code": code}},
            )
            return True

        self._unregistered_codes_logged.get(code, warn)
        return fallback

    def _with_error_code(self, result: mcp_types.CallToolResult, tool_name: str) -> mcp_types.CallToolResult:
        """a failed result a handler returned, carrying its error under ``_meta`` from the server's set.

        the result's content and structured content stay the tool's. a result naming a code (as
        :func:`tool_error` writes one) keeps it, with ``retryable`` the set's; one naming a code the set does not
        hold answers ``HANDLER_ERROR``; one naming none is ``TOOL_ERROR``, its message the tool's own text.

        :param result: the failed result
        :ptype result: mcp_types.CallToolResult
        :param tool_name: the tool that returned it
        :ptype tool_name: str
        :return: the result with its error
        :rtype: mcp_types.CallToolResult
        """
        meta = dict(result.meta or {})
        named = meta.get(ERROR_META_KEY)
        if isinstance(named, Mapping) and isinstance(named.get("code"), str):
            code = self._registered(named["code"], tool_name)
            message = named.get("message")
            argument = named.get("argument")
            detail = error_detail(
                code,
                message if isinstance(message, str) and message else code.description,
                argument if isinstance(argument, str) else None,
            )
        else:
            detail = error_detail(TOOL_ERROR, _failure_text(result) or TOOL_ERROR.description)
        meta[ERROR_META_KEY] = detail
        return result.model_copy(update={"meta": meta})

    async def start(self) -> None:
        """initialize background state (authorizer cache prime + subscribe).

        called once before :meth:`serve_stdio`. delegates to
        :meth:`Authorizer.start` so cold-start grant priming happens
        before the first dispatch can land.

        :return: nothing
        :rtype: None
        """
        await self._authorizer.start()
        self._started = True

    async def stop(self) -> None:
        """tear down background state (authorizer catch-up tick, etc.).

        called once on shutdown (typically from a lifespan teardown
        or a SIGTERM handler). delegates to :meth:`Authorizer.stop`.

        :return: nothing
        :rtype: None
        """
        await self._authorizer.stop()

    async def serve_stdio(self) -> None:
        """run the SDK's stdio transport loop until EOF.

        callers MUST have called :meth:`start` first AND configured
        logging away from stdout / stderr. this method raises a
        clear error rather than silently default-denying every
        dispatch when start() was forgotten -- the
        :class:`LocalGrantAuthorizer` cache would be empty and
        every non-admin call would deny without explanation.

        :return: nothing
        :rtype: None
        :raises RuntimeError: when :meth:`start` has not been called
        """
        if not getattr(self, "_started", False):
            raise RuntimeError(
                "McpServer.serve_stdio called before start; the authorizer "
                "cache has not been primed. Call McpServer.start() first.",
            )
        async with _stdio_server() as (read_stream, write_stream):
            await self._sdk_server.run(
                read_stream,
                write_stream,
                self._sdk_server.create_initialization_options(),
            )

    def build_http_app(
        self,
        *,
        path: str = "/mcp",
        stateless: bool = True,
        json_response: bool = True,
    ) -> Starlette:
        """build the v2 Streamable-HTTP ASGI app serving this server.

        thin delegation into :mod:`threetears.mcp.http_server` (the
        transport bulk lives there so this v1 module stays small). the
        returned app runs the SAME wired ``self._sdk_server`` over the
        SDK's Streamable-HTTP transport -- the RBAC gate, identity
        resolution, and error mapping in :meth:`_dispatch` are reused
        unchanged. start-gated like :meth:`serve_stdio`: an unprimed
        authorizer cache would silently default-deny every non-admin
        call.

        :param path: URL path the MCP endpoint is served at
        :ptype path: str
        :param stateless: run the transport statelessly (the default); ``False`` serves stateful
            sessions, which can be pushed ``tools/list_changed`` and resource updates. the bearer is
            resolved for every message either way
        :ptype stateless: bool
        :param json_response: return plain JSON responses instead of
            SSE streams; a call's progress and log notifications then cannot
            reach the client (they need an SSE stream)
        :ptype json_response: bool
        :return: Starlette ASGI app serving the Streamable-HTTP transport
        :rtype: starlette.applications.Starlette
        :raises RuntimeError: when :meth:`start` has not been called
        """
        if not getattr(self, "_started", False):
            raise RuntimeError(
                "McpServer.build_http_app called before start; the authorizer "
                "cache has not been primed. Call McpServer.start() first.",
            )
        return build_mcp_http_app(
            self._sdk_server,
            path=path,
            stateless=stateless,
            json_response=json_response,
        )

    def build_http_transport(
        self,
        *,
        path: str = "/mcp",
        stateless: bool = True,
        json_response: bool = True,
    ) -> McpHttpTransport:
        """build the Streamable-HTTP transport for a host app to embed.

        the embedding counterpart of :meth:`build_http_app`: the host
        adds the transport's ``routes`` (or ``endpoint``) to its own app
        and enters its ``lifespan()`` in its own lifespan, which a
        mounted :meth:`build_http_app` would never run (see
        :class:`~threetears.mcp.http_server.McpHttpTransport`).
        start-gated like :meth:`build_http_app`.

        :param path: URL path of the endpoint in the transport's routes
        :ptype path: str
        :param stateless: run the transport statelessly (the default); ``False`` serves stateful
            sessions, which can be pushed ``tools/list_changed`` and resource updates. the bearer is
            resolved for every message either way
        :ptype stateless: bool
        :param json_response: return plain JSON responses instead of
            SSE streams; a call's progress and log notifications then cannot
            reach the client (they need an SSE stream)
        :ptype json_response: bool
        :return: the transport
        :rtype: threetears.mcp.http_server.McpHttpTransport
        :raises RuntimeError: when :meth:`start` has not been called
        """
        if not getattr(self, "_started", False):
            raise RuntimeError(
                "McpServer.build_http_transport called before start; the authorizer "
                "cache has not been primed. Call McpServer.start() first.",
            )
        return McpHttpTransport(
            self._sdk_server,
            path=path,
            stateless=stateless,
            json_response=json_response,
        )

    async def serve_http(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8080,
        path: str = "/mcp",
        stateless: bool = True,
        json_response: bool = True,
    ) -> None:
        """serve this server over the v2 Streamable-HTTP transport (uvicorn).

        parallels :meth:`serve_stdio`: same ``_started`` gate, then runs
        ``self._sdk_server`` over the SDK's Streamable-HTTP transport
        instead of stdio. delegates to :mod:`threetears.mcp.http_server`.
        consumers embedding the transport in a larger ASGI app mount
        :meth:`build_http_app`'s result directly instead of calling this.

        :param host: interface to bind
        :ptype host: str
        :param port: TCP port to bind (``0`` selects an ephemeral port)
        :ptype port: int
        :param path: URL path the MCP endpoint is served at
        :ptype path: str
        :param stateless: run the transport statelessly (the default); ``False`` serves stateful
            sessions, which can be pushed ``tools/list_changed`` and resource updates. the bearer is
            resolved for every message either way
        :ptype stateless: bool
        :param json_response: return plain JSON responses instead of
            SSE streams; a call's progress and log notifications then cannot
            reach the client (they need an SSE stream)
        :ptype json_response: bool
        :return: nothing
        :rtype: None
        :raises RuntimeError: when :meth:`start` has not been called
        """
        app = self.build_http_app(
            path=path,
            stateless=stateless,
            json_response=json_response,
        )
        await serve_mcp_http(app, host=host, port=port)


async def _handled(call: Awaitable[Any], dropped: list[bool]) -> Any:
    """a handler's result, its call cancelled if the HTTP request carrying it drops first.

    The handler runs as a task of its own, which the request's drop cancels (:func:`~threetears.mcp.http_server.
    on_request_drop`; nothing drops over stdio), ``True`` then appended to ``dropped``. A cancellation of the dispatch
    reaches it as well: a cancelled task cancels the task it awaits.

    :param call: the handler's call
    :ptype call: Awaitable[Any]
    :param dropped: appended ``True`` when the request's drop cancelled the call
    :ptype dropped: list[bool]
    :return: what the handler returned
    :rtype: Any
    :raises asyncio.CancelledError: when the call was cancelled, by the drop or with the dispatch
    """
    task = asyncio.ensure_future(call)

    def drop() -> None:
        dropped.append(True)
        task.cancel()

    stop = on_request_drop(drop)
    try:
        return await task
    finally:
        if stop is not None:
            stop()


def _call_tool_result(
    *,
    content: list[mcp_types.TextContent],
    is_error: bool,
    structured: dict[str, Any] | None = None,
) -> mcp_types.CallToolResult:
    """build a :class:`CallToolResult`.

    :param content: the content blocks
    :ptype content: list[mcp_types.TextContent]
    :param is_error: whether the call failed
    :ptype is_error: bool
    :param structured: the structured result, when there is one
    :ptype structured: dict[str, Any] | None
    :return: the result
    :rtype: mcp_types.CallToolResult
    """
    if structured is None:
        return mcp_types.CallToolResult(content=list(content), is_error=is_error)
    return mcp_types.CallToolResult(content=list(content), is_error=is_error, structured_content=structured)


def _error_result(
    *,
    tool_name: str,
    code: ErrorCode,
    message: str,
    argument: str | None = None,
) -> mcp_types.CallToolResult:
    """the result of a call the framework refused or a handler failed with a code.

    ``isError`` set, so a client that only checks the flag sees the failure; the error as JSON in a text block
    (``{"tool_name", "error": {"code", "message", "retryable", "argument"?}}``) for a client that reads the text;
    and the same error under ``_meta`` (:data:`~threetears.mcp.errors.ERROR_META_KEY`) for one that branches on
    the code.

    :param tool_name: the tool the call named
    :ptype tool_name: str
    :param code: the error's code
    :ptype code: ErrorCode
    :param message: what went wrong
    :ptype message: str
    :param argument: the argument at fault, when one is
    :ptype argument: str | None
    :return: the failed result
    :rtype: mcp_types.CallToolResult
    """
    detail = error_detail(code, message, argument)
    envelope = {"tool_name": tool_name, "error": detail}
    return mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text=json.dumps(envelope, indent=2))],
        is_error=True,
    ).model_copy(update={"meta": {ERROR_META_KEY: detail}})


def _invalid_arguments_result(message: str, argument: str | None) -> mcp_types.CallToolResult:
    """the result of a call whose arguments its tool's input schema refused, before the tool ran.

    the text is the SDK's own form, ``Input validation error: ...``, which clients already read; the code and the
    argument at fault ride ``_meta``.

    :param message: why the arguments were refused
    :ptype message: str
    :param argument: the argument at fault, when one is
    :ptype argument: str | None
    :return: the failed result
    :rtype: mcp_types.CallToolResult
    """
    return mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text=f"Input validation error: {message}")],
        is_error=True,
    ).model_copy(update={"meta": {ERROR_META_KEY: error_detail(INVALID_ARGUMENTS, message, argument)}})


def _argument_at_fault(exc: jsonschema.ValidationError) -> str | None:
    """the top-level argument a schema refusal is about, when it is about one.

    a refusal inside an argument's value has that argument first on its path; a missing required argument and an
    argument the schema does not admit are named by the refusal's instance and schema, not its path.

    :param exc: the refusal
    :ptype exc: jsonschema.ValidationError
    :return: the argument's name, or ``None`` when the refusal is about the arguments as a whole
    :rtype: str | None
    """
    path = list(exc.absolute_path)
    if path and isinstance(path[0], str):
        return path[0]
    if path or not isinstance(exc.instance, dict):
        return None
    if exc.validator == "required" and isinstance(exc.validator_value, list):
        missing = [name for name in exc.validator_value if name not in exc.instance]
        return str(missing[0]) if missing else None
    if exc.validator == "additionalProperties":
        properties = exc.schema.get("properties") if isinstance(exc.schema, dict) else None
        admitted = set(properties) if isinstance(properties, dict) else set()
        extra = [name for name in exc.instance if name not in admitted]
        return str(extra[0]) if len(extra) == 1 else None
    return None


def _failure_text(result: mcp_types.CallToolResult) -> str | None:
    """the text of a failed result's first text block, cut to a message's length.

    :param result: the failed result
    :ptype result: mcp_types.CallToolResult
    :return: the text, or ``None`` when it has none
    :rtype: str | None
    """
    for block in result.content:
        if isinstance(block, mcp_types.TextContent) and block.text:
            text = block.text
            if len(text) > _MAX_TOOL_ERROR_MESSAGE_CHARS:
                return f"{text[:_MAX_TOOL_ERROR_MESSAGE_CHARS]}...[truncated]"
            return text
    return None


def tool_error(
    error: Mapping[str, Any] | str,
    *,
    code: ErrorCode | str | None = None,
    message: str | None = None,
    argument: str | None = None,
) -> mcp_types.CallToolResult:
    """a tool-execution error a handler returns: ``isError`` set, the tool's own content, and its code.

    the spec's tool-execution error -- the tool ran and the call failed, as opposed to a protocol error. a mapping
    rides as ``structuredContent`` and as its serialized JSON in a text block, the same two faces a successful
    structured result has; a string is the text block alone. :class:`McpServer` passes the content through as the
    call's result, puts the error under ``_meta`` (:data:`~threetears.mcp.errors.ERROR_META_KEY`) with
    ``retryable`` from its set, and does not log it; a tool's ``output_schema`` does not apply to it::

        return tool_error({"ok": False, "error": "no runner 'r1'"}, code=NOT_FOUND, argument="runner_id")

    :param error: the tool's content for the failure: a JSON object, or text
    :ptype error: Mapping[str, Any] | str
    :param code: a code of the server's set, or its name; ``TOOL_ERROR`` when not given. a name the server never
        registered answers ``HANDLER_ERROR``
    :ptype code: ErrorCode | str | None
    :param message: the error's message; the mapping's ``message`` (or the text) when not given
    :ptype message: str | None
    :param argument: the argument at fault, when one is
    :ptype argument: str | None
    :return: the call's result
    :rtype: mcp_types.CallToolResult
    """
    if isinstance(error, str):
        text = error
        structured: dict[str, Any] | None = None
        fallback: str | None = error
    else:
        structured = dict(error)
        text = json.dumps(structured, default=str, indent=2)
        stated = structured.get("message")
        fallback = stated if isinstance(stated, str) else None
    named = TOOL_ERROR.code if code is None else (code.code if isinstance(code, ErrorCode) else code)
    detail: dict[str, Any] = {"code": named, "message": message or fallback or ""}
    if argument is not None:
        detail["argument"] = argument
    result = _call_tool_result(
        content=[mcp_types.TextContent(type="text", text=text)], structured=structured, is_error=True
    )
    return result.model_copy(update={"meta": {ERROR_META_KEY: detail}})


def _listed_tool(tool: McpTool) -> mcp_types.Tool:
    """the ``tools/list`` entry for ``tool``.

    :param tool: the tool
    :ptype tool: McpTool
    :return: the SDK's tool, with only the optional fields the tool sets
    :rtype: mcp_types.Tool
    """
    # only the optional fields the tool sets, so the listing leaves the rest unset rather than null
    optional: dict[str, Any] = {}
    if tool.output_schema is not None:
        optional["output_schema"] = tool.output_schema
    if tool.title is not None:
        optional["title"] = tool.title
    if tool.annotations is not None:
        optional["annotations"] = mcp_types.ToolAnnotations.model_validate(tool.annotations.to_wire())
    return mcp_types.Tool(name=tool.name, description=tool.description, input_schema=tool.input_schema, **optional)


def _as_call_tool_result(result: Any) -> mcp_types.CallToolResult:
    """a handler's return value as the call's result.

    :param result: what the handler returned
    :ptype result: Any
    :return: the result the client receives
    :rtype: mcp_types.CallToolResult
    """
    # a handler may return a fully-formed CallToolResult (e.g. an isError=True envelope carrying a
    # downstream {code,message}, as tool_error builds) and it passes through verbatim
    if isinstance(result, mcp_types.CallToolResult):
        return result

    # a plain content result is wrapped here: the SDK requires the handler to return the result
    if isinstance(result, list) and all(isinstance(item, mcp_types.TextContent) for item in result):
        return _call_tool_result(content=result, is_error=False)
    if isinstance(result, mcp_types.TextContent):
        return _call_tool_result(content=[result], is_error=False)
    if isinstance(result, str):
        return _call_tool_result(content=[mcp_types.TextContent(type="text", text=result)], is_error=False)

    # a structured result rides BOTH faces: the prose a model reads, and the spec's
    # ``structuredContent`` a program reads. returning the full CallToolResult rather than a
    # content list is what lets the structured field be set; the SDK's list wrapper has nowhere
    # to put it (search-spec.md §4.8).
    text = json.dumps(result, default=str, indent=2)
    return _call_tool_result(
        content=[mcp_types.TextContent(type="text", text=text)],
        structured=result if isinstance(result, dict) else {"result": result},
        is_error=False,
    )


def _output_schema_refusal(wire: dict[str, Any], output_schema: dict[str, Any]) -> str | None:
    """why a successful result breaks the tool's ``outputSchema``, or ``None`` when it does not.

    the spec's server MUST: a tool that declares an output schema returns structured results that
    conform to it. an error result is exempt. the SDK's low-level server checks nothing.

    :param wire: the call's result as the JSON the client receives
    :ptype wire: dict[str, Any]
    :param output_schema: the tool's declared output schema
    :ptype output_schema: dict[str, Any]
    :return: the reason, or ``None``
    :rtype: str | None
    """
    if wire.get("isError"):
        return None
    structured = wire.get("structuredContent")
    if structured is None:
        return "the tool declares an outputSchema but returned no structured content"
    try:
        jsonschema.validate(instance=structured, schema=output_schema)
    except jsonschema.ValidationError as exc:
        return f"the tool's structured result does not match its outputSchema: {exc.message}"
    return None


def _loggable_structured(structured: Any) -> str:
    """a refused result's structured content as JSON text, cut to a length a log line can carry.

    :param structured: the structured content, as the client would have received it
    :ptype structured: Any
    :return: its JSON, length-capped
    :rtype: str
    """
    text = json.dumps(structured)
    if len(text) > _MAX_LOGGED_STRUCTURED_CHARS:
        return f"{text[:_MAX_LOGGED_STRUCTURED_CHARS]}...[truncated]"
    return text
